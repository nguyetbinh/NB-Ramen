"""Exercise real curl transfers against a local HTTP server."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest


spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).with_name("prepare-data.py")
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
PAYLOAD = b"NB-Ramen download integrity fixture\n" * 4096
CHECKSUM = hashlib.sha256(PAYLOAD).hexdigest()


def verify(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != CHECKSUM:
        raise prepare.ProvenanceError("checksum mismatch")
    return {"sha256": digest, "path": str(path)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        requested_range = self.headers.get("Range")
        self.server.requests.append((self.path, requested_range))
        if self.path == "/gateway":
            self.send_response(504)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = b"error page returned as HTTP 200" if self.path == "/wrong" else PAYLOAD
        offset = int(requested_range.split("=")[1].split("-")[0]) if requested_range else 0
        if self.path == "/no-range":
            offset = 0
        if offset >= len(payload):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(payload)}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206 if offset else 200)
        if offset:
            self.send_header("Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}")
        self.send_header("Content-Length", str(len(payload) - offset))
        self.end_headers()
        try:
            if self.path == "/disconnect" and not requested_range:
                self.wfile.write(payload[:1024])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(payload[offset:])
        except ConnectionError:
            pass


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "artifact.bin"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.requests = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def download(self, *routes, attempts=2):
        return prepare.download(
            [self.base + route for route in routes], self.path, verify,
            log_path=self.path.with_suffix(".jsonl"), attempts=attempts, retry_delay=0,
        )

    def test_gateway_failure_uses_next_endpoint_and_records_verification(self):
        result = self.download("/gateway", "/good")
        self.assertEqual(result["path"], str(self.path))
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        records = [json.loads(line) for line in self.path.with_suffix(".jsonl").read_text().splitlines()]
        self.assertEqual([r["http_status"] for r in records], ["504", "200"])
        self.assertEqual([r["verified"] for r in records], [False, True])

    def test_interrupted_transfer_resumes_exactly_without_duplicate_bytes(self):
        self.download("/disconnect")
        self.assertEqual(self.server.requests, [("/disconnect", None), ("/disconnect", "bytes=1024-")])
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_endpoint_without_range_support_retries_from_start(self):
        self.path.with_suffix(".bin.part").write_bytes(PAYLOAD[:1024])
        self.download("/no-range")
        self.assertEqual(self.server.requests, [("/no-range", "bytes=1024-"), ("/no-range", None)])
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertEqual(len(list(self.path.parent.glob("*.rejected-*"))), 1)

    def test_completed_partial_is_verified_after_http_416(self):
        self.path.with_suffix(".bin.part").write_bytes(PAYLOAD)
        self.download("/good", attempts=1)
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_bad_http_200_content_never_becomes_the_final_archive(self):
        with self.assertRaises(RuntimeError):
            self.download("/wrong")
        self.assertFalse(self.path.exists())
        self.assertEqual(len(list(self.path.parent.glob("*.rejected-*"))), 2)

    def test_exhausted_network_errors_preserve_partial_for_later_retry(self):
        partial = self.path.with_suffix(".bin.part")
        partial.write_bytes(PAYLOAD[:1024])
        with self.assertRaises(RuntimeError):
            self.download("/gateway")
        self.assertFalse(self.path.exists())
        self.assertEqual(partial.read_bytes(), PAYLOAD[:1024])
        self.assertEqual(len(self.server.requests), 2)

    def test_verified_existing_file_avoids_network(self):
        self.path.write_bytes(PAYLOAD)
        self.download("/gateway")
        self.assertEqual(self.server.requests, [])

    def test_invalid_existing_download_is_preserved_before_replacement(self):
        self.path.write_bytes(b"old invalid download")
        self.download("/good")
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        saved = list(self.path.parent.glob("artifact.bin.rejected-*"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), b"old invalid download")


if __name__ == "__main__":
    unittest.main()
