"""Real filesystem and subprocess checks for the Kaggle campaign helpers."""

import _thread
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile


ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint = load("full-run-checkpoint")
processes = load("full-run-process")


class CheckpointTests(unittest.TestCase):
    def test_round_trip_and_same_archive_resume_keep_new_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            evidence = base / "session-a/campaign"
            evidence.mkdir(parents=True)
            (evidence / "notes.txt").write_text("checkpoint payload\n")
            archive = checkpoint.checkpoint(evidence)
            restored = base / "session-b/campaign"
            checkpoint.restore(archive, restored)
            self.assertEqual((restored / "notes.txt").read_bytes(), (evidence / "notes.txt").read_bytes())
            (restored / "new-progress.txt").write_text("retain this on setup rerun")
            checkpoint.restore(archive, restored)
            self.assertTrue((restored / "new-progress.txt").exists())
            with zipfile.ZipFile(checkpoint.checkpoint(restored)) as z:
                self.assertIsNone(z.testzip())
                self.assertIn("campaign/new-progress.txt", z.namelist())

    def test_restore_never_overwrites_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            target = base / "campaign"
            target.mkdir()
            (target / "notes.txt").write_text("original")
            archive = checkpoint.checkpoint(target)
            with self.assertRaises(FileExistsError):
                checkpoint.restore(archive, target)
            self.assertEqual("original", (target / "notes.txt").read_text())

    def test_invalid_members_are_rejected_before_any_extraction(self):
        for name in ("campaign/../../escape", "/campaign/absolute", "wrong-root/file", "campaign\\escape"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                archive = base / "bad.zip"
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr("campaign/valid.txt", "ordinary payload")
                    z.writestr(name, "invalid path")
                with self.assertRaises(ValueError):
                    checkpoint.restore(archive, base / "campaign")
                self.assertEqual([archive], list(base.iterdir()))

    def test_symlink_and_normalized_duplicate_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "bad.zip"
            link = zipfile.ZipInfo("campaign/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(link, "../../outside")
            with self.assertRaises(ValueError):
                checkpoint.restore(archive, base / "campaign")
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("campaign/a", "first")
                z.writestr("campaign//a", "second")
            with self.assertRaises(ValueError):
                checkpoint.restore(archive, base / "campaign")

    def test_corrupt_payload_does_not_publish_partial_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = base / "corrupt.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
                z.writestr("campaign/notes.txt", "payload")
            contents = bytearray(archive.read_bytes())
            name_len, extra_len = struct.unpack_from("<HH", contents, 26)
            contents[30 + name_len + extra_len] ^= 1
            archive.write_bytes(contents)
            with self.assertRaises(zipfile.BadZipFile):
                checkpoint.restore(archive, base / "campaign")
            self.assertFalse((base / "campaign").exists())

    def test_failed_checkpoint_retains_previous_complete_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / "campaign"
            evidence.mkdir()
            (evidence / "notes.txt").write_text("original")
            archive = checkpoint.checkpoint(evidence)
            previous = archive.read_bytes()
            (evidence / "link").symlink_to(evidence / "notes.txt")
            with self.assertRaises(ValueError):
                checkpoint.checkpoint(evidence)
            self.assertEqual(previous, archive.read_bytes())
            self.assertFalse(archive.with_suffix(".zip.part").exists())


class ProcessTests(unittest.TestCase):
    def test_success_and_failure_are_logged_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            log = Path(tmp) / "command.log"
            processes.run_logged([sys.executable, "-c", "print('completed')"], log=log)
            self.assertEqual("completed\n", log.read_text())
            with self.assertRaises(subprocess.CalledProcessError) as context:
                processes.run_logged([sys.executable, "-c", "print('failed'); raise SystemExit(7)"], log=log)
            self.assertEqual(7, context.exception.returncode)
            self.assertIn("failed", log.read_text())

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "Kaggle uses POSIX process groups")
    def test_interrupt_stops_driver_and_descendant_before_return(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            marker = Path(tmp) / "orphan-wrote-this.txt"
            child = "import time,pathlib; time.sleep(1); pathlib.Path(" + repr(str(marker)) + ").write_text('orphan')"
            driver = "import subprocess,sys,time\nsubprocess.Popen([sys.executable,'-c'," + repr(child) + "])\nwhile True:\n print('alive',flush=True)\n time.sleep(.03)"
            timer = threading.Timer(.25, _thread.interrupt_main)
            timer.start()
            try:
                with self.assertRaises(KeyboardInterrupt):
                    processes.run_logged([sys.executable, "-c", driver])
            finally:
                timer.cancel()
                timer.join()
            time.sleep(1.05)
            self.assertFalse(marker.exists(), "descendant continued running after notebook interrupt")


if __name__ == "__main__":
    unittest.main()
