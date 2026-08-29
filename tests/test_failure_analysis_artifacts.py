"""Contract tests for bounded replay failure-analysis sidecars."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.failure_analysis_artifacts import (
    ReplayArtifactError,
    ReplaySidecarReader,
    ReplaySidecarWriter,
    parse_counterfactual_thresholds,
)


class ReplayArtifactTests(unittest.TestCase):
    MANIFEST = "a" * 64
    STREAM = "b" * 64
    SOURCE = "c" * 64

    def _writer(self, root, **overrides):
        options = {
            "run_id": "diagnostic-run",
            "manifest_sha256": self.MANIFEST,
            "stream_fingerprint": self.STREAM,
            "source_fingerprint": self.SOURCE,
            "config": {"counterfactual_thresholds": [0.5, 0.75, 1.0]},
            "max_samples": 8,
            "max_bytes": 4096,
        }
        options.update(overrides)
        return ReplaySidecarWriter(Path(root) / "failure-analysis", **options)

    def test_bit_exact_tensor_round_trip(self):
        dtypes = [torch.float16, torch.float32]
        if hasattr(torch, "bfloat16"):
            dtypes.append(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory)
            expected = {}
            for index, dtype in enumerate(dtypes):
                tensor = torch.tensor([[1.25, -2.5], [3.75, -4.0]], dtype=dtype)
                expected[index] = tensor
                self.assertTrue(writer.write(items=[{"item_id": index, "feature": tensor}], query={"query_id": index}))
            writer.close()
            reader = ReplaySidecarReader(
                Path(directory) / "failure-analysis",
                manifest_sha256=self.MANIFEST,
                stream_fingerprint=self.STREAM,
                source_fingerprint=self.SOURCE,
            )
            for row in reader.rows("items"):
                actual = reader.tensor(row["feature"])
                wanted = expected[row["item_id"]]
                self.assertEqual(wanted.dtype, actual.dtype)
                self.assertEqual(tuple(wanted.shape), tuple(actual.shape))
                self.assertTrue(torch.equal(wanted.view(torch.uint8), actual.view(torch.uint8)))

    def test_sample_and_byte_limits_close_as_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory, max_samples=1)
            self.assertTrue(writer.write(query={"query_id": 0}))
            self.assertFalse(writer.write(query={"query_id": 1}))
            self.assertEqual("insufficient", writer.close()["status"])

        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory, max_bytes=4)
            self.assertFalse(writer.write(items=[{"item_id": 1, "feature": torch.ones(2)}], query={"query_id": 0}))
            metadata = writer.close()
            self.assertEqual("insufficient", metadata["status"])
            self.assertLessEqual(sum(file["bytes"] for file in metadata["files"].values()), 4)
            self.assertEqual(b"", (Path(directory) / "failure-analysis" / "features.bin").read_bytes())

    def test_segment_scoped_item_identity_preserves_reset_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory)
            first = torch.tensor([1.0, 2.0], dtype=torch.float32)
            second = torch.tensor([3.0, 4.0], dtype=torch.float32)
            self.assertTrue(writer.write(
                items=[{"segment_index": 0, "item_id": 0, "feature": first}],
                query={"segment_index": 0, "item_id": 0, "legal_candidates": [0]},
            ))
            self.assertTrue(writer.write(
                items=[{"segment_index": 1, "item_id": 0, "feature": second}],
                query={"segment_index": 1, "item_id": 0, "legal_candidates": [0]},
            ))
            writer.close()
            reader = ReplaySidecarReader(Path(directory) / "failure-analysis")
            rows = reader.rows("items")
            self.assertEqual([(0, 0), (1, 0)], [(row["segment_index"], row["item_id"]) for row in rows])
            self.assertTrue(torch.equal(first, reader.tensor(rows[0]["feature"])))
            self.assertTrue(torch.equal(second, reader.tensor(rows[1]["feature"])))

    def test_insufficient_requires_explicit_reader_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory, max_samples=1)
            writer.write(query={"query_id": 0})
            writer.write(query={"query_id": 1})
            writer.close()
            path = Path(directory) / "failure-analysis"
            with self.assertRaisesRegex(ReplayArtifactError, "not closed"):
                ReplaySidecarReader(path)
            self.assertEqual("insufficient", ReplaySidecarReader(path, allow_insufficient=True).metadata["status"])

    def test_checksum_and_identity_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory)
            writer.write(query={"query_id": 0})
            writer.close()
            path = Path(directory) / "failure-analysis"
            for keyword, value, message in (
                ("manifest_sha256", "d" * 64, "manifest binding"),
                ("stream_fingerprint", "d" * 64, "stream binding"),
                ("source_fingerprint", "d" * 64, "source binding"),
                ("run_id", "foreign-run", "run binding"),
            ):
                with self.subTest(keyword=keyword), self.assertRaisesRegex(ReplayArtifactError, message):
                    ReplaySidecarReader(path, **{keyword: value})
            with (path / "queries.jsonl").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ReplayArtifactError, "checksum mismatch"):
                ReplaySidecarReader(path)

    def test_interrupted_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self._writer(directory)
            initial = json.loads((Path(directory) / "failure-analysis" / "metadata.json").read_text())
            self.assertEqual("interrupted", initial["status"])
            with self.assertRaisesRegex(ReplayArtifactError, "not closed"):
                ReplaySidecarReader(Path(directory) / "failure-analysis")
            writer.write(query={"query_id": 0})
            metadata = writer.close(completed=False)
            self.assertEqual("interrupted", metadata["status"])
            with self.assertRaisesRegex(ReplayArtifactError, "not closed"):
                ReplaySidecarReader(Path(directory) / "failure-analysis")

    def test_threshold_parser(self):
        self.assertEqual((0.5, 0.75, 1.0), parse_counterfactual_thresholds("0.50,0.75,1.00"))
        for invalid in ("", "0.5", "0.25,0.5,1.0", "0.5,0.5", "-0.1", "1.1", "nan", "inf", "word"):
            with self.subTest(invalid=invalid), self.assertRaises(ReplayArtifactError):
                parse_counterfactual_thresholds(invalid)

    def test_writer_and_reader_reject_noncanonical_threshold_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ReplayArtifactError, "invalid counterfactual thresholds"):
                self._writer(directory, config={"counterfactual_thresholds": [0.5]} )

            writer = self._writer(directory)
            writer.write(query={"query_id": 0})
            writer.close()
            path = Path(directory) / "failure-analysis" / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["config"]["counterfactual_thresholds"] = [0.5]
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ReplayArtifactError, "noncanonical counterfactual thresholds"):
                ReplaySidecarReader(path.parent)


if __name__ == "__main__":
    unittest.main()
