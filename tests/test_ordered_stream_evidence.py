"""End-to-end evidence checks for ordered stream evaluation."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.evidence import (  # noqa: E402
    SUMMARY_SCHEMA_VERSION,
    TRACE_REQUIRED_FIELDS,
    TRACE_SCHEMA_VERSION,
)
from main import ordered_stream_test  # noqa: E402
from streams import (  # noqa: E402
    build_single_domain_stream,
    build_stream,
    verify_stream_fingerprint,
)


class TensorDomainDataset:
    """A small real-tensor domain dataset with deterministic sample metadata."""

    def __init__(self, domain_idx, labels):
        self.domain_idx = domain_idx
        self.targets = list(labels)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, sample_idx):
        # The test method reads only this synthetic input, never evaluator labels.
        return torch.tensor(
            [self.targets[sample_idx], self.domain_idx + 10,
             self.domain_idx * 100 + sample_idx, self.domain_idx + 1],
            dtype=torch.float32,
        ), self.targets[sample_idx]


class TensorMultiDomainDataset:
    def __init__(self):
        self.datasets = (
            TensorDomainDataset(0, [0, 1]),
            TensorDomainDataset(1, [1, 0]),
        )
        self.environments = ("clear", "shifted")

    def __len__(self):
        return len(self.datasets)


class DiagnosticMethod:
    """A real callable TTA-like method whose diagnostics come from its inputs."""

    def __init__(self):
        self._diagnostics = {}
        self.forward_calls = 0
        self.reset_after_forward_calls = []

    def __call__(self, images):
        self.forward_calls += 1
        labels = images[:, 0].to(torch.long)
        logits = torch.full((len(images), 2), -3.0, device=images.device)
        logits.scatter_(1, labels.unsqueeze(1), 3.0)
        self._diagnostics = {
            "inferred_context": images[:, 1].to(torch.long),
            "memory_size": images[:, 2].to(torch.long),
            "num_active_contexts": images[:, 3].to(torch.long),
        }
        return logits

    def get_diagnostics(self):
        return self._diagnostics

    def reset(self):
        self.reset_after_forward_calls.append(self.forward_calls)


class MixedMemoryAvailabilityMethod(DiagnosticMethod):
    """Expose an invalid partial retained-memory timeline for one batch."""

    def __call__(self, images):
        logits = super().__call__(images)
        self._diagnostics["memory_bytes"] = [64, None]
        return logits


class OrderedStreamEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.datasets = TensorMultiDomainDataset()

    @staticmethod
    def _args(stream_mode):
        return SimpleNamespace(
            run_id="ordered-stream-e2e",
            device=torch.device("cpu"),
            batch_size=2,
            num_workers=0,
            metric_window_size=2,
            metric_window_stride=1,
            stream_mode=stream_mode,
            reference_trace=None,
        )

    @staticmethod
    def _paths(directory):
        run_dir = Path(directory) / "run"
        run_dir.mkdir(parents=True)
        paths = {
            "run_dir": run_dir,
            "manifest": run_dir / "manifest.json",
            "stream": run_dir / "stream.json",
            "trace": run_dir / "trace.jsonl",
            "summary": run_dir / "summary.json",
        }
        data_root = str((PROJECT_ROOT / "tests" / "data").resolve())
        dataset_root = str((Path(data_root) / "domainbed" / "domain_net").resolve())
        config_path = str((PROJECT_ROOT / "cfg" / "default" / "NoAdapt.yaml").resolve())
        artifacts = {
            "status": "verified", "mode": "exact",
            "model": {
                "status": "verified", "model": "clip_vitbase16", "official_name": "ViT-B/16",
                "url": "https://example.invalid/model", "expected_sha256": "a" * 64,
                "filename": "ViT-B-16.pt", "publisher": "OpenAI", "trust": "pinned_official",
                "path": str((PROJECT_ROOT / "tests" / "ViT-B-16.pt").resolve()),
                "actual_sha256": "a" * 64, "size_bytes": 1,
            },
            "dataset": {
                "status": "verified", "schema_version": 1, "dataset": "domainnet",
                "root": dataset_root, "sidecar": str((PROJECT_ROOT / "tests" / "data.json").resolve()),
                "verified_exact": True, "content_algorithm": "sha256",
                "root_digest": "b" * 64, "sidecar_sha256": "c" * 64,
                "file_count": 1, "acquisition": {},
            },
        }
        reference_config = {"optimizer": "signsgd", "lr": 1e-9}
        paths["manifest"].write_text(json.dumps({
            "schema_version": 1,
            "run_id": "ordered-stream-e2e",
            "args": {
                "dataset": "DomainNet", "model": "clip_vitbase16", "device": "cpu",
                "data_root": data_root, "tta_mode": "mixed", "batch_size": 2,
                "metric_window_size": 2, "metric_window_stride": 1,
                "stream_block_size": 64,
                "artifact_provenance": "exact", "tta_algo": "NoAdapt",
                "config_path": config_path,
            },
            "config": reference_config,
            "device": "cpu",
            "dataset": {"name": "DomainNet", "environments": ["clear", "shifted"]},
            "artifacts": artifacts,
        }), encoding="utf-8")
        paths["reference_identity"] = {
            "dataset": "DomainNet", "model": "clip_vitbase16", "device": "cpu",
            "data_root": data_root, "tta_mode": "mixed", "batch_size": 2,
            "metric_window_size": 2, "metric_window_stride": 1,
            "stream_block_size": 64,
            "artifact_provenance": "exact", "artifacts": artifacts,
            "reference_config": reference_config, "reference_config_path": config_path,
        }
        return paths

    def _assert_complete_evidence(self, paths, stream, expected_recovery_status):
        stream_payload = json.loads(paths["stream"].read_text(encoding="utf-8"))
        self.assertEqual([list(reference) for reference in stream.references], stream_payload["references"])
        self.assertEqual(stream.fingerprint, stream_payload["fingerprint"])
        self.assertEqual(stream.fingerprint, stream_payload["metadata"]["fingerprint"])
        self.assertTrue(verify_stream_fingerprint(stream_payload))

        rows = [json.loads(line) for line in paths["trace"].read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(stream.references), len(rows))
        expected_contexts = []
        for timestep, ((domain_idx, sample_idx), row) in enumerate(zip(stream.references, rows)):
            label = self.datasets.datasets[domain_idx].targets[sample_idx]
            expected_contexts.append(domain_idx + 10)
            self.assertEqual(set(TRACE_REQUIRED_FIELDS), set(row))
            self.assertEqual("ordered-stream-e2e", row["run_id"])
            self.assertEqual(TRACE_SCHEMA_VERSION, row["schema_version"])
            self.assertEqual(timestep, row["timestep"])
            self.assertEqual(sample_idx, row["sample_idx"])
            self.assertEqual(domain_idx, row["ground_truth_domain"])
            self.assertEqual(label, row["ground_truth_class"])
            self.assertEqual(label, row["prediction"])
            self.assertTrue(row["correct"])
            self.assertEqual(domain_idx + 10, row["inferred_context"])
            self.assertEqual(domain_idx * 100 + sample_idx, row["memory_size"])
            self.assertEqual(domain_idx + 1, row["num_active_contexts"])
            self.assertIsNone(row["memory_bytes"])
            self.assertTrue(math.isfinite(row["predicted_entropy"]))
            self.assertGreaterEqual(row["latency_ms"], 0.0)

        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        self.assertEqual(SUMMARY_SCHEMA_VERSION, summary["schema_version"])
        self.assertEqual("ordered-stream-e2e", summary["run_id"])
        self.assertEqual(len(stream), summary["num_samples"])
        self.assertEqual(1.0, summary["micro_accuracy"])
        self.assertEqual(1.0, summary["macro_domain_accuracy"])
        self.assertEqual(1.0, summary["worst_domain_accuracy"])
        self.assertEqual({"clear": 1.0, "shifted": 1.0}, summary["domain_accuracies"])
        self.assertEqual({"clear": 2, "shifted": 2}, summary["domain_sample_counts"])
        self.assertEqual({"window_size": 2, "stride": 1}, {
            key: summary["sliding_window"][key] for key in ("window_size", "stride")
        })
        self.assertEqual(len(stream) - 1, len(summary["sliding_window"]["values"]))
        self.assertTrue(all(value["accuracy"] == 1.0 for value in summary["sliding_window"]["values"]))
        self.assertEqual(expected_recovery_status, summary["post_shift_recovery_time"]["status"])
        self.assertEqual("reference_required", summary["negative_adaptation_rate"]["status"])
        self.assertEqual("available", summary["routing_diagnostics"]["status"])
        self.assertEqual(1.0, summary["routing_diagnostics"]["normalized_mutual_information"])
        self.assertEqual(1.0, summary["routing_diagnostics"]["adjusted_rand_index"])
        self.assertEqual(1.0, summary["routing_diagnostics"]["context_purity"])
        self.assertEqual(2, summary["routing_diagnostics"]["number_of_discovered_contexts"])
        expected_churn = sum(
            left != right for left, right in zip(expected_contexts, expected_contexts[1:])
        ) / (len(expected_contexts) - 1)
        self.assertEqual(expected_churn, summary["routing_diagnostics"]["assignment_churn_rate"])
        self.assertIsNone(summary["peak_device_memory_bytes"])
        self.assertEqual(
            {"status": "not_applicable", "kind": "unsupported", "bytes": None},
            summary["device_memory"],
        )
        self.assertEqual("unavailable", summary["method_memory"]["status"])
        self.assertIsNone(summary["method_memory"]["max_retained_bytes"])
        self.assertIsNone(summary["method_memory"]["final_retained_bytes"])
        self.assertEqual("computed", summary["forward_latency"]["status"])
        self.assertGreaterEqual(summary["forward_latency"]["total_ms"], 0.0)
        self.assertGreaterEqual(summary["forward_latency"]["mean_per_sample_ms"], 0.0)
        self.assertGreaterEqual(summary["forward_latency"]["median_per_sample_ms"], 0.0)
        self.assertEqual("samples_per_second", summary["throughput"]["unit"])
        self.assertEqual("unavailable", summary["retrieval_latency"]["status"])
        self.assertEqual(stream.fingerprint, summary["stream_fingerprint"])

    def test_mixed_stream_emits_complete_evidence_and_only_final_reset(self):
        stream = build_stream(self.datasets, "iid_mixed", seed=9)
        method = DiagnosticMethod()
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            domain_names, accuracies, metadata = ordered_stream_test(
                self.datasets, method, self._args("iid_mixed"), paths, stream
            )
            self._assert_complete_evidence(paths, stream, "not_applicable")

        self.assertEqual(("clear", "shifted"), domain_names)
        self.assertEqual([1.0, 1.0], accuracies.tolist())
        self.assertEqual(stream.metadata, metadata)
        self.assertEqual([2], method.reset_after_forward_calls)

    def test_novel_domain_stream_marks_persistent_episode_recovery_not_applicable(self):
        stream = build_stream(self.datasets, "novel_domain", seed=9, novel_domain_idx=1)
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            ordered_stream_test(
                self.datasets, DiagnosticMethod(), self._args("novel_domain"), paths, stream
            )
            self._assert_complete_evidence(paths, stream, "not_applicable")

    def test_single_domain_segments_emit_complete_evidence_and_boundary_resets(self):
        stream, segments = build_single_domain_stream(self.datasets, seed=9)
        method = DiagnosticMethod()
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            ordered_stream_test(
                self.datasets, method, self._args("single_domain"), paths, stream, segments
            )
            self._assert_complete_evidence(paths, stream, "not_applicable")

        # One reset separates the two domain segments; the second is the final cleanup.
        self.assertEqual([1, 2], method.reset_after_forward_calls)

    def test_mixed_memory_bytes_availability_fails_the_run(self):
        stream = build_stream(self.datasets, "iid_mixed", seed=9)
        method = MixedMemoryAvailabilityMethod()
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            with self.assertRaisesRegex(ValueError, "memory_bytes must be available"):
                ordered_stream_test(
                    self.datasets, method, self._args("iid_mixed"), paths, stream
                )

    def test_reference_trace_is_paired_with_a_verified_stream_fingerprint(self):
        stream = build_stream(self.datasets, "iid_mixed", seed=9)
        with tempfile.TemporaryDirectory() as directory:
            baseline_root = Path(directory) / "baseline"
            baseline_root.mkdir()
            baseline_paths = self._paths(baseline_root)
            ordered_stream_test(
                self.datasets, DiagnosticMethod(), self._args("iid_mixed"), baseline_paths, stream
            )

            adapted_paths = self._paths(Path(directory) / "adapted")
            adapted_args = self._args("iid_mixed")
            adapted_args.reference_trace = str(baseline_paths["trace"])
            ordered_stream_test(
                self.datasets, DiagnosticMethod(), adapted_args, adapted_paths, stream,
                reference_identity=baseline_paths["reference_identity"],
            )

            summary = json.loads(adapted_paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual("computed", summary["negative_adaptation_rate"]["status"])


if __name__ == "__main__":
    unittest.main()
