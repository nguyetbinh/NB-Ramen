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
from main import _oracle_gradient_summary, ordered_stream_test  # noqa: E402
from methods.NoAdapt import NoAdapt  # noqa: E402
from streams import (  # noqa: E402
    build_open_set_stream,
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
        self.open_set_split_version = "ordered-stream-test-split-v1"
        self.known_class_ids = (0,)
        self.unknown_class_ids = (1,)

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


def _base_open_set_logits(images):
    """Give ID and OOD rows distinct, deterministic base-model energies."""
    is_unknown = images[:, 0].to(torch.bool)
    logits = torch.empty((len(images), 2), device=images.device)
    logits[~is_unknown] = torch.tensor([3.0, -3.0], device=images.device)
    logits[is_unknown] = torch.tensor([-1.0, -1.0], device=images.device)
    return logits


class CountingLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        return _base_open_set_logits(images)


class AdaptedReturnedLogitsMethod:
    """Expose base energy as pre-adaptation evidence, then return changed logits."""

    def __init__(self):
        self.forward_calls = 0
        self._diagnostics = {}

    def __call__(self, images):
        self.forward_calls += 1
        pre_logits = _base_open_set_logits(images)
        self._diagnostics = {
            "pre_adaptation_prediction": pre_logits.argmax(-1).detach(),
            "pre_adaptation_ood_score": -torch.logsumexp(pre_logits.detach(), dim=1),
        }
        # Change the ID decision and shift OOD energy, proving that both
        # classification and detection summaries use their matching phase.
        output = pre_logits + images[:, :1] * 4.0
        is_unknown = images[:, 0].to(torch.bool)
        output[~is_unknown] = output[~is_unknown].flip(dims=(1,))
        return output

    def get_diagnostics(self):
        return self._diagnostics

    def reset(self):
        pass


class AdmissionLogitsMethod(AdaptedReturnedLogitsMethod):
    """Predict model indices from test inputs with original IDs 5/10/20/30."""

    def __call__(self, images):
        logits = torch.stack((images[:, 0] != 10, images[:, 0] == 10), dim=1).float() * 3
        admitted = (images[:, 0] == 10) | (images[:, 0] == 20)
        self._diagnostics = {
            "pre_adaptation_prediction": logits.argmax(-1),
            "pre_adaptation_ood_score": -torch.logsumexp(logits, dim=1),
            "admission_prediction": logits.argmax(-1),
            "admission_normalized_entropy": torch.where(admitted, 0.25, 0.75),
            "admitted_to_memory": admitted,
        }
        return logits


class OrderedStreamEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.datasets = TensorMultiDomainDataset()

    def test_oracle_reductions_use_only_paired_defined_rows(self):
        rows = [
            {"ramen_vs_oracle_id_cosine": 0.0, "consensus_vs_oracle_id_cosine": 0.3,
             "ramen_vs_oracle_id_sign_disagreement": 0.8,
             "consensus_vs_oracle_id_sign_disagreement": 0.5,
             "retrieved_ood_fraction": 0.0, "retrieved_ood_weight_fraction": 0.0,
             "consensus_diagnostic_applied": True,
             "consensus_wrong_sign_coordinate_count": 2,
             "consensus_wrong_sign_removed_count": 1,
             "consensus_correct_sign_coordinate_count": 3,
             "consensus_correct_sign_removed_count": 1},
            {"ramen_vs_oracle_id_cosine": 0.0, "consensus_vs_oracle_id_cosine": None,
             "ramen_vs_oracle_id_sign_disagreement": 0.8,
             "consensus_vs_oracle_id_sign_disagreement": None,
             "retrieved_ood_fraction": 0.0, "retrieved_ood_weight_fraction": 0.0,
             "consensus_diagnostic_applied": False,
             "consensus_wrong_sign_coordinate_count": 0,
             "consensus_wrong_sign_removed_count": 0,
             "consensus_correct_sign_coordinate_count": 0,
             "consensus_correct_sign_removed_count": 0},
        ]
        summary = _oracle_gradient_summary(rows)
        self.assertAlmostEqual(0.3, summary["gdc_reduction_mean"])
        self.assertAlmostEqual(0.3, summary["sdr_reduction_mean"])
        self.assertEqual(0.5, summary["consensus_wrong_sign_removal_rate"])
        self.assertAlmostEqual(1 / 3, summary["consensus_correct_sign_removal_rate"])

    def _open_set_stream(self):
        return build_open_set_stream(
            self.datasets, "iid_mixed", seed=7, ood_ratio=0.5,
        )

    def test_admission_summary_separates_remapped_id_accuracy_from_ood(self):
        self.datasets.datasets = (
            TensorDomainDataset(0, [5, 10, 20, 30]),
            TensorDomainDataset(1, [5, 10, 20, 30]),
        )
        self.datasets.known_class_ids = (5, 10)
        self.datasets.unknown_class_ids = (20, 30)
        stream = self._open_set_stream()
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            ordered_stream_test(
                self.datasets, AdmissionLogitsMethod(), self._args("iid_mixed"), paths, stream,
            )
            summary = json.loads(paths["summary"].read_text())
        self.assertEqual(4, summary["schema_version"])
        self.assertEqual({
            "admitted_count": 4, "rejected_count": 4, "admission_rate": 0.5,
            "mean_normalized_entropy": 0.5,
            "admitted_id_count": 2, "rejected_id_count": 2,
            "admitted_ood_count": 2, "rejected_ood_count": 2,
            "admitted_id_pseudo_label_accuracy": 1.0,
            "rejected_id_pseudo_label_accuracy": 1.0,
            "admitted_ood_fraction": 0.5, "rejected_ood_fraction": 0.5,
        }, summary["admission_diagnostics"])

    @staticmethod
    def _trace_rows(paths):
        return [json.loads(line) for line in paths["trace"].read_text().splitlines()]

    def test_noadapt_pre_and_post_scores_are_equal_without_an_extra_forward(self):
        stream = self._open_set_stream()
        method = object.__new__(NoAdapt)
        torch.nn.Module.__init__(method)
        method.model = CountingLogitModel()
        method.last_diagnostics = {}

        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            ordered_stream_test(
                self.datasets, method, self._args("iid_mixed"), paths, stream,
            )
            rows = self._trace_rows(paths)
            summary = json.loads(paths["summary"].read_text())

        expected_calls = math.ceil(len(stream) / self._args("iid_mixed").batch_size)
        self.assertEqual(expected_calls, method.model.forward_calls)
        for row in rows:
            self.assertEqual(
                row["pre_adaptation_ood_score"], row["post_adaptation_ood_score"],
            )
        pre = summary["open_set"]["pre_adaptation_detection"]
        post = summary["open_set"]["post_adaptation_detection"]
        for field in ("auroc", "fpr95", "fpr95_threshold", "ood_recall_at_fpr95", "h_score"):
            self.assertEqual(pre[field], post[field])

    def test_post_score_uses_adapted_returned_logits_and_remains_single_call(self):
        stream = self._open_set_stream()
        method = AdaptedReturnedLogitsMethod()

        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(directory)
            ordered_stream_test(
                self.datasets, method, self._args("iid_mixed"), paths, stream,
            )
            rows = self._trace_rows(paths)
            summary = json.loads(paths["summary"].read_text())

        expected_calls = math.ceil(len(stream) / self._args("iid_mixed").batch_size)
        self.assertEqual(expected_calls, method.forward_calls)
        id_rows = [row for row in rows if not row["is_ood"]]
        ood_rows = [row for row in rows if row["is_ood"]]
        self.assertTrue(all(row["pre_adaptation_ood_score"] == row["post_adaptation_ood_score"] for row in id_rows))
        self.assertTrue(all(row["pre_adaptation_ood_score"] != row["post_adaptation_ood_score"] for row in ood_rows))
        self.assertEqual(1.0, summary["open_set"]["pre_adaptation_detection"]["auroc"])
        self.assertEqual(0.0, summary["open_set"]["post_adaptation_detection"]["auroc"])
        self.assertEqual(1.0, summary["open_set"]["pre_adaptation_detection"]["id_accuracy"])
        self.assertEqual(0.0, summary["open_set"]["post_adaptation_detection"]["id_accuracy"])
        self.assertEqual(0.0, summary["open_set"]["auroc"])
        self.assertEqual(0.0, summary["open_set"]["id_accuracy"])

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
