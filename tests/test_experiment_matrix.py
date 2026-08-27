import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from src.runtime.experiment_matrix import (
    IncompleteRunError,
    build_command,
    build_experiment_matrix,
    execute_matrix,
    main as matrix_main,
    make_run_id,
    preflight,
    validate_completed_run,
)
from src.evaluation.evidence import SUMMARY_SCHEMA_VERSION, TRACE_SCHEMA_VERSION
from src.runtime.artifact_provenance import (
    CIFAR100C_OFFICIAL_ACQUISITION,
    SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
    default_sidecar_path,
    resolve_clip_model,
)


def _fingerprint(metadata, references):
    clean_metadata = dict(metadata)
    clean_metadata.pop("fingerprint", None)
    encoded = json.dumps(
        {"metadata": clean_metadata, "references": references},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_valid_evidence(run):
    run.run_dir.mkdir(parents=True)
    references = [[0, 0]]
    metadata = {
        "format_version": 1,
        "mode": run.stream_mode,
        "seed": run.seed,
        "domain_lengths": [1],
        "domain_weights": [1.0],
        "block_size": run.stream_block_size,
        "domain_names": ["domain-a"],
        "parameters": {
            "gradual_sharpness": 4.0,
            "sample_budget": None,
            "novel_domain_idx": None,
            "novel_release_fraction": None,
            "novel_release_timestep": None,
            "correlation_strength": None,
            "burst_size": None,
        },
        "num_samples": 1,
    }
    if run.max_eval_samples is not None:
        metadata["evaluation_budget"] = {
            "truncation_strategy": "deterministic_prefix",
            "evidence_scope": "cost_limited",
            "cost_limited_evidence": True,
            "full_sample_count": 2,
            "full_stream_fingerprint": "f" * 64,
            "retained_sample_count": 1,
            "dropped_sample_count": 1,
        }
    fingerprint = _fingerprint(metadata, references)
    metadata["fingerprint"] = fingerprint
    args = {
        "dataset": run.dataset,
        "model": run.model,
        "tta_algo": run.method,
        "tta_mode": "mixed",
        "batch_size": run.batch_size,
        "seed": run.seed,
        "stream_seed": run.seed,
        "max_eval_samples": run.max_eval_samples,
        "stream_block_size": run.stream_block_size,
        "device_request": run.device,
        "metric_window_size": run.metric_window_size,
        "metric_window_stride": run.metric_window_stride,
        "config_path": str(run.config_path) if run.config_path else None,
        "reference_trace": str(run.reference_trace) if run.reference_trace else None,
        "artifact_provenance": run.artifact_provenance,
        "data_root": str(run.data_root),
    }
    model_artifact = resolve_clip_model(run.model)
    model_artifact.update({
        "status": "verified", "path": str(Path.home() / ".cache" / "clip" / model_artifact["filename"]),
        "actual_sha256": model_artifact["expected_sha256"], "size_bytes": 123,
    })
    dataset_key = run.dataset.lower()
    dataset_root = run.data_root / (
        "corruption/CIFAR-100-C" if run.dataset == "CIFAR100C" else "domainbed/domain_net"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run.run_id,
        "args": args,
        "config": run.config_data,
        "device": "cpu" if run.device == "auto" else run.device,
        "dataset": {"name": run.dataset, "environments": ["domain-a"], "original_domain_lengths": [1]},
        "stream": metadata,
        "artifacts": {
            "status": "verified", "mode": run.artifact_provenance,
            "model": model_artifact,
            "dataset": {
                "status": "verified", "schema_version": ARTIFACT_SCHEMA_VERSION,
                "dataset": dataset_key, "root": str(dataset_root),
                "sidecar": str(default_sidecar_path(dataset_root, dataset_key).absolute()),
                "verified_exact": run.artifact_provenance == "exact",
                "content_algorithm": "sha256", "root_digest": "b" * 64,
                "sidecar_sha256": "c" * 64, "file_count": 2,
                "acquisition": CIFAR100C_OFFICIAL_ACQUISITION if run.dataset == "CIFAR100C" else {},
            },
        },
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": run.run_id,
        "num_samples": 1,
        "micro_accuracy": 1.0,
        "macro_domain_accuracy": 1.0,
        "worst_domain_accuracy": 1.0,
        "domain_accuracies": {"domain-a": 1.0},
        "domain_sample_counts": {"domain-a": 1},
        "sliding_window": {
            "window_size": run.metric_window_size,
            "stride": run.metric_window_stride,
            "values": (
                [{"start_timestep": 0, "end_timestep": 0, "accuracy": 1.0}]
                if run.metric_window_size == 1 else []
            ),
        },
        "post_shift_recovery_time": (
            {
                "status": "computed",
                "definition": "full-window recovery within each persistent-domain episode",
                "window_size": run.metric_window_size,
                "shifts": [],
            }
            if run.stream_mode in {"block", "recurring", "bursty"} else
            {
                "status": "not_applicable",
                "reason": "stream does not define discrete persistent-domain episodes",
            }
        ),
        "negative_adaptation_rate": (
            {
                "status": "reference_required",
                "reason": "pass --reference_trace from NoAdapt on the identical stream",
            }
            if run.reference_trace is None else
            {
                "status": "computed",
                "value": 0.0,
                "negative_windows": 0,
                "total_windows": 1,
                "window_size": run.metric_window_size,
                "stride": run.metric_window_stride,
                "reference_trace": str(run.reference_trace),
            }
        ),
        "routing_diagnostics": {
            "status": "unavailable",
            "normalized_mutual_information": None,
            "adjusted_rand_index": None,
            "context_purity": None,
            "number_of_discovered_contexts": None,
            "assignment_churn_rate": None,
        },
        "peak_device_memory_bytes": None,
        "device_memory": {"status": "not_applicable", "kind": "unsupported", "bytes": None},
        "method_memory": {
            "status": "unavailable",
            "reason": "method did not expose retained support-memory bytes",
            "unit": "bytes",
            "max_retained_bytes": None,
            "final_retained_bytes": None,
        },
        "forward_latency": {
            "status": "computed",
            "definition": "per-sample share of synchronized tta_model forward wall-clock latency; includes adaptation and prediction",
            "unit": "milliseconds",
            "total_ms": 1.0,
            "mean_per_sample_ms": 1.0,
            "median_per_sample_ms": 1.0,
        },
        "throughput": {
            "status": "computed",
            "definition": "completed samples divided by total synchronized tta_model forward wall-clock latency",
            "unit": "samples_per_second",
            "samples_per_second": 1000.0,
        },
        "retrieval_latency": {
            "status": "unavailable",
            "reason": "retrieval is interleaved with causal insertion and adaptation; isolating it would require invasive instrumentation and device synchronization that would perturb the measured path",
        },
        "stream_fingerprint": fingerprint,
    }
    stream = {"metadata": metadata, "references": references, "fingerprint": fingerprint}
    (run.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run.run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run.run_dir / "stream.json").write_text(json.dumps(stream), encoding="utf-8")
    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run.run_id,
        "timestep": 0,
        "sample_idx": 0,
        "ground_truth_domain": 0,
        "ground_truth_class": 1,
        "prediction": 1,
        "correct": True,
        "predicted_entropy": 0.25,
        "inferred_context": None,
        "memory_size": 0,
        "num_active_contexts": None,
        "memory_bytes": None,
        "latency_ms": 1.0,
    }
    (run.run_dir / "trace.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")


def _mutate_trace(run, mutate):
    path = run.run_dir / "trace.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    mutate(row)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _mutate_summary(run, mutate):
    path = run.run_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    mutate(summary)
    path.write_text(json.dumps(summary), encoding="utf-8")


def _extend_to_two_sample_memory_timeline(run, memory_bytes):
    """Keep otherwise-valid evidence aligned while replacing memory evidence."""
    stream_path = run.run_dir / "stream.json"
    manifest_path = run.run_dir / "manifest.json"
    summary_path = run.run_dir / "summary.json"
    trace_path = run.run_dir / "trace.jsonl"
    stream = json.loads(stream_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    first_row = json.loads(trace_path.read_text(encoding="utf-8"))

    references = [[0, 0], [0, 0]]
    metadata = stream["metadata"]
    metadata["num_samples"] = 2
    metadata.pop("fingerprint", None)
    fingerprint = _fingerprint(metadata, references)
    metadata["fingerprint"] = fingerprint
    stream.update({"metadata": metadata, "references": references, "fingerprint": fingerprint})
    manifest["stream"] = metadata

    summary["num_samples"] = 2
    summary["domain_sample_counts"] = {"domain-a": 2}
    summary["forward_latency"].update({
        "total_ms": 2.0,
        "mean_per_sample_ms": 1.0,
        "median_per_sample_ms": 1.0,
    })
    summary["stream_fingerprint"] = fingerprint

    second_row = dict(first_row, timestep=1)
    first_row["memory_bytes"], second_row["memory_bytes"] = memory_bytes
    trace_path.write_text(
        json.dumps(first_row) + "\n" + json.dumps(second_row) + "\n", encoding="utf-8"
    )
    stream_path.write_text(json.dumps(stream), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


class ExperimentMatrixTests(unittest.TestCase):
    def test_profile_resume_replays_counters_even_when_summary_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory) / "profile-config"
            (config_root / "CIFAR100C").mkdir(parents=True)
            (config_root / "default").mkdir()
            (config_root / "CIFAR100C" / "LatentRamen.yaml").write_text(
                "max_capacity: 2\ntopk: 1\noptimizer: signsgd\nlr: .01\n"
                "capacity_scope: per_class\ninclude_current: true\nretrieval_profile: causal_sync_v1\n"
            )
            (config_root / "default" / "NoAdapt.yaml").write_text("optimizer: signsgd\nlr: .000000001\n")
            baseline, run = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("iid_mixed",), methods=("LatentRamen",), seeds=(0,),
                evidence_dir=directory, data_root=directory, device="cpu", config_dir=config_root,
                max_eval_samples=1,
            )
            _write_valid_evidence(baseline)
            _write_valid_evidence(run)
            trace_path, summary_path = run.run_dir / "trace.jsonl", run.run_dir / "summary.json"
            row = json.loads(trace_path.read_text())
            row.update({
                "inferred_context": 0, "memory_size": 1, "memory_bytes": 32,
                "admission_prediction": 1, "admission_normalized_entropy": .25, "admitted_to_memory": True,
                "retrieval_profile": "causal_sync_v1", "retrieval_elapsed_ms": .1,
                "retrieval_candidate_count": 1, "retrieval_eligible_candidate_count": 1,
                "retrieval_returned_support_count": 1, "retrieval_active_class_count": 1,
            })
            trace_path.write_text(json.dumps(row) + "\n")
            summary = json.loads(summary_path.read_text())
            summary["retrieval_latency"] = {
                "status": "computed", "profile": "causal_sync_v1",
                "definition": "device-synchronized query-only interval after causal insertion; synchronization perturbs execution and is not comparable to ordinary end-to-end latency",
                "unit": "milliseconds", "total_ms": .1, "p50_ms": .1, "p95_ms": .1, "max_ms": .1,
                "candidate_count": {"min": 1, "p50": 1, "p95": 1, "max": 1},
                "eligible_candidate_count": {"min": 1, "p50": 1, "p95": 1, "max": 1},
                "returned_support_count": {"min": 1, "p50": 1, "p95": 1, "max": 1},
                "active_class_count": {"min": 1, "p50": 1, "p95": 1, "max": 1},
            }
            summary["admission_diagnostics"] = {
                "admitted_count": 1, "rejected_count": 0, "admission_rate": 1.0,
                "mean_normalized_entropy": .25, "admitted_pseudo_label_accuracy": 1.0,
                "rejected_pseudo_label_accuracy": None, "admitted_contamination_rate": 0.0,
            }
            summary["routing_diagnostics"] = {
                "status": "available", "normalized_mutual_information": 1.0,
                "adjusted_rand_index": 1.0, "context_purity": 1.0,
                "number_of_discovered_contexts": 1, "assignment_churn_rate": 0.0,
            }
            summary["method_memory"] = {
                "status": "computed", "definition": (
                    "exact bytes retained by the method support memory at the state exposed for each emitted sample; "
                    "batch-atomic methods repeat the post-admission batch state"
                ),
                "unit": "bytes", "max_retained_bytes": 32, "final_retained_bytes": 32,
            }
            summary_path.write_text(json.dumps(summary))
            validate_completed_run(run)
            row["retrieval_candidate_count"] = 0
            trace_path.write_text(json.dumps(row) + "\n")
            summary["retrieval_latency"]["candidate_count"] = {"min": 0, "p50": 0, "p95": 0, "max": 0}
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(IncompleteRunError, "causal replay"):
                validate_completed_run(run)
            row["memory_size"] = 0
            trace_path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(IncompleteRunError, "memory_size disagrees"):
                validate_completed_run(run)

    def test_entropy_gated_method_is_explicitly_selectable_without_expanding_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("EntropyGatedLatentRamen",),
                seeds=(0,), evidence_dir=directory, data_root=directory, device="cpu", max_eval_samples=1,
            )
        self.assertEqual(["NoAdapt", "EntropyGatedLatentRamen"], [run.method for run in runs])
        self.assertEqual(330, len(build_experiment_matrix(seeds=(0, 1, 2), max_eval_samples=1)))

    def test_admission_evidence_summary_is_recomputed_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("iid_mixed",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=directory, data_root=directory, device="cpu",
            )[0]
            _write_valid_evidence(run)
            row = json.loads((run.run_dir / "trace.jsonl").read_text())
            row.update({
                "admission_prediction": 1,
                "admission_normalized_entropy": .25,
                "admitted_to_memory": True,
            })
            (run.run_dir / "trace.jsonl").write_text(json.dumps(row) + "\n")
            summary_path = run.run_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["admission_diagnostics"] = {
                "admitted_count": 1, "rejected_count": 0, "admission_rate": 1.0,
                "mean_normalized_entropy": .25, "admitted_pseudo_label_accuracy": 1.0,
                "rejected_pseudo_label_accuracy": None, "admitted_contamination_rate": 0.0,
            }
            summary_path.write_text(json.dumps(summary))
            validate_completed_run(run)
            summary["admission_diagnostics"]["admitted_count"] = 0
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(IncompleteRunError, "admission_diagnostics"):
                validate_completed_run(run)

    def test_gated_resume_rejects_entropy_decision_that_disagrees_with_config(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("iid_mixed",), methods=("EntropyGatedLatentRamen",),
                seeds=(0,), evidence_dir=directory, data_root=directory, device="cpu", max_eval_samples=1,
            )
            baseline, gated = runs
            _write_valid_evidence(baseline)
            _write_valid_evidence(gated)
            trace_path = gated.run_dir / "trace.jsonl"
            row = json.loads(trace_path.read_text())
            row.update({
                "admission_prediction": 1,
                "admission_normalized_entropy": .25,
                "admitted_to_memory": False,
            })
            trace_path.write_text(json.dumps(row) + "\n")
            summary_path = gated.run_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["admission_diagnostics"] = {
                "admitted_count": 0, "rejected_count": 1, "admission_rate": 0.0,
                "mean_normalized_entropy": .25, "admitted_pseudo_label_accuracy": None,
                "rejected_pseudo_label_accuracy": 1.0, "admitted_contamination_rate": None,
            }
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(IncompleteRunError, "disagrees with entropy gate"):
                validate_completed_run(gated)

    def test_gated_resume_requires_complete_admission_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("iid_mixed",), methods=("EntropyGatedLatentRamen",),
                seeds=(0,), evidence_dir=directory, data_root=directory, device="cpu", max_eval_samples=1,
            )
            baseline, gated = runs
            _write_valid_evidence(baseline)
            _write_valid_evidence(gated)
            with self.assertRaisesRegex(IncompleteRunError, "lacks complete admission evidence"):
                validate_completed_run(gated)
            trace_path = gated.run_dir / "trace.jsonl"
            row = json.loads(trace_path.read_text())
            row["admitted_to_memory"] = True
            trace_path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(IncompleteRunError, "all present or all absent"):
                validate_completed_run(gated)

    def test_matrix_preflight_requests_semantic_dataset_validation(self):
        with patch(
            "src.runtime.experiment_matrix.validate_dataset_layout",
            return_value={"dataset": "CIFAR100C", "valid": True},
        ) as validator:
            result = preflight(("CIFAR100C", "CIFAR100C"), "/tmp/data")

        self.assertEqual(1, len(result))
        validator.assert_called_once_with(Path("/tmp/data").resolve(), "CIFAR100C", deep=True)

    def test_full_grid_cardinality_order_and_baseline_pairing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = build_experiment_matrix(seeds=(1, 7), evidence_dir=temporary_directory)
        self.assertEqual(2 * 5 * 2 * 11, len(runs))
        first_cell = runs[:11]
        self.assertEqual(
            [
                "NoAdapt", "Tent", "Ramen", "CausalRamen", "StructuredAtomicRamen", "RandomMemoryRamen",
                "SameClassRamen", "GlobalNearestRamen", "ContextOnlyRamen",
                "OracleLatentRamen", "LatentRamen",
            ],
            [run.method for run in first_cell],
        )
        baseline_trace = first_cell[0].run_dir / "trace.jsonl"
        self.assertTrue(all(run.reference_trace == baseline_trace for run in first_cell[1:]))
        self.assertTrue(all(run.device == "auto" and run.max_eval_samples is None for run in first_cell))
        self.assertTrue(all(run.stream_block_size == 64 for run in first_cell))

    def test_default_batch_size_preserves_identity_and_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            default = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(3,),
                evidence_dir=temporary_directory,
            )[0]
            explicit = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(3,),
                evidence_dir=temporary_directory, batch_size=100,
            )[0]
        self.assertEqual(100, default.batch_size)
        self.assertEqual(default.run_id, explicit.run_id)
        self.assertNotIn("-bs-", default.run_id)
        self.assertEqual("100", build_command(default)[build_command(default).index("--batch_size") + 1])

    def test_nondefault_batch_size_binds_identity_command_and_resume_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            default = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(3,),
                evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu",
            )[0]
            atomic = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(3,),
                evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu", batch_size=1,
            )[0]
            self.assertNotEqual(default.run_id, atomic.run_id)
            self.assertIn("-bs-1-", atomic.run_id)
            command = build_command(atomic)
            self.assertEqual("1", command[command.index("--batch_size") + 1])
            with self.assertRaisesRegex(ValueError, "batch_size override"):
                build_command(atomic, batch_size=100)
            _write_valid_evidence(atomic)
            validate_completed_run(atomic)
            manifest_path = atomic.run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["args"]["batch_size"] = 100
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(IncompleteRunError, "manifest.args.batch_size"):
                validate_completed_run(atomic)

    def test_matrix_cli_plans_nondefault_batch_size(self):
        with tempfile.TemporaryDirectory() as temporary_directory, contextlib.redirect_stdout(io.StringIO()) as output:
            result = matrix_main([
                "--dataset", "CIFAR100C", "--stream", "block", "--method", "NoAdapt", "--seed", "3",
                "--evidence-dir", temporary_directory, "--batch-size", "1",
            ])
        self.assertEqual(0, result)
        payload = json.loads(output.getvalue())
        self.assertIn("-bs-1-", payload["runs"][0]["run_id"])
        command = payload["commands"][0]
        self.assertEqual("1", command[command.index("--batch_size") + 1])

    def test_default_block_size_preserves_canonical_identity_and_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            default = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(3,),
                evidence_dir=temporary_directory,
            )[0]
            explicit = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(3,),
                evidence_dir=temporary_directory, stream_block_size=64,
            )[0]
        self.assertEqual(default.run_id, explicit.run_id)
        self.assertNotIn("-blk-", default.run_id)
        self.assertNotIn("--stream_block_size", build_command(default))

    def test_cost_limited_custom_block_size_binds_identity_and_command(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(3,),
                evidence_dir=temporary_directory, max_eval_samples=5, stream_block_size=17,
            )[0]
        self.assertIn("-blk-17-", run.run_id)
        command = build_command(run)
        self.assertEqual("17", command[command.index("--stream_block_size") + 1])
        with self.assertRaisesRegex(ValueError, "stream_block_size override"):
            build_command(run, stream_block_size=64)

    def test_custom_block_size_requires_cost_limited_budget_and_is_positive(self):
        common = dict(datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(0,))
        with self.assertRaisesRegex(ValueError, "requires max_eval_samples"):
            build_experiment_matrix(**common, stream_block_size=17)
        for value in (0, -1, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                build_experiment_matrix(**common, max_eval_samples=1, stream_block_size=value)

    def test_matrix_cli_plans_custom_block_size_with_exact_option_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory, contextlib.redirect_stdout(io.StringIO()) as output:
            result = matrix_main([
                "--dataset", "CIFAR100C", "--stream", "block", "--method", "NoAdapt", "--seed", "3",
                "--evidence-dir", temporary_directory, "--max-eval-samples", "2", "--stream-block-size", "8",
            ])
        self.assertEqual(0, result)
        payload = json.loads(output.getvalue())
        self.assertIn("-blk-8-", payload["runs"][0]["run_id"])
        command = payload["commands"][0]
        self.assertEqual("8", command[command.index("--stream_block_size") + 1])

    def test_resume_rejects_manifest_or_stream_block_size_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(0, 1),
                evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu",
                max_eval_samples=1, stream_block_size=17,
            )
            for run, filename, mutate in (
                (runs[0], "manifest.json", lambda payload: payload["args"].__setitem__("stream_block_size", 64)),
                (runs[1], "stream.json", lambda payload: payload["metadata"].__setitem__("block_size", 64)),
            ):
                with self.subTest(filename=filename):
                    _write_valid_evidence(run)
                    path = run.run_dir / filename
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    mutate(payload)
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    if filename == "stream.json":
                        manifest_path = run.run_dir / "manifest.json"
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        manifest["stream"] = payload["metadata"]
                        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(IncompleteRunError, "block_size"):
                        validate_completed_run(run)

    def test_adapted_subset_includes_baseline_first(self):
        runs = build_experiment_matrix(
            datasets=("CIFAR100C",), streams=("recurring",), methods=("LatentRamen",), seeds=(0,),
        )
        self.assertEqual(["NoAdapt", "LatentRamen"], [run.method for run in runs])

    def test_all_roadmap_streams_are_schedulable_without_expanding_the_default_grid(self):
        extended = ("novel_domain", "class_domain_correlated", "bursty")
        runs = build_experiment_matrix(
            datasets=("CIFAR100C",), streams=extended, methods=("NoAdapt",), seeds=(0,),
        )
        self.assertEqual(extended, tuple(run.stream_mode for run in runs))
        default = build_experiment_matrix(
            datasets=("CIFAR100C",), methods=("NoAdapt",), seeds=(0,),
        )
        self.assertEqual(
            ("iid_mixed", "block", "gradual", "recurring", "imbalanced"),
            tuple(run.stream_mode for run in default),
        )

    def test_persistent_and_novel_domain_recovery_evidence_resume_strictly(self):
        def mutate_to_computed(recovery):
            recovery.clear()
            recovery.update({
                "status": "computed",
                "definition": "full-window recovery within each persistent-domain episode",
                "window_size": 1,
                "shifts": [],
            })

        cases = (
            ("bursty", "computed", lambda recovery: recovery.__setitem__(
                "shifts", [{"status": "recovered", "recovery_samples": -1}]
            )),
            ("novel_domain", "not_applicable", mutate_to_computed),
        )
        for stream_mode, expected_status, mutate_recovery in cases:
            with self.subTest(stream_mode=stream_mode), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("DomainNet",), streams=(stream_mode,), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu",
                )[0]
                _write_valid_evidence(run)
                recovery = json.loads(
                    (run.run_dir / "summary.json").read_text(encoding="utf-8")
                )["post_shift_recovery_time"]
                self.assertEqual(expected_status, recovery["status"])
                with patch("src.runtime.experiment_matrix.validate_dataset_layout", return_value={"valid": True}):
                    outcomes = execute_matrix(
                        (run,), data_root=temporary_directory, resume=True,
                        runner=lambda *_args, **_kwargs: self.fail("model launched"),
                    )
                self.assertEqual("skipped", outcomes[0]["status"])
                _mutate_summary(run, lambda summary: mutate_recovery(summary["post_shift_recovery_time"]))
                with self.assertRaisesRegex(IncompleteRunError, "post_shift_recovery_time"):
                    validate_completed_run(run)

    def test_support_baseline_subset_is_paired_and_uses_method_configs(self):
        methods = (
            "CausalRamen", "RandomMemoryRamen", "SameClassRamen",
            "GlobalNearestRamen", "ContextOnlyRamen",
        )
        runs = build_experiment_matrix(
            datasets=("CIFAR100C",), streams=("block",), methods=methods, seeds=(6,),
        )
        self.assertEqual(("NoAdapt", *methods), tuple(run.method for run in runs))
        baseline_trace = runs[0].run_dir / "trace.jsonl"
        for run in runs[1:]:
            self.assertEqual(baseline_trace, run.reference_trace)
            self.assertEqual(f"{run.method}.yaml", run.config_path.name)
            self.assertNotEqual("missing", run.config_hash)
            command = build_command(run)
            self.assertEqual(run.method, command[command.index("--tta_algo") + 1])
            self.assertEqual(str(baseline_trace), command[command.index("--reference_trace") + 1])

    def test_support_baseline_completed_evidence_resumes_strictly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            baseline, causal = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("CausalRamen",), seeds=(0,),
                evidence_dir=temporary_directory, max_eval_samples=1, data_root=temporary_directory,
                device="cpu",
            )
            _write_valid_evidence(baseline)
            _write_valid_evidence(causal)
            with patch("src.runtime.experiment_matrix.validate_dataset_layout", return_value={"valid": True}):
                outcomes = execute_matrix(
                    (baseline, causal), data_root=temporary_directory, resume=True,
                    runner=lambda *_args, **_kwargs: self.fail("model launched"),
                )
        self.assertEqual(["skipped", "skipped"], [outcome["status"] for outcome in outcomes])

    def test_identity_separates_budget_device_and_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / "cfg" / "CIFAR100C"
            config.mkdir(parents=True)
            config_path = config / "Ramen.yaml"
            config_path.write_text("lr: 0.1\n", encoding="utf-8")
            kwargs = dict(datasets=("CIFAR100C",), streams=("block",), methods=("Ramen",),
                          seeds=(3,), evidence_dir=root / "evidence", config_dir=root / "cfg")
            full = build_experiment_matrix(**kwargs, device="cpu")[-1]
            limited = build_experiment_matrix(**kwargs, device="cpu", max_eval_samples=7)[-1]
            cuda = build_experiment_matrix(**kwargs, device="cuda")[-1]
            config_path.write_text("lr: 0.2\n", encoding="utf-8")
            changed_config = build_experiment_matrix(**kwargs, device="cpu")[-1]
        self.assertEqual(4, len({full.run_id, limited.run_id, cuda.run_id, changed_config.run_id}))
        self.assertIn("-full-cfg-", full.run_id)
        self.assertNotEqual(full.config_hash, changed_config.config_hash)

    def test_identity_and_command_include_artifact_provenance(self):
        common = dict(datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(3,))
        fast = build_experiment_matrix(**common, artifact_provenance="fast")[0]
        exact = build_experiment_matrix(**common, artifact_provenance="exact")[0]
        self.assertNotEqual(fast.run_id, exact.run_id)
        command = build_command(exact)
        self.assertEqual("exact", command[command.index("--artifact-provenance") + 1])

    def test_identity_and_execution_are_bound_to_canonical_data_root(self):
        common = dict(datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,))
        first = build_experiment_matrix(**common, data_root="~/dataset-a")[0]
        second = build_experiment_matrix(**common, data_root="~/dataset-b")[0]
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(Path("~/dataset-a").expanduser().resolve(), first.data_root)
        with self.assertRaisesRegex(ValueError, "data_root override contradicts planned identity"):
            execute_matrix((first,), data_root="~/dataset-b", resume=True,
                           runner=lambda *_args, **_kwargs: None)

    def test_execution_rejects_auto_device_before_preflight_or_runner(self):
        run = build_experiment_matrix(
            datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
            device="auto",
        )[0]
        with patch("src.runtime.experiment_matrix.preflight") as preflight_check:
            with self.assertRaisesRegex(ValueError, "requires an explicit device"):
                execute_matrix(
                    (run,), resume=True,
                    runner=lambda *_args, **_kwargs: self.fail("model launched"),
                )
        preflight_check.assert_not_called()

    def test_cli_resume_rejects_auto_device_during_argument_validation(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                matrix_main(["--resume", "--device", "auto"])
        self.assertEqual(2, error.exception.code)

    def test_resume_rejects_alternate_manifest_and_artifact_data_roots(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=temporary_directory, data_root="/planned/data",
            )[0]
            _write_valid_evidence(run)
            manifest_path = run.run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["args"]["data_root"] = "/alternate/data"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(IncompleteRunError, "manifest.args.data_root"):
                validate_completed_run(run)
            manifest["args"]["data_root"] = str(run.data_root)
            manifest["artifacts"]["dataset"]["root"] = "/alternate/data/domainbed/domain_net"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(IncompleteRunError, "artifacts.dataset.root"):
                validate_completed_run(run)

    def test_resume_rejects_missing_or_mismatched_artifact_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=temporary_directory,
            )[0]
            _write_valid_evidence(run)
            manifest_path = run.run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = manifest.pop("artifacts")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(IncompleteRunError, "artifacts"):
                validate_completed_run(run)
            manifest["artifacts"] = artifacts
            manifest["artifacts"]["mode"] = "exact"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(IncompleteRunError, "artifacts.mode"):
                validate_completed_run(run)

    def test_resume_rejects_tampered_artifact_report_fields(self):
        mutations = {
            "model host": lambda artifacts: artifacts["model"].__setitem__("url", "https://evil.invalid/model.pt"),
            "model hash": lambda artifacts: artifacts["model"].__setitem__("actual_sha256", "0" * 64),
            "dataset digest": lambda artifacts: artifacts["dataset"].__setitem__("root_digest", "not-a-hash"),
            "sidecar": lambda artifacts: artifacts["dataset"].__setitem__("sidecar", "/tmp/foreign.json"),
            "acquisition": lambda artifacts: artifacts["dataset"]["acquisition"].__setitem__("doi", "10.0/evil"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory,
                )[0]
                _write_valid_evidence(run)
                manifest_path = run.run_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest["artifacts"])
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(IncompleteRunError):
                    validate_completed_run(run)

    def test_negative_seed_is_unambiguous_and_duplicates_are_rejected(self):
        negative = make_run_id("CIFAR100C", "iid_mixed", -1, "NoAdapt")
        positive = make_run_id("CIFAR100C", "iid_mixed", 1, "NoAdapt")
        self.assertIn("seed-neg1", negative)
        self.assertNotEqual(negative, positive)
        with self.assertRaisesRegex(ValueError, "duplicate generated run ID"):
            build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(1, 1),
            )

    def test_huge_seed_is_rejected_before_planning_an_invalid_run_id(self):
        huge_seed = 10 ** 200
        with self.assertRaisesRegex(ValueError, "maximum is 128"):
            make_run_id("CIFAR100C", "iid_mixed", huge_seed, "NoAdapt")
        with self.assertRaisesRegex(ValueError, "maximum is 128"):
            build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("iid_mixed",), methods=("NoAdapt",),
                seeds=(huge_seed,),
            )

    def test_commands_use_run_identity_and_small_budget_windows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("CIFAR100C",), streams=("gradual",), methods=("LatentRamen",),
                seeds=(2,), evidence_dir=temporary_directory, device="cpu", max_eval_samples=7,
                data_root="~/matrix-data",
            )[1]
            command = build_command(run, data_root="~/matrix-data")
        self.assertTrue(Path(command[0]).is_absolute())
        self.assertTrue(Path(command[1]).is_absolute())
        self.assertEqual("cpu", command[command.index("--device") + 1])
        self.assertEqual("7", command[command.index("--max-eval-samples") + 1])
        self.assertEqual("7", command[command.index("--metric_window_size") + 1])
        self.assertEqual("7", command[command.index("--metric_window_stride") + 1])
        self.assertEqual(str(run.reference_trace), command[command.index("--reference_trace") + 1])
        with self.assertRaisesRegex(ValueError, "contradicts planned identity"):
            build_command(run, device="cuda")
        with self.assertRaisesRegex(ValueError, "contradicts planned identity"):
            build_command(run, max_eval_samples=None)
        with self.assertRaisesRegex(ValueError, "data_root override contradicts planned identity"):
            build_command(run, data_root="~/other-data")

    def test_to_dict_exports_identity_fields(self):
        run = build_experiment_matrix(
            datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
            device="mps", max_eval_samples=9,
        )[0]
        payload = run.to_dict()
        for field in ("device", "data_root", "max_eval_samples", "config_dir", "config_path", "config_hash", "config_data"):
            self.assertIn(field, payload)

    def test_valid_completed_artifact_is_skipped_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu",
            )[0]
            _write_valid_evidence(run)
            validate_completed_run(run)
            with patch("src.runtime.experiment_matrix.validate_dataset_layout", return_value={"valid": True}):
                outcomes = execute_matrix((run,), data_root=temporary_directory, resume=True,
                                          runner=lambda *_args, **_kwargs: self.fail("model launched"))
        self.assertEqual([{"run_id": run.run_id, "status": "skipped"}], outcomes)

    def test_strict_resume_requires_current_trace_and_summary_schemas(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=temporary_directory,
            )[0]
            _write_valid_evidence(run)
            validate_completed_run(run)

            _mutate_trace(run, lambda row: row.__setitem__("schema_version", 1))
            with self.assertRaisesRegex(IncompleteRunError, "trace\\[1\\]\\.schema_version"):
                validate_completed_run(run)

            _mutate_trace(run, lambda row: row.__setitem__("schema_version", TRACE_SCHEMA_VERSION))
            _mutate_summary(run, lambda summary: summary.__setitem__("schema_version", 1))
            with self.assertRaisesRegex(IncompleteRunError, "summary.schema_version"):
                validate_completed_run(run)

            _mutate_summary(
                run, lambda summary: summary.__setitem__("schema_version", SUMMARY_SCHEMA_VERSION)
            )
            validate_completed_run(run)

    def test_trace_requires_full_schema_and_exact_stream_identity(self):
        mutations = {
            "missing prediction": lambda row: row.pop("prediction"),
            "wrong sample": lambda row: row.__setitem__("sample_idx", 99),
            "wrong domain": lambda row: row.__setitem__("ground_truth_domain", 1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu",
                )[0]
                _write_valid_evidence(run)
                _mutate_trace(run, mutate)
                with self.assertRaises(IncompleteRunError):
                    validate_completed_run(run)

    def test_trace_rejects_malformed_or_contaminated_predictions(self):
        mutations = {
            "boolean prediction": lambda row: row.__setitem__("prediction", True),
            "integer correct": lambda row: row.__setitem__("correct", 1),
            "inconsistent correct": lambda row: row.__setitem__("correct", False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu",
                )[0]
                _write_valid_evidence(run)
                _mutate_trace(run, mutate)
                with self.assertRaises(IncompleteRunError):
                    validate_completed_run(run)

    def test_trace_memory_bytes_requires_null_or_nonnegative_integer(self):
        for value in (-1, True, 1.5, "64"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory,
                )[0]
                _write_valid_evidence(run)
                _mutate_trace(run, lambda row: row.__setitem__("memory_bytes", value))
                with self.assertRaisesRegex(IncompleteRunError, "memory_bytes"):
                    validate_completed_run(run)

    def test_summary_requires_metrics_and_valid_negative_adaptation_status(self):
        mutations = {
            "missing accuracy": lambda summary: summary.pop("micro_accuracy"),
            "missing routing": lambda summary: summary.pop("routing_diagnostics"),
            "invalid negative status": lambda summary: summary["negative_adaptation_rate"].__setitem__(
                "status", "computed"
            ),
            "contaminated accuracy": lambda summary: summary.__setitem__("micro_accuracy", 0.0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory, data_root=temporary_directory,
                )[0]
                _write_valid_evidence(run)
                _mutate_summary(run, mutate)
                with self.assertRaises(IncompleteRunError):
                    validate_completed_run(run)

    def test_adapted_negative_adaptation_is_recomputed_from_paired_trace(self):
        mutations = {
            "value": lambda metric: metric.__setitem__("value", 1.0),
            "counts": lambda metric: metric.__setitem__("negative_windows", 1),
            "reference": lambda metric: metric.__setitem__("reference_trace", "/tmp/foreign.jsonl"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                baseline, adapted = build_experiment_matrix(
                    datasets=("DomainNet",), streams=("block",), methods=("Ramen",), seeds=(0,),
                    evidence_dir=temporary_directory, max_eval_samples=1,
                )
                _write_valid_evidence(baseline)
                _write_valid_evidence(adapted)
                validate_completed_run(adapted)
                _mutate_summary(adapted, lambda summary: mutate(summary["negative_adaptation_rate"]))
                with self.assertRaisesRegex(IncompleteRunError, "negative_adaptation_rate"):
                    validate_completed_run(adapted)

    def test_adapted_negative_adaptation_rejects_paired_trace_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            baseline, adapted = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("Ramen",), seeds=(0,),
                evidence_dir=temporary_directory, max_eval_samples=1,
            )
            _write_valid_evidence(baseline)
            _write_valid_evidence(adapted)
            _mutate_trace(baseline, lambda row: row.__setitem__("ground_truth_class", 2))
            with self.assertRaisesRegex(IncompleteRunError, "negative adaptation"):
                validate_completed_run(adapted)

    def test_routing_diagnostics_are_recomputed_exactly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=temporary_directory,
            )[0]
            _write_valid_evidence(run)
            _mutate_trace(run, lambda row: row.__setitem__("inferred_context", 7))
            available = {
                "status": "available",
                "normalized_mutual_information": 1.0,
                "adjusted_rand_index": 1.0,
                "context_purity": 1.0,
                "number_of_discovered_contexts": 1,
                "assignment_churn_rate": 0.0,
            }
            _mutate_summary(run, lambda summary: summary.__setitem__("routing_diagnostics", available))
            validate_completed_run(run)
            _mutate_summary(
                run,
                lambda summary: summary["routing_diagnostics"].__setitem__(
                    "normalized_mutual_information", 0.5
                ),
            )
            with self.assertRaisesRegex(IncompleteRunError, "routing_diagnostics"):
                validate_completed_run(run)

    def test_domain_shift_recovery_is_recomputed_exactly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=temporary_directory,
            )[0]
            _write_valid_evidence(run)
            _mutate_summary(
                run,
                lambda summary: summary["post_shift_recovery_time"].__setitem__(
                    "shifts", [{"status": "recovered", "recovery_samples": -999}]
                ),
            )
            with self.assertRaisesRegex(IncompleteRunError, "post_shift_recovery_time"):
                validate_completed_run(run)

    def test_method_memory_is_recomputed_exactly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=temporary_directory,
            )[0]
            _write_valid_evidence(run)
            _mutate_trace(run, lambda row: row.__setitem__("memory_bytes", 64))
            expected = {
                "status": "computed",
                "definition": (
                    "exact bytes retained by the method support memory at the state exposed for each emitted sample; "
                    "batch-atomic methods repeat the post-admission batch state"
                ),
                "unit": "bytes",
                "max_retained_bytes": 64,
                "final_retained_bytes": 64,
            }
            _mutate_summary(run, lambda summary: summary.__setitem__("method_memory", expected))
            validate_completed_run(run)
            _mutate_summary(
                run, lambda summary: summary["method_memory"].__setitem__("max_retained_bytes", 65)
            )
            with self.assertRaisesRegex(IncompleteRunError, "method_memory"):
                validate_completed_run(run)

    def test_resume_rejects_mixed_memory_bytes_availability(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                evidence_dir=temporary_directory,
            )[0]
            _write_valid_evidence(run)
            _extend_to_two_sample_memory_timeline(run, [64, None])
            with self.assertRaisesRegex(IncompleteRunError, "mixed memory_bytes availability"):
                validate_completed_run(run)

    def test_latency_and_throughput_are_recomputed_exactly(self):
        mutations = {
            "total": lambda summary: summary["forward_latency"].__setitem__("total_ms", 2.0),
            "mean": lambda summary: summary["forward_latency"].__setitem__("mean_per_sample_ms", 2.0),
            "median": lambda summary: summary["forward_latency"].__setitem__("median_per_sample_ms", 2.0),
            "throughput": lambda summary: summary["throughput"].__setitem__("samples_per_second", 999.0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory,
                )[0]
                _write_valid_evidence(run)
                _mutate_summary(run, mutate)
                with self.assertRaises(IncompleteRunError):
                    validate_completed_run(run)

    def test_retrieval_latency_must_remain_explicitly_unavailable(self):
        mutations = {
            "computed": lambda metric: metric.__setitem__("status", "computed"),
            "empty reason": lambda metric: metric.__setitem__("reason", ""),
            "garbage field": lambda metric: metric.__setitem__("milliseconds", 1.0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory,
                )[0]
                _write_valid_evidence(run)
                _mutate_summary(run, lambda summary: mutate(summary["retrieval_latency"]))
                with self.assertRaisesRegex(IncompleteRunError, "retrieval_latency"):
                    validate_completed_run(run)

    def test_empty_corrupt_and_stale_artifacts_never_skip(self):
        mutations = {
            "empty trace": lambda run: (run.run_dir / "trace.jsonl").write_text("", encoding="utf-8"),
            "corrupt manifest": lambda run: (run.run_dir / "manifest.json").write_text("{", encoding="utf-8"),
            "stale summary": lambda run: (run.run_dir / "summary.json").write_text(
                json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "run_id": "foreign", "num_samples": 1,
                            "stream_fingerprint": "foreign"}), encoding="utf-8"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                run = build_experiment_matrix(
                    datasets=("DomainNet",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
                    evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu",
                )[0]
                _write_valid_evidence(run)
                mutate(run)
                with patch("src.runtime.experiment_matrix.validate_dataset_layout", return_value={"valid": True}):
                    with self.assertRaises(IncompleteRunError):
                        execute_matrix((run,), data_root=temporary_directory, resume=True,
                                       runner=lambda *_args, **_kwargs: self.fail("model launched"))

    def test_stale_config_and_tampered_stream_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "cfg" / "DomainNet" / "Ramen.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("lr: 0.1\n", encoding="utf-8")
            run = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("Ramen",), seeds=(0,),
                evidence_dir=root / "evidence", config_dir=root / "cfg",
            )[-1]
            _write_valid_evidence(run)
            config_path.write_text("lr: 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(IncompleteRunError, "config hash"):
                validate_completed_run(run)

            # Restore the planned config, then corrupt the exported schedule.
            config_path.write_text("lr: 0.1\n", encoding="utf-8")
            stream_path = run.run_dir / "stream.json"
            stream = json.loads(stream_path.read_text(encoding="utf-8"))
            stream["references"] = [[0, 99]]
            stream_path.write_text(json.dumps(stream), encoding="utf-8")
            with self.assertRaisesRegex(IncompleteRunError, "stream.fingerprint"):
                validate_completed_run(run)

    def test_adapted_run_validates_paired_baseline_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            baseline, adapted = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("Ramen",), seeds=(0,),
                evidence_dir=temporary_directory, data_root=temporary_directory, device="cpu",
            )
            baseline.run_dir.mkdir(parents=True)
            (baseline.run_dir / "manifest.json").write_text("{}", encoding="utf-8")
            with patch("src.runtime.experiment_matrix.validate_dataset_layout", return_value={"valid": True}):
                with self.assertRaises(IncompleteRunError):
                    execute_matrix((adapted, baseline), data_root=temporary_directory,
                                   runner=lambda *_args, **_kwargs: self.fail("adapted model launched"))
            self.assertFalse(adapted.run_dir.exists())


if __name__ == "__main__":
    unittest.main()
