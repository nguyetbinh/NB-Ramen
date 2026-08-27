import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.causal_completion_analysis import (
    CausalCompletionThresholds,
    MatrixEvidenceError,
    analyse_completed_runs,
    build_causal_matrix,
    load_completed_runs,
    main,
)
from src.runtime.experiment_matrix import IncompleteRunError


THRESHOLDS = CausalCompletionThresholds.from_mapping({
    "minimum_fixed_seeds": 3,
    "minimum_non_iid_streams": 2,
    "minimum_mean_micro_gain_for_go": 0.01,
    "minimum_mean_micro_gain_for_weak_go": 0.0,
    "minimum_mean_negative_adaptation_reduction_for_weak_go": 0.01,
    "minimum_mean_recovery_reduction_for_weak_go": 1.0,
    "minimum_mean_worst_domain_gain_for_weak_go": 0.005,
    "maximum_micro_std": 0.02,
    "require_full_cifar100c": True,
    "require_natural_domain_dataset": True,
    "allow_weak_go": True,
})


class _Run:
    def __init__(self, dataset, stream, seed, batch_size, method, *, full=True, config=None):
        self.dataset, self.stream_mode, self.seed = dataset, stream, seed
        self.batch_size, self.method = batch_size, method
        self.run_id = f"{dataset}-{stream}-{seed}-b{batch_size}-{method}"
        self.model, self.device, self.max_eval_samples = "clip", "cpu", None if full else 64
        self.stream_block_size, self.metric_window_size, self.metric_window_stride = 64, 50, 50
        self.artifact_provenance, self.data_root = "exact", Path("/data")
        self.config_data = config if config is not None else {"topk": 5, "method": method}


def _evidence(micro, *, macro=None, worst=None, negative=0.1, recovery=4, memory=100, latency=100):
    return {"summary": {
        "micro_accuracy": micro,
        "macro_domain_accuracy": micro if macro is None else macro,
        "worst_domain_accuracy": micro if worst is None else worst,
        "negative_adaptation_rate": {"status": "computed", "value": negative},
        "post_shift_recovery_time": {"status": "computed", "shifts": [{"status": "recovered", "recovery_samples": recovery}]},
        "method_memory": {"status": "computed", "max_retained_bytes": memory},
        "forward_latency": {"status": "computed", "total_ms": latency},
        "retrieval_latency": {"status": "computed", "total_ms": latency / 2},
        "peak_device_memory_bytes": memory,
    }}


def _completed(*, gain=0.015, full=True, include_natural=True):
    completed = []
    datasets = ("CIFAR100C", "DomainNet") if include_natural else ("CIFAR100C",)
    for dataset in datasets:
        for stream in ("block", "gradual"):
            for seed in (0, 1, 2):
                for batch in (1, 100):
                    legacy = _Run(dataset, stream, seed, batch, "Ramen", full=full)
                    atomic = _Run(dataset, stream, seed, batch, "StructuredAtomicRamen", full=full)
                    causal = _Run(dataset, stream, seed, batch, "CausalRamen", full=full)
                    completed.extend(((legacy, _evidence(.49 if batch == 1 else .48)), (atomic, _evidence(.50)), (causal, _evidence(.50 + gain, macro=.52 + gain, worst=.48 + gain, negative=.05, recovery=3))))
    return completed


class CausalCompletionAnalysisTests(unittest.TestCase):
    def test_go_reports_paired_metrics_and_descriptive_batch_sizes(self):
        report = analyse_completed_runs(_completed(), THRESHOLDS)
        self.assertEqual("GO", report["decision"]["status"])
        self.assertEqual([1, 100], [item["batch_size"] for item in report["batch_size_effects"]])
        self.assertEqual([1, 100], [item["batch_size"] for item in report["legacy_vs_causal_batch_size_effects"]])
        self.assertEqual("computed", report["legacy_vs_causal_b1_diagnostic"]["status"])
        b1_micro = report["legacy_vs_causal_b1_diagnostic"]["deltas_causal_minus_legacy"]["micro_accuracy"]
        self.assertAlmostEqual(.025, b1_micro["mean"])
        first = report["comparisons"][0]["deltas_causal_minus_atomic"]
        self.assertAlmostEqual(.015, first["micro_accuracy"])
        self.assertAlmostEqual(.035, first["macro_domain_accuracy"])
        self.assertAlmostEqual(-.005, first["worst_domain_accuracy"])
        self.assertAlmostEqual(-.05, first["negative_adaptation_rate"])
        self.assertAlmostEqual(-1, first["recovery_samples"])
        self.assertEqual("not_tested_as_a_hard_criterion", report["monotonic_batch_size_claim"]["status"])

    def test_missing_paired_cell_is_insufficient_and_fails_closed(self):
        completed = _completed()
        expected = [run for run, _ in completed]
        completed = [(run, evidence) for run, evidence in completed if not (run.dataset == "DomainNet" and run.seed == 1 and run.method == "CausalRamen")]
        report = analyse_completed_runs(completed, THRESHOLDS, expected_runs=expected)
        self.assertEqual("INSUFFICIENT", report["decision"]["status"])
        self.assertTrue(report["missing_cells"])

    def test_missing_legacy_cell_is_insufficient_and_fails_closed(self):
        completed = _completed()
        expected = [run for run, _ in completed]
        completed = [(run, evidence) for run, evidence in completed if not (run.dataset == "DomainNet" and run.seed == 1 and run.method == "Ramen")]
        report = analyse_completed_runs(completed, THRESHOLDS, expected_runs=expected)
        self.assertEqual("INSUFFICIENT", report["decision"]["status"])
        self.assertIn("Ramen", report["missing_cells"][0]["methods"])

    def test_nonidentity_config_difference_is_invalid(self):
        completed = _completed()
        for run, _ in completed:
            if run.method == "CausalRamen":
                run.config_data = {"topk": 7, "method": run.method}
                break
        report = analyse_completed_runs(completed, THRESHOLDS)
        self.assertEqual("INVALID", report["decision"]["status"])
        self.assertIn("config_data_except_method_identity", report["config_equivalence_failures"][0]["mismatches"])

    def test_weak_go_requires_stability_and_secondary_improvement(self):
        report = analyse_completed_runs(_completed(gain=.005), THRESHOLDS)
        self.assertEqual("WEAK_GO", report["decision"]["status"])
        self.assertTrue(report["decision"]["weak_go_improvements"]["negative_adaptation_reduction"]["passed"])

    def test_stable_nonnegative_micro_without_secondary_improvement_is_no_go(self):
        completed = _completed(gain=.005)
        for run, evidence in completed:
            if run.method == "CausalRamen":
                evidence["summary"].update({
                    "worst_domain_accuracy": .50,
                    "negative_adaptation_rate": {"status": "computed", "value": .10},
                    "post_shift_recovery_time": {"status": "computed", "shifts": [{"status": "recovered", "recovery_samples": 4}]},
                })
        report = analyse_completed_runs(completed, THRESHOLDS)
        self.assertEqual("NO_GO", report["decision"]["status"])
        self.assertFalse(any(item["passed"] for item in report["decision"]["weak_go_improvements"].values()))

    def test_go_requires_micro_stability(self):
        completed = _completed()
        index = 0
        for run, evidence in completed:
            if run.method == "CausalRamen":
                evidence["summary"]["micro_accuracy"] = .60 if index % 2 == 0 else .43
                index += 1
        report = analyse_completed_runs(completed, THRESHOLDS)
        self.assertGreaterEqual(report["decision"]["mean_micro_gain"]["mean"], .01)
        self.assertFalse(report["decision"]["stability_passed"])
        self.assertEqual("NO_GO", report["decision"]["status"])

    def test_pilot_requires_full_cifar_and_natural_domain(self):
        report = analyse_completed_runs(_completed(full=False, include_natural=False), THRESHOLDS)
        self.assertEqual("PILOT", report["decision"]["status"])
        self.assertFalse(report["decision"]["requirements"]["full_cifar100c"]["passed"])
        self.assertFalse(report["decision"]["requirements"]["natural_domain_dataset"]["passed"])

    def test_no_go_for_complete_negative_gain(self):
        report = analyse_completed_runs(_completed(gain=-.02), THRESHOLDS)
        self.assertEqual("NO_GO", report["decision"]["status"])

    def test_batch_matrix_reuses_runtime_batch_size_support(self):
        observed = []

        def planner(*, batch_size, **kwargs):
            observed.append(batch_size)
            self.assertEqual(("Ramen", "StructuredAtomicRamen", "CausalRamen"), kwargs["methods"])
            return [_Run("CIFAR100C", "block", 0, batch_size, method) for method in kwargs["methods"]]

        with patch("src.evaluation.causal_completion_analysis.build_experiment_matrix", side_effect=planner):
            runs = build_causal_matrix(batch_sizes=(1, 100), datasets=("CIFAR100C",), streams=("block",), seeds=(0,))
        self.assertEqual([1, 100], observed)
        self.assertEqual(6, len(runs))

    def test_cli_exit_statuses_are_differentiated_and_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            thresholds = Path(temporary) / "thresholds.json"
            thresholds.write_text(json.dumps(THRESHOLDS.__dict__), encoding="utf-8")
            argv = ["--thresholds", str(thresholds), "--evidence-dir", temporary, "--dataset", "CIFAR100C", "--stream", "block", "--seed", "0", "--batch-size", "1"]
            for status, expected_exit in (("GO", 0), ("WEAK_GO", 0), ("NO_GO", 1), ("PILOT", 1), ("INSUFFICIENT", 1), ("INVALID", 2)):
                with self.subTest(status=status), \
                     patch("src.evaluation.causal_completion_analysis.build_causal_matrix", return_value=[]), \
                     patch("src.evaluation.causal_completion_analysis.load_completed_runs", return_value=[]), \
                     patch("src.evaluation.causal_completion_analysis.analyse_completed_runs", return_value={"decision": {"status": status}}), \
                     patch("builtins.print") as printed:
                    result = main(argv)
                self.assertEqual(expected_exit, result)
                self.assertTrue(json.loads(printed.call_args.args[0]))
            with patch("builtins.print") as printed:
                result = main(["--thresholds", str(thresholds), "--evidence-dir", temporary, "--dataset", "CIFAR100C", "--stream", "block", "--seed", "0", "--batch-size", "0"])
            self.assertEqual(2, result)
            self.assertEqual("INVALID", json.loads(printed.call_args.args[0])["decision"]["status"])

    def test_cli_missing_artifact_is_insufficient_not_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            thresholds = Path(temporary) / "thresholds.json"
            thresholds.write_text(json.dumps(THRESHOLDS.__dict__), encoding="utf-8")
            with patch("src.evaluation.causal_completion_analysis.build_causal_matrix", side_effect=IncompleteRunError("missing evidence directory: /nope")), \
                 patch("builtins.print") as printed:
                result = main(["--thresholds", str(thresholds), "--evidence-dir", temporary, "--dataset", "CIFAR100C", "--stream", "block", "--seed", "0", "--batch-size", "1"])
            self.assertEqual(1, result)
            self.assertEqual("INSUFFICIENT", json.loads(printed.call_args.args[0])["decision"]["status"])

    def test_loader_validates_all_runs_and_missing_only_is_insufficient(self):
        runs = [
            _Run("CIFAR100C", "block", 0, 1, "Ramen"),
            _Run("CIFAR100C", "block", 0, 1, "CausalRamen"),
        ]
        with patch("src.evaluation.causal_completion_analysis.validate_completed_run", side_effect=[
            IncompleteRunError("missing evidence directory: /first"),
            IncompleteRunError("missing trace: /second/trace.jsonl"),
        ]) as validator:
            with self.assertRaises(MatrixEvidenceError) as raised:
                load_completed_runs(runs)
        self.assertEqual(2, validator.call_count)
        self.assertEqual("INSUFFICIENT", raised.exception.status)
        self.assertTrue(all(item["classification"] == "missing" for item in raised.exception.failures))

    def test_cli_earlier_missing_does_not_hide_later_malformed_evidence(self):
        runs = [
            _Run("CIFAR100C", "block", 0, 1, "Ramen"),
            _Run("CIFAR100C", "block", 0, 1, "CausalRamen"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            thresholds = Path(temporary) / "thresholds.json"
            thresholds.write_text(json.dumps(THRESHOLDS.__dict__), encoding="utf-8")
            with patch("src.evaluation.causal_completion_analysis.build_causal_matrix", return_value=runs), \
                 patch("src.evaluation.causal_completion_analysis.validate_completed_run", side_effect=[
                     IncompleteRunError("missing evidence directory: /first"),
                     IncompleteRunError("summary.micro_accuracy disagrees with trace: later-run"),
                 ]) as validator, patch("builtins.print") as printed:
                result = main(["--thresholds", str(thresholds), "--evidence-dir", temporary, "--dataset", "CIFAR100C", "--stream", "block", "--seed", "0", "--batch-size", "1"])
        payload = json.loads(printed.call_args.args[0])
        self.assertEqual(2, validator.call_count)
        self.assertEqual(2, result)
        self.assertEqual("INVALID", payload["decision"]["status"])
        self.assertEqual(["missing", "invalid"], [item["classification"] for item in payload["validation_failures"]])


if __name__ == "__main__":
    unittest.main()
