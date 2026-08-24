import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from src.evaluation.experiment_analysis import (
    AnalysisThresholds,
    analyse_completed_runs,
    load_completed_runs,
    main,
)
from src.runtime.experiment_matrix import IncompleteRunError, build_experiment_matrix


THRESHOLDS = AnalysisThresholds.from_mapping({
    "minimum_repeats": 2,
    "max_accuracy_std": 0.1,
    "structured_degradation_min": 0.1,
    "oracle_recovery_min": 0.1,
    "router_closure_min": 0.4,
    "natural_domain_gain_min": 0.05,
    "max_memory_ratio": 1.2,
    "max_forward_latency_ratio": 1.2,
    "min_routing_accuracy_association": 0.1,
    "max_class_context_nmi": 0.8,
})


class _Run:
    def __init__(self, dataset, stream, seed, method):
        self.dataset, self.stream_mode, self.seed, self.method = dataset, stream, seed, method
        self.run_id = f"{dataset}-{stream}-{seed}-{method}"
        self.run_dir = Path("/strictly-validated-evidence")


def _evidence(accuracy, nmi, memory=100, latency=100):
    return {"summary": {
        "micro_accuracy": accuracy,
        "worst_domain_accuracy": accuracy,
        "post_shift_recovery_time": {"status": "computed", "shifts": [{"status": "recovered", "recovery_samples": 3}]},
        "negative_adaptation_rate": {"status": "computed", "value": 0.0},
        "forward_latency": {"status": "computed", "total_ms": latency},
        "throughput": {"status": "computed", "samples_per_second": 10.0},
        "method_memory": {"status": "computed", "max_retained_bytes": memory},
        "peak_device_memory_bytes": memory,
        "device_memory": {"kind": "exact_cuda_allocator_peak", "bytes": memory},
        "routing_diagnostics": {"normalized_mutual_information": nmi},
    }}


class ExperimentAnalysisTests(unittest.TestCase):
    def _completed(self, *, latent_memory=105, latent_latency=105, streams=("iid_mixed", "block")):
        completed = []
        for stream in streams:
            for seed in (0, 1):
                values = {
                    "NoAdapt": (0.50, None),
                    "Ramen": (0.80 if stream == "iid_mixed" else 0.60, None),
                    "OracleLatentRamen": (0.90 if stream == "iid_mixed" else 0.80, 1.0),
                    "LatentRamen": ((0.82 if seed == 0 else 0.84) if stream == "iid_mixed" else (0.70 if seed == 0 else 0.74), 0.2 if seed == 0 else 0.9),
                }
                for method, (accuracy, nmi) in values.items():
                    memory = latent_memory if method == "LatentRamen" else 100
                    latency = latent_latency if method == "LatentRamen" else 100
                    completed.append((_Run("DomainNet", stream, seed, method), _evidence(accuracy, nmi, memory, latency)))
        return completed

    @patch("src.evaluation.experiment_analysis._class_recovery", return_value={"status": "computed", "nmi_inferred_context_vs_ground_truth_class": 0.2})
    def test_aggregation_and_go_gate(self, _class_recovery):
        report = analyse_completed_runs(self._completed(), THRESHOLDS)
        self.assertEqual("go", report["go_no_go"]["status"])
        ramen_block = next(item for item in report["aggregates"] if item["method"] == "Ramen" and item["stream_mode"] == "block")
        self.assertEqual(2, ramen_block["metrics"]["accuracy"]["count"])
        self.assertAlmostEqual(0.6, ramen_block["metrics"]["accuracy"]["mean"])
        self.assertEqual("unavailable", report["comparisons"][0]["retrieval_latency"]["status"])

    @patch("src.evaluation.experiment_analysis._class_recovery", return_value={"status": "computed", "nmi_inferred_context_vs_ground_truth_class": 0.2})
    def test_no_go_when_equal_compute_fails(self, _class_recovery):
        report = analyse_completed_runs(self._completed(latent_memory=300), THRESHOLDS)
        self.assertEqual("no_go", report["go_no_go"]["status"])
        self.assertFalse(report["go_no_go"]["criteria"]["memory_ratio"]["passed"])

    @patch("src.evaluation.experiment_analysis._class_recovery", return_value={"status": "computed", "nmi_inferred_context_vs_ground_truth_class": 0.2})
    def test_insufficient_when_routing_association_is_not_identifiable(self, _class_recovery):
        completed = self._completed()
        for run, evidence in completed:
            if run.method == "LatentRamen":
                evidence["summary"]["routing_diagnostics"]["normalized_mutual_information"] = None
        report = analyse_completed_runs(completed, THRESHOLDS)
        self.assertEqual("insufficient_evidence", report["go_no_go"]["status"])
        self.assertIn("routing_accuracy_association", report["go_no_go"]["insufficient"])

    @patch("src.evaluation.experiment_analysis._class_recovery", return_value={"status": "computed", "nmi_inferred_context_vs_ground_truth_class": 0.2})
    def test_one_missing_seed_fails_each_evidence_family_closed(self, _class_recovery):
        def latent_block_seed_zero(completed):
            return next(evidence for run, evidence in completed if run.method == "LatentRamen" and run.stream_mode == "block" and run.seed == 0)

        cases = (
            ("memory_ratio", lambda completed: latent_block_seed_zero(completed)["summary"].update({
                "peak_device_memory_bytes": None,
                "device_memory": {"kind": "exact_cuda_allocator_peak", "bytes": None},
            })),
            ("forward_latency_ratio", lambda completed: latent_block_seed_zero(completed)["summary"].update({
                "forward_latency": {"status": "unavailable"},
            })),
            ("routing_accuracy_association", lambda completed: latent_block_seed_zero(completed)["summary"]["routing_diagnostics"].update({
                "normalized_mutual_information": None,
            })),
            ("paired_comparison_completeness", lambda completed: completed.__setitem__(
                slice(None), [(run, evidence) for run, evidence in completed if not (
                    run.method == "OracleLatentRamen" and run.stream_mode == "block" and run.seed == 0
                )],
            )),
        )
        for criterion, mutate in cases:
            with self.subTest(criterion=criterion):
                completed = self._completed()
                mutate(completed)
                report = analyse_completed_runs(completed, THRESHOLDS)
                self.assertEqual("insufficient_evidence", report["go_no_go"]["status"])
                self.assertIn(criterion, report["go_no_go"]["insufficient"])
                if criterion == "paired_comparison_completeness":
                    self.assertIn("oracle_recovery", report["go_no_go"]["insufficient"])
                    self.assertIn("router_closure", report["go_no_go"]["insufficient"])

    @patch("src.evaluation.experiment_analysis._class_recovery", return_value={"status": "computed", "nmi_inferred_context_vs_ground_truth_class": 0.2})
    def test_iid_only_gains_do_not_satisfy_structured_gates(self, _class_recovery):
        report = analyse_completed_runs(self._completed(streams=("iid_mixed",)), THRESHOLDS)
        self.assertEqual("insufficient_evidence", report["go_no_go"]["status"])
        for name in ("structured_degradation", "oracle_recovery", "router_closure", "natural_domain_gain"):
            self.assertEqual("insufficient_evidence", report["go_no_go"]["criteria"][name]["status"])

    @patch("src.evaluation.experiment_analysis._class_recovery", return_value={"status": "computed", "nmi_inferred_context_vs_ground_truth_class": 0.2})
    def test_one_strong_structured_degradation_is_sufficient(self, _class_recovery):
        completed = self._completed()
        # Add a second non-IID condition with no Ramen degradation. The block
        # condition remains a strong 0.20 degradation and must control the gate.
        for run, evidence in self._completed(streams=("gradual",)):
            if run.method == "Ramen":
                evidence["summary"]["micro_accuracy"] = 0.80
                evidence["summary"]["worst_domain_accuracy"] = 0.80
            completed.append((run, evidence))
        report = analyse_completed_runs(completed, THRESHOLDS)
        criterion = report["go_no_go"]["criteria"]["structured_degradation"]
        self.assertAlmostEqual(0.2, criterion["value"])
        self.assertTrue(criterion["passed"])

    def test_tampered_evidence_is_rejected_by_strict_loader(self):
        run = _Run("DomainNet", "block", 0, "Ramen")
        with patch("src.evaluation.experiment_analysis.validate_completed_run", side_effect=IncompleteRunError("summary disagrees with trace")):
            with self.assertRaisesRegex(IncompleteRunError, "disagrees"):
                load_completed_runs([run])

    def test_trace_replacement_after_read_is_revalidated(self):
        from src.evaluation import experiment_analysis

        with tempfile.TemporaryDirectory() as temporary:
            run = _Run("DomainNet", "block", 0, "LatentRamen")
            run.run_dir = Path(temporary)
            trace = run.run_dir / "trace.jsonl"
            trace.write_text('{"ground_truth_class": 0, "inferred_context": 0}\n', encoding="utf-8")
            original = trace.read_bytes()

            def replace_during_read(_run):
                rows = [{"ground_truth_class": 0, "inferred_context": 0}]
                trace.write_text('{"ground_truth_class": 1, "inferred_context": 1}\n', encoding="utf-8")
                return rows

            def validator(_run):
                if trace.read_bytes() != original:
                    raise IncompleteRunError("trace replacement detected")
                return {}

            with patch.object(experiment_analysis, "_read_trace", side_effect=replace_during_read), \
                    patch.object(experiment_analysis, "validate_completed_run", side_effect=validator):
                with self.assertRaisesRegex(IncompleteRunError, "replacement"):
                    experiment_analysis._class_recovery(run)

    def test_cli_reconstructs_nondefault_run_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            threshold_path = Path(temporary) / "thresholds.json"
            threshold_path.write_text(__import__("json").dumps(THRESHOLDS.__dict__), encoding="utf-8")
            expected = build_experiment_matrix(
                datasets=["DomainNet"], streams=["block"], seeds=[7],
                methods=("NoAdapt", "Ramen", "OracleLatentRamen", "LatentRamen"),
                evidence_dir=temporary, device="mps", max_eval_samples=11,
                artifact_provenance="exact", data_root="/analysis/data", stream_block_size=13,
            )
            captured = []

            def completed(runs):
                captured.extend(runs)
                return []

            with patch("src.evaluation.experiment_analysis.load_completed_runs", side_effect=completed), \
                    patch("src.evaluation.experiment_analysis.analyse_completed_runs", return_value={"go_no_go": {"status": "go"}}), \
                    patch("builtins.print"):
                result = main([
                    "--thresholds", str(threshold_path), "--evidence-dir", temporary,
                    "--dataset", "DomainNet", "--stream", "block", "--seed", "7",
                    "--device", "mps", "--max-eval-samples", "11",
                    "--stream-block-size", "13",
                    "--artifact-provenance", "exact",
                    "--data-root", "/analysis/data",
                ])
            self.assertEqual(0, result)
            self.assertEqual([run.run_id for run in expected], [run.run_id for run in captured])


if __name__ == "__main__":
    unittest.main()
