"""Contract tests for the separate open-set Consensus evidence analyzer."""

import unittest

from src.evaluation.open_set_consensus_analysis import analyse_open_set_completed_runs
from src.runtime.experiment_matrix import build_open_set_evidence_matrix


def _evidence(run, *, fingerprint="paired"):
    summary = {
        "stream_fingerprint": fingerprint,
        "open_set": {
            "status": "computed", "id_accuracy": 0.5, "auroc": 0.7, "fpr95": 0.2,
            "h_score": 0.4, "worst_domain_id_accuracy": 0.3,
        },
        "negative_adaptation_rate": {"status": "computed", "value": 0.25},
        "post_shift_recovery_time": {
            "status": "computed", "shifts": [{"status": "recovered", "recovery_samples": 4}],
        },
        "forward_latency": {
            "status": "computed", "total_ms": 100.0, "mean_per_sample_ms": 2.0,
            "median_per_sample_ms": 1.5,
        },
        "method_memory": {
            "status": "computed", "max_retained_bytes": 1024, "final_retained_bytes": 512,
        },
        "throughput": {"status": "computed", "samples_per_second": 500.0},
    }
    if run.method in {"OracleDropOODRamen", "OracleIDGradientRamen"}:
        summary["oracle_gradient_diagnostics"] = {
            "gradient_direction_corruption_mean": 0.1, "sign_disagreement_mean": 0.2,
        }
    if run.method in {"ConsensusRamen", "OracleConsensusRamen"}:
        summary["consensus_diagnostics"] = {
            "consensus_applied_sample_fraction": 0.9, "mask_rate": 0.6,
        }
    return {
        "summary": summary,
        "manifest": {"args": {"oracle_ood_contexts": run.method.startswith("Oracle")}},
    }


class OpenSetConsensusAnalysisTests(unittest.TestCase):
    def test_reports_noncanonical_partial_evidence_without_legacy_gate(self):
        runs = build_open_set_evidence_matrix(streams=("block",), ood_ratios=(0.5,), seeds=(0,))
        report = analyse_open_set_completed_runs([(run, _evidence(run)) for run in runs])
        self.assertEqual("open_set_consensus_descriptive_v2", report["analysis_contract"])
        self.assertEqual("noncanonical_pilot", report["classification"])
        self.assertEqual("not_applicable", report["consensus_certification"])
        self.assertEqual("complete", report["comparisons"][0]["status"])

    def test_surfaces_validated_stability_and_cost_evidence(self):
        runs = build_open_set_evidence_matrix(streams=("block",), ood_ratios=(0.5,), seeds=(0,))
        report = analyse_open_set_completed_runs([(run, _evidence(run)) for run in runs])
        ramen = report["comparisons"][0]["methods"]["Ramen"]
        self.assertEqual(0.3, ramen["worst_domain_id_accuracy"])
        self.assertEqual({"status": "computed", "rate": 0.25}, ramen["stability"]["negative_adaptation"])
        self.assertEqual("computed", ramen["stability"]["post_shift_recovery"]["status"])
        self.assertEqual(100.0, ramen["cost"]["synchronized_forward_latency"]["total_ms"])
        self.assertEqual(1024, ramen["cost"]["retained_memory"]["max_retained_bytes"])
        self.assertEqual(500.0, ramen["cost"]["throughput"]["samples_per_second"])

    def test_accepts_ratio_zero_explicitly_unavailable_detection_metrics(self):
        runs = build_open_set_evidence_matrix(streams=("block",), ood_ratios=(0.0,), seeds=(0,))
        completed = [(run, _evidence(run)) for run in runs]
        for _, evidence in completed:
            block = evidence["summary"]["open_set"]
            block.update({
                "status": "unavailable", "reason": "OOD metrics require at least one OOD sample",
                "auroc": None, "fpr95": None, "h_score": None,
            })
        metrics = analyse_open_set_completed_runs(completed)["comparisons"][0]["methods"]
        self.assertIsNone(metrics["Ramen"]["auroc"])
        self.assertIsNone(metrics["ConsensusRamen"]["fpr95"])

    def test_reports_paired_consensus_overhead_without_claiming_isolated_time(self):
        runs = build_open_set_evidence_matrix(streams=("block",), ood_ratios=(0.5,), seeds=(0,))
        completed = [(run, _evidence(run)) for run in runs]
        for run, evidence in completed:
            if run.method == "ConsensusRamen":
                evidence["summary"]["forward_latency"]["total_ms"] = 125.0
                evidence["summary"]["throughput"]["samples_per_second"] = 400.0
                evidence["summary"]["method_memory"]["max_retained_bytes"] = 1536
        overhead = analyse_open_set_completed_runs(completed)["comparisons"][0]["consensus_vs_ramen_cost_overhead"]["ConsensusRamen"]
        self.assertEqual("paired_total_path_proxy", overhead["status"])
        self.assertEqual(25.0, overhead["forward_latency_total_ms_difference"])
        self.assertEqual(1.25, overhead["forward_latency_total_ms_ratio"])
        self.assertEqual(512.0, overhead["max_retained_memory_bytes_difference"])
        self.assertIn("not isolated", overhead["definition"])

    def test_rejects_missing_relevant_cost_summary(self):
        runs = build_open_set_evidence_matrix(streams=("block",), ood_ratios=(0.5,), seeds=(0,))
        completed = [(run, _evidence(run)) for run in runs]
        completed[0][1]["summary"].pop("method_memory")
        with self.assertRaisesRegex(ValueError, "summary method_memory missing"):
            analyse_open_set_completed_runs(completed)

    def test_rejects_evaluator_context_on_consensus(self):
        runs = build_open_set_evidence_matrix(streams=("block",), ood_ratios=(0.5,), seeds=(0,))
        completed = [(run, _evidence(run)) for run in runs]
        for run, evidence in completed:
            if run.method == "ConsensusRamen":
                evidence["manifest"]["args"]["oracle_ood_contexts"] = True
        with self.assertRaisesRegex(ValueError, "non-oracle received evaluator OOD context"):
            analyse_open_set_completed_runs(completed)

    def test_rejects_mismatched_paired_stream_identity(self):
        runs = build_open_set_evidence_matrix(streams=("block",), ood_ratios=(0.5,), seeds=(0,))
        completed = [(run, _evidence(run, fingerprint="other" if run.method == "Ramen" else "paired")) for run in runs]
        with self.assertRaisesRegex(ValueError, "different stream fingerprints"):
            analyse_open_set_completed_runs(completed)
