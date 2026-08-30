import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.failure_decomposition import (
    FailureDecompositionError,
    OUTCOME_BENEFICIAL,
    OUTCOME_HARMFUL,
    OUTCOME_STABLE_CORRECT,
    OUTCOME_UNRESOLVED,
    compare_trace_outcomes,
    oracle_ladder_decomposition,
    paired_outcome_decomposition,
    scalar_diagnostics_by_outcome,
    validate_paired_rows,
)


class FailureDecompositionTests(unittest.TestCase):
    @staticmethod
    def row(timestep, prediction, target, *, domain=0, sample_idx=None, **extra):
        if sample_idx is None:
            sample_idx = timestep
        return {
            "timestep": timestep,
            "sample_idx": sample_idx,
            "ground_truth_domain": domain,
            "ground_truth_class": target,
            "prediction": prediction,
            "correct": prediction == target,
            **extra,
        }

    def make_four_way_pair(self):
        # stable correct, beneficial, harmful, unresolved
        reference = [
            self.row(0, 0, 0, domain=0),
            self.row(1, 1, 0, domain=0),
            self.row(2, 0, 0, domain=1),
            self.row(3, 1, 0, domain=1),
        ]
        adapted = [
            self.row(0, 0, 0, domain=0, consensus_mean=0.9),
            self.row(1, 0, 0, domain=0, consensus_mean=0.8),
            self.row(2, 1, 0, domain=1, consensus_mean=0.2),
            self.row(3, 1, 0, domain=1, consensus_mean=0.1),
        ]
        return reference, adapted

    def test_exact_four_way_decomposition_and_identity(self):
        reference, adapted = self.make_four_way_pair()
        result = paired_outcome_decomposition(reference, adapted)
        self.assertEqual(4, result["num_samples"])
        self.assertEqual(0.5, result["reference_accuracy"])
        self.assertEqual(0.5, result["adapted_accuracy"])
        self.assertEqual(0.0, result["accuracy_delta"])
        self.assertEqual(1, result["counts"][OUTCOME_STABLE_CORRECT])
        self.assertEqual(1, result["counts"][OUTCOME_BENEFICIAL])
        self.assertEqual(1, result["counts"][OUTCOME_HARMFUL])
        self.assertEqual(1, result["counts"][OUTCOME_UNRESOLVED])
        self.assertEqual(0.5, result["help_rate_given_reference_wrong"])
        self.assertEqual(0.5, result["harm_rate_given_reference_correct"])
        self.assertEqual(0.0, result["identity"]["residual"])

    def test_per_domain_and_temporal_windows_are_paired(self):
        reference, adapted = self.make_four_way_pair()
        result = paired_outcome_decomposition(
            reference, adapted, window_size=2, window_stride=2
        )
        self.assertEqual(2, len(result["per_domain"]))
        self.assertEqual(0.5, result["per_domain"]["0"]["reference_accuracy"])
        self.assertEqual(1.0, result["per_domain"]["0"]["adapted_accuracy"])
        self.assertEqual(1.0, result["per_domain"]["1"]["reference_accuracy"])
        self.assertEqual(0.0, result["per_domain"]["1"]["adapted_accuracy"])
        self.assertEqual(2, len(result["windows"]))
        self.assertEqual(0, result["windows"][0]["start_timestep"])
        self.assertEqual(1, result["windows"][0]["end_timestep"])
        self.assertEqual(2, result["windows"][1]["start_timestep"])
        self.assertEqual(3, result["windows"][1]["end_timestep"])

    def test_pair_validation_fails_on_sample_mismatch(self):
        reference, adapted = self.make_four_way_pair()
        adapted[2] = dict(adapted[2], sample_idx=999)
        with self.assertRaisesRegex(FailureDecompositionError, "paired trace mismatch"):
            validate_paired_rows(reference, adapted)

    def test_pair_validation_fails_on_inconsistent_correct_field(self):
        reference, adapted = self.make_four_way_pair()
        adapted[0] = dict(adapted[0], correct=False)
        with self.assertRaisesRegex(FailureDecompositionError, "inconsistent"):
            paired_outcome_decomposition(reference, adapted)

    def test_pair_validation_fails_on_nonmonotonic_timestep(self):
        reference, adapted = self.make_four_way_pair()
        adapted[3] = dict(adapted[3], timestep=2)
        with self.assertRaisesRegex(FailureDecompositionError, "strictly increasing"):
            paired_outcome_decomposition(reference, adapted)

    def test_scalar_mechanisms_are_stratified_by_exact_outcome(self):
        reference, adapted = self.make_four_way_pair()
        result = scalar_diagnostics_by_outcome(reference, adapted, ["consensus_mean"])
        groups = result["consensus_mean"]["groups"]
        self.assertEqual(0.9, groups[OUTCOME_STABLE_CORRECT]["mean"])
        self.assertEqual(0.8, groups[OUTCOME_BENEFICIAL]["mean"])
        self.assertEqual(0.2, groups[OUTCOME_HARMFUL]["mean"])
        self.assertEqual(0.1, groups[OUTCOME_UNRESOLVED]["mean"])

    def test_scalar_mechanism_rejects_non_numeric_populated_values(self):
        reference, adapted = self.make_four_way_pair()
        adapted[0]["consensus_mean"] = "high"
        with self.assertRaisesRegex(FailureDecompositionError, "finite numeric"):
            scalar_diagnostics_by_outcome(reference, adapted, ["consensus_mean"])

    def test_oracle_ladder_keeps_negative_interactions_visible(self):
        reference, _ = self.make_four_way_pair()
        # Ramen: 2/4 correct. Oracle retrieval: 3/4. Oracle aggregation: 2/4.
        retrieval = [dict(row) for row in reference]
        retrieval[1] = dict(retrieval[1], prediction=0, correct=True)
        aggregation = [dict(row) for row in retrieval]
        aggregation[0] = dict(aggregation[0], prediction=1, correct=False)
        result = oracle_ladder_decomposition(OrderedDict([
            ("Ramen", reference),
            ("OracleRetrieval", retrieval),
            ("OracleAggregation", aggregation),
        ]))
        stages = result["stages"]
        self.assertEqual(0.25, stages[1]["incremental_error_reduction"])
        self.assertFalse(stages[1]["regressed_vs_previous_stage"])
        self.assertEqual(-0.25, stages[2]["incremental_error_reduction"])
        self.assertTrue(stages[2]["regressed_vs_previous_stage"])
        self.assertEqual(0.0, result["total_error_recovery"])

    def test_jsonl_comparison_records_hashes_and_mechanism_groups(self):
        reference, adapted = self.make_four_way_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.jsonl"
            adapted_path = root / "adapted.jsonl"
            for path, rows in ((reference_path, reference), (adapted_path, adapted)):
                with path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
            result = compare_trace_outcomes(
                reference_path,
                adapted_path,
                mechanism_fields=["consensus_mean"],
            )
            self.assertEqual(64, len(result["provenance"]["reference_trace_sha256"]))
            self.assertEqual(64, len(result["provenance"]["adapted_trace_sha256"]))
            self.assertIn("consensus_mean", result["mechanism_by_outcome"])

    def test_window_stride_without_window_size_fails(self):
        reference, adapted = self.make_four_way_pair()
        with self.assertRaisesRegex(FailureDecompositionError, "requires window_size"):
            paired_outcome_decomposition(reference, adapted, window_stride=2)


if __name__ == "__main__":
    unittest.main()
