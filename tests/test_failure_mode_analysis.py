import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.failure_mode_analysis import (
    PREREGISTERED_COUNTERFACTUAL_THRESHOLDS,
    analyze_failure_modes,
    atomic_causal_pairing,
    consensus_ramen_decision,
    counterfactual_recovery_analysis,
    entropy_admission_analysis,
    gradient_conflict_association,
    gradient_direction_corruption,
    open_set_gradient_analysis,
    oracle_gap_analysis,
    paired_outcome_decomposition,
)
from src.evaluation.evidence import FAILURE_ANALYSIS_REQUIRED_FIELDS
from src.evaluation.failure_analysis_artifacts import ReplaySidecarWriter, sha256_file
from src.streams import stream_fingerprint
import torch


def row(index, correct):
    return {"timestep": index, "sample_idx": index, "ground_truth_domain": "d",
            "ground_truth_class": index, "correct": correct}


class FailureModeAnalysisTests(unittest.TestCase):

    def _verified_run(self, root, name, correctness, *, schedule="causal", future=0, with_sidecar=True,
                      reset_state_verified=True, corrupt_failure_trace=False, device="cpu",
                      model_digest="a" * 64, dataset_digest="b" * 64,
                      evaluator_overrides=None, reference_trace=None, manifest_mutator=None):
        run = root / name
        run.mkdir()
        args = {"device": device, "dataset": "CIFAR100C", "model": "clip_vitbase16", "seed": 0,
                "stream_seed": 0, "batch_size": 1, "max_eval_samples": 2,
                "stream_mode": "block", "stream_block_size": 2, "reference_trace": reference_trace,
                "tta_algo": "NoAdapt" if not with_sidecar else "CausalRamen",
                "failure_analysis_profile": "off" if not with_sidecar else "replay_v1"}
        if evaluator_overrides:
            args.update(evaluator_overrides)
        manifest = {"schema_version": 1, "run_id": name, "device": device, "args": args,
                    "config": {"max_capacity": 4, "capacity_scope": "per_class", "topk": 2,
                               "beta": 5.0, "include_current": True, "optimizer": "signsgd",
                               "lr": 0.01, "method": args["tta_algo"]},
                    "artifacts": {"status": "verified",
                                  "model": {"status": "verified", "model": "clip_vitbase16", "actual_sha256": model_digest},
                                  "dataset": {"status": "verified", "dataset": "cifar100c", "root_digest": dataset_digest}},
                    "git": {"source": {"fingerprint": "d" * 64}}}
        if manifest_mutator:
            manifest_mutator(manifest)
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        payload = {"metadata": {"mode": "fixture"}, "references": [[0, 10], [0, 11]]}
        payload["fingerprint"] = stream_fingerprint(payload)
        (run / "stream.json").write_text(json.dumps(payload), encoding="utf-8")
        traces = []
        for timestep, correct in enumerate(correctness):
            trace = {"schema_version": 2, "run_id": name, "timestep": timestep, "sample_idx": 10 + timestep,
                     "ground_truth_domain": 0, "ground_truth_class": timestep,
                     "prediction": timestep if correct else 9, "correct": correct}
            if with_sidecar:
                failure = {field: 0 for field in FAILURE_ANALYSIS_REQUIRED_FIELDS}
                failure.update({
                    "query_item_id": 100 + timestep, "producer_query_timestep": timestep,
                    "evaluator_sample_identity": {"sample_idx": 10 + timestep, "ground_truth_domain": 0},
                    "batch_position": timestep,
                    # These are the canonical evaluator diagnostics mirrored by
                    # the replay sidecar below.  Keep the class-balanced shape
                    # (including its padded invalid entry) intact.
                    "support_item_ids": [[100 + timestep, -1]],
                    "support_predicted_classes": [[timestep, -1]],
                    "support_distances": [[0.0, 0.0]],
                    "support_entropies": [[0.1, 0.0]],
                    "support_weights": [[1.0, 0.0]],
                    "support_recencies": [[0, 0]],
                    "support_valid_mask": [[True, False]], "support_count": 1,
                    "active_support_classes": [timestep], "schedule": schedule,
                    "future_support_count": future, "future_support_weight_fraction": float(future),
                    "conflict_metric": "fraction_low_consensus_coordinates_v1", "conflict": .2 + .1 * timestep,
                })
                if corrupt_failure_trace:
                    failure = {"legal_candidates": ["method-must-not-overwrite"]}
                trace["failure_analysis"] = failure
            traces.append(trace)
        (run / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces), encoding="utf-8")
        if not with_sidecar:
            return run
        writer = ReplaySidecarWriter(run / "failure-analysis", run_id=name,
            manifest_sha256=sha256_file(run / "manifest.json"), stream_fingerprint=payload["fingerprint"],
            source_fingerprint="d" * 64, config={"counterfactual_thresholds": [0.5, 0.75, 1.0]},
            max_samples=2, max_bytes=100000)
        for timestep in range(2):
            writer.write(items=[{"item_id": 100 + timestep, "segment_index": timestep,
                                 "predicted_class": timestep, "entropy": .1, "admitted": True,
                                 "ground_truth_class": timestep,
                                 "feature": torch.tensor([1.0])}], query={
                "item_id": 100 + timestep, "segment_index": timestep,
                "producer_query_timestep": timestep,
                "evaluator_sample_identity": {"sample_idx": 10 + timestep, "ground_truth_domain": 0},
                "legal_candidates": [{"item_id": 100 + timestep}],
                # Matches the class-balanced Causal/Structured payload:
                # [active class][rank], with padded invalid entries.
                "retrieved_support_ids": [[100 + timestep, -1]], "retrieved_weights": [[1.0, 0.0]],
                "retrieved_distances": [[0.0, 0.0]],
                "support_valid_mask": [[True, False]], "support_predicted_classes": [[timestep, -1]],
                "schedule": schedule, "conflict_metric": "fraction_low_consensus_coordinates_v1",
                "conflict": .2 + .1 * timestep, "batch_position": timestep,
                "future_support_count": future, "future_support_weight_fraction": float(future),
                "reset_state_verified": reset_state_verified, "ground_truth_class": timestep,
                "counterfactuals": [{"threshold": .5, "prediction": timestep},
                                    {"threshold": .75, "prediction": 9}, {"threshold": 1.0, "prediction": timestep}]})
        writer.close()
        return run

    @staticmethod
    def _mutate_sidecar_query(run, mutate):
        path = run / "failure-analysis" / "queries.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        metadata_path = run / "failure-analysis" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["files"]["queries.jsonl"] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    def test_exact_paired_decomposition_and_identity(self):
        base = [row(0, True), row(1, False), row(2, True), row(3, False)]
        adapted = [row(0, True), row(1, True), row(2, False), row(3, False)]
        report = paired_outcome_decomposition(base, adapted)
        self.assertEqual("computed", report["status"])
        self.assertEqual({"safe": 1, "beneficial": 1, "harmful": 1, "unresolved": 1}, report["counts"])
        self.assertEqual(0.0, report["accuracy_delta"])
        self.assertEqual(report["accuracy_delta"], report["identity_h_minus_a"])
        with self.assertRaises(ValueError):
            paired_outcome_decomposition(base, list(reversed(adapted)))
        with self.assertRaises(ValueError):
            paired_outcome_decomposition(base[:-1], adapted)
        self.assertEqual("insufficient", paired_outcome_decomposition([], [])["status"])

    def test_oracle_entropy_open_set_and_nonfinite_rejection(self):
        query = {"legal_candidates": [{"item_id": "a", "correct_pseudolabel": True}, {"item_id": "b", "correct_pseudolabel": False}],
                 "retrieved_supports": [{"item_id": "b", "correct_pseudolabel": False}]}
        oracle = oracle_gap_analysis([query])
        self.assertEqual("computed", oracle["status"])
        self.assertEqual(1.0, oracle["retrieval_gap"])
        self.assertEqual("unavailable", oracle_gap_analysis([{}])["status"])
        entropy = entropy_admission_analysis([
            {"entropy": .2, "correct_pseudolabel": True, "admitted": True, "downstream_weight": 2.0},
            {"entropy": .8, "correct_pseudolabel": False, "admitted": False, "downstream_weight": 1.0},
        ])
        self.assertEqual("computed", entropy["status"])
        self.assertEqual("computed", entropy["groups"]["low_correct"]["storage_rate"]["status"])
        self.assertEqual({"gdc": 1.0, "sdr": .5}, gradient_direction_corruption([1, 1], [1, -1]))
        self.assertEqual("computed", open_set_gradient_analysis([
            {"all_gradient": [1, 1], "id_gradient": [1, -1]}
        ])["status"])
        with self.assertRaises(ValueError):
            entropy_admission_analysis([{"entropy": float("nan"), "correct_pseudolabel": True}])

    def test_counterfactual_and_composed_states(self):
        rows = [
            {"base_correct": True, "adapted_correct": False, "reset_state_verified": True,
             "counterfactual_predictions": {.5: True, .75: False, 1.0: False}},
            {"base_correct": True, "adapted_correct": True, "reset_state_verified": True,
             "counterfactual_predictions": {.5: True, .75: False, 1.0: True}},
            {"base_correct": False, "adapted_correct": False, "reset_state_verified": True,
             "counterfactual_predictions": {.5: True, .75: False, 1.0: False}},
            {"base_correct": False, "adapted_correct": True, "reset_state_verified": True,
             "counterfactual_predictions": {.5: True, .75: True, 1.0: True}},
        ]
        report = counterfactual_recovery_analysis(rows)
        recovered = report["variants"]["0.50"]
        self.assertEqual("computed", recovered["status"])
        self.assertEqual(1, recovered["harmful_event_count"])
        self.assertEqual(1, recovered["harmful_recovery_count"])
        self.assertEqual(1.0, recovered["harmful_recovery_rate"])
        self.assertEqual(0, recovered["new_harm_count"])
        self.assertEqual(1.0, recovered["counterfactual_accuracy"])
        self.assertEqual(.5, recovered["accuracy_delta_vs_adapted"])
        harmed = report["variants"]["0.75"]
        self.assertEqual(1, harmed["new_harm_count"])
        self.assertEqual(.5, harmed["new_harm_rate"])
        self.assertEqual("0.50", report["best_preregistered_upper_bound"]["threshold"])
        with self.assertRaises(ValueError):
            counterfactual_recovery_analysis(rows, thresholds=(.5,))
        no_harmful = counterfactual_recovery_analysis([
            {"base_correct": True, "adapted_correct": True, "reset_state_verified": True,
             "counterfactual_predictions": {.5: True, .75: True, 1.0: True}}
        ])["variants"]["0.50"]
        self.assertEqual(0, no_harmful["harmful_event_count"])
        self.assertEqual("insufficient", no_harmful["harmful_recovery_status"])
        self.assertIsNone(no_harmful["harmful_recovery_rate"])
        complete = analyze_failure_modes([row(0, True)], [row(0, True)])
        self.assertEqual("unavailable", complete["oracle_gaps"]["status"])
        self.assertEqual("unavailable", complete["open_set"]["status"])

    def test_cli_identity_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, adapted = root / "base.json", root / "adapted.json"
            base.write_text(json.dumps({"identity": {"stream": "a"}, "rows": [row(0, True)]}))
            adapted.write_text(json.dumps({"identity": {"stream": "b"}, "rows": [row(0, True)]}))
            result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis", "--base", str(base), "--adapted", str(adapted)], capture_output=True, text=True)
            self.assertEqual(2, result.returncode)
            self.assertEqual("invalid", json.loads(result.stdout)["status"])

    def test_verified_run_directory_cli_computes_f1_f4_and_f5(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            adapted = self._verified_run(root, "adapted", [False, True])
            atomic = self._verified_run(root, "atomic", [False, False], schedule="atomic", future=0)
            causal = self._verified_run(root, "causal", [True, False], schedule="causal", future=2)
            result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted),
                "--atomic-run-dir", str(atomic), "--causal-run-dir", str(causal)], capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["provenance"]["verified"])
            self.assertEqual("INSUFFICIENT", report["consensus_ramen_decision"]["status"])
            self.assertIn("completed_f4_oracle_recovery", report["consensus_ramen_decision"]["missing_conditions"])
            self.assertEqual("computed", report["oracle_gaps"]["status"])
            self.assertEqual("computed", report["counterfactual"]["variants"]["0.50"]["status"])
            f5 = report["temporal_schedule"]["paired_schedule_comparison"]
            self.assertEqual("computed", f5["status"])
            self.assertEqual(0.5, f5["accuracy_delta"])
            self.assertEqual(2.0, f5["future_support"]["mean_count_delta"])
            self.assertEqual(2.0, f5["future_support"]["mean_weight_fraction_delta"])

    def test_verified_run_rejects_sidecar_f3_f5_diagnostics_that_disagree_with_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            for field, replacement in (("conflict", 0.9), ("future_support_count", 99)):
                with self.subTest(field=field):
                    adapted = self._verified_run(root, f"adapted-{field}", [False, True], future=2)

                    def mutate(rows):
                        rows[0].update({
                            "schedule": "causal", "conflict_metric": "fraction_low_consensus_coordinates_v1",
                            "conflict": 0.2, "batch_position": 0,
                            "future_support_count": 2, "future_support_weight_fraction": 2.0,
                        })
                        rows[0][field] = replacement

                    self._mutate_sidecar_query(adapted, mutate)
                    result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                        "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted)],
                        capture_output=True, text=True)
                    self.assertEqual(2, result.returncode, result.stderr)
                    self.assertEqual("invalid", json.loads(result.stdout)["status"])

    def test_verified_replay_trace_rejects_missing_or_null_failure_analysis_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            for label, mutate in (
                ("missing", lambda failure: failure.pop("support_count")),
                ("null", lambda failure: failure.__setitem__("support_count", None)),
            ):
                with self.subTest(label=label):
                    adapted = self._verified_run(root, f"adapted-{label}", [False, True])
                    trace_path = adapted / "trace.jsonl"
                    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
                    mutate(rows[0]["failure_analysis"])
                    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                    result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                        "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted)],
                        capture_output=True, text=True)
                    self.assertEqual(2, result.returncode, result.stderr)
                    self.assertEqual("invalid", json.loads(result.stdout)["status"])

    def test_verified_pair_rejects_incompatible_required_identity_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            cases = {
                "device": {"device": "mps"},
                "model artifact": {"model_digest": "c" * 64},
                "dataset artifact": {"dataset_digest": "c" * 64},
                "evaluator config": {"evaluator_overrides": {"batch_size": 2}},
                "reference": {"reference_trace": str(root / "not-the-baseline.jsonl")},
                "missing config": {"manifest_mutator": lambda manifest: manifest["args"].pop("stream_seed")},
            }
            for label, options in cases.items():
                with self.subTest(label=label):
                    adapted = self._verified_run(root, f"adapted-{label.replace(' ', '-')}", [False, True], **options)
                    result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                        "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted)], capture_output=True, text=True)
                    self.assertEqual(2, result.returncode, result.stderr)
                    self.assertEqual("invalid", json.loads(result.stdout)["status"])

    def test_verified_pair_accepts_same_backend_v3_shape_and_bound_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for device in ("cpu", "mps"):
                with self.subTest(device=device):
                    baseline = self._verified_run(root, f"baseline-{device}", [True, False], with_sidecar=False,
                                                  device=device)
                    adapted = self._verified_run(root, f"adapted-{device}", [False, True], device=device,
                                                 reference_trace=str(baseline / "trace.jsonl"))
                    result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                        "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted)], capture_output=True, text=True)
                    self.assertEqual(0, result.returncode, result.stderr)

    def test_f5_rejects_unpaired_identity(self):
        atomic, causal = [row(0, True)], [row(1, False)]
        atomic[0].update(schedule="atomic", future_support_count=0, future_support_weight_fraction=0.0)
        causal[0].update(schedule="causal", future_support_count=0, future_support_weight_fraction=0.0)
        with self.assertRaises(ValueError):
            atomic_causal_pairing(atomic, causal)

    def test_f5_verified_runs_require_shared_non_method_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            adapted = self._verified_run(root, "adapted", [False, True])
            cases = {
                "device": {"device": "mps"},
                "model artifact": {"model_digest": "c" * 64},
                "evaluator config": {"evaluator_overrides": {"batch_size": 2}},
            }
            for label, options in cases.items():
                with self.subTest(label=label):
                    atomic = self._verified_run(root, f"atomic-{label.replace(' ', '-')}", [False, False],
                                                schedule="atomic", future=0)
                    causal = self._verified_run(root, f"causal-{label.replace(' ', '-')}", [True, False],
                                                schedule="causal", future=2, **options)
                    result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                        "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted),
                        "--atomic-run-dir", str(atomic), "--causal-run-dir", str(causal)], capture_output=True, text=True)
                    self.assertEqual(2, result.returncode, result.stderr)
                    self.assertEqual("invalid", json.loads(result.stdout)["status"])

    def test_f5_verified_runs_require_identical_adaptation_mechanics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            adapted = self._verified_run(root, "adapted", [False, True])
            mismatches = {
                "learning rate": lambda manifest: manifest["config"].__setitem__("lr", 0.02),
                "topk": lambda manifest: manifest["config"].__setitem__("topk", 3),
                "capacity": lambda manifest: manifest["config"].__setitem__("max_capacity", 5),
                "selection": lambda manifest: manifest["config"].__setitem__("selection", "nearest"),
                "missing config": lambda manifest: manifest.pop("config"),
                "malformed config": lambda manifest: manifest.__setitem__("config", []),
            }
            for label, mutate in mismatches.items():
                with self.subTest(label=label):
                    atomic = self._verified_run(root, f"atomic-{label.replace(' ', '-')}", [False, False],
                                                schedule="atomic", future=0,
                                                evaluator_overrides={"tta_algo": "StructuredAtomicRamen"})
                    causal = self._verified_run(root, f"causal-{label.replace(' ', '-')}", [True, False],
                                                schedule="causal", future=2, manifest_mutator=mutate)
                    result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                        "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted),
                        "--atomic-run-dir", str(atomic), "--causal-run-dir", str(causal)], capture_output=True, text=True)
                    self.assertEqual(2, result.returncode, result.stderr)
                    self.assertEqual("invalid", json.loads(result.stdout)["status"])

    def test_f5_accepts_canonical_schedule_only_method_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            adapted = self._verified_run(root, "adapted", [False, True])
            atomic = self._verified_run(root, "atomic", [False, False], schedule="atomic", future=0,
                                        evaluator_overrides={"tta_algo": "StructuredAtomicRamen"})
            causal = self._verified_run(root, "causal", [True, False], schedule="causal", future=2,
                                        evaluator_overrides={"tta_algo": "CausalRamen"})
            result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted),
                "--atomic-run-dir", str(atomic), "--causal-run-dir", str(causal)], capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("computed", json.loads(result.stdout)["temporal_schedule"]["paired_schedule_comparison"]["status"])

    def test_query_join_rejects_duplicate_exact_f0_identity(self):
        base = [row(0, True), row(0, True)]
        adapted = [row(0, False), row(0, False)]
        with self.assertRaisesRegex(ValueError, "exact F0 trace identities must be unique"):
            analyze_failure_modes(base, adapted, query_rows=[row(0, False), row(0, False)])

    def test_f3_joins_canonical_diagnostic_to_exact_f0_outcomes(self):
        base = [row(0, True), row(1, False)]
        adapted = [row(0, False), row(1, True)]
        queries = [
            {**row(0, False), "conflict_metric": "fraction_low_consensus_coordinates_v1", "conflict": .8},
            {**row(1, True), "conflict_metric": "fraction_low_consensus_coordinates_v1", "conflict": .2},
        ]
        report = analyze_failure_modes(base, adapted, query_rows=queries)
        conflict = report["gradient_conflict"]
        self.assertEqual("computed", conflict["status"])
        self.assertAlmostEqual(.6, conflict["harmful_minus_beneficial"])
        self.assertEqual("fraction_low_consensus_coordinates_v1", conflict["metric"])
        missing = gradient_conflict_association([{**queries[0], "outcome": "harmful"}])
        self.assertEqual("insufficient", missing["status"])
        absent = gradient_conflict_association([{"outcome": "harmful", "conflict": .8}])
        self.assertEqual("unavailable", absent["status"])

    def test_f5_requires_rows_that_explicitly_claim_opposite_schedules(self):
        atomic, causal = [row(0, True)], [row(0, False)]
        for value in atomic + causal:
            value.update(future_support_count=0, future_support_weight_fraction=0.0)
        atomic[0]["schedule"] = "causal"
        causal[0]["schedule"] = "causal"
        result = atomic_causal_pairing(atomic, causal)
        self.assertEqual("insufficient", result["status"])
        self.assertIn("true paired", result["reason"])

    def test_verified_noadapt_baseline_without_sidecar_keeps_unverified_reset_f4_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            adapted = self._verified_run(root, "adapted", [False, True], reset_state_verified=False)
            result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted)], capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual("computed", report["paired_outcomes"]["status"])
            self.assertEqual("computed", report["oracle_gaps"]["status"])
            self.assertEqual("unavailable", report["counterfactual"]["variants"]["0.50"]["status"])

    def test_verified_replay_trace_rejects_malformed_failure_analysis_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._verified_run(root, "baseline", [True, False], with_sidecar=False)
            adapted = self._verified_run(root, "adapted", [False, True], corrupt_failure_trace=True)
            result = subprocess.run([sys.executable, "-m", "src.evaluation.failure_mode_analysis",
                "--baseline-run-dir", str(baseline), "--adapted-run-dir", str(adapted)], capture_output=True, text=True)
            self.assertEqual(2, result.returncode, result.stderr)
            self.assertEqual("invalid", json.loads(result.stdout)["status"])

    def test_consensus_decision_is_fail_closed_and_all_conditions_can_go(self):
        insufficient = consensus_ramen_decision([])
        self.assertEqual("INSUFFICIENT", insufficient["status"])
        self.assertIn("conflict_direction_across_multiple_structured_streams", insufficient["missing_conditions"])
        complete = consensus_ramen_decision([
            {"stream": "bursty", "structured_stream": True, "seed": 1, "harmful_minus_beneficial": .2,
             "f4": {"status": "computed", "harmful_recovery_status": "computed"}},
            {"stream": "recurring", "structured_stream": True, "seed": 2, "harmful_minus_beneficial": .3,
             "f4": {"status": "computed", "harmful_recovery_status": "computed"}},
        ])
        self.assertEqual("GO", complete["status"])


if __name__ == "__main__":
    unittest.main()
