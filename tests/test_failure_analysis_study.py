import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.failure_analysis_study import aggregate_failure_analysis_study, load_and_aggregate_failure_analysis_study


THRESHOLDS = [0.50, 0.75, 1.00]


def _report(direction=0.2, *, f4=True, provenance=None):
    recovery = "computed" if f4 else "insufficient"
    variants = {f"{value:.2f}": {"status": "computed", "harmful_recovery_status": recovery} for value in THRESHOLDS}
    outcomes = {outcome: {metric: {"status": "computed"} for metric in (
        "consensus_mean", "consensus_p10", "consensus_p50", "fraction_low_consensus_coordinates",
        "active_support_classes", "pairwise_sign_agreement_mean", "pairwise_cosine_mean")}
        for outcome in ("safe", "beneficial", "harmful", "unresolved")}
    return {"status": "computed", "provenance": provenance or {"verified": True},
            "paired_outcomes": {"status": "computed"}, "oracle_gaps": {"status": "computed"},
            "gradient_conflict": {"status": "computed", "metric": "fraction_low_consensus_coordinates_v1", "harmful_minus_beneficial": direction},
            "gradient_conflict_distributions": {"status": "computed", "outcomes": outcomes},
            "counterfactual": {"status": "computed", "variants": variants},
            "temporal_schedule": {"status": "computed", "paired_schedule_comparison": {"status": "computed"}},
            "entropy_admission": {"status": "computed"}}


def _identity(stream, seed, *, device="cpu"):
    return {"device": device, "dataset": "CIFAR100C", "stream": stream, "seed": seed,
            "method": "CausalRamen", "analysis_role": "analysis",
            "failure_counterfactual_thresholds": THRESHOLDS}


class FailureAnalysisStudyTests(unittest.TestCase):
    """Fixtures model completed, checksum-bound run directories without replaying a model."""

    def _manifest(self, root, cells):
        declared = []
        for index, (identity, report) in enumerate(cells):
            report_path = root / f"report-{index}.json"; report_path.write_text(json.dumps(report), encoding="utf-8")
            (root / f"base-{index}").mkdir(exist_ok=True); (root / f"adapted-{index}").mkdir(exist_ok=True)
            declared.append({"report": report_path.name, "baseline_run_dir": f"base-{index}",
                             "adapted_run_dir": f"adapted-{index}", "run_identity": identity})
        return {"preregistered_counterfactual_thresholds": THRESHOLDS, "cells": declared}

    def _aggregate(self, manifest, root, *, actual_role=None, recomputed_direction=None, verified_overrides=None):
        verified_overrides = verified_overrides or {}
        def trace(path):
            name = Path(path).name
            index = int(name.rsplit("-", 1)[1])
            cell = manifest["cells"][index]
            identity = cell["run_identity"]
            overrides = verified_overrides.get(index, {})
            is_base = name.startswith("base-")
            run_id = name
            artifact_identity = {"run_id": run_id, "manifest_sha256": ("a" if is_base else "b") * 64,
                                 "stream_fingerprint": "c" * 64,
                                 "source_fingerprint": overrides.get("source_fingerprint", "d" * 64)}
            args = {"dataset": identity["dataset"], "stream_mode": identity["stream"], "seed": identity["seed"],
                    "tta_algo": "NoAdapt" if is_base else identity["method"]}
            args["analysis_role"] = identity.get("analysis_role") if actual_role is None else actual_role
            args.update(overrides.get("args", {}))
            return Path(path), {"device": identity["device"], "args": args,
                                "config": overrides.get("config", {"beta": 5.0, "topk": 5}),
                                "artifacts": {"status": "verified", "model": {"status": "verified", "actual_sha256": overrides.get("model_digest", "e" * 64)},
                                              "dataset": {"status": "verified", "root_digest": overrides.get("dataset_digest", "f" * 64)}}}, [], artifact_identity
        def recompute(_baseline, adapted):
            index = int(Path(adapted).name.rsplit("-", 1)[1])
            report = json.loads((root / manifest["cells"][index]["report"]).read_text())
            if recomputed_direction is not None:
                report["gradient_conflict"]["harmful_minus_beneficial"] = recomputed_direction
            return report
        # Add the expected report bindings after knowing each artificial run identity.
        for index, cell in enumerate(manifest["cells"]):
            report_path = root / cell["report"]
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text())
            source = verified_overrides.get(index, {}).get("source_fingerprint", "d" * 64)
            report["provenance"] = {"verified": True,
                "baseline": {"run_id": f"base-{index}", "manifest_sha256": "a" * 64, "stream_fingerprint": "c" * 64, "source_fingerprint": source},
                "adapted": {"run_id": f"adapted-{index}", "manifest_sha256": "b" * 64, "stream_fingerprint": "c" * 64, "source_fingerprint": source}}
            report_path.write_text(json.dumps(report), encoding="utf-8")
        with patch("src.evaluation.failure_analysis_study._verified_trace_run", side_effect=trace), \
             patch("src.evaluation.failure_analysis_study.analyze_verified_run_dirs", side_effect=recompute):
            return aggregate_failure_analysis_study(manifest, manifest_root=root)

    def test_two_seed_two_structured_stream_bound_go(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root, [(_identity(stream, seed), _report()) for stream in ("block", "gradual") for seed in (1, 2)])
            result = self._aggregate(manifest, root)
        self.assertEqual("GO", result["status"])
        self.assertEqual(4, result["evidence_counts"]["eligible_consensus_cells"])

    def test_rejects_cross_cell_verified_scientific_identity_mismatches(self):
        cases = {
            "source fingerprint": {1: {"source_fingerprint": "z" * 64}},
            "model artifact digest": {1: {"model_digest": "z" * 64}},
            "dataset artifact digest": {1: {"dataset_digest": "z" * 64}},
            "adapted method config": {1: {"config": {"beta": 7.0, "topk": 5}}},
            "non-stream/non-seed evaluator settings": {1: {"args": {"max_eval_samples": 99}}},
        }
        for expected, overrides in cases.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = self._manifest(root, [(_identity(stream, seed), _report()) for stream in ("block", "gradual") for seed in (1, 2)])
                result = self._aggregate(manifest, root, verified_overrides=overrides)
            self.assertEqual("invalid", result["status"])
            self.assertTrue(any(f"verified {expected}" in error for error in result["errors"]))

    def test_rejects_unbound_report_traversal_reuse_and_seed_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root, [(_identity("block", 1), _report()), (_identity("gradual", 2), _report())])
            manifest["cells"][0]["report"] = "../outside.json"
            self.assertEqual("invalid", self._aggregate(manifest, root)["status"])
            manifest = self._manifest(root, [(_identity("block", 1), _report()), (_identity("gradual", "1"), _report())])
            self.assertEqual("invalid", self._aggregate(manifest, root)["status"])
            manifest = self._manifest(root, [(_identity("block", 1), _report()), (_identity("gradual", 2), _report())])
            manifest["cells"][1]["report"] = manifest["cells"][0]["report"]
            self.assertEqual("invalid", self._aggregate(manifest, root)["status"])

    def test_rejects_declared_role_that_disagrees_with_actual_manifest_args(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root, [(_identity("block", 1), _report()), (_identity("gradual", 2), _report())])
            result = self._aggregate(manifest, root, actual_role="final")
            self.assertEqual("invalid", result["status"])
            self.assertTrue(any("analysis_role disagrees" in error for error in result["errors"]))

    def test_rejects_legacy_declared_role_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root, [(_identity("block", 1), _report()), (_identity("gradual", 2), _report())])
            identity = manifest["cells"][0]["run_identity"]
            identity["role"] = identity.pop("analysis_role")
            self.assertEqual("invalid", self._aggregate(manifest, root)["status"])

    def test_threshold_f4_and_per_stream_replication_never_go(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root, [(_identity("block", 1), _report()), (_identity("block", 2), _report()),
                                             (_identity("gradual", 1), _report(f4=False))])
            self.assertEqual("INSUFFICIENT", self._aggregate(manifest, root)["status"])
            manifest = self._manifest(root, [(_identity("block", 1), _report()), (_identity("gradual", 2), _report())])
            report = json.loads((root / manifest["cells"][0]["report"]).read_text())
            report["counterfactual"]["variants"].pop("1.00")
            (root / manifest["cells"][0]["report"]).write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual("invalid", self._aggregate(manifest, root)["status"])

    def test_rejects_report_values_that_disagree_with_verified_recomputation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._manifest(root, [(_identity("block", 1), _report(direction=.9))])
            result = self._aggregate(manifest, root, recomputed_direction=.2)
        self.assertEqual("invalid", result["status"])
        self.assertTrue(any("does not exactly match recomputation" in error for error in result["errors"]))

    def test_loader_defaults_to_repository_root_for_checked_in_manifest_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            repository_root = Path(temp)
            manifest_dir = repository_root / "plans" / "study"
            manifest_dir.mkdir(parents=True)
            manifest = self._manifest(manifest_dir, [(_identity("block", 1), _report())])
            # Checked-in manifests address both plans and evidence from the repository root.
            manifest["cells"][0]["report"] = "plans/study/report-0.json"
            manifest["cells"][0]["baseline_run_dir"] = "evidence/base-0"
            manifest["cells"][0]["adapted_run_dir"] = "evidence/adapted-0"
            (repository_root / "evidence" / "base-0").mkdir(parents=True)
            (repository_root / "evidence" / "adapted-0").mkdir(parents=True)
            manifest_path = manifest_dir / "study-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            def trace(path):
                name = Path(path).name
                is_base = name.startswith("base-")
                identity = manifest["cells"][0]["run_identity"]
                return Path(path), {"device": identity["device"], "args": {
                    "dataset": identity["dataset"], "stream_mode": identity["stream"], "seed": identity["seed"],
                    "tta_algo": "NoAdapt" if is_base else identity["method"], "analysis_role": identity["analysis_role"]},
                    "config": {"beta": 5.0, "topk": 5},
                    "artifacts": {"status": "verified", "model": {"status": "verified", "actual_sha256": "e" * 64},
                                  "dataset": {"status": "verified", "root_digest": "f" * 64}}}, [], {
                    "run_id": name, "manifest_sha256": ("a" if is_base else "b") * 64,
                    "stream_fingerprint": "c" * 64, "source_fingerprint": "d" * 64}

            report_path = manifest_dir / "report-0.json"
            report = json.loads(report_path.read_text())
            report["provenance"] = {"verified": True,
                "baseline": {"run_id": "base-0", "manifest_sha256": "a" * 64, "stream_fingerprint": "c" * 64, "source_fingerprint": "d" * 64},
                "adapted": {"run_id": "adapted-0", "manifest_sha256": "b" * 64, "stream_fingerprint": "c" * 64, "source_fingerprint": "d" * 64}}
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual("invalid", aggregate_failure_analysis_study(manifest, manifest_root=manifest_dir)["status"])
            with patch("src.evaluation.failure_analysis_study._verified_trace_run", side_effect=trace), \
                 patch("src.evaluation.failure_analysis_study.analyze_verified_run_dirs", return_value=report), \
                 patch("src.evaluation.failure_analysis_study._default_artifact_root", return_value=repository_root):
                result = load_and_aggregate_failure_analysis_study(manifest_path)
        self.assertEqual("INSUFFICIENT", result["status"])
        self.assertEqual("plans/study/report-0.json", result["cells"][0]["report"])



if __name__ == "__main__":
    unittest.main()
