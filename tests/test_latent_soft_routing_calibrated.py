import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "latent_soft_calibrated", ROOT / "scripts" / "run-latent-soft-routing-calibrated.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _profile_evidence(directory: Path, values=(.1, .2, .3)):
    run = next(run for run in MODULE._base_runs(
        config_dir=MODULE.MARGIN_PROFILE_CONFIG_ROOT,
        evidence_dir=directory / "profile-evidence", data_root=directory / "data",
        device="cpu", provenance="fast",
    ) if run.method == "OracleSoftRankRamen")
    run.run_dir.mkdir(parents=True, exist_ok=True)
    (run.run_dir / "trace.jsonl").write_text('{"profile": true}\n', encoding="utf-8")
    return {
        "run": run,
        "manifest": {},
        "summary": {
            "stream_fingerprint": "a" * 64,
            "replacement_margin_profile": {
                "replacement_margin_p25": values[0],
                "replacement_margin_p50": values[1],
                "replacement_margin_p75": values[2],
            },
        },
    }


class LatentSoftRoutingCalibratedTests(unittest.TestCase):
    def test_derivation_advances_quantiles_and_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = _profile_evidence(Path(directory))
            first = MODULE.derive_calibration(evidence)
            second = MODULE.derive_calibration(evidence)
        self.assertEqual(first, second)
        self.assertEqual(64, len(first["calibration_id"]))
        gammas = [entry["gamma"] for entry in first["strengths"]]
        self.assertTrue(all(a < b for a, b in zip(gammas, gammas[1:])))
        self.assertGreater(gammas[0], .1)
        self.assertLess(gammas[0] - .1, 1e-7)

    def test_configs_are_deterministic_json_yaml_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = _profile_evidence(Path(directory))
            calibration = MODULE.derive_calibration(evidence)
            first = MODULE.materialize_configs(calibration, Path(directory) / "configs")
            contents = {name: path.read_text(encoding="utf-8") for name, path in first.items()}
            second = MODULE.materialize_configs(calibration, Path(directory) / "configs")
            self.assertEqual(first, second)
            self.assertTrue(all(path.read_text(encoding="utf-8") == contents[name] for name, path in first.items()))
            payload = json.loads(first["medium"].read_text(encoding="utf-8"))
            self.assertEqual(calibration["calibration_id"], payload["calibration_id"])
            self.assertFalse(payload["profile_replacement_margins"])

    def test_calibration_file_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = MODULE.derive_calibration(_profile_evidence(root))
            path = MODULE.write_calibration(calibration, root / "calibration.json")
            MODULE.write_calibration(calibration, path)
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable calibration"):
                MODULE.write_calibration(calibration, path)

    def test_plan_contains_only_three_external_baseline_soft_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = MODULE.derive_calibration(_profile_evidence(root))
            baseline = MODULE.canonical_baseline(
                evidence_dir=root / "baseline", data_root=root / "data", device="cpu", provenance="fast",
            )
            runs = MODULE.plan_calibrated_runs(
                calibration=calibration, config_root=root / "configs", evidence_dir=root / "runs",
                baseline=baseline, data_root=root / "data", device="cpu", provenance="fast",
            )
        self.assertEqual(3, len(runs))
        self.assertEqual(["OracleSoftRankRamen"] * 3, [run.method for run in runs])
        self.assertEqual(3, len({run.run_id for run in runs}))
        self.assertTrue(all(run.reference_trace == baseline.run_dir / "trace.jsonl" for run in runs))
        self.assertTrue(all(run.reference_config_path == baseline.config_path for run in runs))

    def test_rejects_duplicate_gammas_and_tampered_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "strictly ordered"):
                MODULE.derive_calibration(_profile_evidence(Path(directory), (.1, .1, .3)))
            calibration = MODULE.derive_calibration(_profile_evidence(Path(directory), (.1, .2, .3)))
        calibration["strengths"][0]["gamma"] = math.nextafter(calibration["strengths"][0]["gamma"], math.inf)
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.validate_calibration(calibration)

    def test_rejects_empty_profile_and_missing_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = _profile_evidence(Path(directory))
            evidence["summary"]["replacement_margin_profile"] = {
                "replacement_margin_p25": None,
                "replacement_margin_p50": None,
                "replacement_margin_p75": None,
            }
            with self.assertRaisesRegex(ValueError, "empty"):
                MODULE.derive_calibration(evidence)
            evidence = _profile_evidence(Path(directory), (.1, .2, .3))
            (evidence["run"].run_dir / "trace.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                MODULE.derive_calibration(evidence)

    def test_rejects_calibration_source_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = _profile_evidence(Path(directory))
            run = evidence["run"]
            (run.run_dir / "summary.json").write_text(json.dumps({
                "run_id": run.run_id, "stream_fingerprint": "a" * 64,
            }), encoding="utf-8")
            calibration = MODULE.derive_calibration(evidence)
            MODULE.verify_calibration_source(calibration)
            (run.run_dir / "trace.jsonl").write_text('{"profile": false}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trace identity mismatch"):
                MODULE.verify_calibration_source(calibration)


if __name__ == "__main__":
    unittest.main()
