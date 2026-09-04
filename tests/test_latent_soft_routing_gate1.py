import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "latent_soft_gate1", ROOT / "scripts" / "run-latent-soft-routing-gate1.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LatentSoftRoutingGate1Tests(unittest.TestCase):
    def test_protocol_materializes_minimal_fixed_gamma_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            groups = MODULE.build_gate1_groups(
                evidence_dir=Path(directory) / "evidence",
                data_root=Path(directory) / "data",
                device="cpu",
            )
        self.assertEqual(
            ["controls", "gamma-0", "gamma-0.25"],
            [name for name, _ in groups],
        )
        adapted = [run for _, runs in groups for run in runs if run.method != "NoAdapt"]
        self.assertEqual(
            ["CausalRamen", "OracleHardRamen"] + ["OracleSoftRankRamen"] * 2,
            [run.method for run in adapted],
        )
        self.assertEqual(
            [0.0, 0.25],
            [run.config_data["gamma"] for run in adapted[2:]],
        )
        self.assertTrue(all(run.max_eval_samples == 200 for _, runs in groups for run in runs))
        self.assertTrue(all(run.stream_mode == "block" for _, runs in groups for run in runs))
        self.assertEqual(
            len({run.run_id for _, runs in groups for run in runs}),
            sum(len(runs) for _, runs in groups),
        )


if __name__ == "__main__":
    unittest.main()
