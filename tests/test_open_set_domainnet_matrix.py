"""Contract tests for the planner-only DomainNet Phase E matrix."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.runtime.experiment_matrix import build_command
from src.runtime.open_set_domainnet_matrix import (
    DOMAINNET_OPEN_SET_METHODS,
    DOMAINNET_OPEN_SET_OOD_RATIOS,
    DOMAINNET_OPEN_SET_PER_DOMAIN_SOURCE_BUDGET,
    DOMAINNET_OPEN_SET_SEEDS,
    DOMAINNET_OPEN_SET_SPLIT,
    DOMAINNET_OPEN_SET_STREAMS,
    build_domainnet_open_set_evidence_matrix,
    main,
)


class DomainNetOpenSetMatrixTests(unittest.TestCase):
    def test_locked_plan_resolves_dataset_configs_and_exact_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = build_domainnet_open_set_evidence_matrix(
                evidence_dir=Path(directory) / "evidence", data_root=Path(directory) / "data"
            )
        self.assertEqual(252, len(runs))
        self.assertEqual(set(DOMAINNET_OPEN_SET_METHODS), {run.method for run in runs})
        self.assertEqual(set(DOMAINNET_OPEN_SET_STREAMS), {run.stream_mode for run in runs})
        self.assertEqual(set(DOMAINNET_OPEN_SET_OOD_RATIOS), {run.ood_ratio for run in runs})
        self.assertEqual(set(DOMAINNET_OPEN_SET_SEEDS), {run.seed for run in runs})
        self.assertEqual(252, len({run.run_id for run in runs}))

        baselines = {}
        for run in runs:
            self.assertTrue(run.open_set)
            self.assertEqual(DOMAINNET_OPEN_SET_SPLIT, run.known_class_split)
            self.assertEqual(DOMAINNET_OPEN_SET_PER_DOMAIN_SOURCE_BUDGET, run.open_set_per_domain_source_budget)
            key = (run.ood_ratio, run.stream_mode, run.seed)
            if run.method == "NoAdapt":
                baselines[key] = run
                self.assertIsNone(run.reference_trace)
                continue
            self.assertEqual(baselines[key].run_dir / "trace.jsonl", run.reference_trace)
            self.assertEqual("DomainNet", run.config_path.parent.name)
            expected_hash = hashlib.sha256(run.config_path.read_bytes()).hexdigest()[:12]
            self.assertEqual(expected_hash, run.config_hash)
            self.assertNotEqual("missing", run.config_hash)
            command = build_command(run)
            self.assertIn("--open_set", command)
            self.assertEqual(DOMAINNET_OPEN_SET_SPLIT, command[command.index("--known_class_split") + 1])
            self.assertEqual("690", command[command.index("--open-set-per-domain-source-budget") + 1])
            self.assertEqual(str(run.reference_trace), command[command.index("--reference_trace") + 1])

        self.assertEqual(36, len(baselines))
        sample = next(run for run in runs if run.method == "OracleConsensusRamen")
        self.assertEqual("evaluator_is_ood", sample.config_data["oracle_ood_source"])
        self.assertEqual("hard_mask", sample.config_data["consensus_mode"])
        self.assertEqual(.2, sample.config_data["consensus_threshold"])

    def test_rejects_any_noncanonical_identity_or_cost_limited_request(self):
        base = {"evidence_dir": "/tmp/evidence", "data_root": "/tmp/data"}
        with self.assertRaisesRegex(ValueError, "device='cuda'"):
            build_domainnet_open_set_evidence_matrix(**base, device="mps")
        with self.assertRaisesRegex(ValueError, "full streams"):
            build_domainnet_open_set_evidence_matrix(**base, max_eval_samples=32)
        with self.assertRaisesRegex(ValueError, "artifact_provenance"):
            build_domainnet_open_set_evidence_matrix(**base, artifact_provenance="off")
        with self.assertRaisesRegex(ValueError, "source budget is fixed"):
            build_domainnet_open_set_evidence_matrix(**base, per_domain_source_budget=400)
        with self.assertRaisesRegex(ValueError, "streams is fixed"):
            build_domainnet_open_set_evidence_matrix(**base, streams=("block",))
        with self.assertRaisesRegex(ValueError, "OOD ratios is fixed"):
            build_domainnet_open_set_evidence_matrix(**base, ood_ratios=(.3,))
        with self.assertRaisesRegex(ValueError, "seeds is fixed"):
            build_domainnet_open_set_evidence_matrix(**base, seeds=(0,))
        with self.assertRaisesRegex(ValueError, "methods is fixed"):
            build_domainnet_open_set_evidence_matrix(**base, methods=("NoAdapt", "Ramen"))

    def test_cli_is_explicitly_planner_only(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code = main(["--evidence-dir", directory, "--data-root", directory])
        self.assertEqual(0, exit_code)
        payload = json.loads(output.getvalue())
        self.assertEqual("planned_not_executed", payload["status"])
        self.assertTrue(payload["canonical"])
        self.assertEqual("DomainNet", payload["dataset"])
        self.assertEqual(252, payload["run_count"])
        self.assertEqual(252, len(payload["commands"]))
        self.assertTrue(all("--open_set" in command for command in payload["commands"]))


if __name__ == "__main__":
    unittest.main()
