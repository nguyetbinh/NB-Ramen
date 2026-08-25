"""Planning-contract tests for the separately reported §26 ablation grid."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.runtime.consensus_ablation_matrix import (  # noqa: E402
    CONSENSUS_ABLATION_METHODS,
    build_consensus_ablation_matrix,
    main,
)
from src.runtime.experiment_matrix import build_command


class ConsensusAblationMatrixTests(unittest.TestCase):
    def test_held_out_cuda_plan_has_fixed_identities_config_hashes_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = build_consensus_ablation_matrix(
                stream="recurring", ood_ratio=.3, seeds=(4, 9), device="cuda",
                evidence_dir=directory, data_root=directory,
            )
        self.assertEqual(2 * (len(CONSENSUS_ABLATION_METHODS) + 1), len(runs))
        self.assertEqual(
            {"NoAdapt", *CONSENSUS_ABLATION_METHODS}, {run.method for run in runs}
        )
        by_seed = {}
        for run in runs:
            self.assertTrue(run.open_set)
            self.assertEqual("open-set-cifar100-split-v1", run.known_class_split)
            self.assertEqual(.3, run.ood_ratio)
            self.assertEqual(400, run.open_set_per_domain_source_budget)
            by_seed.setdefault(run.seed, []).append(run)
            if run.method != "NoAdapt":
                self.assertIsNotNone(run.config_path)
                self.assertNotEqual("missing", run.config_hash)
                self.assertEqual("cuda", run.device)
                self.assertEqual(
                    next(item for item in runs if item.seed == run.seed and item.method == "NoAdapt").run_dir / "trace.jsonl",
                    run.reference_trace,
                )
                command = build_command(run)
                self.assertIn("--open_set", command)
                self.assertEqual("400", command[command.index("--open-set-per-domain-source-budget") + 1])
        for runs_for_seed in by_seed.values():
            self.assertEqual("NoAdapt", runs_for_seed[0].method)

        identities = {run.method: run for run in by_seed[4] if run.method != "NoAdapt"}
        self.assertEqual(.2, identities["ConsensusRamen"].config_data["consensus_threshold"])
        self.assertEqual("hard_mask", identities["ConsensusRamen"].config_data["consensus_mode"])
        self.assertEqual("soft_weight", identities["ConsensusRamenSoft"].config_data["consensus_mode"])
        self.assertFalse(identities["ConsensusRamenNoSelf"].config_data["include_current"])
        self.assertEqual(.6, identities["ConsensusRamenTau060"].config_data["consensus_threshold"])
        self.assertEqual(2, identities["ConsensusRamenMin2"].config_data["min_consensus_classes"])
        self.assertEqual(4, identities["ConsensusRamenMin4"].config_data["min_consensus_classes"])
        self.assertEqual(len(identities), len({run.config_hash for run in identities.values()}))

    def test_canonical_rejects_non_cuda_and_pilots_require_explicit_opt_in_and_name(self):
        kwargs = {"stream": "block", "ood_ratio": .5, "seeds": (0,)}
        with self.assertRaisesRegex(ValueError, "requires device='cuda'"):
            build_consensus_ablation_matrix(**kwargs, device="mps")
        with self.assertRaisesRegex(ValueError, "requires a non-empty pilot_name"):
            build_consensus_ablation_matrix(**kwargs, canonical=False, device="mps")
        with self.assertRaisesRegex(ValueError, "pilot_name is only valid"):
            build_consensus_ablation_matrix(**kwargs, pilot_name="mps-check")
        with tempfile.TemporaryDirectory() as directory:
            pilot = build_consensus_ablation_matrix(
                **kwargs, canonical=False, pilot_name="mps-check", device="mps",
                evidence_dir=directory, data_root=directory, max_eval_samples=16,
            )
        self.assertTrue(all(run.device == "mps" for run in pilot))
        self.assertTrue(all(run.open_set for run in pilot))

    def test_rejects_non_preregistered_cell_values_and_unmatched_source_budget(self):
        kwargs = {"stream": "block", "ood_ratio": .3, "seeds": (0,), "device": "cuda"}
        with self.assertRaisesRegex(ValueError, "held-out stream"):
            build_consensus_ablation_matrix(**{**kwargs, "stream": "gradual"})
        with self.assertRaisesRegex(ValueError, "held-out OOD ratio"):
            build_consensus_ablation_matrix(**{**kwargs, "ood_ratio": .2})
        with self.assertRaisesRegex(ValueError, "must support"):
            build_consensus_ablation_matrix(**kwargs, per_domain_source_budget=401)

    def test_cli_emits_a_plan_only_for_the_selected_cell(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code = main([
                "--stream", "iid_mixed", "--ood-ratio", "0.1", "--seed", "12",
                "--device", "mps", "--noncanonical-pilot", "--pilot-name", "sanity-mps",
                "--evidence-dir", directory, "--data-root", directory, "--max-eval-samples", "16",
            ])
        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual("planned_not_executed", payload["status"])
        self.assertFalse(payload["canonical"])
        self.assertEqual(len(CONSENSUS_ABLATION_METHODS) + 1, payload["run_count"])
        self.assertTrue(all("--open_set" in command for command in payload["commands"]))


if __name__ == "__main__":
    unittest.main()
