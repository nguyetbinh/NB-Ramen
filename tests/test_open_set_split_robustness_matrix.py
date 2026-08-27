"""Contract tests for the immutable v2/v3 split-robustness planner."""

import json
from pathlib import Path
import tempfile
import unittest

from src.runtime.experiment_matrix import build_command, make_run_id
from src.runtime.open_set_split_robustness_matrix import (
    SPLIT_SHA256,
    build_canonical_open_set_split_robustness_matrix,
    main,
)


class SplitRobustnessMatrixTests(unittest.TestCase):
    def test_plan_is_exactly_96_split_bound_runs_with_three_adaptations_per_baseline(self):
        runs = build_canonical_open_set_split_robustness_matrix(data_root="/tmp/ramen-data")
        self.assertEqual(96, len(runs))
        self.assertEqual(96, len({run.run_id for run in runs}))
        self.assertEqual({"open-set-cifar100-name-rank-v2", "open-set-cifar100-name-rank-v3"},
                         {run.known_class_split for run in runs})
        self.assertTrue(all(run.device == "cuda" and run.max_eval_samples is None
                            and run.stream_block_size == 64 and run.artifact_provenance == "exact"
                            and run.open_set_per_domain_source_budget == 400 for run in runs))
        refs = {}
        for run in runs:
            if run.method != "NoAdapt":
                refs[run.reference_trace] = refs.get(run.reference_trace, 0) + 1
        self.assertEqual(24, len(refs))
        self.assertEqual({3}, set(refs.values()))

    def test_commands_lock_config_and_pass_the_exact_registered_split_path(self):
        runs = build_canonical_open_set_split_robustness_matrix(data_root="/tmp/ramen-data")
        adapted = next(run for run in runs if run.method == "Ramen")
        command = build_command(adapted)
        self.assertIn("--config-lock-path", command)
        self.assertIn(adapted.config_sha256, command)
        index = command.index("--known_class_split_path")
        self.assertTrue(command[index + 1].endswith("open-set-cifar100-split-v2.json"))
        digest_index = command.index("--known_class_split_sha256")
        self.assertEqual(adapted.known_class_split_sha256, command[digest_index + 1])
        self.assertEqual(adapted.known_class_split_sha256, adapted.to_dict()["known_class_split_sha256"])

    def test_v2_and_v3_ids_do_not_collide_while_legacy_ids_remain_compatible(self):
        common = dict(dataset="CIFAR100C", stream_mode="block", seed=0, method="Ramen",
                      device="cuda", config_hash="abc123", artifact_provenance="exact",
                      data_root="/tmp/ramen-data", open_set_ood_ratio=.3,
                      open_set_per_domain_source_budget=400)
        legacy = make_run_id(**common)
        self.assertEqual(
            "cifar100c-block-seed-0-ramen-dev-cuda-full-open-ood-0-3-src-400-"
            "cfg-abc123-prov-exact-data-c6eee46166e8",
            legacy,
        )
        self.assertNotEqual(
            make_run_id(**common, open_set_split_fingerprint=SPLIT_SHA256["open-set-cifar100-name-rank-v2"]),
            make_run_id(**common, open_set_split_fingerprint=SPLIT_SHA256["open-set-cifar100-name-rank-v3"]),
        )

    def test_split_or_config_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "cfg"
            target = config / "CIFAR100C"
            target.mkdir(parents=True)
            repository_config = Path(__file__).parents[1] / "cfg" / "CIFAR100C"
            for name in ("Ramen.yaml", "ConsensusRamen.yaml", "OracleIDGradientRamen.yaml"):
                (target / name).write_bytes((repository_config / name).read_bytes())
            (target / "Ramen.yaml").write_text("drift: true\n")
            with self.assertRaisesRegex(ValueError, "digest drift"):
                build_canonical_open_set_split_robustness_matrix(config_dir=config, data_root="/tmp/ramen-data")

    def test_cli_records_full_hashes_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            # Exercise the plan-only entry point without shell quoting.
            from contextlib import redirect_stdout
            with output.open("w") as handle, redirect_stdout(handle):
                self.assertEqual(0, main(["--data-root", "/tmp/ramen-data"]))
            payload = json.loads(output.read_text())
        self.assertEqual(96, payload["run_count"])
        self.assertEqual(SPLIT_SHA256, payload["split_sha256"])
        self.assertEqual(96, len(payload["artifacts"]))


if __name__ == "__main__":
    unittest.main()
