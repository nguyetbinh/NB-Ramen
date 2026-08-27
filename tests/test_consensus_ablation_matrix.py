"""Planning-contract tests for the preregistered Consensus heldout-v1 grid."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.runtime.consensus_ablation_matrix import (
    CANONICAL_CONFIG_SHA256,
    CANONICAL_CONFIG_SURFACES,
    CANONICAL_SEEDS,
    CANONICAL_STREAMS,
    CONSENSUS_ABLATION_METHODS,
    PREREGISTRATION_VERSION,
    build_canonical_consensus_ablation_heldout_v1,
    build_noncanonical_consensus_ablation_pilot,
    main,
)
from src.runtime.experiment_matrix import REPOSITORY_ROOT, build_command


class ConsensusAblationMatrixTests(unittest.TestCase):
    def test_heldout_v1_is_exactly_72_unique_runs_with_nine_seven_way_pairings(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = build_canonical_consensus_ablation_heldout_v1(
                evidence_dir=directory, data_root=directory,
            )
        self.assertEqual(72, len(runs))
        self.assertEqual(72, len({run.run_id for run in runs}))
        self.assertEqual({"NoAdapt", *CONSENSUS_ABLATION_METHODS}, {run.method for run in runs})
        self.assertEqual({(stream, seed) for stream in CANONICAL_STREAMS for seed in CANONICAL_SEEDS},
                         {(run.stream_mode, run.seed) for run in runs})
        baseline_references = {}
        for run in runs:
            self.assertTrue(run.open_set)
            self.assertEqual("open-set-cifar100-split-v1", run.known_class_split)
            self.assertEqual(.3, run.ood_ratio)
            self.assertEqual(400, run.open_set_per_domain_source_budget)
            self.assertEqual("cuda", run.device)
            self.assertIsNone(run.max_eval_samples)
            self.assertEqual(64, run.stream_block_size)
            self.assertEqual("exact", run.artifact_provenance)
            if run.method != "NoAdapt":
                baseline_references[run.reference_trace] = baseline_references.get(run.reference_trace, 0) + 1
                self.assertEqual(CANONICAL_CONFIG_SURFACES[run.method], run.config_data)
                self.assertEqual(CANONICAL_CONFIG_SHA256[run.method][:12], run.config_hash)
                self.assertIn("--reference_trace", build_command(run))
        self.assertEqual(9, len(baseline_references))
        self.assertEqual({7}, set(baseline_references.values()))

    def test_heldout_v1_rejects_missing_fallback_digest_and_semantic_config_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory) / "cfg"
            shutil.copytree(REPOSITORY_ROOT / "cfg" / "CIFAR100C", config_root / "CIFAR100C")
            (config_root / "CIFAR100C" / "ConsensusRamenMin4.yaml").unlink()
            with self.assertRaisesRegex(ValueError, "missing or resolved a fallback"):
                build_canonical_consensus_ablation_heldout_v1(config_dir=config_root, data_root=directory)

        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory) / "cfg"
            shutil.copytree(REPOSITORY_ROOT / "cfg" / "CIFAR100C", config_root / "CIFAR100C")
            config = config_root / "CIFAR100C" / "Ramen.yaml"
            config.write_text(config.read_text().replace("beta: 5.0", "beta: 5.1"))
            with self.assertRaisesRegex(ValueError, "config digest drift"):
                build_canonical_consensus_ablation_heldout_v1(config_dir=config_root, data_root=directory)

        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory) / "cfg"
            shutil.copytree(REPOSITORY_ROOT / "cfg" / "CIFAR100C", config_root / "CIFAR100C")
            config = config_root / "CIFAR100C" / "ConsensusRamen.yaml"
            config.write_text(config.read_text() + "\nextra_semantic_key: true\n")
            with self.assertRaisesRegex(ValueError, "config digest drift"):
                build_canonical_consensus_ablation_heldout_v1(config_dir=config_root, data_root=directory)

    def test_custom_cells_are_explicit_noncanonical_named_pilots(self):
        kwargs = {"stream": "block", "ood_ratio": .5, "seeds": (0,)}
        with self.assertRaisesRegex(ValueError, "non-empty pilot_name"):
            build_noncanonical_consensus_ablation_pilot(**kwargs, pilot_name="")
        with tempfile.TemporaryDirectory() as directory:
            pilot = build_noncanonical_consensus_ablation_pilot(
                **kwargs, pilot_name="mps-check", device="mps", evidence_dir=directory,
                data_root=directory, max_eval_samples=16,
            )
        self.assertEqual(8, len(pilot))
        self.assertTrue(all(run.device == "mps" for run in pilot))
        self.assertTrue(all(run.open_set for run in pilot))

    def test_cli_emits_preregistered_heldout_v1_plan_only_with_commands_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code = main(["--evidence-dir", directory, "--data-root", directory])
        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual("planned_not_executed", payload["status"])
        self.assertTrue(payload["canonical"])
        self.assertEqual(PREREGISTRATION_VERSION, payload["preregistration_version"])
        self.assertEqual(72, payload["run_count"])
        self.assertEqual(72, len(payload["commands"]))
        self.assertEqual(72, len(payload["artifacts"]))
        self.assertEqual(CANONICAL_CONFIG_SHA256, payload["config_sha256"])
        self.assertTrue(all("--artifact-provenance" in command for command in payload["commands"]))

    def test_cli_rejects_custom_canonical_knobs_and_marks_pilot_noncanonical(self):
        with self.assertRaisesRegex(SystemExit, "2"):
            main(["--stream", "block"])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            exit_code = main([
                "--noncanonical-pilot", "--pilot-name", "sanity-mps", "--stream", "iid_mixed",
                "--ood-ratio", "0.1", "--seed", "12", "--device", "mps",
                "--evidence-dir", directory, "--data-root", directory, "--max-eval-samples", "16",
            ])
        payload = json.loads(output.getvalue())
        self.assertEqual(0, exit_code)
        self.assertFalse(payload["canonical"])
        self.assertIsNone(payload["preregistration_version"])
        self.assertEqual("sanity-mps", payload["pilot_name"])


if __name__ == "__main__":
    unittest.main()
