"""Tests for planner-to-launch configuration byte locks."""

import contextlib
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main import args_parser
from runtime.experiment_matrix import build_command, build_experiment_matrix


class ConfigLaunchLockTests(unittest.TestCase):
    def _parse(self, *arguments):
        with patch("sys.argv", ["main.py", "--device", "cpu", *arguments]):
            return args_parser()

    def _config(self, root: Path, relative: str = "DomainNet/Ramen.yaml", content: str = "lr: 0.1\n") -> Path:
        path = root / "cfg" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path.resolve()

    def test_lock_accepts_exact_resolved_path_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = self._config(root)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            args = self._parse(
                "--dataset", "DomainNet", "--tta_algo", "Ramen", "--config", str(root / "cfg"),
                "--config-lock-path", str(path), "--config-lock-sha256", digest,
            )
        self.assertEqual(str(path), args.config_path)
        self.assertEqual({"lr": 0.1}, args.config)

    def test_lock_rejects_partial_flags_hash_drift_and_fallback_substitution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = self._config(root)
            common = ("--dataset", "DomainNet", "--tta_algo", "Ramen", "--config", str(root / "cfg"))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self._parse(*common, "--config-lock-path", str(path))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self._parse(*common, "--config-lock-path", str(path), "--config-lock-sha256", "0" * 64)

            path.unlink()
            fallback = self._config(root, "default/Ramen.yaml")
            digest = hashlib.sha256(fallback.read_bytes()).hexdigest()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self._parse(*common, "--config-lock-path", str(path), "--config-lock-sha256", digest)

    def test_legacy_unlocked_invocation_still_uses_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fallback = self._config(root, "default/Ramen.yaml")
            args = self._parse(
                "--dataset", "DomainNet", "--tta_algo", "Ramen", "--config", str(root / "cfg"),
            )
        self.assertEqual(str(fallback), args.config_path)

    def test_adapted_command_carries_full_lock_but_baseline_does_not(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._config(root, "DomainNet/Ramen.yaml")
            self._config(root, "DomainNet/NoAdapt.yaml")
            baseline, adapted = build_experiment_matrix(
                datasets=("DomainNet",), streams=("block",), methods=("Ramen",), seeds=(0,),
                config_dir=root / "cfg", evidence_dir=root / "evidence", data_root=root, device="cpu",
            )
            command = build_command(adapted)
            self.assertEqual(str(adapted.config_path), command[command.index("--config-lock-path") + 1])
            self.assertEqual(
                adapted.config_sha256,
                command[command.index("--config-lock-sha256") + 1],
            )
            self.assertEqual(hashlib.sha256(adapted.config_path.read_bytes()).hexdigest(), adapted.config_sha256)
            self.assertNotIn("--config-lock-path", build_command(baseline))
            self.assertIsNone(baseline.config_sha256)

            # This preserves the parsed YAML and the short planned prefix is
            # not used for the decision: any byte drift must fail closed.
            adapted.config_path.write_text("# changed after planning\nlr: 0.1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "effective config changed after planning"):
                build_command(adapted)

    def test_registered_v2_v3_split_byte_locks_parse_exactly(self):
        research = Path(__file__).parents[1] / "cfg" / "research"
        for version in (2, 3):
            with self.subTest(version=version):
                path = (research / f"open-set-cifar100-split-v{version}.json").resolve()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                args = self._parse(
                    "--open_set", "--dataset", "CIFAR100C",
                    "--known_class_split", f"open-set-cifar100-name-rank-v{version}",
                    "--known-class-split-path", str(path),
                    "--known-class-split-sha256", digest,
                )
                self.assertEqual(str(path), args.known_class_split_path)
                self.assertEqual(digest, args.known_class_split_sha256)

    def test_split_lock_flags_are_all_or_none_and_reject_byte_drift(self):
        path = (Path(__file__).parents[1] / "cfg" / "research" / "open-set-cifar100-split-v2.json").resolve()
        common = (
            "--open_set", "--dataset", "CIFAR100C",
            "--known_class_split", "open-set-cifar100-name-rank-v2",
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._parse(*common, "--known-class-split-path", str(path))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._parse(
                *common, "--known-class-split-path", str(path),
                "--known-class-split-sha256", "0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
