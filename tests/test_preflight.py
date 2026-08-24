import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.runtime import preflight


class DatasetPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _cifar10_layout(self):
        directory = self.root / "corruption" / "CIFAR-10-C"
        directory.mkdir(parents=True)
        (directory / "labels.npy").touch()
        for corruption in preflight.MAIN_CORRUPTIONS:
            (directory / (corruption + ".npy")).touch()

    def _imagenet_layout(self):
        directory = self.root / "corruption" / "ImageNet-C"
        directory.mkdir(parents=True)
        (directory / "classnames.txt").touch()
        for corruption in preflight.MAIN_CORRUPTIONS:
            (directory / corruption / "5" / "n01440764").mkdir(parents=True)

    def _small_cifar_arrays(self, *, labels=None, array=None):
        directory = self.root / "small-cifar"
        directory.mkdir()
        first_severity = np.array([0, 9], dtype=np.int64)
        np.save(
            directory / "labels.npy",
            np.tile(first_severity, 5) if labels is None else labels,
        )
        if array is None:
            array = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        np.save(directory / "fog.npy", array)
        return directory

    def _small_domainnet(self, class_names=("bird", "cat")):
        base = self.root / "small-domainnet"
        environments = ("clipart", "real")
        for environment in environments:
            for class_name in class_names:
                class_dir = base / environment / class_name
                class_dir.mkdir(parents=True)
                Image.new("RGB", (2, 2), color=(1, 2, 3)).save(class_dir / "sample.png")
        return base, environments

    def test_cifar_layout_is_valid_with_only_required_files(self):
        self._cifar10_layout()

        result = preflight.validate_dataset_layout(self.root, "CIFAR10C")

        self.assertTrue(result["valid"])
        self.assertEqual([], result["missing"])
        self.assertEqual("not_requested", result["deep"]["status"])

    def test_cifar_reports_missing_files_without_reading_arrays(self):
        self._cifar10_layout()
        missing_path = self.root / "corruption" / "CIFAR-10-C" / "fog.npy"
        missing_path.unlink()

        result = preflight.validate_dataset_layout(self.root, "CIFAR10C")

        self.assertFalse(result["valid"])
        self.assertIn(str(missing_path), [item["path"] for item in result["missing"]])

    def test_imagenet_requires_class_directories(self):
        self._imagenet_layout()
        empty_path = self.root / "corruption" / "ImageNet-C" / "fog" / "5" / "n01440764"
        empty_path.rmdir()

        result = preflight.validate_dataset_layout(self.root, "ImageNetC5K")

        self.assertFalse(result["valid"])
        self.assertIn(str(empty_path.parent), [item["path"] for item in result["missing"]])

    def test_domainbed_environment_requires_class_directories(self):
        base = self.root / "domainbed" / "domain_net"
        for environment in preflight.DOMAINBED_LAYOUTS["DomainNet"][1]:
            (base / environment / "class-a").mkdir(parents=True)

        result = preflight.validate_dataset_layout(self.root, "DomainNet")

        self.assertTrue(result["valid"])

    def test_cli_returns_nonzero_and_json_for_missing_layout(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = preflight.main(["--data-root", str(self.root), "--dataset", "CIFAR10C", "--json"])

        self.assertEqual(1, exit_code)
        self.assertIn('"valid": false', output.getvalue())

    def test_deep_cifar_accepts_exact_arrays_and_reads_edges(self):
        directory = self._small_cifar_arrays()

        details, errors = preflight._deep_check_cifar(
            directory, "CIFAR10C", ("fog",), expected_samples=10,
        )

        self.assertEqual([], errors)
        self.assertEqual([10, 32, 32, 3], details["arrays"]["fog"]["shape"])
        self.assertEqual(0, details["labels"]["min"])
        self.assertEqual(9, details["labels"]["max"])

    def test_deep_cifar_rejects_semantically_invalid_arrays(self):
        directory = self._small_cifar_arrays(
            labels=np.tile(np.array([0.0, 11.0], dtype=np.float32), 5),
            array=np.zeros((10, 32, 32, 3), dtype=np.int16),
        )

        _, errors = preflight._deep_check_cifar(
            directory, "CIFAR10C", ("fog",), expected_samples=10,
        )

        checks = {error["check"] for error in errors}
        self.assertIn("labels_dtype", checks)
        self.assertIn("corruption_dtype", checks)

        np.save(directory / "labels.npy", np.tile(np.array([0, 10], dtype=np.int64), 5))
        np.save(directory / "fog.npy", np.zeros((9, 32, 32, 3), dtype=np.uint8))
        _, errors = preflight._deep_check_cifar(
            directory, "CIFAR10C", ("fog",), expected_samples=10,
        )
        checks = {error["check"] for error in errors}
        self.assertIn("labels_range", checks)
        self.assertIn("corruption_shape", checks)

    def test_deep_cifar_rejects_inconsistent_severity_label_blocks(self):
        labels = np.tile(np.array([0, 9], dtype=np.int64), 5)
        labels[7] = 8
        directory = self._small_cifar_arrays(labels=labels)

        _, errors = preflight._deep_check_cifar(
            directory, "CIFAR10C", ("fog",), expected_samples=10,
        )

        self.assertIn(
            "labels_severity_consistency", {error["check"] for error in errors},
        )

    def test_deep_cifar_rejects_symlinked_root_and_artifacts(self):
        directory = self._small_cifar_arrays()
        labels_target = directory / "labels-target.npy"
        fog_target = directory / "fog-target.npy"
        (directory / "labels.npy").replace(labels_target)
        (directory / "fog.npy").replace(fog_target)
        (directory / "labels.npy").symlink_to(labels_target)
        (directory / "fog.npy").symlink_to(fog_target)

        _, errors = preflight._deep_check_cifar(
            directory, "CIFAR10C", ("fog",), expected_samples=10,
        )
        self.assertEqual(2, sum(error["check"] == "symlink" for error in errors))

        linked_root = self.root / "linked-cifar"
        linked_root.symlink_to(directory, target_is_directory=True)
        _, errors = preflight._deep_check_cifar(
            linked_root, "CIFAR10C", ("fog",), expected_samples=10,
        )
        self.assertEqual(["symlink"], [error["check"] for error in errors])

    def test_deep_domainnet_accepts_identical_nonempty_taxonomy(self):
        base, environments = self._small_domainnet()

        details, errors = preflight._deep_check_domainnet(
            base, environments, expected_class_count=2,
        )

        self.assertEqual([], errors)
        self.assertEqual({"clipart": 2, "real": 2}, details["environment_image_counts"])
        self.assertEqual(4, details["total_images"])

    def test_deep_domainnet_counts_nested_images_deterministically(self):
        base, environments = self._small_domainnet()
        for environment in environments:
            source = base / environment / "bird" / "sample.png"
            nested = base / environment / "bird" / "nested" / "deeper"
            nested.mkdir(parents=True)
            source.replace(nested / "sample.png")

        details, errors = preflight._deep_check_domainnet(
            base, environments, expected_class_count=2,
        )

        self.assertEqual([], errors)
        self.assertEqual({"clipart": 2, "real": 2}, details["environment_image_counts"])

    def test_deep_domainnet_rejects_nested_file_and_directory_symlinks(self):
        base, environments = self._small_domainnet()
        class_path = base / "clipart" / "bird"
        nested = class_path / "nested"
        nested.mkdir()
        (nested / "linked-file.png").symlink_to(class_path / "sample.png")
        (nested / "linked-directory").symlink_to(
            base / "real" / "bird", target_is_directory=True,
        )

        _, errors = preflight._deep_check_domainnet(
            base, environments, expected_class_count=2,
        )

        symlink_paths = {Path(error["path"]).name for error in errors if error["check"] == "symlink"}
        self.assertEqual({"linked-file.png", "linked-directory"}, symlink_paths)

    def test_deep_domainnet_includes_dot_classes_and_rejects_dot_symlinks(self):
        base, environments = self._small_domainnet()
        hidden_class = base / "clipart" / ".hidden"
        hidden_class.mkdir()
        Image.new("RGB", (2, 2)).save(hidden_class / "sample.png")
        hidden_link = base / "real" / "bird" / ".linked.png"
        hidden_link.symlink_to(base / "real" / "bird" / "sample.png")

        _, errors = preflight._deep_check_domainnet(
            base, environments, expected_class_count=2,
        )

        checks = {error["check"] for error in errors}
        self.assertIn("class_count", checks)
        self.assertIn("class_taxonomy", checks)
        self.assertIn("symlink", checks)

    def test_deep_domainnet_rejects_taxonomy_empty_class_and_symlink(self):
        base, environments = self._small_domainnet()
        (base / "real" / "cat" / "sample.png").unlink()
        target = base / "clipart" / "bird" / "sample.png"
        link = base / "clipart" / "cat" / "linked.png"
        (base / "clipart" / "cat" / "sample.png").unlink()
        link.symlink_to(target)

        _, errors = preflight._deep_check_domainnet(
            base, environments, expected_class_count=2,
        )

        checks = {error["check"] for error in errors}
        self.assertIn("class_nonempty", checks)
        self.assertIn("symlink", checks)

    def test_deep_api_and_cli_fail_on_semantic_invalidity(self):
        self._cifar10_layout()

        result = preflight.validate_dataset_layout(
            self.root, "CIFAR10C", deep=True,
        )
        self.assertFalse(result["valid"])
        self.assertEqual("failed", result["deep"]["status"])
        self.assertTrue(result["deep"]["errors"])
        self.assertEqual([], result["missing"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = preflight.main([
                "--data-root", str(self.root), "--dataset", "CIFAR10C", "--deep", "--json",
            ])
        self.assertEqual(1, exit_code)
        self.assertIn('"status": "failed"', output.getvalue())

    def test_runtime_facts_do_not_require_torch_import(self):
        facts = preflight.runtime_facts(self.root)

        self.assertIn("python", facts)
        self.assertIn("packages", facts)
        self.assertIn("git", facts)


if __name__ == "__main__":
    unittest.main()
