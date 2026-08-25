"""Download-free contracts for the non-benchmark CIFAR-100-C pilot builder."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build-cifar100c-pilot.py"
SPEC = importlib.util.spec_from_file_location("build_cifar100c_pilot", SCRIPT)
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class BuildCIFAR100CPilotTests(unittest.TestCase):
    def test_writes_loader_compatible_five_severity_arrays(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = np.arange(12 * 32 * 32 * 3, dtype=np.uint8).reshape(12, 32, 32, 3)
            labels = np.arange(12, dtype=np.int64)
            output = root / "corruption" / "CIFAR-100-C"

            pilot.build_pilot(images, labels, output, samples_per_severity=2, seed=9, source={"dataset": "test"})

            expected_shape = (10, 32, 32, 3)
            self.assertEqual(np.load(output / "labels.npy").shape, (10,))
            self.assertEqual(
                {path.stem for path in output.glob("*.npy")}, {"labels", *pilot.CORRUPTIONS}
            )
            for corruption in pilot.CORRUPTIONS:
                values = np.load(output / f"{corruption}.npy")
                self.assertEqual(values.shape, expected_shape)
                self.assertEqual(values.dtype, np.uint8)
            metadata = json.loads((output / pilot.README_NAME).read_text())
            self.assertIn("NOT canonical", metadata["benchmark_status"])
            self.assertEqual(metadata["parameters"]["samples_per_severity"], 2)

    def test_is_deterministic_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = np.full((10, 32, 32, 3), 127, dtype=np.uint8)
            labels = np.arange(10, dtype=np.int64)
            first, second = root / "first", root / "second"

            pilot.build_pilot(images, labels, first, samples_per_severity=1, seed=3, source={})
            pilot.build_pilot(images, labels, second, samples_per_severity=1, seed=3, source={})

            self.assertTrue(
                np.array_equal(np.load(first / "gaussian_noise.npy"), np.load(second / "gaussian_noise.npy"))
            )
            with self.assertRaises(FileExistsError):
                pilot.build_pilot(images, labels, first, samples_per_severity=1, seed=3, source={})
