"""Focused entrypoint helpers for deterministic evaluation-prefix runs."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main import _prefix_segments, _sync_device, args_parser


class EvaluationBudgetEntrypointTests(unittest.TestCase):
    def test_prefix_clips_single_domain_reset_segments(self):
        self.assertEqual(
            ((0, 6), (6, 7)),
            _prefix_segments(((0, 6), (6, 12), (12, 18)), 7),
        )

    def test_sync_device_synchronizes_mps_when_available(self):
        with patch("main.torch") as mocked_torch:
            mocked_torch.backends.mps.is_available.return_value = True

            _sync_device(SimpleNamespace(type="mps"))

            mocked_torch.mps.synchronize.assert_called_once_with()

    def test_sync_device_uses_backend_availability_api(self):
        with patch("main.torch") as mocked_torch:
            del mocked_torch.mps.is_available
            mocked_torch.backends.mps.is_available.return_value = True

            _sync_device(SimpleNamespace(type="mps"))

            mocked_torch.backends.mps.is_available.assert_called_once_with()
            mocked_torch.mps.synchronize.assert_called_once_with()

    def test_sync_device_preserves_cuda_synchronization(self):
        device = SimpleNamespace(type="cuda")
        with patch("main.torch") as mocked_torch:
            _sync_device(device)

            mocked_torch.cuda.synchronize.assert_called_once_with(device)

    def test_matrix_hyphenated_budget_alias_reaches_main_parser(self):
        with patch("sys.argv", ["main.py", "--device", "cpu", "--max-eval-samples", "7"]):
            args = args_parser()
        self.assertEqual(7, args.max_eval_samples)

    def test_legacy_mixed_order_is_explicitly_scoped_to_mixed_iid(self):
        with patch("sys.argv", ["main.py", "--device", "cpu", "--legacy_mixed_order"]):
            args = args_parser()
        self.assertTrue(args.legacy_mixed_order)

        for incompatible in (
            ["--tta_mode", "single"],
            ["--stream_mode", "block"],
        ):
            with self.subTest(incompatible=incompatible), patch(
                "sys.argv",
                ["main.py", "--device", "cpu", "--legacy_mixed_order", *incompatible],
            ):
                with self.assertRaises(SystemExit):
                    args_parser()


if __name__ == "__main__":
    unittest.main()
