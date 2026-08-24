"""Empirical parity checks for the historical PyTorch mixed-order path."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from src.streams import truncate_stream, verify_stream_fingerprint
from src.streams.legacy import build_legacy_torch_iid_stream
from main import args_parser, ordered_stream_test


class _IndexDataset(Dataset):
    def __init__(self, domain, size):
        self.domain = domain
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor(index), index


class _TaggedFlatDataset(Dataset):
    def __init__(self, domains):
        self.references = [
            (domain_idx, sample_idx)
            for domain_idx, domain in enumerate(domains)
            for sample_idx in range(len(domain))
        ]

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        return torch.tensor(self.references[index])


class _MultiDataset:
    def __init__(self):
        self.datasets = (
            _IndexDataset("a", 4),
            _IndexDataset("b", 3),
            _IndexDataset("c", 5),
        )
        self.environments = ("a", "b", "c")

    def __iter__(self):
        return iter(self.datasets)


class _EvaluationDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.tensor([float(index)]), index % 2


class _EvaluationMultiDataset:
    datasets = (_EvaluationDataset(),)
    environments = ("only",)

    def __iter__(self):
        return iter(self.datasets)

    def __len__(self):
        return len(self.datasets)


class _StochasticMethod:
    def __init__(self):
        self.first_forward_draw = None

    def __call__(self, image):
        if self.first_forward_draw is None:
            self.first_forward_draw = torch.rand(4)
        return torch.zeros((len(image), 2))

    def reset(self):
        pass


def _consume_constructor_rng(draw_count):
    if draw_count:
        torch.rand(draw_count)


def _actual_historical_run(multi, seed, constructor_draws):
    """Run the actual old DataLoader path and capture its next global draw."""
    torch.manual_seed(seed)
    _consume_constructor_rng(constructor_draws)
    loader = DataLoader(
        _TaggedFlatDataset(multi.datasets),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=None,
    )
    references = tuple(
        tuple(int(value) for value in row)
        for batch in loader
        for row in batch.tolist()
    )
    return references, torch.rand(5)


class LegacyTorchStreamTests(unittest.TestCase):
    def test_matches_actual_dataloader_order_and_rng_after_constructor(self):
        multi = _MultiDataset()
        for constructor_draws in (0, 7):
            with self.subTest(constructor_draws=constructor_draws):
                expected_references, expected_first_forward_draw = _actual_historical_run(
                    multi, seed=17, constructor_draws=constructor_draws
                )

                torch.manual_seed(17)
                _consume_constructor_rng(constructor_draws)
                stream = build_legacy_torch_iid_stream(multi, 17)
                actual_first_forward_draw = torch.rand(5)

                self.assertEqual(expected_references, stream.references)
                self.assertTrue(
                    torch.equal(expected_first_forward_draw, actual_first_forward_draw)
                )
                self.assertTrue(verify_stream_fingerprint(stream.to_dict()))

    def test_constructor_rng_can_change_legacy_fingerprint(self):
        multi = _MultiDataset()
        torch.manual_seed(31)
        without_constructor_rng = build_legacy_torch_iid_stream(multi, 31)

        torch.manual_seed(31)
        _consume_constructor_rng(7)
        with_constructor_rng = build_legacy_torch_iid_stream(multi, 31)

        self.assertNotEqual(
            without_constructor_rng.references, with_constructor_rng.references
        )
        self.assertNotEqual(
            without_constructor_rng.fingerprint, with_constructor_rng.fingerprint
        )
        parameters = with_constructor_rng.metadata["parameters"]
        self.assertEqual(
            "process_global_state_after_method_construction",
            parameters["generator_source"],
        )
        self.assertEqual(2, parameters["global_rng_draws_consumed"])
        self.assertIn("fingerprint", parameters["pairing_contract"])

    def test_truncation_is_an_exact_prefix_without_extra_rng_consumption(self):
        multi = _MultiDataset()
        expected_references, expected_next_draw = _actual_historical_run(
            multi, seed=23, constructor_draws=4
        )

        torch.manual_seed(23)
        _consume_constructor_rng(4)
        full_stream = build_legacy_torch_iid_stream(multi, 23)
        prefix = truncate_stream(full_stream, 5)
        actual_next_draw = torch.rand(5)

        self.assertEqual(expected_references[:5], prefix.references)
        self.assertEqual(full_stream.fingerprint, prefix.metadata["evaluation_budget"]["full_stream_fingerprint"])
        self.assertTrue(torch.equal(expected_next_draw, actual_next_draw))

    def test_ordered_loader_preserves_historical_first_forward_rng(self):
        datasets = _EvaluationMultiDataset()
        seed = 123
        constructor_draws = 6

        torch.manual_seed(seed)
        _consume_constructor_rng(constructor_draws)
        historical_loader = DataLoader(
            _TaggedFlatDataset(datasets.datasets),
            batch_size=2,
            shuffle=True,
            num_workers=0,
            generator=None,
        )
        next(iter(historical_loader))
        expected_first_forward_draw = torch.rand(4)

        torch.manual_seed(seed)
        _consume_constructor_rng(constructor_draws)
        stream = build_legacy_torch_iid_stream(datasets, seed)
        method = _StochasticMethod()
        args = SimpleNamespace(
            run_id="legacy-rng-boundary",
            device=torch.device("cpu"),
            batch_size=2,
            num_workers=0,
            metric_window_size=2,
            metric_window_stride=1,
            stream_mode="iid_mixed",
            reference_trace=None,
            legacy_mixed_order=True,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            ordered_stream_test(
                datasets,
                method,
                args,
                {
                    "run_dir": run_dir,
                    "stream": run_dir / "stream.json",
                    "trace": run_dir / "trace.jsonl",
                    "summary": run_dir / "summary.json",
                },
                stream,
            )

        self.assertTrue(torch.equal(expected_first_forward_draw, method.first_forward_draw))

    def test_seed_validation_and_legacy_cli_identity(self):
        with self.assertRaisesRegex(TypeError, "seed"):
            build_legacy_torch_iid_stream(_MultiDataset(), True)

        with patch(
            "sys.argv",
            [
                "main.py", "--device", "cpu", "--legacy_mixed_order",
                "--seed", "7", "--stream_seed", "7",
            ],
        ):
            args = args_parser()
        self.assertIn("legacy-torch-iid-replay-s7-", args.run_id)

        with patch(
            "sys.argv",
            ["main.py", "--device", "cpu", "--max_eval_samples", "3", "--stream_block_size", "12"],
        ):
            args = args_parser()
        self.assertIn("-blk-12-", args.run_id)

        with patch(
            "sys.argv", ["main.py", "--device", "cpu", "--stream_block_size", "12"],
        ), self.assertRaises(SystemExit):
            args_parser()

        with patch(
            "sys.argv",
            [
                "main.py", "--device", "cpu", "--legacy_mixed_order",
                "--seed", "7", "--stream_seed", "8",
            ],
        ), self.assertRaises(SystemExit):
            args_parser()

        with patch(
            "sys.argv",
            [
                "main.py", "--device", "cpu", "--legacy_mixed_order",
                "--num_workers", "1",
            ],
        ), self.assertRaises(SystemExit):
            args_parser()

        with patch(
            "sys.argv",
            [
                "main.py", "--device", "cpu", "--legacy_mixed_order",
                "--reference_trace", "baseline/trace.jsonl",
            ],
        ), self.assertRaises(SystemExit):
            args_parser()


if __name__ == "__main__":
    unittest.main()
