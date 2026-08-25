"""Dependency-light contracts for the open-set CIFAR-100-C stream layer."""

import json
import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streams.builders import build_open_set_stream, verify_stream_fingerprint  # noqa: E402


_OPEN_SET_SPEC = importlib.util.spec_from_file_location("open_set", PROJECT_ROOT / "src" / "datasets" / "open_set.py")
_OPEN_SET = importlib.util.module_from_spec(_OPEN_SET_SPEC)
_OPEN_SET_SPEC.loader.exec_module(_OPEN_SET)
OpenSetDomainDataset = _OPEN_SET.OpenSetDomainDataset
load_cifar100_open_set_split = _OPEN_SET.load_cifar100_open_set_split


class _Domain:
    def __init__(self, labels):
        self.Y = list(labels)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return f"image-{index}", self.Y[index]


class _OpenSetDomains:
    def __init__(self, *, version="open-set-cifar100-split-v1", known=(0, 1, 2, 3), unknown=(4,)):
        # Each domain has eight ID and four OOD items, sufficient for the
        # exact 1/4 OOD selection used below.
        labels = [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 4, 4]
        self.datasets = (_Domain(labels), _Domain(labels))
        self.environments = ("noise", "blur")
        self.open_set_split_version = version
        self.known_class_ids = known
        self.unknown_class_ids = unknown


class OpenSetTests(unittest.TestCase):
    def test_versioned_split_is_a_complete_80_20_partition(self):
        split = load_cifar100_open_set_split()
        self.assertEqual("open-set-cifar100-split-v1", split["version"])
        self.assertEqual(80, len(split["known_class_ids"]))
        self.assertEqual(20, len(split["unknown_class_ids"]))
        self.assertEqual(set(range(100)), set(split["known_class_ids"]) | set(split["unknown_class_ids"]))

    def test_domain_wrapper_keeps_source_label_and_separates_metadata(self):
        source = _Domain([1, 4])
        domain = OpenSetDomainDataset(source, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(("image-1", 4), domain[1])
        self.assertEqual(
            {"original_label": 4, "known_label_or_minus_one": -1, "is_ood": True},
            domain.sample_metadata(1),
        )
        self.assertEqual(1, domain.sample_metadata(0)["known_label_or_minus_one"])

    def test_open_set_stream_is_deterministic_exact_and_metadata_isolated(self):
        datasets = _OpenSetDomains()
        first = build_open_set_stream(datasets, "block", 17, ood_ratio=0.25, block_size=2)
        second = build_open_set_stream(datasets, "block", 17, ood_ratio=0.25, block_size=2)
        self.assertEqual(first.references, second.references)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertTrue(verify_stream_fingerprint(first.to_dict()))
        self.assertEqual(16, len(first))
        self.assertEqual(4, first.metadata["open_set"]["realized_ood_count"])
        self.assertEqual(12, first.metadata["open_set"]["realized_known_count"])
        self.assertEqual(0.25, first.metadata["open_set"]["realized_ood_ratio"])
        self.assertEqual(
            [{"known": 6, "ood": 2, "total": 8}, {"known": 6, "ood": 2, "total": 8}],
            first.metadata["open_set"]["per_domain_counts"],
        )
        item = first[0]
        self.assertEqual(5, len(item))
        image, source_label, domain_idx, sample_idx, evaluator = item
        self.assertEqual(f"image-{sample_idx}", image)
        self.assertEqual(datasets.datasets[domain_idx].Y[sample_idx], source_label)
        self.assertEqual(source_label, evaluator["original_label"])
        self.assertEqual(source_label == 4, evaluator["is_ood"])
        self.assertEqual(-1 if source_label == 4 else source_label, evaluator["known_label_or_minus_one"])

    def test_fingerprint_binds_split_and_requested_ratio(self):
        datasets = _OpenSetDomains()
        baseline = build_open_set_stream(datasets, "iid_mixed", 7, ood_ratio=0.25)
        different_ratio = build_open_set_stream(datasets, "iid_mixed", 7, ood_ratio=0.5)
        different_split = build_open_set_stream(
            _OpenSetDomains(version="open-set-cifar100-split-v2"), "iid_mixed", 7, ood_ratio=0.25
        )
        self.assertNotEqual(baseline.fingerprint, different_ratio.fingerprint)
        self.assertNotEqual(baseline.fingerprint, different_split.fingerprint)
        payload = json.loads(json.dumps(baseline.to_dict()))
        payload["metadata"]["open_set"]["known_class_ids"] = [99]
        self.assertFalse(verify_stream_fingerprint(payload))

    def test_budget_matches_per_domain_exposure_across_ratios(self):
        datasets = _OpenSetDomains()
        budget = 8
        streams = [
            build_open_set_stream(datasets, "iid_mixed", 7, ood_ratio=ratio,
                                  per_domain_source_budget=budget)
            for ratio in (0.0, 0.25, 0.5)
        ]
        self.assertEqual([16, 16, 16], [len(stream) for stream in streams])
        for stream, ratio in zip(streams, (0.0, 0.25, 0.5)):
            open_set = stream.metadata["open_set"]
            self.assertEqual(budget, open_set["requested_per_domain_source_budget"])
            self.assertEqual([budget, budget], open_set["realized_per_domain_source_budget"])
            self.assertEqual(
                [{"known": int(budget * (1 - ratio)), "ood": int(budget * ratio), "total": budget}] * 2,
                open_set["selected_pool_per_domain_counts"],
            )
            self.assertEqual(open_set["selected_pool_per_domain_counts"], open_set["per_domain_counts"])
            self.assertEqual(ratio, open_set["realized_ood_ratio"])
            self.assertTrue(verify_stream_fingerprint(stream.to_dict()))

        repeated = build_open_set_stream(
            datasets, "iid_mixed", 7, ood_ratio=0.25, per_domain_source_budget=budget
        )
        self.assertEqual(streams[1].references, repeated.references)
        self.assertEqual(streams[1].fingerprint, repeated.fingerprint)
        self.assertNotEqual(streams[0].fingerprint, streams[1].fingerprint)

    def test_budget_rejects_non_integral_and_infeasible_allocations(self):
        datasets = _OpenSetDomains()
        with self.assertRaisesRegex(ValueError, "divisible"):
            build_open_set_stream(datasets, "iid_mixed", 7, ood_ratio=0.25,
                                  per_domain_source_budget=7)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            build_open_set_stream(datasets, "iid_mixed", 7, ood_ratio=0.5,
                                  per_domain_source_budget=10)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_open_set_stream(datasets, "iid_mixed", 7, ood_ratio=0.25,
                                  per_domain_source_budget=0)


if __name__ == "__main__":
    unittest.main()
