"""Dependency-light contracts for the open-set CIFAR-100-C stream layer."""

import json
import importlib.util
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streams.builders import build_open_set_stream, verify_stream_fingerprint  # noqa: E402


_OPEN_SET_SPEC = importlib.util.spec_from_file_location("open_set", PROJECT_ROOT / "src" / "datasets" / "open_set.py")
_OPEN_SET = importlib.util.module_from_spec(_OPEN_SET_SPEC)
_OPEN_SET_SPEC.loader.exec_module(_OPEN_SET)
OpenSetDomainDataset = _OPEN_SET.OpenSetDomainDataset
OpenSetCIFAR100C = _OPEN_SET.OpenSetCIFAR100C
load_cifar100_open_set_split = _OPEN_SET.load_cifar100_open_set_split
CIFAR100_CLASS_NAMES = _OPEN_SET.CIFAR100_CLASS_NAMES


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
        self.assertNotIn("fingerprint", split)
        self.assertNotIn("taxonomy_sha256", split)

    def test_name_ranked_robustness_splits_recompute_from_frozen_recipe(self):
        split_dir = PROJECT_ROOT / "cfg" / "research"
        loaded = [
            load_cifar100_open_set_split(split_dir / f"open-set-cifar100-split-v{version}.json")
            for version in (2, 3)
        ]
        self.assertEqual(
            ["open-set-cifar100-name-rank-v2", "open-set-cifar100-name-rank-v3"],
            [split["version"] for split in loaded],
        )
        for split in loaded:
            # Deliberately repeat the published UTF-8 digest recipe here
            # rather than relying on the loader's implementation.
            salt = split["version"]
            ranked = sorted(
                range(100),
                key=lambda class_id: (
                    hashlib.sha256(
                        f"{salt}\0{CIFAR100_CLASS_NAMES[class_id]}".encode("utf-8")
                    ).hexdigest(),
                    CIFAR100_CLASS_NAMES[class_id],
                ),
            )
            self.assertEqual(tuple(ranked[:80]), split["known_class_ids"])
            self.assertEqual(tuple(ranked[80:]), split["unknown_class_ids"])
            self.assertEqual(80, len(set(split["known_class_ids"])))
            self.assertEqual(20, len(set(split["unknown_class_ids"])))
            self.assertFalse(set(split["known_class_ids"]) & set(split["unknown_class_ids"]))
            self.assertEqual(set(range(100)), set(split["known_class_ids"]) | set(split["unknown_class_ids"]))
            self.assertRegex(split["fingerprint"], r"^[0-9a-f]{64}$")
            self.assertRegex(split["taxonomy_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(loaded[0]["known_class_ids"], loaded[1]["known_class_ids"])

    def test_verified_name_ranked_identifiers_are_bound_into_stream_fingerprint(self):
        split_path = PROJECT_ROOT / "cfg" / "research" / "open-set-cifar100-split-v2.json"

        def initialize_without_images(dataset, *args, **kwargs):
            dataset.classes = list(CIFAR100_CLASS_NAMES)
            dataset.environments = ("corruption",)
            dataset.datasets = [_Domain(range(100))]

        with patch.object(_OPEN_SET.CIFAR100C, "__init__", initialize_without_images):
            dataset = OpenSetCIFAR100C("unused", split_path=split_path)
        split = load_cifar100_open_set_split(split_path)
        self.assertEqual(split["fingerprint"], dataset.open_set_split_fingerprint)
        self.assertEqual(split["taxonomy_sha256"], dataset.open_set_taxonomy_sha256)

        stream = build_open_set_stream(dataset, "iid_mixed", 7, ood_ratio=0.2)
        open_set = stream.metadata["open_set"]
        self.assertEqual(split["fingerprint"], open_set["split_fingerprint"])
        self.assertEqual(split["taxonomy_sha256"], open_set["taxonomy_sha256"])
        self.assertTrue(verify_stream_fingerprint(stream.to_dict()))

        tampered = json.loads(json.dumps(stream.to_dict()))
        tampered["metadata"]["open_set"]["taxonomy_sha256"] = "0" * 64
        self.assertFalse(verify_stream_fingerprint(tampered))

    def test_name_ranked_split_rejects_recipe_or_fingerprint_tampering(self):
        source = PROJECT_ROOT / "cfg" / "research" / "open-set-cifar100-split-v2.json"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["known_class_ids"][0], payload["known_class_ids"][1] = (
                payload["known_class_ids"][1], payload["known_class_ids"][0]
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_cifar100_open_set_split(path)

            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["fingerprint"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_cifar100_open_set_split(path)

    def test_split_byte_lock_rejects_semantically_identical_reformatting(self):
        source = PROJECT_ROOT / "cfg" / "research" / "open-set-cifar100-split-v2.json"
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        locked = load_cifar100_open_set_split(source, expected_sha256=expected)
        self.assertEqual("open-set-cifar100-name-rank-v2", locked["version"])
        with tempfile.TemporaryDirectory() as directory:
            reformatted = Path(directory) / "split.json"
            # Same JSON value, different bytes: a preregistered launch must
            # reject even whitespace-only drift.
            reformatted.write_text(
                json.dumps(json.loads(source.read_text(encoding="utf-8")), sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "planned bytes"):
                load_cifar100_open_set_split(reformatted, expected_sha256=expected)

    def test_open_set_wrapper_passes_split_byte_lock_to_loader(self):
        split_path = PROJECT_ROOT / "cfg" / "research" / "open-set-cifar100-split-v3.json"
        expected = hashlib.sha256(split_path.read_bytes()).hexdigest()

        def initialize_without_images(dataset, *args, **kwargs):
            dataset.classes = list(CIFAR100_CLASS_NAMES)
            dataset.environments = ("corruption",)
            dataset.datasets = [_Domain(range(100))]

        with patch.object(_OPEN_SET.CIFAR100C, "__init__", initialize_without_images):
            dataset = OpenSetCIFAR100C(
                "unused", split_path=split_path, split_sha256=expected,
            )
        self.assertEqual("open-set-cifar100-name-rank-v3", dataset.open_set_split_version)

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
