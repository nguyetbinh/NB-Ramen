"""Dependency-light contracts for the DomainNet semantic open-set protocol."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streams.builders import build_open_set_stream, verify_stream_fingerprint  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("open_set", PROJECT_ROOT / "src" / "datasets" / "open_set.py")
_OPEN_SET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OPEN_SET)
OpenSetDomainDataset = _OPEN_SET.OpenSetDomainDataset
load_domainnet_open_set_split = _OPEN_SET.load_domainnet_open_set_split
materialize_domainnet_open_set_split = _OPEN_SET.materialize_domainnet_open_set_split


class _ToyImageFolder:
    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return f"image-{index}", self.targets[index]


class _ToyOpenSetDomainNet:
    def __init__(self, *, split_fingerprint="split-fingerprint", taxonomy_sha256="taxonomy-sha256"):
        # Six known and four unknown examples per domain permit an exact 25%
        # OOD source pool without reading an image.
        labels = [0, 1, 2, 0, 1, 2, 3, 3, 4, 4]
        self.datasets = (_ToyImageFolder(labels), _ToyImageFolder(labels))
        self.environments = ("clipart", "real")
        self.open_set_split_version = "open-set-domainnet-name-rank-v1"
        self.open_set_split_fingerprint = split_fingerprint
        self.open_set_taxonomy_sha256 = taxonomy_sha256
        self.known_class_ids = (0, 1, 2)
        self.unknown_class_ids = (3, 4)


class DomainNetOpenSetTests(unittest.TestCase):
    def _toy_split_path(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "toy-domainnet-split.json"
        path.write_text(
            json.dumps(
                {
                    "version": "toy-domainnet-name-rank-v1",
                    "dataset": "DomainNet",
                    "expected_class_count": 5,
                    "known_class_count": 3,
                    "unknown_class_count": 2,
                    "selection": {
                        "strategy": "sha256-name-rank-v1",
                        "salt": "toy split salt",
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_repository_recipe_is_a_pinned_345_class_protocol(self):
        split = load_domainnet_open_set_split()
        self.assertEqual("open-set-domainnet-name-rank-v1", split["version"])
        self.assertEqual("DomainNet", split["dataset"])
        self.assertEqual(345, split["expected_class_count"])
        self.assertEqual(276, split["known_class_count"])
        self.assertEqual(69, split["unknown_class_count"])
        self.assertEqual("sha256-name-rank-v1", split["selection"]["strategy"])

    def test_materialized_partition_is_deterministic_complete_and_name_based(self):
        path = self._toy_split_path()
        names = ("zebra", "airplane", "bicycle", "cat", "dog")
        first = materialize_domainnet_open_set_split(names, path)
        second = materialize_domainnet_open_set_split(tuple(reversed(names)), path)

        self.assertEqual(first["known_class_names"], second["known_class_names"])
        self.assertEqual(first["unknown_class_names"], second["unknown_class_names"])
        self.assertEqual(first["taxonomy_sha256"], second["taxonomy_sha256"])
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(set(names), set(first["known_class_names"]) | set(first["unknown_class_names"]))
        self.assertFalse(set(first["known_class_names"]) & set(first["unknown_class_names"]))
        self.assertEqual(3, len(first["known_class_ids"]))
        self.assertEqual(2, len(first["unknown_class_ids"]))

    def test_materialization_rejects_mismatched_or_ambiguous_taxonomy(self):
        path = self._toy_split_path()
        with self.assertRaisesRegex(ValueError, "size"):
            materialize_domainnet_open_set_split(("a", "b"), path)
        with self.assertRaisesRegex(ValueError, "unique"):
            materialize_domainnet_open_set_split(("a", "b", "b", "c", "d"), path)

    def test_wrapped_imagefolder_exposes_evaluator_metadata_only(self):
        source = _ToyImageFolder([4, 1])
        wrapped = OpenSetDomainDataset(source, {1: 0, 3: 1})
        self.assertEqual(("image-0", 4), wrapped[0])
        self.assertEqual(
            {"original_label": 4, "known_label_or_minus_one": -1, "is_ood": True},
            wrapped.sample_metadata(0),
        )
        self.assertEqual(
            {"original_label": 1, "known_label_or_minus_one": 0, "is_ood": False},
            wrapped.sample_metadata(1),
        )

    def test_stream_binds_dataset_split_identifiers_into_its_fingerprint(self):
        baseline = build_open_set_stream(_ToyOpenSetDomainNet(), "block", 7, ood_ratio=0.25)
        changed_taxonomy = build_open_set_stream(
            _ToyOpenSetDomainNet(taxonomy_sha256="changed-taxonomy"), "block", 7, ood_ratio=0.25
        )
        open_set = baseline.metadata["open_set"]
        self.assertEqual("split-fingerprint", open_set["split_fingerprint"])
        self.assertEqual("taxonomy-sha256", open_set["taxonomy_sha256"])
        self.assertTrue(verify_stream_fingerprint(baseline.to_dict()))
        self.assertNotEqual(baseline.fingerprint, changed_taxonomy.fingerprint)
        item = baseline[0]
        self.assertEqual(5, len(item))
        self.assertEqual(
            {"original_label", "known_label_or_minus_one", "is_ood"},
            set(item[-1]),
        )

    def test_entrypoint_accepts_domainnet_protocol_and_selects_its_wrapper(self):
        # ``main`` is imported only under the pinned project environment,
        # which supplies torch and the optional model dependencies.
        from main import OpenSetDomainNet, _open_set_dataset_class, args_parser

        with patch(
            "sys.argv",
            [
                "main.py", "--device", "cpu", "--dataset", "DomainNet", "--open_set",
                "--known_class_split", "open-set-domainnet-name-rank-v1",
            ],
        ):
            args = args_parser()
        self.assertEqual("DomainNet", args.dataset)
        self.assertEqual("open-set-domainnet-name-rank-v1", args.known_class_split)
        self.assertTrue(args.known_class_split_path.endswith("open-set-domainnet-split-v1.json"))
        self.assertIs(OpenSetDomainNet, _open_set_dataset_class(args.dataset, args.known_class_split))


if __name__ == "__main__":
    unittest.main()
