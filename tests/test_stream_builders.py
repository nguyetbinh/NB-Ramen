"""Standalone tests for stream construction (no torch or numpy required)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streams import (
    build_single_domain_stream,
    build_stream,
    truncate_stream,
    verify_stream_fingerprint,
)


class FakeDataset:
    def __init__(self, domain, labels):
        self.domain = domain
        self.targets = list(labels)
        self.reads = 0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.reads += 1
        return ("image-%s-%s" % (self.domain, index), self.targets[index])


class FakeMultiDataset:
    def __init__(self):
        self.datasets = (
            FakeDataset("a", [0, 1, 0, 1, 2, 2]),
            FakeDataset("b", [0, 1, 0, 1, 2, 2]),
            FakeDataset("c", [0, 1, 0, 1, 2, 2]),
        )
        self.environments = ("a", "b", "c")


class StreamBuilderTests(unittest.TestCase):
    def setUp(self):
        self.multi = FakeMultiDataset()

    def test_all_modes_are_deterministic_and_lazy(self):
        modes = (
            "iid_mixed", "block", "gradual", "recurring", "imbalanced",
            "novel_domain", "class_domain_correlated", "bursty",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                first = build_stream(self.multi, mode, seed=17, block_size=2)
                second = build_stream(self.multi, mode, seed=17, block_size=2)
                self.assertEqual(first.references, second.references)
                self.assertEqual(first.fingerprint, second.fingerprint)
                self.assertEqual(sum(dataset.reads for dataset in self.multi.datasets), 0)
                if mode == "imbalanced":
                    self.assertLess(len(first), 18)
                    counts = [sum(domain == index for domain, _ in first.references) for index in range(3)]
                    self.assertGreater(counts[0], counts[1])
                    self.assertGreater(counts[1], counts[2])
                else:
                    self.assertEqual(len(first), 18)
                    self.assertEqual(sorted(first.references),
                                     [(domain, sample) for domain in range(3) for sample in range(6)])
                self.assertEqual(json.loads(json.dumps(first.to_dict()))["fingerprint"], first.fingerprint)
                self.assertTrue(verify_stream_fingerprint(first.to_dict()))

    def test_schedule_metadata_and_output_keep_domain_for_evaluation(self):
        stream = build_stream(self.multi, "block", seed=1, block_size=3)
        image, label, domain_idx, sample_idx = stream[0]
        self.assertEqual(image, "image-%s-%s" % (self.multi.environments[domain_idx], sample_idx))
        self.assertEqual(label, self.multi.datasets[domain_idx].targets[sample_idx])
        self.assertEqual(stream.metadata["domain_names"], ["a", "b", "c"])
        self.assertEqual(stream.metadata["num_samples"], 18)
        self.assertEqual(sum(dataset.reads for dataset in self.multi.datasets), 1)

    def test_single_domain_stream_is_deterministic_and_segmented(self):
        first, segments = build_single_domain_stream(self.multi, seed=7)
        second, second_segments = build_single_domain_stream(self.multi, seed=7)
        self.assertEqual(first.references, second.references)
        self.assertEqual(segments, second_segments)
        self.assertEqual(((0, 6), (6, 12), (12, 18)), segments)
        for domain_idx, (start, stop) in enumerate(segments):
            self.assertEqual({domain_idx}, {domain for domain, _ in first.references[start:stop]})
        self.assertTrue(first.metadata["parameters"]["reset_at_domain_boundaries"])

    def test_novel_domain_is_not_seen_before_halfway(self):
        stream = build_stream(self.multi, "novel_domain", seed=3, novel_domain_idx=2)
        self.assertNotIn(2, [domain for domain, _ in stream.references[:len(stream) // 2]])
        self.assertEqual(len(stream) // 2, stream.metadata["parameters"]["novel_release_timestep"])

    def test_novel_domain_rejects_an_infeasible_release_boundary(self):
        multi = FakeMultiDataset()
        multi.datasets = (
            FakeDataset("old", [0]),
            FakeDataset("novel", [0, 0, 0, 0, 0]),
        )
        with self.assertRaisesRegex(ValueError, "cannot be held"):
            build_stream(multi, "novel_domain", seed=0, novel_domain_idx=1)

    def test_imbalanced_budget_and_weights_control_selection(self):
        stream = build_stream(self.multi, "imbalanced", seed=5,
                              domain_weights=[100, 1, 1], sample_budget=8)
        counts = [sum(domain == index for domain, _ in stream.references) for index in range(3)]
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[0], counts[2])
        self.assertEqual(len(stream), 8)

    def test_imbalanced_default_creates_long_tail_domain_counts(self):
        stream = build_stream(self.multi, "imbalanced", seed=5)
        counts = [sum(domain == index for domain, _ in stream.references) for index in range(3)]
        self.assertEqual([6, 3, 2], counts)

    def test_cost_limited_prefix_is_deterministic_and_records_provenance(self):
        full = build_stream(self.multi, "block", seed=5, block_size=2)
        first = truncate_stream(full, 7)
        second = truncate_stream(full, 7)

        self.assertEqual(first.references, full.references[:7])
        self.assertEqual(first.references, second.references)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, full.fingerprint)
        self.assertTrue(verify_stream_fingerprint(first.to_dict()))
        self.assertEqual(0, sum(dataset.reads for dataset in self.multi.datasets))
        self.assertEqual({
            "truncation_strategy": "deterministic_prefix",
            "evidence_scope": "cost_limited",
            "cost_limited_evidence": True,
            "full_sample_count": 18,
            "full_stream_fingerprint": full.fingerprint,
            "retained_sample_count": 7,
            "dropped_sample_count": 11,
        }, first.metadata["evaluation_budget"])

    def test_cost_limited_prefix_validates_its_budget(self):
        stream = build_stream(self.multi, "block", seed=0)
        for value in (0, -1, len(stream) + 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_samples"):
                    truncate_stream(stream, value)
        with self.assertRaisesRegex(TypeError, "max_samples"):
            truncate_stream(stream, 1.5)

    def test_custom_weights_change_iid_block_and_recurring_orders(self):
        for mode in ("iid_mixed", "block", "recurring"):
            with self.subTest(mode=mode):
                equal = build_stream(self.multi, mode, seed=11, block_size=2,
                                     domain_weights=[1, 1, 1])
                weighted = build_stream(self.multi, mode, seed=11, block_size=2,
                                        domain_weights=[100, 1, 1])
                self.assertNotEqual(equal.references, weighted.references)

    def test_cifar_style_Y_metadata_supports_correlated_stream(self):
        class CIFARStyleDataset:
            def __init__(self, labels):
                self.Y = list(labels)

            def __len__(self):
                return len(self.Y)

            def __getitem__(self, index):
                return "image", self.Y[index]

        multi = FakeMultiDataset()
        multi.datasets = tuple(CIFARStyleDataset([0, 1, 0, 1, 2, 2]) for _ in range(3))
        stream = build_stream(multi, "class_domain_correlated", seed=2)
        self.assertEqual(18, len(stream))

    def test_clear_validation_and_non_materializing_label_requirement(self):
        with self.assertRaisesRegex(ValueError, "mode must"):
            build_stream(self.multi, "unknown", seed=0)
        with self.assertRaisesRegex(ValueError, "domain_weights"):
            build_stream(self.multi, "block", seed=0, domain_weights=[1, 2])
        with self.assertRaisesRegex(ValueError, "sample_budget"):
            build_stream(self.multi, "block", seed=0, sample_budget=2)
        class NoMetadata:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise AssertionError("schedule construction must not read samples")
        broken = FakeMultiDataset()
        broken.datasets = (NoMetadata(), FakeDataset("y", [0]), FakeDataset("z", [0]))
        with self.assertRaisesRegex(ValueError, "non-materializing label metadata"):
            build_stream(broken, "class_domain_correlated", seed=0)


if __name__ == "__main__":
    unittest.main()
