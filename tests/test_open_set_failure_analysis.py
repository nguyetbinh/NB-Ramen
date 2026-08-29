import unittest

from src.evaluation.failure_mode_analysis import (
    _open_set_reconstruction,
    open_set_gradient_analysis,
)


OPEN = {
    "original_label": 7, "is_ood": False, "known_label_or_minus_one": 7,
    "split_version": "open-set-cifar100-split-v1", "ood_ratio": 0.3,
}
OOD = {**OPEN, "original_label": 91, "is_ood": True, "known_label_or_minus_one": -1}


class _Reader:
    def tensor(self, descriptor, *, kind):
        class _Vector:
            def __init__(self, value): self.value = value
            def reshape(self, _): return self
            def tolist(self): return self.value
        return _Vector(descriptor["values"])


class OpenSetFailureAnalysisTests(unittest.TestCase):
    def test_reconstructs_production_class_balanced_all_and_id_aggregates(self):
        query = {
            "segment_index": 0, "open_set": OPEN,
            "retrieved_support_ids": [[10, 11], [12, -1]],
            "retrieved_weights": [[.25, .75], [2.0, 0.0]],
            "support_valid_mask": [[True, True], [True, False]],
        }
        items = {
            (0, 10): {"open_set": OPEN, "gradient": {"values": [2., 0.]}},
            (0, 11): {"open_set": OOD, "gradient": {"values": [0., 2.]}},
            (0, 12): {"open_set": OPEN, "gradient": {"values": [2., 2.]}},
        }
        trace = {"open_set": OPEN, "support_item_ids": query["retrieved_support_ids"],
                 "support_weights": query["retrieved_weights"], "support_valid_mask": query["support_valid_mask"]}
        rebuilt = _open_set_reconstruction(query, items, trace, _Reader())
        self.assertEqual([2.25, 2.75], rebuilt["all_gradient"])
        self.assertEqual([2.25, 2.0], rebuilt["id_gradient"])
        self.assertEqual(.3333333333333333, rebuilt["retrieved_ood_count_fraction"])
        self.assertEqual(.25, rebuilt["retrieved_ood_weight_fraction"])
        report = open_set_gradient_analysis([{"open_set": OPEN, "open_set_reconstruction": rebuilt, "outcome": "harmful"}])
        self.assertEqual("computed", report["status"])
        self.assertEqual(1, report["harmful_id"]["count"])

    def test_zero_id_support_is_insufficient_and_tampered_labels_fail_closed(self):
        query = {
            "segment_index": 0, "open_set": OPEN,
            "retrieved_support_ids": [[11]], "retrieved_weights": [[1.]],
            "support_valid_mask": [[True]],
        }
        items = {(0, 11): {"open_set": OOD, "gradient": {"values": [1., -1.]}}}
        trace = {"open_set": OPEN, "support_item_ids": query["retrieved_support_ids"],
                 "support_weights": query["retrieved_weights"], "support_valid_mask": query["support_valid_mask"]}
        rebuilt = _open_set_reconstruction(query, items, trace, _Reader())
        self.assertTrue(rebuilt["zero_id_support"])
        self.assertEqual("insufficient", open_set_gradient_analysis([
            {"open_set": OPEN, "open_set_reconstruction": rebuilt}
        ])["status"])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            _open_set_reconstruction({**query, "open_set": {**OPEN, "ood_ratio": .4}}, items, trace, _Reader())
