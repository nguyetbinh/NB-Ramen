import unittest

from src.evaluation.open_set_metrics import (
    binary_auroc,
    fpr_at_95_tpr,
    h_score,
    id_accuracy,
    open_set_metrics,
    open_set_metrics_from_trace_rows,
)


class OpenSetMetricsTests(unittest.TestCase):
    def test_perfect_scores_and_id_only_accuracy(self):
        metrics = open_set_metrics(
            predictions=[0, 9, 1, 9],
            ground_truth_classes=[0, 7, 1, 8],
            is_ood=[False, True, False, True],
            ood_scores=[0.1, 0.9, 0.2, 0.8],
        )
        self.assertEqual(1.0, metrics.id_accuracy)
        self.assertEqual(1.0, metrics.auroc)
        self.assertEqual(0.0, metrics.fpr_at_95_tpr)
        self.assertEqual(0.8, metrics.fpr95_threshold)
        self.assertEqual(1.0, metrics.ood_recall_at_fpr95)
        self.assertEqual(1.0, metrics.h_score)

    def test_reversed_scores_have_zero_auroc_and_full_fpr95(self):
        flags = [False, True, False, True]
        scores = [0.9, 0.1, 0.8, 0.2]
        self.assertEqual(0.0, binary_auroc(flags, scores))
        self.assertEqual(1.0, fpr_at_95_tpr(flags, scores))

    def test_tied_scores_are_order_independent_and_receive_half_auroc_credit(self):
        flags = [False, True, False, True]
        scores = [0.5, 0.5, 0.5, 0.5]
        self.assertEqual(0.5, binary_auroc(flags, scores))
        self.assertEqual(1.0, fpr_at_95_tpr(flags, scores))
        metrics = open_set_metrics([0, 9, 3, 9], [0, 1, 2, 3], flags, scores)
        self.assertEqual(0.5, metrics.id_accuracy)
        self.assertEqual(1.0, metrics.ood_recall_at_fpr95)
        self.assertEqual(2 / 3, metrics.h_score)

    def test_no_id_or_no_ood_is_undefined_for_detection(self):
        with self.assertRaises(ValueError):
            binary_auroc([False, False], [0.1, 0.2])
        with self.assertRaises(ValueError):
            fpr_at_95_tpr([True, True], [0.1, 0.2])
        with self.assertRaises(ValueError):
            id_accuracy([0], [0], [True])

    def test_trace_builder_checks_known_split_and_inputs(self):
        rows = [
            {"ground_truth_class": 0, "prediction": 0, "is_ood": False, "ood_score": 0.1},
            {"ground_truth_class": 5, "prediction": 0, "is_ood": True, "ood_score": 0.9},
        ]
        self.assertEqual(1.0, open_set_metrics_from_trace_rows(rows, known_class_ids=[0, 1]).id_accuracy)
        with self.assertRaises(ValueError):
            open_set_metrics_from_trace_rows(rows, known_class_ids=[])
        with self.assertRaises(ValueError):
            open_set_metrics_from_trace_rows(
                [{"ground_truth_class": 0, "prediction": 0, "is_ood": True, "ood_score": 0.1}],
                known_class_ids=[0],
            )
        with self.assertRaises(TypeError):
            open_set_metrics([0, 1], [0, 1], [False, 1], [0.1, 0.9])
        with self.assertRaises(ValueError):
            open_set_metrics([0, 1], [0, 1], [False, True], [0.1, float("nan")])

    def test_h_score_validates_range(self):
        self.assertEqual(0.0, h_score(0.0, 0.0))
        with self.assertRaises(ValueError):
            h_score(1.1, 0.5)


if __name__ == "__main__":
    unittest.main()
