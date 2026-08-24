import unittest

from src.evaluation.online_metrics import (
    average_accuracy, domain_accuracies, domain_shift_recovery_times, negative_adaptation_rate,
    post_shift_recovery_time, sliding_window_accuracy, worst_domain_accuracy,
)


class OnlineMetricsTests(unittest.TestCase):
    def test_average_and_worst_domain_accuracy(self):
        rows = [
            {"ground_truth_domain": "a", "correct": True},
            {"ground_truth_domain": "a", "correct": False},
            {"ground_truth_domain": "b", "correct": False},
        ]
        self.assertEqual(1 / 3, average_accuracy(rows))
        self.assertEqual({"a": 0.5, "b": 0.0}, domain_accuracies(rows))
        self.assertEqual(0.0, worst_domain_accuracy(rows))

    def test_sliding_window_defaults_to_overlapping_windows(self):
        rows = [{"timestep": i, "correct": value} for i, value in enumerate([True, False, True, True])]
        windows = sliding_window_accuracy(rows, window_size=2)
        self.assertEqual([0.5, 0.5, 1.0], [window.accuracy for window in windows])
        self.assertEqual((0, 1), (windows[0].start_timestep, windows[0].end_timestep))

    def test_post_shift_recovery_uses_pre_shift_baseline(self):
        rows = [{"timestep": i, "correct": value} for i, value in enumerate(
            [True, True, False, False, True, True, True]
        )]
        self.assertEqual(2, post_shift_recovery_time(rows, shift_timestep=2, window_size=2))

    def test_negative_adaptation_rate_compares_full_windows(self):
        adapted = [True, False, False, False, True, True]
        reference = [True, True, False, False, False, True]
        self.assertEqual(0.5, negative_adaptation_rate(adapted, reference, window_size=3))

    def test_domain_shift_recovery_stays_within_each_episode(self):
        correctness = [True, True, False, False, True, True, True, True]
        domains = [0, 0, 1, 1, 1, 1, 1, 1]
        results = domain_shift_recovery_times(correctness, domains, window_size=2)
        self.assertEqual("recovered", results[0]["status"])
        self.assertEqual(2, results[0]["recovery_samples"])

    def test_short_shift_episode_is_explicitly_insufficient(self):
        results = domain_shift_recovery_times(
            [True, True, False, True], [0, 0, 1, 1], window_size=3
        )
        self.assertEqual("insufficient_episode", results[0]["status"])

    def test_empty_accuracy_and_unaligned_streams_fail(self):
        with self.assertRaises(ValueError):
            average_accuracy([])
        with self.assertRaises(ValueError):
            negative_adaptation_rate([True], [True, False], window_size=1)


if __name__ == "__main__":
    unittest.main()
