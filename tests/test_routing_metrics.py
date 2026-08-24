import unittest

from src.evaluation.routing_metrics import (
    adjusted_rand_index,
    assignment_churn_rate,
    context_purity,
    normalized_mutual_information,
    number_of_discovered_contexts,
    routing_diagnostics,
)


class RoutingMetricsTests(unittest.TestCase):
    def test_perfect_and_permuted_assignments_are_perfect(self):
        domains = ["a", "a", "b", "b"]
        for contexts in ([0, 0, 1, 1], [7, 7, 3, 3]):
            self.assertEqual(1.0, normalized_mutual_information(domains, contexts))
            self.assertEqual(1.0, adjusted_rand_index(domains, contexts))
            self.assertEqual(1.0, context_purity(domains, contexts))

    def test_collapsed_assignment_has_zero_information_and_majority_purity(self):
        domains = ["a", "a", "b", "b"]
        contexts = [0, 0, 0, 0]
        self.assertEqual(0.0, normalized_mutual_information(domains, contexts))
        self.assertEqual(0.0, adjusted_rand_index(domains, contexts))
        self.assertEqual(0.5, context_purity(domains, contexts))
        self.assertEqual(1, number_of_discovered_contexts(contexts))

    def test_alternating_assignments_have_maximum_adjacent_churn(self):
        contexts = [0, 1, 0, 1, 0]
        self.assertEqual(1.0, assignment_churn_rate(contexts))
        self.assertEqual(2, number_of_discovered_contexts(contexts))
        self.assertEqual(0.0, assignment_churn_rate([0]))

    def test_diagnostics_reports_unavailable_for_missing_contexts(self):
        self.assertEqual("unavailable", routing_diagnostics([0, 1]).status)
        self.assertEqual("unavailable", routing_diagnostics([0, 1], [None, None]).status)
        result = routing_diagnostics([0, 0, 1, 1], [4, 4, 9, 9])
        self.assertEqual("available", result.status)
        self.assertEqual(1.0, result.normalized_mutual_information)
        self.assertEqual(2, result.number_of_discovered_contexts)

    def test_validation_and_metric_ranges(self):
        with self.assertRaises(ValueError):
            normalized_mutual_information([], [])
        with self.assertRaises(ValueError):
            adjusted_rand_index([0], [0, 1])
        with self.assertRaises(TypeError):
            context_purity([["unhashable"]], [0])
        with self.assertRaises(ValueError):
            routing_diagnostics([0, 1], [0, None])
        nmi = normalized_mutual_information([0, 0, 1, 1], [0, 1, 0, 1])
        ari = adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertGreaterEqual(nmi, 0.0)
        self.assertLessEqual(nmi, 1.0)
        self.assertGreaterEqual(ari, -1.0)
        self.assertLessEqual(ari, 1.0)


if __name__ == "__main__":
    unittest.main()
