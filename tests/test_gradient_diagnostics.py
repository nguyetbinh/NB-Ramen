import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.gradient_diagnostics import (
    class_balanced_production_aggregate,
    normalized_class_local_gradients,
    pairwise_class_gradient_summaries,
    summarize_class_balanced_gradients,
    trace_payload_from_retrieval,
)
from memory.structured_memory import StructuredGradientMemory


class GradientDiagnosticsTests(unittest.TestCase):
    def test_production_aggregate_and_local_normalisation_are_distinct(self):
        gradients = torch.tensor([[[[2.0], [6.0]], [[10.0], [0.0]]]])
        entropies = torch.zeros(1, 2, 2)
        distances = torch.tensor([[[0.0, 1.0], [0.0, float("inf")]]])
        valid = torch.tensor([[[True, True], [True, False]]])
        aggregate, active_count, weights = class_balanced_production_aggregate(
            gradients, entropies, distances, valid, beta=1.0
        )
        expected_class_zero = 2.0 + 6.0 * torch.exp(torch.tensor(-1.0))
        self.assertTrue(torch.allclose(aggregate.squeeze(), (expected_class_zero + 10.0) / 2))
        self.assertEqual([2], active_count.tolist())
        h, active = normalized_class_local_gradients(gradients, weights, valid)
        self.assertTrue(torch.allclose(h[0, 0, 0], expected_class_zero / (1 + torch.exp(torch.tensor(-1.0)))))
        self.assertEqual([[True, True]], active.tolist())

    def test_consensus_and_pairwise_singleton_and_empty_masks(self):
        gradients = torch.tensor([[[[1.0, 1.0]], [[1.0, -1.0]]], [[[3.0, 4.0]], [[0.0, 0.0]]]])
        entropies = torch.zeros(2, 2, 1)
        distances = torch.zeros(2, 2, 1)
        valid = torch.tensor([[[True], [True]], [[True], [False]]])
        summary = summarize_class_balanced_gradients(gradients, entropies, distances, valid, beta=0.0)
        self.assertTrue(torch.allclose(summary["consensus_mean"][0], torch.tensor(.5)))
        self.assertTrue(torch.allclose(summary["pairwise_cosine_mean"][0], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(summary["pairwise_sign_agreement_mean"][0], torch.tensor(.5)))
        self.assertEqual([1, 0], summary["pairwise_class_gradient_count"].tolist())
        self.assertTrue(torch.isnan(summary["pairwise_cosine_mean"][1]))
        self.assertTrue(torch.isnan(summary["pairwise_sign_agreement_mean"][1]))

    def test_trace_support_classes_align_with_padded_class_axis(self):
        memory = StructuredGradientMemory(2, 2, 1, 1, device="cpu")
        memory.add(torch.tensor([[0.]]), torch.tensor([[1.]]), 1, 0, 0.)
        retrieval = memory.query(torch.tensor([[0.]]), 0, 2)
        payload = trace_payload_from_retrieval(retrieval, 0.)
        self.assertEqual([[[-1, -1], [1, -1]]], payload["support_predicted_classes"].tolist())

    def test_flat_trace_handles_empty_single_and_multiclass_supports(self):
        memory = StructuredGradientMemory(3, 3, 1, 1, device="cpu")
        empty = memory.query_flat(torch.tensor([[0.]]), 2, selection="global_nearest")
        self.assertEqual([0], trace_payload_from_retrieval(empty, 0.)["support_count"].tolist())
        memory.add(torch.tensor([[0.], [1.], [2.]]), torch.tensor([[1.], [2.], [3.]]),
                   torch.tensor([0, 1, 1]), torch.tensor([0, 0, 0]), torch.zeros(3))
        single = memory.query_flat(torch.tensor([[0.]]), 1, selection="global_nearest")
        self.assertEqual([[0]], trace_payload_from_retrieval(single, 0.)["support_predicted_classes"].tolist())
        multi = memory.query_flat(torch.tensor([[0.]]), 3, selection="global_nearest")
        payload = trace_payload_from_retrieval(multi, 0.)
        self.assertEqual([[0, 1, 1]], payload["support_predicted_classes"].tolist())
        self.assertEqual([2], payload["active_support_classes"].tolist())


if __name__ == "__main__":
    unittest.main()
