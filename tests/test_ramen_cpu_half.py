"""CPU regression coverage for half-precision Ramen retrieval."""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.Ramen import PriorityCache  # noqa: E402


class PriorityCacheCpuHalfTests(unittest.TestCase):
    def test_query_matches_half_rounded_float32_cdist_ranking_and_metadata(self):
        cache = PriorityCache(4, 2, 1, "cpu", torch.float16)
        cache.add(
            torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [5.0, 5.0]]),
            torch.tensor([[7.0], [11.0], [13.0], [17.0]]),
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        query = torch.tensor([[0.75, 0.25]], dtype=torch.float16)

        values, priorities, entropies, distances = cache.query(query, topk=3)

        expected_distances, expected_indices = torch.topk(
            torch.cdist(query.float(), cache.keys[:cache.size].float()).half(),
            k=3,
            dim=1,
            largest=False,
            sorted=True,
        )
        self.assertEqual(torch.float16, cache.keys.dtype)
        self.assertEqual(torch.float16, cache.values.dtype)
        self.assertTrue(torch.equal(values, cache.values[expected_indices]))
        self.assertTrue(torch.equal(priorities, cache.priorities[expected_indices]))
        self.assertTrue(torch.equal(entropies, cache.entropies[expected_indices]))
        self.assertTrue(torch.equal(distances, expected_distances))

    def test_ramen_forward_completes_with_cpu_half_cache_and_emits_diagnostics(self):
        class _Model:
            def featurize(self, x):
                return x

            def classify(self, features):
                return torch.stack((features[:, 0], -features[:, 0]), dim=1).requires_grad_()

            def get_by_sample_grad(self):
                return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)

            def set_by_sample_grad(self, gradients):
                self.gradients = gradients

            def step_and_zero_grad(self):
                pass

            def __call__(self, x):
                return torch.zeros((x.shape[0], 2), dtype=torch.float16)

            def reset_parameters(self):
                pass

        from methods.Ramen import Ramen

        method = object.__new__(Ramen)
        method.device = torch.device("cpu")
        method.dtype = torch.float16
        method.cfg = {"topk": 1}
        method.beta = 0.5
        method.counter = 0
        method.cache = [
            PriorityCache(3, 2, 2, "cpu", torch.float16),
            PriorityCache(3, 2, 2, "cpu", torch.float16),
        ]
        method.num_classes = len(method.cache)
        method.model = _Model()
        method.loss_fn = lambda logits: logits.sum()
        method.last_diagnostics = {}

        logits = method.forward(
            torch.tensor([[2.0, 0.0], [-1.0, 0.0]], dtype=torch.float16)
        )

        self.assertEqual((2, 2), tuple(logits.shape))
        self.assertEqual(torch.float16, method.model.gradients.dtype)
        self.assertEqual(2, sum(cache.size for cache in method.cache))
        self.assertEqual(method.memory_bytes, method.get_diagnostics()["memory_bytes"])


if __name__ == "__main__":
    unittest.main()
