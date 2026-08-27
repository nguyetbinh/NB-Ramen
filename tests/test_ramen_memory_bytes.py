"""Retained-support memory diagnostics for Legacy Ramen."""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.Ramen import PriorityCache, Ramen  # noqa: E402


class RamenMemoryBytesTests(unittest.TestCase):
    @staticmethod
    def _cache():
        return PriorityCache(5, 3, 2, "cpu", torch.float32)

    @staticmethod
    def _retained_bytes(cache):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                cache.keys[:cache.size],
                cache.values[:cache.size],
                cache.priorities[:cache.size],
                cache.entropies[:cache.size],
            )
        )

    def test_counts_only_current_entries_not_eager_capacity(self):
        first, second = self._cache(), self._cache()
        first.add(
            torch.ones((2, 3)), torch.ones((2, 2)),
            torch.zeros(2), torch.tensor([0.0, 1.0]),
        )
        second.add(
            torch.ones((1, 3)), torch.ones((1, 2)),
            torch.zeros(1), torch.tensor([0.0]),
        )
        method = object.__new__(Ramen)
        method.cache = [first, second]

        expected = self._retained_bytes(first) + self._retained_bytes(second)
        capacity_bytes = sum(
            tensor.numel() * tensor.element_size()
            for cache in method.cache
            for tensor in (cache.keys, cache.values, cache.priorities, cache.entropies)
        )
        self.assertEqual(expected, method.memory_bytes)
        self.assertLess(method.memory_bytes, capacity_bytes)

    def test_forward_reports_batch_final_retained_state_and_reset_clears_it(self):
        class _Model:
            def featurize(self, x):
                return x

            def classify(self, features):
                return torch.stack((features[:, 0], -features[:, 0]), dim=1).requires_grad_()

            def get_by_sample_grad(self):
                return torch.tensor([[1.0, 2.0], [3.0, 4.0]])

            def set_by_sample_grad(self, gradients):
                self.gradients = gradients

            def step_and_zero_grad(self):
                pass

            def __call__(self, x):
                return torch.zeros((x.shape[0], 2))

            def reset_parameters(self):
                pass

        method = object.__new__(Ramen)
        method.device = torch.device("cpu")
        method.dtype = torch.float32
        method.cfg = {"topk": 1}
        method.beta = 0.0
        method.counter = 0
        method.cache = [self._cache(), self._cache()]
        method.num_classes = len(method.cache)
        method.model = _Model()
        method.loss_fn = lambda logits: logits.sum()
        method.last_diagnostics = {}

        method.forward(torch.tensor([[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

        self.assertEqual(method.memory_bytes, method.get_diagnostics()["memory_bytes"])
        self.assertEqual(2, sum(cache.size for cache in method.cache))

        method.reset()

        self.assertEqual(0, method.memory_bytes)
        self.assertEqual({}, method.get_diagnostics())


if __name__ == "__main__":
    unittest.main()
