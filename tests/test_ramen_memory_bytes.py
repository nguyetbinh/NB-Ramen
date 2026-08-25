"""Retained-support memory diagnostics for ordinary Ramen."""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.Ramen import PriorityCache, Ramen  # noqa: E402


class RamenMemoryBytesTests(unittest.TestCase):
    @staticmethod
    def _cache() -> PriorityCache:
        return PriorityCache(5, 3, 2, "cpu", torch.float32)

    @staticmethod
    def _expected_retained_bytes(cache: PriorityCache) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                cache.keys[:cache.size],
                cache.values[:cache.size],
                cache.priorities[:cache.size],
                cache.entropies[:cache.size],
            )
        )

    def test_empty_cache_has_no_retained_support_bytes(self):
        cache = self._cache()
        method = object.__new__(Ramen)
        method.cache = [cache]

        self.assertEqual(0, cache.retained_bytes)
        self.assertEqual(0, method.memory_bytes)

    def test_memory_bytes_counts_only_actual_retained_tensor_entries(self):
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

        self.assertEqual(self._expected_retained_bytes(first), first.retained_bytes)
        self.assertEqual(self._expected_retained_bytes(second), second.retained_bytes)
        self.assertEqual(
            self._expected_retained_bytes(first) + self._expected_retained_bytes(second),
            method.memory_bytes,
        )

        # Capacity is intentionally not claimed as retained method state.
        full_capacity_bytes = sum(
            tensor.numel() * tensor.element_size()
            for cache in method.cache
            for tensor in (cache.keys, cache.values, cache.priorities, cache.entropies)
        )
        self.assertLess(method.memory_bytes, full_capacity_bytes)

    def test_reset_removes_retained_bytes_without_reallocating_cache_storage(self):
        cache = self._cache()
        cache.add(torch.ones((1, 3)), torch.ones((1, 2)), torch.zeros(1), torch.zeros(1))
        self.assertGreater(cache.retained_bytes, 0)

        cache.reset()

        self.assertEqual(0, cache.retained_bytes)

    def test_forward_exposes_current_retained_bytes_as_diagnostic(self):
        class _Model:
            def featurize(self, x):
                return x

            def classify(self, features):
                return torch.stack((features[:, 0], -features[:, 0]), dim=1).requires_grad_()

            def get_by_sample_grad(self):
                return torch.tensor([[1., 2.], [3., 4.]])

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
        method.beta = 0.
        method.counter = 0
        method.cache = [self._cache(), self._cache()]
        method.num_classes = len(method.cache)
        method.model = _Model()
        method.loss_fn = lambda logits: logits.sum()
        method.last_diagnostics = {}

        method.forward(torch.tensor([[2., 0., 0.], [-1., 0., 0.]]))

        self.assertEqual(method.memory_bytes, method.get_diagnostics()["memory_bytes"])
        self.assertGreater(method.get_diagnostics()["memory_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
