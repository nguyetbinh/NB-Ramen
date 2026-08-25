"""CPU regression coverage for half-precision Ramen retrieval."""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.Ramen import PriorityCache  # noqa: E402


class PriorityCacheCpuHalfTests(unittest.TestCase):
    def test_cpu_half_query_preserves_float_distance_ranking(self):
        cache = PriorityCache(3, 1, 1, "cpu", torch.float16)
        cache.add(
            torch.tensor([[0.0], [2.0], [5.0]]),
            torch.tensor([[7.0], [11.0], [13.0]]),
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([1.0, 2.0, 3.0]),
        )
        query = torch.tensor([[0.6]], dtype=torch.float16)

        values, priorities, entropies, distances = cache.query(query, topk=2)

        expected_distances, expected_indices = torch.topk(
            torch.cdist(query.float(), cache.keys[:cache.size].float()),
            k=2,
            dim=1,
            largest=False,
            sorted=True,
        )
        self.assertTrue(torch.equal(values, cache.values[expected_indices]))
        self.assertTrue(torch.equal(priorities, cache.priorities[expected_indices]))
        self.assertTrue(torch.equal(entropies, cache.entropies[expected_indices]))
        self.assertTrue(torch.equal(distances, expected_distances))


if __name__ == "__main__":
    unittest.main()
