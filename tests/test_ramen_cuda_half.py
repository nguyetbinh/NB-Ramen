"""Real-device regressions for small FP16 CUDA support queries."""

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.Ramen import PriorityCache, cache_distances
from methods.OracleIDGradientRamen import OraclePriorityCache


@unittest.skipUnless(torch.cuda.is_available(), "requires a real CUDA device")
class CudaHalfCacheTests(unittest.TestCase):
    def test_single_query_returns_expected_half_precision_supports(self):
        cache = PriorityCache(3, 2, 1, "cuda", torch.float16)
        cache.add(
            torch.tensor([[0., 0.], [2., 0.], [5., 0.]], device="cuda"),
            torch.tensor([[7.], [11.], [13.]], device="cuda"),
            torch.tensor([.1, .2, .3], device="cuda"),
            torch.tensor([1., 2., 3.], device="cuda"),
        )
        values, priorities, entropies, distances = cache.query(
            torch.tensor([[.5, 0.]], device="cuda", dtype=torch.float16), topk=2,
        )
        self.assertEqual([[[7.], [11.]]], values.tolist())
        self.assertEqual([[1., 2.]], priorities.tolist())
        self.assertEqual([[.5, 1.5]], distances.tolist())
        for tensor in (cache.keys, cache.values, values, priorities, entropies, distances):
            self.assertEqual(torch.float16, tensor.dtype)

    def test_distance_queries_work_on_both_sides_of_dispatch_boundary(self):
        for query_count, support_count in ((1, 1), (1, 25), (25, 1), (25, 25), (1, 26), (26, 1), (100, 3)):
            with self.subTest(queries=query_count, supports=support_count):
                queries = torch.zeros((query_count, 2), device="cuda", dtype=torch.float16)
                supports = torch.tensor([[3., 4.]], device="cuda", dtype=torch.float16).repeat(support_count, 1)
                distances = cache_distances(queries, supports)
                self.assertEqual((query_count, support_count), tuple(distances.shape))
                self.assertTrue(torch.equal(distances, torch.full_like(distances, 5.)))
                self.assertEqual(torch.float16, distances.dtype)

    def test_primary_batch_distances_match_existing_torch_default(self):
        generator = torch.Generator(device="cuda").manual_seed(17)
        queries = torch.randn((100, 8), generator=generator, device="cuda", dtype=torch.float16)
        supports = torch.randn((7, 8), generator=generator, device="cuda", dtype=torch.float16)
        self.assertTrue(torch.equal(torch.cdist(queries, supports), cache_distances(queries, supports)))

    def test_oracle_single_query_preserves_ood_alignment(self):
        cache = OraclePriorityCache(3, 2, 1, "cuda", torch.float16)
        cache.add(
            torch.tensor([[0., 0.], [2., 0.], [5., 0.]], device="cuda"),
            torch.tensor([[7.], [11.], [13.]], device="cuda"),
            torch.zeros(3, device="cuda"), torch.arange(3, device="cuda"),
            torch.tensor([False, True, False], device="cuda"),
        )
        values, entropies, distances, is_ood = cache.query(
            torch.tensor([[.5, 0.]], device="cuda", dtype=torch.float16), topk=2,
        )
        self.assertEqual([[[7.], [11.]]], values.tolist())
        self.assertEqual([[.5, 1.5]], distances.tolist())
        self.assertEqual([[False, True]], is_ood.tolist())
        for tensor in (values, entropies, distances):
            self.assertEqual(torch.float16, tensor.dtype)


if __name__ == "__main__":
    unittest.main()
