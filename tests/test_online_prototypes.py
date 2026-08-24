import unittest

try:
    import torch
except ImportError:  # pragma: no cover - the whole module is intentionally skipped
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for online prototype tests")
class OnlinePrototypeRouterTests(unittest.TestCase):
    def setUp(self):
        from src.routing.online_prototypes import OnlinePrototypeRouter

        self.Router = OnlinePrototypeRouter

    def test_spawns_then_reuses_nearby_context(self):
        router = self.Router(spawn_threshold=0.2, max_contexts=3)
        result = router.route(torch.tensor([[1.0, 0.0], [0.99, 0.01]]))

        self.assertEqual([0, 0], result.assignments.tolist())
        self.assertEqual([True, False], result.spawned.tolist())
        self.assertEqual(1, router.num_contexts)
        self.assertEqual([2], router.context_counts.tolist())
        self.assertEqual(2, router.total_samples)

    def test_far_feature_spawns_until_context_limit(self):
        router = self.Router(spawn_threshold=0.1, max_contexts=2)
        result = router.route(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))

        self.assertEqual([0, 1, 1], result.assignments.tolist())
        self.assertEqual([True, True, False], result.spawned.tolist())
        self.assertEqual(2, router.num_contexts)
        self.assertEqual(2, router.num_spawns)

    def test_running_mean_update_is_normalized(self):
        router = self.Router(spawn_threshold=1.0)
        router.route(torch.tensor([[1.0, 0.0]]))
        router.route(torch.tensor([[0.0, 1.0]]))

        expected = torch.tensor([1.0, 1.0]) / (2**0.5)
        self.assertTrue(torch.allclose(router.prototypes[0], expected))
        self.assertTrue(torch.allclose(router.prototypes.norm(dim=1), torch.ones(1)))

    def test_momentum_update_is_supported(self):
        router = self.Router(spawn_threshold=1.0, momentum=0.75)
        router.route(torch.tensor([[1.0, 0.0]]))
        router.route(torch.tensor([[0.0, 1.0]]))

        expected = torch.tensor([0.75, 0.25])
        expected = expected / expected.norm()
        self.assertTrue(torch.allclose(router.prototypes[0], expected))

    def test_posteriors_are_soft_and_sum_to_one(self):
        router = self.Router(spawn_threshold=0.1, temperature=0.2)
        router.route(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        posterior = router.posterior(torch.tensor([[0.8, 0.2]]))

        self.assertEqual((1, 2), tuple(posterior.shape))
        self.assertTrue(torch.allclose(posterior.sum(dim=1), torch.ones(1)))
        self.assertGreater(posterior[0, 0], posterior[0, 1])

    def test_reset_clears_state_and_counters(self):
        router = self.Router()
        router.route(torch.tensor([[1.0, 0.0]]))
        router.reset()

        self.assertEqual(0, router.num_contexts)
        self.assertEqual(0, router.total_samples)
        self.assertEqual(0, router.num_spawns)
        self.assertEqual([], router.context_counts.tolist())

    def test_batch_is_processed_in_order(self):
        first = self.Router(spawn_threshold=0.1)
        result = first.route(torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]))
        second = self.Router(spawn_threshold=0.1)
        expected = [second.route(row.unsqueeze(0)).assignments.item() for row in torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])]

        self.assertEqual(expected, result.assignments.tolist())
        self.assertEqual([2, 1], first.context_counts.tolist())

    def test_dtype_is_preserved_and_state_moves_safely(self):
        router = self.Router()
        result = router.route(torch.tensor([[1.0, 0.0]], dtype=torch.float64))
        self.assertEqual(torch.float64, router.prototypes.dtype)
        self.assertEqual(torch.float64, result.posteriors.dtype)


if __name__ == "__main__":
    unittest.main()
