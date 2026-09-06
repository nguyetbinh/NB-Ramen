import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.LegacyLatentRamen import (
    ContextPriorityCache,
    LegacyLatentRamen,
    update_and_retrieve_batch_atomic,
    validate_legacy_latent_ramen_config,
)
from methods.Ramen import PriorityCache
from routing.online_prototypes import OnlinePrototypeRouter


class LegacyLatentRamenTests(unittest.TestCase):
    def _cache(self, capacity=8):
        return ContextPriorityCache(capacity, 1, 1, "cpu", torch.float32)

    def test_batch_admits_every_item_before_issuing_any_query(self):
        class RecordingCache(ContextPriorityCache):
            def __init__(self):
                super().__init__(8, 1, 1, "cpu", torch.float32)
                self.added = 0
                self.added_when_queried = []

            def add(self, *args, **kwargs):
                self.added += 1
                return super().add(*args, **kwargs)

            def query(self, *args, **kwargs):
                self.added_when_queried.append(self.added)
                return super().query(*args, **kwargs)

        cache = RecordingCache()
        retrieved = update_and_retrieve_batch_atomic(
            [cache], torch.tensor([[0.], [10.]]), torch.tensor([[1.], [2.]]),
            torch.tensor([0, 0]), torch.tensor([0, 1]), torch.zeros(2), torch.tensor([0., 1.]),
            topk=1, beta=0.,
        )
        self.assertEqual([2], cache.added_when_queried)
        self.assertEqual([[1.], [2.]], retrieved.tolist())

    def test_context_filtering_never_returns_cross_context_support(self):
        cache = self._cache()
        cache.add(torch.tensor([[0.], [10.]]), torch.tensor([[3.], [7.]]), torch.zeros(2),
                  torch.tensor([0., 1.]), torch.tensor([0, 1]))
        values, _, _, valid = cache.query(torch.tensor([[0.], [10.]]), torch.tensor([0, 1]), topk=1)
        self.assertTrue(valid.all())
        self.assertEqual([[3.], [7.]], values[:, 0].tolist())
        no_match = cache.query(torch.tensor([[0.]]), torch.tensor([2]), topk=1)
        self.assertFalse(no_match[3].any())

    def test_context_filtering_keeps_legacy_scalar_class_denominator(self):
        caches = [self._cache(), self._cache()]
        result = update_and_retrieve_batch_atomic(
            caches,
            torch.tensor([[0.], [10.]]),
            torch.tensor([[4.], [8.]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.zeros(2),
            torch.tensor([0., 1.]),
            topk=1,
            beta=0.,
        )
        # Both class caches are nonempty, as in legacy Ramen.  Each query gets
        # support from only its own context, so the other class contributes
        # zero but remains in the original scalar denominator of two.
        self.assertEqual([[2.], [4.]], result.tolist())

    def test_single_context_retrieval_matches_priority_cache(self):
        keys = torch.tensor([[0.], [2.], [4.]])
        values = torch.tensor([[10.], [20.], [30.]])
        entropies = torch.tensor([.2, .4, .6])
        priorities = torch.tensor([1., 2., 3.])
        queries = torch.tensor([[1.], [3.]])
        baseline = PriorityCache(4, 1, 1, "cpu", torch.float32)
        contextual = self._cache(4)
        baseline.add(keys, values, entropies, priorities)
        contextual.add(keys, values, entropies, priorities, torch.zeros(3, dtype=torch.long))
        expected_values, _, expected_entropies, expected_distances = baseline.query(queries, topk=2)
        original_query = PriorityCache.query
        calls = []

        def record_parent_query(cache, *args, **kwargs):
            calls.append(cache)
            return original_query(cache, *args, **kwargs)

        with patch.object(PriorityCache, "query", new=record_parent_query):
            actual_values, actual_entropies, actual_distances, valid = contextual.query(
                queries, torch.zeros(2, dtype=torch.long), topk=2,
            )
        self.assertEqual([contextual], calls)
        self.assertTrue(valid.all())
        self.assertTrue(torch.equal(expected_values, actual_values))
        self.assertTrue(torch.equal(expected_entropies, actual_entropies))
        self.assertTrue(torch.equal(expected_distances, actual_distances))

    def test_single_context_aggregation_matches_legacy_ramen(self):
        features = torch.tensor([[0.], [1.], [4.]])
        gradients = torch.tensor([[2.], [4.], [8.]])
        predicted_classes = torch.tensor([0, 1, 0])
        entropies = torch.tensor([.2, .5, .7])
        priorities = torch.tensor([0., 1., 2.])
        beta = .25
        expected_caches = [PriorityCache(4, 1, 1, "cpu", torch.float32) for _ in range(2)]
        for index in range(features.shape[0]):
            expected_caches[int(predicted_classes[index])].add(
                features[index:index + 1], gradients[index:index + 1], entropies[index:index + 1],
                priorities[index:index + 1],
            )
        expected = torch.zeros_like(gradients)
        for cache in expected_caches:
            values, _, support_entropies, distances = cache.query(features, topk=2)
            weights = torch.exp(-support_entropies) * torch.exp(-beta * distances)
            expected += (values * weights.unsqueeze(-1)).sum(dim=1)
        expected /= len(expected_caches)

        actual = update_and_retrieve_batch_atomic(
            [self._cache(4), self._cache(4)], features, gradients, predicted_classes,
            torch.zeros(3, dtype=torch.long), entropies, priorities, topk=2, beta=beta,
        )
        self.assertTrue(torch.equal(expected, actual))

    def test_reset_clears_router_cache_and_stream_counter(self):
        class Model:
            def __init__(self):
                self.reset_calls = 0

            def reset_parameters(self):
                self.reset_calls += 1

        method = object.__new__(LegacyLatentRamen)
        method.model = Model()
        method.router = OnlinePrototypeRouter(spawn_threshold=.1, max_contexts=4, temperature=.1)
        method.router.route(torch.tensor([[1.]]), update=True)
        method.cache = [self._cache()]
        method.cache[0].add(torch.tensor([[1.]]), torch.tensor([[1.]]), torch.tensor([0.]),
                            torch.tensor([0.]), torch.tensor([0]))
        method.counter = 9
        method.reset()
        self.assertEqual(1, method.model.reset_calls)
        self.assertEqual(0, method.router.num_contexts)
        self.assertEqual(0, method.cache[0].size)
        self.assertEqual(0, method.counter)
        self.assertEqual(0, method.get_diagnostics()["memory_size"])
        self.assertEqual(0, method.get_diagnostics()["num_active_contexts"])
        self.assertIsNone(method.get_diagnostics()["inferred_context"])
        self.assertNotIn("memory_bytes", method.get_diagnostics())

    def test_diagnostics_preserve_per_sample_routing_and_batch_atomic_memory_size(self):
        method = object.__new__(LegacyLatentRamen)
        method.router = OnlinePrototypeRouter(spawn_threshold=.1, max_contexts=4, temperature=.1)
        method.cache = [self._cache(), self._cache()]
        routing = method.router.route(torch.tensor([[1.], [-1.]]), update=True)
        method.cache[0].add(torch.tensor([[1.]]), torch.tensor([[1.]]), torch.tensor([0.]),
                            torch.tensor([0.]), torch.tensor([0]))
        method.cache[1].add(torch.tensor([[-1.]]), torch.tensor([[2.]]), torch.tensor([0.]),
                            torch.tensor([1.]), torch.tensor([1]))
        memory_sizes = torch.full((2,), sum(cache.size for cache in method.cache), dtype=torch.long)
        active_contexts = routing.spawned.long().cumsum(0)
        method.last_diagnostics = method._diagnostics(routing, active_contexts, memory_sizes)
        diagnostics = method.get_diagnostics()
        self.assertEqual(routing.context_ids.tolist(), diagnostics["inferred_context"].tolist())
        self.assertEqual(routing.spawned.tolist(), diagnostics["spawned"].tolist())
        self.assertEqual([2, 2], diagnostics["memory_size"].tolist())
        self.assertEqual([1, 2], diagnostics["num_active_contexts"].tolist())

    def test_configs_are_valid_and_keep_legacy_capacity_surface(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "cfg/CIFAR10C/LegacyLatentRamen.yaml",
            "cfg/CIFAR100C/LegacyLatentRamen.yaml",
            "cfg/DomainNet/LegacyLatentRamen.yaml",
            "cfg/ImageNetC5K/LegacyLatentRamen.yaml",
            "cfg/smoke/CIFAR100C/LegacyLatentRamen.yaml",
        ):
            config = validate_legacy_latent_ramen_config(yaml.safe_load((root / relative).read_text()))
            self.assertNotIn("capacity_scope", config)
            self.assertEqual("signsgd", config["optimizer"])

    def test_benchmark_configs_differ_from_ramen_only_by_router_fields(self):
        root = Path(__file__).resolve().parents[1]
        router_fields = {"spawn_threshold", "max_contexts", "router_temperature", "router_momentum"}
        for dataset in ("CIFAR10C", "CIFAR100C", "DomainNet", "ImageNetC5K"):
            ramen = yaml.safe_load((root / "cfg" / dataset / "Ramen.yaml").read_text())
            latent = yaml.safe_load((root / "cfg" / dataset / "LatentRamen.yaml").read_text())
            legacy_latent = yaml.safe_load((root / "cfg" / dataset / "LegacyLatentRamen.yaml").read_text())
            self.assertEqual(ramen, {key: value for key, value in legacy_latent.items() if key not in router_fields})
            self.assertEqual(
                {key: latent[key] for key in router_fields},
                {key: legacy_latent[key] for key in router_fields},
            )


if __name__ == "__main__":
    unittest.main()
