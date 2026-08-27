import sys
import unittest
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory.structured_memory import StructuredGradientMemory
from methods.SupportAblations import (
    CausalRamen,
    ContextOnlyRamen,
    GlobalNearestRamen,
    RandomMemoryRamen,
    SameClassRamen,
    StructuredAtomicRamen,
    aggregate_unbalanced_gradients,
    causal_active_context_counts,
    update_and_retrieve_support_atomic_batch,
    update_and_retrieve_support_batch,
    update_and_retrieve_support_causal_batch,
    validate_support_ablation_config,
)


class SupportAblationTests(unittest.TestCase):
    def make_memory(self):
        memory = StructuredGradientMemory(3, 10, 1, 1, device="cpu", capacity_scope="per_class")
        memory.add(
            torch.tensor([[8.], [2.], [1.], [9.]]), torch.tensor([[80.], [20.], [10.], [90.]]),
            torch.tensor([0, 1, 0, 2]), torch.tensor([1, 1, 2, 2]), torch.zeros(4),
            item_ids=torch.tensor([8, 2, 1, 9]),
        )
        return memory

    def test_selection_rules_have_exact_topk_pools(self):
        memory = self.make_memory()
        query = torch.tensor([[0.]])
        same_class = memory.query_flat(query, 2, selection="same_class", predicted_classes=0)
        global_nearest = memory.query_flat(query, 2, selection="global_nearest")
        context = memory.query_flat(query, 2, selection="context_nearest", contexts=2)
        self.assertEqual([1, 8], same_class.item_ids[0, same_class.valid_mask[0]].tolist())
        self.assertEqual([1, 2], global_nearest.item_ids[0, global_nearest.valid_mask[0]].tolist())
        self.assertEqual([1, 9], context.item_ids[0, context.valid_mask[0]].tolist())

    def test_random_selection_is_seeded_without_replacement(self):
        memory = self.make_memory()
        kwargs = dict(selection="random", current_item_ids=17, random_seed=123)
        first = memory.query_flat(torch.tensor([[0.]]), 3, **kwargs)
        second = memory.query_flat(torch.tensor([[0.]]), 3, **kwargs)
        self.assertEqual(first.item_ids.tolist(), second.item_ids.tolist())
        self.assertEqual(3, len(set(first.item_ids[0, first.valid_mask[0]].tolist())))

    def test_causal_update_never_reads_a_future_item(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        retrieved, counts, sizes, memory_bytes = update_and_retrieve_support_causal_batch(
            memory, torch.tensor([[0.], [10.]]), torch.tensor([[1.], [2.]]), torch.tensor([0, 0]),
            torch.tensor([0, 0]), torch.tensor([0., 0.]), torch.tensor([0, 1]), topk=1,
            include_current=False, beta=0., selection="global_nearest",
        )
        self.assertEqual([[0.], [1.]], retrieved.tolist())
        self.assertEqual([0, 1], counts.tolist())
        self.assertEqual([1, 2], sizes.tolist())
        self.assertEqual([32, 64], memory_bytes.tolist())

    def test_atomic_update_can_read_later_batch_items_and_reports_post_batch_state(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        retrieved, counts, sizes, memory_bytes = update_and_retrieve_support_atomic_batch(
            memory, torch.tensor([[0.], [10.]]), torch.tensor([[1.], [2.]]), torch.tensor([0, 0]),
            torch.tensor([0, 0]), torch.tensor([0., 0.]), torch.tensor([0, 1]), topk=1,
            include_current=False, beta=0., selection="global_nearest",
        )
        self.assertEqual([[2.], [1.]], retrieved.tolist())
        self.assertEqual([1, 1], counts.tolist())
        self.assertEqual([2, 2], sizes.tolist())
        self.assertEqual([64, 64], memory_bytes.tolist())

    def test_atomic_excludes_only_the_current_item_id(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        retrieved, counts, _, _ = update_and_retrieve_support_atomic_batch(
            memory, torch.tensor([[0.], [1.]]), torch.tensor([[10.], [20.]]), torch.tensor([0, 0]),
            torch.tensor([0, 0]), torch.zeros(2), torch.tensor([3, 7]), topk=2,
            include_current=False, beta=0., selection="class_balanced",
        )
        self.assertEqual([[20.], [10.]], retrieved.tolist())
        self.assertEqual([1, 1], counts.tolist())

    def test_atomic_include_current_retains_each_query_self_support(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        retrieved, counts, _, _ = update_and_retrieve_support_atomic_batch(
            memory, torch.tensor([[0.], [10.]]), torch.tensor([[1.], [2.]]), torch.tensor([0, 0]),
            torch.tensor([0, 0]), torch.zeros(2), torch.tensor([0, 1]), topk=1,
            include_current=True, beta=0., selection="class_balanced",
        )
        self.assertEqual([[1.], [2.]], retrieved.tolist())
        self.assertEqual([1, 1], counts.tolist())

    def test_atomic_schedule_is_exactly_causal_for_single_item_batches(self):
        inputs = dict(
            features=torch.tensor([[0.]]), gradients=torch.tensor([[10.]]),
            predicted_classes=torch.tensor([0]), contexts=torch.tensor([0]), entropies=torch.tensor([0.]),
            item_ids=torch.tensor([4]), topk=1, include_current=True, beta=0., selection="class_balanced",
        )
        causal_memory = StructuredGradientMemory(1, 1, 1, 1, device="cpu", capacity_scope="per_class")
        atomic_memory = StructuredGradientMemory(1, 1, 1, 1, device="cpu", capacity_scope="per_class")
        for memory in (causal_memory, atomic_memory):
            memory.add(torch.tensor([[2.]]), torch.tensor([[20.]]), 0, 0, 0., item_ids=2)
        causal = update_and_retrieve_support_causal_batch(causal_memory, **inputs)
        atomic = update_and_retrieve_support_atomic_batch(atomic_memory, **inputs)
        for causal_value, atomic_value in zip(causal, atomic):
            self.assertTrue(torch.equal(causal_value, atomic_value))
        self.assertEqual(atomic_memory.diagnostics(), causal_memory.diagnostics())
        retained = atomic_memory.query_flat(torch.tensor([[0.]]), 1, selection="global_nearest")
        self.assertEqual([4], retained.item_ids[0, retained.valid_mask[0]].tolist())

    def test_atomic_historical_only_singleton_can_differ_after_full_capacity_eviction(self):
        inputs = dict(
            features=torch.tensor([[0.]]), gradients=torch.tensor([[10.]]),
            predicted_classes=torch.tensor([0]), contexts=torch.tensor([0]), entropies=torch.tensor([0.]),
            item_ids=torch.tensor([4]), topk=1, include_current=False, beta=0., selection="class_balanced",
        )
        causal_memory = StructuredGradientMemory(1, 1, 1, 1, device="cpu", capacity_scope="per_class")
        atomic_memory = StructuredGradientMemory(1, 1, 1, 1, device="cpu", capacity_scope="per_class")
        for memory in (causal_memory, atomic_memory):
            memory.add(torch.tensor([[2.]]), torch.tensor([[20.]]), 0, 0, 0., item_ids=2)
        causal, causal_counts, _, _ = update_and_retrieve_support_causal_batch(causal_memory, **inputs)
        atomic, atomic_counts, _, _ = update_and_retrieve_support_atomic_batch(atomic_memory, **inputs)
        self.assertEqual([[20.]], causal.tolist())
        self.assertEqual([[0.]], atomic.tolist())
        self.assertEqual([1], causal_counts.tolist())
        self.assertEqual([0], atomic_counts.tolist())
        self.assertEqual(atomic_memory.diagnostics(), causal_memory.diagnostics())

    def test_historical_only_queries_before_capacity_eviction_for_every_selection(self):
        selections = ("random", "same_class", "global_nearest", "context_nearest", "class_balanced")
        for selection in selections:
            with self.subTest(selection=selection):
                memory = StructuredGradientMemory(
                    1, 1, 1, 1, device="cpu", capacity_scope="per_class"
                )
                retrieved, counts, sizes, memory_bytes = update_and_retrieve_support_causal_batch(
                    memory, torch.tensor([[0.], [1.], [2.]]), torch.tensor([[10.], [20.], [30.]]),
                    torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0]), torch.zeros(3),
                    torch.tensor([0, 1, 2]), topk=1, include_current=False, beta=0.,
                    selection=selection, random_seed=7,
                )
                self.assertEqual([[0.], [10.], [20.]], retrieved.tolist())
                self.assertEqual([0, 1, 1], counts.tolist())
                self.assertEqual([1, 1, 1], sizes.tolist())
                self.assertEqual([32, 32, 32], memory_bytes.tolist())
                retained = memory.query_flat(torch.tensor([[2.]]), 1, selection="global_nearest")
                self.assertEqual([2], retained.item_ids[0, retained.valid_mask[0]].tolist())

    def test_retained_byte_diagnostics_follow_each_insertion(self):
        memory = StructuredGradientMemory(2, 1, 1, 1, device="cpu", capacity_scope="per_class")
        _, _, sizes, memory_bytes = update_and_retrieve_support_causal_batch(
            memory, torch.tensor([[0.], [1.], [2.]]), torch.tensor([[1.], [2.], [3.]]),
            torch.tensor([0, 1, 0]), torch.tensor([0, 0, 0]), torch.zeros(3),
            torch.tensor([0, 1, 2]), topk=1, include_current=True, beta=0.,
            selection="global_nearest",
        )
        self.assertEqual([1, 2, 2], sizes.tolist())
        self.assertEqual([32, 64, 64], memory_bytes.tolist())

    def test_context_only_reports_two_spawns_as_a_causal_timeline(self):
        timeline = causal_active_context_counts(
            0, torch.tensor([True, True], dtype=torch.bool)
        )
        self.assertEqual([1, 2], timeline.tolist())

        method = object.__new__(ContextOnlyRamen)
        method.router = type("Router", (), {"num_contexts": 2})()
        method.memory = StructuredGradientMemory(
            1, 2, 1, 1, device="cpu", capacity_scope="per_class"
        )
        diagnostics = method._diagnostics(active_contexts=timeline)
        self.assertEqual([1, 2], diagnostics["num_active_contexts"].tolist())

    def test_fixed_context_baseline_reports_one_context_per_item(self):
        method = object.__new__(GlobalNearestRamen)
        method.router = None
        method.memory = StructuredGradientMemory(
            1, 2, 1, 1, device="cpu", capacity_scope="per_class"
        )
        timeline = torch.ones(3, dtype=torch.long)
        diagnostics = method._diagnostics(active_contexts=timeline)
        self.assertEqual([1, 1, 1], diagnostics["num_active_contexts"].tolist())

    def test_causal_ramen_matches_class_balanced_aggregation_without_future_leak(self):
        memory = StructuredGradientMemory(2, 4, 1, 1, device="cpu", capacity_scope="per_class")
        retrieved, active_classes, _, _ = update_and_retrieve_support_causal_batch(
            memory, torch.tensor([[0.], [1.], [2.]]), torch.tensor([[2.], [6.], [8.]]),
            torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]), torch.zeros(3),
            torch.tensor([0, 1, 2]), topk=2, include_current=True, beta=1.,
            selection="class_balanced",
        )
        self.assertTrue(torch.allclose(retrieved[0], torch.tensor([2.])))
        self.assertTrue(torch.allclose(retrieved[1], torch.tensor([6. + 2. * torch.exp(torch.tensor(-1.))])))
        expected = (2. * torch.exp(torch.tensor(-2.)) + 6. * torch.exp(torch.tensor(-1.)) + 8.) / 2
        self.assertTrue(torch.allclose(retrieved[2].squeeze(), expected))
        self.assertEqual([1, 1, 2], active_classes.tolist())

    def test_unbalanced_aggregation_keeps_ramen_weighting(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        memory.add(torch.tensor([[0.], [1.]]), torch.tensor([[2.], [6.]]), 0, 0,
                   torch.tensor([0., 0.]))
        retrieved = memory.query_flat(torch.tensor([[0.]]), 2, selection="global_nearest")
        gradients, counts = aggregate_unbalanced_gradients(retrieved, beta=1.)
        self.assertEqual([2], counts.tolist())
        self.assertTrue(torch.allclose(gradients.squeeze(), 1 + 3 * torch.exp(torch.tensor(-1.))))

    def test_reset_clears_the_shared_memory_and_counter(self):
        method = object.__new__(GlobalNearestRamen)
        method.memory = StructuredGradientMemory(1, 2, 1, 1, device="cpu", capacity_scope="per_class")
        method.memory.add(torch.tensor([[0.]]), torch.tensor([[1.]]), 0, 0, 0.)
        method.counter = 4
        method.router = None
        method.model = type("Model", (), {"reset_parameters": lambda self: None})()
        method.last_diagnostics = {}
        method.reset()
        self.assertEqual(0, method.counter)
        self.assertEqual(0, method.memory.size)

    def test_manifest_classes_are_distinct_and_context_only_routes(self):
        self.assertEqual("random", RandomMemoryRamen.support_selection)
        self.assertEqual("same_class", SameClassRamen.support_selection)
        self.assertEqual("global_nearest", GlobalNearestRamen.support_selection)
        self.assertEqual("context_nearest", ContextOnlyRamen.support_selection)
        self.assertEqual("class_balanced", CausalRamen.support_selection)
        self.assertEqual("class_balanced", StructuredAtomicRamen.support_selection)
        self.assertEqual("causal", CausalRamen.retrieval_schedule)
        self.assertEqual("batch_atomic", StructuredAtomicRamen.retrieval_schedule)
        # This test intentionally records the study mapping instead of adding
        # a copy of the original Ramen implementation to the ablation module.
        from methods import SupportAblations
        self.assertIn("class-balanced-without-context-routing", SupportAblations.__doc__)
        self.assertIn("legacy batch-atomic", SupportAblations.__doc__)

    def test_configs_validate_for_every_dataset_and_smoke(self):
        root = Path(__file__).resolve().parents[1]
        methods = {
            "RandomMemoryRamen": "random", "SameClassRamen": "same_class",
            "GlobalNearestRamen": "global_nearest", "ContextOnlyRamen": "context_nearest",
            "CausalRamen": "class_balanced",
            "StructuredAtomicRamen": "class_balanced",
        }
        for path in root.glob("cfg/*/*.yaml"):
            if path.stem not in methods:
                continue
            with path.open() as config_file:
                cfg = validate_support_ablation_config(yaml.safe_load(config_file), selection=methods[path.stem])
            self.assertEqual("per_class", cfg["capacity_scope"], path)
        for name, selection in methods.items():
            path = root / "cfg" / "smoke" / "CIFAR100C" / f"{name}.yaml"
            with path.open() as config_file:
                validate_support_ablation_config(yaml.safe_load(config_file), selection=selection)

    def test_atomic_configs_match_causal_values(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("CIFAR100C", "DomainNet", "CIFAR10C", "ImageNetC5K", "smoke/CIFAR100C"):
            with (root / "cfg" / dataset / "CausalRamen.yaml").open() as source:
                causal = yaml.safe_load(source)
            with (root / "cfg" / dataset / "StructuredAtomicRamen.yaml").open() as source:
                atomic = yaml.safe_load(source)
            self.assertEqual(causal, atomic, dataset)


if __name__ == "__main__":
    unittest.main()
