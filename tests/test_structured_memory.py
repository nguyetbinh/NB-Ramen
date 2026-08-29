import unittest

try:
    import torch
except ImportError:  # pragma: no cover - enables source-only environments
    torch = None

if torch is not None:
    from src.memory.structured_memory import StructuredGradientMemory


@unittest.skipIf(torch is None, "PyTorch is required for structured memory tests")
class StructuredGradientMemoryTests(unittest.TestCase):
    def make_memory(self, capacity=3, dtype=None, capacity_scope="per_class_context"):
        if dtype is None:
            dtype = torch.float32
        return StructuredGradientMemory(
            3, capacity, 2, 2, device="cpu", dtype=dtype, capacity_scope=capacity_scope,
        )

    @staticmethod
    def add(memory, features, classes, contexts, **kwargs):
        tensor = torch.tensor(features, dtype=torch.float32)
        return memory.add(tensor, tensor + 10, torch.tensor(classes), torch.tensor(contexts),
                          torch.zeros(len(features)), **kwargs)

    def test_oldest_item_is_evicted_per_class_context_bucket(self):
        memory = self.make_memory(capacity=2)
        self.add(memory, [[0, 0], [1, 0], [2, 0]], [0, 0, 0], [7, 7, 7])
        result = memory.query(torch.tensor([[0.0, 0.0]]), 7, topk=2)
        self.assertEqual({1, 2}, set(result.item_ids[0, 0][result.valid_mask[0, 0]].tolist()))
        self.assertEqual(2, memory.size)

    def test_per_class_capacity_is_shared_across_context_buckets(self):
        memory = self.make_memory(capacity=2, capacity_scope="per_class")
        self.add(memory, [[0, 0], [1, 0], [2, 0]], [0, 0, 0], [1, 2, 3])
        self.assertEqual(2, memory.size)
        self.assertLessEqual(memory.size, memory.num_classes * memory.max_capacity)
        self.assertEqual(2, memory.diagnostics()["per_class_sizes"][0])
        evicted_context = memory.query(torch.tensor([[0.0, 0.0]]), 1, topk=2)
        self.assertEqual([], evicted_context.item_ids[0, 0][evicted_context.valid_mask[0, 0]].tolist())
        for context, expected_id in ((2, 1), (3, 2)):
            result = memory.query(torch.tensor([[0.0, 0.0]]), context, topk=1)
            self.assertEqual([expected_id], result.item_ids[0, 0][result.valid_mask[0, 0]].tolist())

    def test_per_class_eviction_removes_oldest_across_buckets_and_keeps_other_classes(self):
        memory = self.make_memory(capacity=2, capacity_scope="per_class")
        self.add(
            memory,
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [0, 0, 1, 0],
            [4, 8, 8, 12],
        )
        self.assertEqual(2, memory.diagnostics()["per_class_sizes"][0])
        self.assertEqual(1, memory.diagnostics()["per_class_sizes"][1])
        self.assertEqual(3, memory.size)
        old_bucket = memory.query(torch.tensor([[0.0, 0.0]]), 4, topk=1)
        self.assertFalse(old_bucket.valid_mask[0, 0].any())
        surviving = memory.query(torch.tensor([[0.0, 0.0]]), 8, topk=1)
        self.assertEqual([1], surviving.item_ids[0, 0][surviving.valid_mask[0, 0]].tolist())
        other_class = memory.query(torch.tensor([[0.0, 0.0]]), 8, topk=1)
        self.assertEqual([2], other_class.item_ids[0, 1][other_class.valid_mask[0, 1]].tolist())
        self.assertEqual(2, memory.active_contexts)

    def test_tied_recencies_use_stable_item_id_tiebreak_and_keep_counters_exact(self):
        memory = self.make_memory(capacity=2, capacity_scope="per_class")
        self.add(
            memory,
            [[0, 0], [1, 0]],
            [0, 0],
            [1, 2],
            item_ids=torch.tensor([20, 10]),
            recencies=torch.tensor([5, 5]),
        )
        self.add(
            memory,
            [[2, 0]],
            [0],
            [3],
            item_ids=torch.tensor([30]),
            recencies=torch.tensor([6]),
        )
        self.assertEqual([20], memory.query(torch.tensor([[0., 0.]]), 1, 1).item_ids[0, 0, :].tolist())
        evicted = memory.query(torch.tensor([[0., 0.]]), 2, 1)
        self.assertFalse(evicted.valid_mask[0, 0].any())
        diagnostics = memory.diagnostics()
        self.assertEqual(2, diagnostics["size"])
        self.assertEqual({0: 2, 1: 0, 2: 0}, diagnostics["per_class_sizes"])
        self.assertEqual(80, diagnostics["bytes"])

    def test_mixed_multi_add_updates_per_class_capacity_and_all_counters(self):
        memory = self.make_memory(capacity=2, capacity_scope="per_class")
        self.add(
            memory,
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
            [0, 1, 0, 1, 0, 2],
            [1, 1, 2, 2, 3, 3],
        )
        diagnostics = memory.diagnostics()
        self.assertEqual(5, diagnostics["size"])
        self.assertEqual(3, diagnostics["active_contexts"])
        self.assertEqual(5, diagnostics["active_buckets"])
        self.assertEqual({0: 2, 1: 2, 2: 1}, diagnostics["per_class_sizes"])
        self.assertEqual(200, diagnostics["bytes"])
        self.assertEqual(diagnostics["bytes"], memory.retained_bytes)

    def test_mixed_multi_add_updates_per_bucket_replacement_counters(self):
        memory = self.make_memory(capacity=1, capacity_scope="per_class_context")
        self.add(
            memory,
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [0, 0, 1, 0],
            [1, 1, 1, 2],
        )
        diagnostics = memory.diagnostics()
        self.assertEqual(3, diagnostics["size"])
        self.assertEqual(2, diagnostics["active_contexts"])
        self.assertEqual(3, diagnostics["active_buckets"])
        self.assertEqual({0: 2, 1: 1, 2: 0}, diagnostics["per_class_sizes"])
        self.assertEqual(120, diagnostics["bytes"])

    def test_unknown_capacity_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_memory(capacity_scope="global")

    def test_per_class_context_scope_retains_independent_bucket_capacity(self):
        memory = self.make_memory(capacity=2, capacity_scope="per_class_context")
        self.add(memory, [[0, 0], [1, 0], [2, 0]], [0, 0, 0], [1, 2, 3])
        self.assertEqual(3, memory.size)
        self.assertEqual(3, memory.diagnostics()["per_class_sizes"][0])

    def test_retrieval_is_nearest_and_class_balanced(self):
        memory = self.make_memory()
        self.add(memory, [[4, 0], [1, 0], [8, 0], [2, 0]], [0, 0, 1, 1], [3, 3, 3, 3])
        result = memory.query(torch.tensor([[0.0, 0.0]]), 3, topk=1)
        self.assertTrue(result.valid_mask[0, 0, 0])
        self.assertTrue(result.valid_mask[0, 1, 0])
        self.assertEqual(1, result.item_ids[0, 0, 0].item())
        self.assertEqual(3, result.item_ids[0, 1, 0].item())
        self.assertFalse(result.valid_mask[0, 2].any())

    def test_context_restriction_does_not_cross_contexts(self):
        memory = self.make_memory()
        self.add(memory, [[0, 0], [9, 0]], [0, 0], [1, 2])
        result = memory.query(torch.tensor([[0.0, 0.0]]), 2, topk=2)
        self.assertEqual([1], result.item_ids[0, 0][result.valid_mask[0, 0]].tolist())

    def test_excluding_current_item_uses_stable_ids(self):
        memory = self.make_memory()
        ids = self.add(memory, [[0, 0], [1, 0]], [0, 0], [4, 4], item_ids=torch.tensor([40, 90]))
        self.assertEqual([40, 90], ids.tolist())
        result = memory.query(torch.tensor([[0.0, 0.0]]), 4, topk=2,
                              include_current=False, current_item_ids=40)
        self.assertEqual([90], result.item_ids[0, 0][result.valid_mask[0, 0]].tolist())
        with self.assertRaises(ValueError):
            memory.query(torch.tensor([[0.0, 0.0]]), 4, topk=1, include_current=False)

    def test_legal_candidate_snapshot_is_read_only_and_uses_retrieval_filters(self):
        memory = self.make_memory()
        self.add(memory, [[0, 0], [1, 0], [2, 0]], [0, 1, 0], [4, 4, 5], item_ids=torch.tensor([40, 50, 60]))
        before = memory.diagnostics()
        snapshot = memory.legal_candidate_snapshot(
            4, schedule="causal", selection="class_balanced", include_current=False, current_item_ids=40,
        )
        self.assertEqual([50], [item["item_id"] for item in snapshot[0]])
        self.assertEqual(before, memory.diagnostics())
        with self.assertRaises(ValueError):
            memory.legal_candidate_snapshot(4, schedule="unknown")

    def test_reset_clears_memory_and_restarts_generated_ids(self):
        memory = self.make_memory()
        self.add(memory, [[0, 0]], [0], [1])
        memory.reset()
        self.assertEqual(0, memory.diagnostics()["size"])
        self.assertEqual(0, memory.diagnostics()["active_contexts"])
        self.assertEqual(0, memory.diagnostics()["active_buckets"])
        self.assertEqual(0, memory.diagnostics()["bytes"])
        ids = self.add(memory, [[1, 1]], [1], [2])
        self.assertEqual([0], ids.tolist())

    def test_diagnostics_and_half_precision_recency_are_safe(self):
        memory = self.make_memory(dtype=torch.float16)
        self.add(memory, [[1, 2]], [2], [8], recencies=torch.tensor([100_000], dtype=torch.long),
                 reliabilities=torch.tensor([0.25]))
        result = memory.query(torch.tensor([[1.0, 2.0]]), 8, topk=1)
        self.assertEqual(100_000, result.recencies[0, 2, 0].item())
        self.assertEqual(torch.float16, result.gradients.dtype)
        diagnostics = memory.diagnostics()
        self.assertEqual(1, diagnostics["size"])
        self.assertEqual(1, diagnostics["active_contexts"])
        self.assertGreater(diagnostics["bytes"], 0)

    def test_invalid_input_is_rejected_without_partial_add(self):
        memory = self.make_memory()
        with self.assertRaises(ValueError):
            self.add(memory, [[0, 0], [1, 1]], [0, 99], [1, 1])
        self.assertEqual(0, memory.size)


if __name__ == "__main__":
    unittest.main()
