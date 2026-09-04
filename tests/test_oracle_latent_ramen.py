import sys
import unittest
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import _provide_oracle_domain_context
from memory.structured_memory import StructuredGradientMemory
from methods.OracleLatentRamen import (
    OracleDomainContextHook,
    OracleLatentRamen,
    update_and_retrieve_oracle_causal_batch,
    validate_oracle_latent_ramen_config,
)


class _OracleHook(OracleDomainContextHook):
    def __init__(self):
        self._initialize_oracle_context_hook()


class OracleLatentRamenPureTests(unittest.TestCase):
    def test_config_requires_explicit_evaluator_domain_provenance(self):
        base = {"max_capacity": 2, "topk": 1, "optimizer": "signsgd", "lr": .01}
        with self.assertRaises(ValueError):
            validate_oracle_latent_ramen_config(base)
        cfg = validate_oracle_latent_ramen_config({
            **base, "oracle_context_source": "evaluator_domain_idx",
        })
        self.assertEqual("evaluator_domain_idx", cfg["oracle_context_source"])

    def test_all_oracle_configs_use_per_class_capacity_scope(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.glob("cfg/*/OracleLatentRamen.yaml"):
            with path.open() as config_file:
                cfg = validate_oracle_latent_ramen_config(yaml.safe_load(config_file))
            self.assertEqual("per_class", cfg["capacity_scope"], path)

    def test_context_hook_is_single_use_and_fail_closed(self):
        hook = _OracleHook()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            hook._consume_oracle_domain_context(1, torch.device("cpu"))
        hook.set_oracle_domain_context(torch.tensor([4]))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            hook.set_oracle_domain_context(torch.tensor([5]))
        self.assertEqual([4], hook._consume_oracle_domain_context(1, torch.device("cpu")).tolist())
        with self.assertRaisesRegex(RuntimeError, "missing"):
            hook._consume_oracle_domain_context(1, torch.device("cpu"))

    def test_mismatched_oracle_context_is_discarded(self):
        hook = _OracleHook()
        hook.set_oracle_domain_context(torch.tensor([1, 2]))
        with self.assertRaisesRegex(RuntimeError, "batch size"):
            hook._consume_oracle_domain_context(1, torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "missing"):
            hook._consume_oracle_domain_context(2, torch.device("cpu"))
        hook.set_oracle_domain_context(torch.tensor([3]))
        hook._clear_oracle_domain_context()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            hook._consume_oracle_domain_context(1, torch.device("cpu"))

    def test_reset_discards_an_unconsumed_oracle_context(self):
        class Model:
            def reset_parameters(self):
                pass

        method = object.__new__(OracleLatentRamen)
        method.model = Model()
        method.memory = StructuredGradientMemory(1, 1, 1, 1, device="cpu")
        method.counter = 12
        method._initialize_oracle_context_hook()
        method._seen_oracle_contexts.update({3, 7})
        method.set_oracle_domain_context(torch.tensor([3]))
        method.reset()
        self.assertEqual(0, method.counter)
        self.assertEqual(set(), method._seen_oracle_contexts)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            method._consume_oracle_domain_context(1, torch.device("cpu"))

    def test_evaluator_context_never_reaches_ordinary_method(self):
        class OrdinaryMethod:
            pass

        class OracleMethod:
            requires_oracle_domain_context = True

            def __init__(self):
                self.seen = None

            def set_oracle_domain_context(self, values):
                self.seen = values

        domains = torch.tensor([2, 7])
        ordinary = OrdinaryMethod()
        _provide_oracle_domain_context(ordinary, domains)
        self.assertFalse(hasattr(ordinary, "seen"))
        oracle = OracleMethod()
        _provide_oracle_domain_context(oracle, domains)
        self.assertIs(oracle.seen, domains)

    def test_causal_oracle_insertions_and_diagnostics_are_vector_aligned(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu")
        result = update_and_retrieve_oracle_causal_batch(
            memory,
            features=torch.tensor([[0.], [10.], [20.]]),
            gradients=torch.tensor([[1.], [2.], [3.]]),
            predicted_classes=torch.tensor([0, 0, 0]),
            contexts=torch.tensor([4, 4, 9]),
            entropies=torch.tensor([0., 0., 0.]),
            item_ids=torch.tensor([0, 1, 2]),
            topk=1,
            include_current=False,
            beta=0.,
        )
        retrieved, active_classes, memory_sizes, memory_bytes, active_contexts = result
        self.assertEqual([[0.], [1.], [0.]], retrieved.tolist())
        self.assertEqual([0, 1, 0], active_classes.tolist())
        self.assertEqual([1, 2, 3], memory_sizes.tolist())
        self.assertEqual([32, 64, 96], memory_bytes.tolist())
        self.assertEqual([1, 1, 2], active_contexts.tolist())

        method = object.__new__(OracleLatentRamen)
        method.memory = memory
        diagnostics = method._diagnostics(
            torch.tensor([4, 4, 9]), active_classes, memory_sizes, memory_bytes, active_contexts,
        )
        self.assertEqual([4, 4, 9], diagnostics["inferred_context"].tolist())
        self.assertEqual([1, 2, 3], diagnostics["memory_size"].tolist())
        self.assertEqual([32, 64, 96], diagnostics["memory_bytes"].tolist())
        self.assertEqual([1, 1, 2], diagnostics["num_active_contexts"].tolist())

    def test_hard_oracle_composition_reports_purity_and_coverage_loss(self):
        memory = StructuredGradientMemory(2, 4, 1, 1, device="cpu", capacity_scope="per_class")
        result = update_and_retrieve_oracle_causal_batch(
            memory,
            features=torch.tensor([[0.], [10.], [20.]]),
            gradients=torch.tensor([[1.], [2.], [3.]]),
            predicted_classes=torch.tensor([0, 0, 0]),
            contexts=torch.tensor([4, 4, 9]),
            entropies=torch.zeros(3), item_ids=torch.tensor([0, 1, 2]), topk=1,
            include_current=False, beta=0., return_composition=True,
        )
        *_, composition = result
        self.assertEqual([0, 1, 0], composition["returned_support_count"].tolist())
        self.assertEqual([0, 1, 0], composition["active_class_count"].tolist())
        self.assertEqual([0., .5, 0.], composition["class_coverage"].tolist())
        self.assertEqual([0., 1., 0.], composition["same_domain_ratio"].tolist())
        self.assertEqual([0., 0., 0.], composition["cross_domain_ratio"].tolist())
        self.assertEqual([0., 1., 0.], composition["effective_sample_size"].tolist())
        self.assertEqual((3, 2, 1), tuple(composition["support_item_ids"].shape))
        self.assertEqual(torch.bool, composition["support_valid_mask"].dtype)
        self.assertEqual([0], composition["support_item_ids"][1, 0, :].tolist())
        self.assertTrue(composition["support_valid_mask"][1, 0, 0])
        self.assertFalse(composition["support_valid_mask"][1, 1, 0])

        method = object.__new__(OracleLatentRamen)
        method.memory = memory
        method._seen_oracle_contexts = {4, 9}
        diagnostics = method._diagnostics(composition=composition)
        self.assertEqual([0., 1., 0.], diagnostics["same_domain_ratio"].tolist())
        self.assertTrue(torch.equal(composition["support_valid_mask"], diagnostics["support_valid_mask"]))

    def test_hard_oracle_diagnostics_exclude_soft_only_context_strength(self):
        method = object.__new__(OracleLatentRamen)
        method.memory = StructuredGradientMemory(1, 1, 1, 1, device="cpu")
        method._seen_oracle_contexts = set()
        diagnostics = method._diagnostics()
        self.assertNotIn("context_strength", diagnostics)

    def test_observed_context_count_survives_per_class_memory_eviction(self):
        memory = StructuredGradientMemory(
            1, 1, 1, 1, device="cpu", capacity_scope="per_class",
        )
        seen_contexts = set()
        result = update_and_retrieve_oracle_causal_batch(
            memory,
            features=torch.tensor([[0.], [10.]]),
            gradients=torch.tensor([[1.], [2.]]),
            predicted_classes=torch.tensor([0, 0]),
            contexts=torch.tensor([4, 9]),
            entropies=torch.tensor([0., 0.]),
            item_ids=torch.tensor([0, 1]),
            topk=1,
            include_current=True,
            beta=0.,
            seen_contexts=seen_contexts,
        )
        _, _, memory_sizes, memory_bytes, observed_contexts = result
        self.assertEqual([1, 1], memory_sizes.tolist())
        self.assertEqual([32, 32], memory_bytes.tolist())
        self.assertEqual([1, 2], observed_contexts.tolist())
        self.assertEqual({4, 9}, seen_contexts)
        self.assertEqual(1, memory.active_contexts)
        first_context = memory.query(torch.tensor([[0.]]), 4, topk=1)
        self.assertFalse(first_context.valid_mask[0, 0].any())

    def test_method_diagnostics_keep_observed_and_live_memory_contexts_separate(self):
        method = object.__new__(OracleLatentRamen)
        method._seen_oracle_contexts = {4, 9}
        method.memory = StructuredGradientMemory(1, 1, 1, 1, device="cpu")
        method.memory.add(torch.tensor([[0.]]), torch.tensor([[1.]]), 0, 9, 0.)
        method.memory.diagnostics = lambda: self.fail("full memory diagnostics scanned in hot path")
        diagnostics = method._diagnostics()
        self.assertEqual(2, diagnostics["num_active_contexts"])
        self.assertEqual(1, diagnostics["memory_active_contexts"])
        self.assertEqual(32, diagnostics["memory_bytes"])


if __name__ == "__main__":
    unittest.main()
