import math
import sys
import unittest
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory.structured_memory import StructuredGradientMemory
from methods import LatentHardRamen, OracleHardRamen, OracleLatentRamen
from methods.LatentRamen import LatentRamen
from methods.OracleLatentRamen import OracleDomainContextHook
from methods.SupportAblations import update_and_retrieve_support_causal_batch
from methods.SoftRoutingRamen import (
    OracleSoftRankRamen,
    soft_routing_influence_diagnostics,
    support_composition_diagnostics,
    update_and_retrieve_oracle_soft_rank_causal_batch,
    validate_oracle_soft_rank_ramen_config,
)


class _OracleHook(OracleDomainContextHook):
    def __init__(self):
        self._initialize_oracle_context_hook()


class SoftRoutingRamenTests(unittest.TestCase):
    def _memory(self):
        return StructuredGradientMemory(2, 8, 1, 1, device="cpu", capacity_scope="per_class")

    def test_config_requires_explicit_oracle_source_per_class_and_valid_gamma(self):
        base = {"max_capacity": 2, "topk": 1, "optimizer": "signsgd", "lr": .01,
                "capacity_scope": "per_class", "oracle_context_source": "evaluator_domain_idx"}
        self.assertEqual(.25, validate_oracle_soft_rank_ramen_config({**base, "gamma": .25})["gamma"])
        for gamma in (True, None, float("nan"), float("inf"), -.01):
            with self.assertRaises(ValueError):
                validate_oracle_soft_rank_ramen_config({**base, "gamma": gamma})
        with self.assertRaises(ValueError):
            validate_oracle_soft_rank_ramen_config({**base, "capacity_scope": "per_class_context"})
        with self.assertRaises(ValueError):
            validate_oracle_soft_rank_ramen_config({**base, "oracle_context_source": "other"})

    def test_cifar_config_is_discoverable_by_exact_method_name(self):
        path = Path(__file__).resolve().parents[1] / "cfg" / "CIFAR100C" / "OracleSoftRankRamen.yaml"
        cfg = validate_oracle_soft_rank_ramen_config(yaml.safe_load(path.read_text()))
        self.assertEqual(.5, cfg["gamma"])
        self.assertEqual("per_class", cfg["capacity_scope"])

    def test_hard_aliases_preserve_old_method_semantics(self):
        self.assertIs(OracleHardRamen, OracleLatentRamen)
        self.assertIs(LatentHardRamen, LatentRamen)

    def test_gamma_zero_recovers_causal_global_ranking(self):
        memory = self._memory()
        memory.add(torch.tensor([[2.], [1.], [3.]]), torch.tensor([[20.], [10.], [30.]]),
                   torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]), torch.zeros(3),
                   item_ids=torch.tensor([2, 1, 3]))
        causal = memory.query(torch.tensor([[0.]]), 0, topk=2)
        soft = memory.query_class_balanced_global(
            torch.tensor([[0.]]), 2, query_contexts=torch.tensor([7]), context_strength=0.,
        )
        self.assertTrue(torch.equal(causal.item_ids, soft.item_ids))
        self.assertTrue(torch.equal(causal.valid_mask, soft.valid_mask))
        self.assertTrue(torch.equal(causal.distances, soft.distances))

    def test_soft_bonus_prefers_matching_context_without_excluding_cross_context(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        memory.add(torch.tensor([[1.], [1.]]), torch.tensor([[1.], [2.]]), torch.tensor([0, 0]),
                   torch.tensor([4, 9]), torch.zeros(2), item_ids=torch.tensor([4, 9]))
        retrieval = memory.query_class_balanced_global(
            torch.tensor([[0.]]), 2, query_contexts=torch.tensor([9]), context_strength=.5,
        )
        self.assertEqual([9, 4], retrieval.item_ids[0, 0, retrieval.valid_mask[0, 0]].tolist())
        self.assertEqual([9, 4], retrieval.contexts[0, 0, retrieval.valid_mask[0, 0]].tolist())

    def test_strict_causal_loop_never_uses_future_support(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        retrieved, active, sizes, _, diagnostics = update_and_retrieve_oracle_soft_rank_causal_batch(
            memory, torch.tensor([[0.], [10.]]), torch.tensor([[1.], [2.]]), torch.tensor([0, 0]),
            torch.tensor([3, 3]), torch.zeros(2), torch.tensor([0, 1]), topk=1,
            include_current=False, beta=0., gamma=1.,
        )
        self.assertEqual([[0.], [1.]], retrieved.tolist())
        self.assertEqual([0, 1], active.tolist())
        self.assertEqual([1, 2], sizes.tolist())
        self.assertEqual([0, 1], diagnostics["returned_support_count"].tolist())

    def test_gamma_zero_matches_causal_ramen_for_both_inclusion_schedules(self):
        stream = dict(
            features=torch.tensor([[0.], [1.], [2.]]), gradients=torch.tensor([[10.], [20.], [30.]]),
            predicted_classes=torch.tensor([0, 0, 0]), contexts=torch.zeros(3, dtype=torch.long),
            entropies=torch.zeros(3), item_ids=torch.tensor([0, 1, 2]), topk=1, beta=0.,
        )
        for include_current in (True, False):
            with self.subTest(include_current=include_current):
                baseline = update_and_retrieve_support_causal_batch(
                    StructuredGradientMemory(1, 1, 1, 1, device="cpu", capacity_scope="per_class"),
                    **stream, include_current=include_current, selection="class_balanced",
                )
                soft = update_and_retrieve_oracle_soft_rank_causal_batch(
                    StructuredGradientMemory(1, 1, 1, 1, device="cpu", capacity_scope="per_class"),
                    **stream, include_current=include_current, gamma=0.,
                )
                for actual, expected in zip(soft[:4], baseline):
                    self.assertTrue(torch.equal(actual, expected))
                for name in ("selection_change_ratio", "mean_context_bonus", "mean_rank_displacement"):
                    self.assertTrue(torch.equal(torch.zeros(3), soft[4][name]))

    def test_influence_diagnostics_and_raw_support_metadata_are_per_sample(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        memory.add(torch.tensor([[1.], [1.]]), torch.tensor([[1.], [2.]]), torch.tensor([0, 0]),
                   torch.tensor([4, 9]), torch.zeros(2), item_ids=torch.tensor([4, 9]))
        soft = memory.query_class_balanced_global(
            torch.tensor([[0.]]), 2, query_contexts=torch.tensor([9]), context_strength=.5,
        )
        reference = memory.query_class_balanced_global(
            torch.tensor([[0.]]), 2, query_contexts=torch.tensor([9]), context_strength=0.,
        )
        influence = soft_routing_influence_diagnostics(
            soft, reference, query_contexts=torch.tensor([9]), gamma=.5,
        )
        self.assertEqual([1.], influence["selection_change_ratio"].tolist())
        self.assertEqual([.25], influence["mean_context_bonus"].tolist())
        self.assertEqual([1.], influence["mean_rank_displacement"].tolist())

        _, _, _, _, diagnostics = update_and_retrieve_oracle_soft_rank_causal_batch(
            self._memory(), torch.tensor([[0.]]), torch.tensor([[1.]]), torch.tensor([0]),
            torch.tensor([9]), torch.zeros(1), torch.tensor([10]), topk=1,
            include_current=True, beta=0., gamma=.5,
        )
        self.assertEqual((1, 2, 1), tuple(diagnostics["support_item_ids"].shape))
        self.assertEqual(torch.bool, diagnostics["support_valid_mask"].dtype)

    def test_historical_only_full_capacity_matches_causal_ramen_semantics(self):
        memory = StructuredGradientMemory(1, 1, 1, 1, device="cpu", capacity_scope="per_class")
        retrieved, active, sizes, _, diagnostics = update_and_retrieve_oracle_soft_rank_causal_batch(
            memory, torch.tensor([[0.], [1.], [2.]]), torch.tensor([[10.], [20.], [30.]]),
            torch.tensor([0, 0, 0]), torch.tensor([3, 3, 3]), torch.zeros(3), torch.tensor([0, 1, 2]),
            topk=1, include_current=False, beta=0., gamma=0.,
        )
        # Historical retrieval happens before insertion, so the current item
        # cannot evict the prior support simply because it is excluded.
        self.assertEqual([[0.], [10.], [20.]], retrieved.tolist())
        self.assertEqual([0, 1, 1], active.tolist())
        self.assertEqual([1, 1, 1], sizes.tolist())
        self.assertEqual([0, 1, 1], diagnostics["returned_support_count"].tolist())

    def test_diagnostics_math_uses_final_entropy_distance_weights(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu", capacity_scope="per_class")
        memory.add(torch.tensor([[0.], [2.]]), torch.tensor([[1.], [1.]]), torch.tensor([0, 0]),
                   torch.tensor([1, 2]), torch.zeros(2), item_ids=torch.tensor([1, 2]))
        retrieval = memory.query_class_balanced_global(
            torch.tensor([[0.]]), 2, query_contexts=torch.tensor([1]), context_strength=0.,
        )
        metrics = support_composition_diagnostics(
            retrieval, query_contexts=torch.tensor([1]), beta=0., num_classes=1, context_strength=.5,
        )
        self.assertEqual([2], metrics["returned_support_count"].tolist())
        self.assertEqual([1], metrics["active_class_count"].tolist())
        self.assertEqual([1.], metrics["class_coverage"].tolist())
        self.assertEqual([.5], metrics["same_domain_ratio"].tolist())
        self.assertEqual([.5], metrics["cross_domain_ratio"].tolist())
        self.assertTrue(math.isclose(2., metrics["effective_sample_size"].item()))
        self.assertEqual([.5], metrics["context_strength"].tolist())

    def test_oracle_hook_isolated_from_support_method_until_consumption(self):
        hook = _OracleHook()
        hook.set_oracle_domain_context(torch.tensor([4]))
        self.assertEqual([4], hook._consume_oracle_domain_context(1, torch.device("cpu")).tolist())
        with self.assertRaisesRegex(RuntimeError, "missing"):
            hook._consume_oracle_domain_context(1, torch.device("cpu"))
        self.assertTrue(OracleSoftRankRamen.requires_oracle_domain_context)


if __name__ == "__main__":
    unittest.main()
