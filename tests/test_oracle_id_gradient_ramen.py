"""CPU-only mechanics tests for evaluator-only oracle gradient controls."""

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.OracleDropOODRamen import OracleDropOODRamen  # noqa: E402
from methods.OracleIDGradientRamen import (  # noqa: E402
    OracleIDGradientRamen,
    OracleOODContextHook,
    OraclePriorityCache,
    _direction_diagnostics,
    aggregate_oracle_supports,
    validate_oracle_id_gradient_config,
)


class _Hook(OracleOODContextHook):
    def __init__(self):
        self._initialize_oracle_ood_hook()


class OracleIDGradientRamenTests(unittest.TestCase):
    def _cache(self):
        return OraclePriorityCache(4, 1, 2, "cpu", torch.float32)

    def test_config_requires_explicit_evaluator_provenance(self):
        base = {"max_capacity": 2, "topk": 1, "optimizer": "signsgd", "lr": .01}
        with self.assertRaises(ValueError):
            validate_oracle_id_gradient_config(base)
        cfg = validate_oracle_id_gradient_config({**base, "oracle_ood_source": "evaluator_is_ood"})
        self.assertEqual("evaluator_is_ood", cfg["oracle_ood_source"])

    def test_ood_hook_is_single_use_fail_closed_and_boolean_only(self):
        hook = _Hook()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            hook._consume_oracle_is_ood(1, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "boolean"):
            hook.set_oracle_is_ood(torch.tensor([1]))
        values = torch.tensor([False, True])
        hook.set_oracle_is_ood(values)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            hook.set_oracle_is_ood(values)
        self.assertEqual([False, True], hook._consume_oracle_is_ood(2, torch.device("cpu")).tolist())
        with self.assertRaisesRegex(RuntimeError, "missing"):
            hook._consume_oracle_is_ood(2, torch.device("cpu"))

    def test_id_oracle_excludes_ood_gradient_but_retains_its_retrieval_evidence(self):
        cache = self._cache()
        cache.add(torch.tensor([[0.], [0.]]), torch.tensor([[1., 0.], [0., 2.]]),
                  torch.zeros(2), torch.tensor([0., 1.]), torch.tensor([False, True]))
        all_gradient, id_gradient, diagnostics = aggregate_oracle_supports(
            torch.tensor([[0.]]), [cache], topk=2, beta=0.
        )
        self.assertEqual([[1., 2.]], all_gradient.tolist())
        self.assertEqual([[1., 0.]], id_gradient.tolist())
        self.assertEqual([0.5], diagnostics["retrieved_ood_fraction"].tolist())
        self.assertEqual([0.5], diagnostics["retrieved_ood_weight_fraction"].tolist())
        self.assertAlmostEqual(1 / (5 ** .5), diagnostics["ramen_vs_oracle_id_cosine"][0])
        self.assertEqual(0.5, diagnostics["ramen_vs_oracle_id_sign_disagreement"][0])
        self.assertGreater(cache.retained_bytes, 0)

    def test_causal_cache_updates_do_not_expose_future_ood_supports(self):
        cache = self._cache()
        # First item is ID: its ID-only direction is defined only from itself.
        cache.add(torch.tensor([[0.]]), torch.tensor([[1., 0.]]), torch.zeros(1),
                  torch.tensor([0.]), torch.tensor([False]))
        _, first_id, first = aggregate_oracle_supports(torch.tensor([[0.]]), [cache], topk=2, beta=0.)
        # Add the future stream item only after evaluating the first item.
        cache.add(torch.tensor([[0.]]), torch.tensor([[0., 2.]]), torch.zeros(1),
                  torch.tensor([1.]), torch.tensor([True]))
        _, second_id, second = aggregate_oracle_supports(torch.tensor([[0.]]), [cache], topk=2, beta=0.)
        self.assertEqual([[1., 0.]], first_id.tolist())
        self.assertEqual([[1., 0.]], second_id.tolist())
        self.assertEqual([0.0], first["retrieved_ood_fraction"].tolist())
        self.assertEqual([0.5], second["retrieved_ood_fraction"].tolist())

    def test_zero_id_direction_has_undefined_direction_metrics(self):
        cache = self._cache()
        cache.add(torch.tensor([[0.]]), torch.tensor([[0., 2.]]), torch.zeros(1),
                  torch.tensor([0.]), torch.tensor([True]))
        _, id_gradient, diagnostics = aggregate_oracle_supports(torch.tensor([[0.]]), [cache], topk=1, beta=0.)
        self.assertEqual([[0., 0.]], id_gradient.tolist())
        self.assertIsNone(diagnostics["ramen_vs_oracle_id_cosine"][0])
        self.assertIsNone(diagnostics["ramen_vs_oracle_id_sign_disagreement"][0])

    def test_empty_supports_have_undefined_direction_metrics(self):
        _, id_gradient, diagnostics = aggregate_oracle_supports(
            torch.tensor([[0.]]), [self._cache()], topk=1, beta=0.
        )
        self.assertEqual([[0., 0.]], id_gradient.tolist())
        self.assertIsNone(diagnostics["ramen_vs_oracle_id_cosine"][0])
        self.assertIsNone(diagnostics["ramen_vs_oracle_id_sign_disagreement"][0])

    def test_float16_overflow_is_measured_in_float32(self):
        direction = torch.full((2_000,), 1_000., dtype=torch.float16)
        cosine, sign_disagreement = _direction_diagnostics(direction, direction)
        self.assertEqual(1.0, cosine)
        self.assertEqual(0.0, sign_disagreement)

    def test_id_only_support_has_a_schema_safe_unit_cosine(self):
        cache = self._cache()
        cache.add(torch.tensor([[0.]]), torch.tensor([[3., -4.]]), torch.zeros(1),
                  torch.tensor([0.]), torch.tensor([False]))
        _, _, diagnostics = aggregate_oracle_supports(
            torch.tensor([[0.]]), [cache], topk=1, beta=0.
        )
        self.assertEqual(1.0, diagnostics["ramen_vs_oracle_id_cosine"][0])

    def test_drop_ood_variant_is_separately_named_and_reset_clears_hook_and_cache(self):
        self.assertTrue(OracleDropOODRamen.drop_ood_from_memory)
        method = object.__new__(OracleIDGradientRamen)
        method.cache = [self._cache()]
        method.cache[0].add(torch.tensor([[0.]]), torch.tensor([[1., 0.]]), torch.zeros(1),
                            torch.tensor([0.]), torch.tensor([False]))
        method.counter = 3
        method.model = type("Model", (), {"reset_parameters": lambda self: None})()
        method._initialize_oracle_ood_hook()
        method.set_oracle_is_ood(torch.tensor([True]))
        method.last_diagnostics = {}
        method.reset()
        self.assertEqual(0, method.counter)
        self.assertEqual(0, method.cache[0].size)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            method._consume_oracle_is_ood(1, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
