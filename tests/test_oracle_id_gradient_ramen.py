"""CPU-only mechanics tests for evaluator-only oracle gradient controls."""

import sys
import json
import math
import tempfile
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
from main import _method_diagnostics, _oracle_gradient_summary  # noqa: E402
from evaluation.evidence import JsonlTraceWriter, ORACLE_GRADIENT_TRACE_FIELDS  # noqa: E402


class _Hook(OracleOODContextHook):
    def __init__(self):
        self._initialize_oracle_ood_hook()


class _ForwardModel:
    feat_dim = 1
    grad_dim = 2
    dtype = torch.float32

    def __init__(self):
        self.logits = torch.tensor(
            [[3., 1., 0.], [0., 3., 1.], [1., 0., 3.]], requires_grad=True
        )
        self.gradients = torch.tensor([[1., 0.], [1., 2.], [1., -1.]])
        self.applied_gradient = None

    def featurize(self, inputs):
        return inputs

    def classify(self, _features):
        return self.logits

    def get_by_sample_grad(self):
        return self.gradients

    def set_by_sample_grad(self, gradient):
        self.applied_gradient = gradient.detach().clone()

    def step_and_zero_grad(self):
        self.logits.grad = None

    def __call__(self, _inputs):
        return self.logits.detach()

    def reset_parameters(self):
        pass


class OracleIDGradientRamenTests(unittest.TestCase):
    def _cache(self):
        return OraclePriorityCache(4, 1, 2, "cpu", torch.float32)

    def _forward_method(self, method_type):
        method = object.__new__(method_type)
        method.cfg = {"topk": 1, "beta": 0.0}
        method.num_classes = 3
        method.device = torch.device("cpu")
        method.dtype = torch.float32
        method.model = _ForwardModel()
        method.cache = [self._cache() for _ in range(3)]
        method.loss_fn = lambda logits: logits.sum()
        method.counter = 0
        method._initialize_oracle_ood_hook()
        method.last_diagnostics = method._diagnostics()
        method.set_oracle_is_ood(torch.tensor([False, True, True]))
        method.forward(torch.tensor([[0.], [0.], [0.]]))
        return method

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
        all_gradient, consensus_gradient, id_gradient, diagnostics = aggregate_oracle_supports(
            torch.tensor([[0.]]), [cache], topk=2, beta=0.
        )
        self.assertEqual([[1., 2.]], all_gradient.tolist())
        self.assertEqual([[1., 0.]], id_gradient.tolist())
        self.assertEqual(all_gradient.tolist(), consensus_gradient.tolist())
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
        _, _, first_id, first = aggregate_oracle_supports(torch.tensor([[0.]]), [cache], topk=2, beta=0.)
        # Add the future stream item only after evaluating the first item.
        cache.add(torch.tensor([[0.]]), torch.tensor([[0., 2.]]), torch.zeros(1),
                  torch.tensor([1.]), torch.tensor([True]))
        _, _, second_id, second = aggregate_oracle_supports(torch.tensor([[0.]]), [cache], topk=2, beta=0.)
        self.assertEqual([[1., 0.]], first_id.tolist())
        self.assertEqual([[1., 0.]], second_id.tolist())
        self.assertEqual([0.0], first["retrieved_ood_fraction"].tolist())
        self.assertEqual([0.5], second["retrieved_ood_fraction"].tolist())

    def test_zero_id_direction_has_undefined_direction_metrics(self):
        cache = self._cache()
        cache.add(torch.tensor([[0.]]), torch.tensor([[0., 2.]]), torch.zeros(1),
                  torch.tensor([0.]), torch.tensor([True]))
        _, _, id_gradient, diagnostics = aggregate_oracle_supports(torch.tensor([[0.]]), [cache], topk=1, beta=0.)
        self.assertEqual([[0., 0.]], id_gradient.tolist())
        self.assertIsNone(diagnostics["ramen_vs_oracle_id_cosine"][0])
        self.assertIsNone(diagnostics["ramen_vs_oracle_id_sign_disagreement"][0])

    def test_empty_supports_have_undefined_direction_metrics(self):
        _, _, id_gradient, diagnostics = aggregate_oracle_supports(
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
        _, _, _, diagnostics = aggregate_oracle_supports(
            torch.tensor([[0.]]), [cache], topk=1, beta=0.
        )
        self.assertEqual(1.0, diagnostics["ramen_vs_oracle_id_cosine"][0])

    def test_consensus_mask_is_independent_of_ood_flags_and_improves_known_sdr(self):
        caches = [self._cache() for _ in range(3)]
        gradients = ([torch.tensor([[1., 0.]])], [torch.tensor([[1., 2.]])], [torch.tensor([[1., -1.]])])
        for index, (cache, values) in enumerate(zip(caches, gradients)):
            cache.add(torch.tensor([[0.]]), values[0], torch.zeros(1), torch.tensor([0.]), torch.tensor([index > 0]))
        ramen, consensus, oracle, diagnostics = aggregate_oracle_supports(torch.tensor([[0.]]), caches, topk=1, beta=0.)
        self.assertEqual([[1., 0.]], consensus.tolist())
        self.assertLess(diagnostics["consensus_vs_oracle_id_sign_disagreement"][0], diagnostics["ramen_vs_oracle_id_sign_disagreement"][0])
        for cache in caches:
            cache.is_ood[:cache.size] = True
        _, changed_consensus, _, changed = aggregate_oracle_supports(torch.tensor([[0.]]), caches, topk=1, beta=0.)
        self.assertEqual(consensus.tolist(), changed_consensus.tolist())
        self.assertEqual(diagnostics["consensus_diagnostic_mask_rate"], changed["consensus_diagnostic_mask_rate"])

    def test_hard_mask_reports_wrong_and_correct_sign_removal_separately(self):
        caches = [self._cache() for _ in range(3)]
        # Coordinate 0 is correct and unanimous. Coordinate 1 is wrong in the
        # ordinary aggregate, while (+, -, 0) yields zero consensus agreement
        # and therefore suppresses that harmful update.
        values = ([1., 1.], [1., -4.], [1., 0.])
        flags = (False, True, True)
        for cache, gradient, is_ood in zip(caches, values, flags):
            cache.add(
                torch.tensor([[0.]]), torch.tensor([gradient]), torch.zeros(1),
                torch.tensor([0.]), torch.tensor([is_ood]),
            )
        ramen, consensus, oracle, diagnostics = aggregate_oracle_supports(
            torch.tensor([[0.]]), caches, topk=1, beta=0.
        )
        self.assertEqual([[1., -1.]], ramen.tolist())
        self.assertEqual([[1., 0.]], consensus.tolist())
        torch.testing.assert_close(oracle, torch.tensor([[1. / 3., 1. / 3.]]))
        self.assertEqual([1], diagnostics["consensus_wrong_sign_coordinate_count"].tolist())
        self.assertEqual([1], diagnostics["consensus_wrong_sign_removed_count"].tolist())
        self.assertEqual([1.0], diagnostics["consensus_wrong_sign_removal_rate"])
        self.assertEqual([1], diagnostics["consensus_correct_sign_coordinate_count"].tolist())
        self.assertEqual([0], diagnostics["consensus_correct_sign_removed_count"].tolist())
        self.assertEqual([0.0], diagnostics["consensus_correct_sign_removal_rate"])

    def test_forward_diagnostics_survive_evaluator_trace_and_summary_plumbing(self):
        f1_fields = (
            "consensus_vs_oracle_id_cosine",
            "consensus_vs_oracle_id_sign_disagreement",
            "consensus_vs_ramen_cosine",
            "consensus_diagnostic_mask_rate",
            "consensus_diagnostic_applied",
        )
        for method_type in (OracleIDGradientRamen, OracleDropOODRamen):
            with self.subTest(method=method_type.__name__):
                method = self._forward_method(method_type)
                expanded = _method_diagnostics(method, 3)
                rows = []
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "trace.jsonl"
                    with JsonlTraceWriter(path, "integration") as writer:
                        for index in range(3):
                            row = {
                                "timestep": index, "sample_idx": index,
                                "ground_truth_domain": 0, "ground_truth_class": index,
                                "prediction": index, "correct": True,
                                "predicted_entropy": 0.1, "inferred_context": None,
                                "memory_size": expanded["memory_size"][index],
                                "num_active_contexts": None,
                                "memory_bytes": expanded["memory_bytes"][index],
                                "latency_ms": 1.0, "original_label": index,
                                "known_label_or_minus_one": index,
                                "is_ood": False, "open_set_split_version": "integration",
                                "ood_ratio": 0.0,
                                "pre_adaptation_prediction": expanded["pre_adaptation_prediction"][index],
                                "pre_adaptation_ood_score": expanded["pre_adaptation_ood_score"][index],
                                "post_adaptation_ood_score": expanded["pre_adaptation_ood_score"][index],
                            }
                            for field in ORACLE_GRADIENT_TRACE_FIELDS:
                                row[field] = expanded[field][index]
                            rows.append(writer.write(row))
                    persisted = [json.loads(line) for line in path.read_text().splitlines()]
                for row in persisted:
                    for field in f1_fields[:4]:
                        self.assertIsNotNone(row[field])
                        self.assertTrue(math.isfinite(row[field]))
                    self.assertIsInstance(row[f1_fields[4]], bool)
                    for field in (
                        "consensus_wrong_sign_coordinate_count",
                        "consensus_wrong_sign_removed_count",
                        "consensus_correct_sign_coordinate_count",
                        "consensus_correct_sign_removed_count",
                    ):
                        self.assertIsInstance(row[field], int)
                summary = _oracle_gradient_summary(rows)
                for field in (
                    "ramen_gdc_mean", "consensus_gdc_mean", "ramen_sdr_mean",
                    "consensus_sdr_mean", "gdc_reduction_mean", "sdr_reduction_mean",
                ):
                    self.assertIsNotNone(summary[field])
                    self.assertTrue(math.isfinite(summary[field]))
                for field in (
                    "consensus_wrong_sign_removal_rate",
                    "consensus_correct_sign_removal_rate",
                ):
                    if summary[field] is not None:
                        self.assertTrue(math.isfinite(summary[field]))
                entropy_weight = float(torch.exp(-method.cache[0].entropies[0]))
                expected_x = entropy_weight if method_type is OracleDropOODRamen else entropy_weight / 3.0
                torch.testing.assert_close(
                    method.model.applied_gradient,
                    torch.tensor([[expected_x, 0.]]).expand(3, 2),
                )

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
