"""CPU-only mechanics tests for the evaluator-only OracleConsensusRamen."""

import sys
import unittest
from pathlib import Path

import torch
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.OracleConsensusRamen import (  # noqa: E402
    OracleConsensusRamen,
    validate_oracle_consensus_ramen_config,
)
from methods.Ramen import PriorityCache  # noqa: E402


class OracleConsensusRamenTests(unittest.TestCase):
    def _cache(self):
        return PriorityCache(4, 1, 2, "cpu", torch.float32)

    def test_config_requires_explicit_evaluator_provenance(self):
        base = {
            "max_capacity": 2, "topk": 1, "optimizer": "signsgd", "lr": .01,
            "consensus_threshold": .2, "min_consensus_classes": 1, "consensus_mode": "hard_mask",
        }
        with self.assertRaises(ValueError):
            validate_oracle_consensus_ramen_config(base)
        cfg = validate_oracle_consensus_ramen_config({**base, "oracle_ood_source": "evaluator_is_ood"})
        self.assertEqual("evaluator_is_ood", cfg["oracle_ood_source"])

    def test_oracle_hook_fails_closed_and_is_single_use(self):
        method = object.__new__(OracleConsensusRamen)
        method._initialize_oracle_ood_hook()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            method._consume_oracle_is_ood(1, torch.device("cpu"))
        method.set_oracle_is_ood(torch.tensor([False, True]))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            method.set_oracle_is_ood(torch.tensor([False, True]))
        self.assertEqual([False, True], method._consume_oracle_is_ood(2, torch.device("cpu")).tolist())
        with self.assertRaisesRegex(RuntimeError, "missing"):
            method._consume_oracle_is_ood(2, torch.device("cpu"))

    def test_ood_support_is_not_retained_or_used_as_consensus_vote(self):
        cache = self._cache()
        # This models forward admission: the evaluator-known OOD item is
        # skipped rather than stored with an oracle flag for later use.
        id_gradient = torch.tensor([[1., 0.]])
        ood_gradient = torch.tensor([[0., 9.]])
        cache.add(torch.tensor([[0.]]), id_gradient, torch.zeros(1), torch.zeros(1))
        self.assertEqual(1, cache.size)
        self.assertTrue(torch.equal(torch.tensor([[1., 0.]]), cache.values[:cache.size]))
        self.assertFalse(torch.equal(ood_gradient, cache.values[:cache.size]))

    def test_forward_admits_only_id_items_from_an_evaluator_labeled_batch(self):
        class _Model:
            def __init__(self):
                self.applied = None

            def featurize(self, x):
                return x

            def classify(self, features):
                # Both examples predict the same class, so an admitted OOD
                # example would be directly observable in the sole cache.
                return torch.stack((features[:, 0], -features[:, 0]), dim=1).requires_grad_()

            def get_by_sample_grad(self):
                return torch.tensor([[1., 0.], [0., 9.]])

            def set_by_sample_grad(self, gradients):
                self.applied = gradients.clone()

            def step_and_zero_grad(self):
                pass

            def __call__(self, x):
                return torch.zeros((x.shape[0], 2))

            def reset_parameters(self):
                pass

        method = object.__new__(OracleConsensusRamen)
        method.cfg = {"topk": 1, "consensus_threshold": .2, "min_consensus_classes": 1}
        method.beta = 0.
        method.device = torch.device("cpu")
        method.dtype = torch.float32
        method.counter = 0
        method.cache = [self._cache(), self._cache()]
        method.model = _Model()
        method.loss_fn = lambda logits: logits.sum()
        method._initialize_oracle_ood_hook()
        method.set_oracle_is_ood(torch.tensor([False, True]))

        method.forward(torch.tensor([[2.], [1.]]))

        self.assertEqual(1, method.cache[0].size)
        self.assertEqual(0, method.cache[1].size)
        self.assertEqual([[1., 0.]], method.cache[0].values[:1].tolist())
        self.assertTrue(torch.all(method.model.applied[:, 0] > 0))
        self.assertTrue(torch.equal(torch.zeros(2), method.model.applied[:, 1]))

    def test_reset_clears_cache_counter_diagnostics_and_pending_oracle_context(self):
        method = object.__new__(OracleConsensusRamen)
        method.cache = [self._cache()]
        method.cache[0].add(torch.tensor([[0.]]), torch.tensor([[1., 0.]]), torch.zeros(1), torch.zeros(1))
        method.counter = 3
        method.model = type("Model", (), {"reset_parameters": lambda self: None})()
        method.last_diagnostics = {"memory_bytes": torch.tensor([12])}
        method._initialize_oracle_ood_hook()
        method.set_oracle_is_ood(torch.tensor([True]))
        method.reset()
        self.assertEqual(0, method.cache[0].size)
        self.assertEqual(0, method.counter)
        self.assertEqual({}, method.last_diagnostics)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            method._consume_oracle_is_ood(1, torch.device("cpu"))

    def test_plain_consensus_method_has_no_oracle_label_hook(self):
        from methods.ConsensusRamen import ConsensusRamen

        self.assertFalse(hasattr(ConsensusRamen, "requires_oracle_ood_context"))
        self.assertFalse(hasattr(ConsensusRamen, "set_oracle_is_ood"))
        self.assertTrue(OracleConsensusRamen.requires_oracle_ood_context)
        self.assertFalse(OracleConsensusRamen.emits_oracle_gradient_diagnostics)

    def test_preregistered_config_is_valid(self):
        root = Path(__file__).resolve().parents[1]
        cfg = validate_oracle_consensus_ramen_config(
            yaml.safe_load((root / "cfg/CIFAR100C/OracleConsensusRamen.yaml").read_text())
        )
        self.assertEqual(.2, cfg["consensus_threshold"])
        self.assertEqual("evaluator_is_ood", cfg["oracle_ood_source"])


if __name__ == "__main__":
    unittest.main()
