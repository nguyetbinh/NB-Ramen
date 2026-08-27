"""CPU mechanics for the clean, non-latent EntropyGatedRamen control."""

import sys
import unittest
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.EntropyGatedRamen import EntropyGatedRamen, validate_entropy_gated_ramen_config  # noqa: E402
from methods.Ramen import PriorityCache, Ramen  # noqa: E402


class _Model:
    def __init__(self, gradients):
        self._gradients = torch.as_tensor(gradients, dtype=torch.float32)
        self.set_gradients = None
        self.steps = 0
        self.resets = 0

    def featurize(self, x):
        return x

    def classify(self, features):
        # The first coordinate controls both prediction and confidence.
        return torch.stack((features[:, 0], -features[:, 0]), dim=1).requires_grad_()

    def get_by_sample_grad(self):
        return self._gradients.clone()

    def set_by_sample_grad(self, gradients):
        self.set_gradients = gradients.clone()

    def step_and_zero_grad(self):
        self.steps += 1

    def __call__(self, x):
        return torch.zeros((x.shape[0], 2))

    def reset_parameters(self):
        self.resets += 1


class EntropyGatedRamenTests(unittest.TestCase):
    @staticmethod
    def _method(gradients, threshold=.5):
        method = object.__new__(EntropyGatedRamen)
        method.device = torch.device("cpu")
        method.dtype = torch.float32
        method.cfg = {"topk": 1, "max_normalized_entropy": threshold}
        method.beta = 0.
        method.counter = 0
        method.cache = [PriorityCache(5, 1, 1, "cpu", torch.float32) for _ in range(2)]
        method.num_classes = 2
        method.model = _Model(gradients)
        method.loss_fn = lambda logits: logits.sum()
        method.last_diagnostics = {}
        return method

    def test_all_admitted_matches_ordinary_ramen(self):
        # Two supports in each predicted-class cache force top-k aggregation;
        # nonzero beta and unequal distances exercise both Ramen weights.
        x = torch.tensor([[2.4], [1.2], [-1.3], [-2.6]])
        gradients = [[2.], [4.], [6.], [8.]]
        gated = self._method(gradients)
        gated.cfg["topk"] = 2
        gated.beta = .3
        ordinary = object.__new__(Ramen)
        ordinary.device, ordinary.dtype = torch.device("cpu"), torch.float32
        ordinary.cfg, ordinary.beta, ordinary.counter = {"topk": 2}, .3, 0
        ordinary.cache = [PriorityCache(5, 1, 1, "cpu", torch.float32) for _ in range(2)]
        ordinary.num_classes, ordinary.model = 2, _Model(gradients)
        ordinary.loss_fn, ordinary.last_diagnostics = lambda logits: logits.sum(), {}

        gated.forward(x)
        ordinary.forward(x)
        self.assertTrue(torch.equal(gated.model.set_gradients, ordinary.model.set_gradients))
        self.assertEqual([2, 2], [cache.size for cache in gated.cache])
        self.assertEqual([cache.size for cache in gated.cache], [cache.size for cache in ordinary.cache])

    def test_all_rejected_with_empty_history_is_zero_update(self):
        method = self._method([[7.]])
        method.forward(torch.tensor([[0.]]))
        diagnostics = method.get_diagnostics()
        self.assertEqual([[0.]], method.model.set_gradients.tolist())
        self.assertEqual(0, diagnostics["memory_size"])
        self.assertEqual(0, diagnostics["memory_bytes"])
        self.assertFalse(bool(diagnostics["admitted_to_memory"][0]))

    def test_mixed_batch_admits_only_confident_items(self):
        method = self._method([[3.], [9.]])
        method.forward(torch.tensor([[10.], [0.]]))
        diagnostics = method.get_diagnostics()
        self.assertEqual([True, False], diagnostics["admitted_to_memory"].tolist())
        self.assertEqual(1, diagnostics["memory_size"])
        # The rejected query can use history, but its own gradient was absent.
        self.assertTrue(torch.allclose(method.model.set_gradients, torch.tensor([[3.], [3.]])))

    def test_admitted_current_item_can_self_retrieve(self):
        method = self._method([[7.]])
        method.forward(torch.tensor([[10.]]))
        self.assertTrue(torch.allclose(method.model.set_gradients, torch.tensor([[7.]])))

    def test_rejected_current_item_cannot_self_retrieve(self):
        method = self._method([[7.]])
        method.forward(torch.tensor([[10.]]))
        method.model._gradients = torch.tensor([[99.]])
        method.forward(torch.tensor([[0.]]))
        self.assertTrue(torch.allclose(method.model.set_gradients, torch.tensor([[7.]])))
        self.assertEqual(1, method.get_diagnostics()["memory_size"])

    def test_reset_clears_memory_counter_and_diagnostics(self):
        method = self._method([[7.]])
        method.forward(torch.tensor([[10.]]))
        method.reset()
        self.assertEqual(0, method.counter)
        self.assertEqual([0, 0], [cache.size for cache in method.cache])
        self.assertEqual({}, method.get_diagnostics())

    def test_is_label_isolated_and_has_no_oracle_context_hook(self):
        method = self._method([[1.], [2.]])
        method.forward(torch.tensor([[10.], [-10.]]))
        self.assertEqual([1, 1], [cache.size for cache in method.cache])
        self.assertFalse(hasattr(EntropyGatedRamen, "requires_oracle_ood_context"))
        self.assertFalse(hasattr(EntropyGatedRamen, "set_oracle_is_ood"))

    def test_configs_lock_threshold(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("CIFAR100C", "DomainNet"):
            cfg = yaml.safe_load((root / "cfg" / dataset / "EntropyGatedRamen.yaml").read_text())
            self.assertEqual(.5, validate_entropy_gated_ramen_config(cfg)["max_normalized_entropy"])

    def test_config_rejects_every_non_preregistered_threshold(self):
        base = {"max_capacity": 2, "topk": 1, "optimizer": "signsgd", "lr": .01}
        for value in (
            None, True, "0.5", float("nan"), float("inf"), float("-inf"),
            0, .49, .5000001, .6, 1,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_entropy_gated_ramen_config(
                        {**base, "max_normalized_entropy": value}
                    )


if __name__ == "__main__":
    unittest.main()
