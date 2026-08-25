"""CPU-only mechanics tests for ConsensusRamen-v0 and its v1 soft ablation."""

import sys
import unittest
from pathlib import Path

import torch
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from methods.ConsensusRamen import (  # noqa: E402
    ConsensusRamen,
    aggregate_consensus_supports,
    validate_consensus_ramen_config,
)
from methods.Ramen import PriorityCache  # noqa: E402
from models.optimizer import SignSGD  # noqa: E402


class ConsensusRamenTests(unittest.TestCase):
    def _cache(self, value_dim=2):
        return PriorityCache(8, 1, value_dim, "cpu", torch.float32)

    @staticmethod
    def _add(cache, gradient, key=0., entropy=0., priority=0.):
        cache.add(torch.tensor([[key]]), torch.tensor([gradient]), torch.tensor([entropy]), torch.tensor([priority]))

    def _aggregate(self, caches, **kwargs):
        return aggregate_consensus_supports(
            torch.tensor([[0.]]), caches, topk=1, beta=0., consensus_threshold=.6,
            min_consensus_classes=3, **kwargs,
        )

    def test_hard_mask_uses_coordinate_sign_consensus_over_class_balanced_gradients(self):
        caches = [self._cache() for _ in range(3)]
        self._add(caches[0], [2., 2.])
        self._add(caches[1], [4., 3.])
        self._add(caches[2], [-9., 6.])
        safe, ordinary, diagnostics = self._aggregate(caches)
        self.assertTrue(torch.allclose(ordinary, torch.tensor([[-1., 11. / 3.]])))
        self.assertTrue(torch.allclose(safe, torch.tensor([[0., 11. / 3.]])))
        self.assertEqual([3], diagnostics["consensus_active_class_count"].tolist())
        self.assertTrue(torch.allclose(diagnostics["consensus_mask_rate"], torch.tensor([.5])))
        self.assertEqual([True], diagnostics["consensus_applied"].tolist())

    def test_zero_signs_are_neutral_votes(self):
        caches = [self._cache(1) for _ in range(3)]
        self._add(caches[0], [5.])
        self._add(caches[1], [0.])
        self._add(caches[2], [-2.])
        safe, ordinary, _ = self._aggregate(caches)
        self.assertEqual([[1.]], ordinary.tolist())
        self.assertEqual([[0.]], safe.tolist())

    def test_soft_weight_uses_seeded_coordinate_admission_that_changes_signsgd_input(self):
        caches = [self._cache(3) for _ in range(3)]
        self._add(caches[0], [2., 2., 2.])
        self._add(caches[1], [4., 3., 3.])
        self._add(caches[2], [-9., 6., -6.])
        safe, ordinary, diagnostics = aggregate_consensus_supports(
            torch.tensor([[0.]]), caches, topk=1, beta=0., consensus_threshold=.6,
            min_consensus_classes=3, consensus_mode="soft_weight", consensus_gamma=1.,
            consensus_seed=0,
        )
        # Seed 0 generates [0.496..., 0.768..., 0.088...].  With q=[1/3,
        # 1,1/3], it suppresses only the first coordinate.  The retained
        # coordinates preserve ordinary Ramen's sign exactly.
        self.assertTrue(torch.allclose(ordinary, torch.tensor([[-1., 11. / 3., -1. / 3.]])))
        self.assertTrue(torch.allclose(safe, torch.tensor([[0., 11. / 3., -1. / 3.]])))
        self.assertTrue(torch.equal(safe != 0, torch.tensor([[False, True, True]])))
        self.assertTrue(torch.allclose(diagnostics["consensus_mask_rate"], torch.tensor([2. / 3.])))

        # This is deliberately optimizer-level: SignSGD would erase ordinary
        # magnitude scaling, but a zeroed admission coordinate changes its
        # actual parameter delta.
        plain_parameter = torch.nn.Parameter(torch.zeros(3))
        soft_parameter = torch.nn.Parameter(torch.zeros(3))
        plain_parameter.grad = ordinary.squeeze(0).clone()
        soft_parameter.grad = safe.squeeze(0).clone()
        SignSGD([plain_parameter], lr=.1).step()
        SignSGD([soft_parameter], lr=.1).step()
        self.assertTrue(torch.allclose(plain_parameter.detach(), torch.tensor([.1, -.1, .1])))
        self.assertTrue(torch.allclose(soft_parameter.detach(), torch.tensor([0., -.1, .1])))

        repeated, _, _ = aggregate_consensus_supports(
            torch.tensor([[0.]]), caches, topk=1, beta=0., consensus_threshold=.6,
            min_consensus_classes=3, consensus_mode="soft_weight", consensus_gamma=1.,
            consensus_seed=0,
        )
        self.assertTrue(torch.equal(safe, repeated))

    def test_soft_weight_zeros_neutral_consensus_and_keeps_below_minimum_fallback(self):
        caches = [self._cache(1) for _ in range(3)]
        self._add(caches[0], [5.])
        self._add(caches[1], [0.])
        self._add(caches[2], [-2.])
        safe, ordinary, diagnostics = aggregate_consensus_supports(
            torch.tensor([[0.]]), caches, topk=1, beta=0., consensus_threshold=.6,
            min_consensus_classes=3, consensus_mode="soft_weight", consensus_gamma=1.,
            consensus_seed=1729,
        )
        self.assertEqual([[1.]], ordinary.tolist())
        self.assertEqual([[0.]], safe.tolist())
        self.assertEqual([0.], diagnostics["consensus_mask_rate"].tolist())

        unanimous = [self._cache(1) for _ in range(3)]
        self._add(unanimous[0], [2.])
        self._add(unanimous[1], [3.])
        self._add(unanimous[2], [4.])
        safe, ordinary, diagnostics = aggregate_consensus_supports(
            torch.tensor([[0.]]), unanimous, topk=1, beta=0., consensus_threshold=.6,
            min_consensus_classes=3, consensus_mode="soft_weight", consensus_gamma=1.,
            consensus_seed=1729,
        )
        # q=1 is admitted for every seed; q=0 above is excluded for every seed.
        self.assertTrue(torch.equal(safe, ordinary))
        self.assertEqual([1.], diagnostics["consensus_mask_rate"].tolist())

        sparse = [self._cache(1) for _ in range(3)]
        self._add(sparse[0], [2.])
        self._add(sparse[1], [-9.])
        safe, ordinary, diagnostics = aggregate_consensus_supports(
            torch.tensor([[0.]]), sparse, topk=1, beta=0., consensus_threshold=.6,
            min_consensus_classes=3, consensus_mode="soft_weight", consensus_gamma=1.,
            consensus_seed=1729,
        )
        self.assertTrue(torch.equal(safe, ordinary))
        self.assertEqual([False], diagnostics["consensus_applied"].tolist())

    def test_empty_history_has_ramen_empty_support_update(self):
        safe, ordinary, diagnostics = self._aggregate([self._cache(), self._cache(), self._cache()])
        self.assertEqual([[0., 0.]], ordinary.tolist())
        self.assertEqual([[0., 0.]], safe.tolist())
        self.assertEqual([0], diagnostics["consensus_active_class_count"].tolist())

    def test_below_minimum_classes_falls_back_exactly_to_ordinary_ramen(self):
        caches = [self._cache(), self._cache(), self._cache()]
        self._add(caches[0], [2., -3.])
        self._add(caches[1], [4., 5.])
        safe, ordinary, diagnostics = self._aggregate(caches)
        self.assertEqual([[3., 1.]], ordinary.tolist())
        self.assertTrue(torch.equal(safe, ordinary))
        self.assertEqual([1.], diagnostics["consensus_mask_rate"].tolist())
        self.assertEqual([False], diagnostics["consensus_applied"].tolist())

    def test_retrieval_and_weighted_aggregate_match_ramen_when_consensus_is_bypassed(self):
        caches = [self._cache(1) for _ in range(3)]
        self._add(caches[0], [2.], key=0., entropy=0.)
        self._add(caches[0], [9.], key=3., entropy=0., priority=1.)
        self._add(caches[1], [4.], key=0., entropy=0.)
        safe, ordinary, _ = aggregate_consensus_supports(
            torch.tensor([[0.]]), caches, topk=1, beta=1., consensus_threshold=.6, min_consensus_classes=3,
        )
        # Ramen retrieves [2] and [4] as nearest class supports, then averages
        # those two class contributions; insufficient classes bypass masking.
        self.assertEqual([[3.]], ordinary.tolist())
        self.assertTrue(torch.equal(safe, ordinary))

    def test_batch_atomic_current_visibility_keeps_same_batch_items_retrievable(self):
        cache = self._cache(1)
        cache.add(torch.tensor([[0.], [10.]]), torch.tensor([[1.], [9.]]), torch.zeros(2), torch.tensor([0., 1.]))
        safe, ordinary, _ = aggregate_consensus_supports(
            torch.tensor([[0.], [10.]]), [cache], topk=1, beta=0., consensus_threshold=.6,
            min_consensus_classes=3,
        )
        self.assertEqual([[1.], [9.]], ordinary.tolist())
        self.assertTrue(torch.equal(safe, ordinary))

    def test_no_self_uses_only_history_before_admitting_current_batch(self):
        """The current support is closer, so its absence proves causal order."""
        class ForwardModel:
            def __init__(self):
                self.safe_gradients = None

            def featurize(self, x):
                return x

            def classify(self, features):
                # Near-zero entropy keeps the deliberately simple support
                # arithmetic below free from Ramen's entropy weight.
                return torch.tensor([[100., -100.]], requires_grad=True)

            def get_by_sample_grad(self):
                return torch.tensor([[11.]])

            def set_by_sample_grad(self, gradients):
                self.safe_gradients = gradients.clone()

            def step_and_zero_grad(self):
                pass

            def __call__(self, x):
                return torch.zeros((x.shape[0], 2))

            def reset_parameters(self):
                pass

        method = object.__new__(ConsensusRamen)
        method.cfg = {
            "topk": 1, "consensus_threshold": .2,
            "min_consensus_classes": 3, "consensus_mode": "hard_mask",
            "include_current": False,
        }
        method.beta = 0.
        method.device = "cpu"
        method.dtype = torch.float32
        method.counter = 1
        method.model = ForwardModel()
        method.loss_fn = lambda logits: logits.sum()
        method.cache = [self._cache(1), self._cache(1)]
        # The only historical support is deliberately farther from query 0
        # than the current support would be.  If it were admitted before
        # retrieval, top-1 would return 11 rather than 7.
        self._add(method.cache[0], [7.], key=10., priority=0.)
        method.last_diagnostics = {}

        method.forward(torch.tensor([[0.]]))

        self.assertEqual([[7.]], method.model.safe_gradients.tolist())
        self.assertEqual(2, method.cache[0].size)
        self.assertEqual([[10.], [0.]], method.cache[0].keys[:2].tolist())

    def test_current_visibility_v0_regression_admits_before_retrieval(self):
        """The same arithmetic confirms explicit v0 keeps current support."""
        class ForwardModel:
            def __init__(self):
                self.safe_gradients = None

            def featurize(self, x): return x
            def classify(self, features): return torch.tensor([[100., -100.]], requires_grad=True)
            def get_by_sample_grad(self): return torch.tensor([[11.]])
            def set_by_sample_grad(self, gradients): self.safe_gradients = gradients.clone()
            def step_and_zero_grad(self): pass
            def __call__(self, x): return torch.zeros((x.shape[0], 2))
            def reset_parameters(self): pass

        method = object.__new__(ConsensusRamen)
        method.cfg = {
            "topk": 1, "consensus_threshold": .2,
            "min_consensus_classes": 3, "consensus_mode": "hard_mask",
            "include_current": True,
        }
        method.beta = 0.
        method.device = "cpu"
        method.dtype = torch.float32
        method.counter = 1
        method.model = ForwardModel()
        method.loss_fn = lambda logits: logits.sum()
        method.cache = [self._cache(1), self._cache(1)]
        self._add(method.cache[0], [7.], key=10., priority=0.)
        method.last_diagnostics = {}

        method.forward(torch.tensor([[0.]]))

        self.assertEqual([[11.]], method.model.safe_gradients.tolist())

    def test_reset_clears_cache_counter_and_diagnostics(self):
        method = object.__new__(ConsensusRamen)
        method.cache = [self._cache()]
        self._add(method.cache[0], [1., 2.])
        method.counter = 4
        method.last_diagnostics = {"pre_adaptation_ood_score": torch.tensor([1.])}
        method.model = type("Model", (), {"reset_parameters": lambda self: None})()
        method.reset()
        self.assertEqual(0, method.cache[0].size)
        self.assertEqual(0, method.counter)
        self.assertEqual({}, method.last_diagnostics)

    def test_memory_bytes_follow_retained_ramen_supports(self):
        method = object.__new__(ConsensusRamen)
        method.cache = [self._cache()]
        self._add(method.cache[0], [1., 2.])
        cache = method.cache[0]
        expected = 1 * (
            cache.keys.shape[1] * cache.keys.element_size()
            + cache.values.shape[1] * cache.values.element_size()
            + cache.priorities.element_size()
            + cache.entropies.element_size()
        )
        self.assertEqual(expected, method.memory_bytes)

    def test_config_matches_preregistered_v0_surface_and_method_has_no_evaluator_fields(self):
        root = Path(__file__).resolve().parents[1]
        cfg = validate_consensus_ramen_config(yaml.safe_load((root / "cfg/CIFAR100C/ConsensusRamen.yaml").read_text()))
        self.assertEqual(.2, cfg["consensus_threshold"])
        self.assertEqual(3, cfg["min_consensus_classes"])
        self.assertEqual("hard_mask", cfg["consensus_mode"])
        self.assertTrue(cfg["include_current"])
        self.assertFalse(hasattr(ConsensusRamen, "requires_oracle_ood_context"))
        self.assertFalse(hasattr(ConsensusRamen, "set_oracle_is_ood"))

    def test_soft_config_is_preregistered_and_requires_valid_mode_specific_gamma(self):
        root = Path(__file__).resolve().parents[1]
        cfg = validate_consensus_ramen_config(
            yaml.safe_load((root / "cfg/CIFAR100C/ConsensusRamenSoft.yaml").read_text())
        )
        self.assertEqual("soft_weight", cfg["consensus_mode"])
        self.assertEqual(1., cfg["consensus_gamma"])
        self.assertEqual(1729, cfg["consensus_seed"])

        base = yaml.safe_load((root / "cfg/CIFAR100C/ConsensusRamen.yaml").read_text())
        base["consensus_mode"] = "soft_weight"
        with self.assertRaisesRegex(ValueError, "consensus_gamma"):
            validate_consensus_ramen_config(base)
        base["consensus_gamma"] = 0.
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_consensus_ramen_config(base)
        base["consensus_gamma"] = 1.
        with self.assertRaisesRegex(ValueError, "consensus_seed"):
            validate_consensus_ramen_config(base)
        base["consensus_seed"] = -1
        with self.assertRaisesRegex(ValueError, "consensus_seed"):
            validate_consensus_ramen_config(base)
        base["consensus_seed"] = 1729
        base["consensus_mode"] = "not-a-mode"
        with self.assertRaisesRegex(ValueError, "consensus_mode"):
            validate_consensus_ramen_config(base)
        base["consensus_mode"] = "hard_mask"
        base["include_current"] = "false"
        with self.assertRaisesRegex(ValueError, "include_current"):
            validate_consensus_ramen_config(base)

    def test_no_self_config_is_explicit_and_preserves_v0_hyperparameters(self):
        root = Path(__file__).resolve().parents[1]
        base = validate_consensus_ramen_config(
            yaml.safe_load((root / "cfg/CIFAR100C/ConsensusRamen.yaml").read_text())
        )
        no_self = validate_consensus_ramen_config(
            yaml.safe_load((root / "cfg/CIFAR100C/ConsensusRamenNoSelf.yaml").read_text())
        )
        self.assertFalse(no_self["include_current"])
        for key in ("max_capacity", "topk", "beta", "optimizer", "lr", "consensus_threshold", "min_consensus_classes", "consensus_mode"):
            self.assertEqual(base[key], no_self[key])


if __name__ == "__main__":
    unittest.main()
