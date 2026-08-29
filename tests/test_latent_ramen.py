import unittest
import sys
from pathlib import Path

import yaml

import torch

# Application modules intentionally use ``src`` as their import root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.memory.structured_memory import StructuredGradientMemory
from methods.LatentRamen import (
    LatentRamen,
    aggregate_class_balanced_gradients,
    update_and_retrieve_causal_batch,
    update_and_retrieve_profiled_causal_batch,
    validate_latent_ramen_config,
    _profile_elapsed_tensor,
    _counterfactual_thresholds,
    evaluate_replay_counterfactuals,
)
from methods.EntropyGatedLatentRamen import (
    update_and_retrieve_entropy_gated_causal_batch,
    validate_entropy_gated_latent_ramen_config,
)


class LatentRamenPureTests(unittest.TestCase):
    def test_replay_counterfactuals_reset_between_fixed_masks(self):
        class FakeModel:
            def __init__(self):
                self.reset_calls = 0
                self.gradients = None
                self.events = []
            def reset_parameters(self):
                self.reset_calls += 1
                self.gradients = None
                self.events.append("reset")
            def featurize(self, x):
                self.events.append("featurize")
                return x
            def set_by_sample_grad(self, gradients):
                self.gradients = gradients.clone()
                self.events.append("set")
            def step_and_zero_grad(self):
                self.events.append("step")
            def __call__(self, x):
                # A kept first coordinate predicts class 1; a masked one class 0.
                self.events.append("predict")
                return torch.stack((1 - self.gradients[:, 0], self.gradients[:, 0]), dim=1)

        model = FakeModel()
        entries = [{"_consensus_strength": torch.tensor([.6, .2])}]
        actual = torch.tensor([[1., 1.]])
        actual_logits = torch.tensor([[0., 1.]])
        returned_logits = actual_logits.clone()
        evaluate_replay_counterfactuals(
            model, torch.zeros(1, 1), actual_logits, actual, entries, (.5, .75, 1.)
        )
        torch.testing.assert_close(actual_logits, returned_logits)
        variants = entries[0]["counterfactuals"]
        self.assertEqual([.5, .75, 1.], [item["threshold"] for item in variants])
        self.assertEqual([1, 0, 0], [item["prediction"].item() for item in variants])
        self.assertTrue(entries[0]["reset_state_verified"])
        self.assertEqual(5, model.reset_calls)  # parity replay, then each variant
        self.assertEqual("reset", model.events[-1])

    def test_replay_counterfactuals_fail_closed_when_reset_cannot_rebuild_actual(self):
        class BadResetModel:
            def reset_parameters(self): pass
            def featurize(self, x): return x
            def set_by_sample_grad(self, gradients): self.gradients = gradients
            def step_and_zero_grad(self): pass
            def __call__(self, x): return torch.tensor([[3., 0.]])

        entries = [{"_consensus_strength": torch.tensor([1.])}]
        evaluate_replay_counterfactuals(
            BadResetModel(), torch.zeros(1, 1), torch.tensor([[0., 3.]]),
            torch.tensor([[1.]]), entries, (.5, .75, 1.)
        )
        self.assertFalse(entries[0]["reset_state_verified"])
        self.assertEqual([], entries[0]["counterfactuals"])

    def test_method_threshold_parser_rejects_noncanonical_tuple(self):
        class Args:
            failure_counterfactual_thresholds = (.5,)

        with self.assertRaisesRegex(ValueError, "preregistered tuple"):
            _counterfactual_thresholds(Args())

    def test_aggregation_preserves_ramen_weights_and_class_balance(self):
        memory = StructuredGradientMemory(3, 4, 2, 1, device="cpu")
        memory.add(torch.tensor([[0., 0.], [1., 0.], [0., 2.]]), torch.tensor([[2.], [5.], [7.]]),
                   torch.tensor([0, 0, 1]), torch.tensor([0, 0, 0]), torch.tensor([0., 0., 0.]))
        result = memory.query(torch.tensor([[0., 0.]]), 0, topk=2)
        gradients, counts = aggregate_class_balanced_gradients(result, beta=1.0)
        # class 0 contributes 2 + 5e^-1; class 1 contributes 7e^-2; the
        # method averages the two active predicted classes, as Ramen does.
        expected = (2 + 5 * torch.exp(torch.tensor(-1.)) + 7 * torch.exp(torch.tensor(-2.))) / 2
        self.assertTrue(torch.allclose(gradients.squeeze(), expected))
        self.assertEqual([2], counts.tolist())

    def test_empty_ablation_support_produces_no_update(self):
        memory = StructuredGradientMemory(2, 2, 1, 1, device="cpu")
        memory.add(torch.tensor([[0.]]), torch.tensor([[9.]]), 0, 0, 0., item_ids=42)
        result = memory.query(torch.tensor([[0.]]), 0, 1, include_current=False, current_item_ids=42)
        gradients, counts = aggregate_class_balanced_gradients(result, beta=0.)
        self.assertEqual([0], counts.tolist())
        self.assertEqual([[0.]], gradients.tolist())

    def test_replay_payload_records_causal_schedule_and_canonical_conflict_metric(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu")
        *_, payloads = update_and_retrieve_causal_batch(
            memory, torch.tensor([[0.]]), torch.tensor([[1.]]), torch.tensor([0]), torch.tensor([0]),
            torch.tensor([0.]), torch.tensor([0]), topk=1, include_current=True, beta=0.,
            failure_analysis_profile="replay_v1",
        )
        self.assertEqual("causal", payloads[0]["schedule"])
        self.assertEqual("fraction_low_consensus_coordinates_v1", payloads[0]["conflict_metric"])
        self.assertIsInstance(payloads[0]["conflict"], torch.Tensor)

    def test_config_validation_defaults_and_rejects_invalid_ablation_flag(self):
        cfg = validate_latent_ramen_config({"max_capacity": 2, "topk": 1, "optimizer": "signsgd", "lr": .01})
        self.assertTrue(cfg["include_current"])
        self.assertEqual(8, cfg["max_contexts"])
        self.assertEqual("per_class_context", cfg["capacity_scope"])
        self.assertEqual("off", cfg["retrieval_profile"])
        with self.assertRaises(ValueError):
            validate_latent_ramen_config({**cfg, "include_current": 1})
        with self.assertRaises(ValueError):
            validate_latent_ramen_config({**cfg, "capacity_scope": "unknown"})
        with self.assertRaises(ValueError):
            validate_latent_ramen_config({**cfg, "retrieval_profile": "timed"})

    def test_all_latent_ramen_configs_use_per_class_capacity_scope(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.glob("cfg/*/LatentRamen.yaml"):
            with path.open() as config_file:
                cfg = validate_latent_ramen_config(yaml.safe_load(config_file))
            self.assertEqual("per_class", cfg["capacity_scope"], path)

    def test_entropy_gate_configs_only_add_preregistered_threshold(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("CIFAR100C", "DomainNet", "smoke/CIFAR100C"):
            gated_path = root / "cfg" / dataset / "EntropyGatedLatentRamen.yaml"
            latent_path = root / "cfg" / dataset / "LatentRamen.yaml"
            gated = yaml.safe_load(gated_path.read_text())
            latent = yaml.safe_load(latent_path.read_text())
            self.assertEqual(.5, gated.pop("max_normalized_entropy"))
            self.assertEqual(latent, gated)

    def test_public_diagnostics_match_evidence_contract(self):
        method = object.__new__(LatentRamen)
        method.last_diagnostics = {
            "inferred_context": torch.tensor([2]),
            "memory_size": 9,
            "num_active_contexts": 3,
        }
        diagnostics = method.get_diagnostics()
        self.assertEqual(9, diagnostics["memory_size"])
        self.assertEqual(3, diagnostics["num_active_contexts"])
        self.assertEqual([2], diagnostics["inferred_context"].tolist())
        diagnostics["memory_size"] = 0
        self.assertEqual(9, method.last_diagnostics["memory_size"])

    def test_method_diagnostics_use_lightweight_memory_accessors(self):
        class Router:
            num_contexts = 2

        method = object.__new__(LatentRamen)
        method.router = Router()
        method.memory = StructuredGradientMemory(1, 2, 1, 1, device="cpu")
        method.memory.add(torch.tensor([[0.]]), torch.tensor([[1.]]), 0, 3, 0.)
        method.memory.diagnostics = lambda: self.fail("full memory diagnostics scanned in hot path")
        diagnostics = method._diagnostics()
        self.assertEqual(1, diagnostics["memory_size"])
        self.assertEqual(1, diagnostics["memory_active_contexts"])
        self.assertEqual(32, diagnostics["memory_bytes"])
        self.assertEqual(2, diagnostics["memory_max_capacity"])

    def test_batch_retrieval_never_reads_future_items(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu")
        retrieved, active_classes, memory_sizes, memory_bytes = update_and_retrieve_causal_batch(
            memory,
            features=torch.tensor([[0.], [10.]]),
            gradients=torch.tensor([[1.], [2.]]),
            predicted_classes=torch.tensor([0, 0]),
            contexts=torch.tensor([0, 0]),
            entropies=torch.tensor([0., 0.]),
            item_ids=torch.tensor([0, 1]),
            topk=1,
            include_current=False,
            beta=0.,
        )
        self.assertEqual([[0.], [1.]], retrieved.tolist())
        self.assertEqual([0, 1], active_classes.tolist())
        self.assertEqual([1, 2], memory_sizes.tolist())
        self.assertEqual([32, 64], memory_bytes.tolist())

    def test_profiled_causal_batch_preserves_supports_and_reports_causal_counts(self):
        kwargs = dict(
            features=torch.tensor([[0.], [10.]]), gradients=torch.tensor([[1.], [2.]]),
            predicted_classes=torch.tensor([0, 0]), contexts=torch.tensor([0, 0]),
            entropies=torch.tensor([0., 0.]), item_ids=torch.tensor([0, 1]),
            topk=1, include_current=False, beta=0.,
        )
        ordinary = update_and_retrieve_causal_batch(StructuredGradientMemory(1, 4, 1, 1, device="cpu"), **kwargs)
        profiled = update_and_retrieve_profiled_causal_batch(StructuredGradientMemory(1, 4, 1, 1, device="cpu"), **kwargs)
        for actual, expected in zip(profiled[:4], ordinary):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual([1, 2], profiled[5].tolist())
        self.assertEqual([0, 1], profiled[6].tolist())
        self.assertEqual([0, 1], profiled[7].tolist())
        self.assertTrue(bool((profiled[4] >= 0).all()))
        self.assertEqual("cpu", profiled[4].device.type)

    def test_profile_elapsed_timestamps_are_cpu_float64_for_mps_portability(self):
        elapsed = _profile_elapsed_tensor([.1])
        self.assertEqual(torch.device("cpu"), elapsed.device)
        self.assertEqual(torch.float64, elapsed.dtype)

    def test_entropy_gate_config_requires_finite_probability(self):
        base = {"max_capacity": 2, "topk": 1, "optimizer": "signsgd", "lr": .01}
        self.assertEqual(.5, validate_entropy_gated_latent_ramen_config(
            {**base, "max_normalized_entropy": .5}
        )["max_normalized_entropy"])
        for value in (None, float("nan"), float("inf"), -0.01, 1.01, True):
            with self.assertRaises(ValueError):
                validate_entropy_gated_latent_ramen_config({**base, "max_normalized_entropy": value})

    def test_entropy_gate_rejection_reads_historical_memory_only(self):
        memory = StructuredGradientMemory(1, 4, 1, 1, device="cpu")
        retrieved, active, sizes, retained = update_and_retrieve_entropy_gated_causal_batch(
            memory, torch.tensor([[0.], [10.], [20.]]), torch.tensor([[1.], [2.], [3.]]),
            torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0]), torch.tensor([0., 0., 0.]),
            torch.tensor([0, 1, 2]), torch.tensor([True, False, True]),
            topk=1, include_current=True, beta=0.,
        )
        self.assertEqual([[1.], [1.], [3.]], retrieved.tolist())
        self.assertEqual([1, 1, 1], active.tolist())
        self.assertEqual([1, 1, 2], sizes.tolist())
        self.assertEqual([32, 32, 64], retained.tolist())

    def test_entropy_gate_empty_history_has_zero_update(self):
        memory = StructuredGradientMemory(1, 2, 1, 1, device="cpu")
        retrieved, active, sizes, retained = update_and_retrieve_entropy_gated_causal_batch(
            memory, torch.tensor([[0.]]), torch.tensor([[7.]]), torch.tensor([0]), torch.tensor([0]),
            torch.tensor([0.]), torch.tensor([0]), torch.tensor([False]),
            topk=1, include_current=True, beta=0.,
        )
        self.assertEqual([[0.]], retrieved.tolist())
        self.assertEqual([0], active.tolist())
        self.assertEqual([0], sizes.tolist())
        self.assertEqual([0], retained.tolist())


if __name__ == "__main__":
    unittest.main()
