"""Adaptation must not modify the pretrained reset state at any capacity."""

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.ModelForBySampleTTA import BySampleBatchNorm, BySampleLayerNorm
from models.optimizer import SignSGD


class BySampleNormalizationTests(unittest.TestCase):
    def check_reset(self, kind, capacity, device="cpu", batch_first=True):
        if kind == "layer":
            base = torch.nn.LayerNorm(4).to(device)
            wrapper = BySampleLayerNorm
            kwargs = {"batch_first": batch_first}
            shape = (1, 3, 4) if batch_first else (3, 1, 4)
        else:
            base = torch.nn.BatchNorm2d(4, track_running_stats=False).to(device)
            wrapper, kwargs, shape = BySampleBatchNorm, {}, (1, 4, 3, 3)
        with torch.no_grad():
            base.weight.copy_(torch.tensor([.5, 1., 1.5, 2.], device=device))
            base.bias.copy_(torch.tensor([.1, .2, .3, .4], device=device))
        reference = copy.deepcopy(base)
        initial_weight, initial_bias = base.weight.detach().clone(), base.bias.detach().clone()
        module = wrapper(base, max_batch_size=capacity, **kwargs)
        x = torch.arange(torch.tensor(shape).prod().item(), device=device, dtype=torch.float32).reshape(shape)
        expected = reference(x).detach()
        torch.testing.assert_close(module(x), expected)
        optimizer = SignSGD(module.parameters(), lr=.125)
        for _ in range(3):
            optimizer.zero_grad()
            module(x).square().mean().backward()
            optimizer.step()
            self.assertFalse(torch.equal(module.weight_by_sample[0], initial_weight))
            self.assertFalse(torch.equal(module.bias_by_sample[0], initial_bias))
            self.assertTrue(torch.equal(module.weight, initial_weight), "pretrained weight was mutated")
            self.assertTrue(torch.equal(module.bias, initial_bias), "pretrained bias was mutated")
            with torch.no_grad():
                module.reset_parameters()
            self.assertIsNone(module.recent_B)
            self.assertTrue(torch.equal(module.weight_by_sample, initial_weight.expand(capacity, -1)))
            self.assertTrue(torch.equal(module.bias_by_sample, initial_bias.expand(capacity, -1)))
            torch.testing.assert_close(module(x), expected)

    def test_layer_norm_resets_after_repeated_adaptation(self):
        for capacity in (1, 2, 100):
            with self.subTest(capacity=capacity):
                self.check_reset("layer", capacity)

    def test_sequence_first_layer_norm_resets_after_adaptation(self):
        for capacity in (1, 100):
            with self.subTest(capacity=capacity):
                self.check_reset("layer", capacity, batch_first=False)

    def test_batch_norm_resets_after_repeated_adaptation(self):
        for capacity in (1, 2, 100):
            with self.subTest(capacity=capacity):
                self.check_reset("batch", capacity)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a real CUDA device")
    def test_cuda_norms_preserve_pretrained_reset_state(self):
        for kind in ("layer", "batch"):
            for capacity in (1, 100):
                with self.subTest(kind=kind, capacity=capacity):
                    self.check_reset(kind, capacity, device="cuda")


if __name__ == "__main__":
    unittest.main()
