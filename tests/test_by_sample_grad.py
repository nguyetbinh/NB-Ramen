import unittest

import torch
import torch.nn as nn

from src.models.ModelForBySampleTTA import (
    BySampleBatchNorm,
    BySampleLayerNorm,
    ModelForBySampleTTA,
)
from src.models.optimizer import SignSGD


class BySampleGradTests(unittest.TestCase):
    def _make_owner(self, module):
        owner = ModelForBySampleTTA()
        owner.model = nn.Sequential(module)
        return owner

    def _assert_missing_buffers_materialize(self, module):
        owner = self._make_owner(module)
        batch_size = 2
        width = module.weight_by_sample.shape[1] + module.bias_by_sample.shape[1]
        gradients = torch.arange(batch_size * width, dtype=torch.float32).reshape(batch_size, width)

        module.recent_B = batch_size
        optimizer = torch.optim.SGD(
            [module.weight_by_sample, module.bias_by_sample], lr=0.01)
        module.weight_by_sample.grad = torch.ones_like(module.weight_by_sample)
        module.bias_by_sample.grad = torch.ones_like(module.bias_by_sample)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(module.weight_by_sample.grad)
        self.assertIsNone(module.bias_by_sample.grad)

        owner.set_by_sample_grad(gradients)

        self.assertEqual(module.weight_by_sample.shape, module.weight_by_sample.grad.shape)
        self.assertEqual(module.weight_by_sample.dtype, module.weight_by_sample.grad.dtype)
        self.assertEqual(module.weight_by_sample.device, module.weight_by_sample.grad.device)
        self.assertEqual(module.bias_by_sample.shape, module.bias_by_sample.grad.shape)
        self.assertEqual(module.bias_by_sample.dtype, module.bias_by_sample.grad.dtype)
        self.assertEqual(module.bias_by_sample.device, module.bias_by_sample.grad.device)
        torch.testing.assert_close(module.weight_by_sample.grad[:batch_size], gradients[:, :2])
        torch.testing.assert_close(module.bias_by_sample.grad[:batch_size], gradients[:, 2:])
        torch.testing.assert_close(module.weight_by_sample.grad[batch_size:], torch.zeros_like(module.weight_by_sample.grad[batch_size:]))
        torch.testing.assert_close(module.bias_by_sample.grad[batch_size:], torch.zeros_like(module.bias_by_sample.grad[batch_size:]))

    def test_missing_grad_buffers_materialize_for_layer_norm_and_batch_norm(self):
        self._assert_missing_buffers_materialize(
            BySampleLayerNorm(nn.LayerNorm(2), max_batch_size=4))
        self._assert_missing_buffers_materialize(
            BySampleBatchNorm(nn.BatchNorm2d(2), max_batch_size=4))

    def test_single_sample_parameters_do_not_alias_reset_buffers(self):
        for module in (
            BySampleLayerNorm(nn.LayerNorm(2), max_batch_size=1),
            BySampleBatchNorm(nn.BatchNorm2d(2), max_batch_size=1),
        ):
            with self.subTest(module=module.__class__.__name__):
                original_weight = module.weight.clone()
                original_bias = module.bias.clone()
                self.assertNotEqual(
                    module.weight.untyped_storage().data_ptr(),
                    module.weight_by_sample.untyped_storage().data_ptr(),
                )
                self.assertNotEqual(
                    module.bias.untyped_storage().data_ptr(),
                    module.bias_by_sample.untyped_storage().data_ptr(),
                )

                with torch.no_grad():
                    module.weight_by_sample.sub_(0.01)
                    module.bias_by_sample.add_(0.01)

                torch.testing.assert_close(module.weight, original_weight)
                torch.testing.assert_close(module.bias, original_bias)
                with torch.no_grad():
                    module.reset_parameters()
                torch.testing.assert_close(module.weight_by_sample[0], original_weight)
                torch.testing.assert_close(module.bias_by_sample[0], original_bias)

    def test_single_sample_signsgd_reset_reapplies_identical_update(self):
        module = BySampleLayerNorm(nn.LayerNorm(2), max_batch_size=1)
        owner = self._make_owner(module)
        owner.optimizer = SignSGD(owner.model.parameters(), lr=0.01)
        gradients = torch.tensor([[1.0, -1.0, -1.0, 1.0]])

        module.recent_B = 1
        owner.set_by_sample_grad(gradients)
        owner.step_and_zero_grad()
        actual_weight = module.weight_by_sample.clone()
        actual_bias = module.bias_by_sample.clone()

        owner.reset_parameters()
        module.recent_B = 1
        owner.set_by_sample_grad(gradients)
        owner.step_and_zero_grad()

        torch.testing.assert_close(module.weight_by_sample, actual_weight)
        torch.testing.assert_close(module.bias_by_sample, actual_bias)

    def test_existing_buffers_only_overwrite_recent_rows(self):
        module = BySampleLayerNorm(nn.LayerNorm(2), max_batch_size=4)
        owner = self._make_owner(module)
        module.recent_B = 2
        module.weight_by_sample.grad = torch.full_like(module.weight_by_sample, -1)
        module.bias_by_sample.grad = torch.full_like(module.bias_by_sample, -1)
        gradients = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

        owner.set_by_sample_grad(gradients)

        torch.testing.assert_close(module.weight_by_sample.grad[:2], gradients[:, :2])
        torch.testing.assert_close(module.bias_by_sample.grad[:2], gradients[:, 2:])
        torch.testing.assert_close(module.weight_by_sample.grad[2:], torch.full_like(module.weight_by_sample.grad[2:], -1))
        torch.testing.assert_close(module.bias_by_sample.grad[2:], torch.full_like(module.bias_by_sample.grad[2:], -1))

    def test_rejects_uninitialized_recent_batch_and_invalid_grad_matrix(self):
        module = BySampleLayerNorm(nn.LayerNorm(2), max_batch_size=4)
        owner = self._make_owner(module)

        with self.assertRaisesRegex(RuntimeError, 'recent_B'):
            owner.set_by_sample_grad(torch.zeros(2, 4))

        module.recent_B = 2
        with self.assertRaisesRegex(ValueError, '2-dimensional'):
            owner.set_by_sample_grad(torch.zeros(4))
        with self.assertRaisesRegex(ValueError, 'rows'):
            owner.set_by_sample_grad(torch.zeros(1, 4))
        with self.assertRaisesRegex(ValueError, 'width'):
            owner.set_by_sample_grad(torch.zeros(2, 3))
        module.recent_B = 0
        with self.assertRaisesRegex(ValueError, 'Invalid recent_B'):
            owner.set_by_sample_grad(torch.zeros(2, 4))


if __name__ == '__main__':
    unittest.main()
