"""Real autograd/SignSGD mechanics on a small independent-row network."""
import random
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from methods.Ramen import Ramen, PriorityCache
from methods.OracleSupportUtilityProbe import OracleSupportUtilityProbe, SupportProvenanceCache, aggregate_pools, aggregate_swaps, legal_swaps
from methods.losses import softmax_entropy
from models.ModelForBySampleTTA import ModelForBySampleTTA, BySampleLayerNorm
from models.optimizer import SignSGD
from evaluation.oracle_support_utility import summarize


class TinyModel(ModelForBySampleTTA):
    def __init__(self, batch_size=8):
        self.model = nn.Sequential(BySampleLayerNorm(nn.LayerNorm(3), batch_size))
        self.optimizer = SignSGD(self.model.parameters(), lr=.1)

    def featurize(self, x):
        return self.model(x)

    def classify(self, x):
        return x[:, :2]


def method(probe=True, device='cpu', batch_size=8):
    cls = OracleSupportUtilityProbe if probe else Ramen
    result = object.__new__(cls)
    result.model = TinyModel(batch_size)
    result.model.model.to(device)
    result.cfg = {'topk': 1, 'candidate_m': 2, 'probe_queries': 2, 'probe_mode': 'exhaustive', 'lr': .1}
    result.beta = 5.
    result.device = torch.device(device)
    result.dtype = torch.half
    result.num_classes = 2
    cache_cls = SupportProvenanceCache if probe else PriorityCache
    result.cache = [cache_cls(750, 3, 6, device, torch.half) for _ in range(2)]
    result.counter = 0
    result.last_diagnostics = {}
    result.loss_fn = lambda logits: softmax_entropy(logits, reduction='sum')
    if probe:
        result._pending = None
        result.rows = []
        result.seed = 0
        result.rng = random.Random(0)
    return result


class SupportUtilityTests(unittest.TestCase):
    def context(self, probe, labels=None):
        labels = torch.tensor([0, 1, -1, 0]) if labels is None else labels
        probe.set_oracle_known_label(labels, is_ood=labels == -1, domains=torch.zeros(len(labels), dtype=torch.long))

    def test_vectorized_aggregation_matches_serial_for_cache_dtypes(self):
        devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
        for device in devices:
            for dtype in (torch.float16, torch.float32):
                for dim in (3, 40000):
                    generator = torch.Generator().manual_seed(14)
                    pools = {}
                    for c, size in enumerate((10, 7, 4)):
                        values = torch.randn(size, dim, generator=generator).to(device=device, dtype=dtype)
                        weights = torch.rand(size, generator=generator).to(device=device, dtype=dtype)
                        k = min(5, size)
                        pools[c] = {'k': k, 'values': values, 'weights': weights,
                                    'base_contribution': (values[:k]*weights[:k,None]).sum(0)}
                    swaps = list(legal_swaps(pools, 10))
                    template = torch.zeros(dim, device=device)
                    batched = aggregate_swaps(pools, template, swaps)
                    for i, swap in enumerate(swaps):
                        torch.testing.assert_close(batched[i], aggregate_pools(pools, template, swap), rtol=0, atol=0)

    def test_hooks_fail_closed_and_consume_on_mismatch(self):
        probe = method()
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            probe._consume(4)
        self.context(probe)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            self.context(probe)
        with self.assertRaisesRegex(RuntimeError, 'batch size'):
            probe._consume(3)
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            probe._consume(4)
        with self.assertRaisesRegex(ValueError, 'OOD'):
            probe.set_oracle_known_label(torch.tensor([0]), is_ood=torch.tensor([True]), domains=torch.tensor([0]))

    def test_real_updates_preserve_baseline_cache_and_reset(self):
        probe, base = method(), method(False)
        x = torch.tensor([[3., 0., 1.], [0., 3., 1.], [2., 0., 1.], [0., 2., 1.]])
        for _ in range(2):
            self.context(probe)
            expected = base.forward(x)
            actual = probe.forward(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(probe.cache, base.cache):
                self.assertEqual(a.size, b.size)
                for name in ('keys', 'values', 'entropies', 'priorities'):
                    torch.testing.assert_close(getattr(a, name)[:a.size], getattr(b, name)[:b.size], rtol=0, atol=0)
            for module in probe.model.model.modules():
                if isinstance(module, BySampleLayerNorm):
                    torch.testing.assert_close(module.weight_by_sample, module.weight.expand(8, -1))
        self.assertEqual(len(probe.rows), 2)
        self.assertNotIn(2, [r['query_index'] for r in probe.rows])
        for row in probe.rows:
            self.assertEqual(row['legal_swap_count'], row['verified_swap_count'])
            self.assertGreaterEqual(row['oracle']['exact_utility'], 0)
            for swap in row['verified_swaps']:
                self.assertEqual(swap['outgoing']['rank'], 1)
                self.assertEqual(swap['incoming']['rank'], 2)
        summary = summarize(probe.rows, 2)
        self.assertTrue(summary['query_budget_complete'])
        self.assertEqual(summary['oracle_scope'], 'exhaustive_one_swap')

    def test_legacy_batch_100_top5_candidate10_baseline_parity(self):
        probe, base = method(batch_size=100), method(False, batch_size=100)
        probe.cfg.update(topk=5, candidate_m=10, probe_queries=16)
        base.cfg['topk'] = 5
        x = torch.randn(100, 3, generator=torch.Generator().manual_seed(123))
        labels = torch.arange(100) % 2
        labels[::2] = -1
        self.context(probe, labels)
        expected = base.forward(x)
        actual = probe.forward(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(probe.rows), 16)
        for row in probe.rows:
            self.assertEqual(row['legal_swap_count'], 50)
            self.assertEqual(row['verified_swap_count'], 50)
            self.assertGreaterEqual(row['oracle']['exact_utility'], 0.)
            self.assertAlmostEqual(row['oracle']['exact_utility'], max(0., max(s['exact_utility'] for s in row['verified_swaps'])))

    def test_grouped_trials_match_serial_at_legacy_dimensions(self):
        grouped, serial = method(batch_size=100), method(batch_size=100)
        grouped.model.model_type = 'vit'
        x = torch.randn(100, 3, generator=torch.Generator().manual_seed(123))
        labels = torch.arange(100) % 2
        labels[::2] = -1
        for probe in (grouped, serial):
            probe.cfg.update(topk=5, candidate_m=10, probe_queries=16)
            self.context(probe, labels)
        torch.testing.assert_close(grouped.forward(x), serial.forward(x), rtol=0, atol=0)
        self.assertEqual(grouped.rows, serial.rows)

    def test_legal_swaps_preserve_class_count_and_recompute_weights(self):
        pools = {0: {'k': 1, 'values': torch.tensor([[2., -1.], [-3., 2.]]), 'weights': torch.tensor([.5, .25])},
                 1: {'k': 1, 'values': torch.tensor([[1., 1.]]), 'weights': torch.tensor([.1])}}
        swaps = list(legal_swaps(pools, 2))
        self.assertEqual(swaps, [{'class': 0, 'out': 0, 'in': 1}])
        torch.testing.assert_close(aggregate_pools(pools, torch.zeros(2), swaps[0]), torch.tensor([-.325, .3]))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'requires MPS')
    def test_mps_half_supervised_backward_and_exact_forward(self):
        probe = method(device='mps', batch_size=100)
        probe.model.model_type = 'vit'
        probe.cfg.update(topk=5, candidate_m=10, probe_queries=16)
        labels = torch.arange(100) % 2
        labels[::2] = -1
        self.context(probe, labels)
        x = torch.randn(100, 3, generator=torch.Generator().manual_seed(123)).to(device='mps', dtype=torch.half)
        output = probe.forward(x)
        torch.mps.synchronize()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(probe.rows), 16)
        self.assertGreaterEqual(sum(r['exact_forward_count'] for r in probe.rows), 2)

    def test_screened_verifies_unique_baseline_choices(self):
        probe = method()
        probe.cfg['probe_mode'] = 'screened'
        self.context(probe)
        probe.forward(torch.tensor([[3., 0., 1.], [0., 3., 1.], [2., 0., 1.], [0., 2., 1.]]))
        self.assertTrue(probe.rows)
        for row in probe.rows:
            self.assertLessEqual(row['verified_swap_count'], 4)
            self.assertEqual(set(row['baselines']), {'feature_similarity', 'best_predicted', 'worst_predicted', 'random', 'gradient_cosine'})
        self.assertEqual(summarize(probe.rows, 2)['oracle_scope'], 'screened_subset_lower_bound')

    def test_temporary_update_returns_to_pretrained_on_failure(self):
        probe = method()
        self.context(probe)
        def fail():
            raise RuntimeError('output failure')
        probe.probe_progress = fail
        with self.assertRaisesRegex(RuntimeError, 'output failure'):
            probe.forward(torch.tensor([[3., 0., 1.], [0., 3., 1.], [2., 0., 1.], [0., 2., 1.]]))
        for module in probe.model.model.modules():
            if isinstance(module, BySampleLayerNorm):
                torch.testing.assert_close(module.weight_by_sample, module.weight.expand(8, -1))
                self.assertIsNone(module.weight_by_sample.grad)
        self.assertIsNone(probe._pending)

    def test_eviction_metadata_matches_admission_including_rejection(self):
        cache = SupportProvenanceCache(1, 1, 1, 'cpu', torch.float32)
        for priority in (2., 1., 3.):
            cache.add_with_metadata(torch.tensor([priority]), torch.tensor([priority]),
                                    torch.tensor(0.), torch.tensor(priority), {'domain': priority})
            self.assertEqual(cache.metadata[0]['domain'], float(cache.keys[0, 0]))
        self.assertEqual(cache.metadata[0]['domain'], 3.)
        cache.reset()
        self.assertEqual(cache.metadata, [None])

    def test_partial_class_extras_are_legal_after_query_eligibility(self):
        pools = {0: {'k': 1, 'values': torch.zeros(3, 2)},
                 1: {'k': 1, 'values': torch.zeros(2, 2)}}
        self.assertEqual(list(legal_swaps(pools, 3)), [
            {'class': 0, 'out': 0, 'in': 1}, {'class': 0, 'out': 0, 'in': 2},
            {'class': 1, 'out': 0, 'in': 1}])
        self.assertEqual(list(legal_swaps({1: pools[1]}, 3)), [])

    def test_ineligible_cache_skips_supervised_backward(self):
        probe = method()
        probe.cfg['candidate_m'] = 10
        class CountingModel(TinyModel):
            forwards = 0
            def featurize(self, x):
                self.forwards += 1
                return super().featurize(x)
        probe.model = CountingModel()
        self.context(probe)
        probe.forward(torch.randn(4, 3))
        self.assertEqual(probe.model.forwards, 2)
        self.assertEqual(probe.rows, [])

    def test_empty_summary_and_all_ood(self):
        probe = method()
        self.context(probe, torch.full((4,), -1))
        probe.forward(torch.randn(4, 3))
        self.assertEqual(probe.rows, [])
        self.assertIsNone(summarize([], 16)['headroom_rate'])
        self.assertEqual(summarize([], 16, 'exhaustive')['oracle_scope'], 'exhaustive_one_swap')
        self.assertFalse(summarize([], 16)['query_budget_complete'])


if __name__ == '__main__':
    unittest.main()
