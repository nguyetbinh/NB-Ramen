"""Real autograd tests of views, anchor/reset, unchanged Ramen, and raw audit."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from test_oracle_support_utility import method, BySampleLayerNorm
from test_query_gradient_selection import context
from methods.MultiViewQueryGradientUtilityProbe import MultiViewQueryGradientUtilityProbe
from methods.query_gradient_multiview import VIEW_SPEC, raw_views, ensemble_log_probabilities, soft_target_ce, direction_scores
from methods.query_gradient_selection import select_positive
from methods.QueryGradientUtilityProbe import tensor_sha
from evaluation.query_gradient_multiview import audit_rows, summarize_cell


def preprocess(image):
    return torch.from_numpy(np.array(image)).float().mean((0, 1)) / 127.5 - 1.


def images(n=100):
    pixels = np.random.default_rng(87).integers(0, 256, size=(n,32,32,3), dtype=np.uint8)
    x = torch.stack([preprocess(Image.fromarray(p)) for p in pixels])
    return pixels, x


def probe(mode='smoke', n=100):
    p = method(batch_size=n)
    p.__class__ = MultiViewQueryGradientUtilityProbe
    p.model.model_type = 'vit'  # independent-row network exercises grouped/isolated parity
    p.cfg.update(topk=5, candidate_m=10, qcgs_mode='smoke', ood_cell='0',
                 dataset_fingerprint='a'*64, source_revision='unit-autograd', view_spec=copy.deepcopy(VIEW_SPEC))
    p.initialize_diagnostic()
    if mode != 'smoke':
        p.mode=mode
        if mode != 'scan': p.registered={0: None, 1: None}
    return p


def feed(p, pixels, labels=None, domains=None):
    context(p, len(pixels), labels, domains)
    p.set_query_images(pixels, preprocess)


class MultiviewTests(unittest.TestCase):
    def test_exact_crop_flip_order_and_no_rng_use(self):
        pixels = np.arange(32*32*3, dtype=np.uint16).reshape(32,32,3).astype(np.uint8)
        before = torch.get_rng_state().clone()
        views = raw_views(pixels)
        for j,(left,top) in enumerate(((0,0),(2,0),(0,2),(2,2))):
            expected = pixels[top:top+30,left:left+30]
            np.testing.assert_array_equal(np.array(views[2*j]), expected)
            np.testing.assert_array_equal(np.array(views[2*j+1]), expected[:,::-1])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for bad in (pixels.astype(float), pixels[:30], pixels[:,:,0]):
            with self.assertRaises(ValueError): raw_views(bad)

    def test_stable_extreme_ensemble_soft_ce_and_direction_guards(self):
        logits = torch.tensor([[[1000.,-1000.],[-1000.,1000.]]]).repeat(8,1,1)
        logp = ensemble_log_probabilities(logits)
        self.assertTrue(torch.isfinite(logp).all())
        self.assertAlmostEqual(float(logp[0,1]), -2000.)
        target = logits.softmax(-1).mean(0).detach()
        torch.testing.assert_close(logp.exp(), target)
        self.assertAlmostEqual(float(soft_target_ce(logits[0],target).sum()),0.)
        with self.assertRaises(ValueError): soft_target_ce(logits[0], target.requires_grad_())
        h,base,c = torch.tensor([1.,2.]),torch.tensor([1.,-1.]).half(),torch.tensor([[-1.,1.],[1.,1.]]).half()
        score,reason = direction_scores(h,base,c,.01)
        torch.testing.assert_close(score,torch.tensor([.02,.04]));self.assertIsNone(reason)
        self.assertEqual(select_positive(*direction_scores(h*0,base,c,.01))['reason'],'zero_query_gradient')
        for bad in (None, torch.ones(3), h*float('nan')):
            with self.assertRaises(ValueError): direction_scores(bad,base,c,.01)

    def test_label_isolation_cache_and_trajectory_parity_across_batches(self):
        pixels,x=images();p,q,baseline=probe(),probe(),method(False,batch_size=100)
        baseline.cfg['topk']=5
        batch_sizes=[]
        p.model.model.register_forward_pre_hook(lambda module,args:batch_sizes.append(len(args[0])))
        for turn in range(2):
            feed(p,pixels)
            labels=torch.arange(100)%2;labels[:2]=1-labels[:2];labels[2:]=-1
            feed(q,pixels,labels,torch.arange(100))
            expected=baseline.forward(x)
            torch.testing.assert_close(p.forward(x),expected,rtol=0,atol=0)
            torch.testing.assert_close(q.forward(x),expected,rtol=0,atol=0)
            for a,b in zip(p.cache,baseline.cache):
                self.assertEqual(a.size,b.size)
                for field in ('keys','values','entropies','priorities'):
                    torch.testing.assert_close(getattr(a,field)[:a.size],getattr(b,field)[:b.size],rtol=0,atol=0)
        for a,b in zip(p.rows,q.rows):
            for name in ('entropy_sign','entropy_ramen','mv_target','random_forced','random_matched'):
                for field in ('index','score','reason'):self.assertEqual(a['policies'][name][field],b['policies'][name][field])
            self.assertEqual(a['multiview']['target'],b['multiview']['target'])
        self.assertTrue(audit_rows(p.rows,'smoke','0'))
        self.assertEqual(set(batch_sizes),{100})
        self.assertFalse(p.model.optimizer.state)
        for layer in p.model.model.modules():
            if isinstance(layer,BySampleLayerNorm):
                torch.testing.assert_close(layer.weight_by_sample,layer.weight.expand(100,-1),rtol=0,atol=0)
                self.assertIsNone(layer.recent_B)  # reset clears the row-count bookkeeping too

    def test_target_and_gradients_use_the_locked_anchors(self):
        pixels,x=images();p=probe()
        captured={}
        original=p.prepare_signals
        def capture(x,base,baseline,query_gradients,selected):
            original(x,base,baseline,query_gradients,selected)
            captured.update(base=base.clone(),baseline=baseline.clone(),target=p.target.clone(),h=p.signals['mv_target'].clone())
        p.prepare_signals=capture
        feed(p,pixels);p.forward(x)
        with torch.no_grad():
            view_logits=torch.stack([p.model(torch.stack([preprocess(raw_views(a)[v]) for a in pixels])) for v in range(8)])
        torch.testing.assert_close(captured['target'],view_logits.float().softmax(-1).mean(0),rtol=0,atol=0)
        theta0=p.parameter_hash()
        p._temporary_logits(x,captured['base'])
        self.assertEqual(p.parameter_hash(),p.rows[0]['multiview']['theta_r_parameter_sha256'])
        self.assertEqual(theta0,p.rows[0]['multiview']['theta_0_parameter_sha256'])
        selected=[r['timestep'] for r in p.rows]
        h=p.checked_gradient(soft_target_ce(p.model(x),captured['target'])[selected].sum(),len(x))
        torch.testing.assert_close(h,captured['h'],rtol=0,atol=0)
        # Non-query slots stay in place and receive zero gradient.
        self.assertEqual(int(torch.count_nonzero(h[2:])),0)
        self.assertEqual(tensor_sha(captured['baseline'][0]),p.rows[0]['multiview']['anchor_logits_sha256'])

    def test_first_order_score_matches_real_finite_difference_at_ramen_anchor(self):
        p=probe();_,x=images()
        base=torch.tensor([1.,-1.,1.,-1.,1.,-1.]).repeat(100,1)
        p._temporary_logits(x,base)
        target=torch.tensor([.25,.75]).repeat(100,1)
        h=p.checked_gradient(soft_target_ce(p.model(x),target)[0],len(x))[0]
        candidate=-base[0]
        scores,_=direction_scores(h,base[0],candidate[None],.01)
        delta=-.01*(candidate.sign()-base[0].sign())
        parameters=dict(p.model.model.named_parameters())
        anchor={k:v.detach().clone() for k,v in parameters.items()}
        losses=[]
        eps=1e-3
        for sign in (-1,1):
            with torch.no_grad():
                for entry in p.parameter_order:
                    par=parameters[entry['name']]
                    par.copy_(anchor[entry['name']])
                    par[0].add_(sign*eps*delta[entry['start']:entry['stop']])
                losses.append(float(soft_target_ce(p.model(x),target)[0]))
        self.assertAlmostEqual(float(scores[0]),(losses[0]-losses[1])/(2*eps),delta=1e-4)

    def test_exhaustive_scope_and_raw_mutation_audit(self):
        pixels,x=images();probes=[]
        for stage in ('stage-a','stage-b'):
            p=probe(stage);feed(p,pixels);p.forward(x)
            self.assertFalse(audit_rows(p.rows,stage,'0'))
            summary=summarize_cell(p.rows,stage)
            self.assertEqual(summary,json.loads(json.dumps(summary)))
            probes.append(p)
        for a,b in zip(probes[0].rows,probes[1].rows):
            self.assertEqual(len(a['verified_swaps']),a['legal_swap_count'])
            self.assertIsNone(b['exact_oracle_utility'])
            self.assertGreaterEqual(a['exact_oracle_utility'],b['best_verified_utility'])
            for name in a['policies']:
                self.assertEqual(a['policies'][name]['index'],b['policies'][name]['index'])
                self.assertEqual(a['policies'][name]['ce'],b['policies'][name]['ce'])
            self.assertEqual(a['prediction_controls'],b['prediction_controls'])
        mutations=[lambda r:r['multiview']['target'].__setitem__(0,.99),
                   lambda r:r['prediction_controls']['ramen_mv'].__setitem__('ce',100.),
                   lambda r:r['prediction_controls']['frozen_mv'].__setitem__('index',None),
                   lambda r:r['policies']['random_matched'].__setitem__('reason','wrong'),
                   lambda r:r['multiview'].__setitem__('anchor_logits_sha256','0'*64),
                   lambda r:r['verified_swaps'].pop(),
                   lambda r:r['policies']['mv_target'].__setitem__('soft_target_ce',99.)]
        for mutate in mutations:
            rows=copy.deepcopy(probes[0].rows);mutate(rows[0])
            with self.assertRaises(ValueError):audit_rows(rows,'stage-a','0')

    def test_no_swap_random_consumes_the_same_draw_and_positive_reference(self):
        p=probe()
        choices={'mv_target':{'index':None}}
        first=p.random_choices(10,choices)
        self.assertIsNone(first['random_matched']['index'])
        choices['mv_target']['index']=1
        second=p.random_choices(10,choices)
        self.assertEqual(second['random_forced'],second['random_matched'])
        import random
        rng=random.Random(0)
        self.assertEqual(first['random_forced']['index'],rng.randrange(10))
        self.assertEqual(second['random_forced']['index'],rng.randrange(10))
        self.assertIsNone(p.reference_choice([-1.,0.])['index'])
        self.assertEqual(p.reference_choice([1.,1.])['index'],0)

    def test_scan_unscored_and_exception_cleanup(self):
        pixels,x=images();p=probe('scan');feed(p,pixels);p.forward(x)
        self.assertEqual(p.rows,[]);self.assertEqual(len(p.scan),100)
        p=probe();feed(p,pixels)
        def fail(*args):raise RuntimeError('test interruption')
        p._verify_jobs=fail
        with self.assertRaisesRegex(RuntimeError,'interruption'):p.forward(x)
        self.assertIsNone(p.raw_context);self.assertIsNone(p.identity);self.assertIsNone(p._pending)
        self.assertFalse(p.model.optimizer.state)
        for layer in p.model.model.modules():
            if isinstance(layer,BySampleLayerNorm):
                torch.testing.assert_close(layer.weight_by_sample,layer.weight.expand(100,-1),rtol=0,atol=0)
        p=probe();feed(p,pixels)
        with self.assertRaisesRegex(ValueError,'reproduce'):p.forward(x+1)


if __name__=='__main__':unittest.main()
