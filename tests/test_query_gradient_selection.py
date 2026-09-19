"""QCGS numerical guards and real autograd baseline/isolation contracts."""
import copy
import inspect
from pathlib import Path
import random
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from methods.query_gradient_selection import score_swaps, select_positive, select_one_swap, random_swap
from methods.QueryGradientUtilityProbe import QueryGradientUtilityProbe
from test_oracle_support_utility import method, BySampleLayerNorm
from evaluation.query_gradient_utility import audit_rows
from runtime.query_gradient_registry import digest


def qcgs(mode='smoke',batch_size=8):
    p=method(batch_size=batch_size)
    p.__class__=QueryGradientUtilityProbe
    p.cfg.update(qcgs_mode=mode,ood_cell='0',dataset_fingerprint='a'*64,source_revision='unit-autograd')
    p.initialize_diagnostic()
    return p


def context(p,n,labels=None,domains=None):
    labels=torch.arange(n)%2 if labels is None else labels
    domains=torch.zeros(n,dtype=torch.long) if domains is None else domains
    p.set_query_identity(torch.arange(p.counter,p.counter+n))
    p.set_oracle_known_label(labels.long(),is_ood=labels==-1,domains=domains)


class SelectorTests(unittest.TestCase):
    def vectors(self):
        return (torch.tensor([1.,2.]),torch.tensor([1.,-1.]),torch.tensor([[-1.,1.],[1.,1.]]),
                torch.tensor([[0.,1.],[1.,1.]]),torch.tensor([[1.,0.],[1.,0.]]),.01)

    def test_manual_scores_and_first_tie(self):
        result=score_swaps(*self.vectors())
        torch.testing.assert_close(result['entropy_sign'],torch.tensor([.02,.04]))
        self.assertEqual(select_positive(result['entropy_sign'])['index'],1)
        self.assertEqual(select_positive(torch.tensor([1.,1.]))['index'],0)
        self.assertEqual(select_positive(torch.tensor([0.,0.]))['reason'],'zero_maximum_no_swap_wins')
        self.assertEqual(select_positive(torch.tensor([-1.,-2.]))['reason'],'negative_maximum')
        h,g,c,i,o,lr=self.vectors()
        expected=torch.nn.functional.cosine_similarity(h[None],i)-torch.nn.functional.cosine_similarity(h[None],o)
        torch.testing.assert_close(result['entropy_cosine'],expected)
        self.assertNotIn('label',inspect.signature(select_one_swap).parameters)

    def test_numerical_fallbacks(self):
        args=list(self.vectors())
        for h,reason in [(None,'missing_query_gradient'),(torch.ones(3),'shape_incompatible_query_gradient'),
                         (torch.tensor([float('nan'),1.]),'nonfinite_query_gradient'),
                         (torch.tensor([float('inf'),1.]),'nonfinite_query_gradient')]:
            args[0]=h
            result=score_swaps(*args)
            self.assertEqual(select_positive(result['entropy_sign'],result['reasons']['entropy_sign']),
                             dict(index=None,score=0.,reason=reason,invalid=True))
        args=list(self.vectors());args[0]=torch.zeros(2)
        self.assertEqual(score_swaps(*args)['reasons']['entropy_sign'],('zero_query_gradient',False))
        args[2][0,0]=float('nan')
        self.assertEqual(score_swaps(*args)['reasons']['entropy_sign'],('nonfinite_candidate',True))
        args=list(self.vectors());args[3][0].zero_()
        result=score_swaps(*args)
        self.assertEqual(result['reasons']['entropy_cosine'],('zero_candidate_norm',False))
        self.assertIsNotNone(result['entropy_sign'])
        self.assertTrue(select_positive(torch.tensor([float('nan')]))['invalid'])
        args=list(self.vectors());args[0]=torch.full((2,),3e38)
        self.assertTrue(score_swaps(*args)['reasons']['entropy_sign'][1])
        args=list(self.vectors());args[3]=torch.ones(3,3)
        with self.assertRaises(ValueError): score_swaps(*args)
        args=list(self.vectors());args[2]=args[1].repeat(2,1)
        self.assertEqual(select_one_swap(*args,'entropy_sign')['reason'],'all_aggregate_directions_unchanged')
        self.assertEqual(select_positive(torch.empty(0))['reason'],'no_legal_swaps')
        with self.assertRaises(ValueError): score_swaps(*self.vectors()[:-1],float('nan'))

    def test_random_is_dedicated_forced_and_reproducible(self):
        a,b=random.Random(0),random.Random(0)
        for _ in range(50):
            torch.rand(3)
            self.assertEqual(random_swap(10,a)['index'],b.randrange(10))
        self.assertEqual(random_swap(0,a)['reason'],'no_legal_swaps')

    def test_score_preserves_half_aggregate_signs(self):
        h,g,c,i,o,lr=self.vectors()
        score=score_swaps(h.half(),g.half(),c.half(),i.half(),o.half(),lr)
        self.assertEqual(score['entropy_sign'].dtype,torch.float32)
        torch.testing.assert_close(score['entropy_sign'],torch.tensor([.02,.04]))


class ProbeTests(unittest.TestCase):
    def x(self):
        return torch.tensor([[3.,0.,1.],[0.,3.,1.],[2.,0.,1.],[0.,2.,1.]])

    def test_real_baseline_reset_and_metadata_isolation(self):
        a,b,base=qcgs(),qcgs(),method(False)
        for turn in range(2):
            context(a,4)
            # Mutate ID labels and domains, as well as OOD flags on non-query rows.
            context(b,4,torch.tensor([1,0,-1,-1]),torch.tensor([3,2,1,0]))
            actual=a.forward(self.x());changed=b.forward(self.x());expected=base.forward(self.x())
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(changed,expected,rtol=0,atol=0)
            for left,right in zip(a.cache,base.cache):
                for field in ('keys','values','entropies','priorities'):
                    torch.testing.assert_close(getattr(left,field)[:left.size],getattr(right,field)[:right.size],rtol=0,atol=0)
            for left,right in zip(a.rows,b.rows):
                for name in ('entropy_sign','entropy_cosine','random'):
                    for field in ('index','score','reason'):
                        self.assertEqual(left['policies'][name][field],right['policies'][name][field])
            self.assertFalse(a.model.optimizer.state)
            for layer in a.model.model.modules():
                if isinstance(layer,BySampleLayerNorm):
                    torch.testing.assert_close(layer.weight_by_sample,layer.weight.expand(8,-1),rtol=0,atol=0)
        a.reset();context(a,4);a.forward(self.x())
        self.assertEqual([r['timestep'] for r in a.scan],list(range(4)))
        self.assertEqual([r['random_draw'] for r in a.rows],[0,1])
        self.assertIsNone(a.identity)

    def test_exception_resets_parameters_and_context(self):
        p=qcgs();context(p,4)
        original=p._verify_jobs
        def fail(*args): raise RuntimeError('verification interruption')
        p._verify_jobs=fail
        with self.assertRaisesRegex(RuntimeError,'interruption'): p.forward(self.x())
        for layer in p.model.model.modules():
            if isinstance(layer,BySampleLayerNorm):
                torch.testing.assert_close(layer.weight_by_sample,layer.weight.expand(8,-1),rtol=0,atol=0)
        self.assertFalse(p.model.optimizer.state)
        self.assertIsNone(p.identity);self.assertIsNone(p._pending)
        p.reset();p._verify_jobs=original;context(p,4);p.forward(self.x())

    def test_scan_is_unscored_and_baseline_exact(self):
        p=qcgs('scan');base=method(False);context(p,4)
        torch.testing.assert_close(p.forward(self.x()),base.forward(self.x()),rtol=0,atol=0)
        self.assertEqual(p.rows,[]);self.assertEqual(len(p.scan),4)

    def test_full_batch_k5_m10_raw_audit_and_mutation(self):
        p=qcgs(batch_size=100);base=method(False,batch_size=100)
        p.cfg.update(topk=5,candidate_m=10);base.cfg.update(topk=5)
        x=torch.randn(100,3,generator=torch.Generator().manual_seed(17))
        context(p,100)
        torch.testing.assert_close(p.forward(x),base.forward(x),rtol=0,atol=0)
        self.assertTrue(audit_rows(p.rows,'smoke','0'))
        bad=copy.deepcopy(p.rows);bad[0]['policies']['entropy_sign']['supervised_utility']+=1
        with self.assertRaisesRegex(ValueError,'arithmetic'): audit_rows(bad,'smoke','0')
        bad=copy.deepcopy(p.rows);bad[0]['policies']['random']['index']=-1
        with self.assertRaisesRegex(ValueError,'random'): audit_rows(bad,'smoke','0')
        bad=copy.deepcopy(p.rows);bad[0]['exact_oracle_utility']=0
        with self.assertRaisesRegex(ValueError,'oracle scope'): audit_rows(bad,'smoke','0')

    def test_exhaustive_and_subset_scopes_share_frozen_actions(self):
        probes=[]
        x=torch.randn(100,3,generator=torch.Generator().manual_seed(27))
        for stage in ('stage-a','stage-b'):
            p=qcgs(batch_size=100)
            p.mode=stage;p.registered={0:None,1:None}
            p.cfg.update(topk=5,candidate_m=10)
            context(p,100);p.forward(x)
            # Two fixture queries intentionally do not satisfy scientific quotas.
            self.assertFalse(audit_rows(p.rows,stage,'0'))
            probes.append(p)
        for a,b in zip(probes[0].rows,probes[1].rows):
            self.assertEqual(len(a['verified_swaps']),a['legal_swap_count'])
            self.assertLessEqual(len(b['verified_swaps']),4)
            self.assertIsNone(b['exact_oracle_utility'])
            self.assertGreaterEqual(a['exact_oracle_utility'],b['best_verified_utility'])
            for name in ('entropy_sign','entropy_cosine','random','supervised_reference'):
                self.assertEqual(a['policies'][name]['index'],b['policies'][name]['index'])
                self.assertEqual(a['policies'][name]['supervised_utility'],b['policies'][name]['supervised_utility'])
        bad=copy.deepcopy(probes[0].rows);bad[0]['verified_swaps'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):audit_rows(bad,'stage-a','0')


if __name__=='__main__': unittest.main()
