"""Frozen registry, decision ordering and descriptive uncertainty checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from runtime.query_gradient_registry import assigned_stage,build_registry,validate_registry,CELLS,digest,write_json,file_sha
from evaluation.query_gradient_utility import decide,sensitivity,uncertainty,correlations,freeze_bins,analyze,ANALYSIS
from runtime.query_gradient_campaign import make_job,ledger_commit,verify_ledger,freeze_file,audit_campaign


def registry():
    scans={c:[dict(timestep=i,sample_idx=i//2,domain=i%2,is_ood=False,eligible=i>=10,
                   baseline_sha256='b'*64,aggregate_sha256='c'*64) for i in range(1600)] for c in CELLS}
    return build_registry(scans,'a'*64,{'base_image_indices':list(range(20)),'smoke':[]},source_revision='f'*40,provenance={})


class RegistryTests(unittest.TestCase):
    def test_hash_partition_base_images_and_earliest_quotas(self):
        r=registry();validate_registry(r)
        a,b=set(),set()
        for cell in CELLS:
            for stage,destination in [('stage-a',a),('stage-b',b)]:
                rows=r['cells'][cell]['selected'][stage]
                self.assertEqual(len(rows),16 if stage=='stage-a' else 128)
                self.assertEqual(len(rows),len({x['sample_idx'] for x in rows}))
                self.assertTrue(all(x['timestep']%2==0 and x['sample_idx']>=20 for x in rows))
                destination.update(x['sample_idx'] for x in rows)
        self.assertFalse(a&b)
        self.assertEqual(r,registry())
        self.assertEqual(assigned_stage('a'*64,40),assigned_stage('a'*64,40))
        with self.assertRaises(ValueError): assigned_stage('A'*64,40)
        with self.assertRaises(ValueError): assigned_stage('a'*64,True)

    def test_registry_tampering_and_insufficient_prefix(self):
        for mutation in ('quota','overlap','hash','later','stream'):
            r=registry();cell=r['cells']['0']
            if mutation=='quota':cell['selected']['stage-a'].pop()
            elif mutation=='overlap':cell['selected']['stage-b'][0]=cell['selected']['stage-a'][0]
            elif mutation=='hash':r['sha256']='0'*64
            elif mutation=='later':cell['selected']['stage-a'][0]=dict(cell['selected']['stage-a'][0],timestep=999)
            else:cell['stream'][0]['timestep']=10
            if mutation!='hash':r.pop('sha256')
            with self.assertRaises(ValueError,msg=mutation):validate_registry(r)
        scans={c:[dict(timestep=i,sample_idx=i,domain=0,is_ood=False,eligible=True) for i in range(100)] for c in CELLS}
        with self.assertRaisesRegex(ValueError,'insufficient'):build_registry(scans,'a'*64,{'base_image_indices':[]},source_revision='f'*40,provenance={})

    def test_launch_and_ledger_are_locked_before_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);locks=root/'locks';locks.mkdir()
            write_json(locks/'registry.json',registry())
            job=make_job(locks,root/'runs',root/'data','stage-b','0',600,'a'*64,'f'*40,locks/'registry.json')
            freeze_file(locks/'launch.json',job)
            first=ledger_commit(locks,'chore: freeze test inputs')
            verify_ledger(locks)
            self.assertEqual(first,ledger_commit(locks,'chore: retain identical inputs'))
            self.assertIn('--config-lock-sha256',job['command'])
            self.assertIn('--known-class-split-sha256',job['command'])
            with self.assertRaisesRegex(ValueError,'frozen input changed'):freeze_file(locks/'launch.json',{'changed':True})
            (locks/'launch.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'committed ledger'):verify_ledger(locks)

    def test_offline_audit_checks_frozen_analysis_and_content_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);locks=root/'locks';locks.mkdir()
            r=registry();write_json(locks/'registry.json',r)
            write_json(locks/'preflight.json',{'source':{'revision':'f'*40},'analysis':ANALYSIS,'artifacts':{}})
            jobs=[make_job(locks,root/'runs',root/'data',stage,cell,r['cells'][cell]['max_eval_samples'],'a'*64,'f'*40,locks/'registry.json')
                  for stage in ('stage-a','stage-b') for cell in CELLS]
            launch={'jobs':jobs,'source_revision':'f'*40,'registry_sha256':r['sha256'],
                    'registry_file_sha256':file_sha(locks/'registry.json'),'preflight_sha256':file_sha(locks/'preflight.json'),
                    'analysis_sha256':digest(ANALYSIS)}
            write_json(locks/'launch.json',launch);ledger_commit(locks,'chore: freeze unexecuted test inputs')
            self.assertEqual(audit_campaign(root)['decision'],'INCONCLUSIVE')
            launch['analysis_sha256']='0'*64
            write_json(locks/'launch.json',launch);ledger_commit(locks,'test: change analysis lock')
            with self.assertRaisesRegex(ValueError,'analysis specification'):audit_campaign(root)


class AnalysisTests(unittest.TestCase):
    def outcomes(self,u=1.,d=1.,ent=1.,swaps=10,drop=1.,net=0):
        a={c:{'exact_oracle':{'mean':1.}} for c in CELLS}
        b={c:{'U':{'mean':u,'drop_two_largest_mean':drop},'D':{'mean':d,'drop_two_largest_mean':drop},
              'policies':{'entropy_sign':{'actual_replacements':swaps,'net_corrections':net,'entropy_utility':{'mean':ent}}}} for c in CELLS}
        return a,b

    def test_first_matching_gates(self):
        a,b=self.outcomes();self.assertEqual(decide(a,b),'GO_PILOT')
        self.assertEqual(decide(a,b,integrity_errors=['bad'],complete=False),'INVALID')
        self.assertEqual(decide(a,b,complete=False),'INCONCLUSIVE')
        a,b=self.outcomes(u=0,d=0,ent=1)
        self.assertEqual(decide(a,b),'STOP') # before mismatch REVISE
        a,b=self.outcomes();b['0.5']['U']['mean']=-1
        self.assertEqual(decide(a,b),'REVISE')
        a,b=self.outcomes(u=-1,d=1,ent=1);self.assertEqual(decide(a,b),'REVISE')
        for change in ({'swaps':2},{'drop':0},{'net':-1}):
            a,b=self.outcomes(**change);self.assertEqual(decide(a,b),'INCONCLUSIVE')
        a,b=self.outcomes(u=1,d=0,ent=-1);self.assertEqual(decide(a,b),'INCONCLUSIVE')

    def test_exact_trim_and_identity_ties(self):
        s=sensitivity(list(range(128)),list(range(128)))
        self.assertEqual(s['symmetric_trim_each_tail'],12)
        self.assertEqual(s['symmetric_trimmed_mean'],63.5)
        self.assertEqual(s['drop_two_largest_mean'],62.5)
        self.assertEqual(s['removed_identities'],[126,127])
        s=sensitivity([1]*4,[3,2,0,1]);self.assertEqual(s['removed_identities'],[2,3])

    def test_bootstrap_query_weighted_and_null_guard(self):
        rows=[{'timestep':t} for t in [0,1,2,64,128,192]]
        u=[0,0,0,10,20,30];d=[2*v for v in u]
        result=uncertainty(rows,u,d)
        self.assertEqual(result['block_count'],4)
        self.assertEqual(result['blocks'][0]['count'],3)
        self.assertEqual(result['leave_one_block_out_range']['U'],[6.,20.])
        a,b=result['bootstrap_95_percentile']['U'],result['bootstrap_95_percentile']['D']
        np.testing.assert_allclose(np.array(a)*2,b)
        self.assertEqual(result,uncertainty(rows,u,d))
        self.assertEqual(uncertainty(rows[:3],u[:3],d[:3])['bootstrap_null_reason'],'too_few_blocks')

    def test_ranking_ties_constant_and_bins(self):
        rows=[{'timestep':0,'candidate_scores':{'entropy_sign':[1,1,3]},'verified_swaps':[{'supervised_utility':v} for v in [2,2,4]]},
              {'timestep':1,'candidate_scores':{'entropy_sign':[0,0]},'verified_swaps':[{'supervised_utility':v} for v in [2,4]]}]
        s=correlations(rows,'entropy_sign')
        self.assertEqual(s['valid_queries'],1);self.assertEqual(s['total_queries'],2)
        self.assertAlmostEqual(s['mean_rho'],1.)
        self.assertEqual(s['per_query'][1]['null_reason'],'constant_vector')
        a={c:[dict(query_entropy=1.,query_gradient_norm=2.,legal_swap_count=25) for _ in range(16)] for c in CELLS}
        bins=freeze_bins(a)
        self.assertEqual(bins['0']['query_entropy'],[1.])
        self.assertEqual(bins['0']['legal_swap_count'],[25.])

    def test_empty_is_incomplete_not_negative_science(self):
        result=analyze({c:[] for c in CELLS},{c:[] for c in CELLS},registry(),None)
        self.assertEqual(result['decision'],'INCONCLUSIVE');self.assertFalse(result['complete'])


if __name__=='__main__': unittest.main()
