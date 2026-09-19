"""Gate ordering, unique identities and fail-closed campaign acceptance."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from runtime.query_gradient_registry import build_registry,validate_registry,CELLS,write_json,file_sha,digest
from runtime.query_gradient_campaign import make_job,ledger_commit,audit_campaign
from evaluation.query_gradient_multiview import stage_a_decision,stage_b_decision,paired_uncertainty,analyze,ANALYSIS


def registry():
    scans={c:[dict(timestep=i,sample_idx=i//2,domain=i%2,is_ood=False,eligible=i>=10,
                   baseline_sha256='b'*64,aggregate_sha256='c'*64) for i in range(1600)] for c in CELLS}
    return build_registry(scans,'a'*64,{'base_image_indices':list(range(20))},source_revision='f'*40,
                          provenance={},namespace='qcgs-multiview-rescue-v1')


class AnalysisTests(unittest.TestCase):
    def outcomes(self):
        return {c:{'exact_oracle':{'mean':1.},'U':{'mean':1.,'drop_two_largest_mean':1.},
                   'D':{'mean':1.,'drop_two_largest_mean':1.},'E':{'mean':1.,'drop_two_largest_mean':1.},
                   'policies':{'mv_target':{'actual_replacements':3,'net_corrections':0}},
                   'ranking':{'mv_target':{'valid_queries':8,'mean_rho':.1}}} for c in CELLS}

    def test_ordered_gates_no_revise_and_each_independent_sensitivity(self):
        cells=self.outcomes()
        self.assertEqual(stage_a_decision(cells),'GO_CONFIRM');self.assertEqual(stage_b_decision(cells),'GO_PILOT')
        for decide in (stage_a_decision,stage_b_decision):
            self.assertEqual(decide(cells,integrity_errors=['bad'],complete=False),'INVALID')
            self.assertEqual(decide(cells,complete=False),'INCONCLUSIVE')
        for name in ('U','D','E'):
            for cell in CELLS:
                bad=copy.deepcopy(cells);bad[cell][name]['mean']=0
                self.assertEqual(stage_a_decision(bad),'STOP');self.assertEqual(stage_b_decision(bad),'STOP')
                bad=copy.deepcopy(cells);bad[cell][name]['drop_two_largest_mean']=0
                self.assertEqual(stage_b_decision(bad),'STOP')
        bad=copy.deepcopy(cells);bad['0']['ranking']['mv_target']['valid_queries']=7
        self.assertEqual(stage_a_decision(bad),'STOP')
        bad=copy.deepcopy(cells);bad['0']['ranking']['mv_target']['mean_rho']=0
        self.assertEqual(stage_a_decision(bad),'STOP')
        bad=copy.deepcopy(cells);bad['0']['policies']['mv_target']['actual_replacements']=2
        self.assertEqual(stage_a_decision(bad),'STOP');self.assertEqual(stage_b_decision(bad),'STOP')
        bad=copy.deepcopy(cells);bad['0']['policies']['mv_target']['net_corrections']=-1
        self.assertEqual(stage_b_decision(bad),'STOP')

    def test_three_metrics_use_paired_query_weighted_blocks(self):
        rows=[{'timestep':t} for t in (0,1,2,64,128,192)]
        u=[0,0,0,10,20,30];values={'U':u,'D':[2*x for x in u],'E':[-x for x in u]}
        result=paired_uncertainty(rows,values)
        self.assertEqual(result['blocks'][0]['count'],3)
        self.assertEqual(result['leave_one_block_out_range']['U'],[6.,20.])
        intervals=result['bootstrap_95_percentile']
        np.testing.assert_allclose(np.array(intervals['U'])*2,intervals['D'])
        np.testing.assert_allclose(-np.array(intervals['U'])[::-1],intervals['E'])
        self.assertEqual(result,paired_uncertainty(rows,values))

    def test_new_namespace_disjoint_earliest_images_and_shortest_prefix(self):
        r=registry();validate_registry(r)
        self.assertEqual(r['namespace'],'qcgs-multiview-rescue-v1')
        a={v['sample_idx'] for c in CELLS for v in r['cells'][c]['selected']['stage-a']}
        b={v['sample_idx'] for c in CELLS for v in r['cells'][c]['selected']['stage-b']}
        self.assertFalse(a&b);self.assertFalse((a|b)&set(range(20)))
        bad=copy.deepcopy(r);bad.pop('sha256');bad['cells']['0']['max_eval_samples']+=100
        with self.assertRaisesRegex(ValueError,'shortest'):validate_registry(bad)
        bad=copy.deepcopy(r);bad.pop('sha256');bad['namespace']='qcgs-label-free-v1'
        with self.assertRaisesRegex(ValueError,'namespace'):validate_registry(bad)
        empty=analyze({c:[] for c in CELLS},{c:[] for c in CELLS},r)
        self.assertEqual(empty['decision'],'INCONCLUSIVE');self.assertFalse(empty['complete'])

    def test_offline_campaign_rejects_b_without_a_committed_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);locks=root/'locks';locks.mkdir()
            r=registry();write_json(locks/'registry.json',r)
            write_json(locks/'preflight.json',{'source':{'revision':'f'*40},'analysis':ANALYSIS,'artifacts':{}})
            jobs=[make_job(locks,root/'runs',root/'data',s,c,r['cells'][c]['max_eval_samples'],'a'*64,'f'*40,locks/'registry.json',rescue=True)
                  for s in ('stage-a','stage-b') for c in CELLS]
            write_json(locks/'launch.json',{'jobs':jobs,'source_revision':'f'*40,'registry_sha256':r['sha256'],
                       'registry_file_sha256':file_sha(locks/'registry.json'),'preflight_sha256':file_sha(locks/'preflight.json'),
                       'analysis_sha256':digest(ANALYSIS)})
            ledger_commit(locks,'test: freeze unexecuted rescue inputs')
            self.assertEqual(audit_campaign(root,rescue=True)['decision'],'INCONCLUSIVE')
            (root/'runs'/jobs[-1]['name']).mkdir(parents=True)
            result=audit_campaign(root,rescue=True)
            self.assertEqual(result['decision'],'INVALID')
            self.assertTrue(any('without GO_CONFIRM' in x for x in result['integrity_errors']))
            write_json(locks/'stage-a-gate.json',{'decision':'GO_CONFIRM'})
            ledger_commit(locks,'test: corrupt the gate')
            self.assertEqual(audit_campaign(root,rescue=True)['decision'],'INVALID')


if __name__=='__main__':unittest.main()
