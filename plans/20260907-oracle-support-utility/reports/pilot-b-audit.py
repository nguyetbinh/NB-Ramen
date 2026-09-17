"""Recompute screened-query results and verify both cells against their traces."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from evaluation.oracle_support_utility import summarize, average_ranks
EVIDENCE = ROOT/'evidence/oracle-support-pilot-b-resumed-20260907'


def audit(job):
    run = EVIDENCE/job['run_id']
    completion = json.loads((run/'diagnostic-completion.json').read_text())
    assert completion['status'] == 'TARGET_COMPLETE'
    manifest = json.loads((run/'manifest.json').read_text())
    assert manifest['artifacts']['status'] == 'verified'
    cfg = manifest['config']
    assert cfg == {'beta':5., 'candidate_m':10, 'lr':.01, 'max_capacity':750,
                   'optimizer':'signsgd', 'oracle_label_source':'evaluator_known_label',
                   'probe_mode':'screened', 'probe_queries':128, 'topk':5}
    rows = [json.loads(s) for s in (run/'oracle-support-queries.jsonl').read_text().splitlines()]
    trace = [json.loads(s) for s in (run/'trace.jsonl').read_text().splitlines()]
    assert len(rows)==128 and len(trace)%100==0
    assert [r['timestep'] for r in trace]==list(range(len(trace)))
    assert len(trace)<=600<cfg['max_capacity']
    expected=[]; snapshots={}; counts=collections.Counter()
    for end in range(100,len(trace)+1,100):
        counts.update(r['pre_adaptation_prediction'] for r in trace[end-100:end])
        snapshots[end]=counts.copy()
        if max(counts.values())>=10:
            expected.extend(r['timestep'] for r in trace[end-100:end] if not r['is_ood'])
    assert [r['query_index'] for r in rows]==expected[:128]
    for r in rows:
        q=r['query_index']; end=(q//100+1)*100; source=trace[q]
        assert not source['is_ood'] and source['known_label_or_minus_one']==r['true_label']
        assert source['ground_truth_domain']==r['query_domain']
        assert source['correct']==r['ramen']['correct']
        legal={(c,o,i) for c,n in snapshots[end].items() for o in range(min(5,n)) for i in range(min(5,n),min(10,n))}
        def key(s): return (s['class'],s['out'],s['in'])
        verified={key(s) for s in r['verified_swaps']}
        assert r['legal_swap_count']==len(legal)
        assert verified <= legal and len(verified)==r['verified_swap_count']==len(r['verified_swaps'])
        assert 1<=len(verified)<=4
        assert verified=={key(s) for name,s in r['baselines'].items() if name!='feature_similarity'}
        assert r['baselines']['feature_similarity']['exact_utility']==0
        for s in r['verified_swaps']:
            assert abs(s['exact_utility']-(r['ramen']['ce']-s['ce']))<1e-12
            assert s['outgoing']['rank']==s['out']+1 and s['incoming']['rank']==s['in']+1
            assert s['outgoing']['query_index']!=s['incoming']['query_index']
            for side in ('outgoing','incoming'):
                support=s[side]; t=support['query_index']; assert 0<=t<end
                assert trace[t]['pre_adaptation_prediction']==s['class']
                assert trace[t]['is_ood']==support['is_ood']
                assert trace[t]['ground_truth_domain']==support['domain']
        assert r['baselines']['best_predicted']['predicted_utility']==max(s['predicted_utility'] for s in r['verified_swaps'])
        assert r['baselines']['worst_predicted']['predicted_utility']==min(s['predicted_utility'] for s in r['verified_swaps'])
        assert r['baselines']['gradient_cosine']['cosine_gain']==max(s['cosine_gain'] for s in r['verified_swaps'])
        assert abs(r['oracle']['exact_utility']-max(0.,max(s['exact_utility'] for s in r['verified_swaps'])))<1e-12
    summary=summarize(rows,128,'screened')
    assert summary==json.loads((run/'oracle-support-summary.json').read_text())
    if job['pilot']=='b-null':
        assert not any(r['is_ood'] for r in trace)
    extra={'query_domains':dict(collections.Counter(manifest['dataset']['environments'][r['query_domain']] for r in rows)),
           'query_timestep_range':[rows[0]['query_index'],rows[-1]['query_index']],
           'committed_trace_rows':len(trace), 'realized_trace_ood_fraction':sum(r['is_ood'] for r in trace)/len(trace),
           'mean_ramen_ce':float(np.mean([r['ramen']['ce'] for r in rows])),
           'mean_oracle_ce':float(np.mean([r['oracle']['ce'] for r in rows])),
           'ramen_correct':sum(r['ramen']['correct'] for r in rows),
           'oracle_correct':sum(r['oracle']['correct'] for r in rows),
           'verified_swaps':sum(r['verified_swap_count'] for r in rows),
           'legal_swaps_ranked':sum(r['legal_swap_count'] for r in rows),
           'baseline_positive_queries':{name:sum(r['baselines'][name]['exact_utility']>0 for r in rows) for name in rows[0]['baselines']}}
    return {'audit':'PASS_DIAGNOSTIC_TARGET','summary':summary,'extra':extra}, rows


def main():
    plan=json.loads((EVIDENCE/'launch-plan.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in plan['source_sha256'].items())
    results={}; raw={}
    for job in plan['jobs']:
        results[job['pilot']],raw[job['pilot']]=audit(job)
    a_run=ROOT/'evidence/oracle-support-pilot-a-complete-20260907/oracle-support-a-mps-seed0'
    a_rows=[json.loads(s) for s in (a_run/'oracle-support-queries.jsonl').read_text().splitlines()]
    open_rows={r['query_index']:r for r in raw['b-open']}
    for a in a_rows:
        b=open_rows[a['query_index']]
        assert a['true_label']==b['true_label'] and a['ramen']==b['ramen']
        all_a={(s['class'],s['out'],s['in']):s for s in a['verified_swaps']}
        for s in b['verified_swaps']:
            assert s==all_a[(s['class'],s['out'],s['in'])]
    a_gain=sum(r['oracle']['exact_utility'] for r in a_rows)
    b_gain=sum(open_rows[r['query_index']]['oracle']['exact_utility'] for r in a_rows)
    results['pilot_a_overlap']={'queries':16,'baseline_and_verified_swaps_exactly_match':True,
                               'screened_to_exhaustive_gain_ratio':b_gain/a_gain}
    (EVIDENCE/'audit-and-analysis.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2,allow_nan=False))


if __name__=='__main__': main()
