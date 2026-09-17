import collections
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.oracle_support_utility import summarize
root=ROOT
evidence=root/'evidence/oracle-support-pilot-a-complete-20260907'
run=evidence/'oracle-support-a-mps-seed0'
launch=json.loads((evidence/'launch-plan.json').read_text())
assert all(hashlib.sha256((evidence/'source'/p).read_bytes()).hexdigest()==h for p,h in launch['source_sha256'].items())
manifest=json.loads((run/'manifest.json').read_text())
assert manifest['artifacts']['status']=='verified'
assert manifest['config']['probe_mode']=='exhaustive'
assert (manifest['config']['topk'],manifest['config']['candidate_m'],manifest['config']['probe_queries'])==(5,10,16)
rows=[json.loads(s) for s in (run/'oracle-support-queries.jsonl').read_text().splitlines()]
trace=[json.loads(s) for s in (run/'trace.jsonl').read_text().splitlines()]
assert len(rows)==16, len(rows)
assert len(trace) >= 200 and len(trace) % 100 == 0, len(trace)
assert [r['timestep'] for r in trace]==list(range(len(trace)))
counts=collections.Counter()
expected=[]
snapshots={}
for end in range(100,len(trace)+1,100):
    counts.update(r['pre_adaptation_prediction'] for r in trace[end-100:end])
    snapshots[end]=counts.copy()
    if max(counts.values())>=10:
        expected.extend(r['timestep'] for r in trace[end-100:end] if not r['is_ood'])
assert [r['query_index'] for r in rows]==expected[:16]
for row in rows:
    q=row['query_index']; source=trace[q]; end=(q//100+1)*100
    assert row['true_label']==source['known_label_or_minus_one'] and not source['is_ood']
    assert row['query_domain']==source['ground_truth_domain']
    assert row['ramen']['correct']==source['correct']
    expected_swaps={(c,o,i) for c,n in snapshots[end].items() for o in range(min(5,n)) for i in range(min(5,n),min(10,n))}
    swaps=row['verified_swaps']
    assert {(s['class'],s['out'],s['in']) for s in swaps}==expected_swaps
    assert len(swaps)==row['legal_swap_count']==row['verified_swap_count']==len(expected_swaps)
    for s in swaps:
        assert s['outgoing']['rank']==s['out']+1 and s['incoming']['rank']==s['in']+1
        for side in ['outgoing','incoming']:
            support=s[side]; t=support['query_index']; assert 0<=t<end
            assert trace[t]['pre_adaptation_prediction']==s['class']
            assert trace[t]['is_ood']==support['is_ood']
            assert trace[t]['ground_truth_domain']==support['domain']
        assert s['outgoing']['query_index']!=s['incoming']['query_index']
        assert abs(s['exact_utility']-(row['ramen']['ce']-s['ce']))<1e-12
    assert abs(row['oracle']['exact_utility']-max(0.,max(s['exact_utility'] for s in swaps)))<1e-12
summary=summarize(rows,16,'exhaustive')
assert summary==json.loads((run/'oracle-support-summary.json').read_text())
audit={'status':'PASS_DIAGNOSTIC_TARGET', 'full_600_sample_run_complete':len(trace)==600,'eligible_queries':len(rows),'trace_rows':len(trace),
       'earliest_eligible_queries':expected[:16], 'all_legal_swaps_verified':sum(r['verified_swap_count'] for r in rows),
       'unique_nonbaseline_directions':sum(r['exact_forward_count'] for r in rows),
       'source_hashes_match_launch':True,'support_provenance_matches_trace':True,
       'complete_earliest_query_selection':True,'summary_recomputed_from_raw_rows':True}
(evidence/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'audit':audit,'summary':summary},indent=2))
