"""Raw-record audit and preregistered QCGS diagnostics; no selector tuning."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random

import numpy as np

from .oracle_support_utility import average_ranks
from runtime.query_gradient_registry import CELLS, STAGES, digest, write_json, validate_registry

POLICIES = ('ramen', 'entropy_sign', 'entropy_cosine', 'random', 'supervised_reference')
ANALYSIS = {'version': 1, 'primary': 'entropy_sign', 'seed': 917, 'bootstrap_draws': 10000,
            'block_size': 64, 'minimum_blocks': 4, 'trim_fraction': .1,
            'remove_largest': 2, 'minimum_replacements': 3, 'threshold': 0.,
            'gate_order': ['INVALID', 'INCONCLUSIVE:incomplete', 'GO_PILOT', 'STOP', 'REVISE', 'INCONCLUSIVE']}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    with tmp.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_query_outputs(method, run_dir):
    root = Path(run_dir)
    write_jsonl(root / 'qcgs-queries.jsonl', method.rows)
    write_jsonl(root / 'qcgs-scan.jsonl', method.scan)
    write_json(root / 'qcgs-state.json', {'parameter_order':method.parameter_order,
               'parameter_order_sha256':method.parameter_order_sha256})
    write_json(root / 'qcgs-status.json', {'schema_version': 1, 'mode': method.mode,
               'scanned': len(method.scan), 'scored': len(method.rows),
               'requested': method.cfg['probe_queries'], 'complete': len(method.rows) == method.cfg['probe_queries']})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def audit_rows(rows, stage, cell, registry=None, config_sha256=None):
    """Reject malformed/nonfinite outcomes; missing rows alone are incomplete.

    Recompute utilities, selected actions, RNG draws, and legal-space coverage.
    This verifies the saved arithmetic; it cannot reconstruct logits from CE.
    """
    rng = random.Random(0)
    expected = None
    if registry is not None:
        validate_registry(registry)
        expected = registry['cells'][cell]['selected'][stage]
    require(len(rows) <= (STAGES[stage] if stage in STAGES else 2), 'too many query records')
    seen = set()
    for position, row in enumerate(rows):
        digest(row)  # Reject JSON NaN/Infinity anywhere, including diagnostic fields.
        require(finite(row['query_entropy']) and row['query_entropy'] >= 0
                and finite(row['query_gradient_norm']) and row['query_gradient_norm'] >= 0, 'invalid query entropy/norm')
        require(row['schema_version'] == 1 and row['stage'] == stage and row['ood_cell'] == cell, 'row schema/stage/cell mismatch')
        require(row['sample_idx'] not in seen, 'duplicate original image')
        seen.add(row['sample_idx'])
        require(row['is_ood'] is False and type(row['known_label']) is int and row['known_label'] >= 0, 'invalid scored ID label')
        require(row['block'] == row['timestep']//64 and row['batch'] == row['timestep']//100, 'batch/block identity mismatch')
        if config_sha256 is not None:
            require(row['config_sha256'] == config_sha256, 'config provenance mismatch')
        if expected is not None:
            require(position < len(expected), 'unexpected query')
            for field in ('timestep', 'sample_idx', 'domain', 'baseline_sha256', 'eligible'):
                require(row[field] == expected[position][field], 'registered query mismatch: '+field)
            require(row['dataset_fingerprint'] == registry['dataset_fingerprint'] and row['source_revision'] == registry['source_revision']
                    and row['registry_sha256'] == registry['sha256'], 'source/dataset/registry provenance mismatch')
        pool = row['candidate_pool']
        require(digest(pool) == row['candidate_pool_sha256'], 'candidate pool hash mismatch')
        actions = []
        require(row['active_class_count'] == len(pool), 'class divisor mismatch')
        for key in sorted(pool, key=int):
            p = pool[key]
            require(0 < p['k'] <= len(p['metadata']) <= 10, 'pool cardinality mismatch')
            require(len({m['timestep'] for m in p['metadata']}) == len(p['metadata']), 'duplicate pool entry')
            for out in range(p['k']):
                for incoming in range(p['k'], len(p['metadata'])):
                    actions.append({'class': int(key), 'out': out, 'in': incoming})
        require(actions == row['legal_actions'] and len(actions) == row['legal_swap_count'] > 0, 'legal action space mismatch')
        require(len(row['direction_changes']) == len(actions) and all(type(v) is bool for v in row['direction_changes']), 'direction flags mismatch')
        require(set(row['policies']) == set(POLICIES), 'missing policy outcomes')
        scores = row['candidate_scores']
        require(set(scores) == set(POLICIES[1:3]) | {'supervised_reference'}, 'score vector schema mismatch')
        for name, vector in scores.items():
            require(len(vector) == len(actions) and (all(v is None for v in vector) or all(finite(v) for v in vector)), 'invalid score vector')
            choice = row['policies'][name]
            require(choice['invalid'] is False, 'numerical integrity failure')
            if name == 'supervised_reference':
                require(all(finite(v) for v in vector), 'missing supervised scores')
                index = max(range(len(vector)), key=vector.__getitem__)
            elif all(v is None for v in vector):
                require(choice['reason'] in ('zero_query_gradient', 'zero_candidate_norm'), 'unexplained missing scores')
                index = None
            else:
                index = max(range(len(vector)), key=vector.__getitem__)
                if vector[index] <= 0:
                    index = None
            require(choice['index'] == index, 'policy is not the frozen canonical argmax: '+name)
            require(close(choice['score'], 0. if index is None else vector[index]), 'selected score mismatch')
            if name != 'supervised_reference' and all(finite(v) for v in vector):
                maximum = max(vector)
                reason = ('positive_score' if index is not None else
                          'all_aggregate_directions_unchanged' if name == 'entropy_sign' and not any(row['direction_changes']) else
                          'negative_maximum' if maximum < 0 else 'zero_maximum_no_swap_wins')
                require(choice['reason'] == reason, 'fallback reason mismatch')
        require(row['random_draw'] == position and row['rng_before_sha256'] == digest(rng.getstate()), 'random RNG provenance mismatch')
        require(row['policies']['random']['index'] == rng.randrange(len(actions)), 'random control is not independent uniform draw')
        require(row['policies']['ramen']['index'] is None, 'Ramen must be no swap')
        verified = {r['index']: r for r in row['verified_swaps']}
        require(len(verified) == len(row['verified_swaps']), 'duplicate verified action')
        require(list(verified) == sorted(verified), 'verified actions not in canonical order')
        needed = {p['index'] for p in row['policies'].values() if p['index'] is not None}
        require(set(verified) == (set(range(len(actions))) if stage == 'stage-a' else needed), 'verification coverage mismatch')
        baseline = row['ramen']
        for trial in [baseline, *verified.values(), *row['policies'].values()]:
            require(all(finite(trial[k]) for k in ('ce', 'entropy', 'margin')), 'nonfinite trial outcome')
            require(type(trial['correct']) is bool and trial['correct'] == (trial['prediction'] == row['known_label']), 'incorrect prediction/correctness arithmetic')
            if 'supervised_utility' in trial:
                require(close(trial['supervised_utility'], baseline['ce'] - trial['ce']), 'CE utility arithmetic mismatch')
                require(close(trial['entropy_utility'], baseline['entropy'] - trial['entropy']), 'entropy utility arithmetic mismatch')
        for index, trial in verified.items():
            require({k: trial[k] for k in ('class', 'out', 'in')} == actions[index], 'verified action identity mismatch')
            require(trial['direction_changed'] == row['direction_changes'][index], 'direction flag mismatch')
            for prefix, rank in (('incoming', actions[index]['in']), ('outgoing', actions[index]['out'])):
                metadata = pool[str(actions[index]['class'])]['metadata'][rank]
                require(all(trial[prefix][k] == v for k, v in metadata.items()), 'support identity mismatch')
                require(trial[prefix]['rank'] == rank, 'support rank mismatch')
            for name in scores:
                require(trial[name+'_score'] == scores[name][index], 'verified score mismatch')
        best = max([0.] + [r['supervised_utility'] for r in verified.values()])
        require(close(row['best_verified_utility'], best), 'best verified arithmetic mismatch')
        require(close(row['exact_oracle_utility'], best) if stage == 'stage-a' else row['exact_oracle_utility'] is None, 'oracle scope mismatch')
        for policy in row['policies'].values():
            original = baseline if policy['index'] is None else verified[policy['index']]
            require(all(policy[k] == original[k] for k in ('prediction', 'correct', 'ce', 'entropy', 'margin')), 'policy outcome differs from verified trial')
            regret = best - policy['supervised_utility']
            require(close(policy['regret_to_verified_set'], regret), 'verified regret mismatch')
            require(close(policy['exact_oracle_regret'], regret) if stage == 'stage-a' else policy['exact_oracle_regret'] is None, 'exact regret scope mismatch')
    return len(rows) == (STAGES[stage] if stage in STAGES else 2)


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'count': 0, 'mean': None, 'median': None, 'p25_p75': None}
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)), 'p25_p75': np.percentile(a, [25,75]).tolist()}


def sensitivity(values, identities):
    order = sorted(range(len(values)), key=lambda i: (values[i], identities[i]))
    n = len(order)
    trim = math.floor(.1*n)
    remaining = order[:-2]
    return {**describe(values), 'symmetric_trim_each_tail': trim,
            'symmetric_trimmed_mean': float(np.mean([values[i] for i in order[trim:n-trim]])) if n else None,
            'drop_two_largest_mean': float(np.mean([values[i] for i in remaining])) if remaining else None,
            'removed_identities': [identities[i] for i in order[-2:]]}


def correlations(rows, name):
    observations = []
    for r in rows:
        x, y = r['candidate_scores'][name], [s['supervised_utility'] for s in r['verified_swaps']]
        reason, rho = None, None
        if len(x) < 2:
            reason = 'fewer_than_two_candidates'
        elif any(v is None or not finite(v) for v in x+y):
            reason = 'undefined_values'
        elif len(set(x)) < 2 or len(set(y)) < 2:
            reason = 'constant_vector'
        else:
            rho = float(np.corrcoef(average_ranks(x), average_ranks(y))[0,1])
        observations.append({'timestep': r['timestep'], 'rho': rho, 'null_reason': reason})
    valid = [o['rho'] for o in observations if o['rho'] is not None]
    return {'per_query': observations, 'valid_queries': len(valid), 'total_queries': len(rows),
            'mean_rho': float(np.mean(valid)) if valid else None, 'median_rho': float(np.median(valid)) if valid else None,
            'positive_rho_count': sum(v > 0 for v in valid),
            'positive_rho_rate_among_valid': sum(v > 0 for v in valid)/len(valid) if valid else None}


def freeze_bins(stage_a):
    require(all(len(stage_a[c]) == 16 for c in CELLS), 'bins require complete Stage A')
    return {c: {field: sorted(set(np.percentile([r[field] for r in stage_a[c]], quantiles).tolist()))
                for field, quantiles in [('query_entropy', [100/3,200/3]), ('query_gradient_norm', [100/3,200/3]), ('legal_swap_count', [50])]}
            for c in CELLS}


def uncertainty(rows, u, d):
    blocks = sorted({r['timestep']//64 for r in rows})
    totals = np.array([[sum(u[i] for i,r in enumerate(rows) if r['timestep']//64 == b),
                        sum(d[i] for i,r in enumerate(rows) if r['timestep']//64 == b),
                        sum(r['timestep']//64 == b for r in rows)] for b in blocks], dtype=float).reshape(-1,3)
    k = len(blocks)
    result = {'block_count': k, 'blocks': [{'block': b, 'count': int(t[2]), 'U_mean': t[0]/t[2], 'D_mean': t[1]/t[2]} for b,t in zip(blocks,totals)],
              'bootstrap_95_percentile': None, 'bootstrap_null_reason': 'too_few_blocks' if k < 4 else None,
              'leave_one_block_out_range': None, 'leave_one_out_null_reason': 'fewer_than_two_blocks' if k < 2 else None,
              'interpretation': 'conditional descriptive one-stream intervals; not independent-run confidence'}
    if k >= 2:
        removed = totals.sum(0) - totals
        means = removed[:,:2]/removed[:,2,None]
        result['leave_one_block_out_range'] = {'U': [float(means[:,0].min()), float(means[:,0].max())], 'D': [float(means[:,1].min()), float(means[:,1].max())]}
    if k >= 4:
        rng = np.random.default_rng(917)
        draws = totals[rng.integers(0,k,size=(10000,k))].sum(1)
        means = draws[:,:2]/draws[:,2,None]
        result['bootstrap_95_percentile'] = dict(zip(('U','D'), np.percentile(means, [2.5,97.5], axis=0).T.tolist()))
    return result


def policy_summary(rows, name):
    n = len(rows)
    outcomes = [r['policies'][name] for r in rows]
    u = [p['supervised_utility'] for p in outcomes]
    e = [p['entropy_utility'] for p in outcomes]
    swaps = sum(p['index'] is not None for p in outcomes)
    w2c = sum(not r['ramen']['correct'] and p['correct'] for r,p in zip(rows,outcomes))
    c2w = sum(r['ramen']['correct'] and not p['correct'] for r,p in zip(rows,outcomes))
    rate = lambda count: count/n if n else None
    return {**describe(u), 'helpful_rate': rate(sum(v>0 for v in u)), 'harmful_rate': rate(sum(v<0 for v in u)),
            'no_swap_rate': rate(n-swaps), 'no_swap_reasons': dict(Counter(p['reason'] for p in outcomes if p['index'] is None)),
            'actual_replacements': swaps, 'replacement_coverage': rate(swaps), 'accuracy': rate(sum(p['correct'] for p in outcomes)),
            'wrong_to_correct': w2c, 'correct_to_wrong': c2w, 'net_corrections': w2c-c2w,
            'entropy_utility': describe(e), 'lower_entropy_worse_ce_rate': rate(sum(a>0 and b<0 for a,b in zip(e,u))),
            'regret_to_verified_set': describe([p['regret_to_verified_set'] for p in outcomes]),
            'exact_oracle_regret': describe([p['exact_oracle_regret'] for p in outcomes]) if rows and rows[0]['stage']=='stage-a' else None}


def summarize_cell(rows, stage, bins=None):
    result = {'queries': len(rows), 'policies': {p: policy_summary(rows,p) for p in POLICIES},
              'exact_oracle': describe([r['exact_oracle_utility'] for r in rows]) if stage=='stage-a' else None,
              'best_verified_subset': describe([r['best_verified_utility'] for r in rows]),
              'costs': {k: describe([r[k] for r in rows if r[k] is not None]) for k in
                        ('score_latency_ms','parity_audit_ms','scoring_peak_extra_bytes','evaluator_peak_extra_bytes','retained_memory_bytes','peak_device_memory_bytes')}}
    batches = {r['batch']: r for r in rows}
    result['costs']['batch_totals_ms'] = {k: sum(r[k] for r in batches.values()) for k in ('retrieval_batch_ms','supervised_reference_batch_ms','baseline_replay_batch_ms','verification_batch_ms')}
    result['costs']['verification_full_batch_forwards'] = sum(r['verification_batch_forward_count'] for r in batches.values())
    result['costs']['note'] = 'score latency includes pool/aggregate/score; retrieval shared per batch. Scoring peak includes label-free diagnostic bookkeeping/audit, an upper bound, not standalone inference memory. Verification includes grouped-parity audit.'
    if stage == 'stage-a':
        result['ranking'] = {name: correlations(rows,name) for name in ('entropy_sign','entropy_cosine')}
    if stage == 'stage-b':
        u = [r['policies']['entropy_sign']['supervised_utility'] for r in rows]
        d = [a-r['policies']['random']['supervised_utility'] for a,r in zip(u,rows)]
        ids = [(r['sample_idx'],r['domain'],r['timestep']) for r in rows]
        result['U'], result['D'] = sensitivity(u,ids), sensitivity(d,ids)
        result['uncertainty'] = uncertainty(rows,u,d)
        selected = [i for i,r in enumerate(rows) if r['policies']['entropy_sign']['index'] is not None]
        result['actual_swap_queries_only'] = {'U': describe([u[i] for i in selected]), 'D': describe([d[i] for i in selected])}
        strata = {}
        for field in ('baseline_correct','self_support_present',*(bins or {})):
            groups = {}
            for r in rows:
                if field == 'baseline_correct':
                    key = str(r['ramen']['correct'])
                elif field == 'self_support_present':
                    key = str(r[field])
                else:
                    key = str(int(np.searchsorted(bins[field],r[field],side='right')))
                groups.setdefault(key,[]).append(r)
            strata[field] = {key: {p: policy_summary(group,p) for p in POLICIES} for key,group in groups.items()}
        result['strata'] = strata
        result['frozen_bin_edges'] = bins
        result['bin_rule'] = 'searchsorted side=right; sorted unique Stage A boundaries'
    return result


def decide(stage_a, stage_b, *, integrity_errors=(), complete=True):
    if integrity_errors:
        return 'INVALID'
    if not complete:
        return 'INCONCLUSIVE'
    positive = [stage_b[c]['U']['mean'] > 0 and stage_b[c]['D']['mean'] > 0 for c in CELLS]
    if (all(positive) and sum(stage_b[c]['policies']['entropy_sign']['net_corrections'] for c in CELLS) >= 0
        and all(stage_b[c]['policies']['entropy_sign']['actual_replacements'] >= 3
                and stage_b[c]['U']['drop_two_largest_mean'] > 0 and stage_b[c]['D']['drop_two_largest_mean'] > 0 for c in CELLS)):
        return 'GO_PILOT'
    if all(stage_b[c]['U']['mean'] <= 0 and stage_b[c]['D']['mean'] <= 0 for c in CELLS):
        return 'STOP'
    mismatch = any(stage_b[c]['policies']['entropy_sign']['entropy_utility']['mean'] > 0 and stage_b[c]['U']['mean'] <= 0 for c in CELLS)
    if all(stage_a[c]['exact_oracle']['mean'] > 0 for c in CELLS) and (sum(positive)==1 or mismatch):
        return 'REVISE'
    return 'INCONCLUSIVE'


def analyze(stage_a, stage_b, registry, bins, *, integrity_errors=()):
    errors = list(integrity_errors)
    complete = True
    for stage, cells in (('stage-a',stage_a),('stage-b',stage_b)):
        for cell in CELLS:
            try:
                complete = audit_rows(cells.get(cell,[]),stage,cell,registry) and complete
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                errors.append(f'{stage}/{cell}: {exc}')
    if not errors and all(len(stage_a.get(c,[]))==16 for c in CELLS):
        if bins is None and not any(stage_b.get(c,[]) for c in CELLS):
            complete = False
        elif bins != freeze_bins(stage_a):
            errors.append('Stage A bins changed or not frozen')
    summary = {'schema_version':1, 'analysis':ANALYSIS, 'integrity_errors':errors, 'complete':complete and not errors,
               'registry_sha256': registry.get('sha256') if registry else None, 'stage_a':{}, 'stage_b':{}}
    if not errors:
        summary['stage_a'] = {c:summarize_cell(stage_a.get(c,[]),'stage-a') for c in CELLS}
        summary['stage_b'] = {c:summarize_cell(stage_b.get(c,[]),'stage-b',(bins or {}).get(c)) for c in CELLS}
    summary['decision'] = decide(summary['stage_a'],summary['stage_b'],integrity_errors=errors,complete=complete)
    return summary


def write_report(root, summary):
    root = Path(root)
    write_json(root/'analysis.json',summary)
    lines = ['# QCGS label-free diagnostic', '', '**Decision: '+summary['decision']+'**.', '',
             f'Complete protocol: {summary["complete"]}. Integrity errors: {len(summary["integrity_errors"])}.', '',
             'Raw records: [queries.jsonl](queries.jsonl). Frozen inputs and commit history: [locks](locks/). '
             'Recomputed metrics: [analysis.json](analysis.json). Run provenance: [runs](runs/).', '']
    if summary['integrity_errors']:
        lines += ['- '+str(e) for e in summary['integrity_errors']]
    for stage, values in (('Stage A',summary.get('stage_a',{})),('Stage B',summary.get('stage_b',{}))):
        for cell,s in values.items():
            lines.append(f'- {stage}, OOD={cell}: {s["queries"]} queries; entropy-sign mean U={s["policies"]["entropy_sign"]["mean"]}; replacements={s["policies"]["entropy_sign"]["actual_replacements"]}.')
            if stage=='Stage A':
                lines.append(f'  Exact one-swap oracle mean={s["exact_oracle"]["mean"]}; per-query ranking and valid denominators are in analysis.json.')
            elif s['queries']:
                lines.append(f'  Mean D={s["D"]["mean"]}; drop-two U/D={s["U"]["drop_two_largest_mean"]}/{s["D"]["drop_two_largest_mean"]}; net corrections={s["policies"]["entropy_sign"]["net_corrections"]}.')
    lines += ['', 'Stage A verifies all legal swaps. Stage B measures only the selected-action union; its best-verified utility/regret are lower bounds, not an exact oracle.', '',
              'One seed, one cache trajectory per OOD cell, shared memory and potentially repeated images across OOD cells. Block intervals are descriptive and conditional. No causal OOD or cross-dataset generalization claim. No thresholds are fitted on Stage B.', '']
    (root/'report.md').write_text('\n'.join(lines))
