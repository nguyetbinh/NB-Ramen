"""Rescue-only raw audit and locked gates. No score or threshold fitting."""
import hashlib
import math
from pathlib import Path

import numpy as np

from .query_gradient_utility import (audit_rows as audit_actions, describe, sensitivity, correlations,
    policy_summary, require, close, finite, write_json, write_jsonl, read_jsonl)
from runtime.query_gradient_registry import CELLS, STAGES, digest, validate_registry
from methods.query_gradient_multiview import VIEW_SPEC

POLICIES = ('ramen', 'entropy_sign', 'entropy_ramen', 'mv_target', 'random_forced', 'random_matched', 'supervised_reference')
SCORES = ('entropy_sign', 'entropy_ramen', 'mv_target', 'supervised_reference')
ANALYSIS = {'version': 2, 'diagnostic': 'qcgs-multiview', 'primary': 'mv_target',
            'policies': list(POLICIES), 'prediction_controls': ['frozen_mv', 'ramen_mv'],
            'view_spec': VIEW_SPEC, 'registry_namespace': 'qcgs-multiview-rescue-v1',
            'seed': 917, 'bootstrap_draws': 10000, 'block_size': 64, 'minimum_blocks': 4,
            'remove_largest': 2, 'minimum_replacements': 3, 'threshold': 0.,
            'stage_a_minimum_defined_rho': 8,
            'stage_a_gate_order': ['INVALID', 'INCONCLUSIVE:incomplete', 'GO_CONFIRM', 'STOP'],
            'stage_b_gate_order': ['INVALID', 'INCONCLUSIVE:incomplete', 'GO_PILOT', 'STOP']}


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def audit_rows(rows, stage, cell, registry=None, config_sha256=None):
    if registry is not None:
        require(registry.get('namespace') == ANALYSIS['registry_namespace'], 'wrong multi-view registry namespace')
    complete = audit_actions(rows, stage, cell, registry, config_sha256, schema_version=2,
                             policies=POLICIES, score_names=SCORES, forced_reference=False,
                             random_policy='random_forced', sign_names=SCORES[:-1])
    batch_shared = {}
    for row in rows:
        v = row['multiview']; state = row['view_state']
        require(state['view_spec'] == VIEW_SPEC and state['preprocess'] is not None, 'view specification mismatch')
        require(state['sha256'] == digest({k: a for k, a in state.items() if k != 'sha256'}), 'view state hash mismatch')
        require(v['anchor_logits_sha256'] == row['baseline_sha256'], 'wrong Ramen anchor')
        for key in ('raw_image_sha256', 'input_sha256', 'theta_0_parameter_sha256', 'theta_r_parameter_sha256', 'target_sha256'):
            require(sha(v[key]), 'missing input/anchor/target provenance: ' + key)
        require(len(v['view_tensor_sha256']) == 8 and all(map(sha, v['view_tensor_sha256'])), 'missing ordered view hashes')
        target = np.asarray(v['target'], dtype=np.float32)
        require(target.ndim == 1 and len(target) > row['known_label'] and np.isfinite(target).all()
                and (target >= 0).all() and np.isclose(target.sum(), 1., rtol=1e-6, atol=1e-6), 'invalid soft target')
        require(hashlib.sha256(target.tobytes()).hexdigest() == v['target_sha256'], 'target content hash mismatch')
        require(set(v['query_signals']) == set(SCORES[:-1]), 'query signal schema mismatch')
        for name, signal in v['query_signals'].items():
            require(sha(signal['sha256']) and finite(signal['norm']) and signal['norm'] >= 0, 'invalid query signal')
            missing = all(a is None for a in row['candidate_scores'][name])
            require(missing == (signal['norm'] == 0), 'zero-norm fallback mismatch')
        primary = row['policies']['mv_target']; forced = row['policies']['random_forced']; matched = row['policies']['random_matched']
        require(matched['index'] == (forced['index'] if primary['index'] is not None else None), 'random-matched abstention/draw mismatch')
        require(row['primary_verification_required'] == (primary['index'] is not None), 'primary verification cost mismatch')
        require(forced['reason'] == 'uniform_random' and forced['score'] is None and forced['invalid'] is False, 'forced random contract mismatch')
        require(matched['reason'] == ('uniform_random' if primary['index'] is not None else 'primary_no_swap')
                and matched['score'] == (None if primary['index'] is not None else 0.) and matched['invalid'] is False, 'matched random contract mismatch')
        for trial in [row['ramen'], *row['verified_swaps'], *row['policies'].values()]:
            require(finite(trial['soft_target_ce']) and trial['soft_target_ce'] >= 0, 'invalid soft-target CE')
        trials = {p['index']: p for p in row['verified_swaps']}
        for policy in row['policies'].values():
            original = row['ramen'] if policy['index'] is None else trials[policy['index']]
            require(close(original['soft_target_ce'], policy['soft_target_ce']), 'soft-target policy outcome mismatch')
        require(set(row['prediction_controls']) == {'frozen_mv', 'ramen_mv'}, 'prediction controls missing')
        for name, control in row['prediction_controls'].items():
            require(not {'index', 'exact_oracle_regret', 'regret_to_verified_set'} & set(control), 'ensemble is not a legal swap action')
            lp = np.asarray(control['log_probabilities'], dtype=np.float64)
            require(lp.shape == target.shape and np.isfinite(lp).all() and np.isclose(np.exp(lp).sum(), 1., atol=1e-6), 'invalid ensemble log probabilities')
            # Stable log probabilities are saved before CE, with no clipping.
            normalized = lp - np.logaddexp.reduce(lp)
            pred = int(np.argmax(lp)); label = row['known_label']
            entropy = -float((np.exp(normalized) * normalized).sum())
            margin = float(lp[label] - np.max(np.delete(lp, label)))
            for field, expected in (('ce', -normalized[label]), ('entropy', entropy),
                                    ('margin', margin), ('soft_target_ce', -float(target @ normalized))):
                require(finite(control[field]) and math.isclose(control[field], expected, rel_tol=2e-6, abs_tol=2e-6), 'ensemble arithmetic mismatch: ' + field)
            require(control['prediction'] == pred and control['correct'] == (pred == label), 'ensemble prediction mismatch')
            if name == 'frozen_mv':
                require(np.allclose(np.exp(lp), target, rtol=2e-6, atol=1e-7), 'teacher target differs from frozen ensemble')
        costs = row['side_costs']
        for name, expected in {'teacher_full_batch_forwards': 8, 'target_full_batch_forwards': 1,
                              'target_backwards': 1, 'entropy_anchor_forwards': 1, 'entropy_anchor_backwards': 1,
                              'ramen_mv_full_batch_forwards': 8, 'anchor_install_forwards': 1,
                              'supervised_anchor_forwards': 1, 'supervised_reference_forwards': 1,
                              'supervised_reference_backwards': 1, 'baseline_replay_forwards': 1}.items():
            require(costs[name] == expected, 'view/gradient cost accounting mismatch')
        require(all(a is None or (finite(a) and a >= 0) for a in costs.values()), 'invalid costs')
        shared = {'costs': costs, 'state': state, 'theta_0': v['theta_0_parameter_sha256'], 'theta_R': v['theta_r_parameter_sha256']}
        require(batch_shared.setdefault(row['batch'], shared) == shared, 'batch anchor/cost mapping mismatch')
    return complete


def paired_uncertainty(rows, values):
    names = ('U', 'D', 'E')
    blocks = sorted({r['timestep'] // 64 for r in rows})
    totals = np.array([[*[sum(values[n][i] for i, r in enumerate(rows) if r['timestep']//64 == b) for n in names],
                        sum(r['timestep']//64 == b for r in rows)] for b in blocks], dtype=float).reshape(-1, 4)
    k = len(blocks)
    result = {'block_count': k, 'blocks': [{'block': b, 'count': int(t[3]), **{n+'_mean': t[j]/t[3] for j,n in enumerate(names)}} for b,t in zip(blocks,totals)],
              'bootstrap_95_percentile': None, 'bootstrap_null_reason': 'too_few_blocks' if k < 4 else None,
              'leave_one_block_out_range': None,
              'interpretation': 'paired query-weighted, conditional on one stream; not independent-run confidence'}
    if k >= 2:
        removed = totals.sum(0) - totals
        means = removed[:, :3] / removed[:, 3, None]
        result['leave_one_block_out_range'] = {n: [float(means[:,j].min()), float(means[:,j].max())] for j,n in enumerate(names)}
    if k >= 4:
        rng = np.random.default_rng(917)
        draws = totals[rng.integers(0, k, size=(10000, k))].sum(1)
        result['bootstrap_95_percentile'] = dict(zip(names, np.percentile(draws[:,:3]/draws[:,3,None], [2.5,97.5], axis=0).T.tolist()))
    return result


def summarize_cell(rows, stage):
    values = {k: [] for k in ('U', 'D', 'E')}
    for r in rows:
        ce = r['policies']['mv_target']['ce']
        values['U'].append(r['ramen']['ce'] - ce)
        values['D'].append(r['policies']['random_matched']['ce'] - ce)
        values['E'].append(r['prediction_controls']['ramen_mv']['ce'] - ce)
    ids = [(r['sample_idx'], r['domain'], r['timestep']) for r in rows]
    result = {'queries': len(rows), **{n: sensitivity(v, ids) for n,v in values.items()},
              'policies': {n: policy_summary(rows, n) for n in POLICIES},
              'exact_oracle': describe([r['exact_oracle_utility'] for r in rows]) if stage == 'stage-a' else None,
              'best_verified_subset': describe([r['best_verified_utility'] for r in rows]),
              'uncertainty': paired_uncertainty(rows, values)}
    selected = [i for i,r in enumerate(rows) if r['policies']['mv_target']['index'] is not None]
    result['actual_swap_queries_only'] = {n: describe([v[i] for i in selected]) for n,v in values.items()}
    result['prediction_controls'] = {n: {'ce': describe([r['prediction_controls'][n]['ce'] for r in rows]),
        'gain_vs_ramen': describe([r['ramen']['ce']-r['prediction_controls'][n]['ce'] for r in rows]),
        'accuracy': float(np.mean([r['prediction_controls'][n]['correct'] for r in rows])) if rows else None}
        for n in ('frozen_mv', 'ramen_mv')}
    result['soft_target_gain'] = {n: describe([r['ramen']['soft_target_ce']-r['policies'][n]['soft_target_ce'] for r in rows]) for n in POLICIES}
    for name, summary in result['policies'].items():
        gains = [r['policies'][name]['supervised_utility'] for r in rows]
        summary['sensitivity'] = sensitivity(gains, ids)
        summary['helpful_magnitude'] = describe([g for g in gains if g > 0])
        summary['harmful_magnitude'] = describe([-g for g in gains if g < 0])
    result['primary_gain_over_controls'] = {
        name: describe([r['policies'][name]['ce']-r['policies']['mv_target']['ce'] for r in rows])
        for name in ('entropy_sign', 'entropy_ramen', 'random_forced')}
    result['primary_gain_over_controls']['frozen_mv'] = describe([
        r['prediction_controls']['frozen_mv']['ce']-r['policies']['mv_target']['ce'] for r in rows])
    wrong = [r for r in rows if not r['ramen']['correct']]
    probability_gain = [r['multiview']['target'][r['known_label']]-math.exp(-r['ramen']['ce']) for r in wrong]
    result['teacher_on_ramen_errors'] = {'queries': len(wrong), 'true_label_probability_gain': describe(probability_gain),
        'increased_true_label_probability': sum(g > 0 for g in probability_gain),
        'teacher_correct_count': sum(r['prediction_controls']['frozen_mv']['correct'] for r in wrong),
        'primary_correct_count': sum(r['policies']['mv_target']['correct'] for r in wrong)}
    if stage == 'stage-a':
        result['ranking'] = {n: correlations(rows, n) for n in SCORES}
    batches = {r['batch']: r for r in rows}
    result['costs'] = {'batch_totals': {k: sum(r['side_costs'][k] for r in batches.values()) for k in
                       ('teacher_batch_ms', 'target_gradient_batch_ms', 'entropy_anchor_batch_ms', 'ramen_mv_batch_ms',
                        'teacher_full_batch_forwards', 'target_full_batch_forwards', 'target_backwards',
                        'entropy_anchor_forwards', 'entropy_anchor_backwards', 'ramen_mv_full_batch_forwards', 'anchor_install_forwards',
                        'supervised_anchor_forwards', 'supervised_reference_forwards', 'supervised_reference_backwards', 'baseline_replay_forwards')},
        'shared_batch_ms': {k: sum(r[k] for r in batches.values()) for k in
                            ('retrieval_batch_ms', 'supervised_reference_batch_ms', 'baseline_replay_batch_ms', 'verification_batch_ms')},
        'primary_selected_query_verifications': sum(r['primary_verification_required'] for r in rows),
        'primary_side_peak_extra_bytes': describe([r['side_costs']['primary_side_peak_extra_bytes'] for r in batches.values()
                                                  if r['side_costs']['primary_side_peak_extra_bytes'] is not None]),
        'verification_batch_forward_count': sum(r['verification_batch_forward_count'] for r in batches.values()),
        'query_costs': {k: describe([r[k] for r in rows if r[k] is not None]) for k in
                        ('score_latency_ms', 'scoring_peak_extra_bytes', 'evaluator_peak_extra_bytes', 'peak_device_memory_bytes')},
        'note': 'Full-batch side forwards retain original normalization slots. Teacher and primary target gradient costs are separate from controls; no equal-latency or standalone deployment claim.'}
    result['baseline_correct_strata'] = {str(c): {n: describe([v[i] for i,r in enumerate(rows) if r['ramen']['correct'] == c]) for n,v in values.items()} for c in (False, True)}
    return result


def stage_a_decision(cells, *, integrity_errors=(), complete=True):
    if integrity_errors: return 'INVALID'
    if not complete: return 'INCONCLUSIVE'
    for c in CELLS:
        s = cells[c]; ranking = s['ranking']['mv_target']
        if not (s['exact_oracle']['mean'] > 0 and all(s[n]['mean'] > 0 for n in ('U','D','E'))
                and s['policies']['mv_target']['actual_replacements'] >= 3
                and ranking['valid_queries'] >= 8 and ranking['mean_rho'] is not None and ranking['mean_rho'] > 0):
            return 'STOP'
    return 'GO_CONFIRM'


def stage_b_decision(cells, *, integrity_errors=(), complete=True):
    if integrity_errors: return 'INVALID'
    if not complete: return 'INCONCLUSIVE'
    for c in CELLS:
        s = cells[c]
        if not (all(s[n]['mean'] > 0 and s[n]['drop_two_largest_mean'] > 0 for n in ('U','D','E'))
                and s['policies']['mv_target']['actual_replacements'] >= 3):
            return 'STOP'
    return 'GO_PILOT' if sum(cells[c]['policies']['mv_target']['net_corrections'] for c in CELLS) >= 0 else 'STOP'


def stage_a_gate(stage_a, registry):
    complete = True
    for cell in CELLS:
        complete &= audit_rows(stage_a.get(cell, []), 'stage-a', cell, registry)
    cells = {c: summarize_cell(stage_a.get(c, []), 'stage-a') for c in CELLS}
    # No timings in the gate: it is exactly reproducible from scientific raw records.
    inputs = {c: {k: cells[c][k] for k in ('queries','U','D','E','exact_oracle','ranking','policies')} for c in CELLS}
    return {'schema_version': 2, 'analysis_sha256': digest(ANALYSIS), 'registry_sha256': registry['sha256'],
            'stage_a_raw_sha256': digest(stage_a), 'inputs': inputs,
            'decision': stage_a_decision(cells, complete=complete)}


def analyze(stage_a, stage_b, registry, bins=None, *, integrity_errors=()):
    errors = list(integrity_errors); complete = {}
    try:
        validate_registry(registry)
        for name, rows in (('stage-a', stage_a), ('stage-b', stage_b)):
            complete[name] = all([audit_rows(rows.get(c, []), name, c, registry) for c in CELLS])
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc))
    if errors:
        return {'schema_version': 2, 'decision': 'INVALID', 'complete': False, 'integrity_errors': errors, 'stage_a': {}, 'stage_b': {}}
    a = {c: summarize_cell(stage_a.get(c, []), 'stage-a') for c in CELLS}
    b = {c: summarize_cell(stage_b.get(c, []), 'stage-b') for c in CELLS}
    gate = stage_a_decision(a, complete=complete['stage-a'])
    if gate != 'GO_CONFIRM' and any(stage_b.get(c) for c in CELLS):
        errors.append('Stage B collected without a passing Stage A gate')
    decision = ('INVALID' if errors else gate if gate != 'GO_CONFIRM' else stage_b_decision(b, complete=complete['stage-b']))
    overlap = {s: sorted({r['sample_idx'] for r in data.get('0', [])} & {r['sample_idx'] for r in data.get('0.5', [])}) for s,data in (('stage-a',stage_a),('stage-b',stage_b))}
    return {'schema_version': 2, 'decision': decision, 'stage_a_decision': gate,
            'complete': not errors and complete['stage-a'] and (gate == 'STOP' or complete['stage-b']),
            'integrity_errors': errors, 'stage_a': a, 'stage_b': b, 'cross_ood_base_image_overlap': overlap,
            'analysis': ANALYSIS, 'registry_sha256': registry['sha256']}


def write_report(root, summary):
    root = Path(root); write_json(root/'analysis.json', summary)
    lines = ['# QCGS multi-view rescue diagnostic', '', '**Decision: '+summary['decision']+'**.', '',
             f'Complete protocol: {summary["complete"]}. Stage A gate: {summary.get("stage_a_decision", "unavailable")}.', '',
             'Raw: [queries.jsonl](queries.jsonl). Metrics: [analysis.json](analysis.json). '
             'Frozen inputs and Git ledger: [locks](locks/). CUDA provenance: [runs](runs/).', '']
    lines.extend('- '+e for e in summary['integrity_errors'])
    for stage in ('stage_a', 'stage_b'):
        for cell, s in summary.get(stage, {}).items():
            lines.append(f'- {stage}, OOD={cell}: n={s["queries"]}; mean U/D/E={s["U"]["mean"]}/{s["D"]["mean"]}/{s["E"]["mean"]}; swaps={s["policies"]["mv_target"]["actual_replacements"]}.')
    lines += ['', 'Stage A exhausts legal same-pseudo-class one-swaps. Stage B uses only the selected-action union; its best-verified subset is not an exact oracle. Ensembles are prediction controls outside the action space.', '',
              'One stream/seed per cell; block intervals are descriptive. Cross-OOD repeats are reported. No Stage B tuning, full matrix, reranking, adaptive support or cross-dataset claim. The previous single-view STOP remains unchanged.', '']
    (root/'report.md').write_text('\n'.join(lines))
