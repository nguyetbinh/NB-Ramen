"""Separate diagnostic sidecars; never interpreted as canonical method evidence."""
import json
from pathlib import Path

import numpy as np
def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks



def summarize(rows, requested_queries, mode=None):
    mode = mode or (rows[0]['mode'] if rows else None)
    gains = [r['oracle']['exact_utility'] for r in rows]
    pairs = [(s['predicted_utility'], s['exact_utility']) for r in rows for s in r['verified_swaps']]
    transitions = {f'{a}->{b}': sum(r['ramen']['correct'] == a and r['oracle']['correct'] == b for r in rows)
                   for a in (False, True) for b in (False, True)}
    correlation = None
    if len(pairs) > 1 and len(set(p[0] for p in pairs)) > 1 and len(set(p[1] for p in pairs)) > 1:
        correlation = float(np.corrcoef(*[average_ranks(v) for v in zip(*pairs)])[0, 1])
    replacements = [r for r in rows if 'outgoing' in r['oracle']]
    def rate(predicate, population=rows):
        return sum(predicate(r) for r in population) / len(population) if population else None
    return {
        'schema_version': 1, 'diagnostic_only': True,
        'oracle_scope': {'exhaustive': 'exhaustive_one_swap', 'screened': 'screened_subset_lower_bound'}.get(mode, 'unavailable'),
        'requested_queries': requested_queries, 'eligible_queries': len(rows),
        'query_budget_complete': len(rows) == requested_queries,
        'mean_oracle_loss_gain': float(np.mean(gains)) if gains else None,
        'median_oracle_loss_gain': float(np.median(gains)) if gains else None,
        'p25_p75': np.percentile(gains, [25, 75]).tolist() if gains else None,
        'headroom_rate': rate(lambda r: r['oracle']['exact_utility'] > 0),
        'accuracy_transitions': transitions,
        'net_correction': transitions['False->True'] - transitions['True->False'],
        'mean_margin_gain': float(np.mean([r['oracle']['margin'] - r['ramen']['margin'] for r in rows])) if rows else None,
        'spearman_predicted_exact': correlation,
        'correlation_population': 'all verified swaps pooled; screened mode has selection bias',
        'selected_swap_count': len(replacements),
        'ood_to_id_best_swap_rate': rate(lambda r: r['oracle']['outgoing']['is_ood'] and not r['oracle']['incoming']['is_ood'], replacements),
        'id_to_ood_best_swap_rate': rate(lambda r: not r['oracle']['outgoing']['is_ood'] and r['oracle']['incoming']['is_ood'], replacements),
        'same_to_cross_domain_rate': rate(lambda r: r['oracle']['outgoing']['domain'] == r['query_domain'] and r['oracle']['incoming']['domain'] != r['query_domain'], replacements),
        'cross_to_same_domain_rate': rate(lambda r: r['oracle']['outgoing']['domain'] != r['query_domain'] and r['oracle']['incoming']['domain'] == r['query_domain'], replacements),
        'baseline_mean_loss_gain': {name: float(np.mean([r['baselines'][name]['exact_utility'] for r in rows]))
                                   for name in rows[0]['baselines']} if rows else {},
        'decision': 'requires_review_of_pilot_A_and_both_pilot_B_cells',
    }


def write_probe_outputs(method, run_dir):
    if getattr(method, 'diagnostic_kind', None) in ('qcgs', 'qcgs-multiview'):
        from .query_gradient_utility import write_query_outputs
        return write_query_outputs(method, run_dir)
    run_dir = Path(run_dir)
    query_path = run_dir / 'oracle-support-queries.jsonl'
    query_tmp = query_path.with_suffix('.tmp')
    with query_tmp.open('w') as handle:
        for row in method.rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    query_tmp.replace(query_path)
    summary = summarize(method.rows, method.cfg['probe_queries'], method.cfg['probe_mode'])
    summary_path = run_dir / 'oracle-support-summary.json'
    summary_tmp = summary_path.with_suffix('.tmp')
    summary_tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    summary_tmp.replace(summary_path)
