#!/usr/bin/env python3
"""Plan or execute the three diagnostic cells, stopping on incomplete evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--evidence-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cuda')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--max-eval-samples', type=int, default=600)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.max_eval_samples < 100 or args.max_eval_samples % 100:
        parser.error('require positive timeout and a positive whole number of legacy 100-row batches')
    evidence = Path(args.evidence_dir).resolve()
    jobs = []
    for pilot, ratio, config in [('a', .5, 'oracle-support-utility'),
                                  ('b-null', 0., 'oracle-support-utility-screened'),
                                  ('b-open', .5, 'oracle-support-utility-screened')]:
        run_id = f'oracle-support-{pilot}-{args.device}-seed0'
        config_root = ROOT / 'cfg' / 'research' / config
        config_file = config_root / 'CIFAR100C' / 'OracleSupportUtilityProbe.yaml'
        command = [sys.executable, str(ROOT / 'src/main.py'), '--dataset', 'CIFAR100C',
                   '--data_root', str(Path(args.data_root).expanduser().resolve()),
                   '--artifact-provenance', 'fast', '--open_set', '--ood_ratio', str(ratio),
                   '--known_class_split', 'open-set-cifar100-split-v1',
                   '--model', 'clip_vitbase16', '--tta_algo', 'OracleSupportUtilityProbe',
                   '--batch_size', '100', '--device', args.device, '--config', str(config_root),
                   '--config-lock-path', str(config_file), '--config-lock-sha256', hashlib.sha256(config_file.read_bytes()).hexdigest(),
                   '--stream_mode', 'block', '--stream_block_size', '64', '--seed', '0',
                   '--open_set_per_domain_source_budget', '400', '--max_eval_samples', str(args.max_eval_samples),
                   '--run_id', run_id, '--evidence_dir', str(evidence), '--save_to', str(evidence / 'runs.csv')]
        jobs.append({'pilot': pilot, 'run_id': run_id, 'command': command})
    sources = ['src/methods/OracleSupportUtilityProbe.py', 'src/methods/Ramen.py',
               'src/evaluation/oracle_support_utility.py', 'src/main.py', 'scripts/run-oracle-support-utility.py']
    plan = {'diagnostic_only': True, 'canonical': False, 'jobs': jobs,
            'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    evidence.mkdir(parents=True, exist_ok=True)
    plan_path = evidence / 'oracle-support-plan.json'
    with plan_path.open('x') as handle:
        json.dump(plan, handle, indent=2)
    for job in jobs:
        with (evidence / (job['run_id'] + '.log')).open('x') as log:
            try:
                subprocess.run(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                raise SystemExit(f"Timed out: {job['run_id']}; partial artifacts retained, later pilots not started")
        summary = json.loads((evidence / job['run_id'] / 'oracle-support-summary.json').read_text())
        if not summary['query_budget_complete']:
            raise SystemExit(f"Incomplete eligible-query budget: {job['run_id']}; later pilots not started")


if __name__ == '__main__':
    if '--diagnostic' in sys.argv:
        index = sys.argv.index('--diagnostic')
        if sys.argv[index+1:index+2] != ['qcgs']:
            raise SystemExit('the explicit diagnostic mode must be qcgs')
        arguments = sys.argv[1:index] + sys.argv[index+2:]
        sys.path.insert(0, str(ROOT / 'src'))
        from runtime.query_gradient_campaign import main as qcgs_main
        qcgs_main(arguments)
    else:
        main()
