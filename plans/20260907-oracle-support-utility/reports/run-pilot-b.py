"""Execute only B cells and stop once each diagnostic target is durably saved."""
import json
import hashlib
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'evidence/oracle-support-pilot-b-resumed-20260907'


def stop(process):
    process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def main():
    EVIDENCE.mkdir()
    plan = json.loads(subprocess.check_output([
        sys.executable, str(ROOT / 'scripts/run-oracle-support-utility.py'),
        '--data-root', '/Users/admin/data', '--evidence-dir', str(EVIDENCE),
        '--device', 'mps', '--max-eval-samples', '600'], text=True))
    plan['jobs'] = plan['jobs'][1:]
    plan['scope'] = 'Pilot B only; 128 eligible ID queries per cell; stop after target and covering trace are saved'
    (EVIDENCE / 'launch-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for relative, expected in plan['source_sha256'].items():
        content = (ROOT / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected
        archived = EVIDENCE / 'source' / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(content)
    for job in plan['jobs']:
        run = EVIDENCE / job['run_id']
        print('Starting', job['run_id'], flush=True)
        with (EVIDENCE / (job['run_id']+'.log')).open('w') as log:
            process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            complete = False
            try:
                while process.poll() is None:
                    try:
                        summary = json.loads((run / 'oracle-support-summary.json').read_text())
                        if summary['query_budget_complete'] and summary['eligible_queries'] == 128:
                            rows = [json.loads(s) for s in (run / 'oracle-support-queries.jsonl').read_text().splitlines()]
                            required_end = (max(r['query_index'] for r in rows)//100+1)*100
                            raw = (run / 'trace.jsonl').read_text()
                            if len(rows) == 128 and raw.endswith('\n'):
                                trace = [json.loads(s) for s in raw.splitlines()]
                                if len(trace) >= required_end and trace[required_end-1]['timestep'] == required_end-1:
                                    complete = True
                                    break
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                    time.sleep(1)
            except BaseException:
                stop(process)
                raise
            exit_code = stop(process) if complete and process.poll() is None else process.wait()
        if not complete:
            raise RuntimeError(f'{job["run_id"]} exited {exit_code} before diagnostic target completion')
        (run / 'diagnostic-completion.json').write_text(json.dumps({
            'status': 'TARGET_COMPLETE', 'eligible_queries': 128,
            'committed_trace_rows': len(trace), 'required_trace_rows': required_end,
            'stop_reason': 'diagnostic target and covering full batches durably saved',
            'evaluator_exit_code': exit_code,
            'full_600_sample_benchmark_complete': len(trace)==600,
        }, indent=2)+'\n')
        print('Completed', job['run_id'], '128 queries;', len(trace), 'trace rows', flush=True)


if __name__ == '__main__':
    main()
