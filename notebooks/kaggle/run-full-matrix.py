"""Run the frozen canonical 252-run matrix with per-run checkpointing."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluation.evidence import atomic_write_json, _source_tree_fingerprint
from evaluation.open_set_consensus_analysis import analyse_open_set_completed_runs
from runtime.experiment_matrix import build_canonical_open_set_evidence_matrix, build_command, validate_completed_run


REVISION = "26a7cd7c847b6630841dbae58067e5bb124f2f9d"


def load_checkpoint_helper():
    path = Path(__file__).with_name("full-run-checkpoint.py")
    spec = importlib.util.spec_from_file_location("full_run_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.checkpoint


def verify_run(run, source_identity, baselines):
    evidence = validate_completed_run(run)
    git = evidence["manifest"]["git"]
    if git.get("commit") != REVISION or git.get("dirty") is not False or git.get("source") != source_identity:
        raise RuntimeError(f"Run source identity differs from the frozen revision: {run.run_id}")
    fingerprint = evidence["summary"]["stream_fingerprint"]
    if run.reference_trace is not None:
        if baselines.get(run.reference_trace) != fingerprint:
            raise RuntimeError(f"Paired NoAdapt is absent or has a different stream: {run.run_id}")
    else:
        baselines[run.run_dir / "trace.jsonl"] = fingerprint
    # Keep summaries/manifests for descriptive analysis, not all stream exports in RAM.
    return {"manifest": evidence["manifest"], "summary": evidence["summary"]}


def run_command(command, repo, log):
    with log.open("w") as handle:
        process = subprocess.Popen(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    print(f"Still running ({time.monotonic() - started:.0f}s); log: {log.name}", flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        with log.open() as handle:
            from collections import deque
            print("".join(deque(handle, maxlen=35)), flush=True)
        raise subprocess.CalledProcessError(code, command)


def write_analysis(root, completed):
    report = analyse_open_set_completed_runs(completed)
    if report["classification"] != "canonical_cuda_expected" or not report["coverage"]["complete"]:
        raise RuntimeError("All 252 runs must satisfy the canonical analysis contract")
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    atomic_write_json(destination / "open-set-consensus.json", report)
    fields = ["ood_ratio", "stream_mode", "seed", "method", "id_accuracy", "auroc", "fpr95", "h_score"]
    with (destination / "per-cell-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in report["comparisons"]:
            for method, metrics in cell["methods"].items():
                writer.writerow({**{key: cell[key] for key in fields[:3]}, "method": method,
                                 **{key: metrics.get(key) for key in fields[4:]}})
    (destination / "README.md").write_text(
        "# Complete canonical CIFAR-100-C evidence\n\n"
        f"Revision: `{REVISION}`. All 252 full-stream runs passed strict validation.\n\n"
        "Read open-set-consensus.json for paired metrics, oracle diagnostics, stability and costs; "
        "per-cell-metrics.csv retains all three seeds and each stream/OOD ratio separately. "
        "Undefined OOD detection metrics at OOD=0 remain empty/null.\n\n"
        "This is descriptive evidence, not a certification that Consensus improves performance. "
        "It covers the primary CIFAR-100-C matrix; DomainNet and split-robustness studies are separate.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="New complete runs this invocation; 0 means all remaining.")
    parser.add_argument("--session-hours", type=float, default=8.0, help="Stop between runs after this budget; 0 disables it.")
    args = parser.parse_args()
    if args.max_runs < 0 or not 0 <= args.session_hours < float("inf"):
        parser.error("budgets must be finite and nonnegative")
    repo = Path(os.environ["RAMEN_REPOSITORY"]).resolve()
    data = Path(os.environ["RAMEN_DATA_ROOT"]).resolve()
    root = Path(os.environ["RAMEN_EVIDENCE_ROOT"]).resolve()
    runtime = Path(os.environ["RAMEN_RUNTIME_ROOT"]).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    runs = build_canonical_open_set_evidence_matrix(
        data_root=data, evidence_dir=root / "canonical", config_dir=repo / "cfg",
        device="cuda", artifact_provenance="fast",
    )
    atomic_write_json(runtime / "canonical-plan.json", {
        "execute": not args.plan_only, "runs": [run.to_dict() for run in runs],
        "commands": [build_command(run, python_executable=sys.executable) for run in runs],
    })
    print(f"Canonical plan: {len(runs)} full-stream runs, B=100, 400 source samples/domain.", flush=True)
    if args.plan_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA device is required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if head != REVISION or os.environ.get("RAMEN_REVISION") != REVISION or dirty:
        raise RuntimeError("Use the clean frozen experiment revision")
    tests = json.loads((runtime / "test-status.json").read_text())
    preflight = json.loads((runtime / "cifar100c-deep-preflight.json").read_text())
    if tests != {"focused_exit": 0, "full_exit": 0} or not preflight.get("valid"):
        raise RuntimeError("Pass this session's tests and deep preflight first")
    source_identity = _source_tree_fingerprint(repo)
    if source_identity is None:
        raise RuntimeError("Cannot fingerprint experiment source")
    device = json.loads((runtime / "device.json").read_text())
    identity = {"revision": REVISION, "data_root": str(data), "runs": 252,
                "runtime": {key: device[key] for key in ("gpu", "torch", "torchvision", "cuda")}}
    campaign_path = root / "campaign.json"
    if campaign_path.exists() and json.loads(campaign_path.read_text()) != identity:
        raise RuntimeError("Campaign identity changed; use the same revision, data path, GPU type and runtime")
    atomic_write_json(campaign_path, identity)
    checkpoint = load_checkpoint_helper()
    completed, baselines = [], {}
    state = {"status": "running", "revision": REVISION, "planned": 252, "completed": [],
             "new_runs_this_session": 0, "session": runtime.name, "full_matrix_complete": False}
    started = time.monotonic()
    failed = False
    try:
        for index, run in enumerate(runs, 1):
            new_run = False
            state["current_run"] = run.run_id
            if run.run_dir.exists() and not (run.run_dir / "summary.json").exists():
                # Preserve interrupted work; restart this whole run, never append a partial trace.
                interrupted = root / "interrupted" / runtime.name
                interrupted.mkdir(parents=True, exist_ok=True)
                preserved = interrupted / f"{run.run_id}-{time.time_ns()}"
                run.run_dir.rename(preserved)
                print(f"Preserved incomplete run under {preserved}", flush=True)
            if run.run_dir.exists():
                checked = verify_run(run, source_identity, baselines)
                print(f"[{index}/252] validated resume: {run.method} {run.stream_mode} seed={run.seed} OOD={run.ood_ratio}", flush=True)
            else:
                if ((args.max_runs and state["new_runs_this_session"] >= args.max_runs)
                        or (args.session_hours and time.monotonic() - started >= args.session_hours * 3600)):
                    state["status"] = "paused_session_budget"
                    break
                if run.reference_trace is not None and run.reference_trace not in baselines:
                    raise RuntimeError("No validated paired NoAdapt before adapted run")
                print(f"[{index}/252] RUN {run.method} {run.stream_mode} seed={run.seed} OOD={run.ood_ratio}", flush=True)
                atomic_write_json(root / "status.json", state)
                run_command(build_command(run, python_executable=sys.executable), repo, runtime / f"{run.run_id}.log")
                checked = verify_run(run, source_identity, baselines)
                state["new_runs_this_session"] += 1
                new_run = True
            completed.append((run, checked))
            state["completed"].append(run.run_id)
            state["current_run"] = None
            atomic_write_json(root / "status.json", state)
            if new_run:
                # Per-run atomic snapshots retain the last complete ZIP if the session is killed.
                print(f"Checkpoint: {checkpoint(root)}", flush=True)
        if len(completed) == 252:
            write_analysis(root, completed)
            state.update(status="complete", full_matrix_complete=True)
    except BaseException as exc:
        failed = True
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_write_json(root / "status.json", state)
        try:
            print(f"Evidence archive: {checkpoint(root)}", flush=True)
        except Exception as exc:
            if not failed:
                raise
            print(f"Checkpoint failed too: {exc}. The previous complete ZIP is retained.", flush=True)
    print(json.dumps({key: value for key, value in state.items() if key != "completed"}, indent=2), flush=True)
    print(f"Validated {len(completed)}/252. Download the ZIP before ending the session.", flush=True)


if __name__ == "__main__":
    main()
