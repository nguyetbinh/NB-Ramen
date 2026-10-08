#!/usr/bin/env python3
"""External, foreground two-T4 launcher for the frozen exact-objective Stage A.

Copy this file OUTSIDE /kaggle/working/NB-Ramen before invoking it with
/tmp/qcgs-venv/bin/python. Root owns artifact preparation, monitoring and export.
No installs, downloads, smokes, Stage B, or scientific decisions are performed.
--prepare-only validates inputs/runtime and saves/prints jobs without inference.

Required artifact receipt is the existing main.py artifact object:
{"status": "verified", "dataset": {"verified_exact": true, ...}, "model": {...}}.
Dataset/model fields must match the frozen provenance verifier outputs. The
root-generated receipt certifies prior exact verification; launch rechecks the
dataset inventory/size/sidecar and checkpoint SHA without rereading all data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


REVISION = "6cd46a8a785e97b1de234af118540cfdd0f562e8"
CELLS = (("0.5", "0"), ("0", "1"))
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def outside(repo, paths):
    for path in paths:
        if Path(path).resolve().is_relative_to(repo.resolve()):
            raise ValueError(f"operational path must be outside audited source: {path}")


def verify_source(repo):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()
    if git("rev-parse", "HEAD") != REVISION or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("audited source must be clean at frozen " + REVISION)
    return {"revision": REVISION, "tree": git("rev-parse", "HEAD^{tree}")}


def child_env(repo, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0",
               PYTHONPATH=str(repo / "src"))
    env.update({name: "1" for name in THREAD_VARS})
    return env


def runtime_probe(repo, gpu):
    code = '''import json, torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
assert "T4" in p.name and 14 * 1024**3 <= p.total_memory <= 17 * 1024**3
print(json.dumps({"python": __import__("sys").version, "torch": torch.__version__,
 "cuda": torch.version.cuda, "device": p.name, "bytes": p.total_memory,
 "capability": list(torch.cuda.get_device_capability(0))}))'''
    facts = json.loads(subprocess.check_output(
        [sys.executable, "-c", code], cwd=repo, env=child_env(repo, gpu), text=True))
    # nvidia-smi checks physical indices and idle state before either job starts.
    output = subprocess.check_output([
        "nvidia-smi", "-i", gpu,
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits"], text=True).strip()
    fields = [field.strip() for field in output.split(",")]
    if (len(fields) != 7 or fields[0] != gpu or "T4" not in fields[2]
            or int(fields[4]) > 100 or int(fields[5]) != 0):
        raise ValueError(f"physical GPU {gpu} is not an idle T4: {output}")
    facts["nvidia_smi"] = output
    facts["physical_gpu"] = gpu
    return facts


def verify_inputs(args):
    from runtime.artifact_provenance import verify_dataset_provenance, verify_cached_clip_checkpoint
    from runtime.qcgs_exact_objective_audit import validate_smoke_registry
    from runtime.query_gradient_registry import validate_exact_objective_registry

    registry, smoke = read(args.registry), read(args.smoke_registry)
    validate_exact_objective_registry(registry)
    validate_smoke_registry(smoke)
    if (registry.get("source_revision") != REVISION or not registry.get("sha256")
            or registry["provenance"]["smoke_registry_sha256"] != smoke["sha256"]):
        raise ValueError("frozen source/registry/smoke binding mismatch")
    smoke_ids = {row["sample_idx"] for value in smoke["cells"].values() for row in value["selected"]}
    if set(registry["exclusions"]["smoke_base_image_indices"]) != smoke_ids:
        raise ValueError("smoke exclusions differ from the provided smoke registry")
    if any(len(value["selected"]["stage-a"]) != 16 for value in registry["cells"].values()):
        raise ValueError("Stage A requires exactly 16 queries per cell")
    receipt = read(args.artifact_receipt)
    dataset = verify_dataset_provenance("cifar100c", args.data_root / "corruption/CIFAR-100-C")
    model = verify_cached_clip_checkpoint("clip_vitbase16", Path.home() / ".cache/clip")
    prior = receipt.get("dataset", {})
    if (receipt.get("status") != "verified" or prior.get("verified_exact") is not True
            or dataset["root_digest"] != registry["dataset_fingerprint"]
            or any(prior.get(key) != dataset[key] for key in
                   ("root_digest", "sidecar_sha256", "file_count", "acquisition"))
            or {key: value for key, value in receipt.get("model", {}).items() if key != "status"} != model):
        raise ValueError("shared exact artifact receipt/data/checkpoint verification mismatch")
    return registry, {"artifact_receipt_sha256": sha(args.artifact_receipt),
                      "registry_file_sha256": sha(args.registry),
                      "smoke_registry_file_sha256": sha(args.smoke_registry),
                      "dataset": dataset, "model": model,
                      "limitation": "data content exact-verified by root; launch uses fast inventory verification"}


def audit_cell(job, registry):
    from evaluation.qcgs_exact_objective_audit import audit_raw, audit_run_provenance, read_jsonl_detailed
    root = Path(job["evidence_root"]) / job["run_id"]
    actions, action_receipt = read_jsonl_detailed(root / "qcgs-exact-actions.jsonl")
    outcomes, outcome_receipt = read_jsonl_detailed(root / "qcgs-exact-outcomes.jsonl")
    for receipt in (action_receipt, outcome_receipt):
        if receipt["errors"] or receipt["truncated_tail"]:
            raise ValueError("malformed or truncated raw record file")
    provenance = audit_run_provenance(root, actions, outcomes)
    audit_raw(actions, outcomes, registry)
    expected = {(row["timestep"], row["sample_idx"])
                for row in registry["cells"][job["cell"]]["selected"]["stage-a"]}
    actual = {(row["query"]["timestep"], row["query"]["sample_idx"]) for row in actions}
    if (len(actions) != 16 or len(outcomes) != 16 or actual != expected
            or provenance["stage"] != "stage-a" or provenance["ood_cell"] != job["cell"]
            or provenance["source_revision"] != REVISION
            or provenance["registry_sha256"] != registry["sha256"]
            or not all(provenance[key] for key in ("scorer_complete", "evaluator_complete", "trace_complete"))
            or provenance["interrupted_jsonl_artifacts"]):
        raise ValueError("missing full durable Stage-A coverage or frozen provenance")
    return {"provenance": provenance, "actions": action_receipt, "outcomes": outcome_receipt}


def worker(plan_path, index):
    plan = read(plan_path)
    repo = Path(plan["repo"])
    sys.path.insert(0, str(repo / "src"))
    job = plan["jobs"][index]
    receipt_path = Path(job["receipt"])
    if receipt_path.exists():
        raise FileExistsError("refusing to replace an existing completion/failure receipt")
    result = {"run_id": job["run_id"], "cell": job["cell"], "status": "failed",
              "exit_code": None, "complete": False, "scientific_decision": None,
              "launcher_sha256": sha(__file__), "source_revision": REVISION}
    try:
        if sha(__file__) != plan["launcher_sha256"]:
            raise ValueError("launcher changed after preparation")
        verify_source(repo)
        if sha(job["config"]) != job["config_sha256"]:
            raise ValueError("job config changed after preparation")
        registry = read(plan["registry_snapshot"])
        if registry != plan["registry"]:
            raise ValueError("registry snapshot changed after preparation")
        with Path(job["log"]).open("x") as log:
            lock_fd = os.environ.get("QCGS_LAUNCH_LOCK_FD")
            result["exit_code"] = subprocess.run(job["command"], cwd=repo,
                env=child_env(repo, job["physical_gpu"]), stdout=log, stderr=subprocess.STDOUT,
                pass_fds=() if lock_fd is None else (int(lock_fd),)).returncode
        # Audit preserved records even when the inference process failed.
        result["audit"] = audit_cell(job, registry)
        verify_source(repo)
        if result["exit_code"] == 0:
            result.update(status="completed_validated", complete=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(receipt_path, result)
    return 0 if result["complete"] else 1


def prepare(args):
    from runtime.qcgs_exact_objective_audit import make_job
    source = verify_source(args.repo)
    registry, artifacts = verify_inputs(args)
    devices = {gpu: runtime_probe(args.repo, gpu) for _, gpu in CELLS}
    campaign = args.output_root / ("stage-a-t4x2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                   + "-" + uuid.uuid4().hex[:12])
    campaign.mkdir(parents=True, exist_ok=False)
    snapshot = campaign / "registry.json"
    write(snapshot, registry)
    write(campaign / "smoke-registry.json", read(args.smoke_registry))
    write(campaign / "artifact-receipt.json", read(args.artifact_receipt))
    jobs = []
    for cell, gpu in CELLS:
        run_id = campaign.name + "-ood" + cell.replace(".", "") + "-gpu" + gpu
        root = campaign / ("gpu" + gpu)
        root.mkdir()
        evidence = root / "evidence"
        job = make_job("stage-a", cell, root / "config", data_root=args.data_root,
                       evidence_dir=evidence, run_id=run_id, registry=registry,
                       action_chunk_size=registry["provenance"]["action_chunk_size"])
        job["command"].extend(["--num_threads", "1"])
        job.update(physical_gpu=gpu, evidence_root=str(evidence),
                   log=str(root / "inference.log"), receipt=str(root / "completion.json"))
        env = child_env(args.repo, gpu)
        job["environment"] = {key: env[key] for key in (*THREAD_VARS, "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "PYTHONPATH")}
        jobs.append(job)
    plan = {"repo": str(args.repo), "source": source, "registry": registry,
            "registry_snapshot": str(snapshot), "launcher_sha256": sha(__file__),
            "launcher_path": str(Path(__file__).resolve()), "artifacts": artifacts,
            "devices": devices, "jobs": jobs, "scientific_decision": None}
    path = campaign / "launch.json"
    write(path, plan)
    return path, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/kaggle/working/NB-Ramen"))
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/qcgs-data"))
    parser.add_argument("--registry", type=Path, default=Path("/kaggle/working/registry.json"))
    parser.add_argument("--smoke-registry", type=Path, default=Path("/kaggle/working/smoke-registry.json"))
    parser.add_argument("--artifact-receipt", type=Path, required=False,
                        default=Path("/kaggle/working/artifact-receipt.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/qcgs-stage-a-t4x2"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, args.index)
    for key in ("repo", "data_root", "registry", "smoke_registry", "artifact_receipt", "output_root"):
        setattr(args, key, getattr(args, key).resolve())
    outside(args.repo, [__file__, args.output_root, args.registry, args.smoke_registry,
                        args.artifact_receipt, args.data_root])
    sys.path.insert(0, str(args.repo / "src"))
    # One stable lock for this audited checkout, independent of output/run IDs.
    lock_path = args.repo.parent / (".qcgs-stage-a-" + hashlib.sha256(str(args.repo).encode()).hexdigest()[:16] + ".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Stage-A launcher/worker is live for this checkout") from exc
        path, plan = prepare(args)
        print(json.dumps({"launch_plan": str(path), "jobs": plan["jobs"]}, indent=2), flush=True)
        if args.prepare_only:
            return 0
        processes = []
        try:
            for index, job in enumerate(plan["jobs"]):
                # Inherited lock remains held if the foreground parent dies.
                env = child_env(args.repo, job["physical_gpu"])
                env["QCGS_LAUNCH_LOCK_FD"] = str(lock.fileno())
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    "--worker", str(path), "--index", str(index)], cwd=args.repo,
                    env=env, pass_fds=(lock.fileno(),))
                processes.append(process)
                write(Path(job["receipt"]).with_name("process.json"), {"pid": process.pid, "run_id": job["run_id"]})
        finally:
            codes = [process.wait() for process in processes]
            write(path.with_name("launcher-exit.json"), {"worker_exit_codes": codes,
                  "scientific_decision": None, "complete": len(codes) == 2 and codes == [0, 0]})
        return 0 if codes == [0, 0] else 1


if __name__ == "__main__":
    sys.exit(main())
