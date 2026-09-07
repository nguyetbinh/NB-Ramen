"""Run the fixed pre-full CUDA smoke controls; never launch the full matrix."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from runtime.experiment_matrix import (
    build_canonical_open_set_evidence_matrix, build_open_set_evidence_matrix,
    build_experiment_matrix, build_command, validate_completed_run,
)
from runtime.open_set_split_robustness_matrix import SPLIT_SHA256
from evaluation.evidence import atomic_write_json, SUMMARY_SCHEMA_VERSION, TRACE_SCHEMA_VERSION


def plan_smokes(repo, data, evidence):
    canonical = build_canonical_open_set_evidence_matrix(
        data_root=data, evidence_dir=evidence / "canonical-not-executed",
        device="cuda", artifact_provenance="fast",
    )
    common = dict(
        streams=("block",), seeds=(0,), evidence_dir=evidence / "smoke",
        device="cuda", max_eval_samples=256, artifact_provenance="fast", data_root=data,
        config_dir=repo / "cfg",
    )
    primary = build_open_set_evidence_matrix(ood_ratios=(.3,), **common)
    ids = ["prefull-noadapt", "prefull-ramen", "prefull-entropy", "prefull-oracle-drop",
           "prefull-oracle-id", "prefull-consensus", "prefull-oracle-consensus"]
    runs = [replace(
        run, run_id=run_id, require_config_lock=run.method != "NoAdapt",
        reference_trace=None if run.method == "NoAdapt" else common["evidence_dir"] / "prefull-noadapt/trace.jsonl",
    ) for run, run_id in zip(primary, ids)]
    b1 = replace(runs[0], run_id="prefull-noadapt-b1", batch_size=1)
    runs += [b1, replace(runs[1], run_id="prefull-ramen-b1", batch_size=1,
                         reference_trace=b1.run_dir / "trace.jsonl")]
    causal = build_experiment_matrix(datasets=("CIFAR100C",), methods=("CausalRamen",), **common)[1]
    runs.append(replace(
        causal, run_id="prefull-causal-b100", open_set=True,
        known_class_split=primary[0].known_class_split, ood_ratio=.3,
        open_set_per_domain_source_budget=400, require_config_lock=True,
        reference_trace=runs[0].run_dir / "trace.jsonl",
    ))
    split = "open-set-cifar100-name-rank-v2"
    split_path = (repo / "cfg/research/open-set-cifar100-split-v2.json").resolve()
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != SPLIT_SHA256[split]:
        raise RuntimeError("Frozen v2 split bytes changed")
    v2 = replace(runs[0], run_id="prefull-v2-noadapt", known_class_split=split,
                 known_class_split_path=split_path, known_class_split_sha256=SPLIT_SHA256[split])
    runs += [v2, replace(runs[2], run_id="prefull-v2-entropy", known_class_split=split,
                         known_class_split_path=split_path, known_class_split_sha256=SPLIT_SHA256[split],
                         reference_trace=v2.run_dir / "trace.jsonl")]
    zero = replace(runs[0], run_id="prefull-ood0-noadapt", ood_ratio=0.0)
    runs += [zero, replace(runs[4], run_id="prefull-ood0-oracle-id", ood_ratio=0.0,
                           reference_trace=zero.run_dir / "trace.jsonl")]
    if len(runs) != 14 or len({r.run_id for r in runs}) != 14 or len(canonical) != 252:
        raise RuntimeError("Unexpected plan size")
    return runs, canonical


def trace(run):
    return [json.loads(line) for line in (run.run_dir / "trace.jsonl").read_text().splitlines() if line.strip()]


def compare(left, right):
    if len(left) != len(right) or not left:
        raise RuntimeError("Sensitivity traces have different lengths or are empty")
    for a, b in zip(left, right):
        for field in ("timestep", "sample_idx", "ground_truth_domain", "original_label", "is_ood"):
            if a[field] != b[field]:
                raise RuntimeError(f"Sensitivity traces disagree on {field}")
    def id_accuracy(rows):
        ids = [r for r in rows if not r["is_ood"]]
        return sum(r["prediction"] == r["known_label_or_minus_one"] for r in ids) / len(ids) if ids else None
    return {
        "samples": len(left),
        "prediction_disagreement_count": sum(a["prediction"] != b["prediction"] for a, b in zip(left, right)),
        "pre_prediction_disagreement_count": sum(a["pre_adaptation_prediction"] != b["pre_adaptation_prediction"] for a, b in zip(left, right)),
        "left_id_accuracy": id_accuracy(left), "right_id_accuracy": id_accuracy(right),
        "max_absolute_post_ood_score_difference": max(abs(a["post_adaptation_ood_score"] - b["post_adaptation_ood_score"]) for a, b in zip(left, right)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    repo = Path(os.environ["RAMEN_REPOSITORY"]).resolve()
    data = Path(os.environ["RAMEN_DATA_ROOT"]).resolve()
    evidence = Path(os.environ["RAMEN_EVIDENCE_ROOT"]).resolve()
    runtime = evidence / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    runs, canonical = plan_smokes(repo, data, evidence)
    for name, planned in (("smoke-plan", runs), ("canonical-plan", canonical)):
        atomic_write_json(runtime / f"{name}.json", {
            "execution_requested": name == "smoke-plan" and not args.plan_only,
            "runs": [r.to_dict() for r in planned],
            "commands": [build_command(r) for r in planned],
        })
    print("Plan: 14 smoke runs; 252 canonical runs (plan only).", flush=True)
    if args.plan_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != os.environ["RAMEN_REVISION"] or subprocess.check_output(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("The experiment checkout must match the pinned revision and be clean")
    status = {"classification": "noncanonical_pilot", "trace_schema": TRACE_SCHEMA_VERSION,
              "summary_schema": SUMMARY_SCHEMA_VERSION, "completed": [], "status": "running"}
    atomic_write_json(runtime / "smoke-status.json", status)
    completed = {}
    for index, run in enumerate(runs, 1):
        print(f"[{index}/{len(runs)}] {run.run_id}", flush=True)
        try:
            if not run.run_dir.exists():
                with (runtime / f"{run.run_id}.log").open("w") as log:
                    result = subprocess.run(build_command(run), cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    tail = (runtime / f"{run.run_id}.log").read_text(errors="replace")[-8000:]
                    print(tail, flush=True)
                    raise RuntimeError(f"{run.run_id} failed with exit {result.returncode}")
            # A partial or corrupted directory is rejected. Do not silently overwrite it.
            checked = validate_completed_run(run)
            if checked["manifest"]["git"].get("commit") != head or checked["manifest"]["git"].get("dirty"):
                raise RuntimeError("Completed run was produced by a different or dirty revision")
            if run.reference_trace is not None:
                baseline = completed[run.reference_trace]
                if checked["summary"]["stream_fingerprint"] != baseline["summary"]["stream_fingerprint"]:
                    raise RuntimeError("Paired stream fingerprints disagree")
            completed[run.run_dir / "trace.jsonl"] = checked
            status["completed"].append({"run_id": run.run_id,
                                        "fingerprint": checked["summary"]["stream_fingerprint"],
                                        "id_accuracy": checked["summary"]["open_set"]["id_accuracy"]})
            atomic_write_json(runtime / "smoke-status.json", status)
        except Exception as exc:
            status.update(status="failed", failed_run=run.run_id, error=str(exc))
            atomic_write_json(runtime / "smoke-status.json", status)
            raise
    by_id = {r.run_id: r for r in runs}
    try:
        causal_ids = ["prefull-ramen", "prefull-ramen-b1", "prefull-causal-b100"]
        if len({completed[by_id[i].run_dir / "trace.jsonl"]["summary"]["stream_fingerprint"] for i in causal_ids}) != 1:
            raise RuntimeError("Causal controls must share the exact stream fingerprint")
        sensitivity = {
            "status": "measured_review_required",
            "pretrained_ramen_b100_vs_noadapt_b100": compare(trace(by_id["prefull-ramen"]), trace(by_id["prefull-noadapt"])),
            "pretrained_ramen_b1_vs_noadapt_b1": compare(trace(by_id["prefull-ramen-b1"]), trace(by_id["prefull-noadapt-b1"])),
            "packaging_ramen_b100_vs_b1": compare(trace(by_id[causal_ids[0]]), trace(by_id[causal_ids[1]])),
            "causal_ramen_b1_vs_causal_b100": compare(trace(by_id[causal_ids[1]]), trace(by_id[causal_ids[2]])),
        }
        atomic_write_json(runtime / "causal-sensitivity.json", sensitivity)
        null_rows = trace(by_id["prefull-ood0-oracle-id"])
        for row in null_rows:
            if row["retrieved_ood_fraction"] != 0 or row["retrieved_ood_weight_fraction"] != 0:
                raise RuntimeError("OOD=0 control retrieved OOD supports")
            cosine = row["ramen_vs_oracle_id_cosine"]
            disagreement = row["ramen_vs_oracle_id_sign_disagreement"]
            if cosine is not None and abs(cosine - 1) > 1e-5:
                raise RuntimeError("OOD=0 defined cosine differs from one beyond 1e-5")
            if disagreement is not None and disagreement != 0:
                raise RuntimeError("OOD=0 sign disagreement is nonzero")
        status.update(status="smokes_validated_causal_review_required", null_control="passed",
                      full_matrix_executed=False)
    except Exception as exc:
        status.update(status="control_check_failed", error=str(exc))
        raise
    finally:
        atomic_write_json(runtime / "smoke-status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
