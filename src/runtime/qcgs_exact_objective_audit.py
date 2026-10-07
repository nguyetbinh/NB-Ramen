"""Prepare locked exact-objective jobs and independently analyze their raw records.

Typical launcher sequence (the commands only emit or analyze; the returned
``main.py`` command is run by the external launcher after review):

    PYTHONPATH=src python -m runtime.qcgs_exact_objective_audit --help
    PYTHONPATH=src python -m runtime.qcgs_exact_objective_audit job --stage scan --cell 0 --max-eval-samples 600 --dataset-fingerprint SHA256 --source-revision COMMIT --data-root DATA --evidence-dir EVIDENCE --output-config LOCKS/scan-0
    PYTHONPATH=src python -m runtime.qcgs_exact_objective_audit make-smoke-registry --scan-0 SCAN0 --scan-05 SCAN05 --dataset-fingerprint SHA256 --source-revision COMMIT --output smoke-registry.json
    PYTHONPATH=src python -m runtime.qcgs_exact_objective_audit build-registry --scan-0 SCAN0 --scan-05 SCAN05 --smoke-registry smoke-registry.json --smoke-run-0 SMOKE0 --smoke-run-05 SMOKE05 --dataset-fingerprint SHA256 --source-revision COMMIT --output registry.json
    PYTHONPATH=src python -m runtime.qcgs_exact_objective_audit job --stage stage-a --cell 0 --registry registry.json --data-root DATA --evidence-dir EVIDENCE --output-config LOCKS/stage-a-0
    PYTHONPATH=src python -m runtime.qcgs_exact_objective_audit analyze --evidence-dir EVIDENCE --registry registry.json --output plans/.../stage-a-decision.json
    PYTHONPATH=src python -m runtime.qcgs_exact_objective_audit job --stage stage-b --cell 0 --registry registry.json --analysis plans/.../stage-a-decision.json --stage-a-evidence-dir EVIDENCE/stage-a --data-root DATA --evidence-dir EVIDENCE/stage-b --output-config LOCKS/stage-b-0
    # After both cells finish, analyze the shared campaign root containing Stage A and Stage B runs.
    PYTHONPATH=src python -m runtime.qcgs_exact_objective_audit analyze --evidence-dir EVIDENCE --registry registry.json --output plans/.../final-analysis.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import uuid

from runtime.query_gradient_registry import (
    CELLS, EXACT_OBJECTIVE_NAMESPACE, EXACT_OBJECTIVE_SCAN_CAPS,
    build_exact_objective_registry, digest, exact_objective_query_schedule,
    file_sha, validate_exact_objective_registry,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIT_CONFIG = ROOT / "cfg/research/qcgs-exact-objective-audit"
HISTORICAL_EXCLUSIONS = AUDIT_CONFIG / "historical-exclusions.json"
SPLIT_PATH = ROOT / "cfg/research/open-set-cifar100-split-v1.json"
METHOD_CONFIG = "QCGSExactObjectiveAudit.yaml"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def _historical(path):
    value = json.loads(Path(path).read_text())
    indices = value["base_image_indices"]
    value_sha = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    if len(indices) != 1499 or value_sha != "c6ea2be7074c89ca0ab8d9a6479c2a0a6aa43e4726e2a6a485d527ab22ddb72c":
        raise ValueError("historical exclusions differ from the frozen 1,499-ID list")
    return indices, value_sha


def _git_output(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def analyze_evidence(evidence_dir, registry=None, *, allowed_stages=None):
    """Recompute gates from raw records and bind them to registry and inputs."""
    from evaluation.qcgs_exact_objective_audit import (
        analyze_raw, audit_raw, audit_run_provenance, read_jsonl_detailed,
    )

    root = Path(evidence_dir)
    run_dirs = {path.parent for pattern in ("manifest.json", "qcgs-exact-status.json",
                                             "qcgs-exact-actions.jsonl", "qcgs-exact-outcomes.jsonl")
                for path in root.rglob(pattern)}
    if not run_dirs:
        raise ValueError("analysis requires a campaign root with preserved run manifests or statuses")
    action_rows, outcome_rows = [], []
    all_action_rows, all_outcome_rows = [], []
    integrity_errors = []
    incomplete_runs = []
    status_reconciliations = []
    raw_file_hashes = []
    stage_b_run_present = False
    stage_b_objectives = []
    for run_dir in sorted(run_dirs):
        manifest_path = run_dir / "manifest.json"
        status_path = run_dir / "qcgs-exact-status.json"
        action_path = run_dir / "qcgs-exact-actions.jsonl"
        outcome_path = run_dir / "qcgs-exact-outcomes.jsonl"
        rows, action_file = read_jsonl_detailed(action_path)
        outcomes, outcome_file = read_jsonl_detailed(outcome_path)
        for artifact, receipt in (("qcgs-exact-actions.jsonl", action_file),
                                  ("qcgs-exact-outcomes.jsonl", outcome_file)):
            integrity_errors.extend(f"{run_dir}/{artifact}: {error}" for error in receipt["errors"])
        if not manifest_path.exists():
            if rows or outcomes or status_path.exists():
                integrity_errors.append(f"{run_dir}: run status or raw records lack a run manifest")
            continue
        manifest = json.loads(manifest_path.read_text())
        stage = manifest.get("config", {}).get("qcgs_mode")
        if stage == "stage-b":
            stage_b_run_present = True
            stage_b_objectives.append(manifest.get("config", {}).get("frozen_objective"))
        if stage in ("stage-a", "stage-b"):
            run_id = manifest.get("args", {}).get("run_id", run_dir.name)
            for artifact, receipt in (("actions", action_file), ("outcomes", outcome_file)):
                if receipt["sha256"] is not None:
                    raw_file_hashes.append({"run_id": run_id, "artifact": artifact,
                                            "sha256": receipt["sha256"]})
        if status_path.exists() and json.loads(status_path.read_text()).get("stage") != stage:
            integrity_errors.append(f"{run_dir}: run status stage differs from its locked configuration")
            continue
        if stage in ("scan", "smoke"):
            continue
        if stage not in ("stage-a", "stage-b"):
            integrity_errors.append(f"{run_dir}: unknown exact-objective campaign stage")
            continue
        if allowed_stages is not None and stage not in allowed_stages:
            raise ValueError("campaign root contains a stage outside the permitted analysis scope")
        if not status_path.exists():
            if rows or outcomes:
                integrity_errors.append(f"{run_dir}: scored raw records lack a run completion status")
            else:
                incomplete_runs.append({"run_dir": str(run_dir), "stage": stage,
                                        "scored_queries": 0, "evaluator_queries": 0,
                                        "reason": "run ended before the first scorer status was written"})
            continue
        all_action_rows.extend(rows)
        all_outcome_rows.extend(outcomes)
        truncated_artifacts = [name for name, receipt in (
            ("qcgs-exact-actions.jsonl", action_file),
            ("qcgs-exact-outcomes.jsonl", outcome_file),
        ) if receipt["truncated_tail"]]
        try:
            cell = format(float(manifest["args"]["ood_ratio"]), "g")
            if registry is not None:
                if cell not in CELLS:
                    raise ValueError("run OOD cell is outside the frozen registry")
                for row in rows:
                    expected = {item["timestep"]: item["sample_idx"]
                                for item in registry["cells"][cell]["selected"][row["stage"]]}
                    if expected.get(row["query"]["timestep"]) != row["query"]["sample_idx"]:
                        raise ValueError("scorer identity is outside its run cell in the frozen registry")
            run_status = audit_run_provenance(run_dir, rows, outcomes)
            if run_status["stage"] != stage:
                raise ValueError("run manifest/status stage differs from its raw records")
            # Validate every complete durable record, including the trailing
            # batch excluded from metrics when ordinary trace evidence is short.
            audit_raw(rows, outcomes, registry, allow_unpaired_scorer=True)
            if run_status["status_reconciled"]:
                status_reconciliations.append({
                    "run_dir": str(run_dir),
                    "fields": run_status["status_reconciled_fields"],
                })
            truncated_artifacts.extend(run_status["interrupted_jsonl_artifacts"])
            if not run_status["scorer_complete"] or (
                    not run_status["evaluator_complete"] or not run_status["trace_complete"] or
                    truncated_artifacts):
                incomplete_runs.append({"run_dir": str(run_dir), "stage": stage,
                                        "scored_queries": len(rows), "evaluator_queries": len(outcomes),
                                        "reason": ("truncated trailing JSONL write" if truncated_artifacts else
                                                   "run ended before scorer/evaluator/trace completion"),
                                        "truncated_artifacts": sorted(set(truncated_artifacts))})
            trace_count = run_status["trace_prefix_rows"]
            valid_hashes = {row["record_sha256"] for row in rows
                            if row["query"]["timestep"] < trace_count}
            action_rows.extend(row for row in rows if row["record_sha256"] in valid_hashes)
            outcome_rows.extend(row for row in outcomes if row["scorer_record_sha256"] in valid_hashes)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            integrity_errors.append(f"{run_dir}: {exc}")
    result = analyze_raw(action_rows, outcome_rows, registry, integrity_errors=integrity_errors,
                         allow_unpaired_scorer=bool(incomplete_runs))
    result = _apply_incomplete_run_status(result, incomplete_runs)
    if stage_b_run_present and result.get("decision") != "INVALID":
        stage_a = result.get("stage_a") or {}
        if stage_a.get("decision") != "GO_CONFIRM":
            reason = "Stage B run exists without a recomputed Stage-A GO_CONFIRM gate"
        elif any(objective != stage_a.get("winner") for objective in stage_b_objectives):
            reason = "Stage-B run configuration differs from the recomputed Stage-A winner"
        else:
            reason = None
        if reason:
            result["decision"] = "INVALID"
            result.setdefault("integrity_errors", []).append(reason)
            result["stage_b"] = {"decision": "INVALID", "complete": False,
                                 "integrity_errors": [reason]}
    if status_reconciliations:
        result["status_reconciliations"] = status_reconciliations
    if registry is not None:
        result["registry_sha256"] = registry["sha256"]
        result["source_revision"] = registry["source_revision"]
    result["raw_input_sha256"] = digest({
        "scorer_records": sorted(row["record_sha256"] for row in all_action_rows),
        "evaluator_records": sorted(row["outcome_record_sha256"] for row in all_outcome_rows),
        "raw_jsonl_files": sorted(raw_file_hashes,
                                   key=lambda row: (row["run_id"], row["artifact"], row["sha256"])),
    })
    return result


def _apply_incomplete_run_status(result, incomplete_runs):
    """Scope an interrupted run's disposition to the stage that was incomplete."""
    if not incomplete_runs or result.get("decision") == "INVALID":
        return result
    result["incomplete_runs"] = incomplete_runs
    stages = {row["stage"] for row in incomplete_runs}
    reason = "run status reports missing scorer/evaluator records"
    if "stage-a" in stages:
        if result.get("stage_b") is not None:
            result["decision"] = "INVALID"
            result.setdefault("integrity_errors", []).append(
                "Stage B is present while Stage-A evidence is incomplete")
            result["stage_b"]["decision"] = "INVALID"
            result["stage_b"]["complete"] = False
            return result
        stage_a = result.get("stage_a")
        if stage_a:
            stage_a["decision"] = "INCONCLUSIVE"
            stage_a["complete"] = False
            stage_a["incomplete_reason"] = reason
        result["decision"] = "INCONCLUSIVE"
    if "stage-b" in stages:
        stage_b = result.get("stage_b")
        if stage_b is None:
            stage_b = {"decision": "INCONCLUSIVE", "complete": False,
                       "incomplete_reason": reason, "cells": {}}
            result["stage_b"] = stage_b
        else:
            stage_b["decision"] = "INCONCLUSIVE"
            stage_b["complete"] = False
            stage_b["incomplete_reason"] = reason
        result["decision"] = "INCONCLUSIVE"
    return result


def _verify_committed_stage_a_decision(analysis, analysis_path, evidence_dir, registry):
    """Require a tracked, clean decision that exactly matches raw recomputation."""
    if analysis is None or analysis_path is None or evidence_dir is None:
        raise ValueError("Stage B requires a committed Stage-A decision and its raw evidence directory")
    if registry is None:
        raise ValueError("Stage B requires the frozen registry used by Stage A")
    path = Path(analysis_path).expanduser().resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("Stage-A decision must be committed inside the repository") from exc
    if not relative.startswith("plans/"):
        raise ValueError("Stage-A decision must be stored under plans/")
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative], cwd=ROOT,
                   check=True, capture_output=True, text=True)
    if _git_output("status", "--porcelain", "--", relative):
        raise ValueError("Stage-A decision must be committed and clean")
    source_revision = registry.get("source_revision")
    if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("Stage-A registry source revision is not a full Git commit hash")
    head = _git_output("rev-parse", "HEAD")
    subprocess.run(["git", "merge-base", "--is-ancestor", source_revision, head], cwd=ROOT,
                   check=True, capture_output=True, text=True)
    frozen_paths = ("src", "cfg/research", "docs/research")
    changed = subprocess.run(["git", "diff", "--name-only", source_revision, head, "--", *frozen_paths],
                             cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = _git_output("status", "--porcelain", "--", *frozen_paths)
    if changed or dirty or _git_output("status", "--porcelain"):
        raise ValueError("Stage-A source, protocol or configuration changed after its locked revision")
    recomputed = analyze_evidence(evidence_dir, registry, allowed_stages={"stage-a"})
    if (recomputed.get("decision") != "GO_CONFIRM" or
            recomputed["stage_a"].get("decision") != "GO_CONFIRM" or recomputed["stage_b"] is not None):
        raise ValueError("raw Stage-A evidence does not independently recompute to GO_CONFIRM")
    if analysis != recomputed:
        raise ValueError("committed Stage-A decision differs from independent raw recomputation")
    return recomputed, head


def _scan_run(run_dir, *, cell=None, fingerprint=None, source_revision=None,
              expected_protocol_sha256=None):
    root = Path(run_dir)
    scanner = read_jsonl(root / "qcgs-exact-scan.jsonl")
    trace = read_jsonl(root / "trace.jsonl")
    if len(scanner) != len(trace) or not scanner:
        raise ValueError("scan sidecar and evaluator trace must cover the same nonempty prefix")
    rows = []
    for timestep, (score, evaluated) in enumerate(zip(scanner, trace)):
        if score["timestep"] != timestep or evaluated["timestep"] != timestep:
            raise ValueError("scan must be contiguous from stream timestep zero")
        if score["sample_idx"] != evaluated["sample_idx"]:
            raise ValueError("scan-sidecar/stream identity mismatch")
        rows.append({"timestep": timestep, "sample_idx": int(score["sample_idx"]),
                     "domain": int(evaluated["ground_truth_domain"]),
                     "is_ood": bool(evaluated["is_ood"]),
                     "eligible": bool(score["eligible"]),
                     "baseline_sha256": score["baseline_sha256"],
                     "aggregate_sha256": score["aggregate_sha256"],
                     "cache_snapshot_sha256": score["cache_snapshot_sha256"]})
    if len(rows) not in EXACT_OBJECTIVE_SCAN_CAPS:
        raise ValueError("scan row count must follow 600-row doubling, capped at 6000")
    if cell is not None:
        from evaluation.qcgs_exact_objective_audit import audit_run_provenance
        receipt = audit_run_provenance(root, [], [],
                                       expected_protocol_sha256=expected_protocol_sha256)
        manifest = json.loads((root / "manifest.json").read_text())
        config = manifest["config"]
        if (receipt["ood_cell"] != cell or receipt["dataset_fingerprint"] != fingerprint or
                receipt["source_revision"] != source_revision or config.get("qcgs_mode") != "scan" or
                len(rows) != manifest["args"].get("max_eval_samples") or
                receipt["scan_rows"] != len(rows) or not receipt["trace_complete"]):
            raise ValueError("unscored scan source/data/cell/prefix differs from the frozen registry inputs")
    return rows


def build_smoke_registry(scan_runs, historical_path=HISTORICAL_EXCLUSIONS):
    if set(scan_runs) != set(CELLS):
        raise ValueError("smoke registry requires both OOD cells")
    historical, historical_sha = _historical(historical_path)
    excluded = set(historical)
    cells = {}
    for cell in CELLS:
        rows = scan_runs[cell]
        if len(rows) != 600:
            raise ValueError("smoke registry is built from the initial 600-row unscored scan")
        selected, seen = [], set()
        for row in rows:
            sample_idx = row["sample_idx"]
            if row["is_ood"] or not row["eligible"] or sample_idx in excluded or sample_idx in seen:
                continue
            selected.append(dict(row))
            seen.add(sample_idx)
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise ValueError(f"fewer than two eligible smoke queries at OOD={cell}")
        prefix = 100 * (1 + max(row["timestep"] for row in selected) // 100)
        cells[cell] = {"selected": selected, "max_eval_samples": prefix}
    value = {"schema_version": 1, "namespace": EXACT_OBJECTIVE_NAMESPACE + "-smoke",
             "historical_base_image_indices_sha256": historical_sha,
             "cells": cells}
    value["sha256"] = digest(value)
    validate_smoke_registry(value)
    return value


def validate_smoke_registry(value):
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if (value.get("schema_version") != 1 or
            value.get("namespace") != EXACT_OBJECTIVE_NAMESPACE + "-smoke" or
            value.get("sha256") != digest(payload) or set(value.get("cells", {})) != set(CELLS)):
        raise ValueError("smoke registry version/hash/cells mismatch")
    historical, historical_sha = _historical(HISTORICAL_EXCLUSIONS)
    if value["historical_base_image_indices_sha256"] != historical_sha:
        raise ValueError("smoke registry historical exclusions changed")
    excluded = set(historical)
    for cell in CELLS:
        selected = value["cells"][cell]["selected"]
        if len(selected) != 2 or len({row["sample_idx"] for row in selected}) != 2:
            raise ValueError("smoke registry must contain two distinct queries per cell")
        if any(row["sample_idx"] in excluded or row["is_ood"] or not row["eligible"] for row in selected):
            raise ValueError("smoke registry contains excluded or ineligible query")
        last = max(row["timestep"] for row in selected)
        if value["cells"][cell]["max_eval_samples"] != 100 * (1 + last // 100):
            raise ValueError("smoke registry prefix is not shortest complete batch prefix")


def build_registry(scan_runs, smoke_registry, smoke_runs, fingerprint, source_revision,
                   historical_path=HISTORICAL_EXCLUSIONS, *, scan_source_revision=None,
                   smoke_source_revision=None, scan_protocol_sha256=None,
                   smoke_protocol_sha256=None):
    validate_smoke_registry(smoke_registry)
    if set(scan_runs) != set(CELLS) or set(smoke_runs) != set(CELLS):
        raise ValueError("scientific registry requires scans and smoke runs for both cells")
    scan_source_revision = scan_source_revision or source_revision
    smoke_source_revision = smoke_source_revision or source_revision
    active_protocol_sha256 = file_sha(ROOT / "docs/research/qcgs-exact-label-free-objective-audit.md")
    scan_protocol_sha256 = scan_protocol_sha256 or active_protocol_sha256
    smoke_protocol_sha256 = smoke_protocol_sha256 or active_protocol_sha256
    from evaluation.qcgs_exact_objective_audit import (
        CURRENT_PROTOCOL, LEGACY_PROTOCOL_SHA256, LEGACY_SMOKE_PROTOCOL,
    )
    supported_protocol_hashes = {active_protocol_sha256, LEGACY_PROTOCOL_SHA256}
    if scan_protocol_sha256 not in supported_protocol_hashes or smoke_protocol_sha256 not in supported_protocol_hashes:
        raise ValueError("registry inputs must use the current protocol or the frozen v1.1 receipts")
    for name, revision in (("registry", source_revision), ("scan", scan_source_revision),
                           ("smoke", smoke_source_revision)):
        if (not isinstance(revision, str) or len(revision) != 40 or
                any(char not in "0123456789abcdef" for char in revision)):
            raise ValueError("invalid " + name + " source revision")
    historical, historical_sha = _historical(historical_path)
    smoke_ids = []
    smoke_rows_by_cell = {}
    from evaluation.qcgs_exact_objective_audit import audit_run_provenance, audit_scorer_rows
    accepted_smoke_protocol = (CURRENT_PROTOCOL if smoke_protocol_sha256 == active_protocol_sha256
                               else LEGACY_SMOKE_PROTOCOL)
    for cell in CELLS:
        selected = smoke_registry["cells"][cell]["selected"]
        run_dir = Path(smoke_runs[cell])
        actual = read_jsonl(run_dir / "qcgs-exact-actions.jsonl")
        audit_scorer_rows(actual, allowed_protocols={accepted_smoke_protocol})
        if len(actual) != 2 or any(row["stage"] != "smoke" for row in actual):
            raise ValueError("smoke run must preserve exactly two scorer-only query records")
        receipt = audit_run_provenance(run_dir, actual, [],
                                       expected_protocol_sha256=smoke_protocol_sha256)
        if (receipt["ood_cell"] != cell or receipt["registry_sha256"] != smoke_registry["sha256"] or
                receipt["dataset_fingerprint"] != fingerprint or
                receipt["source_revision"] != smoke_source_revision or
                receipt["protocol_sha256"] != smoke_protocol_sha256 or
                not receipt["scorer_complete"] or receipt["evaluator_complete"] is not None or
                not receipt["trace_complete"] or receipt["interrupted_jsonl_artifacts"] or
                receipt["scan_rows"] != json.loads((run_dir / "manifest.json").read_text())[
                    "args"]["max_eval_samples"] or
                any(row.get("registry_sha256") != smoke_registry["sha256"] for row in actual)):
            raise ValueError("smoke receipt is incomplete or bound to different source, data, cell or registry")
        if any(row.get("dataset_fingerprint") != fingerprint or
               row.get("source_revision") != smoke_source_revision or
               row.get("protocol_sha256") != smoke_protocol_sha256 for row in actual):
            raise ValueError("smoke source/data provenance differs from the frozen registry inputs")
        expected = {(row["timestep"], row["sample_idx"]) for row in selected}
        observed = {(row["query"]["timestep"], row["query"]["sample_idx"]) for row in actual}
        if observed != expected:
            raise ValueError("smoke run identities differ from its separate registry")
        if (Path(smoke_runs[cell]) / "qcgs-exact-outcomes.jsonl").exists():
            raise ValueError("smoke evaluator outcomes must not be used or persisted")
        smoke_ids.extend(row["query"]["sample_idx"] for row in actual)
        smoke_rows_by_cell[cell] = actual
    combined = sorted(set(historical) | set(smoke_ids))
    smoke_chunk_sizes = {row["costs"]["action_chunk_size"]
                         for cell in CELLS for row in read_jsonl(Path(smoke_runs[cell]) / "qcgs-exact-actions.jsonl")}
    if len(smoke_chunk_sizes) != 1:
        raise ValueError("smoke runs must use one identical locked action chunk size")
    locked_chunk_size = next(iter(smoke_chunk_sizes))
    peaks_by_cell = {
        cell: [row["costs"].get("peak_device_memory_bytes") for row in smoke_rows_by_cell[cell]]
        for cell in CELLS
    }
    if any(type(peak) is not int or peak < 0 for peaks in peaks_by_cell.values() for peak in peaks):
        raise ValueError("smoke must record a nonnegative CUDA allocator peak for every query")
    view_state_hashes = {row.get("view_state", {}).get("sha256")
                         for rows in smoke_rows_by_cell.values() for row in rows}
    parameter_order_hashes = {row.get("parameter_order_sha256")
                              for rows in smoke_rows_by_cell.values() for row in rows}
    if (len(view_state_hashes) != 1 or not all(isinstance(value, str) and len(value) == 64
                                               for value in view_state_hashes) or
            len(parameter_order_hashes) != 1 or not all(isinstance(value, str) and len(value) == 64
                                                        for value in parameter_order_hashes)):
        raise ValueError("both smoke cells must agree on locked transform and adapted-parameter order hashes")
    exclusions = {"base_image_indices": combined,
                  "base_image_indices_sha256": digest(combined),
                  "historical_base_image_indices": historical,
                  "historical_base_image_indices_sha256": historical_sha,
                  "smoke_base_image_indices": sorted(set(smoke_ids)),
                  "smoke_base_image_indices_sha256": digest(sorted(set(smoke_ids))),
                  "interpretation_limit": "Historical raw archives are unavailable; prior unscored cache exposure is not certified."}
    scans = {cell: scan_runs[cell] for cell in CELLS}
    provenance = {"smoke_registry_sha256": smoke_registry["sha256"],
                  "active_protocol": CURRENT_PROTOCOL,
                  "active_protocol_sha256": active_protocol_sha256,
                  "peak_memory_policy": "report_only",
                  "scan_source_revision": scan_source_revision,
                  "scan_protocol_sha256": scan_protocol_sha256,
                  "smoke_source_revision": smoke_source_revision,
                  "smoke_protocol_sha256": smoke_protocol_sha256,
                  "smoke_peak_device_memory_bytes": {cell: max(peaks_by_cell[cell]) for cell in CELLS},
                  "smoke_peak_device_memory_by_query_bytes": peaks_by_cell,
                  "action_chunk_size": locked_chunk_size,
                  "view_state_sha256": next(iter(view_state_hashes)),
                  "parameter_order_sha256": next(iter(parameter_order_hashes)),
                  "scan_content_sha256": {cell: digest(scans[cell]) for cell in CELLS},
                  "scan_row_counts": {cell: len(scans[cell]) for cell in CELLS},
                  "smoke_action_sha256": {cell: file_sha(Path(smoke_runs[cell]) / "qcgs-exact-actions.jsonl")
                                          for cell in CELLS},
                  "historical_exclusion_sha256": file_sha(historical_path),
                  "extension_reason": "shortest outcome-blind row prefix covering both frozen stage quotas"}
    return build_exact_objective_registry(scans, fingerprint, exclusions,
                                          source_revision=source_revision, provenance=provenance)


def _main_config():
    import yaml
    path = AUDIT_CONFIG / "CIFAR100C" / METHOD_CONFIG
    return yaml.safe_load(path.read_text())


def _stage_schedule(stage, cell, registry, smoke_registry, analysis):
    if stage == "scan":
        return [], None, None, None
    if stage == "smoke":
        validate_smoke_registry(smoke_registry)
        value = smoke_registry["cells"][cell]
        schedule = [{"timestep": row["timestep"], "sample_idx": row["sample_idx"]}
                    for row in value["selected"]]
        return schedule, smoke_registry["sha256"], None, value["max_eval_samples"]
    validate_exact_objective_registry(registry)
    schedule = exact_objective_query_schedule(registry, stage, cell)
    frozen = None
    if stage == "stage-b":
        if analysis is None or analysis.get("stage_a", {}).get("decision") != "GO_CONFIRM":
            raise ValueError("Stage B requires independently recomputed Stage-A GO_CONFIRM")
        frozen = analysis["stage_a"].get("winner")
        if frozen not in ("single_view_entropy", "memo_marginal_entropy", "cliptta_image_to_text_entropy"):
            raise ValueError("Stage-A analysis has no frozen objective winner")
    return schedule, registry["sha256"], frozen, registry["cells"][cell]["max_eval_samples"]


def make_job(stage, cell, output_config, *, data_root, evidence_dir, run_id,
             max_eval_samples=None, fingerprint=None, source_revision=None,
             registry=None, smoke_registry=None, analysis=None, analysis_path=None,
             stage_a_evidence_dir=None, action_chunk_size=1, profile_cost_components=False,
             profile_only=False, profile_query_index=0):
    if stage not in ("scan", "smoke", "stage-a", "stage-b") or cell not in CELLS:
        raise ValueError("unknown exact-objective stage/cell")
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ValueError("run_id must be path-safe")
    if type(action_chunk_size) is not int or action_chunk_size <= 0:
        raise ValueError("action chunk size must be a positive integer")
    if profile_cost_components and stage != "smoke":
        raise ValueError("component profiling is restricted to cost-only smoke jobs")
    if profile_only and (stage != "smoke" or not profile_cost_components or profile_query_index not in (0, 1)):
        raise ValueError("profile-only jobs require one of the two separately excluded smoke IDs")
    if stage in ("scan", "smoke") and (not fingerprint or not source_revision):
        raise ValueError("scan/smoke jobs require the preflight dataset fingerprint and frozen source revision")
    run_source_revision = source_revision
    if stage == "scan":
        if max_eval_samples not in EXACT_OBJECTIVE_SCAN_CAPS:
            raise ValueError("scan prefix must be one of the frozen doubling caps")
        schedule, registry_sha, frozen, prefix = [], None, None, max_eval_samples
    else:
        run_source_revision = registry["source_revision"] if registry is not None else source_revision
        if stage == "stage-b":
            analysis, run_source_revision = _verify_committed_stage_a_decision(
                analysis, analysis_path, stage_a_evidence_dir, registry)
        schedule, registry_sha, frozen, prefix = _stage_schedule(
            stage, cell, registry, smoke_registry, analysis)
        if profile_only:
            schedule = [schedule[profile_query_index]]
        if stage in ("stage-a", "stage-b"):
            locked_chunk = registry.get("provenance", {}).get("action_chunk_size")
            if locked_chunk is None or locked_chunk != action_chunk_size:
                raise ValueError("Stage A/B action chunk size must match both separately excluded smoke runs")
    config = _main_config()
    config.update(qcgs_mode=stage, objective_stage="all" if stage != "stage-b" else frozen,
                  query_schedule=schedule, dataset_fingerprint=fingerprint,
                  source_revision=run_source_revision, registry_sha256=registry_sha,
                  protocol_sha256=file_sha(ROOT / "docs/research/qcgs-exact-label-free-objective-audit.md"),
                  random_seed=0, run_token=uuid.uuid4().hex,
                  action_chunk_size=action_chunk_size,
                  profile_cost_components=bool(profile_cost_components),
                  profile_only=bool(profile_only))
    if stage in ("stage-a", "stage-b"):
        config["dataset_fingerprint"] = registry["dataset_fingerprint"]
        config["registry_source_revision"] = registry["source_revision"]
        config["view_state_sha256"] = registry["provenance"]["view_state_sha256"]
        config["parameter_order_sha256"] = registry["provenance"]["parameter_order_sha256"]
    if stage == "smoke":
        config["dataset_fingerprint"] = fingerprint
        config["source_revision"] = source_revision
    if frozen is not None:
        config["frozen_objective"] = frozen
    import yaml
    config_root = Path(output_config).expanduser().resolve()
    config_path = config_root / "CIFAR100C" / METHOD_CONFIG
    config_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(config, sort_keys=True)
    if config_path.exists() and config_path.read_text() != encoded:
        raise FileExistsError("refusing to overwrite a different frozen exact-objective config")
    config_path.write_text(encoded)
    ratio = float(cell)
    command = [sys.executable, str(ROOT / "src/main.py"),
        "--dataset", "CIFAR100C", "--data_root", str(Path(data_root).expanduser().resolve()),
        "--artifact-provenance", "fast", "--open_set", "--ood_ratio", str(ratio),
        "--known_class_split", "open-set-cifar100-split-v1",
        "--known-class-split-path", str(SPLIT_PATH), "--known-class-split-sha256", file_sha(SPLIT_PATH),
        "--model", "clip_vitbase16", "--tta_algo", "QCGSExactObjectiveAudit", "--batch_size", "100",
        "--device", "cuda", "--config", str(config_root),
        "--config-lock-path", str(config_path), "--config-lock-sha256", file_sha(config_path),
        "--stream_mode", "block", "--stream_block_size", "64", "--seed", "0", "--stream_seed", "0",
        "--num_workers", "0", "--open_set_per_domain_source_budget", "400",
        "--max_eval_samples", str(prefix), "--run_id", run_id,
        "--evidence_dir", str(Path(evidence_dir).expanduser().resolve()),
        "--save_to", str(Path(evidence_dir).expanduser().resolve() / "qcgs-exact-runs.csv")]
    return {"stage": stage, "cell": cell, "run_id": run_id, "max_eval_samples": prefix,
            "config": str(config_path), "config_sha256": file_sha(config_path), "command": command}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("make-smoke-registry", help="select two separate, outcome-blind smoke IDs per cell")
    for cell in CELLS:
        smoke.add_argument(f"--scan-{cell.replace('.', '')}", required=True, dest=f"scan_{cell.replace('.', '')}")
    smoke.add_argument("--dataset-fingerprint", required=True)
    smoke.add_argument("--source-revision", required=True)
    smoke.add_argument("--historical-exclusions", type=Path, default=HISTORICAL_EXCLUSIONS)
    smoke.add_argument("--output", type=Path, required=True)
    registry = subparsers.add_parser("build-registry", help="freeze historical/smoke exclusions and Stage A/B IDs")
    registry.add_argument("--scan-0", required=True, type=Path)
    registry.add_argument("--scan-05", required=True, type=Path)
    registry.add_argument("--smoke-registry", required=True, type=Path)
    registry.add_argument("--smoke-run-0", required=True, type=Path)
    registry.add_argument("--smoke-run-05", required=True, type=Path)
    registry.add_argument("--historical-exclusions", type=Path, default=HISTORICAL_EXCLUSIONS)
    registry.add_argument("--dataset-fingerprint", required=True)
    registry.add_argument("--source-revision", required=True)
    registry.add_argument("--scan-source-revision",
                          help="source revision of reused scans; defaults to --source-revision")
    registry.add_argument("--smoke-source-revision",
                          help="source revision of reused smoke receipts; defaults to --source-revision")
    registry.add_argument("--scan-protocol-sha256",
                          help="protocol hash of reused scans; defaults to the active protocol")
    registry.add_argument("--smoke-protocol-sha256",
                          help="protocol hash of reused smokes; defaults to the active protocol")
    registry.add_argument("--output", type=Path, required=True)
    job = subparsers.add_parser("job", help="write a locked per-cell config and print the exact launcher command")
    job.add_argument("--stage", choices=("scan", "smoke", "stage-a", "stage-b"), required=True)
    job.add_argument("--cell", choices=CELLS, required=True)
    job.add_argument("--data-root", type=Path, required=True)
    job.add_argument("--evidence-dir", type=Path, required=True)
    job.add_argument("--output-config", type=Path, required=True)
    job.add_argument("--run-id")
    job.add_argument("--max-eval-samples", type=int)
    job.add_argument("--action-chunk-size", type=int, default=1,
                     help="maximum concurrent query actions per exact trial; every action is still evaluated")
    job.add_argument("--profile-cost-components", action="store_true",
                     help="record detailed timings for a cost-only smoke run")
    job.add_argument("--profile-only", action="store_true",
                     help="run one separately excluded smoke ID for component profiling")
    job.add_argument("--profile-query-index", type=int, default=0,
                     help="0 or 1: select one query from the frozen smoke registry for profile-only mode")
    job.add_argument("--dataset-fingerprint")
    job.add_argument("--source-revision")
    job.add_argument("--registry", type=Path)
    job.add_argument("--smoke-registry", type=Path)
    job.add_argument("--analysis", type=Path)
    job.add_argument("--stage-a-evidence-dir", type=Path,
                     help="required with Stage B; directory containing committed-decision raw Stage-A runs")
    audit = subparsers.add_parser("analyze", help="recompute gates from a campaign root containing Stage A and optional Stage B runs")
    audit.add_argument("--evidence-dir", type=Path, required=True,
                       help="campaign root; include both Stage A and Stage B runs for final confirmation analysis")
    audit.add_argument("--registry", type=Path)
    audit.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "make-smoke-registry":
        paths = {"0": args.scan_0, "0.5": args.scan_05}
        value = build_smoke_registry({cell: _scan_run(
            paths[cell], cell=cell, fingerprint=args.dataset_fingerprint,
            source_revision=args.source_revision) for cell in CELLS}, args.historical_exclusions)
        write_json(args.output, value)
        print(json.dumps({"smoke_registry_sha256": value["sha256"], "queries_per_cell": 2,
                          "output": str(args.output)}, indent=2))
        return
    if args.command == "build-registry":
        paths = {"0": args.scan_0, "0.5": args.scan_05}
        scan_source_revision = args.scan_source_revision or args.source_revision
        smoke_source_revision = args.smoke_source_revision or args.source_revision
        active_protocol_sha256 = file_sha(ROOT / "docs/research/qcgs-exact-label-free-objective-audit.md")
        scan_protocol_sha256 = args.scan_protocol_sha256 or active_protocol_sha256
        smoke_protocol_sha256 = args.smoke_protocol_sha256 or active_protocol_sha256
        scans = {cell: _scan_run(paths[cell], cell=cell, fingerprint=args.dataset_fingerprint,
                                 source_revision=scan_source_revision,
                                 expected_protocol_sha256=scan_protocol_sha256) for cell in CELLS}
        smoke_registry = json.loads(args.smoke_registry.read_text())
        smoke_runs = {"0": args.smoke_run_0, "0.5": args.smoke_run_05}
        try:
            value = build_registry(scans, smoke_registry, smoke_runs, args.dataset_fingerprint,
                                   args.source_revision, args.historical_exclusions,
                                   scan_source_revision=scan_source_revision,
                                   smoke_source_revision=smoke_source_revision,
                                   scan_protocol_sha256=scan_protocol_sha256,
                                   smoke_protocol_sha256=smoke_protocol_sha256)
        except ValueError as exc:
            if not str(exc).startswith("insufficient eligible unseen exact-objective queries"):
                raise
            cap = len(scans["0"])
            if cap == 6000:
                value = {"status": "INCONCLUSIVE", "reason": str(exc), "maximum_scan_rows_per_cell": 6000}
                write_json(args.output, value)
                print(json.dumps(value, indent=2))
                return
            value = {"status": "EXTEND_UNSCORED_SCAN", "reason": str(exc),
                     "next_scan_rows_per_cell": min(cap * 2, 6000)}
            print(json.dumps(value, indent=2))
            return
        write_json(args.output, value)
        print(json.dumps({"status": "REGISTRY_FROZEN", "registry_sha256": value["sha256"],
                          "max_eval_samples": {cell: value["cells"][cell]["max_eval_samples"] for cell in CELLS},
                          "output": str(args.output)}, indent=2))
        return
    if args.command == "job":
        registry = json.loads(args.registry.read_text()) if args.registry else None
        smoke_registry = json.loads(args.smoke_registry.read_text()) if args.smoke_registry else None
        analysis = json.loads(args.analysis.read_text()) if args.analysis else None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = args.run_id or f"{args.stage}-ood-{args.cell}-s0-{timestamp}"
        result = make_job(args.stage, args.cell, args.output_config, data_root=args.data_root,
                          evidence_dir=args.evidence_dir, run_id=run_id,
                          max_eval_samples=args.max_eval_samples,
                          action_chunk_size=args.action_chunk_size,
                          fingerprint=args.dataset_fingerprint,
                          source_revision=args.source_revision, registry=registry,
                          smoke_registry=smoke_registry, analysis=analysis,
                          analysis_path=args.analysis, stage_a_evidence_dir=args.stage_a_evidence_dir,
                          profile_cost_components=args.profile_cost_components,
                          profile_only=args.profile_only, profile_query_index=args.profile_query_index)
        print(json.dumps(result, indent=2))
        return
    if args.command == "analyze":
        registry = json.loads(args.registry.read_text()) if args.registry else None
        result = analyze_evidence(args.evidence_dir, registry, allowed_stages={"stage-a", "stage-b"})
        output = args.output or (args.evidence_dir / "qcgs-exact-analysis.json")
        write_json(output, result)
        print(json.dumps({"decision": result["decision"], "output": str(output)}, indent=2))
        return


if __name__ == "__main__":
    main()
