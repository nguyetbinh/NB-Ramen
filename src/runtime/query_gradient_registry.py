"""Outcome-independent query identity, exclusions and immutable pilot registries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


STAGES = {"stage-a": 16, "stage-b": 128}
CELLS = ("0", "0.5")
EXACT_OBJECTIVE_NAMESPACE = "qcgs-exact-objective-audit-v1"
EXACT_OBJECTIVE_STAGES = {"stage-a": 16, "stage-b": 128}
EXACT_OBJECTIVE_SCAN_CAPS = (600, 1200, 2400, 4800, 6000)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def assigned_stage(fingerprint, sample_idx, namespace="qcgs-label-free-v1"):
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("dataset fingerprint must be lowercase SHA-256")
    if type(sample_idx) is not int or not 0 <= sample_idx < 10000:
        raise ValueError("CIFAR base sample index out of range")
    if namespace not in ("qcgs-label-free-v1", "qcgs-multiview-rescue-v1", EXACT_OBJECTIVE_NAMESPACE):
        raise ValueError("unknown registry namespace")
    key = f"{namespace}|{fingerprint}|{sample_idx}"
    return "stage-a" if int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 9 == 0 else "stage-b"


def build_registry(scans, fingerprint, exclusions, *, source_revision, provenance, namespace="qcgs-label-free-v1"):
    """Take the earliest eligible distinct base images; never inspect utility."""
    if set(scans) != set(CELLS):
        raise ValueError("require both OOD cells")
    excluded = set(exclusions["base_image_indices"])
    cells = {}
    for cell in CELLS:
        rows = scans[cell]
        if [r["timestep"] for r in rows] != list(range(len(rows))):
            raise ValueError("scan must contain the entire contiguous stream prefix")
        selected, seen = {s: [] for s in STAGES}, set()
        for row in rows:
            i = row["sample_idx"]
            stage = assigned_stage(fingerprint, i, namespace)
            if row["is_ood"] or not row["eligible"] or i in excluded or i in seen:
                continue
            if len(selected[stage]) < STAGES[stage]:
                selected[stage].append(dict(row))
                seen.add(i)
        missing = {s: STAGES[s] - len(selected[s]) for s in STAGES if len(selected[s]) != STAGES[s]}
        if missing:
            raise ValueError(f"insufficient eligible unseen queries at OOD={cell}: {missing}")
        cells[cell] = {"stream": rows, "selected": selected,
                       "max_eval_samples": 100 * (1 + max(r["timestep"] for v in selected.values() for r in v) // 100)}
    value = {"schema_version": 1, "dataset_fingerprint": fingerprint,
             "source_revision": source_revision, "provenance": provenance,
             "exclusions": exclusions, "cells": cells}
    if namespace != "qcgs-label-free-v1":
        value.update(schema_version=2, namespace=namespace)
    validate_registry(value)
    value["sha256"] = digest(value)
    return value


def validate_registry(value):
    namespace = value.get("namespace", "qcgs-label-free-v1")
    if (value["schema_version"], namespace) not in ((1, "qcgs-label-free-v1"), (2, "qcgs-multiview-rescue-v1")):
        raise ValueError("registry version/namespace mismatch")
    payload = {k: v for k, v in value.items() if k != "sha256"}
    if "sha256" in value and value["sha256"] != digest(payload):
        raise ValueError("registry hash mismatch")
    if set(value["cells"]) != set(CELLS):
        raise ValueError("registry cells mismatch")
    excluded = set(value["exclusions"]["base_image_indices"])
    all_ids = {s: set() for s in STAGES}
    for cell in value["cells"].values():
        stream = cell["stream"]
        if [r["timestep"] for r in stream] != list(range(len(stream))):
            raise ValueError("registry stream is not contiguous")
        if cell["max_eval_samples"] % 100 or cell["max_eval_samples"] > len(stream):
            raise ValueError("invalid registry stream budget")
        if namespace == "qcgs-multiview-rescue-v1":
            shortest = 100 * (1 + max(r["timestep"] for rows in cell["selected"].values() for r in rows) // 100)
            if cell["max_eval_samples"] != shortest:
                raise ValueError("registry must freeze the shortest whole-batch prefix")
        seen = set()
        for stage, count in STAGES.items():
            rows = cell["selected"][stage]
            if len(rows) != count:
                raise ValueError("registry query count mismatch")
            for row in rows:
                i = row["sample_idx"]
                if i in excluded or i in seen or row["is_ood"] or not row["eligible"]:
                    raise ValueError("registry overlap, excluded image or invalid eligibility")
                if assigned_stage(value["dataset_fingerprint"], i, namespace) != stage:
                    raise ValueError("registry hash assignment mismatch")
                if stream[row["timestep"]] != row or row["timestep"] >= cell["max_eval_samples"]:
                    raise ValueError("registry stream identity mismatch")
                seen.add(i)
                all_ids[stage].add(i)
            # Enforce earliest eligible observations, not an arbitrary valid subset.
            expected, taken = [], set()
            for row in stream:
                i = row["sample_idx"]
                if not row["is_ood"] and row["eligible"] and i not in excluded and i not in taken and assigned_stage(value["dataset_fingerprint"], i, namespace) == stage:
                    expected.append(row)
                    taken.add(i)
                    if len(expected) == count:
                        break
            if rows != expected:
                raise ValueError("registry does not use earliest eligible queries")
    if all_ids["stage-a"] & all_ids["stage-b"]:
        raise ValueError("base images overlap between stages")


def historical_exclusions(runs):
    indices, sources = set(), []
    for run in map(Path, runs):
        raw, trace = run / "oracle-support-queries.jsonl", run / "trace.jsonl"
        mapping = {r["timestep"]: r for r in map(json.loads, trace.read_text().splitlines())}
        rows = list(map(json.loads, raw.read_text().splitlines()))
        for row in rows:
            entry = mapping[row["query_index"]]
            if entry["ground_truth_domain"] != row["query_domain"]:
                raise ValueError("historical trace/query identity mismatch")
            indices.add(entry["sample_idx"])
        sources.append({"run": run.name, "queries_sha256": file_sha(raw),
                        "trace_sha256": file_sha(trace), "queries": len(rows)})
    return {"schema_version": 1, "base_image_indices": sorted(indices), "sources": sources}


def build_exact_objective_registry(scans, fingerprint, exclusions, *, source_revision, provenance):
    """Freeze shortest-prefix, outcome-blind exact-objective query assignments."""
    if set(scans) != set(CELLS):
        raise ValueError("require both exact-objective OOD cells")
    _validate_exact_objective_exclusions(exclusions)
    excluded = set(exclusions["base_image_indices"])
    cells = {}
    for cell in CELLS:
        rows = scans[cell]
        if len(rows) not in EXACT_OBJECTIVE_SCAN_CAPS:
            raise ValueError("scan must use the shortest-prefix doubling schedule through 6000 rows")
        if [row.get("timestep") for row in rows] != list(range(len(rows))):
            raise ValueError("scan must contain the contiguous stream prefix")
        selected = {stage: [] for stage in EXACT_OBJECTIVE_STAGES}
        seen = set()
        for row in rows:
            if any(key in row for key in ("label", "known_label", "correct", "utility", "oracle")):
                raise ValueError("registry scans must not contain evaluator outcomes")
            sample_idx = row["sample_idx"]
            stage = assigned_stage(fingerprint, sample_idx, EXACT_OBJECTIVE_NAMESPACE)
            if row["is_ood"] or not row["eligible"] or sample_idx in excluded or sample_idx in seen:
                continue
            if len(selected[stage]) < EXACT_OBJECTIVE_STAGES[stage]:
                selected[stage].append(dict(row))
                seen.add(sample_idx)
        missing = {stage: count - len(selected[stage]) for stage, count in EXACT_OBJECTIVE_STAGES.items()
                   if len(selected[stage]) != count}
        if missing:
            raise ValueError(f"insufficient eligible unseen exact-objective queries at OOD={cell}: {missing}")
        last_timestep = max(row["timestep"] for members in selected.values() for row in members)
        prefix_rows = 100 * (1 + last_timestep // 100)
        cells[cell] = {"stream": rows, "selected": selected, "max_eval_samples": prefix_rows}
    value = {"schema_version": 3, "namespace": EXACT_OBJECTIVE_NAMESPACE,
             "dataset_fingerprint": fingerprint, "source_revision": source_revision,
             "provenance": provenance, "exclusions": exclusions, "cells": cells}
    validate_exact_objective_registry(value)
    value["sha256"] = digest(value)
    return value


def validate_exact_objective_registry(value):
    """Validate the exact-audit split, historical exclusions and shortest prefix."""
    if value.get("schema_version") != 3 or value.get("namespace") != EXACT_OBJECTIVE_NAMESPACE:
        raise ValueError("exact-objective registry version/namespace mismatch")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if "sha256" in value and value["sha256"] != digest(payload):
        raise ValueError("exact-objective registry hash mismatch")
    fingerprint = value["dataset_fingerprint"]
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("invalid exact-objective dataset fingerprint")
    source_revision = value.get("source_revision")
    if (not isinstance(source_revision, str) or len(source_revision) != 40 or
            any(char not in "0123456789abcdef" for char in source_revision)):
        raise ValueError("invalid exact-objective source revision")
    if set(value["cells"]) != set(CELLS):
        raise ValueError("exact-objective registry cell mismatch")
    chunk_size = value.get("provenance", {}).get("action_chunk_size")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("exact-objective registry lacks the smoke-locked action chunk size")
    for key in ("view_state_sha256", "parameter_order_sha256"):
        locked_hash = value.get("provenance", {}).get(key)
        if not isinstance(locked_hash, str) or len(locked_hash) != 64 or any(
                char not in "0123456789abcdef" for char in locked_hash):
            raise ValueError("exact-objective registry lacks smoke-locked " + key)
    exclusions = value["exclusions"]
    _validate_exact_objective_exclusions(exclusions)
    historical = exclusions["historical_base_image_indices"]
    excluded = set(exclusions["base_image_indices"])
    stage_ids = {stage: set() for stage in EXACT_OBJECTIVE_STAGES}
    for cell in value["cells"].values():
        stream = cell["stream"]
        if len(stream) not in EXACT_OBJECTIVE_SCAN_CAPS or [r["timestep"] for r in stream] != list(range(len(stream))):
            raise ValueError("exact-objective registry scan prefix is invalid")
        selected = cell["selected"]
        if set(selected) != set(EXACT_OBJECTIVE_STAGES):
            raise ValueError("exact-objective stage assignment mismatch")
        selected_ids = set()
        all_selected = []
        for stage, count in EXACT_OBJECTIVE_STAGES.items():
            members = selected[stage]
            if len(members) != count:
                raise ValueError("exact-objective registry quota mismatch")
            expected, taken = [], set()
            for row in stream:
                sample_idx = row["sample_idx"]
                if (not row["is_ood"] and row["eligible"] and sample_idx not in excluded and
                        sample_idx not in taken and assigned_stage(fingerprint, sample_idx, EXACT_OBJECTIVE_NAMESPACE) == stage):
                    expected.append(row)
                    taken.add(sample_idx)
                    if len(expected) == count:
                        break
            if members != expected:
                raise ValueError("exact-objective registry does not use earliest eligible rows")
            for row in members:
                if row["sample_idx"] in selected_ids or row["is_ood"] or not row["eligible"]:
                    raise ValueError("exact-objective stage/cell contains duplicate or ineligible query")
                if row["sample_idx"] in excluded:
                    raise ValueError("exact-objective query uses an excluded image")
                if assigned_stage(fingerprint, row["sample_idx"], EXACT_OBJECTIVE_NAMESPACE) != stage:
                    raise ValueError("exact-objective query hash bucket mismatch")
                selected_ids.add(row["sample_idx"])
                stage_ids[stage].add(row["sample_idx"])
                all_selected.append(row)
        if stage_ids["stage-a"] & stage_ids["stage-b"]:
            raise ValueError("Stage A and Stage B base-image IDs overlap")
        maximum = max(row["timestep"] for row in all_selected)
        shortest = 100 * (1 + maximum // 100)
        if cell["max_eval_samples"] != shortest or shortest > len(stream):
            raise ValueError("exact-objective registry is not the shortest complete batch prefix")
        if cell["max_eval_samples"] > 6000:
            raise ValueError("exact-objective scan prefix exceeds 6000 rows")
    if stage_ids["stage-a"] & stage_ids["stage-b"]:
        raise ValueError("Stage A and Stage B base-image IDs overlap")


def _validate_exact_objective_exclusions(exclusions):
    historical = exclusions.get("historical_base_image_indices", [])
    smoke = exclusions.get("smoke_base_image_indices", [])
    combined = exclusions.get("base_image_indices", [])
    if (len(historical) != 1499 or len(set(historical)) != len(historical) or
            historical != sorted(historical) or digest(historical) !=
            exclusions.get("historical_base_image_indices_sha256")):
        raise ValueError("exact-objective registry lacks the frozen 1499 historical exclusions")
    if exclusions.get("historical_base_image_indices_sha256") != "c6ea2be7074c89ca0ab8d9a6479c2a0a6aa43e4726e2a6a485d527ab22ddb72c":
        raise ValueError("exact-objective historical exclusion hash differs from protocol")
    if (len(set(smoke)) != len(smoke) or smoke != sorted(smoke) or len(smoke) < 2 or len(smoke) > 4 or
            set(historical) & set(smoke) or digest(smoke) != exclusions.get("smoke_base_image_indices_sha256")):
        raise ValueError("exact-objective smoke exclusions are missing, duplicated or invalid")
    expected = sorted(set(historical) | set(smoke))
    if (combined != expected or len(set(combined)) != len(combined) or
            digest(combined) != exclusions.get("base_image_indices_sha256")):
        raise ValueError("combined exclusions must exactly match the historical and smoke IDs")


def exact_objective_query_schedule(registry, stage, cell):
    """Project registry rows to opaque identity keys consumed by the scorer."""
    validate_exact_objective_registry(registry)
    if stage not in EXACT_OBJECTIVE_STAGES or cell not in CELLS:
        raise ValueError("unknown exact-objective stage or OOD cell")
    return [{"timestep": row["timestep"], "sample_idx": row["sample_idx"]}
            for row in registry["cells"][cell]["selected"][stage]]
