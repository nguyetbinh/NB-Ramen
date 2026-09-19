"""Outcome-independent query identity, exclusions and immutable pilot registries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


STAGES = {"stage-a": 16, "stage-b": 128}
CELLS = ("0", "0.5")


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
    if namespace not in ("qcgs-label-free-v1", "qcgs-multiview-rescue-v1"):
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
