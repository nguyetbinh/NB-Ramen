"""Export evaluator-only probe inputs from a verified ``replay_v1`` run.

The exporter deliberately operates after evaluation has finished.  It binds
each decoded query feature to the immutable trace and never exposes labels to
the evaluated method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - supports direct module execution
    from .domain_probe_analysis import representation_probe_report
    from .failure_analysis_artifacts import ReplayArtifactError, ReplaySidecarReader, sha256_file
    from .failure_mode_analysis import _verified_trace_run
except ImportError:  # pragma: no cover
    from domain_probe_analysis import representation_probe_report
    from failure_analysis_artifacts import ReplayArtifactError, ReplaySidecarReader, sha256_file
    from failure_mode_analysis import _verified_trace_run


EXPORT_SCHEMA_VERSION = 1
_ROLES = frozenset({"analysis", "final"})


def _split_role(*, identity: tuple[Any, ...], seed: int, provenance: Mapping[str, str]) -> str:
    """Assign a row by identity and immutable provenance, never by a label."""
    material = json.dumps({"identity": identity, "seed": seed, "provenance": provenance},
                          sort_keys=True, separators=(", ", ":"), default=str)
    bucket = hashlib.sha256(material.encode("utf-8")).digest()[0]
    return "train" if bucket < 153 else "validation" if bucket < 204 else "test"


def _analysis_role(manifest: Mapping[str, Any]) -> str:
    args = manifest.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("manifest args are malformed")
    role = args.get("analysis_role")
    if role not in _ROLES:
        raise ValueError("manifest args.analysis_role must be 'analysis' or 'final'")
    if role != "analysis":
        raise ValueError("verified feature export rejects final-evaluation runs")
    return role


def _model_fingerprint(manifest: Mapping[str, Any]) -> str:
    artifacts = manifest.get("artifacts")
    model = artifacts.get("model") if isinstance(artifacts, Mapping) else None
    value = model.get("actual_sha256", model.get("expected_sha256")) if isinstance(model, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("manifest model artifact fingerprint is absent")
    return value


def export_verified_replay_features(run_dir: str | Path, *, seed: int = 0) -> dict[str, Any]:
    """Decode every verified replay query feature into a domain-probe payload.

    Full trace coverage is required, so a bounded/partial sidecar cannot be
    used as a convenient subset of the evaluation stream.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("split seed must be an integer")
    root, manifest, trace, identity = _verified_trace_run(str(run_dir))
    role = _analysis_role(manifest)
    args = manifest.get("args", {})
    if args.get("failure_analysis_profile") != "replay_v1":
        raise ValueError("manifest is not a replay_v1 run")
    source = identity["source_fingerprint"]
    if not isinstance(source, str) or not source:
        raise ValueError("manifest source fingerprint is absent")
    model = _model_fingerprint(manifest)
    try:
        reader = ReplaySidecarReader(root / "failure-analysis", manifest_sha256=sha256_file(root / "manifest.json"),
                                     stream_fingerprint=identity["stream_fingerprint"], source_fingerprint=source,
                                     run_id=identity["run_id"])
    except ReplayArtifactError as exc:
        raise ValueError(f"invalid replay sidecar: {exc}") from exc
    if reader.metadata.get("status") != "completed":  # reader currently also enforces this; retain the contract here.
        raise ValueError("replay sidecar must be completed")

    trace_by_identity: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for trace_row in trace:
        key = (trace_row["timestep"], trace_row["sample_idx"], trace_row["ground_truth_domain"])
        if key in trace_by_identity:
            raise ValueError("trace evaluator identities are ambiguous")
        trace_by_identity[key] = trace_row
    items: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    for item in reader.rows("items"):
        key = (item.get("segment_index", 0), item.get("item_id"))
        if item.get("item_id") is None or key in items:
            raise ValueError("sidecar item identities are incomplete or ambiguous")
        if "feature" in item:
            items[key] = item

    provenance = {"manifest_sha256": identity["manifest_sha256"], "stream_fingerprint": identity["stream_fingerprint"],
                  "source_fingerprint": source, "model_fingerprint": model}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    feature_dtype: str | None = None
    feature_dim: int | None = None
    for query in reader.rows("queries"):
        sample = query.get("evaluator_sample_identity")
        if not isinstance(sample, Mapping):
            raise ValueError("sidecar query lacks evaluator sample identity")
        query_identity = (query.get("producer_query_timestep"), sample.get("sample_idx"),
                          sample.get("ground_truth_domain"))
        trace_row = trace_by_identity.get(query_identity)
        if trace_row is None or query.get("ground_truth_class") != trace_row["ground_truth_class"]:
            raise ValueError("sidecar query labels do not exactly match evaluator trace")
        trace_sample = trace_row.get("evaluator_sample_identity")
        trace_query_identity = (trace_row.get("producer_query_timestep"),
                                trace_sample.get("sample_idx") if isinstance(trace_sample, Mapping) else None,
                                trace_sample.get("ground_truth_domain") if isinstance(trace_sample, Mapping) else None)
        if trace_query_identity != query_identity:
            raise ValueError("sidecar query identity does not exactly match evaluator trace")
        segment = query.get("segment_index", 0)
        if (query.get("item_id") != trace_row.get("query_item_id")
                or segment != trace_row.get("segment_index")):
            raise ValueError("sidecar query item does not exactly match evaluator trace")
        item = items.get((segment, query.get("item_id")))
        if item is None:
            raise ValueError("sidecar query has no feature-bearing item")
        item_sample = item.get("evaluator_sample_identity")
        item_identity = (item.get("producer_query_timestep"),
                         item_sample.get("sample_idx") if isinstance(item_sample, Mapping) else None,
                         item_sample.get("ground_truth_domain") if isinstance(item_sample, Mapping) else None)
        if item_identity != query_identity or item.get("segment_index", 0) != segment:
            raise ValueError("feature-bearing item does not exactly match sidecar query")
        tensor = reader.tensor(item["feature"])
        if tensor.numel() <= 0:
            raise ValueError("query feature must be non-empty")
        tensor = tensor.detach().cpu().reshape(-1)
        values = tensor.tolist()
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("query feature values must be finite")
        dtype, dimension = str(tensor.dtype).removeprefix("torch."), int(tensor.numel())
        if feature_dtype is None:
            feature_dtype, feature_dim = dtype, dimension
        elif (dtype, dimension) != (feature_dtype, feature_dim):
            raise ValueError("replay query features must have one dtype and dimension")
        row_identity = (trace_row["timestep"], trace_row["sample_idx"], trace_row["ground_truth_domain"], trace_row["ground_truth_class"])
        if row_identity in seen:
            raise ValueError("sidecar queries duplicate evaluator identities")
        seen.add(row_identity)
        rows.append({"timestep": trace_row["timestep"], "sample_idx": trace_row["sample_idx"],
                     "ground_truth_domain": trace_row["ground_truth_domain"],
                     "ground_truth_class": trace_row["ground_truth_class"], "feature": values,
                     "split_role": _split_role(identity=(trace_row["sample_idx"],), seed=seed, provenance=provenance)})
    expected = {(row["timestep"], row["sample_idx"], row["ground_truth_domain"], row["ground_truth_class"]) for row in trace}
    if seen != expected:
        raise ValueError("replay sidecar does not provide full evaluator-trace coverage")
    if feature_dtype is None or feature_dim is None:
        raise ValueError("replay sidecar has no query features")
    metadata = {**provenance, "model_artifact_fingerprint": model, "feature_dtype": feature_dtype,
                "feature_dim": feature_dim, "split_role": "per_row", "analysis_role": role,
                "split_assignment": "sha256(provenance, seed, immutable_sample_identity)"}
    return {"schema_version": EXPORT_SCHEMA_VERSION, "status": "computed", "rows": rows, "metadata": metadata}


def run_verified_replay_probes(run_dir: str | Path, *, seed: int = 0) -> dict[str, Any]:
    payload = export_verified_replay_features(run_dir, seed=seed)
    return representation_probe_report(payload["rows"], payload["metadata"], seed=seed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export verified replay_v1 features for evaluator-only domain probes")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-probes", action="store_true", help="emit the probe report instead of the probe input")
    args = parser.parse_args(argv)
    try:
        result = (run_verified_replay_probes if args.run_probes else export_verified_replay_features)(args.run_dir, seed=args.seed)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = {"schema_version": EXPORT_SCHEMA_VERSION, "status": "invalid", "error": str(exc)}
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0 if result["status"] != "invalid" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
