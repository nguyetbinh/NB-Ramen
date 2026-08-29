"""Machine-readable temporal and stratified failure-mode summaries."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

REPORT_SCHEMA_VERSION = 1
_STATES = frozenset({"computed", "insufficient", "unavailable"})
_NUMERIC_METRICS = ("memory_occupancy", "routing_nmi", "cache_purity", "future_support_weight_fraction", "future_support_count", "consensus_mean", "consensus_p10", "consensus_p50", "fraction_low_consensus_coordinates", "active_support_classes", "pairwise_sign_agreement_mean", "pairwise_cosine_mean", "sign_disagreement", "retrieved_ood_fraction", "ood_ratio", "memory_oracle_gap", "retrieval_oracle_gap", "aggregation_oracle_gap")


def _state(status: str, **values: Any) -> dict[str, Any]:
    if status not in _STATES: raise ValueError("invalid analysis state")
    return {"status": status, **values}


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _outcome(row: Mapping[str, Any]) -> str | None:
    if isinstance(row.get("outcome"), str) and row["outcome"] in {"safe", "beneficial", "harmful", "unresolved"}: return row["outcome"]
    base, adapted = row.get("base_correct"), row.get("adapted_correct")
    if not isinstance(base, bool) or not isinstance(adapted, bool): return None
    return {(True, True): "safe", (False, True): "beneficial", (True, False): "harmful", (False, False): "unresolved"}[(base, adapted)]


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    """Read production diagnostics while retaining their report-only meaning."""
    value = row.get(metric)
    if metric == "active_support_classes" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return float(len(value))
    if metric == "sign_disagreement" and value is None:
        agreement = _finite(row.get("pairwise_sign_agreement_mean"))
        return None if agreement is None else 1.0 - agreement
    return _finite(value)


def annotate_time_since_shift(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return copies with deterministic time-since-domain-shift annotations."""
    values = [dict(row) for row in rows]
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in values: groups[(row.get("stream", row.get("stream_mode")), row.get("seed"), row.get("run_id"))].append(row)
    for group in groups.values():
        if any(not isinstance(row.get("timestep"), int) or isinstance(row.get("timestep"), bool) for row in group):
            continue
        last_domain, shift_time = object(), None
        for row in sorted(group, key=lambda item: item["timestep"]):
            domain = row.get("ground_truth_domain", row.get("domain"))
            if domain is None: continue
            if domain != last_domain: shift_time = row["timestep"]; last_domain = domain
            row["time_since_shift"] = row["timestep"] - shift_time
    return values


def _summary(rows: Sequence[Mapping[str, Any]], *, minimum_count: int) -> dict[str, Any]:
    if len(rows) < minimum_count: return _state("insufficient", count=len(rows), reason="too few observations")
    outcomes = [_outcome(row) for row in rows]
    if any(value is None for value in outcomes): return _state("unavailable", count=len(rows), reason="paired base/adapted outcomes absent")
    count = len(rows); counts = {name: outcomes.count(name) for name in ("safe", "beneficial", "harmful", "unresolved")}
    values: dict[str, Any] = {}
    for metric in _NUMERIC_METRICS:
        observed = [number for row in rows if (number := _metric_value(row, metric)) is not None]
        values[metric] = _state("computed", count=len(observed), mean=statistics.mean(observed)) if observed else _state("unavailable", count=0, reason="metric absent")
    base_error = (counts["beneficial"] + counts["unresolved"]) / count
    return _state("computed", count=count, counts=counts, base_error=base_error,
                  beneficial_rate=counts["beneficial"] / count, harmful_rate=counts["harmful"] / count, metrics=values)


def temporal_failure_report(rows: Iterable[Mapping[str, Any]], *, timestep_bin_size: int = 50, occupancy_bin_size: float = 1.0, minimum_count: int = 2) -> dict[str, Any]:
    if timestep_bin_size <= 0 or occupancy_bin_size <= 0 or minimum_count <= 0: raise ValueError("bin sizes and minimum_count must be positive")
    values = annotate_time_since_shift(rows)
    if not values: return _state("insufficient", count=0, reason="no paired rows")
    axes = {"timestep": lambda r: (r["timestep"] // timestep_bin_size) * timestep_bin_size if isinstance(r.get("timestep"), int) else None,
            "time_since_shift": lambda r: (r["time_since_shift"] // timestep_bin_size) * timestep_bin_size if isinstance(r.get("time_since_shift"), int) else None,
            "memory_occupancy": lambda r: math.floor(r["memory_occupancy"] / occupancy_bin_size) * occupancy_bin_size if _finite(r.get("memory_occupancy")) is not None else None,
            "domain": lambda r: r.get("ground_truth_domain", r.get("domain")), "seed": lambda r: r.get("seed"),
            "stream": lambda r: r.get("stream", r.get("stream_mode")), "batch_size": lambda r: r.get("batch_size"), "ood_ratio": lambda r: r.get("ood_ratio")}
    strata: dict[str, list[dict[str, Any]]] = {}
    for name, getter in axes.items():
        grouped: dict[str, tuple[Any, list[Mapping[str, Any]]]] = {}
        for row in values:
            key = getter(row)
            if key is None: continue
            encoded = json.dumps(key, sort_keys=True, default=str); grouped.setdefault(encoded, (key, []))[1].append(row)
        strata[name] = [{"value": key, **_summary(group, minimum_count=minimum_count)} for _, (key, group) in sorted(grouped.items())]
        if not strata[name]: strata[name] = [{"value": None, **_state("unavailable", count=0, reason="axis absent")}]
    panel = paired_panel_series(values, timestep_bin_size=timestep_bin_size, minimum_count=minimum_count)
    return {"schema_version": REPORT_SCHEMA_VERSION, "status": "computed", "count": len(values), "strata": strata, "paired_panels": panel}


def paired_panel_series(rows: Iterable[Mapping[str, Any]], *, timestep_bin_size: int = 50, minimum_count: int = 2) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("timestep"), int): grouped[(row["timestep"] // timestep_bin_size) * timestep_bin_size].append(row)
    if not grouped: return _state("unavailable", reason="timestep absent")
    series = []
    for start, group in sorted(grouped.items()):
        summary = _summary(group, minimum_count=minimum_count)
        task = {name: summary.get(name) for name in ("base_error", "beneficial_rate", "harmful_rate")}
        metrics = summary.get("metrics", {})
        # Oracle gaps explain outcome failures, so expose them in the task
        # panel as well as retaining the complete metric dictionary below.
        task.update({name: metrics.get(name, _state("unavailable", reason="metric absent"))
                     for name in ("memory_oracle_gap", "retrieval_oracle_gap", "aggregation_oracle_gap")})
        mechanism = {name: metrics.get(name, _state("unavailable", reason="metric absent")) for name in _NUMERIC_METRICS}
        series.append({"timestep_start": start, "timestep_end": start + timestep_bin_size - 1, "count": len(group), "status": summary["status"], "task_failure": task, "mechanism": mechanism})
    return _state("computed", bin_size=timestep_bin_size, series=series)


summarize_temporal_failures = temporal_failure_report
build_paired_panel_series = paired_panel_series


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temporal paired-panel failure analysis")
    parser.add_argument("--input", required=True); parser.add_argument("--output"); parser.add_argument("--timestep-bin-size", type=int, default=50); parser.add_argument("--minimum-count", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8")); rows = payload["rows"] if isinstance(payload, Mapping) else payload
        report = temporal_failure_report(rows, timestep_bin_size=args.timestep_bin_size, minimum_count=args.minimum_count)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc: report = {"schema_version": REPORT_SCHEMA_VERSION, "status": "invalid", "error": str(exc)}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output: Path(args.output).write_text(encoded, encoding="utf-8")
    else: sys.stdout.write(encoded)
    return 0 if report["status"] != "invalid" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
