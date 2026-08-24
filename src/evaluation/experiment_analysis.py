"""Post-hoc, dependency-free analysis of strictly validated experiment evidence.

This module deliberately consumes completed evidence only.  It never writes
run artifacts and does not make labels available to adaptation or routing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

try:
    from ..runtime.experiment_matrix import (
        ExperimentRun, IncompleteRunError, SUPPORTED_ARTIFACT_PROVENANCE,
        SUPPORTED_DEVICES, build_experiment_matrix, validate_completed_run,
    )
    from .routing_metrics import normalized_mutual_information
except ImportError:  # pragma: no cover - direct-file support
    source_root = str(Path(__file__).resolve().parents[1])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from runtime.experiment_matrix import (
        ExperimentRun, IncompleteRunError, SUPPORTED_ARTIFACT_PROVENANCE,
        SUPPORTED_DEVICES, build_experiment_matrix, validate_completed_run,
    )
    from evaluation.routing_metrics import normalized_mutual_information


REPORT_SCHEMA_VERSION = 1
REQUIRED_THRESHOLD_FIELDS = (
    "minimum_repeats", "max_accuracy_std", "structured_degradation_min",
    "oracle_recovery_min", "router_closure_min", "natural_domain_gain_min",
    "max_memory_ratio", "max_forward_latency_ratio", "min_routing_accuracy_association",
    "max_class_context_nmi",
)
NATURAL_DATASETS = frozenset({"domainnet", "officehome", "pacs", "vlcs", "terraincognita"})


@dataclass(frozen=True)
class AnalysisThresholds:
    minimum_repeats: int
    max_accuracy_std: float
    structured_degradation_min: float
    oracle_recovery_min: float
    router_closure_min: float
    natural_domain_gain_min: float
    max_memory_ratio: float
    max_forward_latency_ratio: float
    min_routing_accuracy_association: float
    max_class_context_nmi: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AnalysisThresholds":
        missing = [key for key in REQUIRED_THRESHOLD_FIELDS if key not in value]
        if missing:
            raise ValueError("thresholds missing required field(s): " + ", ".join(missing))
        unknown = sorted(set(value).difference(REQUIRED_THRESHOLD_FIELDS))
        if unknown:
            raise ValueError("thresholds contain unknown field(s): " + ", ".join(unknown))
        parsed: dict[str, object] = {}
        for key in REQUIRED_THRESHOLD_FIELDS:
            number = value[key]
            if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number):
                raise ValueError(f"threshold {key} must be a finite number")
            parsed[key] = number
        if int(parsed["minimum_repeats"]) != parsed["minimum_repeats"] or parsed["minimum_repeats"] < 1:
            raise ValueError("threshold minimum_repeats must be a positive integer")
        for key in ("max_accuracy_std", "max_memory_ratio", "max_forward_latency_ratio", "max_class_context_nmi"):
            if parsed[key] < 0:
                raise ValueError(f"threshold {key} must be non-negative")
        return cls(**parsed)  # type: ignore[arg-type]


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _summary_value(summary: Mapping[str, object], key: str, field: str) -> float | None:
    block = summary.get(key)
    return _number(block.get(field)) if isinstance(block, Mapping) and block.get("status") == "computed" else None


def _recovery_value(summary: Mapping[str, object]) -> float | None:
    block = summary.get("post_shift_recovery_time")
    if not isinstance(block, Mapping) or block.get("status") != "computed":
        return None
    values = [item.get("recovery_samples") for item in block.get("shifts", []) if isinstance(item, Mapping) and item.get("status") == "recovered"]
    numeric = [_number(value) for value in values]
    return statistics.mean(value for value in numeric if value is not None) if numeric and all(value is not None for value in numeric) else None


def _aggregate(values: Iterable[float | None]) -> dict[str, object]:
    observed = [value for value in values if value is not None]
    if not observed:
        return {"count": 0, "mean": None, "std": None, "ci95": None}
    std = statistics.stdev(observed) if len(observed) > 1 else 0.0
    return {"count": len(observed), "mean": statistics.mean(observed), "std": std, "ci95": 1.96 * std / math.sqrt(len(observed))}


def _read_trace(run: ExperimentRun) -> list[dict[str, object]]:
    # ``validate_completed_run`` already checks the schema and evidence linkage.
    with (run.run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _class_recovery(run: ExperimentRun) -> dict[str, object]:
    rows = _read_trace(run)
    # Bind the parsed trace to the strict evidence check.  This catches a
    # replacement occurring between the analysis-level read and the report.
    validate_completed_run(run)
    contexts = [row["inferred_context"] for row in rows]
    if any(context is None for context in contexts):
        return {"status": "unavailable", "nmi_inferred_context_vs_ground_truth_class": None}
    return {"status": "computed", "nmi_inferred_context_vs_ground_truth_class": normalized_mutual_information(
        [row["ground_truth_class"] for row in rows], contexts,
    )}


def load_completed_runs(runs: Iterable[ExperimentRun]) -> list[tuple[ExperimentRun, dict[str, object]]]:
    """Validate every requested run before exposing it to analysis."""
    return [(run, validate_completed_run(run)) for run in runs]


def analyse_completed_runs(
    completed: Iterable[tuple[ExperimentRun, Mapping[str, object]]], thresholds: AnalysisThresholds,
) -> dict[str, object]:
    """Build a deterministic report and evaluate only supplied numeric gates."""
    grouped: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for run, evidence in completed:
        summary = evidence["summary"]
        if not isinstance(summary, Mapping):
            raise ValueError(f"validated run has no summary: {run.run_id}")
        routing = summary.get("routing_diagnostics")
        class_recovery = _class_recovery(run)
        device_peak, device_peak_source = _device_memory_peak(summary)
        metrics = {
            "accuracy": _number(summary.get("micro_accuracy")),
            "worst_domain_accuracy": _number(summary.get("worst_domain_accuracy")),
            "recovery_samples": _recovery_value(summary),
            "negative_adaptation_rate": _summary_value(summary, "negative_adaptation_rate", "value"),
            "forward_total_ms": _summary_value(summary, "forward_latency", "total_ms"),
            "throughput": _summary_value(summary, "throughput", "samples_per_second"),
            "method_memory_bytes": _summary_value(summary, "method_memory", "max_retained_bytes"),
            "device_memory_peak_bytes": device_peak,
            "routing_nmi": _number(routing.get("normalized_mutual_information")) if isinstance(routing, Mapping) else None,
            "routing_ari": _number(routing.get("adjusted_rand_index")) if isinstance(routing, Mapping) else None,
            "routing_context_purity": _number(routing.get("context_purity")) if isinstance(routing, Mapping) else None,
            "routing_context_count": _number(routing.get("number_of_discovered_contexts")) if isinstance(routing, Mapping) else None,
            "routing_assignment_churn_rate": _number(routing.get("assignment_churn_rate")) if isinstance(routing, Mapping) else None,
            "class_context_nmi": _number(class_recovery.get("nmi_inferred_context_vs_ground_truth_class")),
        }
        grouped[(run.dataset, run.stream_mode, run.seed, run.method)].append({
            "run_id": run.run_id, "metrics": metrics, "device_memory_peak_source": device_peak_source,
            "class_recovery": class_recovery,
        })

    groups = []
    condition_records: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    seed_records: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for key in sorted(grouped):
        rows = grouped[key]
        metrics = {name: _aggregate(row["metrics"][name] for row in rows) for name in rows[0]["metrics"]}
        record = {"dataset": key[0], "stream_mode": key[1], "seed": key[2], "method": key[3], "runs": rows, "metrics": metrics}
        groups.append(record)
        seed_records[key] = record
        condition_records[(key[0], key[1], key[3])].append(record)

    aggregates = []
    for (dataset, stream, method), rows in sorted(condition_records.items()):
        aggregates.append({"dataset": dataset, "stream_mode": stream, "method": method, "metrics": {
            name: _aggregate(row["metrics"][name]["mean"] for row in rows) for name in rows[0]["metrics"]
        }})

    comparisons = []
    association_points: list[tuple[float, float]] = []
    comparison_methods = ("NoAdapt", "Ramen", "OracleLatentRamen", "LatentRamen")
    expected_cells = sorted({key[:3] for key in seed_records if key[3] in comparison_methods})
    for dataset, stream, seed in expected_cells:
        latent = seed_records.get((dataset, stream, seed, "LatentRamen"))
        controls = {name: seed_records.get((dataset, stream, seed, name)) for name in ("Ramen", "NoAdapt", "OracleLatentRamen")}
        latent_accuracy = _record_metric(latent, "accuracy")
        values = {name: None if control is None else control["metrics"]["accuracy"]["mean"] for name, control in controls.items()}
        oracle_gap = None if None in (latent_accuracy, values["Ramen"], values["OracleLatentRamen"]) else values["OracleLatentRamen"] - values["Ramen"]
        closure = None if oracle_gap is None or oracle_gap <= 0 else (latent_accuracy - values["Ramen"]) / oracle_gap
        memory_ratio = _device_memory_ratio(latent, controls["Ramen"])
        latency_ratio = _ratio(_record_metric(latent, "forward_total_ms"), controls["Ramen"], "forward_total_ms")
        routing_nmi = _record_metric(latent, "routing_nmi")
        if routing_nmi is not None and values["Ramen"] is not None and latent_accuracy is not None:
            association_points.append((routing_nmi, latent_accuracy - values["Ramen"]))
        comparisons.append({
            "dataset": dataset, "stream_mode": stream, "seed": seed,
            "routing_nmi": routing_nmi,
            "class_context_nmi": _record_metric(latent, "class_context_nmi"),
            "latent_vs_ramen_accuracy_gain": _difference(latent_accuracy, values["Ramen"]),
            "latent_vs_noadapt_accuracy_gain": _difference(latent_accuracy, values["NoAdapt"]),
            "latent_vs_oracle_accuracy_gap": _difference(values["OracleLatentRamen"], latent_accuracy),
            "oracle_recovery_accuracy_gain": _difference(values["OracleLatentRamen"], values["Ramen"]),
            "oracle_gap_closure": closure,
            "equal_memory_control": _match_status(memory_ratio, "device-memory peak evidence"),
            "logical_method_memory": {
                "status": "diagnostic_only",
                "latent_max_retained_bytes": _record_metric(latent, "method_memory_bytes"),
                "ramen_max_retained_bytes": None if controls["Ramen"] is None else controls["Ramen"]["metrics"]["method_memory_bytes"]["mean"],
                "reason": "logical support-memory bytes are not available for every legacy method and are never mixed with device-memory evidence",
            },
            "equal_total_forward_latency_control": _match_status(latency_ratio, "forward_latency.total_ms"),
            "retrieval_latency": {"status": "unavailable", "reason": "summary evidence intentionally does not isolate retrieval latency"},
        })

    gate = evaluate_go_no_go(groups, aggregates, comparisons, association_points, thresholds)
    return {"schema_version": REPORT_SCHEMA_VERSION, "thresholds": asdict(thresholds), "groups": groups, "aggregates": aggregates, "comparisons": comparisons, "go_no_go": gate}


def _difference(left: object, right: object) -> float | None:
    return left - right if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None


def _record_metric(record: Mapping[str, object] | None, metric: str) -> float | None:
    return None if record is None else record["metrics"][metric]["mean"]


def _ratio(numerator: object, control: object, metric: str) -> float | None:
    denominator = None if control is None else control["metrics"][metric]["mean"]
    return numerator / denominator if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)) and denominator > 0 else None


def _device_memory_peak(summary: Mapping[str, object]) -> tuple[float | None, str | None]:
    peak = _number(summary.get("peak_device_memory_bytes"))
    if peak is not None:
        return peak, "peak_device_memory_bytes"
    device = summary.get("device_memory")
    if isinstance(device, Mapping):
        value = _number(device.get("bytes"))
        kind = device.get("kind")
        if value is not None and isinstance(kind, str):
            return value, f"device_memory.{kind}"
    return None, None


def _device_memory_ratio(latent: Mapping[str, object] | None, control: Mapping[str, object] | None) -> float | None:
    if latent is None or control is None:
        return None
    latent_sources = {run["device_memory_peak_source"] for run in latent["runs"]}
    control_sources = {run["device_memory_peak_source"] for run in control["runs"]}
    if len(latent_sources) != 1 or latent_sources != control_sources or None in latent_sources:
        return None
    return _ratio(latent["metrics"]["device_memory_peak_bytes"]["mean"], control, "device_memory_peak_bytes")


def _match_status(ratio: float | None, source: str) -> dict[str, object]:
    return {"status": "available" if ratio is not None else "unavailable", "ratio": ratio, "source": source}


def _pearson(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs, ys = zip(*points)
    dx = [value - statistics.mean(xs) for value in xs]
    dy = [value - statistics.mean(ys) for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denominator if denominator else None


def evaluate_go_no_go(groups: Sequence[Mapping[str, object]], aggregates: Sequence[Mapping[str, object]], comparisons: Sequence[Mapping[str, object]], association_points: Sequence[tuple[float, float]], thresholds: AnalysisThresholds) -> dict[str, object]:
    """Apply all decision criteria; absent numeric evidence is never a pass."""
    group_by_key = {(item["dataset"], item["stream_mode"], item["seed"], item["method"]): item for item in groups}
    repeat = list(aggregates)
    repeat_value: float | None
    if not repeat or any(item["metrics"]["accuracy"]["count"] < thresholds.minimum_repeats for item in repeat):
        repeat_value = None
    else:
        repeat_value = max(item["metrics"]["accuracy"]["std"] for item in repeat)
    structured_comparisons = [item for item in comparisons if item["stream_mode"] != "iid_mixed"]
    structured = []
    for item in structured_comparisons:
        ramen = group_by_key.get((item["dataset"], item["stream_mode"], item["seed"], "Ramen"))
        iid = group_by_key.get((item["dataset"], "iid_mixed", item["seed"], "Ramen"))
        structured.append(_difference(_record_metric(iid, "accuracy"), _record_metric(ramen, "accuracy")))
    oracle_recoveries = [item["oracle_recovery_accuracy_gain"] for item in structured_comparisons]
    closures = [item["oracle_gap_closure"] for item in structured_comparisons]
    natural_comparisons = [item for item in structured_comparisons if item["dataset"].lower() in NATURAL_DATASETS]
    natural = [item["latent_vs_ramen_accuracy_gain"] for item in natural_comparisons]
    memories = [item["equal_memory_control"]["ratio"] for item in comparisons]
    latencies = [item["equal_total_forward_latency_control"]["ratio"] for item in comparisons]
    association = _pearson(association_points) if len(association_points) == len(comparisons) else None
    class_context = [item["class_context_nmi"] for item in comparisons]
    paired_complete = bool(comparisons) and all(
        item[metric] is not None for item in comparisons for metric in (
            "latent_vs_ramen_accuracy_gain", "latent_vs_noadapt_accuracy_gain",
            "latent_vs_oracle_accuracy_gap",
        )
    )
    criteria = {
        "paired_comparison_completeness": _criterion(1.0 if paired_complete else None, 1.0, "minimum"),
        "repeated_run_tolerance": _criterion(repeat_value, thresholds.max_accuracy_std, "maximum"),
        "structured_degradation": _criterion(_complete_max(structured), thresholds.structured_degradation_min, "minimum"),
        "oracle_recovery": _criterion(_complete_mean(oracle_recoveries), thresholds.oracle_recovery_min, "minimum"),
        "router_closure": _criterion(_complete_mean(closures), thresholds.router_closure_min, "minimum"),
        "natural_domain_gain": _criterion(_complete_mean(natural), thresholds.natural_domain_gain_min, "minimum"),
        "memory_ratio": _criterion(_complete_max(memories), thresholds.max_memory_ratio, "maximum"),
        "forward_latency_ratio": _criterion(_complete_max(latencies), thresholds.max_forward_latency_ratio, "maximum"),
        "routing_accuracy_association": _criterion(association, thresholds.min_routing_accuracy_association, "minimum"),
        "class_context_nmi": _criterion(_complete_max(class_context), thresholds.max_class_context_nmi, "maximum"),
    }
    missing = [name for name, value in criteria.items() if value["status"] == "insufficient_evidence"]
    status = "insufficient_evidence" if missing else ("go" if all(value["passed"] for value in criteria.values()) else "no_go")
    return {"status": status, "criteria": criteria, "insufficient": missing}


def _complete_mean(values: Sequence[float | None]) -> float | None:
    return statistics.mean(values) if values and all(value is not None for value in values) else None


def _complete_max(values: Sequence[float | None]) -> float | None:
    return max(values) if values and all(value is not None for value in values) else None


def _criterion(value: object, threshold: float, direction: str) -> dict[str, object]:
    if value is None:
        return {"status": "insufficient_evidence", "value": None, "threshold": threshold, "passed": False}
    if direction == "maximum":
        passed = value <= threshold
    else:
        passed = value >= threshold
    return {"status": "evaluated", "value": value, "threshold": threshold, "passed": passed}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", required=True, type=Path, help="JSON file containing every numeric gate.")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--config-dir", default=Path(__file__).resolve().parents[2] / "cfg")
    parser.add_argument("--dataset", required=True, action="append")
    parser.add_argument("--stream", required=True, action="append")
    parser.add_argument("--seed", required=True, type=int, action="append")
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--stream-block-size", type=int, default=64)
    parser.add_argument("--artifact-provenance", choices=SUPPORTED_ARTIFACT_PROVENANCE, default="fast")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        thresholds = AnalysisThresholds.from_mapping(json.loads(args.thresholds.read_text(encoding="utf-8")))
        runs = build_experiment_matrix(datasets=args.dataset, streams=args.stream, seeds=args.seed,
            methods=("NoAdapt", "Ramen", "OracleLatentRamen", "LatentRamen"),
            evidence_dir=args.evidence_dir, config_dir=args.config_dir, device=args.device,
            max_eval_samples=args.max_eval_samples, artifact_provenance=args.artifact_provenance,
            data_root=args.data_root, stream_block_size=args.stream_block_size)
        report = analyse_completed_runs(load_completed_runs(runs), thresholds)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, IncompleteRunError) as exc:
        print(json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "status": "invalid_evidence", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["go_no_go"]["status"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
