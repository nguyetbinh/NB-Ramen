"""Fail-closed evidence analysis for the StructuredAtomicRamen causal ablation."""

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
        BATCH_SIZE_BY_DATASET, ExperimentRun, IncompleteRunError,
        SUPPORTED_ARTIFACT_PROVENANCE, SUPPORTED_DEVICES, build_experiment_matrix,
        validate_completed_run,
    )
except ImportError:  # pragma: no cover - direct-file invocation
    source_root = str(Path(__file__).resolve().parents[1])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from runtime.experiment_matrix import (
        BATCH_SIZE_BY_DATASET, ExperimentRun, IncompleteRunError,
        SUPPORTED_ARTIFACT_PROVENANCE, SUPPORTED_DEVICES, build_experiment_matrix,
        validate_completed_run,
    )


REPORT_SCHEMA_VERSION = 1
ATTRIBUTION_METHODS = ("StructuredAtomicRamen", "CausalRamen")
REQUIRED_METHODS = ("Ramen", *ATTRIBUTION_METHODS)
NATURAL_DATASETS = frozenset({"domainnet", "officehome", "pacs", "vlcs", "terraincognita"})
IDENTITY_FIELDS = (
    "dataset", "stream_mode", "seed", "model", "batch_size", "device",
    "max_eval_samples", "stream_block_size", "metric_window_size", "metric_window_stride",
    "artifact_provenance", "data_root",
)
METHOD_IDENTITY_CONFIG_KEYS = frozenset({"method", "tta_algo", "algorithm", "name"})
ABSENT_EVIDENCE_PREFIXES = (
    "missing evidence directory:", "missing manifest:", "missing summary:",
    "missing stream export:", "missing trace:",
)


class MatrixEvidenceError(RuntimeError):
    """Aggregated strict-validation failures for one requested matrix."""

    def __init__(self, status: str, failures: Sequence[Mapping[str, str]]):
        self.status = status
        self.failures = [dict(item) for item in failures]
        super().__init__(f"{len(self.failures)} requested run(s) failed strict validation")


def _is_absent_evidence_error(error: BaseException, run: object | None = None) -> bool:
    """Classify only genuinely absent required artifacts as insufficient."""
    if not isinstance(error, IncompleteRunError):
        return False
    if str(error).startswith(ABSENT_EVIDENCE_PREFIXES):
        return True
    run_dir = getattr(run, "run_dir", None)
    if isinstance(run_dir, Path):
        if not run_dir.is_dir():
            return True
        required = ("manifest.json", "summary.json", "stream.json", "trace.jsonl")
        return any(not (run_dir / name).is_file() for name in required)
    return False


@dataclass(frozen=True)
class CausalCompletionThresholds:
    minimum_fixed_seeds: int
    minimum_non_iid_streams: int
    minimum_mean_micro_gain_for_go: float
    minimum_mean_micro_gain_for_weak_go: float
    minimum_mean_negative_adaptation_reduction_for_weak_go: float
    minimum_mean_recovery_reduction_for_weak_go: float
    minimum_mean_worst_domain_gain_for_weak_go: float
    maximum_micro_std: float
    require_full_cifar100c: bool
    require_natural_domain_dataset: bool
    allow_weak_go: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CausalCompletionThresholds":
        expected = set(cls.__annotations__)
        missing, unknown = expected.difference(value), set(value).difference(expected)
        if missing or unknown:
            pieces = []
            if missing:
                pieces.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                pieces.append("unknown " + ", ".join(sorted(unknown)))
            raise ValueError("causal thresholds " + "; ".join(pieces))
        integers = ("minimum_fixed_seeds", "minimum_non_iid_streams")
        parsed: dict[str, object] = {}
        for key in integers:
            number = value[key]
            if not isinstance(number, int) or isinstance(number, bool) or number < 1:
                raise ValueError(f"threshold {key} must be a positive integer")
            parsed[key] = number
        numeric = (
            "minimum_mean_micro_gain_for_go", "minimum_mean_micro_gain_for_weak_go",
            "minimum_mean_negative_adaptation_reduction_for_weak_go",
            "minimum_mean_recovery_reduction_for_weak_go",
            "minimum_mean_worst_domain_gain_for_weak_go", "maximum_micro_std",
        )
        for key in numeric:
            number = value[key]
            if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number):
                raise ValueError(f"threshold {key} must be finite")
            if key in {
                "minimum_mean_negative_adaptation_reduction_for_weak_go",
                "minimum_mean_recovery_reduction_for_weak_go",
                "minimum_mean_worst_domain_gain_for_weak_go", "maximum_micro_std",
            } and number < 0:
                raise ValueError(f"threshold {key} must be non-negative")
            parsed[key] = float(number)
        if parsed["minimum_mean_micro_gain_for_weak_go"] > parsed["minimum_mean_micro_gain_for_go"]:
            raise ValueError("weak-go micro threshold cannot exceed GO threshold")
        for key in ("require_full_cifar100c", "require_natural_domain_dataset", "allow_weak_go"):
            if not isinstance(value[key], bool):
                raise ValueError(f"threshold {key} must be boolean")
            parsed[key] = value[key]
        return cls(**parsed)  # type: ignore[arg-type]


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _optional_number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _metric(summary: Mapping[str, object], name: str) -> float:
    return _number(summary.get(name), f"summary.{name}")


def _nested_metric(summary: Mapping[str, object], name: str, field: str) -> float | None:
    block = summary.get(name)
    if not isinstance(block, Mapping):
        return None
    if block.get("status") not in {"computed", "collected", "sampled"}:
        return None
    return _optional_number(block.get(field))


def _recovery(summary: Mapping[str, object]) -> float | None:
    block = summary.get("post_shift_recovery_time")
    if not isinstance(block, Mapping) or block.get("status") != "computed":
        return None
    values = []
    for shift in block.get("shifts", []):
        if isinstance(shift, Mapping) and shift.get("status") == "recovered":
            number = _optional_number(shift.get("recovery_samples"))
            if number is None:
                return None
            values.append(number)
    return statistics.mean(values) if values else None


def _config_without_method(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    return {str(key): item for key, item in value.items() if str(key).lower() not in METHOD_IDENTITY_CONFIG_KEYS}


def _config_equivalence(atomic: object, causal: object) -> tuple[bool, list[str]]:
    mismatches = []
    for field in IDENTITY_FIELDS:
        if hasattr(atomic, field) and hasattr(causal, field) and getattr(atomic, field) != getattr(causal, field):
            mismatches.append(field)
    if hasattr(atomic, "config_data") and hasattr(causal, "config_data"):
        if _config_without_method(getattr(atomic, "config_data")) != _config_without_method(getattr(causal, "config_data")):
            mismatches.append("config_data_except_method_identity")
    return not mismatches, mismatches


def load_completed_runs(runs: Iterable[ExperimentRun]) -> list[tuple[ExperimentRun, dict[str, object]]]:
    """Validate the entire requested matrix, with malformed evidence taking precedence."""
    completed = []
    failures = []
    for run in runs:
        try:
            completed.append((run, validate_completed_run(run)))
        except (IncompleteRunError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append({
                "run_id": getattr(run, "run_id", "<unknown>"),
                "classification": "missing" if _is_absent_evidence_error(exc, run) else "invalid",
                "error": str(exc),
            })
    if failures:
        status = "INVALID" if any(item["classification"] == "invalid" for item in failures) else "INSUFFICIENT"
        raise MatrixEvidenceError(status, failures)
    return completed


def _run_record(run: object, evidence: Mapping[str, object]) -> dict[str, object]:
    summary = evidence.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"validated run has no summary: {getattr(run, 'run_id', '<unknown>')}")
    return {
        "run": run,
        "run_id": getattr(run, "run_id", None),
        "metrics": {
            "micro_accuracy": _metric(summary, "micro_accuracy"),
            "macro_domain_accuracy": _metric(summary, "macro_domain_accuracy"),
            "worst_domain_accuracy": _metric(summary, "worst_domain_accuracy"),
            "negative_adaptation_rate": _nested_metric(summary, "negative_adaptation_rate", "value"),
            "recovery_samples": _recovery(summary),
            "method_memory_bytes": _nested_metric(summary, "method_memory", "max_retained_bytes"),
            "forward_total_ms": _nested_metric(summary, "forward_latency", "total_ms"),
            "retrieval_total_ms": _nested_metric(summary, "retrieval_latency", "total_ms"),
            "device_memory_bytes": _optional_number(summary.get("peak_device_memory_bytes")),
        },
    }


def _cell_key(run: object) -> tuple[str, str, int, int]:
    try:
        return (str(run.dataset), str(run.stream_mode), int(run.seed), int(run.batch_size))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("run lacks a deterministic dataset/stream/seed/batch_size identity") from exc


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _aggregate(values: Sequence[float]) -> dict[str, object]:
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "mean": statistics.mean(values), "std": std}


def _available_aggregate(values: Sequence[float | None]) -> dict[str, object]:
    observed = [value for value in values if value is not None]
    if not observed:
        return {"status": "unavailable", "count": 0, "mean": None, "std": None}
    return {"status": "computed", **_aggregate(observed)}


def _metric_aggregates(comparisons: Sequence[Mapping[str, object]], delta_key: str) -> dict[str, dict[str, object]]:
    metric_names = (
        "micro_accuracy", "macro_domain_accuracy", "worst_domain_accuracy",
        "negative_adaptation_rate", "recovery_samples", "method_memory_bytes",
        "forward_total_ms", "retrieval_total_ms", "device_memory_bytes",
    )
    return {
        name: _available_aggregate([
            item[delta_key][name]  # type: ignore[index]
            for item in comparisons
        ])
        for name in metric_names
    }


def analyse_completed_runs(
    completed: Iterable[tuple[ExperimentRun, Mapping[str, object]]], thresholds: CausalCompletionThresholds,
    *, expected_runs: Iterable[ExperimentRun] | None = None,
) -> dict[str, object]:
    """Report paired scheduling deltas and make a conservative completion decision."""
    records: dict[tuple[str, str, int, int], dict[str, dict[str, object]]] = defaultdict(dict)
    config_failures: list[dict[str, object]] = []
    for run, evidence in completed:
        method = getattr(run, "method", None)
        if method not in REQUIRED_METHODS:
            continue
        cell, row = _cell_key(run), _run_record(run, evidence)
        if method in records[cell]:
            raise ValueError(f"duplicate validated evidence for {cell!r} {method}")
        records[cell][method] = row

    expected_cells: set[tuple[str, str, int, int]] = set()
    if expected_runs is not None:
        for run in expected_runs:
            if getattr(run, "method", None) in REQUIRED_METHODS:
                expected_cells.add(_cell_key(run))
    else:
        expected_cells = set(records)
    missing = []
    comparisons = []
    for cell in sorted(expected_cells | set(records)):
        pair = records.get(cell, {})
        absent = [method for method in REQUIRED_METHODS if method not in pair]
        if absent:
            missing.append({"dataset": cell[0], "stream_mode": cell[1], "seed": cell[2], "batch_size": cell[3], "methods": absent})
            continue
        legacy, atomic, causal = pair["Ramen"], pair["StructuredAtomicRamen"], pair["CausalRamen"]
        equivalent, mismatches = _config_equivalence(atomic["run"], causal["run"])
        if not equivalent:
            config_failures.append({"cell": {"dataset": cell[0], "stream_mode": cell[1], "seed": cell[2], "batch_size": cell[3]}, "mismatches": mismatches})
        a_metrics, c_metrics = atomic["metrics"], causal["metrics"]
        comparisons.append({
            "dataset": cell[0], "stream_mode": cell[1], "seed": cell[2], "batch_size": cell[3],
            "structured_atomic_run_id": atomic["run_id"], "causal_run_id": causal["run_id"],
            "config_equivalent_except_method_identity": equivalent,
            "deltas_causal_minus_atomic": {key: _difference(c_metrics[key], a_metrics[key]) for key in a_metrics},
            "legacy_run_id": legacy["run_id"],
            "deltas_causal_minus_legacy": {key: _difference(c_metrics[key], legacy["metrics"][key]) for key in c_metrics},
        })

    micro = [item["deltas_causal_minus_atomic"]["micro_accuracy"] for item in comparisons]
    assert all(value is not None for value in micro)
    micro_values = [float(value) for value in micro]
    by_batch: dict[int, list[float]] = defaultdict(list)
    for comparison in comparisons:
        by_batch[comparison["batch_size"]].append(comparison["deltas_causal_minus_atomic"]["micro_accuracy"])
    batch_effects = [{"batch_size": size, "micro_gain": _aggregate([float(value) for value in values])}
                     for size, values in sorted(by_batch.items())]
    legacy_by_batch: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comparison in comparisons:
        legacy_by_batch[comparison["batch_size"]].append(comparison)
    legacy_batch_effects = [
        {"batch_size": size, "deltas_causal_minus_legacy": _metric_aggregates(rows, "deltas_causal_minus_legacy")}
        for size, rows in sorted(legacy_by_batch.items())
    ]
    legacy_b1_rows = legacy_by_batch.get(1, [])
    legacy_b1_diagnostic = {
        "status": "computed" if legacy_b1_rows else "unavailable",
        "purpose": "At B=1 legacy Ramen has no future-within-batch visibility; residual deltas diagnose implementation or numerical differences.",
        "deltas_causal_minus_legacy": _metric_aggregates(legacy_b1_rows, "deltas_causal_minus_legacy") if legacy_b1_rows else None,
    }
    seeds = {item["seed"] for item in comparisons}
    non_iid_streams = {item["stream_mode"] for item in comparisons if item["stream_mode"] != "iid_mixed"}
    cifar_comparisons = [item for item in comparisons if item["dataset"].lower() == "cifar100c"]
    full_cifar = bool(cifar_comparisons) and all(
        getattr(records[(item["dataset"], item["stream_mode"], item["seed"], item["batch_size"])]["CausalRamen"]["run"], "max_eval_samples", "missing") is None
        for item in cifar_comparisons
    )
    natural = any(item["dataset"].lower() in NATURAL_DATASETS for item in comparisons)
    requirements = {
        "fixed_seeds": {"passed": len(seeds) >= thresholds.minimum_fixed_seeds, "observed": len(seeds), "minimum": thresholds.minimum_fixed_seeds},
        "non_iid_streams": {"passed": len(non_iid_streams) >= thresholds.minimum_non_iid_streams, "observed": sorted(non_iid_streams), "minimum": thresholds.minimum_non_iid_streams},
        "full_cifar100c": {"passed": full_cifar or not thresholds.require_full_cifar100c, "required": thresholds.require_full_cifar100c},
        "natural_domain_dataset": {"passed": natural or not thresholds.require_natural_domain_dataset, "required": thresholds.require_natural_domain_dataset},
        "batch_size_one_diagnostic": {"passed": 1 in legacy_by_batch, "observed": sorted(legacy_by_batch)},
        "batch_size_sensitivity": {"passed": len(legacy_by_batch) >= 2, "observed": sorted(legacy_by_batch), "minimum_distinct_batch_sizes": 2},
    }
    aggregate = _aggregate(micro_values) if micro_values else {"count": 0, "mean": None, "std": None}
    stable = bool(micro_values) and aggregate["std"] <= thresholds.maximum_micro_std
    attribution_aggregates = _metric_aggregates(comparisons, "deltas_causal_minus_atomic")
    negative_delta = attribution_aggregates["negative_adaptation_rate"]["mean"]
    recovery_delta = attribution_aggregates["recovery_samples"]["mean"]
    worst_delta = attribution_aggregates["worst_domain_accuracy"]["mean"]
    weak_go_improvements = {
        "negative_adaptation_reduction": {
            "passed": negative_delta is not None and -negative_delta >= thresholds.minimum_mean_negative_adaptation_reduction_for_weak_go,
            "mean_reduction": None if negative_delta is None else -negative_delta,
            "minimum": thresholds.minimum_mean_negative_adaptation_reduction_for_weak_go,
        },
        "recovery_reduction": {
            "passed": recovery_delta is not None and -recovery_delta >= thresholds.minimum_mean_recovery_reduction_for_weak_go,
            "mean_reduction_samples": None if recovery_delta is None else -recovery_delta,
            "minimum": thresholds.minimum_mean_recovery_reduction_for_weak_go,
        },
        "worst_domain_gain": {
            "passed": worst_delta is not None and worst_delta >= thresholds.minimum_mean_worst_domain_gain_for_weak_go,
            "mean_gain": worst_delta,
            "minimum": thresholds.minimum_mean_worst_domain_gain_for_weak_go,
        },
    }
    meaningful_weak_improvement = any(item["passed"] for item in weak_go_improvements.values())
    sufficient = not missing and all(value["passed"] for value in requirements.values())
    if config_failures:
        decision, reason = "INVALID", "paired methods differ outside method identity"
    elif not sufficient:
        decision, reason = ("PILOT", "complete paired pilot evidence does not meet completion coverage") if not missing else ("INSUFFICIENT", "requested paired evidence is missing")
    elif stable and aggregate["mean"] >= thresholds.minimum_mean_micro_gain_for_go:
        decision, reason = "GO", "stable mean micro gain meets the predeclared GO threshold"
    elif thresholds.allow_weak_go and stable and aggregate["mean"] >= thresholds.minimum_mean_micro_gain_for_weak_go and meaningful_weak_improvement:
        decision, reason = "WEAK_GO", "stable micro result plus a configured secondary improvement; micro gain does not meet the GO threshold"
    else:
        decision, reason = "NO_GO", "complete evidence does not support a stable scheduling gain"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "analysis": "structured_atomic_vs_causal_scheduling",
        "thresholds": asdict(thresholds),
        "comparisons": comparisons,
        "attribution_metric_aggregates": attribution_aggregates,
        "batch_size_effects": batch_effects,
        "legacy_vs_causal_batch_size_effects": legacy_batch_effects,
        "legacy_vs_causal_b1_diagnostic": legacy_b1_diagnostic,
        "monotonic_batch_size_claim": {"status": "not_tested_as_a_hard_criterion", "reason": "batch-size effects are descriptive unless a future protocol explicitly configures monotonicity"},
        "missing_cells": missing,
        "config_equivalence_failures": config_failures,
        "decision": {"status": decision, "reason": reason, "mean_micro_gain": aggregate, "stability_passed": stable, "weak_go_improvements": weak_go_improvements, "requirements": requirements},
    }


def build_causal_matrix(*, batch_sizes: Sequence[int], **kwargs: Any) -> list[ExperimentRun]:
    """Use the runtime batch-size identity API, with a narrow legacy compatibility seam."""
    runs: list[ExperimentRun] = []
    for batch_size in batch_sizes:
        try:
            planned = build_experiment_matrix(batch_size=batch_size, methods=REQUIRED_METHODS, **kwargs)
        except TypeError as exc:
            defaults = {BATCH_SIZE_BY_DATASET[dataset] for dataset in kwargs["datasets"]}
            if batch_size not in defaults or len(defaults) != 1:
                raise ValueError("runtime experiment_matrix does not yet support batch_size planning; update it before analysing B=1 or repeated batch sizes") from exc
            planned = build_experiment_matrix(methods=REQUIRED_METHODS, **kwargs)
        runs.extend(planned)
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("batch-size matrix produced duplicate deterministic run identities")
    return runs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=Path, default=Path(__file__).resolve().parents[2] / "cfg/research/causal-completion-go-no-go.json")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--config-dir", default=Path(__file__).resolve().parents[2] / "cfg")
    parser.add_argument("--dataset", required=True, action="append")
    parser.add_argument("--stream", required=True, action="append")
    parser.add_argument("--seed", required=True, type=int, action="append")
    parser.add_argument("--batch-size", required=True, type=int, action="append", help="Repeatable; B=1 is supported.")
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--stream-block-size", type=int, default=64)
    parser.add_argument("--artifact-provenance", choices=SUPPORTED_ARTIFACT_PROVENANCE, default="fast")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if any(value <= 0 for value in args.batch_size):
            raise ValueError("batch sizes must be positive")
        thresholds = CausalCompletionThresholds.from_mapping(json.loads(args.thresholds.read_text(encoding="utf-8")))
        runs = build_causal_matrix(batch_sizes=args.batch_size, datasets=args.dataset, streams=args.stream, seeds=args.seed,
            evidence_dir=args.evidence_dir, config_dir=args.config_dir, device=args.device, max_eval_samples=args.max_eval_samples,
            artifact_provenance=args.artifact_provenance, data_root=args.data_root, stream_block_size=args.stream_block_size)
        report = analyse_completed_runs(load_completed_runs(runs), thresholds, expected_runs=runs)
    except MatrixEvidenceError as exc:
        print(json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "decision": {"status": exc.status, "reason": str(exc)}, "validation_failures": exc.failures}, sort_keys=True))
        return 1 if exc.status == "INSUFFICIENT" else 2
    except IncompleteRunError as exc:
        # Absent artifacts are an evidence-coverage problem, whereas an
        # existing artifact that fails strict validation is invalid evidence.
        status = "INSUFFICIENT" if _is_absent_evidence_error(exc) else "INVALID"
        print(json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "decision": {"status": status, "reason": str(exc)}}, sort_keys=True))
        return 1 if status == "INSUFFICIENT" else 2
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "decision": {"status": "INVALID", "reason": str(exc)}}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    status = report["decision"]["status"]
    return 0 if status in {"GO", "WEAK_GO"} else 2 if status == "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
