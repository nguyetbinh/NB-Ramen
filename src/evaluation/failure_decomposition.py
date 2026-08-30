"""Failure-mode decomposition utilities for paired Ramen evidence traces.

This module deliberately starts from *outcomes* rather than a proposed new
mechanism.  It provides two layers of analysis:

1. exact paired NoAdapt/Ramen outcome decomposition into stable-correct,
   beneficial, harmful, and unresolved samples; and
2. a generic oracle-ladder decomposition that measures how much error is
   removed when one pipeline stage at a time is replaced by an evaluator-only
   diagnostic intervention.

The functions fail closed when paired traces are not sample-identical.  Ground
truth is used only by the evaluator; this module is not imported by deployable
TTA methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OUTCOME_STABLE_CORRECT = "stable_correct"
OUTCOME_BENEFICIAL = "beneficial_adaptation"
OUTCOME_HARMFUL = "harmful_adaptation"
OUTCOME_UNRESOLVED = "unresolved"
OUTCOME_ORDER = (
    OUTCOME_STABLE_CORRECT,
    OUTCOME_BENEFICIAL,
    OUTCOME_HARMFUL,
    OUTCOME_UNRESOLVED,
)

PAIR_IDENTITY_FIELDS = (
    "timestep",
    "sample_idx",
    "ground_truth_domain",
    "ground_truth_class",
)


class FailureDecompositionError(ValueError):
    """Raised when evidence cannot support a valid paired decomposition."""


def _require_int(row: Mapping[str, Any], field: str, *, row_index: int) -> int:
    if field not in row:
        raise FailureDecompositionError(f"row {row_index} is missing required field {field!r}")
    value = row[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise FailureDecompositionError(f"row {row_index} field {field!r} must be an integer")
    return value


def _prediction_and_correct(row: Mapping[str, Any], *, row_index: int) -> tuple[int, bool]:
    prediction = _require_int(row, "prediction", row_index=row_index)
    target = _require_int(row, "ground_truth_class", row_index=row_index)
    recomputed = prediction == target
    if "correct" in row:
        stored = row["correct"]
        if not isinstance(stored, bool):
            raise FailureDecompositionError(f"row {row_index} field 'correct' must be boolean")
        if stored != recomputed:
            raise FailureDecompositionError(
                f"row {row_index} has inconsistent prediction/ground_truth_class/correct"
            )
    return prediction, recomputed


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, name: str) -> None:
    previous_timestep: int | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FailureDecompositionError(f"{name} row {index} is not a JSON object")
        timestep = _require_int(row, "timestep", row_index=index)
        for field in PAIR_IDENTITY_FIELDS[1:]:
            _require_int(row, field, row_index=index)
        _prediction_and_correct(row, row_index=index)
        if previous_timestep is not None and timestep <= previous_timestep:
            raise FailureDecompositionError(
                f"{name} timesteps must be strictly increasing; row {index} has {timestep}"
            )
        previous_timestep = timestep


def validate_paired_rows(
    reference_rows: Sequence[Mapping[str, Any]],
    adapted_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed unless two traces contain exactly the same evaluated items."""
    _validate_rows(reference_rows, name="reference")
    _validate_rows(adapted_rows, name="adapted")
    if len(reference_rows) != len(adapted_rows):
        raise FailureDecompositionError(
            f"paired traces have different lengths: {len(reference_rows)} != {len(adapted_rows)}"
        )
    for index, (reference, adapted) in enumerate(zip(reference_rows, adapted_rows)):
        for field in PAIR_IDENTITY_FIELDS:
            if reference[field] != adapted[field]:
                raise FailureDecompositionError(
                    f"paired trace mismatch at row {index} for {field!r}: "
                    f"{reference[field]!r} != {adapted[field]!r}"
                )


def classify_outcome(reference_correct: bool, adapted_correct: bool) -> str:
    """Return the exact paired adaptation-outcome category."""
    if reference_correct and adapted_correct:
        return OUTCOME_STABLE_CORRECT
    if not reference_correct and adapted_correct:
        return OUTCOME_BENEFICIAL
    if reference_correct and not adapted_correct:
        return OUTCOME_HARMFUL
    return OUTCOME_UNRESOLVED


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _outcome_summary_from_flags(
    reference_correct: Sequence[bool],
    adapted_correct: Sequence[bool],
) -> dict[str, Any]:
    if len(reference_correct) != len(adapted_correct):
        raise FailureDecompositionError("correctness vectors must have identical lengths")
    counts = {name: 0 for name in OUTCOME_ORDER}
    for before, after in zip(reference_correct, adapted_correct):
        counts[classify_outcome(bool(before), bool(after))] += 1

    total = len(reference_correct)
    reference_correct_count = sum(bool(value) for value in reference_correct)
    adapted_correct_count = sum(bool(value) for value in adapted_correct)
    reference_wrong_count = total - reference_correct_count
    rescued = counts[OUTCOME_BENEFICIAL]
    harmed = counts[OUTCOME_HARMFUL]
    reference_accuracy = _safe_rate(reference_correct_count, total)
    adapted_accuracy = _safe_rate(adapted_correct_count, total)
    accuracy_delta = (
        None if reference_accuracy is None or adapted_accuracy is None
        else adapted_accuracy - reference_accuracy
    )
    net_flip_rate = _safe_rate(rescued - harmed, total)
    identity_residual = (
        None if accuracy_delta is None or net_flip_rate is None
        else accuracy_delta - net_flip_rate
    )
    if identity_residual is not None and abs(identity_residual) > 1e-12:
        raise AssertionError("paired accuracy identity failed")

    return {
        "num_samples": total,
        "reference_accuracy": reference_accuracy,
        "adapted_accuracy": adapted_accuracy,
        "accuracy_delta": accuracy_delta,
        "counts": counts,
        "rates": {name: _safe_rate(count, total) for name, count in counts.items()},
        "help_rate_given_reference_wrong": _safe_rate(rescued, reference_wrong_count),
        "harm_rate_given_reference_correct": _safe_rate(harmed, reference_correct_count),
        "net_flip_rate": net_flip_rate,
        "identity": {
            "definition": "adapted_accuracy - reference_accuracy = beneficial_rate - harmful_rate",
            "residual": identity_residual,
        },
    }


def _paired_flags(
    reference_rows: Sequence[Mapping[str, Any]],
    adapted_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[bool], list[bool], list[str]]:
    validate_paired_rows(reference_rows, adapted_rows)
    reference_flags: list[bool] = []
    adapted_flags: list[bool] = []
    outcomes: list[str] = []
    for index, (reference, adapted) in enumerate(zip(reference_rows, adapted_rows)):
        _, before = _prediction_and_correct(reference, row_index=index)
        _, after = _prediction_and_correct(adapted, row_index=index)
        reference_flags.append(before)
        adapted_flags.append(after)
        outcomes.append(classify_outcome(before, after))
    return reference_flags, adapted_flags, outcomes


def paired_outcome_decomposition(
    reference_rows: Sequence[Mapping[str, Any]],
    adapted_rows: Sequence[Mapping[str, Any]],
    *,
    window_size: int | None = None,
    window_stride: int | None = None,
) -> dict[str, Any]:
    """Compute exact sample-level adaptation outcomes for paired traces.

    ``reference_rows`` should normally come from NoAdapt on the identical
    stream.  ``adapted_rows`` should come from Ramen or a controlled variant.
    The function also returns per-domain and optional temporal decompositions.
    """
    before, after, outcomes = _paired_flags(reference_rows, adapted_rows)
    result = _outcome_summary_from_flags(before, after)
    result["outcome_by_timestep"] = [
        {
            "timestep": int(adapted_rows[index]["timestep"]),
            "outcome": outcome,
        }
        for index, outcome in enumerate(outcomes)
    ]

    by_domain: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(adapted_rows):
        by_domain[int(row["ground_truth_domain"])].append(index)
    result["per_domain"] = {}
    for domain, indices in sorted(by_domain.items()):
        result["per_domain"][str(domain)] = _outcome_summary_from_flags(
            [before[index] for index in indices],
            [after[index] for index in indices],
        )

    if window_size is None:
        if window_stride is not None:
            raise FailureDecompositionError("window_stride requires window_size")
        result["windows"] = None
        return result
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
        raise FailureDecompositionError("window_size must be a positive integer")
    if window_stride is None:
        window_stride = window_size
    if not isinstance(window_stride, int) or isinstance(window_stride, bool) or window_stride <= 0:
        raise FailureDecompositionError("window_stride must be a positive integer")

    windows = []
    for start in range(0, max(0, len(before) - window_size + 1), window_stride):
        stop = start + window_size
        summary = _outcome_summary_from_flags(before[start:stop], after[start:stop])
        summary.update({
            "start_row": start,
            "end_row_exclusive": stop,
            "start_timestep": int(adapted_rows[start]["timestep"]),
            "end_timestep": int(adapted_rows[stop - 1]["timestep"]),
        })
        windows.append(summary)
    result["windows"] = windows
    return result


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def scalar_diagnostics_by_outcome(
    reference_rows: Sequence[Mapping[str, Any]],
    adapted_rows: Sequence[Mapping[str, Any]],
    fields: Iterable[str],
) -> dict[str, Any]:
    """Summarize scalar mechanism diagnostics separately by outcome class.

    Missing/``None`` values are ignored, but non-numeric populated values fail
    closed.  This makes the helper suitable for future compact diagnostics such
    as consensus mean, sign-disagreement rate, support count, and entropy.
    """
    _, _, outcomes = _paired_flags(reference_rows, adapted_rows)
    result: dict[str, Any] = {}
    for field in fields:
        grouped: dict[str, list[float]] = {name: [] for name in OUTCOME_ORDER}
        missing = 0
        for index, row in enumerate(adapted_rows):
            value = row.get(field)
            if value is None:
                missing += 1
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise FailureDecompositionError(
                    f"adapted row {index} field {field!r} must be finite numeric or null"
                )
            grouped[outcomes[index]].append(float(value))
        field_result = {"missing_count": missing, "groups": {}}
        for outcome in OUTCOME_ORDER:
            values = grouped[outcome]
            field_result["groups"][outcome] = {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "p10": _percentile(values, 0.10),
                "p90": _percentile(values, 0.90),
            }
        result[field] = field_result
    return result


def oracle_ladder_decomposition(
    stages: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Measure incremental error reduction across an ordered oracle ladder.

    The first stage should be the ordinary Ramen/control trace.  Each later
    stage should differ by one preregistered evaluator-only intervention when
    the goal is single-component attribution.  Negative increments are kept
    rather than clipped because pipeline interactions are scientifically
    meaningful and should not be hidden.
    """
    if not isinstance(stages, Mapping) or len(stages) < 2:
        raise FailureDecompositionError("oracle ladder requires at least two ordered stages")
    ordered = OrderedDict(stages)
    names = list(ordered)
    reference_name = names[0]
    reference_rows = ordered[reference_name]
    _validate_rows(reference_rows, name=reference_name)

    stage_results: list[dict[str, Any]] = []
    previous_error: float | None = None
    for stage_name, rows in ordered.items():
        if not isinstance(stage_name, str) or not stage_name:
            raise FailureDecompositionError("oracle stage names must be non-empty strings")
        validate_paired_rows(reference_rows, rows)
        correct = [
            _prediction_and_correct(row, row_index=index)[1]
            for index, row in enumerate(rows)
        ]
        accuracy = _safe_rate(sum(correct), len(correct))
        error = None if accuracy is None else 1.0 - accuracy
        incremental_reduction = (
            None if previous_error is None or error is None else previous_error - error
        )
        stage_results.append({
            "stage": stage_name,
            "num_samples": len(rows),
            "accuracy": accuracy,
            "error": error,
            "incremental_error_reduction": incremental_reduction,
            "regressed_vs_previous_stage": (
                None if incremental_reduction is None else incremental_reduction < 0
            ),
        })
        previous_error = error

    first_error = stage_results[0]["error"]
    final_error = stage_results[-1]["error"]
    total_recovery = (
        None if first_error is None or final_error is None else first_error - final_error
    )
    return {
        "reference_stage": reference_name,
        "stages": stage_results,
        "total_error_recovery": total_recovery,
        "note": (
            "Incremental reductions are ordered intervention effects, not guaranteed additive causal effects; "
            "negative values are retained to expose interactions."
        ),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read an evidence JSONL file with strict object/line validation."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                raise FailureDecompositionError(f"{path}: blank JSONL line {line_number}")
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise FailureDecompositionError(
                    f"{path}: malformed JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise FailureDecompositionError(f"{path}: line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_trace_outcomes(
    reference_trace: str | Path,
    adapted_trace: str | Path,
    *,
    window_size: int | None = None,
    window_stride: int | None = None,
    mechanism_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Load paired trace files and return provenance-bearing decomposition."""
    reference_trace = Path(reference_trace)
    adapted_trace = Path(adapted_trace)
    reference_rows = read_jsonl(reference_trace)
    adapted_rows = read_jsonl(adapted_trace)
    result = paired_outcome_decomposition(
        reference_rows,
        adapted_rows,
        window_size=window_size,
        window_stride=window_stride,
    )
    fields = tuple(mechanism_fields)
    if fields:
        result["mechanism_by_outcome"] = scalar_diagnostics_by_outcome(
            reference_rows, adapted_rows, fields
        )
    result["provenance"] = {
        "reference_trace": str(reference_trace.resolve()),
        "reference_trace_sha256": file_sha256(reference_trace),
        "adapted_trace": str(adapted_trace.resolve()),
        "adapted_trace_sha256": file_sha256(adapted_trace),
    }
    return result


def _parse_stage(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("stage must use NAME=TRACE.jsonl")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("stage name must be non-empty")
    if not path.strip():
        raise argparse.ArgumentTypeError("stage path must be non-empty")
    return name, Path(path).expanduser()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ramen failure-mode decomposition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    outcomes = subparsers.add_parser("outcomes", help="exact NoAdapt-vs-adapted flip decomposition")
    outcomes.add_argument("--reference-trace", required=True)
    outcomes.add_argument("--adapted-trace", required=True)
    outcomes.add_argument("--output", required=True)
    outcomes.add_argument("--window-size", type=int)
    outcomes.add_argument("--window-stride", type=int)
    outcomes.add_argument(
        "--mechanism-field", action="append", default=[],
        help="optional scalar adapted-trace field to stratify by outcome; repeatable",
    )

    ladder = subparsers.add_parser("ladder", help="ordered oracle-stage error decomposition")
    ladder.add_argument(
        "--stage", action="append", type=_parse_stage, required=True,
        help="ordered NAME=TRACE.jsonl; repeat at least twice",
    )
    ladder.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "outcomes":
        payload = compare_trace_outcomes(
            args.reference_trace,
            args.adapted_trace,
            window_size=args.window_size,
            window_stride=args.window_stride,
            mechanism_fields=args.mechanism_field,
        )
    else:
        if len(args.stage) < 2:
            raise FailureDecompositionError("ladder requires at least two --stage arguments")
        ordered: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        provenance: dict[str, Any] = {}
        for name, path in args.stage:
            if name in ordered:
                raise FailureDecompositionError(f"duplicate oracle stage name: {name}")
            ordered[name] = read_jsonl(path)
            provenance[name] = {"trace": str(path.resolve()), "sha256": file_sha256(path)}
        payload = oracle_ladder_decomposition(ordered)
        payload["provenance"] = provenance
    _write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
