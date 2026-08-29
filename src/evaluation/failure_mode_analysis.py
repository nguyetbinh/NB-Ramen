"""Pure, fail-closed offline diagnostics for Ramen failure-mode analysis.

This module deliberately accepts plain JSON-like mappings rather than replay
objects.  The replay artifact reader owns provenance verification; its caller
passes the verified rows and their shared identity here.  That keeps the
mathematics useful for fixtures and prevents this evaluator from becoming a
second artifact parser.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

try:
    from .causal_completion_analysis import _config_without_method
    from .failure_analysis_artifacts import ReplayArtifactError, ReplaySidecarReader, sha256_file
    from .evidence import validate_failure_analysis
    from ..streams import verify_stream_fingerprint
except ImportError:  # pragma: no cover - direct-file invocation
    from evaluation.causal_completion_analysis import _config_without_method
    from evaluation.failure_analysis_artifacts import ReplayArtifactError, ReplaySidecarReader, sha256_file
    from evaluation.evidence import validate_failure_analysis
    from streams import verify_stream_fingerprint


REPORT_SCHEMA_VERSION = 1
IDENTITY_FIELDS = ("timestep", "sample_idx", "ground_truth_domain", "ground_truth_class")
PREREGISTERED_COUNTERFACTUAL_THRESHOLDS = (0.50, 0.75, 1.00)
CANONICAL_CONFLICT_METRIC = "fraction_low_consensus_coordinates_v1"
_STATES = frozenset({"computed", "insufficient", "unavailable"})
_SIDECAR_TRACE_DIAGNOSTIC_FIELDS = (
    "schedule", "conflict_metric", "conflict", "batch_position",
    "future_support_count", "future_support_weight_fraction",
)


def _state(status: str, **values: Any) -> dict[str, Any]:
    if status not in _STATES:
        raise ValueError("invalid analysis state")
    return {"status": status, **values}


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value: object, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be a non-empty numeric sequence")
    return [_finite(item, name) for item in value]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity for finite, equal-size non-zero vectors."""
    a, b = _vector(left, "left"), _vector(right, "right")
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    aa, bb = sum(x * x for x in a), sum(x * x for x in b)
    if aa == 0.0 or bb == 0.0:
        raise ValueError("cosine is undefined for a zero vector")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / math.sqrt(aa * bb)))


def sign_disagreement_rate(left: Sequence[float], right: Sequence[float]) -> float:
    """SignSGD disagreement, with zero retaining its own deterministic sign."""
    a, b = _vector(left, "left"), _vector(right, "right")
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return sum((x > 0) != (y > 0) or (x == 0) != (y == 0) for x, y in zip(a, b)) / len(a)


def gradient_direction_corruption(all_gradient: Sequence[float], id_gradient: Sequence[float]) -> dict[str, float]:
    """Return open-set GDC and SignSGD-relevant SDR for one query."""
    return {"gdc": 1.0 - cosine_similarity(all_gradient, id_gradient),
            "sdr": sign_disagreement_rate(all_gradient, id_gradient)}


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    missing = [field for field in IDENTITY_FIELDS if field not in row]
    if missing:
        raise ValueError("missing trace identity fields: " + ", ".join(missing))
    return tuple(row[field] for field in IDENTITY_FIELDS)


def paired_outcome_decomposition(base_rows: Iterable[Mapping[str, Any]], adapted_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Strictly pair ordered traces and return the exact F0 flip decomposition."""
    base, adapted = list(base_rows), list(adapted_rows)
    if not base:
        return _state("insufficient", reason="no paired rows", count=0)
    if len(base) != len(adapted):
        raise ValueError("paired traces have different lengths")
    counts = {"safe": 0, "beneficial": 0, "harmful": 0, "unresolved": 0}
    for index, (reference, actual) in enumerate(zip(base, adapted)):
        if _identity(reference) != _identity(actual):
            raise ValueError(f"trace identity mismatch at row {index}")
        if not isinstance(reference.get("correct"), bool) or not isinstance(actual.get("correct"), bool):
            raise TypeError("trace correct fields must be booleans")
        key = ((reference["correct"], actual["correct"]))
        counts[{(True, True): "safe", (False, True): "beneficial", (True, False): "harmful", (False, False): "unresolved"}[key]] += 1
    total = len(base)
    base_correct = counts["safe"] + counts["harmful"]
    adapted_correct = counts["safe"] + counts["beneficial"]
    help_denominator, harm_denominator = counts["beneficial"] + counts["unresolved"], base_correct
    h, a = counts["beneficial"] / total, counts["harmful"] / total
    identity = adapted_correct / total - base_correct / total
    if not math.isclose(identity, h - a, rel_tol=0.0, abs_tol=1e-15):  # defensive invariant
        raise AssertionError("paired accuracy identity failed")
    return _state("computed", count=total, counts=counts, H=h, A=a,
                  HelpRate=None if help_denominator == 0 else counts["beneficial"] / help_denominator,
                  HarmRate=None if harm_denominator == 0 else counts["harmful"] / harm_denominator,
                  base_accuracy=base_correct / total, adapted_accuracy=adapted_correct / total,
                  accuracy_delta=identity, identity_h_minus_a=h - a)


def _mean(values: Sequence[float], *, minimum: int = 1) -> dict[str, Any]:
    if len(values) < minimum:
        return _state("insufficient", count=len(values), reason="too few finite observations")
    return _state("computed", count=len(values), mean=statistics.mean(values))


def oracle_gap_analysis(query_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Closed-set all-legal versus retrieved correctly-pseudolabelled oracle.

    Candidate mappings need ``correct_pseudolabel`` and optionally ``weight``.
    The metric is per-query availability, so it distinguishes absent legal
    evidence (F1) from legal evidence omitted by retrieval (F2).
    """
    rows = list(query_rows)
    if not rows:
        return _state("insufficient", count=0, reason="no query rows")
    legal_available = retrieved_available = 0
    for row in rows:
        legal, retrieved = row.get("legal_candidates"), row.get("retrieved_supports")
        if not isinstance(legal, list) or not isinstance(retrieved, list):
            return _state("unavailable", reason="legal candidates or retrieved supports absent")
        legal_ids = set()
        for item in legal:
            if not isinstance(item, Mapping) or "item_id" not in item or not isinstance(item.get("correct_pseudolabel"), bool):
                return _state("unavailable", reason="correct-pseudolabel oracle labels absent")
            legal_ids.add(item["item_id"])
        for item in retrieved:
            if not isinstance(item, Mapping) or item.get("item_id") not in legal_ids:
                raise ValueError("retrieved support is not a legal candidate")
        legal_available += int(any(item["correct_pseudolabel"] for item in legal))
        retrieved_available += int(any(item.get("correct_pseudolabel") is True for item in retrieved))
    n = len(rows)
    return _state("computed", oracle="correctly_pseudolabeled_legal_support_v1", count=n,
                  legal_oracle_rate=legal_available / n, retrieved_oracle_rate=retrieved_available / n,
                  memory_insufficiency_rate=1.0 - legal_available / n,
                  retrieval_gap=(legal_available - retrieved_available) / n)


def entropy_admission_analysis(items: Iterable[Mapping[str, Any]], *, entropy_threshold: float = 0.5) -> dict[str, Any]:
    """Group item influence by entropy and pseudo-label correctness."""
    threshold = _finite(entropy_threshold, "entropy_threshold")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    values = list(items)
    if not values:
        return _state("insufficient", count=0, reason="no memory items")
    for item in values:
        entropy = _finite(item.get("entropy"), "item.entropy")
        correct = item.get("correct_pseudolabel")
        if not isinstance(correct, bool):
            return _state("unavailable", reason="correct-pseudolabel labels absent")
        groups[("low" if entropy <= threshold else "high") + ("_correct" if correct else "_wrong")].append(item)
    result: dict[str, Any] = {}
    for name in ("low_correct", "low_wrong", "high_correct", "high_wrong"):
        group = groups[name]
        if not group:
            result[name] = _state("insufficient", count=0, reason="empty entropy/correctness group")
            continue
        def optional(field: str) -> list[float]:
            return [_finite(item[field], f"item.{field}") for item in group if field in item and item[field] is not None]
        admitted = [item.get("admitted") for item in group]
        if any(not isinstance(value, bool) for value in admitted):
            storage: Any = _state("unavailable", reason="admission flags absent")
        else:
            storage = _state("computed", count=len(group), value=sum(admitted) / len(group))
        downstream = optional("downstream_weight")
        result[name] = {"count": len(group), "storage_rate": storage,
                        "retrieval_frequency": _mean(optional("retrieval_frequency")),
                        "total_downstream_weight": _state("insufficient", count=0, reason="too few finite observations") if not downstream
                        else _state("computed", count=len(downstream), value=sum(downstream)),
                        "mean_retrieved_distance": _mean(optional("mean_retrieved_distance")),
                        "gradient_sign_agreement": _mean(optional("gradient_sign_agreement")),
                        "gradient_cosine": _mean(optional("gradient_cosine"))}
    return _state("computed", count=len(values), threshold=threshold, groups=result)


def gradient_conflict_association(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Associate the documented F3 conflict scalar with exact F0 flips."""
    values: dict[str, list[float]] = {"beneficial": [], "harmful": []}
    for row in rows:
        if row.get("conflict_metric") != CANONICAL_CONFLICT_METRIC:
            return _state("unavailable", metric=CANONICAL_CONFLICT_METRIC,
                          reason="canonical conflict metric absent from verified query diagnostics")
        category, conflict = row.get("outcome"), row.get("conflict")
        if category in values and conflict is not None:
            values[category].append(_finite(conflict, "conflict"))
    if not values["beneficial"] or not values["harmful"]:
        return _state("insufficient", metric=CANONICAL_CONFLICT_METRIC,
                      beneficial_count=len(values["beneficial"]), harmful_count=len(values["harmful"]),
                      reason="both beneficial and harmful exact F0 outcome groups required")
    beneficial, harmful = statistics.mean(values["beneficial"]), statistics.mean(values["harmful"])
    return _state("computed", metric=CANONICAL_CONFLICT_METRIC,
                  beneficial_count=len(values["beneficial"]), harmful_count=len(values["harmful"]),
                  beneficial_mean=beneficial, harmful_mean=harmful, harmful_minus_beneficial=harmful - beneficial)


def consensus_ramen_decision(evidence: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return GO only for replicated structured-stream conflict and completed F4 evidence."""
    rows = list(evidence)
    missing: list[str] = []
    structured = [row for row in rows if row.get("structured_stream") is True]
    streams = {row.get("stream") for row in structured if isinstance(row.get("stream"), str) and row.get("stream")}
    seeds = {row.get("seed") for row in structured if isinstance(row.get("seed"), (int, str)) and not isinstance(row.get("seed"), bool)}
    directions: list[int] = []
    completed_f4 = bool(structured)
    for row in structured:
        value = row.get("harmful_minus_beneficial")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value != 0:
            directions.append(1 if value > 0 else -1)
        f4 = row.get("f4")
        completed_f4 = completed_f4 and isinstance(f4, Mapping) and f4.get("status") == "computed" and f4.get("harmful_recovery_status") == "computed"
    if len(streams) <= 1:
        missing.append("conflict_direction_across_multiple_structured_streams")
    if not structured:
        missing.append("structured_stream_evidence")
    if len(seeds) <= 1:
        missing.append("conflict_direction_across_fixed_seeds")
    if len(directions) != len(rows) or not directions or len(set(directions)) != 1:
        missing.append("stable_harmful_vs_beneficial_conflict_direction")
    if not completed_f4:
        missing.append("completed_f4_oracle_recovery")
    return {"status": "GO" if not missing else "INSUFFICIENT", "evidence_count": len(rows),
            "stream_count": len(streams), "seed_count": len(seeds), "missing_conditions": missing}


def open_set_gradient_analysis(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise ID-only aggregate corruption; OOD labels remain evaluator-only."""
    measurements: list[dict[str, float]] = []
    for row in rows:
        all_gradient, id_gradient = row.get("all_gradient"), row.get("id_gradient")
        if all_gradient is None and id_gradient is None:
            continue
        if all_gradient is None or id_gradient is None:
            raise ValueError("open-set gradients must be complete per query")
        measurements.append(gradient_direction_corruption(all_gradient, id_gradient))
    if not measurements:
        return _state("unavailable", reason="ID-only and all-support replay gradients absent")
    return _state("computed", count=len(measurements),
                  mean_gdc=statistics.mean(item["gdc"] for item in measurements),
                  mean_sdr=statistics.mean(item["sdr"] for item in measurements))


def counterfactual_recovery_analysis(rows: Iterable[Mapping[str, Any]], *, thresholds: Sequence[float] = PREREGISTERED_COUNTERFACTUAL_THRESHOLDS) -> dict[str, Any]:
    """F4 exact prediction recovery from fixed-support replay predictions."""
    checked = tuple(_finite(value, "threshold") for value in thresholds)
    if checked != PREREGISTERED_COUNTERFACTUAL_THRESHOLDS:
        raise ValueError("counterfactual thresholds must equal the preregistered tuple")
    rows = list(rows)
    result: dict[str, Any] = {}
    for threshold in checked:
        key = f"{threshold:.2f}"
        predictions = []
        for row in rows:
            variants = row.get("counterfactual_predictions")
            # JSON object keys are strings, while in-process callers often use
            # the preregistered float itself.  Accept both representations.
            prediction = None if not isinstance(variants, Mapping) else variants.get(
                threshold, variants.get(key, variants.get(str(threshold)))
            )
            if prediction is None or row.get("reset_state_verified") is not True:
                result[key] = _state("unavailable", reason="complete reset-verified counterfactual predictions absent")
                break
            if not all(isinstance(row.get(name), bool) for name in ("base_correct", "adapted_correct")) or not isinstance(prediction, bool):
                raise ValueError("counterfactual correctness fields must be booleans")
            predictions.append((row["base_correct"], row["adapted_correct"], prediction))
        else:
            if not predictions:
                result[key] = _state("insufficient", count=0, reason="no counterfactual rows")
            else:
                harmful_events = sum(base and not actual for base, actual, _ in predictions)
                harmful_recoveries = sum(base and not actual and cf for base, actual, cf in predictions)
                new_harm_eligible = sum(actual for _, actual, _ in predictions)
                new_harm = sum(actual and not cf for _, actual, cf in predictions)
                adapted_accuracy = sum(actual for _, actual, _ in predictions) / len(predictions)
                counterfactual_accuracy = sum(cf for _, _, cf in predictions) / len(predictions)
                result[key] = _state(
                    "computed", count=len(predictions),
                    harmful_event_count=harmful_events,
                    harmful_recovery_count=harmful_recoveries,
                    harmful_recovery_rate=None if harmful_events == 0 else harmful_recoveries / harmful_events,
                    harmful_recovery_status="insufficient" if harmful_events == 0 else "computed",
                    new_harm_eligible_count=new_harm_eligible,
                    new_harm_count=new_harm,
                    new_harm_rate=None if new_harm_eligible == 0 else new_harm / new_harm_eligible,
                    new_harm_status="insufficient" if new_harm_eligible == 0 else "computed",
                    adapted_accuracy=adapted_accuracy,
                    counterfactual_accuracy=counterfactual_accuracy,
                    accuracy_delta_vs_adapted=counterfactual_accuracy - adapted_accuracy,
                )
    computed = [(name, data) for name, data in result.items() if data["status"] == "computed"]
    upper = _state("unavailable", reason="no complete preregistered variants") if not computed else _state(
        "computed", label="best_of_preregistered_evaluator_only_oracle_upper_bound",
        threshold=max(computed, key=lambda item: item[1]["accuracy_delta_vs_adapted"])[0])
    return {"status": "computed" if computed else "unavailable", "variants": result, "best_preregistered_upper_bound": upper}


def atomic_causal_pairing(atomic_rows: Iterable[Mapping[str, Any]], causal_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the same evaluated samples under atomic and causal schedules."""
    atomic, causal = list(atomic_rows), list(causal_rows)
    if not atomic or not causal:
        return _state("insufficient", reason="both atomic and causal rows required", atomic_count=len(atomic), causal_count=len(causal))
    if len(atomic) != len(causal):
        raise ValueError("atomic and causal traces have different lengths")
    if any(row.get("schedule") != "atomic" for row in atomic) or any(row.get("schedule") != "causal" for row in causal):
        return _state("insufficient", reason="true paired atomic and causal schedule rows required",
                      atomic_count=len(atomic), causal_count=len(causal))
    flips = {"atomic_only_correct": 0, "causal_only_correct": 0, "both_correct": 0, "both_wrong": 0}
    future_atomic: list[float] = []
    future_causal: list[float] = []
    future_weight_atomic: list[float] = []
    future_weight_causal: list[float] = []
    for index, (left, right) in enumerate(zip(atomic, causal)):
        if _identity(left) != _identity(right):
            raise ValueError(f"atomic/causal trace identity mismatch at row {index}")
        if not isinstance(left.get("correct"), bool) or not isinstance(right.get("correct"), bool):
            raise ValueError("atomic/causal correctness fields must be booleans")
        key = {(True, False): "atomic_only_correct", (False, True): "causal_only_correct",
               (True, True): "both_correct", (False, False): "both_wrong"}[(left["correct"], right["correct"])]
        flips[key] += 1
        for row, count_target, weight_target in ((left, future_atomic, future_weight_atomic),
                                                 (right, future_causal, future_weight_causal)):
            if "future_support_count" not in row or "future_support_weight_fraction" not in row:
                raise ValueError("atomic/causal future-support fields absent")
            count_target.append(_finite(row["future_support_count"], "future_support_count"))
            weight_target.append(_finite(row["future_support_weight_fraction"], "future_support_weight_fraction"))
    n = len(atomic)
    atomic_accuracy = (flips["atomic_only_correct"] + flips["both_correct"]) / n
    causal_accuracy = (flips["causal_only_correct"] + flips["both_correct"]) / n
    return _state("computed", count=n, atomic_accuracy=atomic_accuracy, causal_accuracy=causal_accuracy,
                  accuracy_delta=causal_accuracy - atomic_accuracy, paired_flips=flips,
                  future_support={"atomic_mean_count": statistics.mean(future_atomic),
                                  "causal_mean_count": statistics.mean(future_causal),
                                  "mean_count_delta": statistics.mean(future_causal) - statistics.mean(future_atomic),
                                  "atomic_mean_weight_fraction": statistics.mean(future_weight_atomic),
                                  "causal_mean_weight_fraction": statistics.mean(future_weight_causal),
                                  "mean_weight_fraction_delta": (statistics.mean(future_weight_causal)
                                                                  - statistics.mean(future_weight_atomic))})


def temporal_schedule_summary(rows: Iterable[Mapping[str, Any]], *, atomic_rows: Iterable[Mapping[str, Any]] | None = None,
                              causal_rows: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    values = list(rows)
    if not values:
        return _state("insufficient", count=0, reason="no schedule rows")
    fields = ("batch_position", "future_support_count", "future_support_weight_fraction")
    if any(field not in row for row in values for field in fields):
        return _state("unavailable", reason="batch/future-support fields absent")
    metrics = {field: _mean([_finite(row[field], field) for row in values]) for field in fields}
    atomic = [row for row in values if row.get("schedule") == "atomic"]
    causal = [row for row in values if row.get("schedule") == "causal"]
    return _state("computed", count=len(values), metrics=metrics,
                  atomic_count=len(atomic), causal_count=len(causal),
                  paired_schedule_comparison=(
                      atomic_causal_pairing(atomic_rows, causal_rows)
                      if atomic_rows is not None and causal_rows is not None else
                      _state("insufficient", reason="paired atomic/causal outcomes not supplied")))


def analyze_failure_modes(base_rows: Iterable[Mapping[str, Any]], adapted_rows: Iterable[Mapping[str, Any]], *, query_rows: Iterable[Mapping[str, Any]] | None = None, items: Iterable[Mapping[str, Any]] | None = None,
                          atomic_rows: Iterable[Mapping[str, Any]] | None = None, causal_rows: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Compose a canonical report; absent optional sidecars remain explicit."""
    base, adapted = list(base_rows), list(adapted_rows)
    paired = paired_outcome_decomposition(base, adapted)
    queries = None if query_rows is None else _join_query_outcomes(base, adapted, query_rows)
    return {"schema_version": REPORT_SCHEMA_VERSION, "status": "computed" if paired["status"] == "computed" else paired["status"],
            "paired_outcomes": paired,
            "oracle_gaps": _state("unavailable", reason="replay query rows absent") if queries is None else oracle_gap_analysis(queries),
            "entropy_admission": _state("unavailable", reason="memory item rows absent") if items is None else entropy_admission_analysis(items),
            "gradient_conflict": _state("unavailable", reason="conflict rows absent") if queries is None else gradient_conflict_association(queries),
            "open_set": _state("unavailable", reason="raw replay gradients absent") if queries is None else open_set_gradient_analysis(queries),
            "counterfactual": _state("unavailable", reason="counterfactual replay rows absent") if queries is None else counterfactual_recovery_analysis(queries),
            "temporal_schedule": _state("unavailable", reason="schedule rows absent") if queries is None else temporal_schedule_summary(queries, atomic_rows=atomic_rows, causal_rows=causal_rows)}


def _join_query_outcomes(base_rows: Iterable[Mapping[str, Any]], adapted_rows: Iterable[Mapping[str, Any]],
                         query_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Bind evaluator-only F0 outcomes to replay diagnostics by exact identity."""
    base, adapted, queries = list(base_rows), list(adapted_rows), list(query_rows)
    if len(base) != len(adapted):
        raise ValueError("paired traces have different lengths")
    paired: dict[tuple[Any, ...], tuple[bool, bool]] = {}
    for index, (reference, actual) in enumerate(zip(base, adapted)):
        identity = _identity(reference)
        if identity != _identity(actual):
            raise ValueError(f"trace identity mismatch at row {index}")
        if identity in paired:
            raise ValueError("exact F0 trace identities must be unique")
        if not isinstance(reference.get("correct"), bool) or not isinstance(actual.get("correct"), bool):
            raise ValueError("paired trace correctness fields must be booleans")
        paired[identity] = (reference["correct"], actual["correct"])
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    categories = {(True, True): "safe", (False, True): "beneficial",
                  (True, False): "harmful", (False, False): "unresolved"}
    for query in queries:
        identity = _identity(query)
        if identity in seen or identity not in paired:
            raise ValueError("replay query does not strictly join an exact F0 pair")
        seen.add(identity)
        base_correct, adapted_correct = paired[identity]
        merged = dict(query)
        merged.update(base_correct=base_correct, adapted_correct=adapted_correct,
                      outcome=categories[(base_correct, adapted_correct)])
        result.append(merged)
    if len(result) != len(paired):
        raise ValueError("replay query rows do not cover every exact F0 pair")
    return result


def _read_json_rows(path: str) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows, identity = payload, None
    elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        rows, identity = payload["rows"], payload.get("identity")
    else:
        raise ValueError("input must be a JSON row list or {rows, identity} object")
    if not all(isinstance(row, Mapping) for row in rows) or identity is not None and not isinstance(identity, Mapping):
        raise ValueError("rows and identity must be mappings")
    return list(rows), identity


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid trace.jsonl") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("trace.jsonl must contain object rows")
    return rows


def _query_identity(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    evidence = row.get("evaluator_sample_identity")
    if not isinstance(evidence, Mapping) or "producer_query_timestep" not in row:
        raise ValueError("sidecar query has no evaluator identity")
    if "sample_idx" not in evidence or "ground_truth_domain" not in evidence:
        raise ValueError("sidecar query evaluator identity is incomplete")
    return (row["producer_query_timestep"], evidence["sample_idx"], evidence["ground_truth_domain"])


def _segment(row: Mapping[str, Any], *, label: str) -> Any:
    # New artifacts emit an explicit segment index.  A producer timestep is a
    # safe legacy segment only for rows that are otherwise uniquely identified.
    if "segment_index" in row:
        return row["segment_index"]
    if "producer_query_timestep" in row:
        return row["producer_query_timestep"]
    raise ValueError(f"{label} has no segment_index")


def _candidate_key(value: Any, segment: Any) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        if "item_id" not in value:
            raise ValueError("legal candidate has no item_id")
        return (_segment(value, label="legal candidate") if "segment_index" in value or "producer_query_timestep" in value else segment,
                value["item_id"])
    return (segment, value)


def _nested_supports(ids: Any, weights: Any, valid: Any = None, predicted: Any = None,
                     distances: Any = None) -> list[tuple[Any, Any, Any, Any]]:
    """Flatten class-balanced [class][rank] support tensors without losing alignment."""
    sequence = lambda value: isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if sequence(ids):
        if not sequence(weights) or len(ids) != len(weights):
            raise ValueError("retrieved support IDs and weights have different nested shapes")
        if valid is not None and (not sequence(valid) or len(ids) != len(valid)):
            raise ValueError("retrieved support valid-mask shape differs from IDs")
        if predicted is not None and (not sequence(predicted) or len(ids) != len(predicted)):
            raise ValueError("retrieved support predicted-class shape differs from IDs")
        if distances is not None and (not sequence(distances) or len(ids) != len(distances)):
            raise ValueError("retrieved support distance shape differs from IDs")
        result: list[tuple[Any, Any, Any]] = []
        for index, item_id in enumerate(ids):
            result.extend(_nested_supports(item_id, weights[index], None if valid is None else valid[index],
                                           None if predicted is None else predicted[index],
                                           None if distances is None else distances[index]))
        return result
    if sequence(weights) or sequence(valid) or sequence(predicted) or sequence(distances):
        raise ValueError("retrieved support IDs and aligned fields have different nested shapes")
    if valid is not None and not isinstance(valid, bool):
        raise ValueError("retrieved support valid mask must be boolean")
    if valid is False or ids == -1:
        return []
    if isinstance(ids, (list, dict)) or ids is None:
        raise ValueError("retrieved support item_id is invalid")
    if predicted is not None and not isinstance(predicted, int):
        raise ValueError("retrieved support predicted class must be integer")
    return [(ids, _finite(weights, "retrieved weight"), predicted,
             None if distances is None else _finite(distances, "retrieved distance"))]


def _validate_sidecar_trace_diagnostics(query: Mapping[str, Any], trace_row: Mapping[str, Any]) -> None:
    """Reject a sidecar that tries to supply F3/F5 diagnostics inconsistent with its trace.

    Replay sidecars are bound to the run manifest, but the canonical F3/F5
    source is the validated trace family.  Older v4 sidecars intentionally do
    not duplicate these values.  If a sidecar does provide any of them, require
    the complete family and exact equality to the same evaluator trace row.
    """
    present = [field in query for field in _SIDECAR_TRACE_DIAGNOSTIC_FIELDS]
    if not any(present):
        return
    if not all(present):
        raise ValueError("sidecar F3/F5 diagnostics must be all present or all absent")
    for field in _SIDECAR_TRACE_DIAGNOSTIC_FIELDS:
        value, expected = query[field], trace_row.get(field)
        if value is None or expected is None or value != expected:
            raise ValueError(f"sidecar F3/F5 diagnostic {field} disagrees with evaluator trace")


def _verified_trace_run(run_dir: str) -> tuple[Path, Mapping[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate baseline-compatible manifest, stream, and evaluator trace evidence."""
    root = Path(run_dir)
    manifest_path, stream_path, trace_path = root / "manifest.json", root / "stream.json", root / "trace.jsonl"
    manifest, stream, trace = _read_json(manifest_path, "manifest"), _read_json(stream_path, "stream export"), _trace_rows(trace_path)
    if not verify_stream_fingerprint(stream):
        raise ValueError("stream fingerprint is invalid")
    stream_fingerprint = stream.get("fingerprint") or stream.get("metadata", {}).get("fingerprint")
    if not isinstance(stream_fingerprint, str):
        raise ValueError("stream fingerprint is absent")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("manifest run_id is absent")
    args = manifest.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("manifest args are malformed")
    has_failure_analysis = ["failure_analysis" in row for row in trace]
    if any(has_failure_analysis) and not all(has_failure_analysis):
        raise ValueError("failure_analysis trace fields must be all present or all absent")
    is_adapted_replay = (
        args.get("failure_analysis_profile") == "replay_v1"
        and args.get("tta_algo") != "NoAdapt"
    )
    if is_adapted_replay and not all(has_failure_analysis):
        raise ValueError("adapted replay_v1 trace requires failure_analysis on every row")
    for index, row in enumerate(trace):
        if "failure_analysis" not in row:
            continue
        try:
            validate_failure_analysis(row["failure_analysis"])
        except ValueError as exc:
            raise ValueError(f"invalid failure_analysis at trace row {index}: {exc}") from exc
    # Flatten only validated method diagnostics.  NoAdapt traces may omit this
    # optional family entirely, while adapted replay_v1 traces are required to
    # provide it above.
    trace = [{**row, **(row["failure_analysis"] if "failure_analysis" in row else {})}
             for row in trace]
    git = manifest.get("git", {})
    if not isinstance(git, Mapping):
        raise ValueError("manifest git evidence is malformed")
    source_evidence = git.get("source", {})
    if not isinstance(source_evidence, Mapping):
        raise ValueError("manifest source evidence is malformed")
    source = source_evidence.get("fingerprint")
    if source is not None and not isinstance(source, str):
        raise ValueError("manifest source fingerprint is malformed")
    identities: set[tuple[Any, Any, Any]] = set()
    required = ("schema_version", "run_id", "timestep", "sample_idx", "ground_truth_domain", "ground_truth_class", "prediction", "correct")
    for row in trace:
        if any(field not in row for field in required) or row.get("run_id") != run_id or not isinstance(row.get("correct"), bool):
            raise ValueError("trace schema or run identity is invalid")
        key = (row["timestep"], row["sample_idx"], row["ground_truth_domain"])
        if key in identities:
            raise ValueError("trace identities must be complete and unique")
        identities.add(key)
    identity = {"run_id": run_id, "stream_fingerprint": stream_fingerprint, "source_fingerprint": source,
                "manifest_sha256": sha256_file(manifest_path)}
    return root, manifest, trace, identity


_PAIR_EVALUATOR_ARGUMENTS = (
    "dataset", "model", "seed", "stream_seed", "batch_size", "max_eval_samples",
    "stream_mode", "stream_block_size",
)


def _pair_contract(root: Path, manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the non-method identity required for a verified paired result.

    ``manifest.config`` deliberately is not compared: it contains the method's
    hyperparameters, which must differ for a NoAdapt-versus-adapted comparison.
    The common evaluator configuration lives in the canonical manifest args.
    """
    args = manifest.get("args")
    artifacts = manifest.get("artifacts")
    if not isinstance(args, Mapping):
        raise ValueError("manifest pairing args are missing")
    if not isinstance(artifacts, Mapping) or artifacts.get("status") != "verified":
        raise ValueError("manifest pairing artifact verification is missing")
    device = manifest.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("manifest pairing device is missing")
    if args.get("device") != device:
        raise ValueError("manifest pairing device disagrees with args.device")
    evaluator = {}
    for field in _PAIR_EVALUATOR_ARGUMENTS:
        if field not in args:
            raise ValueError(f"manifest pairing evaluator config is missing args.{field}")
        evaluator[field] = args[field]
    model_artifact, dataset_artifact = artifacts.get("model"), artifacts.get("dataset")
    if not isinstance(model_artifact, Mapping) or model_artifact.get("status") != "verified":
        raise ValueError("manifest pairing model artifact verification is missing")
    if not isinstance(dataset_artifact, Mapping) or dataset_artifact.get("status") != "verified":
        raise ValueError("manifest pairing dataset artifact verification is missing")
    model_digest, dataset_digest = model_artifact.get("actual_sha256"), dataset_artifact.get("root_digest")
    if not isinstance(model_digest, str) or not model_digest:
        raise ValueError("manifest pairing model artifact digest is missing")
    if not isinstance(dataset_digest, str) or not dataset_digest:
        raise ValueError("manifest pairing dataset artifact digest is missing")
    if model_artifact.get("model") != args["model"]:
        raise ValueError("manifest pairing model artifact disagrees with args.model")
    if "reference_trace" not in args or (args["reference_trace"] is not None and not isinstance(args["reference_trace"], str)):
        raise ValueError("manifest pairing reference binding is missing or malformed")
    return {
        "device": device,
        "evaluator_config": evaluator,
        "model_artifact_sha256": model_digest,
        "dataset_artifact_digest": dataset_digest,
        "stream_fingerprint": identity["stream_fingerprint"],
        "source_fingerprint": identity["source_fingerprint"],
        "reference_trace": args["reference_trace"],
        "trace_path": str((root / "trace.jsonl").resolve()),
    }


def _schedule_adaptation_config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the canonical method-normalized config for an F5 schedule pair.

    Baseline-versus-adapted F0 pairing intentionally does not use this: a
    NoAdapt baseline has no corresponding adaptation mechanics.  F5 instead
    attributes a schedule effect, so every adaptation mechanism must match.
    """
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("manifest schedule-pair adaptation config is missing or malformed")
    normalized = _config_without_method(config)
    if not isinstance(normalized, Mapping) or not normalized:  # defensive: preserve fail-closed semantics if normalizer changes
        raise ValueError("manifest schedule-pair adaptation config is missing or malformed")
    return normalized


def _validate_verified_compatibility(left_root: Path, left_manifest: Mapping[str, Any], left_identity: Mapping[str, Any],
                                     right_root: Path, right_manifest: Mapping[str, Any], right_identity: Mapping[str, Any],
                                     *, left_name: str, right_name: str, reference_mode: str,
                                     require_same_adaptation_config: bool = False) -> None:
    """Fail closed unless verified runs share their non-method evaluator identity."""
    left = _pair_contract(left_root, left_manifest, left_identity)
    right = _pair_contract(right_root, right_manifest, right_identity)
    for field in ("device", "evaluator_config", "model_artifact_sha256", "dataset_artifact_digest",
                  "stream_fingerprint", "source_fingerprint"):
        if left[field] != right[field]:
            raise ValueError(f"{left_name} and {right_name} verified run {field} mismatch")
    if require_same_adaptation_config and _schedule_adaptation_config(left_manifest) != _schedule_adaptation_config(right_manifest):
        raise ValueError(f"{left_name} and {right_name} verified run adaptation config mismatch")
    if reference_mode == "bind-right-to-left":
        # NoAdapt may omit a reference trace, but an adapted run that records
        # one must point at the exact baseline trace used for F0.
        if right["reference_trace"] is not None and Path(right["reference_trace"]).resolve() != Path(left["trace_path"]):
            raise ValueError(f"{right_name} reference trace does not bind to the {left_name} trace")
    elif reference_mode == "same-optional-reference":
        # Schedule variants may legitimately have no baseline binding in older
        # artifacts.  When present, both must bind to the same reference run.
        if (left["reference_trace"] is None) != (right["reference_trace"] is None):
            raise ValueError(f"{left_name} and {right_name} reference trace binding mismatch")
        if left["reference_trace"] is not None and Path(left["reference_trace"]).resolve() != Path(right["reference_trace"]).resolve():
            raise ValueError(f"{left_name} and {right_name} reference traces differ")
    else:  # pragma: no cover - internal caller invariant
        raise ValueError("unknown verified reference-binding mode")


def _validate_verified_pair(base_root: Path, base_manifest: Mapping[str, Any], base_identity: Mapping[str, Any],
                           adapted_root: Path, adapted_manifest: Mapping[str, Any], adapted_identity: Mapping[str, Any]) -> None:
    """Fail closed unless a baseline/adapted pair shares its evaluator identity."""
    _validate_verified_compatibility(base_root, base_manifest, base_identity,
                                    adapted_root, adapted_manifest, adapted_identity,
                                    left_name="baseline", right_name="adapted",
                                    reference_mode="bind-right-to-left")


def _verified_run_rows(run_dir: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read a completed adapted run and turn its replay sidecar into rows."""
    root, manifest, trace, identity = _verified_trace_run(run_dir)
    manifest_path = root / "manifest.json"
    stream_fingerprint, source, run_id = identity["stream_fingerprint"], identity["source_fingerprint"], identity["run_id"]
    try:
        reader = ReplaySidecarReader(root / "failure-analysis", manifest_sha256=sha256_file(manifest_path),
                                     stream_fingerprint=stream_fingerprint, source_fingerprint=source, run_id=run_id)
    except ReplayArtifactError as exc:
        raise ValueError(f"invalid replay sidecar: {exc}") from exc
    if reader.metadata.get("status") != "completed":
        raise ValueError("replay sidecar must be completed")
    trace_by_identity: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for row in trace:
        key = (row.get("timestep"), row.get("sample_idx"), row.get("ground_truth_domain"))
        if None in key or key in trace_by_identity:
            raise ValueError("trace identities must be complete and unique")
        trace_by_identity[key] = row
    items: dict[tuple[Any, Any], dict[str, Any]] = {}
    for item in reader.rows("items"):
        if "item_id" not in item:
            raise ValueError("sidecar item has no item_id")
        key = (_segment(item, label="sidecar item"), item["item_id"])
        if key in items:
            raise ValueError("ambiguous sidecar item reference")
        items[key] = item
    queries: list[dict[str, Any]] = []
    retrieval_stats: dict[tuple[Any, Any], dict[str, Any]] = defaultdict(lambda: {"count": 0, "weight": 0.0, "distances": []})
    for query in reader.rows("queries"):
        trace_row = trace_by_identity.get(_query_identity(query))
        if trace_row is None or query.get("ground_truth_class") != trace_row.get("ground_truth_class"):
            raise ValueError("sidecar query does not match evaluator trace")
        _validate_sidecar_trace_diagnostics(query, trace_row)
        segment = _segment(query, label="sidecar query")
        legal = query.get("legal_candidates")
        retrieved_ids, weights = query.get("retrieved_support_ids"), query.get("retrieved_weights")
        valid = query.get("retrieved_valid_mask", query.get("support_valid_mask", trace_row.get("support_valid_mask")))
        predicted = query.get("retrieved_predicted_classes", query.get("support_predicted_classes", trace_row.get("support_predicted_classes")))
        distances = query.get("retrieved_distances", query.get("support_distances", trace_row.get("support_distances")))
        if not isinstance(legal, list):
            raise ValueError("sidecar query support evidence is incomplete")
        converted_legal: list[dict[str, Any]] = []
        legal_keys: set[tuple[Any, Any]] = set()
        for candidate in legal:
            key = _candidate_key(candidate, segment)
            if key in legal_keys or key not in items:
                raise ValueError("missing or ambiguous legal candidate reference")
            legal_keys.add(key)
            item = items[key]
            if not isinstance(item.get("predicted_class"), int) or not isinstance(item.get("ground_truth_class"), int):
                raise ValueError("legal candidate evaluator pseudo-label evidence is absent")
            converted_legal.append({"item_id": key, "correct_pseudolabel": item["predicted_class"] == item["ground_truth_class"]})
        retrieved: list[dict[str, Any]] = []
        for support_id, weight, support_prediction, distance in _nested_supports(retrieved_ids, weights, valid, predicted, distances):
            key = _candidate_key(support_id, segment)
            if key not in legal_keys:
                raise ValueError("retrieved support is missing from legal candidates")
            item = items[key]
            if support_prediction is not None and support_prediction != item["predicted_class"]:
                raise ValueError("retrieved support predicted class disagrees with item")
            if not isinstance(item.get("ground_truth_class"), int):
                raise ValueError("retrieved support evaluator pseudo-label evidence is absent")
            support = {"item_id": key, "correct_pseudolabel": item["predicted_class"] == item["ground_truth_class"], "weight": weight}
            if distance is not None:
                support["distance"] = distance
            retrieved.append(support)
            retrieval_stats[key]["count"] += 1
            retrieval_stats[key]["weight"] += weight
            if distance is not None:
                retrieval_stats[key]["distances"].append(distance)
        variants: dict[float, bool] = {}
        for cf in query.get("counterfactuals", []):
            if not isinstance(cf, Mapping) or "threshold" not in cf or "prediction" not in cf:
                raise ValueError("counterfactual evidence is malformed")
            threshold = _finite(cf["threshold"], "counterfactual threshold")
            if threshold in variants or not isinstance(cf["prediction"], int):
                raise ValueError("ambiguous counterfactual evidence")
            variants[threshold] = cf["prediction"] == trace_row["ground_truth_class"]
        merged = dict(query)
        merged.update(trace_row)
        failure_trace = trace_row.get("failure_analysis")
        if isinstance(failure_trace, Mapping):
            merged.update(failure_trace)
        # Evaluator-side joins are authoritative over opaque method payloads.
        merged.update(legal_candidates=converted_legal, retrieved_supports=retrieved,
                      base_correct=None, adapted_correct=trace_row["correct"],
                      counterfactual_predictions=variants, reset_state_verified=query.get("reset_state_verified") is True)
        queries.append(merged)
    if len(queries) != len(trace):
        raise ValueError("completed sidecar must cover every trace row")
    enriched_items: list[dict[str, Any]] = []
    query_count = len(queries)
    for key, item in items.items():
        enriched = dict(item)
        if isinstance(item.get("predicted_class"), int) and isinstance(item.get("ground_truth_class"), int):
            enriched["correct_pseudolabel"] = item["predicted_class"] == item["ground_truth_class"]
        if "entropy" not in enriched and "normalized_entropy" in enriched:
            enriched["entropy"] = enriched["normalized_entropy"]
        if "admitted" not in enriched and "admitted_to_memory" in enriched:
            enriched["admitted"] = enriched["admitted_to_memory"]
        stats = retrieval_stats[key]
        enriched["retrieval_frequency"] = stats["count"] / query_count
        enriched["downstream_weight"] = stats["weight"]
        if stats["distances"]:
            enriched["mean_retrieved_distance"] = statistics.mean(stats["distances"])
        enriched_items.append(enriched)
    return trace, queries, enriched_items, identity


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Ramen failure-mode analysis")
    parser.add_argument("--base", help="raw JSON base rows (unverified mode)")
    parser.add_argument("--adapted", help="raw JSON adapted rows (unverified mode)")
    parser.add_argument("--baseline-run-dir", help="verified baseline run directory")
    parser.add_argument("--adapted-run-dir", help="verified adapted run directory")
    parser.add_argument("--atomic-run-dir", help="verified atomic adapted run for F5")
    parser.add_argument("--causal-run-dir", help="verified causal adapted run for F5")
    parser.add_argument("--queries")
    parser.add_argument("--items")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        verified = args.baseline_run_dir is not None or args.adapted_run_dir is not None
        if verified:
            if not args.baseline_run_dir or not args.adapted_run_dir or args.base or args.adapted or args.queries or args.items:
                raise ValueError("verified mode requires only --baseline-run-dir and --adapted-run-dir (plus optional F5 runs)")
            base_root, base_manifest, base, base_identity = _verified_trace_run(args.baseline_run_dir)
            adapted_root, adapted_manifest, adapted_trace, adapted_identity = _verified_trace_run(args.adapted_run_dir)
            _validate_verified_pair(base_root, base_manifest, base_identity,
                                    adapted_root, adapted_manifest, adapted_identity)
            adapted, queries, items, sidecar_identity = _verified_run_rows(args.adapted_run_dir)
            if sidecar_identity != adapted_identity or adapted != adapted_trace:
                raise AssertionError("verified adapted run changed while reading its replay sidecar")
            base_by_key = {(r["timestep"], r["sample_idx"], r["ground_truth_domain"]): r for r in base}
            for query in queries:
                key = (query["timestep"], query["sample_idx"], query["ground_truth_domain"])
                reference = base_by_key.get(key)
                if reference is None or reference.get("ground_truth_class") != query.get("ground_truth_class"):
                    raise ValueError("baseline trace does not strictly pair adapted query")
                query["base_correct"] = reference.get("correct")
            atomic_rows = causal_rows = None
            if args.atomic_run_dir or args.causal_run_dir:
                if not args.atomic_run_dir or not args.causal_run_dir:
                    raise ValueError("F5 requires both --atomic-run-dir and --causal-run-dir")
                atomic_root, atomic_manifest, _, atomic_identity = _verified_trace_run(args.atomic_run_dir)
                causal_root, causal_manifest, _, causal_identity = _verified_trace_run(args.causal_run_dir)
                _validate_verified_compatibility(atomic_root, atomic_manifest, atomic_identity,
                                                causal_root, causal_manifest, causal_identity,
                                                left_name="atomic", right_name="causal",
                                                reference_mode="same-optional-reference",
                                                require_same_adaptation_config=True)
                atomic_rows, _, _, atomic_sidecar_identity = _verified_run_rows(args.atomic_run_dir)
                causal_rows, _, _, causal_sidecar_identity = _verified_run_rows(args.causal_run_dir)
                if atomic_sidecar_identity != atomic_identity or causal_sidecar_identity != causal_identity:
                    raise AssertionError("verified F5 run changed while reading its replay sidecar")
            report = analyze_failure_modes(base, adapted, query_rows=queries, items=items,
                                           atomic_rows=atomic_rows, causal_rows=causal_rows)
            report["provenance"] = {"verified": True,
                                    "baseline": base_identity, "adapted": adapted_identity}
            report["consensus_ramen_decision"] = consensus_ramen_decision([])
        else:
            if not args.base or not args.adapted:
                raise ValueError("raw mode requires --base and --adapted")
            base, base_identity = _read_json_rows(args.base)
            adapted, adapted_identity = _read_json_rows(args.adapted)
            if base_identity is not None and adapted_identity is not None and base_identity != adapted_identity:
                raise ValueError("base and adapted manifest/stream identity mismatch")
            queries = _read_json_rows(args.queries)[0] if args.queries else None
            items = _read_json_rows(args.items)[0] if args.items else None
            report = analyze_failure_modes(base, adapted, query_rows=queries, items=items)
            report["provenance"] = {"verified": False, "reason": "raw JSON inputs are unverified"}
            report["consensus_ramen_decision"] = consensus_ramen_decision([])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {"schema_version": REPORT_SCHEMA_VERSION, "status": "invalid", "error": str(exc)}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0 if report["status"] != "invalid" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
