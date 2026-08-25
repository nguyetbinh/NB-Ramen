"""Deterministic, dependency-free metrics for open-set evaluation.

An OOD score follows one convention throughout this module: larger scores mean
that a sample is more likely to be OOD.  Ground-truth ``is_ood`` flags and
class labels are evaluator-only inputs and must not be used by adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class OpenSetMetrics:
    """Open-set metrics computed from one fully labelled evaluation stream.

    ``fpr95_threshold`` is the largest observed score whose OOD recall is at
    least 95%.  With the decision rule ``score >= threshold`` this minimizes
    FPR among thresholds meeting the target, while treating score ties as one
    indivisible group.  ``h_score`` combines ID-only classification accuracy
    and the resulting OOD recall at that threshold.
    """

    id_accuracy: float
    auroc: float
    fpr_at_95_tpr: float
    fpr95_threshold: float
    ood_recall_at_fpr95: float
    h_score: float
    id_count: int
    ood_count: int


def _bool_values(values: Iterable[bool], name: str) -> list[bool]:
    result = list(values)
    if any(not isinstance(value, bool) for value in result):
        raise TypeError(f"{name} values must be booleans")
    return result


def _finite_scores(values: Iterable[float], name: str = "ood_scores") -> list[float]:
    result = list(values)
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{name} values must be finite real numbers")
    return [float(value) for value in result]


def _validate_detection_inputs(
    is_ood: Iterable[bool], ood_scores: Iterable[float]
) -> tuple[list[bool], list[float]]:
    flags = _bool_values(is_ood, "is_ood")
    scores = _finite_scores(ood_scores)
    if not flags:
        raise ValueError("OOD metrics are undefined for an empty stream")
    if len(flags) != len(scores):
        raise ValueError("is_ood and ood_scores must have equal length")
    if not any(flags):
        raise ValueError("OOD metrics require at least one OOD sample")
    if all(flags):
        raise ValueError("OOD metrics require at least one ID sample")
    return flags, scores


def id_accuracy(
    predictions: Iterable[Any], ground_truth_classes: Iterable[Any], is_ood: Iterable[bool]
) -> float:
    """Return classification accuracy restricted to ground-truth ID samples."""
    predicted = list(predictions)
    truth = list(ground_truth_classes)
    flags = _bool_values(is_ood, "is_ood")
    if len(predicted) != len(truth) or len(truth) != len(flags):
        raise ValueError("predictions, ground_truth_classes, and is_ood must have equal length")
    id_indices = [index for index, flag in enumerate(flags) if not flag]
    if not id_indices:
        raise ValueError("ID accuracy is undefined when no ID samples are present")
    return sum(predicted[index] == truth[index] for index in id_indices) / len(id_indices)


def binary_auroc(is_ood: Iterable[bool], ood_scores: Iterable[float]) -> float:
    """Return AUROC for detecting OOD, assigning half credit to tied scores."""
    flags, scores = _validate_detection_inputs(is_ood, ood_scores)
    positives = sum(flags)
    negatives = len(flags) - positives
    # Process equal scores as groups.  Each positive wins against every lower
    # negative and receives half credit against tied negatives.  This is the
    # Mann-Whitney definition of AUROC in O(n log n), rather than a quadratic
    # pairwise comparison over a full CIFAR-C stream.
    grouped: dict[float, tuple[int, int]] = {}
    for flag, score in zip(flags, scores):
        positive, negative = grouped.get(score, (0, 0))
        grouped[score] = (positive + int(flag), negative + int(not flag))
    wins = 0.0
    lower_negatives = 0
    for score in sorted(grouped):
        positive, negative = grouped[score]
        wins += positive * (lower_negatives + 0.5 * negative)
        lower_negatives += negative
    return wins / (positives * negatives)


def fpr_at_95_tpr(is_ood: Iterable[bool], ood_scores: Iterable[float]) -> float:
    """Return ID false-positive rate at an OOD recall target of 95%.

    The decision is OOD when ``ood_score >= threshold``.  The threshold is the
    largest distinct observed score for which recall is at least 0.95, so ties
    cannot be split by incidental input ordering.
    """
    return _fpr95_operating_point(is_ood, ood_scores)[0]


def _fpr95_operating_point(
    is_ood: Iterable[bool], ood_scores: Iterable[float]
) -> tuple[float, float, float]:
    flags, scores = _validate_detection_inputs(is_ood, ood_scores)
    positives = sum(flags)
    negatives = len(flags) - positives
    target = 0.95
    grouped: dict[float, tuple[int, int]] = {}
    for flag, score in zip(flags, scores):
        positive, negative = grouped.get(score, (0, 0))
        grouped[score] = (positive + int(flag), negative + int(not flag))
    true_positives = 0
    false_positives = 0
    for threshold in sorted(grouped, reverse=True):
        positive, negative = grouped[threshold]
        true_positives += positive
        false_positives += negative
        recall = true_positives / positives
        if recall >= target:
            return false_positives / negatives, threshold, recall
    raise AssertionError("the minimum observed score must reach full OOD recall")


def h_score(known_accuracy: float, ood_detection_recall: float) -> float:
    """Return the harmonic mean of known-class accuracy and OOD recall."""
    for value, name in ((known_accuracy, "known_accuracy"), (ood_detection_recall, "ood_detection_recall")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{name} must be a finite real number in [0, 1]")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    denominator = known_accuracy + ood_detection_recall
    return 0.0 if denominator == 0.0 else 2.0 * known_accuracy * ood_detection_recall / denominator


def open_set_metrics(
    predictions: Iterable[Any],
    ground_truth_classes: Iterable[Any],
    is_ood: Iterable[bool],
    ood_scores: Iterable[float],
) -> OpenSetMetrics:
    """Build all open-set metrics from aligned evaluator-only labels and scores."""
    predicted = list(predictions)
    truth = list(ground_truth_classes)
    flags = _bool_values(is_ood, "is_ood")
    scores = _finite_scores(ood_scores)
    if len(predicted) != len(truth) or len(truth) != len(flags) or len(flags) != len(scores):
        raise ValueError("predictions, ground_truth_classes, is_ood, and ood_scores must have equal length")
    accuracy = id_accuracy(predicted, truth, flags)
    auroc = binary_auroc(flags, scores)
    fpr95, threshold, recall = _fpr95_operating_point(flags, scores)
    return OpenSetMetrics(
        id_accuracy=accuracy,
        auroc=auroc,
        fpr_at_95_tpr=fpr95,
        fpr95_threshold=threshold,
        ood_recall_at_fpr95=recall,
        h_score=h_score(accuracy, recall),
        id_count=sum(not flag for flag in flags),
        ood_count=sum(flags),
    )


def open_set_metrics_from_trace_rows(
    trace_rows: Iterable[Mapping[str, Any]],
    *,
    known_class_ids: Iterable[Any],
    class_key: str = "ground_truth_class",
    prediction_key: str = "prediction",
    is_ood_key: str = "is_ood",
    ood_score_key: str = "ood_score",
) -> OpenSetMetrics:
    """Build metrics from trace rows and validate their known/unknown split.

    ID rows must have a class in ``known_class_ids``; OOD rows must not.  This
    catches a mismatched split manifest before it produces misleading metrics.
    """
    known_classes: set[Any] = set()
    for label in known_class_ids:
        try:
            known_classes.add(label)
        except TypeError as error:
            raise TypeError("known_class_ids labels must be hashable") from error
    if not known_classes:
        raise ValueError("known_class_ids must be non-empty")
    predictions: list[Any] = []
    truth: list[Any] = []
    flags: list[bool] = []
    scores: list[float] = []
    required = (class_key, prediction_key, is_ood_key, ood_score_key)
    for index, row in enumerate(trace_rows):
        if not isinstance(row, Mapping):
            raise TypeError("trace rows must be mappings")
        missing = [key for key in required if key not in row]
        if missing:
            raise KeyError(f"trace row {index} is missing: {', '.join(missing)}")
        label = row[class_key]
        try:
            is_known = label in known_classes
        except TypeError as error:
            raise TypeError(f"trace row {index} has an unhashable class label") from error
        flag = row[is_ood_key]
        if not isinstance(flag, bool):
            raise TypeError(f"trace row {index} has a non-boolean {is_ood_key}")
        if flag == is_known:
            expected = "outside" if flag else "inside"
            raise ValueError(f"trace row {index} class must be {expected} known_class_ids")
        predictions.append(row[prediction_key])
        truth.append(label)
        flags.append(flag)
        scores.append(row[ood_score_key])
    return open_set_metrics(predictions, truth, flags, scores)
