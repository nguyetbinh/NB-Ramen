"""Dependency-free diagnostics for latent-context routing.

Ground-truth domains are accepted solely as evaluation labels.  None of the
functions in this module performs routing or exposes labels to a router.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class RoutingDiagnostics:
    """Alignment and stability metrics for one ordered routing stream.

    ``status`` is ``"unavailable"`` when a method did not emit inferred
    contexts.  In that case every metric is ``None`` rather than a misleading
    numeric value.
    """

    status: str
    normalized_mutual_information: Optional[float]
    adjusted_rand_index: Optional[float]
    context_purity: Optional[float]
    number_of_discovered_contexts: Optional[int]
    assignment_churn_rate: Optional[float]


def _labels(values: Iterable[Any], name: str) -> list[Any]:
    labels = list(values)
    if not labels:
        raise ValueError(f"{name} must be non-empty")
    for label in labels:
        try:
            hash(label)
        except TypeError as error:
            raise TypeError(f"{name} labels must be hashable") from error
    return labels


def _aligned_labels(
    ground_truth_domains: Iterable[Any], inferred_contexts: Iterable[Any]
) -> tuple[list[Any], list[Any]]:
    truth = _labels(ground_truth_domains, "ground_truth_domains")
    inferred = _labels(inferred_contexts, "inferred_contexts")
    if len(truth) != len(inferred):
        raise ValueError("ground_truth_domains and inferred_contexts must have equal length")
    return truth, inferred


def _contingency(truth: list[Any], inferred: list[Any]) -> tuple[Counter[Any], Counter[Any], Counter[tuple[Any, Any]]]:
    return Counter(truth), Counter(inferred), Counter(zip(truth, inferred))


def normalized_mutual_information(
    ground_truth_domains: Iterable[Any], inferred_contexts: Iterable[Any]
) -> float:
    """Return geometric-mean normalized mutual information in ``[0, 1]``.

    A pair of constant partitions has NMI 1.0 because they contain exactly the
    same (lack of) partition information; only one constant partition has NMI
    0.0.  The value is invariant to arbitrary context-ID permutations.
    """
    truth, inferred = _aligned_labels(ground_truth_domains, inferred_contexts)
    truth_counts, inferred_counts, cells = _contingency(truth, inferred)
    count = len(truth)

    def entropy(counts: Counter[Any]) -> float:
        return -sum((size / count) * log(size / count) for size in counts.values())

    truth_entropy = entropy(truth_counts)
    inferred_entropy = entropy(inferred_counts)
    if truth_entropy == 0.0 or inferred_entropy == 0.0:
        return 1.0 if truth_entropy == 0.0 and inferred_entropy == 0.0 else 0.0

    mutual_information = sum(
        (cell_count / count)
        * log((cell_count * count) / (truth_counts[domain] * inferred_counts[context]))
        for (domain, context), cell_count in cells.items()
    )
    return min(1.0, max(0.0, mutual_information / sqrt(truth_entropy * inferred_entropy)))


def adjusted_rand_index(
    ground_truth_domains: Iterable[Any], inferred_contexts: Iterable[Any]
) -> float:
    """Return the chance-adjusted Rand index in ``[-1, 1]``.

    The implementation uses the standard pair-counting definition.  For the
    degenerate zero-pair case (one sample), two identical partitions score 1.
    """
    truth, inferred = _aligned_labels(ground_truth_domains, inferred_contexts)
    truth_counts, inferred_counts, cells = _contingency(truth, inferred)

    def choose_two(value: int) -> int:
        return value * (value - 1) // 2

    total_pairs = choose_two(len(truth))
    if total_pairs == 0:
        return 1.0
    cell_pairs = sum(choose_two(value) for value in cells.values())
    truth_pairs = sum(choose_two(value) for value in truth_counts.values())
    inferred_pairs = sum(choose_two(value) for value in inferred_counts.values())
    expected = truth_pairs * inferred_pairs / total_pairs
    maximum = (truth_pairs + inferred_pairs) / 2
    denominator = maximum - expected
    if denominator == 0.0:
        # Both partitions either place every pair together or every pair apart.
        return 1.0 if cell_pairs == maximum else 0.0
    return min(1.0, max(-1.0, (cell_pairs - expected) / denominator))


def context_purity(
    ground_truth_domains: Iterable[Any], inferred_contexts: Iterable[Any]
) -> float:
    """Return majority-domain purity averaged over samples, in ``[0, 1]``."""
    truth, inferred = _aligned_labels(ground_truth_domains, inferred_contexts)
    per_context: dict[Any, Counter[Any]] = defaultdict(Counter)
    for domain, context in zip(truth, inferred):
        per_context[context][domain] += 1
    return sum(max(domains.values()) for domains in per_context.values()) / len(truth)


def number_of_discovered_contexts(inferred_contexts: Iterable[Any]) -> int:
    """Return the number of distinct inferred-context labels."""
    return len(set(_labels(inferred_contexts, "inferred_contexts")))


def assignment_churn_rate(inferred_contexts: Iterable[Any]) -> float:
    """Return adjacent assignment changes divided by adjacent pairs.

    A one-sample stream has no adjacent pair and therefore a churn rate of 0.
    """
    contexts = _labels(inferred_contexts, "inferred_contexts")
    if len(contexts) == 1:
        return 0.0
    return sum(left != right for left, right in zip(contexts, contexts[1:])) / (len(contexts) - 1)


def routing_diagnostics(
    ground_truth_domains: Iterable[Any], inferred_contexts: Optional[Iterable[Any]] = None
) -> RoutingDiagnostics:
    """Evaluate inferred contexts, or report an explicit unavailable status.

    ``None`` and an all-``None`` context sequence both represent a method that
    did not emit contexts (as baseline trace rows do).  A partially missing
    sequence is rejected because it cannot support a well-defined alignment.
    """
    truth = _labels(ground_truth_domains, "ground_truth_domains")
    if inferred_contexts is None:
        return RoutingDiagnostics("unavailable", None, None, None, None, None)

    inferred = list(inferred_contexts)
    if inferred and all(context is None for context in inferred):
        if len(truth) != len(inferred):
            raise ValueError("ground_truth_domains and inferred_contexts must have equal length")
        return RoutingDiagnostics("unavailable", None, None, None, None, None)
    if any(context is None for context in inferred):
        raise ValueError("inferred_contexts must be fully present or fully absent")
    truth, inferred = _aligned_labels(truth, inferred)
    return RoutingDiagnostics(
        "available",
        normalized_mutual_information(truth, inferred),
        adjusted_rand_index(truth, inferred),
        context_purity(truth, inferred),
        number_of_discovered_contexts(inferred),
        assignment_churn_rate(inferred),
    )
