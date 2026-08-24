"""Pure-Python metrics for ordered test-time adaptation streams.

All accuracy values are fractions in ``[0, 1]``.  A window is contiguous in
the provided observation order; no hidden shuffling or domain labels are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class WindowAccuracy:
    """Accuracy for one contiguous evaluation window, inclusive endpoints."""

    start_timestep: int
    end_timestep: int
    accuracy: float
    count: int


def _correct_values(observations: Iterable[Any], correct_key: str) -> list[bool]:
    values: list[bool] = []
    for observation in observations:
        value = observation[correct_key] if isinstance(observation, Mapping) else observation
        if not isinstance(value, bool):
            raise TypeError(f"{correct_key} values must be booleans")
        values.append(value)
    return values


def average_accuracy(observations: Iterable[Any], *, correct_key: str = "correct") -> float:
    """Return mean correctness; raises ``ValueError`` for an empty stream."""
    values = _correct_values(observations, correct_key)
    if not values:
        raise ValueError("accuracy is undefined for an empty stream")
    return sum(values) / len(values)


def domain_accuracies(
    observations: Iterable[Mapping[str, Any]],
    *,
    domain_key: str = "ground_truth_domain",
    correct_key: str = "correct",
) -> dict[Any, float]:
    """Return per-domain accuracy, using ground-truth domain only for analysis."""
    counts: dict[Any, list[int]] = {}
    for observation in observations:
        if domain_key not in observation:
            raise KeyError(f"observation is missing {domain_key!r}")
        value = observation[correct_key]
        if not isinstance(value, bool):
            raise TypeError(f"{correct_key} values must be booleans")
        bucket = counts.setdefault(observation[domain_key], [0, 0])
        bucket[0] += int(value)
        bucket[1] += 1
    if not counts:
        raise ValueError("domain accuracy is undefined for an empty stream")
    return {domain: correct / total for domain, (correct, total) in counts.items()}


def worst_domain_accuracy(
    observations: Iterable[Mapping[str, Any]],
    *,
    domain_key: str = "ground_truth_domain",
    correct_key: str = "correct",
) -> float:
    """Return the minimum observed per-domain accuracy (macro worst case)."""
    return min(domain_accuracies(observations, domain_key=domain_key, correct_key=correct_key).values())


def sliding_window_accuracy(
    observations: Iterable[Any],
    *,
    window_size: int = 50,
    stride: int = 1,
    correct_key: str = "correct",
    timestep_key: str = "timestep",
    include_partial: bool = False,
) -> list[WindowAccuracy]:
    """Compute chronological accuracy windows.

    Defaults are a 50-sample, one-sample-stride sliding window.  Partial tail
    windows are excluded by default to keep window comparisons equally sized.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    rows = list(observations)
    correctness = _correct_values(rows, correct_key)
    result: list[WindowAccuracy] = []
    for start in range(0, len(rows), stride):
        stop = min(start + window_size, len(rows))
        if stop - start < window_size and not include_partial:
            break
        if stop == start:
            break
        start_time = rows[start].get(timestep_key, start) if isinstance(rows[start], Mapping) else start
        end_time = rows[stop - 1].get(timestep_key, stop - 1) if isinstance(rows[stop - 1], Mapping) else stop - 1
        if not isinstance(start_time, int) or not isinstance(end_time, int):
            raise TypeError("timesteps must be integers")
        values = correctness[start:stop]
        result.append(WindowAccuracy(start_time, end_time, sum(values) / len(values), len(values)))
    return result


def post_shift_recovery_time(
    observations: Iterable[Mapping[str, Any]],
    *,
    shift_timestep: int,
    baseline_accuracy: Optional[float] = None,
    window_size: int = 50,
    consecutive_windows: int = 1,
    correct_key: str = "correct",
    timestep_key: str = "timestep",
) -> Optional[int]:
    """Return samples from shift to first sustained recovery, or ``None``.

    Recovery is the first *full*, one-sample-stride post-shift window whose
    accuracy reaches ``baseline_accuracy``.  When omitted, the baseline is the
    accuracy of all observations strictly before the shift.  Consecutive
    qualifying windows (default one) prevent a transient window from counting.
    """
    if consecutive_windows <= 0:
        raise ValueError("consecutive_windows must be positive")
    rows = list(observations)
    for row in rows:
        if timestep_key not in row:
            raise KeyError(f"observation is missing {timestep_key!r}")
    if baseline_accuracy is None:
        prefix = [row for row in rows if row[timestep_key] < shift_timestep]
        baseline_accuracy = average_accuracy(prefix, correct_key=correct_key)
    if not 0.0 <= baseline_accuracy <= 1.0:
        raise ValueError("baseline_accuracy must be between 0 and 1")
    post_shift = [row for row in rows if row[timestep_key] >= shift_timestep]
    windows = sliding_window_accuracy(
        post_shift, window_size=window_size, stride=1, correct_key=correct_key,
        timestep_key=timestep_key, include_partial=False,
    )
    streak = 0
    for index, window in enumerate(windows):
        streak = streak + 1 if window.accuracy >= baseline_accuracy else 0
        if streak >= consecutive_windows:
            first = windows[index - consecutive_windows + 1]
            return first.start_timestep - shift_timestep
    return None


def negative_adaptation_rate(
    adapted_observations: Iterable[Any],
    reference_observations: Iterable[Any],
    *,
    window_size: int = 50,
    stride: Optional[int] = None,
    correct_key: str = "correct",
) -> float:
    """Fraction of aligned windows where adaptation is strictly worse.

    The reference normally is zero-shot correctness on the identical ordered
    stream.  Default stride equals the window size, yielding non-overlapping
    windows and avoiding duplicate evidence.  Ties are not negative.
    """
    if stride is None:
        stride = window_size
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    adapted = _correct_values(adapted_observations, correct_key)
    reference = _correct_values(reference_observations, correct_key)
    if len(adapted) != len(reference):
        raise ValueError("adapted and reference observations must have equal length")
    windows = 0
    negative = 0
    for start in range(0, len(adapted) - window_size + 1, stride):
        stop = start + window_size
        windows += 1
        if sum(adapted[start:stop]) < sum(reference[start:stop]):
            negative += 1
    if windows == 0:
        raise ValueError("at least one full comparison window is required")
    return negative / windows


def domain_shift_recovery_times(
    correctness: Iterable[bool],
    domains: Iterable[Any],
    *,
    window_size: int = 50,
) -> list[dict[str, Any]]:
    """Measure recovery within each persistent-domain episode.

    A shift is an adjacent domain change.  The pre-shift baseline is the final
    full ``window_size`` samples of the preceding domain episode.  Recovery is
    the first full window in the new episode whose accuracy reaches that
    baseline.  Episodes shorter than a full window are reported as
    ``insufficient_episode`` rather than silently pooled across another shift.
    """
    values = list(correctness)
    domain_values = list(domains)
    if len(values) != len(domain_values):
        raise ValueError("correctness and domains must have equal length")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("correctness values must be booleans")

    boundaries = [index for index in range(1, len(domain_values))
                  if domain_values[index] != domain_values[index - 1]]
    episode_starts = [0, *boundaries]
    episode_ends = [*boundaries, len(domain_values)]
    results = []
    for episode_index in range(1, len(episode_starts)):
        shift = episode_starts[episode_index]
        previous_start = episode_starts[episode_index - 1]
        episode_end = episode_ends[episode_index]
        result = {
            "shift_timestep": shift,
            "from_domain": domain_values[shift - 1],
            "to_domain": domain_values[shift],
            "recovery_samples": None,
        }
        if shift - previous_start < window_size or episode_end - shift < window_size:
            result["status"] = "insufficient_episode"
            results.append(result)
            continue
        baseline = sum(values[shift - window_size:shift]) / window_size
        result["baseline_accuracy"] = baseline
        result["status"] = "not_recovered"
        for start in range(shift, episode_end - window_size + 1):
            accuracy = sum(values[start:start + window_size]) / window_size
            if accuracy >= baseline:
                result["status"] = "recovered"
                result["recovery_samples"] = start - shift
                break
        results.append(result)
    return results
