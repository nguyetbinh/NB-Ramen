"""Build reproducible heterogeneous evaluation streams without reading images.

``build_stream`` returns a map-style dataset compatible with ``DataLoader``.
Its order is a schedule of ``(domain_idx, sample_idx)`` references; source
datasets are only indexed later by ``StreamDataset.__getitem__``.  The emitted
``domain_idx`` is evaluation metadata, never an input to scheduling after the
schedule is built and never a routing signal for a TTA method.

Defaults are intentionally conservative: streams other than ``imbalanced``
include every source sample once, ``block_size`` is 64, and gradual transitions
use a moderate sharpness of 4.  ``imbalanced`` creates a deterministic long-tail
subset from Zipf-like domain quotas when no explicit sample budget is supplied.
``truncate_stream`` is a separate deterministic evaluation-prefix budget for
cost-limited evidence runs.
"""

from __future__ import annotations

import hashlib
import json
import random
from fractions import Fraction

from . import schedules

_MODES = {
    "iid_mixed", "block", "gradual", "recurring", "imbalanced",
    "novel_domain", "class_domain_correlated", "bursty",
}


class StreamDataset:
    """Lazy dataset view over a deterministic, serializable schedule."""

    def __init__(self, datasets, references, metadata, *, evaluator_metadata=None):
        self.datasets = tuple(datasets)
        self.references = tuple((int(domain), int(sample)) for domain, sample in references)
        self.metadata = dict(metadata)
        self.metadata["num_samples"] = len(self.references)
        self.fingerprint = stream_fingerprint({
            "metadata": self.metadata,
            "references": self.references,
        })
        self.metadata["fingerprint"] = self.fingerprint
        if evaluator_metadata is None:
            self._evaluator_metadata = None
        else:
            metadata_by_reference = {
                (int(domain), int(sample)): dict(value)
                for (domain, sample), value in evaluator_metadata.items()
            }
            if set(metadata_by_reference) != set(self.references):
                raise ValueError("evaluator_metadata must contain exactly one mapping per stream reference")
            self._evaluator_metadata = metadata_by_reference

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        domain_idx, sample_idx = self.references[index]
        item = self.datasets[domain_idx][sample_idx]
        if isinstance(item, tuple):
            result = (*item, domain_idx, sample_idx)
        else:
            result = (item, domain_idx, sample_idx)
        if self._evaluator_metadata is not None:
            # This mapping is a separate evaluator-only tail, never part of
            # the source image/label contract consumed by a TTA method.
            return (*result, dict(self._evaluator_metadata[(domain_idx, sample_idx)]))
        return result

    def to_dict(self):
        """Return a JSON-serializable representation of the schedule."""
        return {
            "metadata": dict(self.metadata),
            "references": [list(reference) for reference in self.references],
            "fingerprint": self.fingerprint,
        }


def stream_fingerprint(payload):
    """Hash the canonical stream payload, excluding self-referential digests."""
    metadata = dict(payload["metadata"])
    metadata.pop("fingerprint", None)
    canonical = {
        "metadata": metadata,
        "references": [list(reference) for reference in payload["references"]],
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_stream_fingerprint(payload):
    """Return whether an exported ``stream.json`` payload is internally valid."""
    expected = payload.get("fingerprint") or payload.get("metadata", {}).get("fingerprint")
    return isinstance(expected, str) and stream_fingerprint(payload) == expected


def truncate_stream(stream, max_samples):
    """Return a deterministic cost-limited prefix of an existing stream.

    This deliberately applies *after* stream scheduling.  Unlike the
    imbalanced stream's ``sample_budget``, it never changes which samples are
    selected or their order: it retains the first ``max_samples`` references.
    Source datasets remain lazy and are not read while truncating.
    """
    if not isinstance(stream, StreamDataset):
        raise TypeError("stream must be a StreamDataset")
    full_sample_count = len(stream)
    if not isinstance(max_samples, int) or isinstance(max_samples, bool):
        raise TypeError("max_samples must be a positive integer")
    if not 0 < max_samples <= full_sample_count:
        raise ValueError("max_samples must be between 1 and the full stream sample count")

    metadata = dict(stream.metadata)
    metadata["evaluation_budget"] = {
        "truncation_strategy": "deterministic_prefix",
        "evidence_scope": "cost_limited",
        "cost_limited_evidence": True,
        "full_sample_count": full_sample_count,
        "full_stream_fingerprint": stream.fingerprint,
        "retained_sample_count": max_samples,
        "dropped_sample_count": full_sample_count - max_samples,
    }
    evaluator_metadata = None
    if stream._evaluator_metadata is not None:
        retained = set(stream.references[:max_samples])
        evaluator_metadata = {
            reference: value for reference, value in stream._evaluator_metadata.items()
            if reference in retained
        }
        _refresh_open_set_realized_counts(metadata, stream.references[:max_samples], evaluator_metadata)
    return StreamDataset(stream.datasets, stream.references[:max_samples], metadata,
                         evaluator_metadata=evaluator_metadata)


def _refresh_open_set_realized_counts(metadata, references, evaluator_metadata):
    """Bind reported OOD counts to the actual emitted (possibly truncated) stream."""
    open_set = metadata.get("open_set")
    if not isinstance(open_set, dict):
        return
    per_domain = {}
    for domain_idx, sample_idx in references:
        value = evaluator_metadata[(domain_idx, sample_idx)]
        counts = per_domain.setdefault(domain_idx, {"known": 0, "ood": 0, "total": 0})
        counts["ood" if value["is_ood"] else "known"] += 1
        counts["total"] += 1
    ordered = [per_domain.get(index, {"known": 0, "ood": 0, "total": 0})
               for index in range(len(metadata.get("domain_lengths", [])))]
    known = sum(item["known"] for item in ordered)
    ood = sum(item["ood"] for item in ordered)
    open_set["per_domain_counts"] = ordered
    open_set["realized_ood_count"] = ood
    open_set["realized_known_count"] = known
    open_set["realized_ood_ratio"] = ood / (known + ood) if known + ood else None


def _open_set_split_from(multi_datasets):
    version = getattr(multi_datasets, "open_set_split_version", None)
    known = getattr(multi_datasets, "known_class_ids", None)
    unknown = getattr(multi_datasets, "unknown_class_ids", None)
    if not isinstance(version, str) or not version:
        raise ValueError("open-set stream requires a dataset with open_set_split_version")
    if not isinstance(known, (list, tuple)) or not isinstance(unknown, (list, tuple)):
        raise ValueError("open-set stream requires known_class_ids and unknown_class_ids")
    known_ids, unknown_ids = tuple(int(value) for value in known), tuple(int(value) for value in unknown)
    if not known_ids or not unknown_ids or set(known_ids).intersection(unknown_ids):
        raise ValueError("open-set split must contain disjoint known and unknown IDs")
    identifiers = {}
    for attribute, metadata_key in (
        ("open_set_split_fingerprint", "split_fingerprint"),
        ("open_set_taxonomy_sha256", "taxonomy_sha256"),
    ):
        value = getattr(multi_datasets, attribute, None)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"open-set dataset {attribute} must be a non-empty string when provided")
        identifiers[metadata_key] = value
    return version, known_ids, unknown_ids, identifiers


def _exact_open_set_selection(labels, known_ids, unknown_ids, ratio, rng):
    """Select the largest exact-ratio subset without decoding source samples."""
    known_set, unknown_set = set(known_ids), set(unknown_ids)
    if any(label not in known_set.union(unknown_set) for label in labels):
        raise ValueError("open-set stream labels must belong to the declared split")
    known_indices = [index for index, label in enumerate(labels) if label in known_set]
    unknown_indices = [index for index, label in enumerate(labels) if label in unknown_set]
    numerator, denominator = ratio.numerator, ratio.denominator
    if numerator == 0:
        selected_known, selected_unknown = known_indices, []
    elif numerator == denominator:
        selected_known, selected_unknown = [], unknown_indices
    else:
        multiplier = min(len(unknown_indices) // numerator, len(known_indices) // (denominator - numerator))
        if multiplier == 0:
            raise ValueError("requested ood_ratio is infeasible for at least one domain")
        unknown_count, known_count = numerator * multiplier, (denominator - numerator) * multiplier
        rng.shuffle(known_indices)
        rng.shuffle(unknown_indices)
        selected_known, selected_unknown = known_indices[:known_count], unknown_indices[:unknown_count]
    selected = selected_known + selected_unknown
    rng.shuffle(selected)
    return selected, len(selected_known), len(selected_unknown)


def _balanced_open_set_selection(labels, class_ids, count, rng):
    """Select a deterministic, as-even-as-possible prefix across classes.

    Each class has one independently shuffled pool.  Round-robin selection
    makes allocations differ by at most one when capacities allow it, and its
    prefix property keeps smaller requested allocations nested in larger ones.
    """
    pools = {class_id: [] for class_id in sorted(class_ids)}
    for index, label in enumerate(labels):
        if label in pools:
            pools[label].append(index)
    for pool in pools.values():
        rng.shuffle(pool)

    selected, offsets = [], {class_id: 0 for class_id in pools}
    while len(selected) < count:
        progressed = False
        for class_id in pools:
            offset = offsets[class_id]
            if offset >= len(pools[class_id]):
                continue
            selected.append(pools[class_id][offset])
            offsets[class_id] = offset + 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError("requested per_domain_source_budget is infeasible for at least one domain")
    return selected


def _budgeted_open_set_selection(labels, known_ids, unknown_ids, ratio, budget, rng):
    """Select one exact, class-balanced source pool of ``budget`` examples."""
    known_set, unknown_set = set(known_ids), set(unknown_ids)
    if any(label not in known_set.union(unknown_set) for label in labels):
        raise ValueError("open-set stream labels must belong to the declared split")
    if budget % ratio.denominator:
        raise ValueError(
            "per_domain_source_budget must be divisible by the requested ood_ratio denominator"
        )
    unknown_count = budget * ratio.numerator // ratio.denominator
    known_count = budget - unknown_count
    known = _balanced_open_set_selection(labels, known_ids, known_count, rng)
    unknown = _balanced_open_set_selection(labels, unknown_ids, unknown_count, rng)
    selected = known + unknown
    rng.shuffle(selected)
    return selected, known_count, unknown_count


def build_open_set_stream(multi_datasets, mode, seed, *, ood_ratio, domain_weights=None,
                          block_size=64, gradual_sharpness=4.0, sample_budget=None,
                          per_domain_source_budget=None,
                          novel_domain_idx=None, novel_release_fraction=0.5,
                          correlation_strength=0.9, burst_size=None):
    """Build a deterministic open-set stream with an exact OOD ratio per domain.

    The ratio is represented by the caller's decimal string and honored exactly
    whenever each domain has enough known and unknown examples.  Selection is
    metadata-only; images remain lazy until ``StreamDataset.__getitem__``.
    """
    if not isinstance(ood_ratio, (int, float)) or isinstance(ood_ratio, bool) or not 0.0 <= float(ood_ratio) <= 1.0:
        raise ValueError("ood_ratio must be a finite value between 0 and 1")
    ratio = Fraction(str(ood_ratio))
    if per_domain_source_budget is not None:
        if (not isinstance(per_domain_source_budget, int)
                or isinstance(per_domain_source_budget, bool)
                or per_domain_source_budget <= 0):
            raise ValueError("per_domain_source_budget must be a positive integer")
        if per_domain_source_budget % ratio.denominator:
            raise ValueError(
                "per_domain_source_budget must be divisible by the requested ood_ratio denominator"
            )
    version, known_ids, unknown_ids, split_identifiers = _open_set_split_from(multi_datasets)
    datasets, _ = _datasets_from(multi_datasets)

    selected_by_domain, selected_counts = [], []
    for domain_idx, dataset in enumerate(datasets):
        labels = _labels_from_metadata(dataset)
        rng = random.Random(seed + domain_idx)
        if per_domain_source_budget is None:
            selected, known_count, unknown_count = _exact_open_set_selection(
                labels, known_ids, unknown_ids, ratio, rng
            )
        else:
            selected, known_count, unknown_count = _budgeted_open_set_selection(
                labels, known_ids, unknown_ids, ratio, per_domain_source_budget, rng
            )
        selected_by_domain.append(selected)
        selected_counts.append({"known": known_count, "ood": unknown_count, "total": len(selected)})

    class _SelectedDataset:
        def __init__(self, labels):
            self.targets = labels

        def __len__(self):
            return len(self.targets)

    class _SelectedDomains:
        def __init__(self):
            self.datasets = tuple(
                _SelectedDataset([_labels_from_metadata(source)[sample] for sample in selected])
                for source, selected in zip(datasets, selected_by_domain)
            )
            self.environments = getattr(multi_datasets, "environments", None)

    scheduled = build_stream(
        _SelectedDomains(), mode, seed, domain_weights, block_size,
        gradual_sharpness=gradual_sharpness, sample_budget=sample_budget,
        novel_domain_idx=novel_domain_idx, novel_release_fraction=novel_release_fraction,
        correlation_strength=correlation_strength, burst_size=burst_size,
    )
    references = [
        (domain_idx, selected_by_domain[domain_idx][selected_index])
        for domain_idx, selected_index in scheduled.references
    ]
    known_label_by_original = {original: known for known, original in enumerate(known_ids)}
    evaluator_metadata = {}
    for domain_idx, sample_idx in references:
        original_label = _labels_from_metadata(datasets[domain_idx])[sample_idx]
        known_label = known_label_by_original.get(original_label, -1)
        evaluator_metadata[(domain_idx, sample_idx)] = {
            "original_label": original_label,
            "known_label_or_minus_one": known_label,
            "is_ood": known_label == -1,
        }
    metadata = dict(scheduled.metadata)
    metadata["open_set"] = {
        "split_version": version,
        "known_class_ids": list(known_ids),
        "unknown_class_ids": list(unknown_ids),
        **split_identifiers,
        "requested_ood_ratio": float(ood_ratio),
        "requested_ood_ratio_fraction": f"{ratio.numerator}/{ratio.denominator}",
        "requested_per_domain_source_budget": per_domain_source_budget,
        "realized_per_domain_source_budget": [item["total"] for item in selected_counts],
        "selected_pool_per_domain_counts": selected_counts,
    }
    _refresh_open_set_realized_counts(metadata, references, evaluator_metadata)
    return StreamDataset(datasets, references, metadata, evaluator_metadata=evaluator_metadata)


def _datasets_from(multi_datasets):
    datasets = getattr(multi_datasets, "datasets", multi_datasets)
    try:
        datasets = tuple(datasets)
    except TypeError as exc:
        raise TypeError("multi_datasets must be iterable or expose a datasets attribute") from exc
    if not datasets:
        raise ValueError("multi_datasets must contain at least one domain dataset")
    lengths = tuple(len(dataset) for dataset in datasets)
    if any(length <= 0 for length in lengths):
        raise ValueError("each domain dataset must contain at least one sample")
    return datasets, lengths


def _labels_from_metadata(dataset):
    for attribute in ("targets", "labels", "Y"):
        labels = getattr(dataset, attribute, None)
        if labels is not None and len(labels) == len(dataset):
            return [int(label) for label in labels]
    samples = getattr(dataset, "samples", None)
    if samples is not None and len(samples) == len(dataset):
        try:
            return [int(sample[1]) for sample in samples]
        except (IndexError, TypeError, ValueError):
            pass
    raise ValueError(
        "class_domain_correlated requires non-materializing label metadata "
        "via dataset.targets, dataset.labels, dataset.Y, or dataset.samples"
    )


def _label_pools(datasets):
    pools = []
    for dataset in datasets:
        labels = _labels_from_metadata(dataset)
        per_label = {}
        for sample_idx, label in enumerate(labels):
            per_label.setdefault(label, []).append(sample_idx)
        pools.append(per_label)
    return pools


def build_stream(multi_datasets, mode, seed, domain_weights=None, block_size=64,
                 *, gradual_sharpness=4.0, sample_budget=None,
                 novel_domain_idx=None, novel_release_fraction=0.5,
                 correlation_strength=0.9,
                 burst_size=None):
    """Build a lazy deterministic stream for a heterogeneous TTA evaluation.

    ``domain_weights`` controls choices among currently available domains.  It
    defaults to equal weights except for ``imbalanced``, where ``1/(i+1)`` is
    used.  All other modes use each sample once.  ``domain_idx`` remains only
    output metadata.
    """
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("mode must be one of: " + ", ".join(sorted(_MODES)))
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(gradual_sharpness, (int, float)) or gradual_sharpness <= 0:
        raise ValueError("gradual_sharpness must be positive")
    if not 0.0 < correlation_strength < 1.0:
        raise ValueError("correlation_strength must be strictly between 0 and 1")
    if not isinstance(novel_release_fraction, (int, float)) or not 0.0 < novel_release_fraction < 1.0:
        raise ValueError("novel_release_fraction must be strictly between 0 and 1")
    if burst_size is not None and (not isinstance(burst_size, int) or burst_size <= 0):
        raise ValueError("burst_size must be a positive integer")

    datasets, lengths = _datasets_from(multi_datasets)
    if sample_budget is not None:
        if mode != "imbalanced":
            raise ValueError("sample_budget is supported only by imbalanced streams")
        if not isinstance(sample_budget, int) or not 0 < sample_budget <= sum(lengths):
            raise ValueError("sample_budget must be between 1 and the total sample count")
    if novel_domain_idx is None:
        novel_domain_idx = len(datasets) - 1
    if not isinstance(novel_domain_idx, int) or not 0 <= novel_domain_idx < len(datasets):
        raise ValueError("novel_domain_idx must identify an existing domain")

    if domain_weights is None and mode == "imbalanced":
        weights = tuple(1.0 / (idx + 1) for idx in range(len(datasets)))
    else:
        weights = schedules.validate_weights(domain_weights, len(datasets))
    rng = random.Random(seed)

    if mode == "iid_mixed":
        references = schedules.iid_mixed(lengths, rng, weights)
    elif mode == "block":
        references = schedules.block(lengths, rng, weights, block_size)
    elif mode == "recurring":
        references = schedules.block(lengths, rng, weights, block_size, recurring=True)
    elif mode == "gradual":
        references = schedules.gradual(lengths, rng, weights, float(gradual_sharpness))
    elif mode == "imbalanced":
        references = schedules.imbalanced(lengths, rng, weights, sample_budget)
    elif mode == "novel_domain":
        release_at = round(sum(lengths) * float(novel_release_fraction))
        references = schedules.novel_domain(lengths, rng, weights, novel_domain_idx, release_at)
    elif mode == "class_domain_correlated":
        references = schedules.class_domain_correlated(
            _label_pools(datasets), rng, weights, float(correlation_strength))
    else:
        references = schedules.bursty(lengths, rng, weights, burst_size or max(1, block_size // 4))

    environments = getattr(multi_datasets, "environments", None)
    metadata = {
        "format_version": 1,
        "mode": mode,
        "seed": seed,
        "domain_lengths": list(lengths),
        "domain_weights": list(weights),
        "block_size": block_size,
        "domain_names": list(environments) if environments is not None else None,
        "parameters": {
            "gradual_sharpness": gradual_sharpness,
            "sample_budget": sample_budget,
            "novel_domain_idx": novel_domain_idx if mode == "novel_domain" else None,
            "novel_release_fraction": novel_release_fraction if mode == "novel_domain" else None,
            "novel_release_timestep": (
                next(index for index, (domain, _) in enumerate(references) if domain == novel_domain_idx)
                if mode == "novel_domain" else None
            ),
            "correlation_strength": correlation_strength if mode == "class_domain_correlated" else None,
            "burst_size": burst_size or max(1, block_size // 4) if mode == "bursty" else None,
        },
    }
    return StreamDataset(datasets, references, metadata)


def build_single_domain_stream(multi_datasets, seed):
    """Build independently shuffled domain segments for legacy single-domain TTA."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    datasets, lengths = _datasets_from(multi_datasets)
    references = []
    segments = []
    for domain_idx, length in enumerate(lengths):
        sample_indices = list(range(length))
        random.Random(seed + domain_idx).shuffle(sample_indices)
        start = len(references)
        references.extend((domain_idx, sample_idx) for sample_idx in sample_indices)
        segments.append((start, len(references)))
    environments = getattr(multi_datasets, "environments", None)
    stream = StreamDataset(
        datasets,
        references,
        {
            "format_version": 1,
            "mode": "single_domain",
            "seed": seed,
            "domain_lengths": list(lengths),
            "domain_names": list(environments) if environments is not None else None,
            "parameters": {"reset_at_domain_boundaries": True},
        },
    )
    return stream, tuple(segments)
