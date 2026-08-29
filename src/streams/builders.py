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

from . import schedules

_MODES = {
    "iid_mixed", "block", "gradual", "recurring", "imbalanced",
    "novel_domain", "class_domain_correlated", "bursty",
}


class StreamDataset:
    """Lazy dataset view over a deterministic, serializable schedule."""

    def __init__(self, datasets, references, metadata):
        self.datasets = tuple(datasets)
        self.references = tuple((int(domain), int(sample)) for domain, sample in references)
        self.metadata = dict(metadata)
        self.metadata["num_samples"] = len(self.references)
        self.fingerprint = stream_fingerprint({
            "metadata": self.metadata,
            "references": self.references,
        })
        self.metadata["fingerprint"] = self.fingerprint

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        domain_idx, sample_idx = self.references[index]
        item = self.datasets[domain_idx][sample_idx]
        if isinstance(item, tuple):
            return (*item, domain_idx, sample_idx)
        return item, domain_idx, sample_idx

    def evaluator_metadata(self, index):
        """Private evaluator join; never part of a DataLoader batch given to methods."""
        domain_idx, sample_idx = self.references[index]
        getter = getattr(self.datasets[domain_idx], "evaluator_metadata", None)
        return getter(sample_idx) if callable(getter) else None

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
    return StreamDataset(stream.datasets, stream.references[:max_samples], metadata)


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


def _label_pools(datasets, source_indices=None):
    """Return label pools keyed by the source index used in references."""
    pools = []
    for domain_idx, dataset in enumerate(datasets):
        labels = _labels_from_metadata(dataset)
        per_label = {}
        indices = range(len(labels)) if source_indices is None else source_indices[domain_idx]
        for sample_idx in indices:
            label = labels[sample_idx]
            per_label.setdefault(label, []).append(sample_idx)
        pools.append(per_label)
    return pools


def build_stream(multi_datasets, mode, seed, domain_weights=None, block_size=64,
                 *, gradual_sharpness=4.0, sample_budget=None,
                 novel_domain_idx=None, novel_release_fraction=0.5,
                 correlation_strength=0.9,
                 burst_size=None, ood_ratio=None, open_set_split_version=None):
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
    if ood_ratio is not None:
        if ood_ratio not in {0, 0.1, 0.3, 0.5}:
            raise ValueError("ood_ratio must be one of 0, 0.1, 0.3, 0.5")
        selected = []
        for dataset in datasets:
            getter = getattr(dataset, "evaluator_metadata", None)
            if not callable(getter):
                raise ValueError("ood_ratio requires an open-set dataset")
            known = [i for i in range(len(dataset)) if not getter(i)["is_ood"]]
            unknown = [i for i in range(len(dataset)) if getter(i)["is_ood"]]
            if ood_ratio == 0:
                chosen = known
            else:
                # Ratios are preregistered finite decimals: choose the largest exact subset.
                denominator = {0.1: 10, 0.3: 10, 0.5: 2}[ood_ratio]
                unknown_per_group = int(ood_ratio * denominator)
                groups = min(len(unknown) // unknown_per_group, len(known) // (denominator - unknown_per_group))
                chosen = known[:groups * (denominator - unknown_per_group)] + unknown[:groups * unknown_per_group]
            selected.append(chosen)
        source_indices = tuple(tuple(indices) for indices in selected)
        lengths = tuple(len(indices) for indices in source_indices)
        if any(length == 0 for length in lengths):
            raise ValueError("open-set ratio leaves an empty domain")
    else:
        source_indices = None
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
            _label_pools(datasets, source_indices), rng, weights, float(correlation_strength))
    else:
        references = schedules.bursty(lengths, rng, weights, burst_size or max(1, block_size // 4))
    # This schedule consumes pool values directly; all other schedules use
    # compact filtered positions and need translating to source IDs.
    if source_indices is not None and mode != "class_domain_correlated":
        references = [(domain, source_indices[domain][sample]) for domain, sample in references]

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
            "ood_ratio": ood_ratio,
            "open_set_split_version": open_set_split_version,
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
