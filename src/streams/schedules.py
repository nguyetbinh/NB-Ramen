"""Pure-Python schedule algorithms used by :mod:`streams.builders`.

The functions in this module operate exclusively on integer sample references.
They deliberately never access ``dataset[index]``: decoding images belongs to
the data loader, not stream construction.
"""

from __future__ import annotations

import math
import random


def validate_weights(domain_weights, num_domains):
    if domain_weights is None:
        return tuple(1.0 for _ in range(num_domains))
    if len(domain_weights) != num_domains:
        raise ValueError("domain_weights must have one value per domain")
    weights = tuple(float(value) for value in domain_weights)
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("domain_weights must contain only finite positive values")
    return weights


def shuffled_pools(domain_lengths, rng):
    pools = []
    for domain_idx, length in enumerate(domain_lengths):
        pool = list(range(length))
        rng.shuffle(pool)
        pools.append(pool)
    return pools


def _weighted_choice(candidates, weights, rng):
    total = sum(weights)
    if total <= 0:
        raise ValueError("at least one scheduling weight must be positive")
    target = rng.random() * total
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += weight
        if target < cumulative:
            return candidate
    return candidates[-1]  # floating-point rounding at the upper boundary


def _remaining_domains(pools, positions):
    return [idx for idx, pool in enumerate(pools) if positions[idx] < len(pool)]


def _take(pools, positions, domain_idx):
    sample_idx = pools[domain_idx][positions[domain_idx]]
    positions[domain_idx] += 1
    return (domain_idx, sample_idx)


def iid_mixed(domain_lengths, rng, weights):
    if len(set(weights)) == 1:
        references = [(domain_idx, sample_idx)
                      for domain_idx, length in enumerate(domain_lengths)
                      for sample_idx in range(length)]
        rng.shuffle(references)
        return references
    pools = shuffled_pools(domain_lengths, rng)
    positions = [0] * len(pools)
    references = []
    while (active := _remaining_domains(pools, positions)):
        domain_idx = _weighted_choice(active, [weights[idx] for idx in active], rng)
        references.append(_take(pools, positions, domain_idx))
    return references


def block(domain_lengths, rng, weights, block_size, recurring=False):
    pools = shuffled_pools(domain_lengths, rng)
    positions = [0] * len(pools)
    references = []
    order = []
    remaining = list(range(len(pools)))
    while remaining:
        selected = _weighted_choice(remaining, [weights[idx] for idx in remaining], rng)
        order.append(selected)
        remaining.remove(selected)
    cursor = 0
    last_domain = None
    while (active := _remaining_domains(pools, positions)):
        if recurring:
            domain_idx = order[cursor % len(order)]
            cursor += 1
            if domain_idx not in active:
                continue
        else:
            candidates = [idx for idx in active if idx != last_domain] or active
            domain_idx = _weighted_choice(candidates, [weights[idx] for idx in candidates], rng)
        for _ in range(min(block_size, len(pools[domain_idx]) - positions[domain_idx])):
            references.append(_take(pools, positions, domain_idx))
        last_domain = domain_idx
    return references


def gradual(domain_lengths, rng, weights, sharpness):
    pools = shuffled_pools(domain_lengths, rng)
    positions = [0] * len(pools)
    references = []
    total = sum(domain_lengths)
    num_domains = len(pools)
    for step in range(total):
        active = _remaining_domains(pools, positions)
        progress = step / max(1, total - 1)
        center = progress * max(0, num_domains - 1)
        local_weights = [weights[idx] * math.exp(-sharpness * abs(idx - center))
                         for idx in active]
        domain_idx = _weighted_choice(active, local_weights, rng)
        references.append(_take(pools, positions, domain_idx))
    return references


def imbalanced(domain_lengths, rng, weights, sample_budget=None):
    pools = shuffled_pools(domain_lengths, rng)
    positions = [0] * len(pools)
    if sample_budget is None:
        max_weight = max(weights)
        quotas = [max(1, min(length, round(length * weight / max_weight)))
                  for length, weight in zip(domain_lengths, weights)]
        budget = sum(quotas)
    else:
        quotas = list(domain_lengths)
        budget = sample_budget
    references = []
    for _ in range(budget):
        active = [idx for idx, quota in enumerate(quotas) if positions[idx] < quota]
        if not active:
            break
        domain_idx = _weighted_choice(active, [weights[idx] for idx in active], rng)
        references.append(_take(pools, positions, domain_idx))
    return references


def novel_domain(domain_lengths, rng, weights, novel_domain_idx, release_at):
    pools = shuffled_pools(domain_lengths, rng)
    positions = [0] * len(pools)
    total = sum(domain_lengths)
    non_novel_count = total - domain_lengths[novel_domain_idx]
    if non_novel_count < release_at:
        raise ValueError(
            "novel domain cannot be held until the requested release timestep: "
            f"only {non_novel_count} non-novel samples are available for {release_at} slots"
        )
    references = []
    for step in range(total):
        active = _remaining_domains(pools, positions)
        eligible = [idx for idx in active if step >= release_at or idx != novel_domain_idx]
        domain_idx = _weighted_choice(eligible, [weights[idx] for idx in eligible], rng)
        references.append(_take(pools, positions, domain_idx))
    return references


def class_domain_correlated(label_pools, rng, weights, correlation_strength):
    """Sample without replacement while favoring domain ``class % num_domains``."""
    num_domains = len(label_pools)
    positions = {domain_idx: {label: 0 for label in labels}
                 for domain_idx, labels in enumerate(label_pools)}
    for labels in label_pools:
        for samples in labels.values():
            rng.shuffle(samples)
    references = []
    while True:
        available_labels = sorted({label for domain_idx, labels in enumerate(label_pools)
                                   for label, samples in labels.items()
                                   if positions[domain_idx][label] < len(samples)})
        if not available_labels:
            return references
        label = available_labels[rng.randrange(len(available_labels))]
        domains = [idx for idx, labels in enumerate(label_pools)
                   if label in labels and positions[idx][label] < len(labels[label])]
        preferred = label % num_domains
        local_weights = [weights[idx] * (correlation_strength if idx == preferred else 1.0 - correlation_strength)
                         for idx in domains]
        # A missing preferred domain should not make the remaining choices zero.
        if not any(local_weights):
            local_weights = [weights[idx] for idx in domains]
        domain_idx = _weighted_choice(domains, local_weights, rng)
        sample_idx = label_pools[domain_idx][label][positions[domain_idx][label]]
        positions[domain_idx][label] += 1
        references.append((domain_idx, sample_idx))


def bursty(domain_lengths, rng, weights, burst_size):
    pools = shuffled_pools(domain_lengths, rng)
    positions = [0] * len(pools)
    references = []
    last_domain = None
    while (active := _remaining_domains(pools, positions)):
        candidates = [idx for idx in active if idx != last_domain] or active
        domain_idx = _weighted_choice(candidates, [weights[idx] for idx in candidates], rng)
        for _ in range(min(burst_size, len(pools[domain_idx]) - positions[domain_idx])):
            references.append(_take(pools, positions, domain_idx))
        last_domain = domain_idx
    return references
