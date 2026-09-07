"""Shared contract for the preregistered ConsensusRamen-v0 policy."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


PRIMARY_CONSENSUS_THRESHOLD = 0.2
PRIMARY_MIN_CONSENSUS_CLASSES = 3
PRIMARY_CONSENSUS_MODE = "hard_mask"
PRIMARY_INCLUDE_CURRENT = True

# Keep the primary method, evaluator-only diagnostics, and canonical planners
# tied to one immutable policy surface. Named ablations remain free to use
# other values through their own configs.
PRIMARY_CONSENSUS_POLICY = MappingProxyType({
    "consensus_threshold": PRIMARY_CONSENSUS_THRESHOLD,
    "min_consensus_classes": PRIMARY_MIN_CONSENSUS_CLASSES,
    "consensus_mode": PRIMARY_CONSENSUS_MODE,
    "include_current": PRIMARY_INCLUDE_CURRENT,
})


def validate_primary_consensus_policy(config: Mapping[str, Any]) -> None:
    """Reject drift from the preregistered deployable v0 policy."""
    if any(config.get(key) != value for key, value in PRIMARY_CONSENSUS_POLICY.items()):
        raise ValueError("ConsensusRamen-v0 policy does not match the preregistered contract")
