"""Small, stateful online routing based on cosine-distance prototypes.

The router deliberately has no domain labels or learned parameters.  It is
intended to be used by a test-time memory as a compact estimate of local
deployment context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RoutingResult:
    """Diagnostics collected while sequentially routing a batch.

    ``posteriors`` and ``distances`` describe the contexts used for each
    assignment.  A spawning row includes its newly created context at zero
    distance; contexts created by later samples are padded with zero posterior
    and ``inf`` distance.
    """

    assignments: torch.Tensor
    posteriors: torch.Tensor
    distances: torch.Tensor
    nearest_distances: torch.Tensor
    spawned: torch.Tensor

    @property
    def context_ids(self) -> torch.Tensor:
        """Alias that makes downstream memory indexing explicit."""
        return self.assignments


class OnlinePrototypeRouter:
    """Route normalized features to online cosine-distance prototypes.

    Samples in a batch are processed in order: a sample may spawn or update a
    context before the next sample is assigned.  With ``momentum=None`` the
    selected prototype is a normalized running mean.  Otherwise its update is
    ``normalize(momentum * prototype + (1 - momentum) * feature)``.
    """

    def __init__(
        self,
        *,
        spawn_threshold: float = 0.25,
        max_contexts: int = 8,
        temperature: float = 0.1,
        momentum: Optional[float] = None,
        eps: float = 1e-12,
    ) -> None:
        if spawn_threshold < 0:
            raise ValueError("spawn_threshold must be non-negative")
        if max_contexts < 1:
            raise ValueError("max_contexts must be at least one")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if momentum is not None and not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.spawn_threshold = float(spawn_threshold)
        self.max_contexts = int(max_contexts)
        self.temperature = float(temperature)
        self.momentum = momentum
        self.eps = float(eps)
        self.reset()

    @property
    def num_contexts(self) -> int:
        return 0 if self.prototypes is None else int(self.prototypes.shape[0])

    @property
    def context_counts(self) -> torch.Tensor:
        """Number of samples hard-assigned to each active context."""
        if self.counts is None:
            return torch.empty(0, dtype=torch.long)
        return self.counts

    def reset(self) -> None:
        """Clear learned contexts and all stream counters."""
        self.prototypes: Optional[torch.Tensor] = None
        self.counts: Optional[torch.Tensor] = None
        self.total_samples = 0
        self.num_spawns = 0

    def route(self, features: torch.Tensor, *, update: bool = True) -> RoutingResult:
        """Hard-route a feature batch and return soft routing diagnostics.

        ``update=False`` provides read-only routing against existing contexts;
        it cannot initialize a router because that would require spawning.
        """
        features = self._validate_features(features)
        if not update and self.num_contexts == 0:
            raise RuntimeError("cannot route without updates before a context exists")
        self._move_state_to(features)
        normalized = F.normalize(features, dim=-1, eps=self._normalization_eps(features))

        assignments = []
        posterior_rows = []
        distance_rows = []
        nearest_distances = []
        spawned = []

        for feature in normalized:
            if self.num_contexts == 0:
                assignment = self._spawn(feature) if update else 0
                distances = feature.new_zeros(1) if update else feature.new_empty(0)
                posterior = feature.new_ones(1) if update else feature.new_empty(0)
                nearest = feature.new_tensor(float("inf"))
                did_spawn = bool(update)
            else:
                distances = self._cosine_distances(feature)
                posterior = torch.softmax(-distances / self.temperature, dim=0)
                nearest, nearest_index = torch.min(distances, dim=0)
                did_spawn = bool(update and nearest.item() > self.spawn_threshold and self.num_contexts < self.max_contexts)
                if did_spawn:
                    assignment = self._spawn(feature)
                    distances = torch.cat((distances, feature.new_zeros(1)))
                    posterior = torch.softmax(-distances / self.temperature, dim=0)
                else:
                    assignment = int(nearest_index.item())
                if update and not did_spawn:
                    self._update_prototype(assignment, feature)

            assignments.append(assignment)
            posterior_rows.append(posterior)
            distance_rows.append(distances)
            nearest_distances.append(nearest)
            spawned.append(did_spawn)
            if update:
                self.total_samples += 1

        return self._result(
            assignments, posterior_rows, distance_rows, nearest_distances, spawned, normalized
        )

    def posterior(self, features: torch.Tensor) -> torch.Tensor:
        """Return read-only soft context probabilities for established contexts."""
        features = self._validate_features(features)
        if self.num_contexts == 0:
            raise RuntimeError("cannot compute a posterior before a context exists")
        self._move_state_to(features)
        normalized = F.normalize(features, dim=-1, eps=self._normalization_eps(features))
        distances = 1 - normalized @ self.prototypes.T
        return torch.softmax(-distances.clamp_min(0) / self.temperature, dim=-1)

    def _spawn(self, feature: torch.Tensor) -> int:
        prototype = feature.unsqueeze(0)
        if self.prototypes is None:
            self.prototypes = prototype.clone()
            self.counts = torch.ones(1, device=feature.device, dtype=torch.long)
        else:
            self.prototypes = torch.cat((self.prototypes, prototype), dim=0)
            self.counts = torch.cat((self.counts, torch.ones(1, device=feature.device, dtype=torch.long)))
        self.num_spawns += 1
        return self.num_contexts - 1

    def _update_prototype(self, context_id: int, feature: torch.Tensor) -> None:
        assert self.prototypes is not None and self.counts is not None
        old_count = self.counts[context_id]
        if self.momentum is None:
            candidate = (self.prototypes[context_id] * old_count.to(self.prototypes.dtype) + feature) / (old_count + 1)
        else:
            candidate = self.momentum * self.prototypes[context_id] + (1 - self.momentum) * feature
        self.prototypes[context_id] = F.normalize(candidate, dim=0, eps=self._normalization_eps(candidate))
        self.counts[context_id] += 1

    def _cosine_distances(self, feature: torch.Tensor) -> torch.Tensor:
        assert self.prototypes is not None
        # Clamp handles tiny dtype-dependent excursions outside [-1, 1].
        return (1 - self.prototypes @ feature).clamp_min(0)

    def _result(self, assignments, posterior_rows, distance_rows, nearest_distances, spawned, features):
        final_contexts = self.num_contexts
        posterior = features.new_zeros((len(assignments), final_contexts))
        distances = features.new_full((len(assignments), final_contexts), float("inf"))
        for index, (row_posterior, row_distances) in enumerate(zip(posterior_rows, distance_rows)):
            if row_posterior.numel():
                posterior[index, : row_posterior.numel()] = row_posterior
                distances[index, : row_distances.numel()] = row_distances
        return RoutingResult(
            assignments=torch.tensor(assignments, device=features.device, dtype=torch.long),
            posteriors=posterior,
            distances=distances,
            nearest_distances=torch.stack(nearest_distances),
            spawned=torch.tensor(spawned, device=features.device, dtype=torch.bool),
        )

    def _move_state_to(self, features: torch.Tensor) -> None:
        if self.prototypes is not None:
            self.prototypes = self.prototypes.to(device=features.device, dtype=features.dtype)
            self.counts = self.counts.to(device=features.device)

    def _validate_features(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must have shape [batch, feature_dim] with a non-empty batch")
        if not features.is_floating_point():
            raise TypeError("features must use a floating-point dtype")
        if self.prototypes is not None and features.shape[1] != self.prototypes.shape[1]:
            raise ValueError("feature dimension does not match existing prototypes")
        return features.detach()

    def _normalization_eps(self, tensor: torch.Tensor) -> float:
        return max(self.eps, torch.finfo(tensor.dtype).eps)
