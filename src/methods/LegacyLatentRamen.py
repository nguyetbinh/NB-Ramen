"""Legacy Ramen with hard latent-context filtering.

This ablation intentionally retains Ramen's cache, batch schedule, weighting,
and adaptation behavior.  The only added state is a latent prototype context
label for each cached item; a query can retrieve supports only from its own
context label.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from models.ModelForBySampleTTA import CLIPModelForBySampleTTA
from routing.online_prototypes import OnlinePrototypeRouter

from .Ramen import PriorityCache
from .TTABase import TTABase
from .losses import softmax_entropy


_REQUIRED_CONFIG = ("max_capacity", "topk", "optimizer", "lr")
_ROUTER_DEFAULTS = {
    "spawn_threshold": 0.25,
    "max_contexts": 8,
    "router_temperature": 0.1,
    "router_momentum": None,
}


def validate_legacy_latent_ramen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the legacy Ramen options plus its latent router options."""
    if not isinstance(config, Mapping):
        raise TypeError("LegacyLatentRamen config must be a mapping")
    missing = [key for key in _REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError("LegacyLatentRamen config is missing: " + ", ".join(missing))
    cfg = dict(config)
    for key, value in _ROUTER_DEFAULTS.items():
        cfg.setdefault(key, value)
    for key in ("max_capacity", "topk", "max_contexts"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("beta", "spawn_threshold", "router_temperature", "lr"):
        value = cfg.get(key, 0.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not torch.isfinite(torch.tensor(float(value))):
            raise ValueError(f"{key} must be finite")
        cfg[key] = float(value)
    if cfg["beta"] < 0 or cfg["spawn_threshold"] < 0:
        raise ValueError("beta and spawn_threshold must be non-negative")
    if cfg["router_temperature"] <= 0 or cfg["lr"] <= 0:
        raise ValueError("router_temperature and lr must be positive")
    momentum = cfg["router_momentum"]
    if momentum is not None:
        if (not isinstance(momentum, (int, float)) or isinstance(momentum, bool)
                or not 0 <= momentum < 1):
            raise ValueError("router_momentum must be null or in [0, 1)")
        cfg["router_momentum"] = float(momentum)
    if not isinstance(cfg["optimizer"], str) or not cfg["optimizer"]:
        raise ValueError("optimizer must be a non-empty string")
    return cfg


class ContextPriorityCache(PriorityCache):
    """Ramen's priority cache with a context index used only at query time."""

    def __init__(self, max_capacity, key_dim, value_dim, device, dtype):
        super().__init__(max_capacity, key_dim, value_dim, device, dtype)
        self.contexts = torch.full((max_capacity,), -1, device=device, dtype=torch.long)

    def add(self, keys, values, entropies, priorities, contexts):
        """Use the exact legacy replacement policy while retaining a context ID."""
        keys = keys.detach().to(device=self.device, dtype=self.dtype)
        values = values.detach().to(device=self.device, dtype=self.dtype)
        contexts = torch.as_tensor(contexts, device=self.device, dtype=torch.long).reshape(-1)
        if keys.ndim == 1:
            keys = keys.unsqueeze(0)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if contexts.numel() != keys.shape[0] or values.shape[0] != keys.shape[0]:
            raise ValueError("keys, values, and contexts must have the same batch size")

        # This is deliberately the same per-item admission and replacement
        # order as PriorityCache.add; context is metadata for the selected slot.
        for index in range(keys.shape[0]):
            if self.size < self.max_capacity:
                slot = self.size
                self.size += 1
            else:
                slot = torch.argmin(self.priorities)
                if priorities[index] <= self.priorities[slot]:
                    continue
            self.keys[slot] = keys[index]
            self.values[slot] = values[index]
            self.priorities[slot] = priorities[index]
            self.entropies[slot] = entropies[index]
            self.contexts[slot] = contexts[index]

    def query(self, queries: torch.Tensor, contexts: torch.Tensor, topk: int):
        """Return padded nearest supports from the matching context only."""
        queries = queries.detach().to(device=self.device, dtype=self.dtype)
        contexts = torch.as_tensor(contexts, device=self.device, dtype=torch.long).reshape(-1)
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        if contexts.numel() != queries.shape[0]:
            raise ValueError("queries and contexts must have the same batch size")
        # When routing has collapsed to one context, use the parent cache's
        # batched cdist/topk path verbatim.  This keeps the no-intervention
        # comparison numerically identical to legacy Ramen, rather than only
        # algebraically equivalent to it.
        if (self.size > 0 and bool((self.contexts[:self.size] == contexts[0]).all())
                and bool((contexts == contexts[0]).all())):
            values, _, entropies, distances = super().query(queries, topk)
            valid = torch.ones(values.shape[:2], device=self.device, dtype=torch.bool)
            return values, entropies, distances, valid
        batch_size = queries.shape[0]
        values = torch.zeros((batch_size, topk, self.value_dim), device=self.device, dtype=self.dtype)
        entropies = torch.zeros((batch_size, topk), device=self.device, dtype=self.dtype)
        distances = torch.zeros((batch_size, topk), device=self.device, dtype=self.dtype)
        valid = torch.zeros((batch_size, topk), device=self.device, dtype=torch.bool)
        for index in range(batch_size):
            matching = torch.nonzero(self.contexts[:self.size] == contexts[index], as_tuple=False).flatten()
            if matching.numel() == 0:
                continue
            count = min(topk, matching.numel())
            candidate_distances = torch.cdist(queries[index:index + 1], self.keys[matching]).squeeze(0)
            sorted_distances, nearest = torch.topk(candidate_distances, k=count, largest=False, sorted=True)
            slots = matching[nearest]
            values[index, :count] = self.values[slots]
            entropies[index, :count] = self.entropies[slots]
            distances[index, :count] = sorted_distances
            valid[index, :count] = True
        return values, entropies, distances, valid

    def reset(self):
        super().reset()


def update_and_retrieve_batch_atomic(
    caches: list[ContextPriorityCache], features: torch.Tensor, gradients: torch.Tensor,
    predicted_classes: torch.Tensor, contexts: torch.Tensor, entropies: torch.Tensor,
    priorities: torch.Tensor, *, topk: int, beta: float,
) -> torch.Tensor:
    """Admit every batch item, then retrieve using Ramen's class aggregation."""
    for index in range(features.shape[0]):
        predicted_class = int(predicted_classes[index])
        caches[predicted_class].add(
            features[index:index + 1], gradients[index:index + 1], entropies[index:index + 1],
            priorities[index:index + 1], contexts[index:index + 1],
        )

    # A collapsed router is the identity intervention.  Preserve Ramen's
    # operations exactly in that case: the same batched query, the same lack
    # of a validity-mask multiply, and the same scalar class denominator.
    # Besides strengthening the control, this avoids mistaking harmless
    # floating-point changes for a routing effect.
    one_context = contexts.numel() > 0 and bool((contexts == contexts[0]).all())
    if one_context:
        one_context = all(
            cache.size == 0 or bool((cache.contexts[:cache.size] == contexts[0]).all())
            for cache in caches
        )
    if one_context:
        retrieved_sum = torch.zeros_like(gradients)
        active_classes = 0
        for cache in caches:
            if cache.size == 0:
                continue
            retrieved, _, support_entropies, distances = PriorityCache.query(cache, features, topk)
            weights = torch.exp(-support_entropies)
            weights = weights * torch.exp(-beta * distances)
            retrieved_sum += (retrieved * weights.unsqueeze(-1)).sum(dim=1)
            active_classes += 1
        return retrieved_sum / active_classes

    retrieved_sum = torch.zeros_like(gradients)
    # Keep Ramen's scalar denominator.  Context routing may zero a class's
    # contribution for one query, but it must not also renormalize the update.
    active_classes = 0
    for cache in caches:
        if cache.size == 0:
            continue
        retrieved, support_entropies, distances, valid = cache.query(features, contexts, topk)
        weights = torch.exp(-support_entropies) * torch.exp(-beta * distances)
        weights = weights * valid.to(weights.dtype)
        retrieved_sum += (retrieved * weights.unsqueeze(-1)).sum(dim=1)
        active_classes += 1
    return retrieved_sum / active_classes


class LegacyLatentRamen(TTABase):
    """Ramen with hard latent-context eligibility and otherwise legacy behavior."""

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = validate_legacy_latent_ramen_config(args.config)
        self.beta = self.cfg.get("beta", 0.0)
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.dtype = torch.half
        self.model = CLIPModelForBySampleTTA(model, datasets.classes, self.cfg, args)
        self.feat_dim = self.model.feat_dim
        self.grad_dim = self.model.grad_dim
        self.router = OnlinePrototypeRouter(
            spawn_threshold=self.cfg["spawn_threshold"], max_contexts=self.cfg["max_contexts"],
            temperature=self.cfg["router_temperature"], momentum=self.cfg["router_momentum"],
        )
        self.loss_fn = lambda logits: softmax_entropy(logits, reduction="sum")
        self.cache = [ContextPriorityCache(self.cfg["max_capacity"], self.feat_dim, self.grad_dim,
                                           self.device, self.dtype) for _ in range(self.num_classes)]
        self.counter = 0
        self.last_diagnostics: dict[str, Any] = self._diagnostics()

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.model.featurize(x)
        logits = self.model.classify(features)
        predicted_classes = logits.argmax(-1)
        self.loss_fn(logits).backward()
        gradients = self.model.get_by_sample_grad()

        with torch.no_grad():
            contexts_before_batch = self.router.num_contexts
            routing = self.router.route(features, update=True)
            contexts = routing.context_ids
            entropies = softmax_entropy(logits, reduction="none")
            priorities = torch.arange(self.counter, self.counter + batch_size,
                                      device=self.device, dtype=self.dtype)
            self.counter += batch_size
            retrieved_gradients = update_and_retrieve_batch_atomic(
                self.cache, features, gradients, predicted_classes, contexts, entropies, priorities,
                topk=self.cfg["topk"], beta=self.beta,
            )
            self.model.set_by_sample_grad(retrieved_gradients)
            # Legacy Ramen admits the whole batch before querying.  Its cache
            # therefore has one final post-admission size, not causal sizes.
            memory_sizes = torch.full(
                (batch_size,), sum(cache.size for cache in self.cache),
                device=features.device, dtype=torch.long,
            )
            active_contexts = contexts_before_batch + routing.spawned.long().cumsum(dim=0)
            self.last_diagnostics = self._diagnostics(routing, active_contexts, memory_sizes)

        self.model.step_and_zero_grad()
        with torch.no_grad():
            logits = self.model(x)
        self.model.reset_parameters()
        return logits

    def reset(self):
        self.model.reset_parameters()
        self.router.reset()
        self.counter = 0
        for cache in self.cache:
            cache.reset()
        self.last_diagnostics = self._diagnostics()

    def get_diagnostics(self) -> dict[str, Any]:
        """Return intervention diagnostics without exposing mutable state."""
        return dict(self.last_diagnostics)

    def diagnostics(self) -> dict[str, Any]:
        """Backward-compatible alias for interactive inspection."""
        return self.get_diagnostics()

    def _diagnostics(self, routing=None, active_contexts=None, memory_sizes=None) -> dict[str, Any]:
        """Describe routing and the batch-atomic legacy cache state.

        PriorityCache does not maintain retained-byte accounting, so this
        intentionally omits ``memory_bytes`` instead of estimating one.
        """
        result: dict[str, Any] = {
            "memory_size": sum(cache.size for cache in self.cache) if memory_sizes is None else memory_sizes,
            "num_active_contexts": self.router.num_contexts if active_contexts is None else active_contexts,
            "inferred_context": None,
            "spawned": None,
        }
        if routing is not None:
            result.update({
                "inferred_context": routing.context_ids.detach().clone(),
                "spawned": routing.spawned.detach().clone(),
            })
        return result
