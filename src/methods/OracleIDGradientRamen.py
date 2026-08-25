"""Evaluator-only oracle analysis of OOD gradient contamination in Ramen.

This module is deliberately not an unsupervised TTA method: ``is_ood`` is
accepted only through a one-shot evaluator hook.  It retains OOD supports to
measure the ordinary Ramen direction, then suppresses only their gradient
contributions for the applied ID-only oracle direction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from models.ModelForBySampleTTA import CLIPModelForBySampleTTA

from .TTABase import TTABase
from .losses import softmax_entropy


_ORACLE_OOD_SOURCE = "evaluator_is_ood"


def validate_oracle_id_gradient_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small Ramen configuration surface and oracle provenance."""
    if not isinstance(config, Mapping):
        raise TypeError("OracleIDGradientRamen config must be a mapping")
    required = ("max_capacity", "topk", "optimizer", "lr")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("OracleIDGradientRamen config is missing: " + ", ".join(missing))
    cfg = dict(config)
    if cfg.get("oracle_ood_source") != _ORACLE_OOD_SOURCE:
        raise ValueError("oracle_ood_source must be 'evaluator_is_ood'")
    for key in ("max_capacity", "topk"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("beta", "lr"):
        value = cfg.get(key, 0.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
        cfg[key] = float(value)
    if cfg["beta"] < 0 or cfg["lr"] <= 0:
        raise ValueError("beta must be non-negative and lr must be positive")
    if not isinstance(cfg["optimizer"], str) or not cfg["optimizer"]:
        raise ValueError("optimizer must be a non-empty string")
    return cfg


class OracleOODContextHook:
    """One-shot, fail-closed evaluator hand-off for ID/OOD membership."""

    requires_oracle_ood_context = True

    def _initialize_oracle_ood_hook(self) -> None:
        self._pending_oracle_is_ood: torch.Tensor | None = None

    def set_oracle_is_ood(self, is_ood: torch.Tensor) -> None:
        if self._pending_oracle_is_ood is not None:
            raise RuntimeError("stale oracle OOD context is pending consumption")
        if not isinstance(is_ood, torch.Tensor) or is_ood.ndim != 1 or is_ood.numel() == 0:
            raise ValueError("oracle OOD context must be a non-empty one-dimensional tensor")
        if is_ood.dtype != torch.bool:
            raise ValueError("oracle OOD context must have boolean dtype")
        self._pending_oracle_is_ood = is_ood.detach().clone()

    def _consume_oracle_is_ood(self, batch_size: int, device: torch.device) -> torch.Tensor:
        is_ood = self._pending_oracle_is_ood
        self._pending_oracle_is_ood = None
        if is_ood is None:
            raise RuntimeError("missing oracle OOD context for this forward batch")
        if is_ood.numel() != batch_size:
            raise RuntimeError("oracle OOD context batch size does not match the forward batch")
        return is_ood.to(device=device, dtype=torch.bool)

    def _clear_oracle_ood_context(self) -> None:
        self._pending_oracle_is_ood = None


class OraclePriorityCache:
    """Ramen-compatible class cache with evaluator-only OOD provenance."""

    def __init__(self, max_capacity, key_dim, value_dim, device, dtype):
        self.max_capacity = max_capacity
        self.keys = torch.empty((max_capacity, key_dim), device=device, dtype=dtype)
        self.values = torch.empty((max_capacity, value_dim), device=device, dtype=dtype)
        self.priorities = torch.full((max_capacity,), float("-inf"), device=device, dtype=dtype)
        self.entropies = torch.full((max_capacity,), float("inf"), device=device, dtype=dtype)
        self.is_ood = torch.zeros((max_capacity,), device=device, dtype=torch.bool)
        self.size = 0

    @property
    def retained_bytes(self) -> int:
        tensors = (self.keys[:self.size], self.values[:self.size], self.priorities[:self.size],
                   self.entropies[:self.size], self.is_ood[:self.size])
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def add(self, keys, values, entropies, priorities, is_ood):
        keys = keys.detach().to(device=self.keys.device, dtype=self.keys.dtype)
        values = values.detach().to(device=self.values.device, dtype=self.values.dtype)
        for index in range(keys.shape[0]):
            if self.size < self.max_capacity:
                target = self.size
                self.size += 1
            else:
                target = int(torch.argmin(self.priorities).item())
                if priorities[index] <= self.priorities[target]:
                    continue
            self.keys[target] = keys[index]
            self.values[target] = values[index]
            self.priorities[target] = priorities[index]
            self.entropies[target] = entropies[index]
            self.is_ood[target] = is_ood[index]

    def query(self, queries, topk):
        if self.size == 0:
            return None
        count = min(topk, self.size)
        distances = torch.cdist(queries.detach().to(self.keys.dtype), self.keys[:self.size])
        distances, indices = torch.topk(distances, k=count, dim=1, largest=False, sorted=True)
        return (
            self.values[indices], self.entropies[indices], distances,
            self.is_ood[indices],
        )

    def reset(self):
        self.size = 0


def _direction_diagnostics(all_gradient: torch.Tensor, id_gradient: torch.Tensor):
    """Return scalar contamination metrics, using ``None`` for zero directions."""
    # Keep evidence diagnostics numerically independent from the support-cache
    # dtype. In particular, reductions in low precision/MPS can otherwise
    # produce non-finite norms for a finite update direction.
    all_metrics = all_gradient.to(dtype=torch.float32)
    id_metrics = id_gradient.to(dtype=torch.float32)
    all_norm = torch.linalg.vector_norm(all_metrics)
    id_norm = torch.linalg.vector_norm(id_metrics)
    if (
        not bool(torch.isfinite(all_norm).item())
        or not bool(torch.isfinite(id_norm).item())
        or float(all_norm) == 0.0
        or float(id_norm) == 0.0
    ):
        return None, None
    cosine = torch.dot(all_metrics, id_metrics) / (all_norm * id_norm)
    if not bool(torch.isfinite(cosine).item()):
        return None, None
    # Floating-point reduction can overshoot the mathematical cosine bounds
    # by a few ulps (observed on MPS when both directions are identical).
    # Clamp before serialising into the strict evidence schema.
    cosine = torch.clamp(cosine, min=-1.0, max=1.0)
    sign_disagreement = torch.sign(all_gradient).ne(torch.sign(id_gradient)).float().mean()
    return float(cosine), float(sign_disagreement)


def aggregate_oracle_supports(queries, caches, *, topk: int, beta: float):
    """Return Ramen all-support and oracle-ID directions plus per-query evidence."""
    batch_size = queries.shape[0]
    gradient_dim = next(cache.values for cache in caches).shape[1]
    all_sum = torch.zeros((batch_size, gradient_dim), device=queries.device, dtype=queries.dtype)
    id_sum = torch.zeros_like(all_sum)
    active_classes = torch.zeros(batch_size, device=queries.device, dtype=torch.long)
    ood_count = torch.zeros(batch_size, device=queries.device, dtype=torch.long)
    total_count = torch.zeros(batch_size, device=queries.device, dtype=torch.long)
    ood_weight = torch.zeros(batch_size, device=queries.device, dtype=queries.dtype)
    total_weight = torch.zeros_like(ood_weight)
    for cache in caches:
        result = cache.query(queries, topk)
        if result is None:
            continue
        values, entropies, distances, flags = result
        weights = torch.exp(-entropies) * torch.exp(-float(beta) * distances)
        all_sum += (values * weights.unsqueeze(-1)).sum(dim=1)
        id_sum += (values * weights.masked_fill(flags, 0).unsqueeze(-1)).sum(dim=1)
        active_classes += 1
        ood_count += flags.sum(dim=1)
        total_count += flags.shape[1]
        ood_weight += weights.masked_fill(~flags, 0).sum(dim=1)
        total_weight += weights.sum(dim=1)
    divisor = active_classes.clamp_min(1).to(all_sum.dtype).unsqueeze(-1)
    all_gradient, id_gradient = all_sum / divisor, id_sum / divisor
    fractions = torch.where(total_count > 0, ood_count.to(all_sum.dtype) / total_count.to(all_sum.dtype), torch.zeros_like(ood_weight))
    weight_fractions = torch.where(total_weight > 0, ood_weight / total_weight, torch.zeros_like(ood_weight))
    cosine, sign_disagreement = [], []
    for index in range(batch_size):
        item_cosine, item_sign = _direction_diagnostics(all_gradient[index], id_gradient[index])
        cosine.append(item_cosine)
        sign_disagreement.append(item_sign)
    return all_gradient, id_gradient, {
        "retrieved_ood_fraction": fractions,
        "retrieved_ood_weight_fraction": weight_fractions,
        "ramen_vs_oracle_id_cosine": cosine,
        "ramen_vs_oracle_id_sign_disagreement": sign_disagreement,
        "active_classes": active_classes,
    }


class OracleIDGradientRamen(OracleOODContextHook, TTABase):
    """Apply an ID-only Ramen gradient while retaining OOD supports for analysis."""

    drop_ood_from_memory = False
    emits_oracle_gradient_diagnostics = True

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = validate_oracle_id_gradient_config(args.config)
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.model = CLIPModelForBySampleTTA(model, datasets.classes, self.cfg, args)
        self.dtype = self.model.dtype if self.model.dtype.is_floating_point else torch.float32
        self.cache = [OraclePriorityCache(self.cfg["max_capacity"], self.model.feat_dim, self.model.grad_dim,
                                          self.device, self.dtype) for _ in range(self.num_classes)]
        self.loss_fn = lambda logits: softmax_entropy(logits, reduction="sum")
        self.counter = 0
        self._initialize_oracle_ood_hook()
        self.last_diagnostics = self._diagnostics()

    @property
    def memory_size(self):
        return sum(cache.size for cache in self.cache)

    @property
    def memory_bytes(self):
        return sum(cache.retained_bytes for cache in self.cache)

    def forward(self, x):
        batch_size = x.shape[0]
        is_ood = self._consume_oracle_is_ood(batch_size, self.device)
        features = self.model.featurize(x)
        logits = self.model.classify(features)
        predicted_classes = logits.argmax(-1)
        self.loss_fn(logits).backward()
        gradients = self.model.get_by_sample_grad()
        with torch.no_grad():
            entropies = softmax_entropy(logits, reduction="none")
            priorities = torch.arange(self.counter, self.counter + batch_size, device=self.device, dtype=self.dtype)
            self.counter += batch_size
            # Match the original Ramen cache timeline exactly: its entire
            # batch is inserted before any query.  The oracle then changes
            # only which already-retrieved support gradients contribute to
            # the applied update, not batch visibility or support selection.
            for index in range(batch_size):
                item = slice(index, index + 1)
                if not (self.drop_ood_from_memory and bool(is_ood[index])):
                    self.cache[int(predicted_classes[index])].add(features[item], gradients[item], entropies[item],
                                                                   priorities[item], is_ood[item])
            _, id_gradient, diagnostics = aggregate_oracle_supports(
                features, self.cache, topk=self.cfg["topk"], beta=self.cfg["beta"],
            )
            rows: dict[str, list[Any]] = {}
            for key, value in diagnostics.items():
                rows[key] = value.detach().cpu().tolist() if torch.is_tensor(value) else list(value)
            self.model.set_by_sample_grad(id_gradient)
            ood_score = -torch.logsumexp(logits, dim=1)
            self.last_diagnostics = self._diagnostics(
                [self.memory_size] * batch_size, [self.memory_bytes] * batch_size, ood_score, rows
            )
        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        self.model.reset_parameters()
        return output

    def reset(self):
        self._clear_oracle_ood_context()
        self.model.reset_parameters()
        for cache in self.cache:
            cache.reset()
        self.counter = 0
        self.last_diagnostics = self._diagnostics()

    def get_diagnostics(self):
        return dict(self.last_diagnostics)

    def _diagnostics(self, memory_sizes=None, memory_bytes=None, ood_score=None, rows=None):
        return {
            "memory_size": self.memory_size if memory_sizes is None else torch.tensor(memory_sizes, device=self.device),
            "memory_bytes": self.memory_bytes if memory_bytes is None else torch.tensor(memory_bytes, device=self.device),
            "pre_adaptation_ood_score": ood_score,
            "retrieved_ood_fraction": None if rows is None else rows["retrieved_ood_fraction"],
            "retrieved_ood_weight_fraction": None if rows is None else rows["retrieved_ood_weight_fraction"],
            "ramen_vs_oracle_id_cosine": None if rows is None else rows["ramen_vs_oracle_id_cosine"],
            "ramen_vs_oracle_id_sign_disagreement": None if rows is None else rows["ramen_vs_oracle_id_sign_disagreement"],
            "active_classes": None if rows is None else rows["active_classes"],
            "oracle_ood_source": _ORACLE_OOD_SOURCE,
        }
