"""Oracle soft context-aware ranking for strictly causal Ramen.

``OracleSoftRankRamen`` keeps the global per-predicted-class support pool used
by :class:`CausalRamen`.  Evaluator domain labels are used solely as a bonus
while ranking otherwise eligible supports; they never control admission,
prediction, gradient construction, or final inference.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch

from memory.structured_memory import RetrievalBatch, StructuredGradientMemory
from models.ModelForBySampleTTA import CLIPModelForBySampleTTA

from .LatentRamen import LatentRamen, aggregate_class_balanced_gradients, validate_latent_ramen_config
from .OracleLatentRamen import OracleDomainContextHook, OracleLatentRamen
from .TTABase import TTABase
from .losses import softmax_entropy


_ORACLE_CONTEXT_SOURCE = "evaluator_domain_idx"


def validate_oracle_soft_rank_ramen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the oracle soft-ranking experiment contract.

    ``gamma`` is deliberately named after the preregistered context-strength
    sweep in the research plan.  Zero is the exact CausalRamen recovery point.
    """
    cfg = validate_latent_ramen_config(config)
    if cfg.get("oracle_context_source") != _ORACLE_CONTEXT_SOURCE:
        raise ValueError(
            "OracleSoftRankRamen requires oracle_context_source='evaluator_domain_idx'"
        )
    if cfg["capacity_scope"] != "per_class":
        raise ValueError("OracleSoftRankRamen requires capacity_scope='per_class'")
    gamma = cfg.get("gamma", 0.0)
    if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or not math.isfinite(float(gamma)):
        raise ValueError("gamma must be a finite number")
    gamma = float(gamma)
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    cfg["gamma"] = gamma
    return cfg


def support_composition_diagnostics(
    retrieval: RetrievalBatch,
    *,
    query_contexts: torch.Tensor,
    beta: float,
    num_classes: int,
    context_strength: float,
) -> dict[str, torch.Tensor]:
    """Compute per-query composition metrics from final Ramen aggregation weights."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if not isinstance(context_strength, (int, float)) or not math.isfinite(float(context_strength)):
        raise ValueError("context_strength must be finite")
    if not hasattr(retrieval, "contexts"):
        raise TypeError("soft-routing retrieval must expose support contexts")
    if query_contexts.ndim != 1 or query_contexts.shape[0] != retrieval.valid_mask.shape[0]:
        raise ValueError("query_contexts must have shape [batch]")

    valid = retrieval.valid_mask
    support_count = valid.sum(dim=(1, 2)).to(torch.long)
    active_class_count = valid.any(dim=2).sum(dim=1).to(torch.long)
    class_coverage = active_class_count.to(torch.float32) / float(num_classes)
    same = valid & (retrieval.contexts == query_contexts.to(retrieval.contexts.device)[:, None, None])
    same_count = same.sum(dim=(1, 2)).to(torch.float32)
    safe_count = support_count.clamp_min(1).to(torch.float32)
    same_domain_ratio = same_count / safe_count
    cross_domain_ratio = (support_count.to(torch.float32) - same_count) / safe_count

    # These are precisely the entropy/distance weights used by the existing
    # class-balanced aggregation.  ESS is invariant to their normalization.
    distance_weights = torch.where(
        valid, torch.exp(-float(beta) * retrieval.distances), torch.zeros_like(retrieval.distances)
    )
    weights = torch.exp(-retrieval.entropies) * distance_weights
    weight_sum = weights.sum(dim=(1, 2))
    effective_sample_size = torch.where(
        support_count > 0,
        weight_sum.square() / weights.square().sum(dim=(1, 2)).clamp_min(torch.finfo(weights.dtype).tiny),
        torch.zeros_like(weight_sum),
    )
    return {
        "returned_support_count": support_count,
        "active_class_count": active_class_count,
        "class_coverage": class_coverage,
        "same_domain_ratio": same_domain_ratio,
        "cross_domain_ratio": cross_domain_ratio,
        "effective_sample_size": effective_sample_size,
        "context_strength": torch.full(
            (support_count.shape[0],), float(context_strength), device=support_count.device, dtype=torch.float32
        ),
    }


def soft_routing_influence_diagnostics(
    retrieval: RetrievalBatch,
    reference: RetrievalBatch,
    *,
    query_contexts: torch.Tensor,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Measure how soft ranking changed a fixed-memory gamma-zero retrieval.

    ``selection_change_ratio`` is the fraction of selected valid
    ``(predicted-class, rank)`` slots whose item ID differs from the same slot
    in the deterministic gamma-zero reference.  ``mean_rank_displacement``
    averages ``|soft_rank - reference_rank|`` only for selected item IDs that
    occur in both top-k lists for their predicted class; newly selected items
    are intentionally not comparable.  Both measures are zero at gamma zero.
    """
    if retrieval.item_ids.shape != reference.item_ids.shape:
        raise ValueError("retrieval and reference shapes must match")
    if query_contexts.ndim != 1 or query_contexts.shape[0] != retrieval.valid_mask.shape[0]:
        raise ValueError("query_contexts must have shape [batch]")
    valid = retrieval.valid_mask
    denominator = valid.sum(dim=(1, 2)).clamp_min(1).to(torch.float32)
    changed = valid & (retrieval.item_ids != reference.item_ids)
    selection_change_ratio = changed.sum(dim=(1, 2)).to(torch.float32) / denominator
    same_context = valid & (retrieval.contexts == query_contexts.to(retrieval.contexts.device)[:, None, None])
    mean_context_bonus = float(gamma) * same_context.sum(dim=(1, 2)).to(torch.float32) / denominator

    displacements = []
    for batch_index in range(retrieval.item_ids.shape[0]):
        per_item = []
        for predicted_class in range(retrieval.item_ids.shape[1]):
            reference_ids = reference.item_ids[batch_index, predicted_class]
            reference_valid = reference.valid_mask[batch_index, predicted_class]
            for rank in torch.nonzero(valid[batch_index, predicted_class], as_tuple=False).flatten().tolist():
                matches = torch.nonzero(
                    reference_valid & (reference_ids == retrieval.item_ids[batch_index, predicted_class, rank]),
                    as_tuple=False,
                ).flatten()
                if matches.numel():
                    per_item.append(abs(rank - int(matches[0])))
        displacements.append(
            torch.tensor(per_item, device=retrieval.item_ids.device, dtype=torch.float32).mean()
            if per_item else torch.zeros((), device=retrieval.item_ids.device, dtype=torch.float32)
        )
    return {
        "selection_change_ratio": selection_change_ratio,
        "mean_context_bonus": mean_context_bonus,
        "mean_rank_displacement": torch.stack(displacements),
    }


def update_and_retrieve_oracle_soft_rank_causal_batch(
    memory: StructuredGradientMemory,
    features: torch.Tensor,
    gradients: torch.Tensor,
    predicted_classes: torch.Tensor,
    contexts: torch.Tensor,
    entropies: torch.Tensor,
    item_ids: torch.Tensor,
    *,
    topk: int,
    include_current: bool,
    beta: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run exact stream-order insertion and global, softly ranked retrieval."""
    retrieved, active_classes, memory_sizes, memory_bytes = [], [], [], []
    metrics: dict[str, list[torch.Tensor]] = {}
    for index in range(features.shape[0]):
        current = slice(index, index + 1)
        # Preserve CausalRamen's historical-only schedule.  Query before
        # insertion avoids evicting useful history merely to exclude the
        # current item; current-inclusive queries add first as usual.
        if include_current:
            memory.add(features[current], gradients[current], predicted_classes[current], contexts[current],
                       entropies[current], item_ids=item_ids[current])
        support = memory.query_class_balanced_global(
            features[current], topk, query_contexts=contexts[current], context_strength=gamma,
            include_current=include_current, current_item_ids=item_ids[current],
        )
        reference = memory.query_class_balanced_global(
            features[current], topk, query_contexts=contexts[current], context_strength=0.0,
            include_current=include_current, current_item_ids=item_ids[current],
        )
        item_gradient, item_active_classes = aggregate_class_balanced_gradients(support, beta)
        composition = support_composition_diagnostics(
            support, query_contexts=contexts[current], beta=beta,
            num_classes=memory.num_classes, context_strength=gamma,
        )
        composition.update(soft_routing_influence_diagnostics(
            support, reference, query_contexts=contexts[current], gamma=gamma,
        ))
        # These evaluator-facing arrays allow downstream reconstruction of
        # domain composition.  The runtime trace deliberately does not
        # serialize them; they remain only in method diagnostics.
        composition["support_item_ids"] = support.item_ids
        composition["support_valid_mask"] = support.valid_mask
        if not include_current:
            memory.add(features[current], gradients[current], predicted_classes[current], contexts[current],
                       entropies[current], item_ids=item_ids[current])
        retrieved.append(item_gradient)
        active_classes.append(item_active_classes)
        memory_sizes.append(memory.size)
        memory_bytes.append(memory.retained_bytes)
        for name, value in composition.items():
            metrics.setdefault(name, []).append(value)
    return (
        torch.cat(retrieved, dim=0),
        torch.cat(active_classes, dim=0),
        torch.tensor(memory_sizes, device=features.device, dtype=torch.long),
        torch.tensor(memory_bytes, device=features.device, dtype=torch.long),
        {name: torch.cat(values, dim=0) for name, values in metrics.items()},
    )


class OracleSoftRankRamen(OracleDomainContextHook, TTABase):
    """Strictly causal Ramen with oracle domain used only as a rank bonus."""

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = validate_oracle_soft_rank_ramen_config(args.config)
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.model = CLIPModelForBySampleTTA(model, datasets.classes, self.cfg, args)
        cache_dtype = self.model.dtype if self.model.dtype.is_floating_point else torch.float32
        self.memory = StructuredGradientMemory(
            self.num_classes, self.cfg["max_capacity"], self.model.feat_dim, self.model.grad_dim,
            device=self.device, dtype=cache_dtype, capacity_scope="per_class",
        )
        self.loss_fn = lambda logits: softmax_entropy(logits, reduction="sum")
        self.counter = 0
        self._initialize_oracle_context_hook()
        self.last_diagnostics: dict[str, Any] = self._diagnostics()

    def forward(self, x):
        batch_size = x.shape[0]
        contexts = self._consume_oracle_domain_context(batch_size, self.device)
        features = self.model.featurize(x)
        logits = self.model.classify(features)
        predicted_classes = logits.argmax(-1)
        self.loss_fn(logits).backward()
        gradients = self.model.get_by_sample_grad()
        with torch.no_grad():
            entropies = softmax_entropy(logits, reduction="none")
            item_ids = torch.arange(self.counter, self.counter + batch_size, device=self.device, dtype=torch.long)
            self.counter += batch_size
            retrieved, active_classes, memory_sizes, memory_bytes, composition = update_and_retrieve_oracle_soft_rank_causal_batch(
                self.memory, features, gradients, predicted_classes, contexts, entropies, item_ids,
                topk=self.cfg["topk"], include_current=self.cfg["include_current"],
                beta=self.cfg["beta"], gamma=self.cfg["gamma"],
            )
            self.model.set_by_sample_grad(retrieved)
            self.last_diagnostics = self._diagnostics(
                contexts, active_classes, memory_sizes, memory_bytes, composition
            )
        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        self.model.reset_parameters()
        return output

    def reset(self):
        self._clear_oracle_domain_context()
        self.model.reset_parameters()
        self.memory.reset()
        self.counter = 0
        self.last_diagnostics = self._diagnostics()

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)

    def diagnostics(self) -> dict[str, Any]:
        return self.get_diagnostics()

    def _diagnostics(self, contexts=None, active_classes=None, memory_sizes=None, memory_bytes=None, composition=None):
        diagnostics: dict[str, Any] = {
            "inferred_context": None if contexts is None else contexts.detach().clone(),
            "memory_size": self.memory.size if memory_sizes is None else memory_sizes.detach().clone(),
            "memory_bytes": self.memory.retained_bytes if memory_bytes is None else memory_bytes.detach().clone(),
            "memory_active_contexts": self.memory.active_contexts,
            "active_classes": None if active_classes is None else active_classes.detach().clone(),
            "oracle_context_source": _ORACLE_CONTEXT_SOURCE,
            "capacity_scope": self.memory.capacity_scope,
            "memory_capacity_scope": self.memory.capacity_scope,
            "memory_max_capacity": self.memory.max_capacity,
            "gamma": self.cfg["gamma"],
        }
        if composition is None:
            diagnostics.update({
                "returned_support_count": None, "active_class_count": None,
                "class_coverage": None, "same_domain_ratio": None,
                "cross_domain_ratio": None, "effective_sample_size": None,
                "context_strength": None, "selection_change_ratio": None,
                "mean_context_bonus": None, "mean_rank_displacement": None,
                "support_item_ids": None, "support_valid_mask": None,
            })
        else:
            diagnostics.update({name: value.detach().clone() for name, value in composition.items()})
        return diagnostics


# Phase-1 names preserve the prior hard-routing behavior without breaking
# evidence artifacts that still refer to LatentRamen / OracleLatentRamen.
LatentHardRamen = LatentRamen
OracleHardRamen = OracleLatentRamen
