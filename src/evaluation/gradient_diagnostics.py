"""Pure, evaluator-safe summaries for retrieved Ramen gradients.

The functions in this module deliberately do not change the gradient passed
to the optimiser.  They reproduce Ramen's production aggregation and compute
the separately normalised class-local quantities used by failure analysis.
"""

from __future__ import annotations

from typing import Any

import torch


def production_support_weights(entropies: torch.Tensor, distances: torch.Tensor,
                               valid_mask: torch.Tensor, beta: float) -> torch.Tensor:
    """Return Ramen's exact entropy/distance weight for every padded support."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    distance_weight = torch.where(
        valid_mask, torch.exp(-float(beta) * distances), torch.zeros_like(distances)
    )
    return torch.exp(-entropies) * distance_weight


def class_balanced_production_aggregate(
    gradients: torch.Tensor, entropies: torch.Tensor, distances: torch.Tensor,
    valid_mask: torch.Tensor, beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the deployed class-balanced aggregate without normalising it.

    Returns ``(aggregate, active_classes, support_weights)``.  ``aggregate``
    is intentionally the same weighted class sum / active-class mean that the
    production path uses; diagnostic normalisation belongs only to ``h``.
    """
    weights = production_support_weights(entropies, distances, valid_mask, beta)
    per_class = (gradients * weights.to(gradients.dtype).unsqueeze(-1)).sum(dim=2)
    active = valid_mask.any(dim=2)
    active_count = active.sum(dim=1)
    summed = (per_class * active.to(per_class.dtype).unsqueeze(-1)).sum(dim=1)
    aggregate = summed / active_count.clamp_min(1).to(summed.dtype).unsqueeze(-1)
    return aggregate, active_count, weights


def normalized_class_local_gradients(
    gradients: torch.Tensor, support_weights: torch.Tensor, valid_mask: torch.Tensor,
    *, eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute framework ``h[q,c]`` and its active-class mask.

    ``h`` is the weighted mean within each predicted support class, rather
    than the production class sum.  Empty padded classes are exactly zero.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    numerator = (gradients * support_weights.to(gradients.dtype).unsqueeze(-1)).sum(dim=2)
    denominator = support_weights.sum(dim=2, keepdim=True).to(numerator.dtype)
    active = valid_mask.any(dim=2)
    h = numerator / (denominator + eps)
    return h * active.to(h.dtype).unsqueeze(-1), active


def consensus_strength(class_local_gradients: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Return coordinate-wise SignSGD consensus ``abs(mean(sign(h)))``."""
    signs = torch.sign(class_local_gradients)
    masked = signs * active_mask.to(signs.dtype).unsqueeze(-1)
    count = active_mask.sum(dim=1, keepdim=True).clamp_min(1).to(signs.dtype)
    return (masked.sum(dim=1) / count).abs()


def _quantile_or_nan(values: torch.Tensor, quantile: float) -> torch.Tensor:
    # torch.quantile supports empty inputs poorly across accelerator versions.
    result = torch.full((values.shape[0],), float("nan"), device=values.device, dtype=torch.float32)
    if values.shape[-1]:
        result = torch.quantile(values.float(), quantile, dim=-1)
    return result


def pairwise_class_gradient_summaries(
    class_local_gradients: torch.Tensor, active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean cosine, sign agreement, and valid-pair counts per query.

    Queries with fewer than two active classes have no pairwise statistic, so
    their means are NaN and their pair count is zero.  This avoids presenting
    "perfect agreement" for an untestable singleton support set.
    """
    batch, classes, _ = class_local_gradients.shape
    cosine_sum = torch.zeros(batch, device=class_local_gradients.device, dtype=torch.float32)
    sign_sum = torch.zeros_like(cosine_sum)
    pair_count = torch.zeros(batch, device=class_local_gradients.device, dtype=torch.long)
    for left in range(classes):
        for right in range(left + 1, classes):
            valid = active_mask[:, left] & active_mask[:, right]
            if not bool(valid.any()):
                continue
            lhs, rhs = class_local_gradients[:, left].float(), class_local_gradients[:, right].float()
            cosine = torch.nn.functional.cosine_similarity(lhs, rhs, dim=-1, eps=1e-12)
            # Zero coordinates agree only when both signs are zero; this is
            # the direct equality requested by the SignSGD diagnostic.
            sign_agreement = (torch.sign(lhs) == torch.sign(rhs)).float().mean(dim=-1)
            cosine_sum += cosine * valid.float()
            sign_sum += sign_agreement * valid.float()
            pair_count += valid.long()
    denominator = pair_count.clamp_min(1).float()
    nan = torch.full_like(cosine_sum, float("nan"))
    return (
        torch.where(pair_count > 0, cosine_sum / denominator, nan),
        torch.where(pair_count > 0, sign_sum / denominator, nan),
        pair_count,
    )


def summarize_class_balanced_gradients(
    gradients: torch.Tensor, entropies: torch.Tensor, distances: torch.Tensor,
    valid_mask: torch.Tensor, beta: float, *, low_consensus_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Produce compact per-query diagnostic tensors from a retrieval result."""
    if not 0 <= low_consensus_threshold <= 1:
        raise ValueError("low_consensus_threshold must be in [0, 1]")
    aggregate, active_count, weights = class_balanced_production_aggregate(
        gradients, entropies, distances, valid_mask, beta
    )
    h, active = normalized_class_local_gradients(gradients, weights, valid_mask)
    strength = consensus_strength(h, active)
    cosine, sign_agreement, pair_count = pairwise_class_gradient_summaries(h, active)
    return {
        "production_aggregate": aggregate,
        "support_weights": weights,
        "class_local_gradients": h,
        "active_class_mask": active,
        "active_support_classes": active_count,
        "consensus_strength": strength,
        "consensus_mean": strength.float().mean(dim=-1),
        "consensus_p10": _quantile_or_nan(strength, .10),
        "consensus_p50": _quantile_or_nan(strength, .50),
        "fraction_low_consensus_coordinates": (strength < low_consensus_threshold).float().mean(dim=-1),
        "pairwise_cosine_mean": cosine,
        "pairwise_sign_agreement_mean": sign_agreement,
        "pairwise_class_gradient_count": pair_count,
    }


def trace_payload_from_retrieval(retrieval: Any, beta: float) -> dict[str, torch.Tensor]:
    """Create model-visible provenance plus scalar summaries, never raw gradients."""
    if retrieval.gradients.ndim == 3:
        return _flat_trace_payload_from_retrieval(retrieval, beta)
    summary = summarize_class_balanced_gradients(
        retrieval.gradients, retrieval.entropies, retrieval.distances, retrieval.valid_mask, beta
    )
    classes = torch.arange(retrieval.item_ids.shape[1], device=retrieval.item_ids.device, dtype=torch.long)
    classes = classes.view(1, -1, 1).expand_as(retrieval.item_ids)
    classes = torch.where(retrieval.valid_mask, classes, torch.full_like(classes, -1))
    return {
        "support_item_ids": retrieval.item_ids.detach().clone(),
        "support_predicted_classes": classes.detach().clone(),
        "support_distances": retrieval.distances.detach().clone(),
        "support_entropies": retrieval.entropies.detach().clone(),
        "support_recencies": retrieval.recencies.detach().clone(),
        "support_valid_mask": retrieval.valid_mask.detach().clone(),
        "support_weights": summary["support_weights"].detach().clone(),
        "support_count": retrieval.valid_mask.sum(dim=(1, 2)).to(torch.long),
        "active_support_classes": summary["active_support_classes"].detach().clone(),
        "consensus_mean": summary["consensus_mean"].detach().clone(),
        "consensus_p10": summary["consensus_p10"].detach().clone(),
        "consensus_p50": summary["consensus_p50"].detach().clone(),
        "fraction_low_consensus_coordinates": summary["fraction_low_consensus_coordinates"].detach().clone(),
        "pairwise_cosine_mean": summary["pairwise_cosine_mean"].detach().clone(),
        "pairwise_sign_agreement_mean": summary["pairwise_sign_agreement_mean"].detach().clone(),
        "pairwise_class_gradient_count": summary["pairwise_class_gradient_count"].detach().clone(),
        "production_aggregate_norm": summary["production_aggregate"].float().norm(dim=-1),
        "consensus_strength": summary["consensus_strength"].detach().clone(),
    }


def _flat_trace_payload_from_retrieval(retrieval: Any, beta: float) -> dict[str, torch.Tensor]:
    """Summarise an unbalanced pool while preserving its production aggregate.

    Flat retrievals retain the predicted class of every selected item.  For
    failure analysis we regroup those supports into compact class-local
    buckets; this does not alter the production (single-pool) aggregation.
    """
    valid = retrieval.valid_mask
    weights = production_support_weights(retrieval.entropies, retrieval.distances, valid, beta)
    summed = (retrieval.gradients * weights.to(retrieval.gradients.dtype).unsqueeze(-1)).sum(dim=1)
    counts = valid.sum(dim=1)
    aggregate = summed / counts.clamp_min(1).to(summed.dtype).unsqueeze(-1)
    batch, topk = valid.shape
    gradients = retrieval.gradients.new_zeros((batch, topk, topk, retrieval.gradients.shape[-1]))
    entropies = retrieval.entropies.new_zeros((batch, topk, topk))
    distances = retrieval.distances.new_full((batch, topk, topk), float("inf"))
    grouped_valid = torch.zeros((batch, topk, topk), device=valid.device, dtype=torch.bool)
    for index in range(batch):
        classes = torch.unique(retrieval.predicted_classes[index][valid[index]], sorted=True)
        for class_index, predicted_class in enumerate(classes.tolist()):
            members = (retrieval.predicted_classes[index] == predicted_class) & valid[index]
            count = int(members.sum())
            gradients[index, class_index, :count] = retrieval.gradients[index, members]
            entropies[index, class_index, :count] = retrieval.entropies[index, members]
            distances[index, class_index, :count] = retrieval.distances[index, members]
            grouped_valid[index, class_index, :count] = True
    summary = summarize_class_balanced_gradients(gradients, entropies, distances, grouped_valid, beta)
    return {
        "support_item_ids": retrieval.item_ids.detach().clone(),
        "support_predicted_classes": torch.where(
            valid, retrieval.predicted_classes, torch.full_like(retrieval.predicted_classes, -1)
        ).detach().clone(),
        "support_distances": retrieval.distances.detach().clone(),
        "support_entropies": retrieval.entropies.detach().clone(),
        "support_recencies": retrieval.recencies.detach().clone(),
        "support_valid_mask": valid.detach().clone(),
        "support_weights": weights.detach().clone(),
        "support_count": counts.to(torch.long),
        "active_support_classes": summary["active_support_classes"].detach().clone(),
        "consensus_mean": summary["consensus_mean"].detach().clone(),
        "consensus_p10": summary["consensus_p10"].detach().clone(),
        "consensus_p50": summary["consensus_p50"].detach().clone(),
        "fraction_low_consensus_coordinates": summary["fraction_low_consensus_coordinates"].detach().clone(),
        "pairwise_cosine_mean": summary["pairwise_cosine_mean"].detach().clone(),
        "pairwise_sign_agreement_mean": summary["pairwise_sign_agreement_mean"].detach().clone(),
        "pairwise_class_gradient_count": summary["pairwise_class_gradient_count"].detach().clone(),
        "production_aggregate_norm": aggregate.float().norm(dim=-1),
        "consensus_strength": summary["consensus_strength"].detach().clone(),
    }
