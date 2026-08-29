"""Latent Ramen with causal entropy-gated support-memory admission."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from memory.structured_memory import StructuredGradientMemory
from .LatentRamen import (
    LatentRamen,
    _single_failure_payload,
    aggregate_class_balanced_gradients,
    evaluate_replay_counterfactuals,
    validate_latent_ramen_config,
)
from .losses import softmax_entropy


def validate_entropy_gated_latent_ramen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Latent Ramen contract plus a preregistered gate bound."""
    cfg = validate_latent_ramen_config(config)
    value = cfg.get("max_normalized_entropy")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("max_normalized_entropy must be a finite number in [0, 1]")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("max_normalized_entropy must be in [0, 1]")
    cfg["max_normalized_entropy"] = float(value)
    return cfg


def update_and_retrieve_entropy_gated_causal_batch(
    memory: StructuredGradientMemory,
    features: torch.Tensor,
    gradients: torch.Tensor,
    predicted_classes: torch.Tensor,
    contexts: torch.Tensor,
    entropies: torch.Tensor,
    item_ids: torch.Tensor,
    admitted_to_memory: torch.Tensor,
    *, topk: int, include_current: bool, beta: float, failure_analysis_profile: str = "off",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Process each item causally, exposing only admitted current support."""
    if failure_analysis_profile not in {"off", "trace_v1", "replay_v1"}:
        raise ValueError("unknown failure_analysis_profile")
    retrieved, active_counts, sizes, bytes_, failure_payloads = [], [], [], [], []
    for index in range(features.shape[0]):
        current = slice(index, index + 1)
        admitted = bool(admitted_to_memory[index].item())
        if admitted:
            memory.add(features[current], gradients[current], predicted_classes[current], contexts[current],
                       entropies[current], item_ids=item_ids[current])
        candidates = None
        if failure_analysis_profile != "off":
            candidates = memory.legal_candidate_snapshot(
                contexts[current], schedule="causal", selection="class_balanced",
                include_current=include_current if admitted else False, current_item_ids=item_ids[current],
            )[0]
        support = memory.query(
            features[current], contexts[current], topk,
            include_current=include_current if admitted else False,
            current_item_ids=item_ids[current],
        )
        item_gradient, item_counts = aggregate_class_balanced_gradients(support, beta)
        retrieved.append(item_gradient)
        active_counts.append(item_counts)
        sizes.append(memory.size)
        bytes_.append(memory.retained_bytes)
        if failure_analysis_profile != "off":
            replay_item = None if failure_analysis_profile != "replay_v1" else {
                "query_feature": features[index].detach().clone(), "query_gradient": gradients[index].detach().clone(),
                "predicted_class": predicted_classes[index].detach().clone(), "context": contexts[index].detach().clone(),
                "entropy": entropies[index].detach().clone(), "admitted_to_memory": admitted_to_memory[index].detach().clone(),
            }
            failure_payloads.append(_single_failure_payload(
                support, beta, item_id=item_ids[index], batch_position=index, legal_candidates=candidates,
                profile=failure_analysis_profile, replay_item=replay_item,
            ))
    result = (
        torch.cat(retrieved, dim=0),
        torch.cat(active_counts, dim=0),
        torch.tensor(sizes, device=features.device, dtype=torch.long),
        torch.tensor(bytes_, device=features.device, dtype=torch.long),
    )
    return result if failure_analysis_profile == "off" else (*result, failure_payloads)


class EntropyGatedLatentRamen(LatentRamen):
    """LatentRamen whose only changed behavior is support-memory admission."""

    def __init__(self, model, datasets, args):
        # Construct explicitly rather than calling the parent validator, whose
        # config surface intentionally does not require the gate parameter.
        super().__init__(model, datasets, args)
        self.cfg = validate_entropy_gated_latent_ramen_config(args.config)

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
            entropies = softmax_entropy(logits, reduction="none")
            normalized_entropy = (entropies / math.log(self.num_classes)).clamp(0.0, 1.0)
            admitted = normalized_entropy <= self.cfg["max_normalized_entropy"]
            item_ids = torch.arange(self.counter, self.counter + batch_size, device=self.device, dtype=torch.long)
            self.counter += batch_size
            retrieval = update_and_retrieve_entropy_gated_causal_batch(
                self.memory, features, gradients, predicted_classes, routing.context_ids, entropies,
                item_ids, admitted, topk=self.cfg["topk"], include_current=self.cfg["include_current"],
                beta=self.cfg["beta"], failure_analysis_profile=self.failure_analysis_profile,
            )
            if self.failure_analysis_profile == "off":
                retrieved, active_classes, memory_sizes, memory_bytes = retrieval
                failure_analysis = None
            else:
                retrieved, active_classes, memory_sizes, memory_bytes, failure_analysis = retrieval
            self.model.set_by_sample_grad(retrieved)
            active_contexts = contexts_before_batch + routing.spawned.long().cumsum(dim=0)
            self.last_diagnostics = self._diagnostics(
                routing, active_classes, memory_sizes, memory_bytes, active_contexts,
                admission_prediction=predicted_classes,
                admission_normalized_entropy=normalized_entropy,
                admitted_to_memory=admitted,
                failure_analysis=failure_analysis,
                failure_analysis_profile=self.failure_analysis_profile,
            )

        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        if self.failure_analysis_profile == "replay_v1":
            evaluate_replay_counterfactuals(
                self.model, x, output, retrieved, failure_analysis, self.failure_counterfactual_thresholds
            )
        self.model.reset_parameters()
        return output
