"""Confidence-gated, otherwise ordinary, Ramen support memory.

This is intentionally not a latent-memory method: it retains Ramen's
per-predicted-class caches, batch-atomic admission order, and aggregation.
Only whether a sample becomes a support is changed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .Ramen import Ramen
from .losses import softmax_entropy


def validate_entropy_gated_ramen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the ordinary Ramen surface plus its fixed admission gate."""
    if not isinstance(config, Mapping):
        raise TypeError("EntropyGatedRamen config must be a mapping")
    required = ("max_capacity", "topk", "optimizer", "lr", "max_normalized_entropy")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("EntropyGatedRamen config is missing: " + ", ".join(missing))
    cfg = dict(config)
    value = cfg["max_normalized_entropy"]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("max_normalized_entropy must be the finite numeric value 0.50")
    if float(value) != 0.5:
        raise ValueError("max_normalized_entropy is preregistered and must equal 0.50")
    cfg["max_normalized_entropy"] = 0.5
    return cfg


class EntropyGatedRamen(Ramen):
    """Ramen which admits only predictions with normalized entropy <= 0.50."""

    def __init__(self, model, datasets, args):
        # Ramen constructs precisely the model, loss, and cache layout that
        # this control needs.  Validate before handing its compatible mapping
        # to the ordinary implementation.
        args.config = validate_entropy_gated_ramen_config(args.config)
        super().__init__(model, datasets, args)

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.model.featurize(x)
        logits = self.model.classify(features)
        predicted_classes = logits.argmax(-1)
        self.loss_fn(logits).backward()
        gradients = self.model.get_by_sample_grad()

        with torch.no_grad():
            entropies = softmax_entropy(logits, reduction="none")
            # All supported datasets have at least two classes.  The clamp
            # also keeps finite numerical roundoff within the documented
            # [0, 1] diagnostic range.
            normalized_entropy = (entropies / math.log(self.num_classes)).clamp(0.0, 1.0)
            admitted = normalized_entropy <= self.cfg["max_normalized_entropy"]
            priorities = torch.arange(
                self.counter, self.counter + batch_size, device=self.device, dtype=self.dtype
            )
            self.counter += batch_size

            # Insert the entire admitted portion before any retrieval.  This
            # preserves Ramen's batch-atomic current-support visibility while
            # making rejected current samples impossible to self-retrieve.
            for index in range(batch_size):
                if not bool(admitted[index].item()):
                    continue
                predicted_class = predicted_classes[index]
                self.cache[predicted_class].add(
                    features[index].unsqueeze(0), gradients[index].unsqueeze(0),
                    entropies[index].unsqueeze(0), priorities[index].unsqueeze(0),
                )

            retrieved_sum = torch.zeros_like(gradients)
            active_class_count = 0
            for cache in self.cache:
                if cache.size == 0:
                    continue
                values, _, support_entropies, distances = cache.query(features, topk=self.cfg["topk"])
                weights = torch.exp(-support_entropies) * torch.exp(-self.beta * distances)
                retrieved_sum += (values * weights.unsqueeze(-1)).sum(dim=1)
                active_class_count += 1

            # Unlike ordinary Ramen's implicit non-empty-cache assumption,
            # an all-rejected first batch has no support and must be a no-op.
            retrieved = retrieved_sum if active_class_count == 0 else retrieved_sum / active_class_count
            self.model.set_by_sample_grad(retrieved)
            self.last_diagnostics = {
                "admission_prediction": predicted_classes.detach(),
                "admission_normalized_entropy": normalized_entropy.detach(),
                "admitted_to_memory": admitted.detach(),
                "memory_size": sum(cache.size for cache in self.cache),
                "memory_bytes": self.memory_bytes,
                "pre_adaptation_ood_score": -torch.logsumexp(logits.detach(), dim=1),
            }

        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        self.model.reset_parameters()
        return output
