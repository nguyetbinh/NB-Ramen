"""Evaluator-only ID-support upper bound for ``ConsensusRamen``.

The evaluator supplies ID/OOD membership through a one-shot hook.  Membership
is used only to prevent OOD items entering this method's support caches; the
consensus calculation itself remains the ordinary class-consensus calculation
over the retained ID supports.  This is an oracle diagnostic, not a deployable
adaptation method.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from models.ModelForBySampleTTA import CLIPModelForBySampleTTA

from .ConsensusRamen import aggregate_consensus_supports, validate_consensus_ramen_config
from .OracleIDGradientRamen import OracleOODContextHook
from .Ramen import PriorityCache
from .TTABase import TTABase
from .losses import softmax_entropy


_ORACLE_OOD_SOURCE = "evaluator_is_ood"


def validate_oracle_consensus_ramen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed ConsensusRamen-v0 surface plus oracle provenance."""
    cfg = validate_consensus_ramen_config(config)
    if cfg.get("oracle_ood_source") != _ORACLE_OOD_SOURCE:
        raise ValueError("oracle_ood_source must be 'evaluator_is_ood'")
    return cfg


class OracleConsensusRamen(OracleOODContextHook, TTABase):
    """ConsensusRamen upper bound that admits evaluator-known ID supports only."""

    # This opt-in is consumed only by main's evaluator hand-off; ordinary
    # ConsensusRamen has no corresponding hook or evaluator-derived input.
    requires_oracle_ood_context = True
    emits_oracle_gradient_diagnostics = False

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = validate_oracle_consensus_ramen_config(args.config)
        self.beta = self.cfg["beta"]
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.dtype = torch.half
        self.model = CLIPModelForBySampleTTA(model, datasets.classes, self.cfg, args)
        self.feat_dim = self.model.feat_dim
        self.grad_dim = self.model.grad_dim
        self.loss_fn = lambda logits: softmax_entropy(logits, reduction="sum")
        self.cache = [
            PriorityCache(self.cfg["max_capacity"], self.feat_dim, self.grad_dim, self.device, self.dtype)
            for _ in range(self.num_classes)
        ]
        self.counter = 0
        self._initialize_oracle_ood_hook()
        self.last_diagnostics = {}

    @property
    def memory_size(self) -> int:
        return sum(cache.size for cache in self.cache)

    @property
    def memory_bytes(self) -> int:
        """Bytes occupied by the retained ID-only Ramen-compatible supports."""
        return sum(
            cache.size * (
                cache.keys.shape[1] * cache.keys.element_size()
                + cache.values.shape[1] * cache.values.element_size()
                + cache.priorities.element_size()
                + cache.entropies.element_size()
            )
            for cache in self.cache
        )

    def forward(self, x):
        batch_size = x.shape[0]
        is_ood = self._consume_oracle_is_ood(batch_size, self.device)
        features = self.model.featurize(x)
        logits = self.model.classify(features)
        predicted_classes = logits.argmax(-1)
        self.loss_fn(logits).backward()
        gradients = self.model.get_by_sample_grad()

        with torch.no_grad():
            priorities = torch.arange(self.counter, self.counter + batch_size, device=self.device, dtype=self.dtype)
            entropies = softmax_entropy(logits, reduction="none")
            self.counter += batch_size
            # Keep ConsensusRamen's batch-atomic visibility and retrieval
            # mechanics.  The oracle sole deviation is admission: OOD support
            # gradients never become consensus votes or cache entries.
            for index in range(batch_size):
                if bool(is_ood[index]):
                    continue
                predicted_class = int(predicted_classes[index])
                self.cache[predicted_class].add(
                    features[index].unsqueeze(0), gradients[index].unsqueeze(0),
                    entropies[index].unsqueeze(0), priorities[index].unsqueeze(0),
                )
            safe_gradients, _, consensus_diagnostics = aggregate_consensus_supports(
                features, self.cache, topk=self.cfg["topk"], beta=self.beta,
                consensus_threshold=self.cfg["consensus_threshold"],
                min_consensus_classes=self.cfg["min_consensus_classes"],
            )
            self.model.set_by_sample_grad(safe_gradients)
            self.last_diagnostics = {
                "memory_size": torch.full((batch_size,), self.memory_size, device=self.device, dtype=torch.long),
                "memory_bytes": torch.full((batch_size,), self.memory_bytes, device=self.device, dtype=torch.long),
                "pre_adaptation_ood_score": -torch.logsumexp(logits.detach(), dim=1),
                "oracle_ood_source": _ORACLE_OOD_SOURCE,
                **consensus_diagnostics,
            }

        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        self.model.reset_parameters()
        return output

    def reset(self):
        self._clear_oracle_ood_context()
        self.model.reset_parameters()
        self.counter = 0
        for cache in self.cache:
            cache.reset()
        self.last_diagnostics = {}

    def get_diagnostics(self):
        return dict(self.last_diagnostics)
