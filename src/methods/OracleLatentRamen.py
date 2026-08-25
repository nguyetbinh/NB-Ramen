"""Explicit oracle-domain diagnostic for structured-memory Ramen.

This method is an upper-bound diagnostic, not an unsupervised TTA method.  It
accepts evaluator ``domain_idx`` values only through ``set_oracle_domain_context``
immediately before a forward pass.  The single-use hook deliberately fails
closed so a context cannot silently leak between batches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from memory.structured_memory import StructuredGradientMemory
from models.ModelForBySampleTTA import CLIPModelForBySampleTTA

from .LatentRamen import aggregate_class_balanced_gradients, validate_latent_ramen_config
from .TTABase import TTABase
from .losses import softmax_entropy


_ORACLE_CONTEXT_SOURCE = "evaluator_domain_idx"


def validate_oracle_latent_ramen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the shared Ramen settings plus explicit oracle provenance."""
    cfg = validate_latent_ramen_config(config)
    if cfg.get("oracle_context_source") != _ORACLE_CONTEXT_SOURCE:
        raise ValueError(
            "OracleLatentRamen requires oracle_context_source='evaluator_domain_idx'"
        )
    return cfg


class OracleDomainContextHook:
    """A one-shot, fail-closed hand-off for evaluator-only domain IDs."""

    requires_oracle_domain_context = True

    def _initialize_oracle_context_hook(self) -> None:
        self._pending_oracle_domain_context: torch.Tensor | None = None
        self._seen_oracle_contexts: set[int] = set()

    def set_oracle_domain_context(self, domain_idx: torch.Tensor) -> None:
        """Stage exactly one batch of evaluator domain IDs for the next forward."""
        if self._pending_oracle_domain_context is not None:
            raise RuntimeError("stale oracle domain context is pending consumption")
        if not isinstance(domain_idx, torch.Tensor):
            raise TypeError("oracle domain context must be a tensor")
        if domain_idx.ndim != 1:
            raise ValueError("oracle domain context must have shape [batch]")
        if domain_idx.numel() == 0:
            raise ValueError("oracle domain context must not be empty")
        if domain_idx.dtype.is_floating_point or domain_idx.dtype == torch.bool:
            raise ValueError("oracle domain context must contain integer IDs")
        domain_idx = domain_idx.detach().to(dtype=torch.long)
        if bool((domain_idx < 0).any()):
            raise ValueError("oracle domain context IDs must be non-negative")
        self._pending_oracle_domain_context = domain_idx.clone()

    def _consume_oracle_domain_context(self, batch_size: int, device: torch.device) -> torch.Tensor:
        contexts = self._pending_oracle_domain_context
        # Clear before validation/use: a failed or interrupted forward must
        # never leave a previous evaluator label available to a later batch.
        self._pending_oracle_domain_context = None
        if contexts is None:
            raise RuntimeError("missing oracle domain context for this forward batch")
        if contexts.numel() != batch_size:
            raise RuntimeError(
                "oracle domain context batch size does not match the forward batch"
            )
        return contexts.to(device=device, dtype=torch.long)

    def _clear_oracle_domain_context(self) -> None:
        self._pending_oracle_domain_context = None


def update_and_retrieve_oracle_causal_batch(
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
    seen_contexts: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causally insert/retrieve with exact per-item retained-memory evidence."""
    if seen_contexts is None:
        seen_contexts = set()
    retrieved, class_counts, memory_sizes, memory_bytes, active_contexts = [], [], [], [], []
    for index in range(features.shape[0]):
        item_slice = slice(index, index + 1)
        memory.add(
            features[item_slice], gradients[item_slice], predicted_classes[item_slice],
            contexts[item_slice], entropies[item_slice], item_ids=item_ids[item_slice],
        )
        seen_contexts.add(int(contexts[index]))
        support = memory.query(
            features[item_slice], contexts[item_slice], topk,
            include_current=include_current, current_item_ids=item_ids[item_slice],
        )
        item_gradient, item_class_count = aggregate_class_balanced_gradients(support, beta)
        retrieved.append(item_gradient)
        class_counts.append(item_class_count)
        memory_sizes.append(memory.size)
        memory_bytes.append(memory.retained_bytes)
        active_contexts.append(len(seen_contexts))
    return (
        torch.cat(retrieved, dim=0),
        torch.cat(class_counts, dim=0),
        torch.tensor(memory_sizes, device=features.device, dtype=torch.long),
        torch.tensor(memory_bytes, device=features.device, dtype=torch.long),
        torch.tensor(active_contexts, device=features.device, dtype=torch.long),
    )


class OracleLatentRamen(OracleDomainContextHook, TTABase):
    """Ramen support retrieval indexed by evaluator-provided oracle domains."""

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = validate_oracle_latent_ramen_config(args.config)
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.model = CLIPModelForBySampleTTA(model, datasets.classes, self.cfg, args)
        cache_dtype = self.model.dtype if self.model.dtype.is_floating_point else torch.float32
        self.memory = StructuredGradientMemory(
            self.num_classes, self.cfg["max_capacity"], self.model.feat_dim, self.model.grad_dim,
            device=self.device, dtype=cache_dtype, capacity_scope=self.cfg["capacity_scope"],
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
            item_ids = torch.arange(self.counter, self.counter + batch_size,
                                    device=self.device, dtype=torch.long)
            self.counter += batch_size
            retrieved, active_classes, memory_sizes, memory_bytes, active_contexts = update_and_retrieve_oracle_causal_batch(
                self.memory, features, gradients, predicted_classes, contexts, entropies, item_ids,
                topk=self.cfg["topk"], include_current=self.cfg["include_current"], beta=self.cfg["beta"],
                seen_contexts=self._seen_oracle_contexts,
            )
            self.model.set_by_sample_grad(retrieved)
            self.last_diagnostics = self._diagnostics(
                contexts, active_classes, memory_sizes, memory_bytes, active_contexts
            )
            self.last_diagnostics["pre_adaptation_ood_score"] = -torch.logsumexp(
                logits.detach(), dim=1
            )

        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        self.model.reset_parameters()
        return output

    def reset(self):
        # Reset is also a context boundary, including when a caller aborts a
        # batch before ``forward`` consumes its evaluator metadata.
        self._clear_oracle_domain_context()
        self.model.reset_parameters()
        self.memory.reset()
        self._seen_oracle_contexts.clear()
        self.counter = 0
        self.last_diagnostics = self._diagnostics()

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)

    def diagnostics(self) -> dict[str, Any]:
        return self.get_diagnostics()

    def _diagnostics(
        self, contexts=None, active_classes=None, memory_sizes=None,
        memory_bytes=None, active_contexts=None,
    ):
        return {
            "inferred_context": None if contexts is None else contexts.detach().clone(),
            "memory_size": self.memory.size if memory_sizes is None else memory_sizes.detach().clone(),
            "num_active_contexts": (
                len(self._seen_oracle_contexts)
                if active_contexts is None else active_contexts.detach().clone()
            ),
            "memory_active_contexts": self.memory.active_contexts,
            "memory_bytes": self.memory.retained_bytes if memory_bytes is None else memory_bytes,
            "active_classes": None if active_classes is None else active_classes.detach().clone(),
            "oracle_context_source": _ORACLE_CONTEXT_SOURCE,
            "capacity_scope": self.memory.capacity_scope,
            "memory_capacity_scope": self.memory.capacity_scope,
            "memory_max_capacity": self.memory.max_capacity,
        }
