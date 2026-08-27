"""Causal support-selection ablations for Ramen.

The five exported methods isolate *which previous items* may support an
update while retaining Ramen's entropy and feature-distance weighting:

* ``RandomMemoryRamen`` samples from all retained historical/current items;
* ``SameClassRamen`` uses only the query's predicted class;
* ``GlobalNearestRamen`` uses nearest neighbours across every class/context;
* ``ContextOnlyRamen`` uses nearest neighbours in the inferred context,
  without balancing predicted classes.
* ``CausalRamen`` uses class-balanced nearest neighbours at fixed context 0.
* ``StructuredAtomicRamen`` is CausalRamen with only its batch schedule
  changed: all evaluator-batch items are admitted before any item is queried.

The original :class:`methods.Ramen.Ramen` remains the legacy batch-atomic
reference: for ``B > 1`` it inserts a whole batch before any query.  The
strictly sequential ``CausalRamen`` is therefore the fair
``class-balanced-without-context-routing`` control for these ablations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from memory.structured_memory import FlatRetrievalBatch, StructuredGradientMemory
from models.ModelForBySampleTTA import CLIPModelForBySampleTTA
from routing.online_prototypes import OnlinePrototypeRouter

from .LatentRamen import aggregate_class_balanced_gradients, validate_latent_ramen_config
from .TTABase import TTABase
from .losses import softmax_entropy


_SELECTIONS = frozenset(("random", "same_class", "global_nearest", "context_nearest", "class_balanced"))
_CONTEXT_SELECTION = "context_nearest"
_RETRIEVAL_SCHEDULES = frozenset(("causal", "batch_atomic"))


def validate_support_ablation_config(config: Mapping[str, Any], *, selection: str) -> dict[str, Any]:
    """Validate shared Ramen settings and the selection-specific controls."""
    if selection not in _SELECTIONS:
        raise ValueError("unknown support selection")
    cfg = validate_latent_ramen_config(config)
    # These baselines use a single historical cache bounded the same way as
    # original Ramen: a class's capacity is shared across all of its buckets.
    if cfg["capacity_scope"] != "per_class":
        raise ValueError("support-selection ablations require capacity_scope='per_class'")
    if selection == "random":
        seed = cfg.get("random_seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("random_seed must be an integer")
        cfg["random_seed"] = seed
    if selection == _CONTEXT_SELECTION:
        # validate_latent_ramen_config already supplies and validates router
        # controls, which are meaningful only for this baseline.
        return cfg
    return cfg


def aggregate_unbalanced_gradients(retrieval: FlatRetrievalBatch, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Ramen entropy/distance weights over one unbalanced support pool."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    valid = retrieval.valid_mask
    distance_weights = torch.where(
        valid, torch.exp(-float(beta) * retrieval.distances), torch.zeros_like(retrieval.distances)
    )
    weights = torch.exp(-retrieval.entropies) * distance_weights
    summed = (retrieval.gradients * weights.to(retrieval.gradients.dtype).unsqueeze(-1)).sum(dim=1)
    counts = valid.sum(dim=1)
    return summed / counts.clamp_min(1).to(summed.dtype).unsqueeze(-1), counts


def causal_active_context_counts(contexts_before_batch: int, spawned: torch.Tensor) -> torch.Tensor:
    """Return the active-router-context count after each sequential item."""
    if not isinstance(contexts_before_batch, int) or isinstance(contexts_before_batch, bool) \
            or contexts_before_batch < 0:
        raise ValueError("contexts_before_batch must be a non-negative integer")
    if not isinstance(spawned, torch.Tensor) or spawned.ndim != 1 or spawned.dtype != torch.bool:
        raise ValueError("spawned must be a one-dimensional boolean tensor")
    return contexts_before_batch + spawned.long().cumsum(dim=0)


def update_and_retrieve_support_causal_batch(
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
    selection: str,
    random_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process a batch in stream order, preventing all future-item reads.

    Historical-only retrieval occurs before insertion.  This matters at full
    capacity: inserting the current item first could evict the very history
    the query is meant to use.  The current item is inserted immediately
    afterwards, so externally visible state still advances one item at a time.
    """
    retrieved, support_counts, memory_sizes, memory_bytes = [], [], [], []
    for index in range(features.shape[0]):
        current = slice(index, index + 1)
        kwargs: dict[str, Any] = {}
        if selection == "same_class":
            kwargs["predicted_classes"] = predicted_classes[current]
        elif selection == _CONTEXT_SELECTION:
            kwargs["contexts"] = contexts[current]
        if include_current:
            memory.add(features[current], gradients[current], predicted_classes[current], contexts[current],
                       entropies[current], item_ids=item_ids[current])
        if selection == "class_balanced":
            support = memory.query(
                features[current], contexts[current], topk, include_current=include_current,
                current_item_ids=item_ids[current],
            )
            item_gradient, item_count = aggregate_class_balanced_gradients(support, beta)
        else:
            support = memory.query_flat(
                features[current], topk, selection=selection, include_current=include_current,
                current_item_ids=item_ids[current], random_seed=random_seed, **kwargs,
            )
            item_gradient, item_count = aggregate_unbalanced_gradients(support, beta)
        if not include_current:
            memory.add(features[current], gradients[current], predicted_classes[current], contexts[current],
                       entropies[current], item_ids=item_ids[current])
        retrieved.append(item_gradient)
        support_counts.append(item_count)
        memory_sizes.append(memory.size)
        memory_bytes.append(memory.retained_bytes)
    return (
        torch.cat(retrieved, dim=0),
        torch.cat(support_counts, dim=0),
        torch.tensor(memory_sizes, device=features.device, dtype=torch.long),
        torch.tensor(memory_bytes, device=features.device, dtype=torch.long),
    )


def update_and_retrieve_support_atomic_batch(
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
    selection: str,
    random_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Admit an evaluator batch before querying any of its items.

    This is intentionally a scheduling-only counterpart to the causal helper:
    storage, selection, ranking, aggregation, and exclusion-by-stable-ID are
    all shared.  Consequently, when ``include_current=False`` each query
    excludes only itself and may still use the other (including later) items
    admitted from the same evaluator batch.
    """
    memory.add(features, gradients, predicted_classes, contexts, entropies, item_ids=item_ids)
    kwargs: dict[str, Any] = {}
    if selection == "same_class":
        kwargs["predicted_classes"] = predicted_classes
    elif selection == _CONTEXT_SELECTION:
        kwargs["contexts"] = contexts
    if selection == "class_balanced":
        support = memory.query(
            features, contexts, topk, include_current=include_current, current_item_ids=item_ids,
        )
        retrieved, support_counts = aggregate_class_balanced_gradients(support, beta)
    else:
        support = memory.query_flat(
            features, topk, selection=selection, include_current=include_current,
            current_item_ids=item_ids, random_seed=random_seed, **kwargs,
        )
        retrieved, support_counts = aggregate_unbalanced_gradients(support, beta)
    batch_size = features.shape[0]
    return (
        retrieved,
        support_counts,
        torch.full((batch_size,), memory.size, device=features.device, dtype=torch.long),
        torch.full((batch_size,), memory.retained_bytes, device=features.device, dtype=torch.long),
    )


def update_and_retrieve_support_batch(
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
    selection: str,
    schedule: str,
    random_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch retrieval through the method's explicit scheduling invariant.

    Atomic scheduling always admits the complete evaluator batch before its
    first query, including singleton batches.
    """
    if schedule not in _RETRIEVAL_SCHEDULES:
        raise ValueError("unknown retrieval schedule")
    helper = (
        update_and_retrieve_support_causal_batch
        if schedule == "causal"
        else update_and_retrieve_support_atomic_batch
    )
    return helper(
        memory, features, gradients, predicted_classes, contexts, entropies, item_ids,
        topk=topk, include_current=include_current, beta=beta, selection=selection,
        random_seed=random_seed,
    )


class SupportSelectionRamen(TTABase):
    """Parameterized implementation shared by manifest-visible ablations."""

    support_selection: str = "global_nearest"
    retrieval_schedule: str = "causal"

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = validate_support_ablation_config(args.config, selection=self.support_selection)
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.model = CLIPModelForBySampleTTA(model, datasets.classes, self.cfg, args)
        self.router = None
        if self.support_selection == _CONTEXT_SELECTION:
            self.router = OnlinePrototypeRouter(
                spawn_threshold=self.cfg["spawn_threshold"], max_contexts=self.cfg["max_contexts"],
                temperature=self.cfg["router_temperature"], momentum=self.cfg["router_momentum"],
            )
        cache_dtype = self.model.dtype if self.model.dtype.is_floating_point else torch.float32
        self.memory = StructuredGradientMemory(
            self.num_classes, self.cfg["max_capacity"], self.model.feat_dim, self.model.grad_dim,
            device=self.device, dtype=cache_dtype, capacity_scope=self.cfg["capacity_scope"],
        )
        self.loss_fn = lambda logits: softmax_entropy(logits, reduction="sum")
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
            routing = None
            if self.router is None:
                contexts = torch.zeros(batch_size, device=self.device, dtype=torch.long)
                # Every non-routing baseline uses one explicit fixed context
                # for every processed item.  Before the first forward there is
                # no observed context, but its per-item trace is consistently 1.
                active_contexts = torch.ones(batch_size, device=self.device, dtype=torch.long)
            else:
                contexts_before_batch = self.router.num_contexts
                routing = self.router.route(features, update=True)
                contexts = routing.context_ids
                active_contexts = causal_active_context_counts(
                    contexts_before_batch, routing.spawned
                )
            entropies = softmax_entropy(logits, reduction="none")
            item_ids = torch.arange(self.counter, self.counter + batch_size, device=self.device, dtype=torch.long)
            self.counter += batch_size
            retrieved, support_counts, memory_sizes, memory_bytes = update_and_retrieve_support_batch(
                self.memory, features, gradients, predicted_classes, contexts, entropies, item_ids,
                topk=self.cfg["topk"], include_current=self.cfg["include_current"], beta=self.cfg["beta"],
                selection=self.support_selection, schedule=self.retrieval_schedule,
                random_seed=self.cfg.get("random_seed", 0),
            )
            self.model.set_by_sample_grad(retrieved)
            self.last_diagnostics = self._diagnostics(
                routing, contexts, support_counts, memory_sizes, memory_bytes, active_contexts
            )
        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        self.model.reset_parameters()
        return output

    def reset(self):
        self.model.reset_parameters()
        self.memory.reset()
        if self.router is not None:
            self.router.reset()
        self.counter = 0
        self.last_diagnostics = self._diagnostics()

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)

    def diagnostics(self) -> dict[str, Any]:
        return self.get_diagnostics()

    def _diagnostics(
        self, routing=None, contexts=None, support_counts=None, memory_sizes=None, memory_bytes=None,
        active_contexts=None,
    ):
        return {
            "support_selection": self.support_selection,
            "memory_size": self.memory.size if memory_sizes is None else memory_sizes.detach().clone(),
            "memory_bytes": self.memory.retained_bytes if memory_bytes is None else memory_bytes.detach().clone(),
            "memory_active_contexts": self.memory.active_contexts,
            "capacity_scope": self.memory.capacity_scope,
            "memory_capacity_scope": self.memory.capacity_scope,
            "memory_max_capacity": self.memory.max_capacity,
            "inferred_context": None if contexts is None else contexts.detach().clone(),
            "num_active_contexts": (
                (0 if self.router is None else self.router.num_contexts)
                if active_contexts is None else active_contexts.detach().clone()
            ),
            "support_count": None if support_counts is None else support_counts.detach().clone(),
            "spawned": None if routing is None else routing.spawned.detach().clone(),
        }


class RandomMemoryRamen(SupportSelectionRamen):
    support_selection = "random"


class SameClassRamen(SupportSelectionRamen):
    support_selection = "same_class"


class GlobalNearestRamen(SupportSelectionRamen):
    support_selection = "global_nearest"


class ContextOnlyRamen(SupportSelectionRamen):
    support_selection = "context_nearest"


class CausalRamen(SupportSelectionRamen):
    """Strictly causal, class-balanced Ramen without context routing."""

    support_selection = "class_balanced"


class StructuredAtomicRamen(SupportSelectionRamen):
    """Batch-atomic scheduling control for :class:`CausalRamen`."""

    support_selection = "class_balanced"
    retrieval_schedule = "batch_atomic"
