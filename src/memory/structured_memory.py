"""Class- and context-indexed gradient memory.

The memory intentionally has no dependency on a TTA method.  It stores the
evidence produced for previous samples and returns a class-balanced support
set for a query's inferred context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class _MemoryItem:
    feature: torch.Tensor
    gradient: torch.Tensor
    entropy: torch.Tensor
    recency: torch.Tensor
    reliability: torch.Tensor
    item_id: torch.Tensor


@dataclass(frozen=True)
class RetrievalBatch:
    """Padded, class-balanced retrieval results.

    All tensors have shape ``[batch, num_classes, topk, ...]`` except
    ``valid_mask``, whose shape is ``[batch, num_classes, topk]``.  Empty
    class/context buckets are represented by ``False`` entries in the mask.
    Distances and metadata are float32/int64 so half-precision cache storage
    cannot reduce their ordering precision.
    """

    features: torch.Tensor
    gradients: torch.Tensor
    entropies: torch.Tensor
    recencies: torch.Tensor
    reliabilities: torch.Tensor
    item_ids: torch.Tensor
    distances: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class FlatRetrievalBatch:
    """Padded top-``k`` results for a single, unbalanced support pool.

    This is intentionally separate from :class:`RetrievalBatch`: ablations
    that remove class balance must not accidentally average a padded class
    dimension.  ``valid_mask`` has shape ``[batch, topk]``.
    """

    features: torch.Tensor
    gradients: torch.Tensor
    entropies: torch.Tensor
    recencies: torch.Tensor
    reliabilities: torch.Tensor
    item_ids: torch.Tensor
    distances: torch.Tensor
    valid_mask: torch.Tensor


class StructuredGradientMemory:
    """A bounded cache indexed by ``(predicted_class, inferred_context)``.

    ``capacity_scope='per_class'`` bounds the aggregate contents of every
    predicted class across all of its context buckets, matching Ramen's
    per-class cache capacity.  ``'per_class_context'`` keeps the original
    structured-memory behavior where each bucket has its own bound.
    """

    _CAPACITY_SCOPES = frozenset(("per_class", "per_class_context"))

    def __init__(
        self,
        num_classes: int,
        max_capacity: int,
        feature_dim: int,
        gradient_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        capacity_scope: str = "per_class_context",
    ) -> None:
        for name, value in {
            "num_classes": num_classes,
            "max_capacity": max_capacity,
            "feature_dim": feature_dim,
            "gradient_dim": gradient_dim,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        if not isinstance(capacity_scope, str) or capacity_scope not in self._CAPACITY_SCOPES:
            allowed = ", ".join(sorted(self._CAPACITY_SCOPES))
            raise ValueError(f"capacity_scope must be one of: {allowed}")

        self.num_classes = num_classes
        self.max_capacity = max_capacity
        self.feature_dim = feature_dim
        self.gradient_dim = gradient_dim
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = dtype
        self.capacity_scope = capacity_scope
        self._buckets: Dict[Tuple[int, int], List[_MemoryItem]] = {}
        self._active_ids: set[int] = set()
        self._size = 0
        self._per_class_sizes = {predicted_class: 0 for predicted_class in range(num_classes)}
        self._context_bucket_counts: Dict[int, int] = {}
        self._retained_bytes = 0
        self._next_item_id = 0
        self._next_recency = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def active_contexts(self) -> int:
        return len(self._context_bucket_counts)

    @property
    def retained_bytes(self) -> int:
        """Exact bytes held by live item tensors, maintained in O(1)."""
        return self._retained_bytes

    @property
    def per_class_sizes(self) -> dict[int, int]:
        """Return a snapshot of maintained per-class live-item counts."""
        return dict(self._per_class_sizes)

    def diagnostics(self) -> dict[str, int | str | dict[int, int]]:
        """Return live-item counts and exact tensor bytes currently retained."""
        return {
            "size": self.size,
            "active_contexts": self.active_contexts,
            "active_buckets": len(self._buckets),
            "bytes": self.retained_bytes,
            "capacity_scope": self.capacity_scope,
            "max_capacity": self.max_capacity,
            "per_class_sizes": self.per_class_sizes,
        }

    def add(
        self,
        features: torch.Tensor,
        gradients: torch.Tensor,
        predicted_classes: torch.Tensor | int,
        contexts: torch.Tensor | int,
        entropies: torch.Tensor | float,
        *,
        reliabilities: torch.Tensor | float | None = None,
        item_ids: torch.Tensor | int | None = None,
        recencies: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Add a batch and return its stable int64 item IDs.

        ``item_ids`` is useful when callers have their own stream IDs.  IDs
        must be unique among active items; omitting them allocates monotonic
        IDs.  ``recencies`` is primarily provided for deterministic replay.
        """
        features = self._matrix(features, "features", self.feature_dim)
        gradients = self._matrix(gradients, "gradients", self.gradient_dim)
        if features.shape[0] != gradients.shape[0]:
            raise ValueError("features and gradients must have the same batch size")
        batch_size = features.shape[0]
        classes = self._integer_vector(predicted_classes, "predicted_classes", batch_size)
        contexts = self._integer_vector(contexts, "contexts", batch_size)
        if bool(((classes < 0) | (classes >= self.num_classes)).any()):
            raise ValueError("predicted_classes contains an out-of-range class")
        if bool((contexts < 0).any()):
            raise ValueError("contexts must be non-negative")
        entropies = self._float_vector(entropies, "entropies", batch_size)
        if not bool(torch.isfinite(entropies).all()):
            raise ValueError("entropies must be finite")
        if reliabilities is None:
            reliabilities = torch.ones(batch_size, device=self.device, dtype=torch.float32)
        else:
            reliabilities = self._float_vector(reliabilities, "reliabilities", batch_size)
            if not bool(torch.isfinite(reliabilities).all()):
                raise ValueError("reliabilities must be finite")

        if item_ids is None:
            ids = torch.arange(self._next_item_id, self._next_item_id + batch_size,
                               device=self.device, dtype=torch.long)
        else:
            ids = self._integer_vector(item_ids, "item_ids", batch_size)
            if bool((ids < 0).any()):
                raise ValueError("item_ids must be non-negative")
        id_values = ids.cpu().tolist()
        if len(set(id_values)) != batch_size or any(item_id in self._active_ids for item_id in id_values):
            raise ValueError("item_ids must be unique among active memory items")

        if recencies is None:
            recencies = torch.arange(self._next_recency, self._next_recency + batch_size,
                                     device=self.device, dtype=torch.long)
        else:
            recencies = self._integer_vector(recencies, "recencies", batch_size)
        if bool((recencies < 0).any()):
            raise ValueError("recencies must be non-negative")

        # All input is validated before mutating the cache.
        for index in range(batch_size):
            predicted_class = int(classes[index])
            bucket_key = (predicted_class, int(contexts[index]))
            item = _MemoryItem(
                feature=features[index].clone(),
                gradient=gradients[index].clone(),
                entropy=entropies[index].clone(),
                recency=recencies[index].clone(),
                reliability=reliabilities[index].clone(),
                item_id=ids[index].clone(),
            )
            if self.capacity_scope == "per_class":
                self._evict_oldest_for_class_if_full(predicted_class)
                self._append_item(bucket_key, item)
            else:
                bucket = self._buckets.get(bucket_key, [])
                if len(bucket) >= self.max_capacity:
                    oldest_index = min(range(len(bucket)), key=lambda i: int(bucket[i].recency))
                    self._remove_item(bucket_key, oldest_index)
                self._append_item(bucket_key, item)

        self._next_item_id = max(self._next_item_id, max(id_values) + 1)
        self._next_recency = max(self._next_recency, max(recencies.cpu().tolist()) + 1)
        return ids.clone()

    def query(
        self,
        features: torch.Tensor,
        contexts: torch.Tensor | int,
        topk: int,
        *,
        include_current: bool = True,
        current_item_ids: torch.Tensor | int | None = None,
    ) -> RetrievalBatch:
        """Retrieve the nearest ``topk`` per class from the query's context.

        With ``include_current=False``, ``current_item_ids`` is required and
        entries with the same stable ID are excluded before ranking.  This
        makes self-gradient and historical-memory ablations unambiguous.
        """
        if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
            raise ValueError("topk must be a positive integer")
        features = self._matrix(features, "features", self.feature_dim)
        batch_size = features.shape[0]
        contexts = self._integer_vector(contexts, "contexts", batch_size)
        if bool((contexts < 0).any()):
            raise ValueError("contexts must be non-negative")
        if not include_current and current_item_ids is None:
            raise ValueError("current_item_ids is required when include_current=False")
        current_ids = None if current_item_ids is None else self._integer_vector(
            current_item_ids, "current_item_ids", batch_size
        )

        shape = (batch_size, self.num_classes, topk)
        result = RetrievalBatch(
            features=torch.zeros(*shape, self.feature_dim, device=self.device, dtype=self.dtype),
            gradients=torch.zeros(*shape, self.gradient_dim, device=self.device, dtype=self.dtype),
            entropies=torch.zeros(shape, device=self.device, dtype=torch.float32),
            recencies=torch.zeros(shape, device=self.device, dtype=torch.long),
            reliabilities=torch.zeros(shape, device=self.device, dtype=torch.float32),
            item_ids=torch.full(shape, -1, device=self.device, dtype=torch.long),
            distances=torch.full(shape, float("inf"), device=self.device, dtype=torch.float32),
            valid_mask=torch.zeros(shape, device=self.device, dtype=torch.bool),
        )

        for batch_index in range(batch_size):
            context = int(contexts[batch_index])
            for predicted_class in range(self.num_classes):
                candidates = self._buckets.get((predicted_class, context), [])
                if current_ids is not None and not include_current:
                    candidates = [item for item in candidates if int(item.item_id) != int(current_ids[batch_index])]
                if not candidates:
                    continue
                candidate_features = torch.stack([item.feature for item in candidates]).float()
                distances = torch.linalg.vector_norm(candidate_features - features[batch_index].float(), dim=1)
                order = torch.argsort(distances, stable=True)[:topk]
                for rank, candidate_index in enumerate(order.tolist()):
                    item = candidates[candidate_index]
                    result.features[batch_index, predicted_class, rank] = item.feature
                    result.gradients[batch_index, predicted_class, rank] = item.gradient
                    result.entropies[batch_index, predicted_class, rank] = item.entropy
                    result.recencies[batch_index, predicted_class, rank] = item.recency
                    result.reliabilities[batch_index, predicted_class, rank] = item.reliability
                    result.item_ids[batch_index, predicted_class, rank] = item.item_id
                    result.distances[batch_index, predicted_class, rank] = distances[candidate_index]
                    result.valid_mask[batch_index, predicted_class, rank] = True
        return result

    def query_candidate_counts(
        self,
        contexts: torch.Tensor | int,
        *,
        current_item_ids: torch.Tensor | int | None = None,
        include_current: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return live and context-eligible support counts without ranking.

        This intentionally reports only cache state that is available to a
        retrieval query.  It is used by opt-in profiling and never affects
        support selection.
        """
        if not include_current and current_item_ids is None:
            raise ValueError("current_item_ids is required when include_current=False")
        batch_size = 1 if isinstance(contexts, int) else (1 if contexts.ndim == 0 else contexts.numel())
        contexts = self._integer_vector(contexts, "contexts", batch_size)
        current_ids = None if current_item_ids is None else self._integer_vector(
            current_item_ids, "current_item_ids", batch_size
        )
        live = torch.full((batch_size,), self.size, device=self.device, dtype=torch.long)
        eligible = []
        for index, context in enumerate(contexts.tolist()):
            items = [item for (klass, bucket_context), bucket in self._buckets.items()
                     if bucket_context == context for item in bucket]
            if current_ids is not None and not include_current:
                items = [item for item in items if int(item.item_id) != int(current_ids[index])]
            eligible.append(len(items))
        return live, torch.tensor(eligible, device=self.device, dtype=torch.long)

    def query_flat(
        self,
        features: torch.Tensor,
        topk: int,
        *,
        selection: str,
        contexts: torch.Tensor | int | None = None,
        predicted_classes: torch.Tensor | int | None = None,
        include_current: bool = True,
        current_item_ids: torch.Tensor | int | None = None,
        random_seed: int = 0,
    ) -> FlatRetrievalBatch:
        """Retrieve an unbalanced support set for a controlled ablation.

        ``selection`` is one of ``random``, ``same_class``, ``global_nearest``
        or ``context_nearest``.  The nearest modes return the globally closest
        eligible ``topk`` entries (not ``topk`` per class).  Random selection
        is deterministic for a seed and query item ID, and samples without
        replacement from only items already present in the cache.
        """
        if selection not in {"random", "same_class", "global_nearest", "context_nearest"}:
            raise ValueError("unknown flat retrieval selection")
        if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
            raise ValueError("topk must be a positive integer")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool):
            raise ValueError("random_seed must be an integer")
        features = self._matrix(features, "features", self.feature_dim)
        batch_size = features.shape[0]
        if selection == "same_class":
            if predicted_classes is None:
                raise ValueError("predicted_classes is required for same_class retrieval")
            predicted_classes = self._integer_vector(predicted_classes, "predicted_classes", batch_size)
            if bool(((predicted_classes < 0) | (predicted_classes >= self.num_classes)).any()):
                raise ValueError("predicted_classes contains an out-of-range class")
        if selection == "context_nearest":
            if contexts is None:
                raise ValueError("contexts is required for context_nearest retrieval")
            contexts = self._integer_vector(contexts, "contexts", batch_size)
            if bool((contexts < 0).any()):
                raise ValueError("contexts must be non-negative")
        if not include_current and current_item_ids is None:
            raise ValueError("current_item_ids is required when include_current=False")
        current_ids = None if current_item_ids is None else self._integer_vector(
            current_item_ids, "current_item_ids", batch_size
        )
        shape = (batch_size, topk)
        result = FlatRetrievalBatch(
            features=torch.zeros(*shape, self.feature_dim, device=self.device, dtype=self.dtype),
            gradients=torch.zeros(*shape, self.gradient_dim, device=self.device, dtype=self.dtype),
            entropies=torch.zeros(shape, device=self.device, dtype=torch.float32),
            recencies=torch.zeros(shape, device=self.device, dtype=torch.long),
            reliabilities=torch.zeros(shape, device=self.device, dtype=torch.float32),
            item_ids=torch.full(shape, -1, device=self.device, dtype=torch.long),
            distances=torch.full(shape, float("inf"), device=self.device, dtype=torch.float32),
            valid_mask=torch.zeros(shape, device=self.device, dtype=torch.bool),
        )
        # Sort by stable ID so ties and random ranking are independent of
        # dictionary/bucket insertion details.
        all_items = sorted(
            ((predicted_class, context, item) for (predicted_class, context), bucket in self._buckets.items()
             for item in bucket),
            key=lambda candidate: int(candidate[2].item_id),
        )
        for batch_index in range(batch_size):
            candidates = all_items
            if selection == "same_class":
                candidates = [candidate for candidate in candidates if candidate[0] == int(predicted_classes[batch_index])]
            elif selection == "context_nearest":
                candidates = [candidate for candidate in candidates if candidate[1] == int(contexts[batch_index])]
            if current_ids is not None and not include_current:
                candidates = [candidate for candidate in candidates if int(candidate[2].item_id) != int(current_ids[batch_index])]
            if not candidates:
                continue
            candidate_features = torch.stack([candidate[2].feature for candidate in candidates]).float()
            distances = torch.linalg.vector_norm(candidate_features - features[batch_index].float(), dim=1)
            if selection == "random":
                query_id = batch_index if current_ids is None else int(current_ids[batch_index])
                order = sorted(
                    range(len(candidates)),
                    key=lambda index: self._random_rank(random_seed, query_id, int(candidates[index][2].item_id)),
                )[:topk]
            else:
                order = torch.argsort(distances, stable=True)[:topk].tolist()
            for rank, candidate_index in enumerate(order):
                item = candidates[candidate_index][2]
                result.features[batch_index, rank] = item.feature
                result.gradients[batch_index, rank] = item.gradient
                result.entropies[batch_index, rank] = item.entropy
                result.recencies[batch_index, rank] = item.recency
                result.reliabilities[batch_index, rank] = item.reliability
                result.item_ids[batch_index, rank] = item.item_id
                result.distances[batch_index, rank] = distances[candidate_index]
                result.valid_mask[batch_index, rank] = True
        return result

    @staticmethod
    def _random_rank(seed: int, query_id: int, item_id: int) -> int:
        """A small platform-independent mixer for deterministic random support."""
        value = (seed & ((1 << 64) - 1)) ^ ((query_id * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
        value ^= (item_id * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)

    def reset(self) -> None:
        """Clear all retained evidence and restart generated IDs/recencies."""
        self._buckets.clear()
        self._active_ids.clear()
        self._size = 0
        self._per_class_sizes = {
            predicted_class: 0 for predicted_class in range(self.num_classes)
        }
        self._context_bucket_counts.clear()
        self._retained_bytes = 0
        self._next_item_id = 0
        self._next_recency = 0

    def _evict_oldest_for_class_if_full(self, predicted_class: int) -> None:
        """Evict one globally oldest item when a per-class cache is full."""
        if self._per_class_sizes[predicted_class] < self.max_capacity:
            return
        candidates = (
            (bucket_key, index, item)
            for bucket_key, bucket in self._buckets.items()
            if bucket_key[0] == predicted_class
            for index, item in enumerate(bucket)
        )
        bucket_key, item_index, _ = min(
            candidates, key=lambda candidate: (int(candidate[2].recency), int(candidate[2].item_id))
        )
        self._remove_item(bucket_key, item_index)

    def _append_item(self, bucket_key: Tuple[int, int], item: _MemoryItem) -> None:
        """Append one validated item and update all live-memory counters."""
        predicted_class, context = bucket_key
        bucket = self._buckets.get(bucket_key)
        if bucket is None:
            bucket = self._buckets[bucket_key] = []
            self._context_bucket_counts[context] = self._context_bucket_counts.get(context, 0) + 1
        bucket.append(item)
        self._active_ids.add(int(item.item_id))
        self._size += 1
        self._per_class_sizes[predicted_class] += 1
        self._retained_bytes += self._item_bytes(item)

    def _remove_item(self, bucket_key: Tuple[int, int], item_index: int) -> _MemoryItem:
        """Remove one item and update IDs, size, context, and byte counters."""
        predicted_class, context = bucket_key
        bucket = self._buckets[bucket_key]
        item = bucket.pop(item_index)
        self._active_ids.remove(int(item.item_id))
        self._size -= 1
        self._per_class_sizes[predicted_class] -= 1
        self._retained_bytes -= self._item_bytes(item)
        if not bucket:
            del self._buckets[bucket_key]
            remaining = self._context_bucket_counts[context] - 1
            if remaining:
                self._context_bucket_counts[context] = remaining
            else:
                del self._context_bucket_counts[context]
        return item

    @staticmethod
    def _item_bytes(item: _MemoryItem) -> int:
        return (
            item.feature.numel() * item.feature.element_size()
            + item.gradient.numel() * item.gradient.element_size()
            + item.entropy.numel() * item.entropy.element_size()
            + item.recency.numel() * item.recency.element_size()
            + item.reliability.numel() * item.reliability.element_size()
            + item.item_id.numel() * item.item_id.element_size()
        )

    def _matrix(self, value: torch.Tensor, name: str, width: int) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{name} must have shape [batch, {width}]")
        if not value.dtype.is_floating_point:
            raise ValueError(f"{name} must have a floating-point dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return value.detach().to(device=self.device, dtype=self.dtype)

    def _integer_vector(self, value: torch.Tensor | int, name: str, batch_size: int) -> torch.Tensor:
        if isinstance(value, int) and not isinstance(value, bool):
            value = torch.tensor([value], device=self.device, dtype=torch.long)
        elif isinstance(value, torch.Tensor):
            if value.ndim == 0:
                value = value.reshape(1)
            if value.ndim != 1 or value.numel() not in (1, batch_size):
                raise ValueError(f"{name} must have one value or one value per batch item")
            if value.dtype.is_floating_point and not bool(torch.equal(value, value.round())):
                raise ValueError(f"{name} must contain integers")
            value = value.detach().to(device=self.device, dtype=torch.long)
        else:
            raise TypeError(f"{name} must be an integer or torch.Tensor")
        return value.expand(batch_size) if value.numel() == 1 else value

    def _float_vector(self, value: torch.Tensor | float, name: str, batch_size: int) -> torch.Tensor:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = torch.tensor([value], device=self.device, dtype=torch.float32)
        elif isinstance(value, torch.Tensor):
            if value.ndim == 0:
                value = value.reshape(1)
            if value.ndim != 1 or value.numel() not in (1, batch_size):
                raise ValueError(f"{name} must have one value or one value per batch item")
            if not value.dtype.is_floating_point:
                raise ValueError(f"{name} must have a floating-point dtype")
            value = value.detach().to(device=self.device, dtype=torch.float32)
        else:
            raise TypeError(f"{name} must be a number or torch.Tensor")
        return value.expand(batch_size) if value.numel() == 1 else value
