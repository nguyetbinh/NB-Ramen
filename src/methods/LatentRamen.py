"""Online latent-context variant of Ramen.

The adaptation model is deliberately identical to :class:`Ramen`; only the
support memory changes.  CLIP visual features are routed, in stream order, to
unsupervised prototype contexts and supports are then retrieved from the
matching ``(predicted_class, context)`` bucket.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import time
from typing import Any

import torch

from evaluation.gradient_diagnostics import trace_payload_from_retrieval
from memory.structured_memory import RetrievalBatch, StructuredGradientMemory
from models.ModelForBySampleTTA import CLIPModelForBySampleTTA
from routing.online_prototypes import OnlinePrototypeRouter

from .TTABase import TTABase
from .losses import softmax_entropy


_REQUIRED_CONFIG = ("max_capacity", "topk", "optimizer", "lr")
_ROUTER_DEFAULTS = {
    "spawn_threshold": 0.25,
    "max_contexts": 8,
    "router_temperature": 0.1,
    "router_momentum": None,
    "include_current": True,
    "capacity_scope": "per_class_context",
    "retrieval_profile": "off",
    "failure_analysis_profile": "off",
}


def validate_latent_ramen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the small public configuration surface.

    Keeping this separate from model construction makes experiment configs
    testable without loading CLIP.
    """
    if not isinstance(config, Mapping):
        raise TypeError("LatentRamen config must be a mapping")
    missing = [key for key in _REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError("LatentRamen config is missing: " + ", ".join(missing))
    cfg = dict(config)
    for key, value in _ROUTER_DEFAULTS.items():
        cfg.setdefault(key, value)
    for key in ("max_capacity", "topk", "max_contexts"):
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("beta", "spawn_threshold", "router_temperature", "lr"):
        value = cfg.get(key, 0.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be a finite number")
        cfg[key] = float(value)
    if cfg["beta"] < 0 or cfg["spawn_threshold"] < 0:
        raise ValueError("beta and spawn_threshold must be non-negative")
    if cfg["router_temperature"] <= 0 or cfg["lr"] <= 0:
        raise ValueError("router_temperature and lr must be positive")
    momentum = cfg["router_momentum"]
    if momentum is not None:
        if not isinstance(momentum, (int, float)) or isinstance(momentum, bool) or not 0 <= momentum < 1:
            raise ValueError("router_momentum must be null or in [0, 1)")
        cfg["router_momentum"] = float(momentum)
    if not isinstance(cfg["include_current"], bool):
        raise ValueError("include_current must be a boolean")
    if not isinstance(cfg["capacity_scope"], str) or cfg["capacity_scope"] not in {
        "per_class", "per_class_context"
    }:
        raise ValueError("capacity_scope must be 'per_class' or 'per_class_context'")
    if not isinstance(cfg["optimizer"], str) or not cfg["optimizer"]:
        raise ValueError("optimizer must be a non-empty string")
    if not isinstance(cfg["retrieval_profile"], str) or cfg["retrieval_profile"] not in {"off", "causal_sync_v1"}:
        raise ValueError("retrieval_profile must be 'off' or 'causal_sync_v1'")
    if not isinstance(cfg["failure_analysis_profile"], str) or cfg["failure_analysis_profile"] not in {
        "off", "trace_v1", "replay_v1"
    }:
        raise ValueError("failure_analysis_profile must be 'off', 'trace_v1', or 'replay_v1'")
    return cfg


def _synchronize_for_retrieval(device: torch.device) -> None:
    """Synchronize the selected accelerator for diagnostic query timing only."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _profile_elapsed_tensor(elapsed_ms: list[float]) -> torch.Tensor:
    """Keep diagnostic wall-clock values off accelerators without float64."""
    # MPS does not implement float64 tensors.  These host-side timestamps are
    # consumed only by evidence serialization, so CPU float64 is both portable
    # and avoids adding a device transfer to the timed query path.
    return torch.tensor(elapsed_ms, dtype=torch.float64)


def aggregate_class_balanced_gradients(retrieval: RetrievalBatch, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Ramen's entropy/distance weights and average active classes.

    The return value includes per-query active-class counts so empty support
    sets (the ``include_current=False`` first-sample ablation) are explicit.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative")
    valid = retrieval.valid_mask
    # Invalid padded entries use ``inf`` distance.  Mask before aggregation so
    # beta=0 does not create the otherwise undefined 0 * inf expression.
    distance_weights = torch.where(
        valid, torch.exp(-float(beta) * retrieval.distances), torch.zeros_like(retrieval.distances)
    )
    weights = torch.exp(-retrieval.entropies) * distance_weights
    per_class = (retrieval.gradients * weights.to(retrieval.gradients.dtype).unsqueeze(-1)).sum(dim=2)
    active = valid.any(dim=2)
    counts = active.sum(dim=1)
    summed = (per_class * active.to(per_class.dtype).unsqueeze(-1)).sum(dim=1)
    # Do not manufacture an update when no historical/current support exists.
    gradients = summed / counts.clamp_min(1).to(summed.dtype).unsqueeze(-1)
    return gradients, counts


def _failure_analysis_profile(args: Any, cfg: Mapping[str, Any]) -> str:
    """Read the public evaluator argument while retaining config-only callers."""
    value = getattr(args, "failure_analysis_profile", cfg["failure_analysis_profile"])
    if value not in {"off", "trace_v1", "replay_v1"}:
        raise ValueError("failure_analysis_profile must be 'off', 'trace_v1', or 'replay_v1'")
    return value


def _counterfactual_thresholds(args: Any) -> tuple[float, ...]:
    """Validate immutable replay thresholds without depending on evaluator code."""
    raw = getattr(args, "failure_counterfactual_thresholds", (0.50, 0.75, 1.00))
    if isinstance(raw, str):
        raw = raw.split(",")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("failure_counterfactual_thresholds must be finite values in [0, 1]") from exc
    if not values or len(set(values)) != len(values) or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("failure_counterfactual_thresholds must be unique finite values in [0, 1]")
    if values != (0.50, 0.75, 1.00):
        raise ValueError(
            "failure_counterfactual_thresholds must equal the preregistered tuple "
            "(0.50, 0.75, 1.00)"
        )
    return values


def _single_failure_payload(support: Any, beta: float, *, item_id: torch.Tensor,
                            batch_position: int, future_support_count: int = 0,
                            future_support_weight_fraction: float = 0.0,
                            schedule: str = "causal",
                            legal_candidates: list[dict[str, int | float]] | None = None,
                            profile: str = "trace_v1", replay_item: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    """Convert one exact retrieval into a trace-safe, model-visible payload."""
    trace = trace_payload_from_retrieval(support, beta)
    payload = {key: value[0].detach().clone() for key, value in trace.items()}
    if schedule not in {"causal", "atomic"}:
        raise ValueError("failure-analysis schedule must be causal or atomic")
    payload.update({
        "query_item_id": item_id.detach().clone(),
        "batch_position": batch_position,
        "future_support_count": future_support_count,
        "future_support_weight_fraction": future_support_weight_fraction,
        # This is production provenance, not an evaluator label.  It makes
        # replay rows self-describing for F5 schedule diagnostics.
        "schedule": schedule,
        # F3's canonical scalar is the fraction of aggregate coordinates
        # without strong class-local SignSGD consensus.  Keep its definition
        # alongside every exact query retrieval so offline analysis never
        # guesses between the several related diagnostic summaries.
        "conflict_metric": "fraction_low_consensus_coordinates_v1",
        "conflict": trace["fraction_low_consensus_coordinates"][0].detach().clone(),
    })
    if legal_candidates is not None:
        payload["legal_candidates"] = legal_candidates
    if profile == "replay_v1" and replay_item is not None:
        # Opaque evaluator-only material: it is deliberately separate from
        # the scalar trace payload, so JSON trace writers never see gradients.
        payload["replay"] = replay_item
    # Internal-only tensors are consumed for evaluator counterfactuals and
    # deliberately stripped by the public trace formatter below.
    payload["_consensus_strength"] = trace["consensus_strength"][0].detach().clone()
    return payload


def _failure_analysis_method_payloads(entries: Any) -> list[dict[str, Any]]:
    """Split trace-safe fields from opaque replay vectors for the evaluator."""
    def serialisable(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {key: serialisable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serialisable(item) for item in value]
        return value

    if not isinstance(entries, list):
        return []
    result: list[dict[str, Any]] = []
    for entry in entries:
        trace = serialisable({
            key: value for key, value in entry.items()
            if key not in {"replay", "counterfactuals"} and not key.startswith("_")
        })
        replay = entry.get("replay")
        item: list[dict[str, Any]] = []
        if isinstance(replay, dict):
            item_id = entry.get("query_item_id")
            item.append({
                "item_id": serialisable(item_id),
                "feature": replay.get("query_feature"),
                "gradient": replay.get("query_gradient"),
                "predicted_class": serialisable(replay.get("predicted_class")),
                "context": serialisable(replay.get("context")),
                "entropy": serialisable(replay.get("entropy")),
                "admitted": serialisable(replay.get("admitted_to_memory")),
            })
        query = {
            "item_id": serialisable(entry.get("query_item_id")), "legal_candidates": serialisable(entry.get("legal_candidates", [])),
            "retrieved_support_ids": serialisable(entry.get("support_item_ids")), "retrieved_weights": serialisable(entry.get("support_weights")),
            "reset_state_verified": bool(entry.get("reset_state_verified", False)),
        }
        result.append({"trace": trace, "query": query, "item": item,
                       "counterfactuals": serialisable(entry.get("counterfactuals", [])),
                       "reset_state_verified": bool(entry.get("reset_state_verified", False))})
    return result


def evaluate_replay_counterfactuals(model: Any, x: torch.Tensor, actual_logits: torch.Tensor,
                                    actual_aggregate: torch.Tensor,
                                    entries: list[dict[str, Any]], thresholds: tuple[float, ...]) -> None:
    """Evaluate fixed consensus masks from the same reset per-sample state.

    The actual prediction has already been captured by the caller.  This
    helper only mutates evaluator-only payloads and always leaves the model at
    its pretrained state.  A warm feature pass before setting gradients is
    required because the by-sample norm modules initialise ``recent_B`` there.
    """
    variants: list[list[dict[str, Any]]] = [[] for _ in entries]
    parity_error = torch.full((len(entries),), float("inf"), device=actual_logits.device)
    verified = False
    try:
        # Rebuild the exact unmasked production update from base state before
        # trusting any masked replay.  This is the counterfactual's causal
        # anchor, not merely a reset call that may be incomplete in a model.
        model.reset_parameters()
        with torch.no_grad():
            model.featurize(x)
        model.set_by_sample_grad(actual_aggregate)
        model.step_and_zero_grad()
        with torch.no_grad():
            replay_logits = model(x)
        parity_error = (replay_logits - actual_logits).abs().amax(dim=-1)
        verified = bool(torch.allclose(replay_logits, actual_logits, rtol=0., atol=0.))
        model.reset_parameters()
        if not verified:
            # Fail closed: the model cannot substantiate its baseline reset,
            # so exposing masked alternatives would be misleading.
            for index, entry in enumerate(entries):
                entry["counterfactuals"] = []
                entry["reset_state_verified"] = False
                entry["actual_prediction"] = actual_logits[index].argmax(dim=-1).detach().clone()
                entry["actual_entropy"] = softmax_entropy(actual_logits, reduction="none")[index].detach().clone()
                entry["reset_parity_error"] = parity_error[index].detach().clone()
            return
        for threshold in thresholds:
            strengths = torch.stack([
                entry.get("_consensus_strength", torch.ones_like(actual_aggregate[index]))
                for index, entry in enumerate(entries)
            ]).to(device=actual_aggregate.device, dtype=actual_aggregate.dtype)
            masked = actual_aggregate * (strengths >= threshold).to(actual_aggregate.dtype)
            # This initializes BySampleLayerNorm/BatchNorm.recent_B after the
            # reset, before set_by_sample_grad slices their gradients.
            with torch.no_grad():
                model.featurize(x)
            model.set_by_sample_grad(masked)
            model.step_and_zero_grad()
            with torch.no_grad():
                logits = model(x)
                predictions = logits.argmax(dim=-1)
                entropy = softmax_entropy(logits, reduction="none")
            for index in range(len(entries)):
                variants[index].append({"threshold": threshold, "prediction": predictions[index], "entropy": entropy[index]})
            model.reset_parameters()
    except Exception:
        model.reset_parameters()
        raise
    for index, (entry, values) in enumerate(zip(entries, variants)):
        entry["counterfactuals"] = values
        entry["reset_state_verified"] = verified
        entry["actual_prediction"] = actual_logits[index].argmax(dim=-1).detach().clone()
        entry["actual_entropy"] = softmax_entropy(actual_logits, reduction="none")[index].detach().clone()
        entry["reset_parity_error"] = parity_error[index].detach().clone()


def update_and_retrieve_causal_batch(
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
    failure_analysis_profile: str = "off",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Insert and retrieve one stream item at a time.

    Per-sample gradients may be computed in a batch, but evidence from item
    ``i + 1`` must never be visible while adapting item ``i``.  Returning the
    memory size after each insertion also lets the trace preserve that causal
    timeline instead of repeating a post-batch scalar.  Retained-byte values
    are read from the memory's O(1) counter immediately after each causal
    insertion; they are not reconstructed by scanning live tensors.
    """
    if failure_analysis_profile not in {"off", "trace_v1", "replay_v1"}:
        raise ValueError("unknown failure_analysis_profile")
    retrieved = []
    active_class_counts = []
    memory_sizes = []
    memory_bytes = []
    failure_payloads: list[dict[str, Any]] = []
    for index in range(features.shape[0]):
        item_slice = slice(index, index + 1)
        memory.add(
            features[item_slice],
            gradients[item_slice],
            predicted_classes[item_slice],
            contexts[item_slice],
            entropies[item_slice],
            item_ids=item_ids[item_slice],
        )
        candidates = None
        if failure_analysis_profile != "off":
            candidates = memory.legal_candidate_snapshot(
                contexts[item_slice], schedule="causal", selection="class_balanced",
                include_current=include_current, current_item_ids=item_ids[item_slice],
            )[0]
        support = memory.query(
            features[item_slice],
            contexts[item_slice],
            topk,
            include_current=include_current,
            current_item_ids=item_ids[item_slice],
        )
        item_gradient, item_active_classes = aggregate_class_balanced_gradients(support, beta)
        retrieved.append(item_gradient)
        active_class_counts.append(item_active_classes)
        memory_sizes.append(memory.size)
        memory_bytes.append(memory.retained_bytes)
        if failure_analysis_profile != "off":
            replay_item = None if failure_analysis_profile != "replay_v1" else {
                "query_feature": features[index].detach().clone(),
                "query_gradient": gradients[index].detach().clone(),
                "predicted_class": predicted_classes[index].detach().clone(),
                "context": contexts[index].detach().clone(),
                "entropy": entropies[index].detach().clone(),
                "admitted_to_memory": torch.tensor(True, device=features.device),
            }
            failure_payloads.append(_single_failure_payload(
                support, beta, item_id=item_ids[index], batch_position=index,
                legal_candidates=candidates, schedule="causal", profile=failure_analysis_profile, replay_item=replay_item,
            ))
    result = (
        torch.cat(retrieved, dim=0),
        torch.cat(active_class_counts, dim=0),
        torch.tensor(memory_sizes, device=features.device, dtype=torch.long),
        torch.tensor(memory_bytes, device=features.device, dtype=torch.long),
    )
    return result if failure_analysis_profile == "off" else (*result, failure_payloads)


def update_and_retrieve_profiled_causal_batch(
    memory: StructuredGradientMemory, features: torch.Tensor, gradients: torch.Tensor,
    predicted_classes: torch.Tensor, contexts: torch.Tensor, entropies: torch.Tensor,
    item_ids: torch.Tensor, *, topk: int, include_current: bool, beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal retrieval with deliberately intrusive, query-only timing."""
    retrieved, active_class_counts, memory_sizes, memory_bytes = [], [], [], []
    elapsed, live_counts, eligible_counts, returned_counts = [], [], [], []
    for index in range(features.shape[0]):
        item_slice = slice(index, index + 1)
        memory.add(features[item_slice], gradients[item_slice], predicted_classes[item_slice],
                   contexts[item_slice], entropies[item_slice], item_ids=item_ids[item_slice])
        live, eligible = memory.query_candidate_counts(
            contexts[item_slice], include_current=include_current, current_item_ids=item_ids[item_slice]
        )
        _synchronize_for_retrieval(features.device)
        started = time.perf_counter()
        support = memory.query(features[item_slice], contexts[item_slice], topk,
                               include_current=include_current, current_item_ids=item_ids[item_slice])
        _synchronize_for_retrieval(features.device)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        item_gradient, item_active_classes = aggregate_class_balanced_gradients(support, beta)
        retrieved.append(item_gradient)
        active_class_counts.append(item_active_classes)
        memory_sizes.append(memory.size)
        memory_bytes.append(memory.retained_bytes)
        live_counts.append(live)
        eligible_counts.append(eligible)
        returned_counts.append(support.valid_mask.sum(dim=(1, 2)).to(torch.long))
    return (
        torch.cat(retrieved), torch.cat(active_class_counts),
        torch.tensor(memory_sizes, device=features.device, dtype=torch.long),
        torch.tensor(memory_bytes, device=features.device, dtype=torch.long),
        _profile_elapsed_tensor(elapsed),
        torch.cat(live_counts), torch.cat(eligible_counts), torch.cat(returned_counts),
    )


class LatentRamen(TTABase):
    """Ramen with hard sequential context routing and structured memory."""

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = validate_latent_ramen_config(args.config)
        self.failure_analysis_profile = _failure_analysis_profile(args, self.cfg)
        self.failure_counterfactual_thresholds = _counterfactual_thresholds(args)
        self.num_classes = datasets.num_classes
        self.device = next(model.parameters()).device
        self.model = CLIPModelForBySampleTTA(model, datasets.classes, self.cfg, args)

        self.router = OnlinePrototypeRouter(
            spawn_threshold=self.cfg["spawn_threshold"],
            max_contexts=self.cfg["max_contexts"],
            temperature=self.cfg["router_temperature"],
            momentum=self.cfg["router_momentum"],
        )
        # Metadata in StructuredGradientMemory is always float32/int64, even
        # when CLIP features and gradients are retained in half precision.
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
            # Routing is necessarily based only on CLIP visual features.  The
            # router processes the batch sequentially, so transitions inside a
            # batch are observable rather than hidden by batch averaging.
            contexts_before_batch = self.router.num_contexts
            routing = self.router.route(features, update=True)
            entropies = softmax_entropy(logits, reduction="none")
            admission_normalized_entropy = (entropies / math.log(self.num_classes)).clamp(0.0, 1.0)
            item_ids = torch.arange(self.counter, self.counter + batch_size, device=self.device, dtype=torch.long)
            self.counter += batch_size
            # Timing and replay instrumentation are intentionally separate;
            # replay keeps the ordinary production query path observable.
            profile = self.cfg["retrieval_profile"] == "causal_sync_v1" and self.failure_analysis_profile == "off"
            failure_analysis = None
            if profile and self.failure_analysis_profile == "off":
                (retrieved_gradients, active_classes, memory_sizes, memory_bytes, retrieval_elapsed_ms,
                 retrieval_candidate_count, retrieval_eligible_candidate_count,
                 retrieval_returned_support_count) = update_and_retrieve_profiled_causal_batch(
                    self.memory, features, gradients, predicted_classes, routing.context_ids, entropies, item_ids,
                    topk=self.cfg["topk"], include_current=self.cfg["include_current"], beta=self.cfg["beta"],
                )
            else:
                retrieval = update_and_retrieve_causal_batch(
                    self.memory, features, gradients, predicted_classes, routing.context_ids, entropies, item_ids,
                    topk=self.cfg["topk"], include_current=self.cfg["include_current"], beta=self.cfg["beta"],
                    failure_analysis_profile=self.failure_analysis_profile,
                )
                if self.failure_analysis_profile != "off":
                    retrieved_gradients, active_classes, memory_sizes, memory_bytes, failure_analysis = retrieval
                else:
                    retrieved_gradients, active_classes, memory_sizes, memory_bytes = retrieval
            self.model.set_by_sample_grad(retrieved_gradients)
            active_contexts = contexts_before_batch + routing.spawned.long().cumsum(dim=0)
            self.last_diagnostics = self._diagnostics(
                routing, active_classes, memory_sizes, memory_bytes, active_contexts,
                admission_prediction=predicted_classes,
                admission_normalized_entropy=admission_normalized_entropy,
                admitted_to_memory=torch.ones(batch_size, device=self.device, dtype=torch.bool),
                retrieval_profile=self.cfg["retrieval_profile"] if profile else None,
                retrieval_elapsed_ms=retrieval_elapsed_ms if profile else None,
                retrieval_candidate_count=retrieval_candidate_count if profile else None,
                retrieval_eligible_candidate_count=retrieval_eligible_candidate_count if profile else None,
                retrieval_returned_support_count=retrieval_returned_support_count if profile else None,
                failure_analysis=failure_analysis,
                failure_analysis_profile=self.failure_analysis_profile,
            )

        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        if self.failure_analysis_profile == "replay_v1":
            evaluate_replay_counterfactuals(
                self.model, x, output, retrieved_gradients, failure_analysis, self.failure_counterfactual_thresholds
            )
        # Adaptation is per sample/batch.  Router and memory deliberately
        # persist; their reset is owned by ``reset`` below.
        self.model.reset_parameters()
        return output

    def get_diagnostics(self) -> dict[str, Any]:
        """Return evidence fields without exposing mutable router state."""
        return dict(self.last_diagnostics)

    def get_failure_analysis_payload(self) -> list[dict[str, Any]]:
        """Return evaluator-facing trace/replay payloads after one forward."""
        return _failure_analysis_method_payloads(self.last_diagnostics.get("failure_analysis"))

    def diagnostics(self) -> dict[str, Any]:
        """Backward-compatible alias for interactive inspection."""
        return self.get_diagnostics()

    def reset(self):
        self.model.reset_parameters()
        self.router.reset()
        self.memory.reset()
        self.counter = 0
        self.last_diagnostics = self._diagnostics()

    def _diagnostics(
        self,
        routing=None,
        active_classes=None,
        memory_sizes=None,
        memory_bytes=None,
        active_contexts=None,
        admission_prediction=None,
        admission_normalized_entropy=None,
        admitted_to_memory=None,
        retrieval_profile=None,
        retrieval_elapsed_ms=None,
        retrieval_candidate_count=None,
        retrieval_eligible_candidate_count=None,
        retrieval_returned_support_count=None,
        failure_analysis=None,
        failure_analysis_profile="off",
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "memory_size": self.memory.size if memory_sizes is None else memory_sizes,
            "num_active_contexts": self.router.num_contexts if active_contexts is None else active_contexts,
            "router_active_contexts": self.router.num_contexts,
            "inferred_context": None,
            "spawned": None,
            "router_distances": None,
            "nearest_router_distances": None,
            "active_classes": None,
            "memory_active_contexts": self.memory.active_contexts,
            "memory_bytes": self.memory.retained_bytes if memory_bytes is None else memory_bytes,
            "capacity_scope": self.memory.capacity_scope,
            "memory_capacity_scope": self.memory.capacity_scope,
            "memory_max_capacity": self.memory.max_capacity,
            # Kept for ungated runs too, so admission contamination can be
            # compared without changing the underlying adaptation behavior.
            "admission_prediction": admission_prediction,
            "admission_normalized_entropy": admission_normalized_entropy,
            "admitted_to_memory": admitted_to_memory,
        }
        if failure_analysis_profile != "off":
            result["failure_analysis_profile"] = failure_analysis_profile
            result["failure_analysis"] = failure_analysis
        if routing is not None:
            result.update({
                "inferred_context": routing.context_ids.detach().clone(),
                "spawned": routing.spawned.detach().clone(),
                "router_distances": routing.distances.detach().clone(),
                "nearest_router_distances": routing.nearest_distances.detach().clone(),
                "active_classes": active_classes.detach().clone() if active_classes is not None else None,
            })
        if retrieval_profile is not None:
            result.update({
                "retrieval_profile": retrieval_profile,
                "retrieval_elapsed_ms": retrieval_elapsed_ms.detach().clone(),
                "retrieval_candidate_count": retrieval_candidate_count.detach().clone(),
                "retrieval_eligible_candidate_count": retrieval_eligible_candidate_count.detach().clone(),
                "retrieval_returned_support_count": retrieval_returned_support_count.detach().clone(),
                "retrieval_active_class_count": active_classes.detach().clone(),
            })
        return result
