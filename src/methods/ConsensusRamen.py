"""Class-consensus gradient aggregation for Ramen.

ConsensusRamen deliberately keeps Ramen's feature retrieval, cache admission,
entropy/distance weighting, temporary SignSGD update, and parameter reset.  It
only suppresses aggregate-gradient coordinates whose directions are not
corroborated across predicted-class support caches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from models.ModelForBySampleTTA import CLIPModelForBySampleTTA

from .Ramen import PriorityCache
from .TTABase import TTABase
from .losses import softmax_entropy


_REQUIRED_CONFIG = ("max_capacity", "topk", "optimizer", "lr")
_HARD_MASK_MODE = "hard_mask"
_SOFT_WEIGHT_MODE = "soft_weight"
_CONSENSUS_MODES = frozenset({_HARD_MASK_MODE, _SOFT_WEIGHT_MODE})


def validate_consensus_ramen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Ramen configuration and its consensus aggregation mode.

    ``hard_mask`` is the preregistered v0 default.  ``soft_weight`` is the
    deferred v1 ablation: it admits each coordinate to SignSGD with probability
    agreement raised to ``consensus_gamma``.  This changes the actual update
    support while retaining Ramen's sign on an admitted coordinate.
    """
    if not isinstance(config, Mapping):
        raise TypeError("ConsensusRamen config must be a mapping")
    missing = [key for key in _REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError("ConsensusRamen config is missing: " + ", ".join(missing))
    cfg = dict(config)
    for key in ("max_capacity", "topk", "min_consensus_classes"):
        value = cfg.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("beta", "lr", "consensus_threshold"):
        value = cfg.get(key, 0.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
        cfg[key] = float(value)
    if cfg["beta"] < 0 or cfg["lr"] <= 0:
        raise ValueError("beta must be non-negative and lr must be positive")
    if not 0.0 <= cfg["consensus_threshold"] <= 1.0:
        raise ValueError("consensus_threshold must be in [0, 1]")
    consensus_mode = cfg.get("consensus_mode")
    if consensus_mode not in _CONSENSUS_MODES:
        raise ValueError(
            "consensus_mode must be one of: " + ", ".join(sorted(_CONSENSUS_MODES))
        )
    if consensus_mode == _SOFT_WEIGHT_MODE:
        gamma = cfg.get("consensus_gamma")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or not math.isfinite(float(gamma)):
            raise ValueError("consensus_gamma must be finite for soft_weight")
        # gamma=0 reduces every nonzero q to one and silently becomes ordinary
        # Ramen; require a real attenuation exponent for this v1 ablation.
        if float(gamma) <= 0:
            raise ValueError("consensus_gamma must be positive for soft_weight")
        cfg["consensus_gamma"] = float(gamma)
        seed = cfg.get("consensus_seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("consensus_seed must be a non-negative integer for soft_weight")
        cfg["consensus_seed"] = seed
    if not isinstance(cfg["optimizer"], str) or not cfg["optimizer"]:
        raise ValueError("optimizer must be a non-empty string")
    # Batch-atomic current-support visibility is the original Ramen behavior
    # and is therefore the v0 default.  The false setting is an explicitly
    # named causal-history ablation, not a silent change to the baseline.
    include_current = cfg.get("include_current", True)
    if not isinstance(include_current, bool):
        raise ValueError("include_current must be a boolean")
    cfg["include_current"] = include_current
    return cfg


def aggregate_consensus_supports(
    queries: torch.Tensor,
    caches: list[PriorityCache],
    *,
    topk: int,
    beta: float,
    consensus_threshold: float,
    min_consensus_classes: int,
    consensus_mode: str = _HARD_MASK_MODE,
    consensus_gamma: float | None = None,
    consensus_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return consensus-weighted and ordinary Ramen aggregates for each query.

    Each non-empty cache remains one class-balanced contributor, exactly as in
    Ramen.  The returned ordinary aggregate is useful both for the mandated
    below-minimum fallback and for testing that retrieval itself is unchanged.
    """
    if consensus_mode not in _CONSENSUS_MODES:
        raise ValueError("unsupported consensus_mode: " + repr(consensus_mode))
    if consensus_mode == _SOFT_WEIGHT_MODE:
        if consensus_gamma is None or not math.isfinite(float(consensus_gamma)) or float(consensus_gamma) <= 0:
            raise ValueError("soft_weight requires finite positive consensus_gamma")
        if not isinstance(consensus_seed, int) or isinstance(consensus_seed, bool) or consensus_seed < 0:
            raise ValueError("soft_weight requires non-negative integer consensus_seed")
    if not caches:
        raise ValueError("caches must contain one cache per predicted class")
    batch_size = queries.shape[0]
    gradient_dim = caches[0].value_dim
    class_gradients = []
    for cache in caches:
        if cache.size == 0:
            continue
        values, _, entropies, distances = cache.query(queries, topk=topk)
        weights = torch.exp(-entropies) * torch.exp(-float(beta) * distances)
        # This is Ramen's weighted per-class contribution.  Do not normalize
        # within a class: doing so would change its existing update magnitude.
        class_gradients.append((values * weights.unsqueeze(-1)).sum(dim=1))

    if not class_gradients:
        empty = torch.zeros((batch_size, gradient_dim), device=queries.device, dtype=queries.dtype)
        agreement = torch.zeros_like(empty)
        active_classes = torch.zeros(batch_size, device=queries.device, dtype=torch.long)
        mask = torch.ones_like(agreement, dtype=torch.bool)
        return empty, empty.clone(), _consensus_diagnostics(
            agreement, mask, active_classes, applied=False
        )

    per_class = torch.stack(class_gradients, dim=1)
    ordinary = per_class.mean(dim=1)
    active_class_count = torch.full(
        (batch_size,), per_class.shape[1], device=queries.device, dtype=torch.long
    )

    # torch.sign(0) is zero, so zero coordinates are neutral votes rather
    # than agreement with either direction.
    agreement = torch.sign(per_class).mean(dim=1).abs()
    if per_class.shape[1] < min_consensus_classes:
        # This is an ordinary Ramen update, so report a fully open mask rather
        # than making a fallback look like coordinates were suppressed.
        mask = torch.ones_like(agreement, dtype=torch.bool)
        return ordinary, ordinary.clone(), _consensus_diagnostics(
            agreement, mask, active_class_count, applied=False
        )
    if consensus_mode == _HARD_MASK_MODE:
        mask = agreement >= float(consensus_threshold)
        safe = ordinary * mask.to(dtype=ordinary.dtype)
    else:
        # SignSGD discards a nonzero gradient's magnitude.  Therefore a
        # multiplication by q**gamma would be observationally identical to
        # Ramen.  Draw coordinate admissions instead: each coordinate is
        # retained with probability q**gamma, preserving its Ramen sign if
        # retained and giving the desired agreement-weighted update in
        # expectation.  Draw on CPU so a configured seed has identical
        # semantics on CPU, CUDA, and MPS, then move the Boolean support back
        # to the gradient device.
        admission_probability = agreement.pow(float(consensus_gamma))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(consensus_seed)
        random_uniform = torch.rand(
            admission_probability.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=agreement.device)
        mask = random_uniform < admission_probability
        safe = ordinary * mask.to(dtype=ordinary.dtype)
    return safe, ordinary, _consensus_diagnostics(
        agreement, mask, active_class_count, applied=True
    )


def _consensus_diagnostics(
    agreement: torch.Tensor,
    mask: torch.Tensor,
    active_classes: torch.Tensor,
    *,
    applied: bool,
) -> dict[str, torch.Tensor]:
    """Summarize coordinate agreement without introducing evaluator inputs."""
    sorted_agreement = agreement.sort(dim=1).values
    count = sorted_agreement.shape[1]
    p10_index = int((count - 1) * .1)
    p50_index = int((count - 1) * .5)
    return {
        "consensus_mean_agreement": agreement.mean(dim=1),
        "consensus_p10_agreement": sorted_agreement[:, p10_index],
        "consensus_p50_agreement": sorted_agreement[:, p50_index],
        "consensus_mask_rate": mask.to(dtype=agreement.dtype).mean(dim=1),
        "consensus_active_class_count": active_classes,
        # False distinguishes empty/below-minimum ordinary-Ramen fallback from
        # a real hard-mask decision in downstream evidence summaries.
        "consensus_applied": torch.full(
            (agreement.shape[0],), applied, device=agreement.device, dtype=torch.bool
        ),
    }


class ConsensusRamen(TTABase):
    """Ramen with a hard-mask v0 default and an opt-in soft-weight v1 ablation."""

    def __init__(self, model, datasets, args):
        super().__init__()
        self.cfg = validate_consensus_ramen_config(args.config)
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
        self.soft_admission_step = 0
        self.last_diagnostics = {}

    @property
    def memory_bytes(self) -> int:
        """Bytes occupied by the currently retained Ramen-compatible supports."""
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
        features = self.model.featurize(x)
        logits = self.model.classify(features)
        self.last_diagnostics = {"pre_adaptation_ood_score": -torch.logsumexp(logits.detach(), dim=1)}
        predicted_classes = logits.argmax(-1)
        self.loss_fn(logits).backward()
        gradients = self.model.get_by_sample_grad()

        with torch.no_grad():
            priorities = torch.arange(self.counter, self.counter + batch_size, device=self.device, dtype=self.dtype)
            entropies = softmax_entropy(logits, reduction="none")
            self.counter += batch_size

            # v0 deliberately retains Ramen's batch-atomic visibility,
            # including every current input.  The no-self ablation reverses
            # this ordering: retrieval is restricted to cache state that
            # predates this forward, then the batch is admitted for future
            # inputs after its safe gradients have been fixed.
            if self.cfg["include_current"]:
                self._admit_batch(
                    features, gradients, entropies, priorities, predicted_classes
                )
            soft_seed = None
            if self.cfg["consensus_mode"] == _SOFT_WEIGHT_MODE:
                # A distinct deterministic seed per forward avoids reusing one
                # mask forever while preserving exact replay under the same
                # stream order and configuration.
                soft_seed = self.cfg["consensus_seed"] + self.soft_admission_step
                self.soft_admission_step += 1
            safe_gradients, _, consensus_diagnostics = aggregate_consensus_supports(
                features, self.cache, topk=self.cfg["topk"], beta=self.beta,
                consensus_threshold=self.cfg["consensus_threshold"],
                min_consensus_classes=self.cfg["min_consensus_classes"],
                consensus_mode=self.cfg["consensus_mode"],
                consensus_gamma=self.cfg.get("consensus_gamma"),
                consensus_seed=soft_seed,
            )
            self.model.set_by_sample_grad(safe_gradients)
            if not self.cfg["include_current"]:
                self._admit_batch(
                    features, gradients, entropies, priorities, predicted_classes
                )
            self.last_diagnostics.update(consensus_diagnostics)
            self.last_diagnostics["memory_bytes"] = self.memory_bytes

        self.model.step_and_zero_grad()
        with torch.no_grad():
            output = self.model(x)
        self.model.reset_parameters()
        return output

    def _admit_batch(
        self,
        features: torch.Tensor,
        gradients: torch.Tensor,
        entropies: torch.Tensor,
        priorities: torch.Tensor,
        predicted_classes: torch.Tensor,
    ) -> None:
        """Admit a fully observed batch into predicted-class support caches."""
        for index in range(features.shape[0]):
            predicted_class = predicted_classes[index]
            self.cache[predicted_class].add(
                features[index].unsqueeze(0), gradients[index].unsqueeze(0),
                entropies[index].unsqueeze(0), priorities[index].unsqueeze(0),
            )

    def reset(self):
        self.model.reset_parameters()
        self.counter = 0
        self.soft_admission_step = 0
        for cache in self.cache:
            cache.reset()
        self.last_diagnostics = {}

    def get_diagnostics(self):
        return dict(self.last_diagnostics)
