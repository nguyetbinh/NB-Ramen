"""Label-free scorer for the frozen exact QCGS objective audit."""
from __future__ import annotations

import base64
import hashlib
import time

import numpy as np
from PIL import Image
import torch

from .OracleSupportUtilityProbe import aggregate_pools, aggregate_swaps, legal_swaps
from .Ramen import Ramen, PriorityCache, cache_distances
from .losses import softmax_entropy
from .query_gradient_multiview import raw_views
from .qcgs_exact_objectives import OBJECTIVE_NAMES, objective_deltas, objective_values, select_objective_action
from .OracleSupportUtilityProbe import BySampleBatchNorm, BySampleLayerNorm
from runtime.query_gradient_registry import digest


EXACT_VIEW_SPEC = {"name": "four-corners-30-flip-v1", "input_shape": [32, 32, 3],
                   "corners": [[0, 0], [2, 0], [0, 2], [2, 2]], "crop_size": 30,
                   "flip_order": [False, True], "views": 8, "crop_before_preprocess": True}


def tensor_sha(tensor):
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def preprocess_spec(transform):
    if hasattr(transform, "transforms"):
        return [preprocess_spec(child) for child in transform.transforms]
    result = {"callable": getattr(transform, "__module__", type(transform).__module__) + "." +
              getattr(transform, "__qualname__", type(transform).__qualname__)}
    for field in ("size", "interpolation", "max_size", "antialias", "mean", "std", "inplace"):
        if hasattr(transform, field):
            value = getattr(transform, field)
            result[field] = str(value) if field == "interpolation" else value
    return result


class IdentityCache(PriorityCache):
    """Ramen cache with opaque stream identities, never evaluator metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identities = [None] * self.max_capacity

    def add_identity(self, key, value, entropy, priority, identity):
        target = self.size if self.size < self.max_capacity else int(self.priorities.argmin())
        admitted = self.size < self.max_capacity or bool(priority > self.priorities[target])
        super().add(key, value, entropy.reshape(1), priority.reshape(1))
        if admitted:
            self.identities[target] = identity

    def reset(self):
        super().reset()
        self.identities = [None] * self.max_capacity


def _pack_signs(values):
    """Losslessly pack a {-1, 0, +1} SignSGD direction into two bit planes."""
    signs = values.detach().sign().to(torch.int8).cpu().numpy()
    positive = np.packbits((signs > 0).astype(np.uint8)).tobytes()
    negative = np.packbits((signs < 0).astype(np.uint8)).tobytes()
    return {"size": int(signs.size), "positive": base64.b64encode(positive).decode("ascii"),
            "negative": base64.b64encode(negative).decode("ascii")}


class QCGSExactObjectiveAudit(Ramen):
    """Exact action scorer. This class has no label/OOD/domain input hook."""

    diagnostic_kind = "qcgs-exact-objective"
    schema_version = 1

    def __init__(self, model, datasets, args):
        cfg = args.config
        if cfg.get("optimizer") != "signsgd" or cfg.get("lr") != 0.01:
            raise ValueError("exact-objective audit requires frozen state-free SignSGD at lr=.01")
        if (cfg.get("topk"), cfg.get("candidate_m"), cfg.get("max_capacity"), cfg.get("beta")) != (5, 10, 750, 5.0):
            raise ValueError("exact-objective audit cache/retrieval parameters differ from protocol")
        if type(cfg.get("action_chunk_size")) is not int or cfg["action_chunk_size"] <= 0:
            raise ValueError("exact-objective audit requires a positive locked action chunk size")
        scale = getattr(model, "logit_scale", None)
        configured_scale = cfg.get("clip_logit_scale")
        if scale is None and configured_scale is None:
            raise ValueError("exact-objective audit requires the pretrained CLIP logit scale")
        self.clip_logit_scale = float(scale.detach().exp().cpu()) if scale is not None else float(configured_scale)
        super().__init__(model, datasets, args)
        if self.num_classes != 80:
            raise ValueError("exact-objective audit requires the locked 80-known-class label space")
        self.cache = [IdentityCache(cfg["max_capacity"], self.feat_dim, self.grad_dim,
                                    self.device, self.dtype) for _ in self.cache]
        self._initialize_audit(args)

    def _initialize_audit(self, args):
        self.mode = self.cfg.get("qcgs_mode", "scan")
        if self.mode not in ("scan", "smoke", "stage-a", "stage-b"):
            raise ValueError("invalid exact-objective audit mode")
        self.objective_stage = self.cfg.get("objective_stage", "all")
        if self.mode == "stage-b":
            self.frozen_objective = self.cfg.get("frozen_objective")
            if self.frozen_objective not in OBJECTIVE_NAMES:
                raise ValueError("Stage B requires one frozen Stage-A objective")
            self.score_names = (self.frozen_objective,)
        else:
            self.frozen_objective = None
            self.score_names = OBJECTIVE_NAMES
        self.fingerprint = self.cfg.get("dataset_fingerprint")
        if self.mode != "scan" and (not isinstance(self.fingerprint, str) or len(self.fingerprint) != 64):
            raise ValueError("scored exact-objective runs require a dataset fingerprint")
        self.source_revision = self.cfg.get("source_revision")
        self.registry_source_revision = self.cfg.get("registry_source_revision", self.source_revision)
        self.protocol_sha256 = self.cfg.get("protocol_sha256")
        run_token = self.cfg.get("run_token")
        if self.mode != "scan" and (not isinstance(run_token, str) or len(run_token) != 32):
            raise ValueError("scored exact-objective run requires an opaque unique run token")
        self.run_identity_sha256 = (hashlib.sha256(run_token.encode("ascii")).hexdigest()
                                    if isinstance(run_token, str) else None)
        self.registry_sha256 = self.cfg.get("registry_sha256")
        self.config_sha256 = self.cfg.get("config_sha256") or digest(self.cfg)
        self.query_schedule = self.cfg.get("query_schedule", [])
        self.registered = {}
        scheduled_ids = set()
        for row in self.query_schedule:
            if set(row) != {"timestep", "sample_idx"}:
                raise ValueError("scorer schedule may contain only opaque timestep/sample identity keys")
            timestep, sample_idx = row["timestep"], row["sample_idx"]
            if type(timestep) is not int or timestep < 0 or type(sample_idx) is not int or not 0 <= sample_idx < 10000:
                raise ValueError("invalid opaque query identity")
            if timestep in self.registered:
                raise ValueError("duplicate registered query timestep")
            if sample_idx in scheduled_ids:
                raise ValueError("duplicate registered base-image identity")
            self.registered[timestep] = sample_idx
            scheduled_ids.add(sample_idx)
        if self.mode in ("stage-a", "stage-b") and not self.query_schedule:
            raise ValueError("scored run requires a frozen registry schedule")
        if self.mode == "stage-a" and len(self.query_schedule) != 16:
            raise ValueError("Stage A must score exactly 16 registered queries")
        if self.mode == "stage-b" and len(self.query_schedule) != 128:
            raise ValueError("Stage B must score exactly 128 registered queries")
        if self.mode == "smoke" and len(self.query_schedule) != 2:
            raise ValueError("smoke mode requires two separately registered queries per cell")
        self.identity = None
        self.raw_context = None
        self.view_state = {"view_spec": EXACT_VIEW_SPEC, "preprocess": None}
        self.scan = []
        self.action_rows = []
        self._pending_evaluator = []
        self._pending_reference_context = None
        self._written_action_count = 0
        self._written_scan_count = 0
        self.parameter_order = []
        offset = 0
        for name, module in self.model.model.named_modules():
            if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                for field in ("weight_by_sample", "bias_by_sample"):
                    parameter = getattr(module, field)
                    width = parameter.shape[1]
                    self.parameter_order.append({"name": name + "." + field,
                        "shape": list(parameter.shape[1:]), "dtype": str(parameter.dtype),
                        "start": offset, "stop": offset + width})
                    offset += width
        self.parameter_order_sha256 = digest(self.parameter_order)
        expected_order = self.cfg.get("parameter_order_sha256")
        if expected_order is not None and expected_order != self.parameter_order_sha256:
            raise ValueError("adapted parameter order changed")
        self.view_state_sha256 = None
        self.forward_counts = {"ramen_full_batch": 0, "baseline_view_rows": 0,
                               "action_full_batch": 0, "action_view_rows": 0}

    def set_query_identity(self, sample_indices):
        if self.identity is not None:
            raise RuntimeError("stale exact-objective query identity")
        if not torch.is_tensor(sample_indices) or sample_indices.ndim != 1 or sample_indices.dtype != torch.long:
            raise ValueError("query identities must be one-dimensional integer sample indices")
        self.identity = sample_indices.detach().cpu().tolist()

    def set_query_images(self, pixels, preprocess):
        if self.raw_context is not None:
            raise RuntimeError("stale exact-objective raw-image context")
        self.raw_context = (np.array(pixels, copy=True), preprocess)
        state = {"view_spec": EXACT_VIEW_SPEC, "preprocess": preprocess_spec(preprocess)}
        state["sha256"] = digest(state)
        if self.view_state["preprocess"] is not None and self.view_state != state:
            raise ValueError("preprocessing changed within the stream")
        expected = self.cfg.get("view_state_sha256")
        if expected is not None and expected != state["sha256"]:
            raise ValueError("preprocessing differs from the locked smoke state")
        self.view_state = state
        self.view_state_sha256 = state["sha256"]

    def pop_evaluator_payload(self):
        result = {"queries": self._pending_evaluator,
                  "reference_context": self._pending_reference_context}
        self._pending_evaluator = []
        self._pending_reference_context = None
        return result

    def reset(self):
        super().reset()
        self.identity = None
        self.raw_context = None
        self.scan = []
        self.action_rows = []
        self._pending_evaluator = []
        self._pending_reference_context = None
        self._written_action_count = 0
        self._written_scan_count = 0
        self.forward_counts = {"ramen_full_batch": 0, "baseline_view_rows": 0,
                               "action_full_batch": 0, "action_view_rows": 0}

    def forward(self, x):
        identity, self.identity = self.identity, None
        raw_context, self.raw_context = self.raw_context, None
        if identity is None or len(identity) != len(x):
            raise RuntimeError("missing or mismatched query identities")
        if self.mode != "scan" and (raw_context is None or len(raw_context[0]) != len(x)):
            raise RuntimeError("missing or mismatched current raw corrupted images")
        self.batch_ids = identity
        self.current_pixels, self.preprocess = raw_context if raw_context is not None else (None, None)
        try:
            return self._forward_exact(x)
        finally:
            self.identity = None
            self.raw_context = None
            self.current_pixels = None
            self.preprocess = None
            if hasattr(self, "_current_batch"):
                del self._current_batch
            self.model.reset_parameters()
            self.model.optimizer.zero_grad()

    def _temporary_outputs(self, x, gradient):
        self.model.reset_parameters()
        self.model.optimizer.zero_grad()
        with torch.no_grad():
            for module in self.model.model.modules():
                if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                    module.recent_B = len(x)
            for group in self.model.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.requires_grad:
                        parameter.grad = torch.zeros_like(parameter)
        self.model.set_by_sample_grad(gradient)
        self.model.step_and_zero_grad()
        with torch.no_grad():
            features = self.model.featurize(x).detach().clone()
            logits = self.model.classify(features).detach().clone()
        return logits, features

    def _capture_packed_source_rows(self, source_rows):
        snapshots = []
        for module in self.model.model.modules():
            if not isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                continue
            for field in ("weight_by_sample", "bias_by_sample"):
                parameter = getattr(module, field)
                snapshots.append((parameter, parameter[source_rows].detach().clone()))
        return snapshots

    @torch.no_grad()
    def _install_packed_rows(self, snapshots, start, count, repeat=8):
        packed_count = count * repeat
        capacities = [module.max_batch_size for module in self.model.model.modules()
                      if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm))]
        if capacities and packed_count > min(capacities):
            raise ValueError("packed view batch exceeds adapted model capacity")
        for parameter, selected in snapshots:
            parameter[:packed_count].copy_(selected[start:start + count].repeat_interleave(repeat, dim=0))

    def _view_logits(self, source_rows, view_tensors):
        results = []
        max_rows = min(12, self.cfg.get("max_batch_size", 100) // 8)
        if max_rows < 1:
            raise ValueError("max adapted batch size must fit all eight views")
        snapshots = self._capture_packed_source_rows(source_rows)
        for start in range(0, len(source_rows), max_rows):
            rows = source_rows[start:start + max_rows]
            self._install_packed_rows(snapshots, start, len(rows))
            packed = torch.stack([tensor for row in view_tensors[start:start + max_rows] for tensor in row]).to(
                device=self.device, dtype=next(self.model.model.parameters()).dtype)
            with torch.no_grad():
                logits = self.model(packed).detach().float().reshape(len(rows), 8, -1)
            results.append(logits)
        return torch.cat(results, dim=0) if results else torch.empty((0, 8, self.num_classes), device=self.device)

    def _cache_membership_snapshot(self):
        rows = []
        for class_id, cache in enumerate(self.cache):
            rows.append({"class": class_id, "size": cache.size,
                         "identities": cache.identities[:cache.size]})
        return digest(rows)

    def _candidate_pools(self, features, row, retrieved):
        pools = {}
        identity_pools = {}
        for class_id in sorted(retrieved):
            distances, base_indices, top_m = retrieved[class_id]
            cache = self.cache[class_id]
            indices = base_indices[row].tolist()
            indices += [idx for idx in top_m[row].tolist() if idx not in indices]
            indices = indices[:self.cfg["candidate_m"]]
            selected_distances = distances[row, indices]
            entropies = cache.entropies[indices]
            weights = torch.exp(-entropies) * torch.exp(-self.beta * selected_distances)
            pool = {"k": len(base_indices[row]), "values": cache.values[indices], "weights": weights,
                    "distances": selected_distances, "entropies": entropies,
                    "metadata": [cache.identities[idx] for idx in indices]}
            pool["base_contribution"] = (pool["values"][:pool["k"]] * pool["weights"][:pool["k"], None]).sum(0)
            pools[class_id] = pool
            identity_pools[str(class_id)] = {
                "k": pool["k"], "values_sha256": tensor_sha(pool["values"]),
                "weights_sha256": tensor_sha(pool["weights"]),
                "candidates": [{**pool["metadata"][rank], "rank": rank,
                                "distance": float(pool["distances"][rank]),
                                "cached_entropy": float(pool["entropies"][rank]),
                                "ramen_weight": float(pool["weights"][rank])}
                               for rank in range(len(pool["metadata"]))]}
        return pools, identity_pools

    def _raw_views_for_query(self, row):
        source = self.current_pixels[row]
        original = self.preprocess(Image.fromarray(source, mode="RGB")).to(self.device)
        if not torch.equal(original, self._current_batch[row]):
            raise ValueError("raw corrupted pixels do not reproduce the original CLIP input")
        images = raw_views(source)
        tensors = [self.preprocess(image).to(self.device) for image in images]
        return tensors, {"raw_image_sha256": tensor_sha(torch.from_numpy(source)),
                         "input_sha256": tensor_sha(self._current_batch[row]),
                         "view_tensor_sha256": [tensor_sha(tensor) for tensor in tensors]}

    def _forward_exact(self, x):
        if len(x) != 100:
            raise ValueError("exact-objective audit requires complete 100-row Ramen batches")
        started = time.perf_counter()
        self._current_batch = x
        start = self.counter
        features = self.model.featurize(x)
        logits = self.model.classify(features)
        predictions = logits.argmax(-1)
        self.last_diagnostics = {"pre_adaptation_prediction": predictions.detach(),
                                 "pre_adaptation_ood_score": -torch.logsumexp(logits.detach(), dim=1)}
        self.loss_fn(logits).backward()
        gradients = self.model.get_by_sample_grad().detach().clone()
        with torch.no_grad():
            entropies = softmax_entropy(logits, reduction="none")
            priorities = torch.arange(start, start + len(x), device=self.device, dtype=self.dtype)
            self.counter += len(x)
            for row in range(len(x)):
                self.cache[int(predictions[row])].add_identity(
                    features[row], gradients[row], entropies[row], priorities[row],
                    {"timestep": start + row, "sample_idx": self.batch_ids[row]})
            self.last_diagnostics["memory_bytes"] = self.memory_bytes
            base = torch.zeros_like(gradients)
            active_classes = 0
            for cache in self.cache:
                if not cache.size:
                    continue
                values, _, entropy, distances = cache.query(features, self.cfg["topk"])
                weights = torch.exp(-entropy) * torch.exp(-self.beta * distances)
                base += (values * weights.unsqueeze(2)).sum(1)
                active_classes += 1
            base /= active_classes
        baseline, baseline_features = self._temporary_outputs(x, base)
        self.forward_counts["ramen_full_batch"] += 1
        if not torch.isfinite(base).all() or not torch.isfinite(baseline).all():
            raise ValueError("nonfinite exact-objective Ramen baseline")
        self._current_batch = x
        eligible = any(cache.size >= self.cfg["candidate_m"] for cache in self.cache)
        cache_snapshot_sha = self._cache_membership_snapshot()
        for row, sample_idx in enumerate(self.batch_ids):
            self.scan.append({"timestep": start + row, "sample_idx": sample_idx,
                              "eligible": bool(eligible), "baseline_sha256": tensor_sha(baseline[row]),
                              "aggregate_sha256": tensor_sha(base[row]),
                              "cache_snapshot_sha256": cache_snapshot_sha})
        scheduled = []
        for row, sample_idx in enumerate(self.batch_ids):
            timestep = start + row
            if timestep not in self.registered:
                continue
            if self.registered[timestep] != sample_idx:
                raise ValueError("scored stream identity differs from the frozen opaque schedule")
            if not eligible:
                raise ValueError("registered query lacks ten-candidate cache eligibility")
            scheduled.append(row)
        if self.mode == "scan" or not scheduled:
            return baseline

        retrieval_started = time.perf_counter()
        retrieved = {}
        with torch.no_grad():
            for class_id, cache in enumerate(self.cache):
                if cache.size:
                    distances = cache_distances(features.detach().to(cache.dtype), cache.keys[:cache.size])
                    k = min(self.cfg["topk"], cache.size)
                    retrieved[class_id] = (
                        distances,
                        distances.topk(k, dim=1, largest=False, sorted=True).indices,
                        distances.topk(min(self.cfg["candidate_m"], cache.size), dim=1,
                                       largest=False, sorted=True).indices)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        preprocess = self.preprocess
        state = {"view_spec": EXACT_VIEW_SPEC, "preprocess": preprocess_spec(preprocess)}
        state["sha256"] = digest(state)
        if self.view_state_sha256 is not None and self.view_state_sha256 != state["sha256"]:
            raise ValueError("view preprocessing state is not stable")
        self.view_state = state
        view_tensors, view_records = [], {}
        for row in scheduled:
            tensors, record = self._raw_views_for_query(row)
            view_tensors.append(tensors)
            view_records[row] = record
        # θ_R view objectives use the exact Ramen parameter row copied to each
        # of its eight raw views. The ordinary output/cache have already been frozen.
        baseline_view_logits = self._view_logits(scheduled, view_tensors)
        self.forward_counts["baseline_view_rows"] += len(scheduled) * 8
        baseline_predictions = baseline.argmax(-1)
        caption_ids = baseline_predictions.detach().cpu().tolist()
        text_features = self.model.text_feature.detach().float()
        caption_bank = text_features[baseline_predictions]
        caption_bank_sha = tensor_sha(caption_bank)
        baseline_objectives = {}
        for position, row in enumerate(scheduled):
            baseline_objectives[row] = objective_values(
                baseline[row], baseline_view_logits[position], baseline_features[row],
                caption_bank, self.clip_logit_scale)
        # Restore baseline Ramen state after the packed view forwards and verify
        # that the auxiliary branch did not change ordinary batch output.
        replay_logits, _ = self._temporary_outputs(x, base)
        if not torch.equal(replay_logits, baseline):
            raise ValueError("exact-objective baseline replay changed Ramen logits")

        jobs = []
        with torch.no_grad():
            for row in scheduled:
                pools, candidate_pool = self._candidate_pools(features, row, retrieved)
                swaps = sorted(legal_swaps(pools, self.cfg["candidate_m"]),
                               key=lambda action: (action["class"], action["out"], action["in"]))
                if not swaps:
                    raise ValueError("registered query has no legal same-pseudo-class substitutions")
                if not torch.equal(aggregate_pools(pools, base[row]), base[row]):
                    raise ValueError("candidate pool aggregation differs from ordinary Ramen")
                # Exact action enumeration is checked against serial arithmetic;
                # no identity or action is removed after this check.
                base_direction = _pack_signs(base[row]) if self.mode == "stage-a" else None
                action_records = []
                packed_directions = []
                for index, action in enumerate(swaps):
                    serial_gradient = aggregate_pools(pools, base[row], action)
                    vectorized_gradient = aggregate_swaps(pools, base[row], [action])[0]
                    if not torch.equal(serial_gradient, vectorized_gradient):
                        raise ValueError("vectorized candidate aggregate differs from serial Ramen arithmetic")
                    pool = pools[action["class"]]
                    direction = _pack_signs(serial_gradient) if self.mode == "stage-a" else None
                    if direction is not None:
                        packed_directions.append(direction)
                    action_record = {
                        "index": index,
                        "action_id": f"swap-{index:05d}",
                        "class": action["class"], "out": action["out"], "in": action["in"],
                        "outgoing": {**pool["metadata"][action["out"]], "rank": action["out"]},
                        "incoming": {**pool["metadata"][action["in"]], "rank": action["in"]},
                        "aggregate_gradient_sha256": tensor_sha(serial_gradient),
                    }
                    if direction is not None:
                        action_record["parameter_delta_sha256"] = digest({
                            "candidate_direction": direction, "base_direction": base_direction,
                            "lr": self.cfg["lr"]})
                    action_records.append(action_record)
                candidate_pool_sha = digest(candidate_pool)
                jobs.append({"row": row, "pools": pools, "swaps": swaps,
                             "packed_directions": packed_directions,
                             "candidate_pool": candidate_pool, "candidate_pool_sha256": candidate_pool_sha,
                             "action_records": action_records, "trial_logits": [None] * len(swaps),
                             "trial_objectives": [None] * len(swaps),
                             "baseline_objectives": baseline_objectives[row],
                             "view_tensors": view_tensors[scheduled.index(row)],
                             "view_record": view_records[row]})

        max_actions = max(len(job["swaps"]) for job in jobs)
        view_forward_calls = 0
        action_forward_calls = 0
        grouped_isolation_forwards = 0
        view_query_chunk = min(12, self.cfg.get("max_batch_size", 100) // 8)
        action_started = time.perf_counter()
        for action_index in range(max_actions):
            active_jobs = [job for job in jobs if action_index < len(job["swaps"])]
            for chunk_start in range(0, len(active_jobs), self.cfg["action_chunk_size"]):
                chunk = active_jobs[chunk_start:chunk_start + self.cfg["action_chunk_size"]]
                trial_gradient = base.clone()
                for job in chunk:
                    action = job["swaps"][action_index]
                    trial_gradient[job["row"]] = aggregate_pools(job["pools"], base[job["row"]], action)
                trial_logits, trial_features = self._temporary_outputs(x, trial_gradient)
                action_forward_calls += 1
                self.forward_counts["action_full_batch"] += 1
                active_rows = [job["row"] for job in chunk]
                packed_views = [job["view_tensors"] for job in chunk]
                action_view_logits = self._view_logits(active_rows, packed_views)
                view_forward_calls += (len(chunk) + view_query_chunk - 1) // view_query_chunk
                self.forward_counts["action_view_rows"] += len(chunk) * 8
                for position, job in enumerate(chunk):
                    row = job["row"]
                    job["trial_logits"][action_index] = trial_logits[row].detach().float().cpu()
                    job["trial_objectives"][action_index] = objective_values(
                        trial_logits[row], action_view_logits[position], trial_features[row],
                        caption_bank, self.clip_logit_scale)
                if (action_index == 0 and chunk_start == 0 and
                        getattr(self.model, "model_type", None) == "vit" and len(chunk) > 1):
                    for job in chunk[:2]:
                        row = job["row"]
                        isolated_gradient = base.clone()
                        isolated_gradient[row] = aggregate_pools(
                            job["pools"], base[row], job["swaps"][action_index])
                        isolated_logits, _ = self._temporary_outputs(x, isolated_gradient)
                        grouped_isolation_forwards += 1
                        if not torch.equal(isolated_logits[row], trial_logits[row]):
                            raise ValueError("grouped ViT action differs from isolated full-batch trial")

        replay_logits, _ = self._temporary_outputs(x, base)
        if not torch.equal(replay_logits, baseline):
            raise ValueError("ordinary Ramen replay changed after exact action scoring")
        cache_snapshot_end = self._cache_membership_snapshot()
        if cache_snapshot_end != cache_snapshot_sha:
            raise ValueError("exact action trials changed Ramen cache membership/history")
        scoring_ms = (time.perf_counter() - action_started) * 1000.0
        for job in jobs:
            row = job["row"]
            objective_vectors = {name: [] for name in self.score_names}
            for index, action_record in enumerate(job["action_records"]):
                values = job["trial_objectives"][index]
                if set(values) != set(OBJECTIVE_NAMES):
                    raise ValueError("candidate action omitted a frozen exact objective")
                deltas = objective_deltas(job["baseline_objectives"], values)
                action_record["objective_losses"] = {name: values[name] for name in self.score_names}
                action_record["objective_deltas"] = {name: deltas[name] for name in self.score_names}
                for name in self.score_names:
                    objective_vectors[name].append(deltas[name])
            selections = {name: select_objective_action(values)
                          for name, values in objective_vectors.items()}
            baseline_record = {"objective_losses": {name: job["baseline_objectives"][name]
                                                     for name in self.score_names},
                               "objective_deltas": {name: 0.0 for name in self.score_names}}
            all_actions = [{"index": None, "action_id": "no-swap", **baseline_record},
                           *job["action_records"]]
            row_identity = {"timestep": start + row, "sample_idx": self.batch_ids[row],
                            "batch": (start + row) // 100, "block": (start + row) // 64,
                            "batch_row": row}
            record = {
                "schema_version": self.schema_version, "protocol": "qcgs-exact-objective-audit-v1.1",
                "stage": self.mode, "query": row_identity,
                "dataset_fingerprint": self.fingerprint, "source_revision": self.source_revision,
                "registry_source_revision": self.registry_source_revision,
                "protocol_sha256": self.protocol_sha256, "run_identity_sha256": self.run_identity_sha256,
                "config_sha256": self.config_sha256, "registry_sha256": self.registry_sha256,
                "parameter_order_sha256": self.parameter_order_sha256,
                "view_state": self.view_state,
                "view_record": job["view_record"], "caption_bank_sha256": caption_bank_sha,
                "caption_bank_source": "per_row_ordinary_ramen_theta_R",
                "caption_pseudo_classes": caption_ids,
                "caption_bank_pseudo_classes_sha256": digest(caption_ids),
                "caption_bank_baseline_logits_sha256": tensor_sha(baseline),
                "ordinary_ramen_batch_features_sha256": tensor_sha(baseline_features),
                "baseline_logits_sha256": tensor_sha(baseline[row]),
                "baseline_pseudo_class": int(baseline_predictions[row]),
                "baseline_feature_sha256": tensor_sha(baseline_features[row]),
                "ramen_aggregate_sha256": tensor_sha(base[row]),
                "baseline_objectives": {name: job["baseline_objectives"][name] for name in self.score_names},
                "cache_snapshot_sha256": cache_snapshot_sha,
                "active_class_count": len(job["pools"]),
                "candidate_pool": job["candidate_pool"],
                "candidate_pool_sha256": job["candidate_pool_sha256"],
                "legal_swap_count": len(job["swaps"]), "legal_actions": job["swaps"],
                "actions": all_actions, "selections": selections,
                **({"stage_b_objective": self.frozen_objective} if self.mode == "stage-b" else {}),
                "parity": {"serial_aggregate_all_actions": True,
                           "baseline_replay_before_trials": True,
                           "baseline_replay_after_trials": True,
                           "cache_membership_unchanged": True,
                           "candidate_parameters_reset_per_trial": True,
                           "action_chunk_isolation_verified": (self.cfg["action_chunk_size"] == 1 or
                               len(jobs) <= 1 or grouped_isolation_forwards >= 2),
                           "action_chunk_isolation_scope": (
                               "serial_no_grouped_queries" if self.cfg["action_chunk_size"] == 1 else
                               "single_query_chunk_no_grouping" if len(jobs) <= 1 else
                               "first_action_first_chunk_first_two_queries"),
                           "ordinary_ramen_logits_returned": True},
                "costs": {"retrieval_batch_ms": retrieval_ms, "candidate_action_ms": scoring_ms,
                          "candidate_trial_count": len(job["swaps"]),
                          "full_batch_action_forwards_batch": action_forward_calls,
                          "eight_view_action_forward_calls_batch": view_forward_calls,
                          "grouped_isolation_check_forwards_batch": grouped_isolation_forwards,
                          "action_chunk_size": self.cfg["action_chunk_size"],
                          "view_tensor_rows": (len(job["swaps"]) + 1) * 8,
                          "theta0_forward_count": 1, "theta_r_forward_count": 1,
                          "baseline_parity_replays_batch": 2,
                          "peak_device_memory_bytes": (self.device_memory_summary()["bytes"]
                              if self.device.type == "cuda" and hasattr(self, "device_memory_summary") else None)},
            }
            record["record_sha256"] = digest(record)
            self.action_rows.append(record)
            evaluator_actions = []
            for index, action_record in enumerate(job["action_records"]):
                evaluator_actions.append({"action": action_record,
                    "logits": job["trial_logits"][index],
                    "packed_direction": job["packed_directions"][index] if self.mode == "stage-a" else None})
            if self.mode in ("stage-a", "stage-b"):
                self._pending_evaluator.append({
                    "stage": self.mode, "query": row_identity,
                    "baseline_logits": baseline[row].detach().float().cpu(),
                    "base_direction": _pack_signs(base[row]) if self.mode == "stage-a" else None,
                    "actions": evaluator_actions,
                    "selections": selections, "score_names": list(self.score_names),
                    "record_sha256": record["record_sha256"],
                    "candidate_pool_sha256": job["candidate_pool_sha256"],
                    "registry_sha256": self.registry_sha256,
                })
        self._pending_reference_context = ({"ramen_gradient": base.detach().cpu()}
                                           if self.mode == "stage-a" else None)
        del self._current_batch
        return baseline
