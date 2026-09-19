"""Evaluator-only QCGS diagnostic. Its continuing trajectory remains Ramen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .OracleSupportUtilityProbe import (
    OracleSupportUtilityProbe, aggregate_pools, aggregate_swaps, legal_swaps, logit_metrics,
    BySampleLayerNorm, BySampleBatchNorm,
)
from .Ramen import cache_distances
from .losses import softmax_entropy
from .query_gradient_selection import score_swaps, select_positive, random_swap
from runtime.query_gradient_registry import digest, file_sha, validate_registry


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class QueryGradientUtilityProbe(OracleSupportUtilityProbe):
    diagnostic_kind = "qcgs"

    def __init__(self, model, datasets, args):
        super().__init__(model, datasets, args)
        self.initialize_diagnostic()

    def initialize_diagnostic(self):
        self.mode = self.cfg["qcgs_mode"]
        if self.mode not in ("scan", "smoke", "stage-a", "stage-b"):
            raise ValueError("invalid QCGS mode")
        self.cell = str(self.cfg["ood_cell"])
        self.fingerprint = self.cfg["dataset_fingerprint"]
        self.identity = None
        self.scan = []
        self.draw_number = 0
        self.registry = None
        self.registered = {}
        self.parameter_order = []
        offset = 0
        for name, module in self.model.model.named_modules():
            if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                for field in ('weight_by_sample', 'bias_by_sample'):
                    parameter = getattr(module, field)
                    size = parameter.shape[1]
                    self.parameter_order.append({'name':name+'.'+field, 'shape':list(parameter.shape[1:]),
                                                 'dtype':str(parameter.dtype), 'start':offset, 'stop':offset+size})
                    offset += size
        self.parameter_order_sha256 = digest(self.parameter_order)
        if self.cfg.get('parameter_order_sha256', self.parameter_order_sha256) != self.parameter_order_sha256:
            raise ValueError('adapted parameter order changed')
        if self.mode.startswith("stage-"):
            path = Path(self.cfg["registry_path"])
            if file_sha(path) != self.cfg["registry_file_sha256"]:
                raise ValueError("registry file changed")
            self.registry = json.loads(path.read_text())
            validate_registry(self.registry)
            if self.registry["dataset_fingerprint"] != self.fingerprint:
                raise ValueError("registry dataset changed")
            self.registered = {r["timestep"]: r for r in self.registry["cells"][self.cell]["selected"][self.mode]}
            if len(self.registered) != self.cfg["probe_queries"]:
                raise ValueError("registry/config quota mismatch")

    def set_query_identity(self, sample_indices):
        if self.identity is not None:
            raise RuntimeError("stale query identity")
        if sample_indices.ndim != 1 or sample_indices.dtype != torch.long:
            raise ValueError("query identities must be integer sample indices")
        self.identity = sample_indices.detach().cpu().tolist()

    def reset(self):
        super().reset()
        self.identity = None
        self.batch_ids = []
        self.scan = []
        self.draw_number = 0

    @staticmethod
    def trial_metrics(logits, label):
        result = logit_metrics(logits, label)
        result.update(prediction=int(logits.argmax()), entropy=float(softmax_entropy(logits.float()[None], reduction="none")[0]))
        if not all(torch.isfinite(torch.tensor(result[k])) for k in ("ce", "entropy", "margin")):
            raise ValueError("nonfinite trial metrics")
        return result

    def forward(self, x):
        identities, self.identity = self.identity, None
        if identities is None or len(identities) != len(x):
            self._pending = None
            raise RuntimeError("missing or mismatched query identities")
        self.batch_ids = identities
        return super().forward(x)

    def _temporary_logits(self, x, gradient):
        self.verification_forwards = getattr(self, 'verification_forwards', 0) + 1
        return super()._temporary_logits(x, gradient)

    def _forward(self, x, labels, is_ood, domains):
        start = self.counter
        features = self.model.featurize(x)
        logits = self.model.classify(features)
        predictions = logits.argmax(-1)
        self.last_diagnostics = {"pre_adaptation_prediction": predictions.detach(),
                                 "pre_adaptation_ood_score": -torch.logsumexp(logits.detach(), dim=1)}
        self.loss_fn(logits).backward()
        query_gradients = self.model.get_by_sample_grad().detach().clone()
        with torch.no_grad():
            entropies = softmax_entropy(logits, reduction="none")
            priorities = torch.arange(start, start + len(x), device=self.device, dtype=self.dtype)
            self.counter += len(x)
            for b in range(len(x)):
                self.cache[int(predictions[b])].add_with_metadata(
                    features[b], query_gradients[b], entropies[b], priorities[b],
                    {"timestep": start+b, "sample_idx": self.batch_ids[b], "domain": int(domains[b]),
                     "is_ood": bool(is_ood[b])})
            self.last_diagnostics["memory_bytes"] = self.memory_bytes
            base, active = torch.zeros_like(query_gradients), 0
            for cache in self.cache:
                if not cache.size:
                    continue
                values, _, entropy, distances = cache.query(features, self.cfg["topk"])
                weights = torch.exp(-entropy) * torch.exp(-self.beta * distances)
                base += (values * weights.unsqueeze(2)).sum(1)
                active += 1
            base /= active
            self.model.set_by_sample_grad(base)
        self.model.step_and_zero_grad()
        with torch.no_grad():
            baseline = self.model(x).detach().clone()
        if not torch.isfinite(baseline).all() or not torch.isfinite(base).all():
            raise ValueError("nonfinite Ramen baseline")
        eligible = any(c.size >= self.cfg["candidate_m"] for c in self.cache)
        selected = []
        for b in range(len(x)):
            scan = {"timestep": start+b, "sample_idx": self.batch_ids[b], "domain": int(domains[b]),
                    "is_ood": bool(is_ood[b]), "eligible": eligible,
                    "baseline_sha256": tensor_sha(baseline[b]), "aggregate_sha256": tensor_sha(base[b])}
            self.scan.append(scan)
            if self.registry is not None:
                expected = self.registry["cells"][self.cell]["stream"][start+b]
                if scan != expected:
                    raise ValueError(f"registry replay mismatch at timestep {start+b}")
            if self.mode == "smoke" and eligible and not is_ood[b] and len(self.rows)+len(selected) < self.cfg["probe_queries"]:
                selected.append(b)
            elif start+b in self.registered:
                selected.append(b)
        if self.mode == "scan" or not selected:
            return baseline

        with torch.no_grad():
            synchronize(self.device)
            retrieval_started = time.perf_counter()
            if self.device.type == "cuda":
                # The stream tracker preserves the global peak before this reset.
                self.reset_cuda_peak()
                scoring_memory_start = torch.cuda.memory_allocated(self.device)
            retrieved = {}
            for c, cache in enumerate(self.cache):
                if cache.size:
                    dist = cache_distances(features.detach().to(cache.dtype), cache.keys[:cache.size])
                    k = min(self.cfg["topk"], cache.size)
                    retrieved[c] = (dist, dist.topk(k, dim=1, largest=False, sorted=True).indices,
                                    dist.topk(min(self.cfg["candidate_m"], cache.size), dim=1, largest=False, sorted=True).indices)
            synchronize(self.device)
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            jobs = []
            # Freeze every label-free action before supervised backward/reference.
            for b in selected:
                synchronize(self.device)
                pool_started = time.perf_counter()
                pools = {}
                for c, (dist, base_indices, top_m) in retrieved.items():
                    cache = self.cache[c]
                    indices = base_indices[b].tolist()
                    indices += [i for i in top_m[b].tolist() if i not in indices]
                    indices = indices[:self.cfg["candidate_m"]]
                    entropy, distances = cache.entropies[indices], dist[b, indices]
                    pool = {"k": len(base_indices[b]), "values": cache.values[indices],
                            "weights": torch.exp(-entropy) * torch.exp(-self.beta * distances),
                            "distances": distances, "entropies": entropy,
                            "metadata": [cache.metadata[i] for i in indices]}
                    pool["base_contribution"] = (pool["values"][:pool["k"]] * pool["weights"][:pool["k"], None]).sum(0)
                    pools[c] = pool
                swaps = list(legal_swaps(pools, self.cfg["candidate_m"]))
                if not swaps:
                    raise ValueError("registered query is not eligible")
                synchronize(self.device)
                pool_ms = (time.perf_counter() - pool_started) * 1000
                audit_started = time.perf_counter()
                if not torch.equal(aggregate_pools(pools, base[b]), base[b]):
                    raise ValueError("QCGS aggregation differs from Ramen")
                synchronize(self.device)
                audit_ms = (time.perf_counter() - audit_started) * 1000
                started = time.perf_counter()
                gradients = aggregate_swaps(pools, base[b], swaps)
                incoming = torch.stack([pools[s["class"]]["values"][s["in"]] for s in swaps])
                outgoing = torch.stack([pools[s["class"]]["values"][s["out"]] for s in swaps])
                scores = score_swaps(query_gradients[b], base[b], gradients, incoming, outgoing, self.cfg["lr"])
                choices = {name: select_positive(scores[name], scores["reasons"].get(name))
                           for name in ("entropy_sign", "entropy_cosine")}
                rng_hash = digest(self.rng.getstate())
                choices["random"] = random_swap(len(swaps), self.rng)
                draw = self.draw_number
                self.draw_number += 1
                synchronize(self.device)
                latency = (time.perf_counter() - started) * 1000
                if not jobs:
                    audit_started = time.perf_counter()
                    for i, swap in enumerate(swaps):
                        if not torch.equal(gradients[i], aggregate_pools(pools, base[b], swap)):
                            raise ValueError("vectorized aggregate differs from serial")
                    synchronize(self.device)
                    audit_ms += (time.perf_counter() - audit_started) * 1000
                if any(choice["invalid"] for choice in choices.values()):
                    raise ValueError(f"invalid selector numerics: {choices}")
                directions = [g.sign().to(torch.int8).cpu().numpy().tobytes() for g in gradients]
                base_key = base[b].sign().to(torch.int8).cpu().numpy().tobytes()
                qnorm = float(query_gradients[b].float().norm())
                pool_identity = {str(c): {"metadata": p["metadata"], "k": p["k"],
                    "values_sha256": tensor_sha(p["values"]), "weights_sha256": tensor_sha(p["weights"])} for c, p in pools.items()}
                records = {}
                for i, swap in enumerate(swaps):
                    record = {"index": i, **swap, "direction_changed": directions[i] != base_key}
                    for name in ("entropy_sign", "entropy_cosine"):
                        record[name+"_score"] = float(scores[name][i]) if scores[name] is not None else None
                    p = pools[swap["class"]]
                    for prefix, rank in (("incoming", swap["in"]), ("outgoing", swap["out"])):
                        value = p["values"][rank].float()
                        norm = float(value.norm())
                        cosine = float(torch.dot(value, query_gradients[b].float()) / (norm*qnorm)) if norm and qnorm else None
                        record[prefix] = {**p["metadata"][rank], "rank": rank,
                                          "feature_distance": float(p["distances"][rank]),
                                          "cached_entropy": float(p["entropies"][rank]), "query_gradient_cosine": cosine}
                    records[i] = record
                jobs.append({"b": b, "choices": choices, "all_records": records,
                    "all_directions": directions, "base_key": base_key, "pool_sha256": digest(pool_identity),
                    "candidate_pool": pool_identity,
                    "legal_actions": swaps, "legal_swap_count": len(swaps), "query_gradient_norm": qnorm,
                    "query_entropy": float(entropies[b]), "score_latency_ms": latency + pool_ms,
                    "retrieval_batch_ms": retrieval_ms, "parity_audit_ms": audit_ms,
                    "rng_before_sha256": rng_hash, "random_draw": draw,
                    "self_support_present": any(m["timestep"] == start+b for p in pools.values() for m in p["metadata"][:p["k"]])})
                del gradients, incoming, outgoing

        scoring_peak = (torch.cuda.max_memory_allocated(self.device) - scoring_memory_start
                        if self.device.type == "cuda" else None)
        if self.device.type == 'cuda':
            self.reset_cuda_peak()
            evaluator_memory_start = torch.cuda.memory_allocated(self.device)
        synchronize(self.device)
        reference_started = time.perf_counter()
        # This reference uses labels solely inside the evaluator, after selection.
        if any(bool(is_ood[j["b"]]) for j in jobs):
            raise ValueError("registered query became OOD")
        self.model.reset_parameters()
        self.model.optimizer.zero_grad()
        reference = self.model(x)
        selected_indices = [j["b"] for j in jobs]
        F.cross_entropy(reference.float()[selected_indices], labels[selected_indices], reduction="sum").backward()
        supervised = self.model.get_by_sample_grad().detach().float().clone()
        del reference
        with torch.no_grad():
            for job in jobs:
                b = job["b"]
                values = []
                for i, key in enumerate(job["all_directions"]):
                    direction = torch.frombuffer(bytearray(key), dtype=torch.int8).to(self.device).float()
                    score = float(self.cfg["lr"] * torch.dot(supervised[b], direction - base[b].sign().float()))
                    job["all_records"][i]["supervised_reference_score"] = score
                    values.append(score)
                if not all(torch.isfinite(torch.tensor(v)) for v in values):
                    raise ValueError("nonfinite supervised reference")
                # Preserve the old first-order control: forced best legal swap.
                index = max(range(len(values)), key=values.__getitem__)
                job["choices"]["supervised_reference"] = {"index": index, "score": values[index], "reason": "supervised_reference", "invalid": False}
                selected_swaps = (range(len(values)) if self.mode == "stage-a" else
                                  sorted({c["index"] for c in job["choices"].values() if c["index"] is not None}))
                job["records"] = {i: job["all_records"][i] for i in selected_swaps}
                job["directions"] = {i: job["all_directions"][i] for i in selected_swaps}
                job["base"] = self.trial_metrics(baseline[b], labels[b])
                job["metrics"] = {job["base_key"]: job["base"]}
                job["pending"] = list(dict.fromkeys(k for k in job["directions"].values() if k != job["base_key"]))
                job["candidate_scores"] = {name: [r[name+"_score"] for r in job["all_records"].values()]
                                           for name in ("entropy_sign", "entropy_cosine", "supervised_reference")}
                job["direction_changes"] = [r["direction_changed"] for r in job["all_records"].values()]
                del job["all_directions"], job["all_records"]
            synchronize(self.device)
            reference_ms = (time.perf_counter() - reference_started) * 1000
            replay_started = time.perf_counter()
            with torch.enable_grad():
                replay = self._temporary_logits(x, base)
            if not torch.equal(replay, baseline):
                raise ValueError("QCGS baseline logits parity failed")
            synchronize(self.device)
            replay_ms = (time.perf_counter() - replay_started) * 1000
            self.verification_forwards = 0
            verify_start = time.perf_counter()
            self._verify_jobs(x, base, labels, jobs)
            synchronize(self.device)
            elapsed = (time.perf_counter() - verify_start) * 1000
            evaluator_peak = (torch.cuda.max_memory_allocated(self.device) - evaluator_memory_start
                              if self.device.type == 'cuda' else None)
            for job in jobs:
                b = job["b"]
                for i, record in job["records"].items():
                    m = job["metrics"][job["directions"][i]]
                    record.update(m, supervised_utility=job["base"]["ce"] - m["ce"], entropy_utility=job["base"]["entropy"] - m["entropy"])
                no_swap = {**job["base"], "supervised_utility": 0., "entropy_utility": 0., "index": None}
                policies = {"ramen": {**no_swap, "reason": "no_swap", "score": 0., "invalid": False}}
                for name, choice in job["choices"].items():
                    policies[name] = {**(no_swap if choice["index"] is None else job["records"][choice["index"]]), **choice}
                best = max([0.] + [r["supervised_utility"] for r in job["records"].values()])
                for policy in policies.values():
                    policy["exact_oracle_regret"] = best - policy["supervised_utility"] if self.mode == "stage-a" else None
                    policy["regret_to_verified_set"] = best - policy["supervised_utility"]
                self.rows.append({"schema_version": 1, "stage": self.mode, "ood_cell": self.cell,
                    "timestep": start+b, "sample_idx": self.batch_ids[b], "domain": int(domains[b]),
                    "batch": (start+b)//100, "block": (start+b)//64, "eligible": eligible,
                    "source_revision": self.cfg.get("source_revision"), "config_sha256": digest(self.cfg),
                    "parameter_order_sha256": self.parameter_order_sha256,
                    "known_label": int(labels[b]), "is_ood": False, "severity": 5,
                    "dataset_fingerprint": self.fingerprint, "registry_sha256": self.registry["sha256"] if self.registry else None,
                    "candidate_pool_sha256": job["pool_sha256"], "baseline_sha256": tensor_sha(baseline[b]),
                    "candidate_pool": job["candidate_pool"], "candidate_scores": job["candidate_scores"],
                    "direction_changes": job["direction_changes"],
                    "ramen": job["base"], "policies": policies, "verified_swaps": list(job["records"].values()),
                    "legal_actions": job["legal_actions"], "legal_swap_count": job["legal_swap_count"],
                    "active_class_count": active, "query_entropy": job["query_entropy"],
                    "query_gradient_norm": job["query_gradient_norm"], "self_support_present": job["self_support_present"],
                    "random_draw": job["random_draw"], "rng_before_sha256": job["rng_before_sha256"],
                    "exact_oracle_utility": best if self.mode == "stage-a" else None, "best_verified_utility": best,
                    "exact_forward_count": len(job["pending"]), "score_latency_ms": job["score_latency_ms"],
                    "retrieval_batch_ms": job["retrieval_batch_ms"], "parity_audit_ms": job["parity_audit_ms"],
                    "supervised_reference_batch_ms": reference_ms, "scoring_peak_extra_bytes": scoring_peak,
                    "evaluator_peak_extra_bytes": evaluator_peak, "baseline_replay_batch_ms": replay_ms,
                    "verification_batch_forward_count": self.verification_forwards,
                    "verification_batch_ms": elapsed, "retained_memory_bytes": self.memory_bytes,
                    "peak_device_memory_bytes": self.device_memory_summary()['bytes'] if self.device.type == "cuda" else None})
                if getattr(self, "probe_progress", None):
                    self.probe_progress()
        return baseline
