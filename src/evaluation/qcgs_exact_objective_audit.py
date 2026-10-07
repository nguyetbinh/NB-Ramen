"""Evaluator-owned outcomes and independent raw analysis for exact QCGS."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import random
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from methods.qcgs_exact_objectives import OBJECTIVE_NAMES, select_objective_action
from runtime.query_gradient_registry import CELLS, digest, file_sha, validate_exact_objective_registry


SCHEMA_VERSION = 1
CURRENT_PROTOCOL = "qcgs-exact-objective-audit-v1.2"
LEGACY_SMOKE_PROTOCOL = "qcgs-exact-objective-audit-v1.1"
LEGACY_PROTOCOL_SHA256 = "f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860"
ANALYSIS = {"version": 1, "protocol": CURRENT_PROTOCOL, "bootstrap_seed": 917,
            "bootstrap_draws": 10000, "block_size": 64, "minimum_blocks": 4,
            "stage_a_minimum_defined_rho": 8, "minimum_replacements": 3,
            "stage_b_remove_largest": 2, "stage_b_tie_rule": "utility descending, sample_idx ascending",
            "objective_tie_order": list(OBJECTIVE_NAMES)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())


def read_jsonl(path):
    rows, receipt = read_jsonl_detailed(path)
    if receipt["errors"] or receipt["truncated_tail"]:
        raise ValueError("malformed or truncated JSONL: " + "; ".join(receipt["errors"] or ["trailing record is incomplete"]))
    return rows


def read_jsonl_detailed(path):
    """Read complete JSONL records and report a malformed final partial write."""
    path = Path(path)
    if not path.exists():
        return [], {"sha256": None, "truncated_tail": False, "errors": []}
    raw = path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [], {"sha256": file_sha256, "truncated_tail": False,
                    "errors": ["file is not valid UTF-8"]}
    lines = content.splitlines(keepends=True)
    rows, errors = [], []
    truncated_tail = False
    for index, line in enumerate(lines):
        value = line.rstrip("\r\n")
        if not value.strip():
            continue
        try:
            rows.append(json.loads(value))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
                truncated_tail = True
            else:
                errors.append(f"malformed JSONL record at line {index + 1}")
    return rows, {"sha256": file_sha256, "truncated_tail": truncated_tail, "errors": errors}


def write_scorer_outputs(method, run_dir):
    """Persist label-free scoring records before evaluator code is called."""
    root = Path(run_dir)
    action_rows = method.action_rows[method._written_action_count:]
    scan_rows = method.scan[method._written_scan_count:]
    serialization_started = time.perf_counter()
    append_jsonl(root / "qcgs-exact-actions.jsonl", action_rows)
    append_jsonl(root / "qcgs-exact-scan.jsonl", scan_rows)
    jsonl_serialization_ms = (time.perf_counter() - serialization_started) * 1000.0
    method._written_action_count = len(method.action_rows)
    method._written_scan_count = len(method.scan)
    write_json(root / "qcgs-exact-state.json", {
        "schema_version": SCHEMA_VERSION,
        "parameter_order": method.parameter_order,
        "parameter_order_sha256": method.parameter_order_sha256,
        "view_state": method.view_state,
        "clip_logit_scale": method.clip_logit_scale,
        "protocol_sha256": method.protocol_sha256,
        "score_boundary": "no labels, OOD/domain metadata, correctness, utility, gradient or oracle values",
    })
    status_path = root / "qcgs-exact-status.json"
    previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    complete = len(method.action_rows) == len(method.query_schedule) if method.mode != "scan" else True
    status = {
        "schema_version": SCHEMA_VERSION, "stage": method.mode,
        "scanned_rows": len(method.scan), "scored_queries": len(method.action_rows),
        "requested_queries": len(method.query_schedule),
        "complete": complete,
    }
    if method.mode in ("stage-a", "stage-b"):
        evaluator_queries = previous_status.get("evaluator_queries", 0)
        status["evaluator_queries"] = evaluator_queries
        status["evaluator_complete"] = complete and evaluator_queries == len(method.query_schedule)
    write_json(status_path, status)
    if getattr(method, "profile_cost_components", False) and action_rows:
        profile_path = root / "qcgs-exact-profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.exists() else {
            "schema_version": 1,
            "stage": method.mode,
            "config_sha256": action_rows[-1]["config_sha256"],
            "protocol_sha256": action_rows[-1]["protocol_sha256"],
            "source_revision": action_rows[-1]["source_revision"],
            "batches": [],
            "profile_scope": "each entry describes one scored stream batch; row-hash time is sidecar-only because it cannot be embedded in its own hash; host wall and GPU event buckets may overlap",
        }
        profile["batches"].append({
            "query_count": len(action_rows),
            "queries": [{"timestep": row["query"]["timestep"],
                         "sample_idx": row["query"]["sample_idx"]} for row in action_rows],
            "component_timing_ms": action_rows[-1]["costs"].get("profile_components_ms", {}),
            "record_sha256_ms": getattr(method, "profile_record_hash_ms", 0.0),
            "jsonl_serialization_ms": jsonl_serialization_ms,
        })
        profile["scored_query_count"] = sum(batch["query_count"] for batch in profile["batches"])
        write_json(profile_path, profile)


def _metric(logits, label):
    logits = logits.detach().float().cpu()
    label = int(label)
    if logits.ndim != 1 or label < 0 or label >= logits.numel() or not torch.isfinite(logits).all():
        raise ValueError("invalid evaluator logits or known-class label")
    logp = F.log_softmax(logits, dim=0)
    probs = logp.exp()
    other = logits.clone()
    other[label] = -torch.inf
    result = {"ce": float(-logp[label]),
              "entropy": float(-torch.special.xlogy(probs, probs).sum()),
              "margin": float(logits[label] - other.max()),
              "prediction": int(logits.argmax()),
              "correct": bool(int(logits.argmax()) == label)}
    if not all(math.isfinite(result[key]) for key in ("ce", "entropy", "margin")):
        raise ValueError("nonfinite evaluator metric")
    return result


def _unpack_signs(value, device):
    size = value["size"]
    positive = np.unpackbits(np.frombuffer(base64.b64decode(value["positive"]), dtype=np.uint8))[:size]
    negative = np.unpackbits(np.frombuffer(base64.b64decode(value["negative"]), dtype=np.uint8))[:size]
    signs = positive.astype(np.float32) - negative.astype(np.float32)
    return torch.from_numpy(signs.copy()).to(device=device)


def _supervised_reference(payloads, labels, model, batch_images, ramen_gradient, lr):
    """Compute the protocol's evaluator-only first-order reference after freeze."""
    if model is None or batch_images is None or ramen_gradient is None:
        raise ValueError("Stage A supervised reference lacks its evaluator-owned model context")
    device = batch_images.device
    model.reset_parameters()
    model.optimizer.zero_grad()
    with torch.no_grad():
        for module in model.model.modules():
            if hasattr(module, "recent_B") and hasattr(module, "weight_by_sample"):
                module.recent_B = len(batch_images)
        for group in model.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad:
                    parameter.grad = torch.zeros_like(parameter)
    model.set_by_sample_grad(ramen_gradient.to(device=device, dtype=next(model.model.parameters()).dtype))
    model.step_and_zero_grad()
    try:
        anchor_logits = model(batch_images)
        row_positions = [int(payload["query"]["batch_row"]) for payload in payloads]
        row_labels = labels.to(device=device, dtype=torch.long)[row_positions]
        for payload, row in zip(payloads, row_positions):
            expected = payload["baseline_logits"].to(anchor_logits)
            if not torch.equal(anchor_logits[row].detach().float(), expected.float()):
                raise ValueError("supervised reference anchor differs from frozen Ramen logits")
        F.cross_entropy(anchor_logits.float()[row_positions], row_labels, reduction="sum").backward()
        supervised = model.get_by_sample_grad().detach().float()
        result = {}
        with torch.no_grad():
            for payload, row in zip(payloads, row_positions):
                base_direction = _unpack_signs(payload["base_direction"], device)
                values = []
                for action in payload["actions"]:
                    candidate = _unpack_signs(action["packed_direction"], device)
                    score = float(lr * torch.dot(supervised[row], candidate - base_direction))
                    if not math.isfinite(score):
                        raise ValueError("nonfinite supervised reference score")
                    values.append(score)
                maximum = max(values)
                selected = values.index(maximum) if maximum > 0 else None
                result[(payload["query"]["timestep"], payload["query"]["sample_idx"])] = {
                    "scores": values, "selected_index": selected,
                    "reason": "positive_canonical_argmax" if selected is not None else "no_positive_reference_score"}
        return result
    finally:
        model.reset_parameters()
        model.optimizer.zero_grad()


class ExactObjectiveEvaluator:
    """One-way evaluator: accepts only frozen scorer payloads and evaluator context."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.draw_number = 0
        self.rows = []

    def evaluate_batch(self, frozen_payload, labels, is_ood, domains, *, model=None, batch_images=None,
                       reference_context=None, lr=0.01, ood_cell=None):
        payloads = frozen_payload["queries"]
        if reference_context is None:
            reference_context = frozen_payload.get("reference_context")
        stage = payloads[0]["stage"] if payloads else frozen_payload.get("stage")
        if not payloads:
            return []
        if any(payload["stage"] != stage for payload in payloads):
            raise ValueError("mixed stages in one evaluator batch")
        labels = labels.detach().cpu()
        is_ood = is_ood.detach().cpu().bool()
        domains = domains.detach().cpu().long()
        reference_scores = {}
        reference_started = time.perf_counter()
        if stage == "stage-a":
            if reference_context is None or "ramen_gradient" not in reference_context:
                raise ValueError("Stage A requires frozen Ramen gradients for its supervised reference")
            reference_scores = _supervised_reference(
                payloads, labels, model, batch_images,
                reference_context["ramen_gradient"], lr)
        reference_batch_ms = (time.perf_counter() - reference_started) * 1000.0
        result = []
        for payload in payloads:
            evaluator_started = time.perf_counter()
            row = payload["query"]["batch_row"]
            timestep, sample_idx = payload["query"]["timestep"], payload["query"]["sample_idx"]
            if row >= len(labels) or bool(is_ood[row]) or int(labels[row]) < 0:
                raise ValueError("registered exact-objective query is not known-ID")
            if int(domains[row]) < 0:
                raise ValueError("invalid evaluator domain identity")
            action_payloads = payload["actions"]
            if not action_payloads:
                raise ValueError("registered query has no legal swaps")
            rng_before = digest(self.rng.getstate())
            random_index = self.rng.randrange(len(action_payloads))
            random_draw = self.draw_number
            self.draw_number += 1
            baseline = _metric(payload["baseline_logits"], labels[row])
            action_indices = range(len(action_payloads)) if stage == "stage-a" else sorted({
                random_index,
                payload["selections"][payload["score_names"][0]]["index"]
                if payload["selections"][payload["score_names"][0]]["index"] is not None else random_index,
            })
            action_metrics = {}
            for index in action_indices:
                action_metrics[index] = _metric(action_payloads[index]["logits"], labels[row])
            utilities = {index: baseline["ce"] - value["ce"] for index, value in action_metrics.items()}
            action_rows = []
            for index in action_indices:
                action = action_payloads[index]["action"]
                action_rows.append({**action, **action_metrics[index], "utility": utilities[index]})
            utility_by_index = utilities
            random_utility = utility_by_index[random_index]
            if stage == "stage-a":
                all_utilities = [baseline["ce"] - _metric(action["logits"], labels[row])["ce"]
                                 for action in action_payloads]
                exact_expected_random = float(np.mean(all_utilities))
                oracle_maximum = max([0.0, *all_utilities])
                oracle_index = all_utilities.index(oracle_maximum) if oracle_maximum > 0 else None
                oracle_utility = oracle_maximum
            else:
                exact_expected_random = None
                oracle_utility = None
                oracle_index = None
            policies = {"no_swap": {"index": None, "utility": 0.0}}
            for objective in payload["score_names"]:
                choice = payload["selections"][objective]
                index = choice["index"]
                if index is not None and index not in utility_by_index:
                    utility_by_index[index] = baseline["ce"] - _metric(action_payloads[index]["logits"], labels[row])["ce"]
                    action_metrics[index] = _metric(action_payloads[index]["logits"], labels[row])
                policies[objective] = {**choice, "utility": 0.0 if index is None else utility_by_index[index]}
                matched_index = random_index if index is not None else None
                policies[objective + "_random_matched"] = {
                    "index": matched_index, "utility": 0.0 if matched_index is None else random_utility}
            policies["random_forced"] = {"index": random_index, "utility": random_utility}
            if stage == "stage-a":
                reference = reference_scores[(timestep, sample_idx)]
                ref_index = reference["selected_index"]
                if ref_index is not None and ref_index not in utility_by_index:
                    utility_by_index[ref_index] = baseline["ce"] - _metric(action_payloads[ref_index]["logits"], labels[row])["ce"]
                    action_metrics[ref_index] = _metric(action_payloads[ref_index]["logits"], labels[row])
                policies["supervised_first_order_reference"] = {
                    "index": ref_index, "utility": 0.0 if ref_index is None else utility_by_index[ref_index],
                    "score": 0.0 if ref_index is None else reference["scores"][ref_index],
                    "scores": reference["scores"], "reason": reference["reason"]}
            record = {
                "schema_version": SCHEMA_VERSION, "protocol": CURRENT_PROTOCOL,
                "stage": stage, "ood_cell": str(ood_cell),
                "query": {**payload["query"], "domain": int(domains[row]),
                          "known_label": int(labels[row]), "is_ood": False},
                "scorer_record_sha256": payload["record_sha256"],
                "candidate_pool_sha256": payload["candidate_pool_sha256"],
                "registry_sha256": payload["registry_sha256"],
                "ramen": {**baseline, "utility": 0.0}, "actions": action_rows,
                "policies": policies, "random_draw": random_draw,
                "random_index": random_index, "rng_before_sha256": rng_before,
                "evaluated_action_indices": sorted(action_metrics),
                "costs": {"evaluator_ms": (time.perf_counter() - evaluator_started) * 1000.0,
                          "evaluator_action_count": len(action_metrics),
                          "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(batch_images.device))
                              if batch_images is not None and batch_images.device.type == "cuda" else None},
            }
            if stage == "stage-a":
                record.update(exact_random_forced_expected_utility=exact_expected_random,
                              exact_oracle_utility=oracle_utility,
                              exact_oracle_action_index=oracle_index)
                record["costs"]["supervised_reference_batch_ms"] = reference_batch_ms
            record["outcome_record_sha256"] = digest(record)
            result.append(record)
            self.rows.append(record)
        return result


def write_evaluator_outputs(rows, run_dir):
    """Write evaluator-owned labels/utilities separately from scorer raw values."""
    root = Path(run_dir)
    path = root / "qcgs-exact-outcomes.jsonl"
    append_jsonl(path, rows)
    status_path = root / "qcgs-exact-status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status["evaluator_queries"] = len(read_jsonl(path))
        status["evaluator_complete"] = (status.get("complete") is True and
                                         status["evaluator_queries"] == status.get("requested_queries"))
        write_json(status_path, status)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _rho(scores, utilities):
    if len(scores) < 2:
        return None, "fewer_than_two_legal_swaps"
    if not all(_finite(v) for v in scores + utilities):
        return None, "undefined_or_nonfinite_action_value"
    if len(set(scores)) < 2:
        return None, "constant_objective_vector"
    if len(set(utilities)) < 2:
        return None, "constant_utility_vector"
    value = float(np.corrcoef(_average_ranks(scores), _average_ranks(utilities))[0, 1])
    return (value, None) if math.isfinite(value) else (None, "undefined_rank_correlation")


def _canonical_actions(row):
    pools = row["candidate_pool"]
    expected = []
    for key in sorted(pools, key=int):
        pool = pools[key]
        k = pool["k"]
        for outgoing in range(k):
            for incoming in range(k, len(pool["candidates"])):
                expected.append({"class": int(key), "out": outgoing, "in": incoming})
    return expected


def audit_scorer_rows(action_rows, *, allowed_protocols=None):
    """Check label-free action records without opening evaluator outcomes."""
    allowed_protocols = {CURRENT_PROTOCOL} if allowed_protocols is None else set(allowed_protocols)
    forbidden = {"label", "known_label", "is_ood", "domain", "correct", "correctness", "utility", "oracle",
                 "supervised_gradient", "supervised_reference", "known_label_or_minus_one",
                 "corruption", "corruption_type", "corruption_name", "corruption_id", "severity",
                 "metadata", "evaluator_metadata", "correlations", "rho", "rank_correlation",
                 "sign_precision", "action_utilities", "all_action_utilities",
                 "supervised_first_order_reference"}

    def check_keys(value):
        if isinstance(value, dict):
            if forbidden & set(value):
                raise ValueError("label/metadata/utility field entered a scorer record")
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    seen = set()
    for row in action_rows:
        check_keys(row)
        if row.get("schema_version") != SCHEMA_VERSION or row.get("protocol") not in allowed_protocols:
            raise ValueError("unknown exact-objective scorer schema")
        if row.get("stage") not in ("smoke", "stage-a", "stage-b"):
            raise ValueError("scorer record lacks its frozen stage identity")
        expected_hash = row["record_sha256"]
        if expected_hash != digest({key: value for key, value in row.items() if key != "record_sha256"}):
            raise ValueError("exact-objective scorer record hash mismatch")
        # The same base image and timestep may be audited in both OOD cells;
        # config identity separates those streams while still rejecting a
        # duplicate within one frozen run configuration.
        identity = (row["stage"], row["config_sha256"], row["query"]["sample_idx"])
        if identity in seen:
            raise ValueError("duplicate exact-objective scorer query")
        seen.add(identity)
        query = row["query"]
        if (query["batch"] != query["timestep"] // 100 or query["block"] != query["timestep"] // 64 or
                query["batch_row"] != query["timestep"] % 100):
            raise ValueError("scorer batch-row/block identity mismatch")
        for key in ("config_sha256", "parameter_order_sha256", "candidate_pool_sha256",
                    "cache_snapshot_sha256", "caption_bank_sha256", "run_identity_sha256",
                    "baseline_logits_sha256", "baseline_feature_sha256", "ramen_aggregate_sha256",
                    "caption_bank_pseudo_classes_sha256", "caption_bank_baseline_logits_sha256",
                    "ordinary_ramen_batch_features_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("missing exact-objective provenance hash: " + key)
        if type(row["baseline_pseudo_class"]) is not int or not 0 <= row["baseline_pseudo_class"] < 80:
            raise ValueError("invalid baseline Ramen pseudo-class")
        parity = row.get("parity", {})
        required_parity = {"serial_aggregate_all_actions", "baseline_replay_before_trials",
                           "baseline_replay_after_trials", "cache_membership_unchanged",
                           "candidate_parameters_reset_per_trial", "action_chunk_isolation_verified",
                           "action_chunk_isolation_scope", "ordinary_ramen_logits_returned"}
        if set(parity) != required_parity or not all(
                               value is True for key, value in parity.items()
                               if key != "action_chunk_isolation_scope"):
            raise ValueError("Ramen parity/reset/cache checks are incomplete")
        if row["candidate_pool_sha256"] != digest(row["candidate_pool"]):
            raise ValueError("candidate pool content hash mismatch")
        actions = _canonical_actions(row)
        if actions != row["legal_actions"] or len(actions) != row["legal_swap_count"]:
            raise ValueError("legal exact-objective actions are not exhaustive/canonical")
        if len(row["actions"]) != len(actions) + 1 or row["actions"][0]["action_id"] != "no-swap":
            raise ValueError("scorer no-swap/action coverage mismatch")
        for index, expected in enumerate(actions):
            saved = row["actions"][index + 1]
            if (saved.get("index") != index or saved.get("action_id") != f"swap-{index:05d}" or
                    any(saved.get(key) != expected[key] for key in ("class", "out", "in"))):
                raise ValueError("scorer action identity/order differs from canonical enumeration")
            pool = row["candidate_pool"].get(str(expected["class"]))
            if pool is None:
                raise ValueError("scorer action refers to a missing candidate class")
            for side, rank_key in (("outgoing", "out"), ("incoming", "in")):
                rank = expected[rank_key]
                try:
                    candidate = pool["candidates"][rank]
                    saved_identity = saved[side]
                except (IndexError, KeyError, TypeError) as exc:
                    raise ValueError("scorer action candidate identity is missing") from exc
                if any(saved_identity.get(key) != candidate.get(key)
                       for key in ("timestep", "sample_idx", "rank")):
                    raise ValueError("scorer action candidate differs from its frozen pool rank")
            aggregate_hash = saved.get("aggregate_gradient_sha256")
            if not isinstance(aggregate_hash, str) or len(aggregate_hash) != 64:
                raise ValueError("scorer action lacks an aggregate-gradient hash")
        objective_names = set(row["selections"])
        expected_names = set(OBJECTIVE_NAMES) if row["stage"] in ("stage-a", "smoke") else {row["stage_b_objective"]}
        if objective_names != expected_names:
            raise ValueError("scorer objective set differs from frozen stage")
        for name in objective_names:
            baseline = row["baseline_objectives"][name]
            if not _finite(baseline):
                raise ValueError("nonfinite Ramen objective")
            if row["actions"][0]["objective_losses"][name] != baseline or row["actions"][0]["objective_deltas"][name] != 0.0:
                raise ValueError("no-swap must preserve baseline loss and zero improvement")
            scores = []
            for action in row["actions"][1:]:
                loss = action["objective_losses"][name]
                delta = action["objective_deltas"][name]
                if not _finite(loss) or not _finite(delta) or not math.isclose(
                        delta, baseline - loss, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("objective action loss/delta arithmetic mismatch")
                scores.append(delta)
            expected_choice = select_objective_action(scores)
            actual_choice = row["selections"][name]
            if (actual_choice.get("index") != expected_choice["index"] or
                    actual_choice.get("reason") != expected_choice["reason"] or
                    not math.isclose(actual_choice.get("score"), expected_choice["score"],
                                     rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("objective selection/tie/constant rule mismatch")
        view = row["view_record"]
        if (len(view["view_tensor_sha256"]) != 8 or
                any(not isinstance(value, str) or len(value) != 64 for value in view["view_tensor_sha256"])):
            raise ValueError("exact ordered view tensor hashes are missing")
        if len(row["caption_pseudo_classes"]) != 100:
            raise ValueError("pseudo-caption bank must map all 100 visible stream rows")
        if (row.get("caption_bank_source") != "per_row_ordinary_ramen_theta_R" or
                row["caption_bank_pseudo_classes_sha256"] != digest(row["caption_pseudo_classes"])):
            raise ValueError("caption bank is not bound to per-row ordinary Ramen baseline outputs")
        if not all(type(value) is int and 0 <= value < 80 for value in row["caption_pseudo_classes"]):
            raise ValueError("pseudo-caption bank contains an invalid known-class prediction")
        chunk_size = row["costs"].get("action_chunk_size")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("missing positive locked action chunk size")
        profile = row["costs"].get("profile_components_ms")
        if profile is not None and (not isinstance(profile, dict) or any(
                not isinstance(key, str) or not _finite(value) or value < 0
                for key, value in profile.items())):
            raise ValueError("invalid component-level smoke timing record")
        grouped_checks = row["costs"].get("grouped_isolation_check_forwards_batch")
        if type(grouped_checks) is not int or grouped_checks < 0:
            raise ValueError("missing grouped-query isolation check count")
        expected_scope = ("serial_no_grouped_queries" if chunk_size == 1 else
                          "first_action_first_chunk_first_two_queries" if grouped_checks >= 2 else
                          "single_query_chunk_no_grouping" if grouped_checks == 0 else None)
        if parity["action_chunk_isolation_scope"] != expected_scope:
            raise ValueError("action-chunk isolation receipt overstates its checked query scope")
    return True


def _verify_pair(action_row, outcome_row):
    audit_scorer_rows([action_row])
    for row, hash_key in ((action_row, "record_sha256"), (outcome_row, "outcome_record_sha256")):
        expected_hash = row[hash_key]
        payload = {key: value for key, value in row.items() if key != hash_key}
        if expected_hash != digest(payload):
            raise ValueError("exact-objective raw record hash mismatch")
    if (outcome_row.get("schema_version") != SCHEMA_VERSION or
            outcome_row.get("protocol") != CURRENT_PROTOCOL):
        raise ValueError("unknown exact-objective evaluator outcome schema")
    if action_row["stage"] != outcome_row["stage"] or action_row["query"] != {
            key: outcome_row["query"][key] for key in action_row["query"]}:
        raise ValueError("scorer/evaluator query identity mismatch")
    if action_row["record_sha256"] != outcome_row["scorer_record_sha256"]:
        raise ValueError("evaluator outcome is detached from the frozen scorer record")
    if action_row["candidate_pool_sha256"] != outcome_row["candidate_pool_sha256"]:
        raise ValueError("candidate pool hash differs across scorer/evaluator boundary")
    if action_row["stage"] == "stage-b":
        forbidden = {
            "exact_oracle_utility", "exact_oracle_action_index",
            "exact_random_forced_expected_utility", "supervised_first_order_reference",
            "supervised_reference_batch_ms", "correlations", "rho", "rank_correlation",
            "sign_precision", "action_utilities", "all_action_utilities",
        }

        def check_stage_b_scope(value):
            if isinstance(value, dict):
                if forbidden & set(value):
                    raise ValueError("Stage B raw outcome contains a forbidden oracle/rank/sign/reference field")
                for child in value.values():
                    check_stage_b_scope(child)
            elif isinstance(value, list):
                for child in value:
                    check_stage_b_scope(child)

        check_stage_b_scope(outcome_row)
    if action_row.get("registry_sha256") != outcome_row.get("registry_sha256"):
        raise ValueError("registry provenance differs across scorer/evaluator boundary")
    if outcome_row["query"]["is_ood"] is not False or type(outcome_row["query"]["known_label"]) is not int:
        raise ValueError("invalid evaluator-known query")
    if not 0 <= outcome_row["query"]["known_label"] < 80:
        raise ValueError("evaluator label is outside the frozen known-class space")
    actions = _canonical_actions(action_row)
    if actions != action_row["legal_actions"] or len(actions) != action_row["legal_swap_count"]:
        raise ValueError("legal action set is not exhaustive and canonical")
    if len(action_row["actions"]) != len(actions) + 1 or action_row["actions"][0]["action_id"] != "no-swap":
        raise ValueError("no-swap/action record coverage mismatch")
    for index, expected in enumerate(actions):
        saved = action_row["actions"][index + 1]
        if saved["index"] != index or {key: saved[key] for key in expected} != expected:
            raise ValueError("action record identity/order mismatch")
        pool = action_row["candidate_pool"][str(expected["class"])]
        for prefix, rank in (("outgoing", expected["out"]), ("incoming", expected["in"])):
            identity = pool["candidates"][rank]
            if any(saved[prefix][key] != identity[key] for key in ("timestep", "sample_idx", "rank")):
                raise ValueError("action support identity/rank mismatch")
    names = action_row["selections"].keys()
    for objective in names:
        scores = [entry["objective_deltas"][objective] for entry in action_row["actions"][1:]]
        choice = select_objective_action(scores)
        saved_choice = action_row["selections"][objective]
        if (choice["index"] != saved_choice.get("index") or
                choice["reason"] != saved_choice.get("reason") or
                not math.isclose(choice["score"], saved_choice.get("score"), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("objective selection differs from the frozen score vector")
        baseline_z = action_row["baseline_objectives"][objective]
        if action_row["actions"][0]["objective_losses"][objective] != baseline_z:
            raise ValueError("no-swap objective loss differs from the Ramen anchor")
        for entry in action_row["actions"][1:]:
            if not _finite(entry["objective_losses"][objective]) or not _finite(entry["objective_deltas"][objective]):
                raise ValueError("nonfinite objective action value")
            if not math.isclose(entry["objective_deltas"][objective],
                                baseline_z - entry["objective_losses"][objective], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("objective improvement sign/arithmetic mismatch")
        if action_row["actions"][0]["objective_deltas"][objective] != 0.0:
            raise ValueError("no-swap objective improvement must be exactly zero")
    evaluated = outcome_row["actions"]
    evaluated_map = {entry["index"]: entry for entry in evaluated}
    if len(evaluated_map) != len(evaluated):
        raise ValueError("duplicate evaluator action")
    if action_row["stage"] == "stage-a":
        if set(evaluated_map) != set(range(len(actions))):
            raise ValueError("Stage A did not evaluate every legal action")
    elif action_row["stage"] == "stage-b":
        objective = next(iter(names))
        selected = action_row["selections"][objective]["index"]
        expected_evaluated = {outcome_row["random_index"]}
        if selected is not None:
            expected_evaluated.add(selected)
        if set(evaluated_map) != expected_evaluated:
            raise ValueError("Stage B evaluated actions outside frozen policy/random controls")
    baseline_ce = outcome_row["ramen"]["ce"]
    if not all(_finite(outcome_row["ramen"][field]) for field in ("ce", "entropy", "margin")):
        raise ValueError("nonfinite evaluator Ramen metrics")
    baseline = outcome_row["ramen"]
    if baseline["correct"] != (baseline["prediction"] == outcome_row["query"]["known_label"]):
        raise ValueError("Ramen baseline correctness does not match its prediction and label")
    utilities = {}
    for index, entry in evaluated_map.items():
        if not all(_finite(entry[field]) for field in ("ce", "entropy", "margin")):
            raise ValueError("nonfinite evaluator action metrics")
        if entry["correct"] != (entry["prediction"] == outcome_row["query"]["known_label"]):
            raise ValueError("evaluator correctness does not match prediction and label")
        utility = baseline_ce - entry["ce"]
        if not math.isclose(entry["utility"], utility, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("evaluator utility does not recompute from CE")
        utilities[index] = utility
    if outcome_row["ramen"]["utility"] != 0.0:
        raise ValueError("no-swap utility must be exactly zero")
    no_swap_policy = outcome_row["policies"].get("no_swap", {})
    if (set(no_swap_policy) != {"index", "utility"} or no_swap_policy["index"] is not None or
            not _finite(no_swap_policy["utility"]) or no_swap_policy["utility"] != 0.0):
        raise ValueError("no-swap policy must preserve the baseline with zero utility")
    random_index = outcome_row["random_index"]
    if type(random_index) is not int or random_index not in range(len(actions)):
        raise ValueError("random-forced control is not a legal swap")
    random_policy = outcome_row["policies"]["random_forced"]
    if (set(random_policy) != {"index", "utility"} or
            type(random_policy["index"]) is not int or random_policy["index"] != random_index):
        raise ValueError("random-forced index mismatch")
    if (not _finite(random_policy["utility"]) or
            not math.isclose(random_policy["utility"], utilities[random_index],
                             rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError("random-forced utility mismatch")
    for objective in names:
        choice = action_row["selections"][objective]
        if action_row["stage"] == "stage-b" and objective not in names:
            continue
        policy = outcome_row["policies"][objective]
        if (policy["index"] != choice["index"] or
                (choice["index"] is not None and type(policy["index"]) is not int)):
            raise ValueError("evaluator policy differs from frozen objective choice")
        expected_utility = 0.0 if choice["index"] is None else utilities[choice["index"]]
        if (not _finite(policy["utility"]) or
                not math.isclose(policy["utility"], expected_utility, rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("selected-action utility mismatch")
        matched = outcome_row["policies"][objective + "_random_matched"]
        expected_index = outcome_row["random_index"] if choice["index"] is not None else None
        if (matched["index"] != expected_index or
                (expected_index is not None and type(matched["index"]) is not int)):
            raise ValueError("random-matched draw/abstention mismatch")
        expected_matched_utility = utilities[outcome_row["random_index"]] if expected_index is not None else 0.0
        if (not _finite(matched["utility"]) or
                not math.isclose(matched["utility"], expected_matched_utility,
                                 rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("random-matched utility mismatch")
    if action_row["stage"] == "stage-a":
        if not {"exact_random_forced_expected_utility", "exact_oracle_utility",
                "exact_oracle_action_index"} <= set(outcome_row):
            raise ValueError("Stage A evaluator outcome lacks frozen reference values")
        expected_random = float(np.mean([utilities[i] for i in range(len(actions))]))
        expected_oracle = max([0.0, *[utilities[i] for i in range(len(actions))]])
        if not math.isclose(outcome_row["exact_random_forced_expected_utility"], expected_random, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("exact uniform-random expectation mismatch")
        if not math.isclose(outcome_row["exact_oracle_utility"], expected_oracle, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Stage A exact oracle mismatch")
        expected_oracle_index = None
        if expected_oracle > 0:
            expected_oracle_index = next(i for i in range(len(actions)) if utilities[i] == expected_oracle)
        if outcome_row["exact_oracle_action_index"] != expected_oracle_index:
            raise ValueError("Stage A exact oracle action identity mismatch")
        reference = outcome_row["policies"]["supervised_first_order_reference"]
        scores = reference["scores"]
        if len(scores) != len(actions) or not all(_finite(value) for value in scores):
            raise ValueError("invalid Stage A supervised-reference score vector")
        maximum = max(scores)
        expected_reference = scores.index(maximum) if maximum > 0 else None
        if reference["index"] != expected_reference:
            raise ValueError("supervised-reference choice does not follow the positive canonical argmax")
        reference_utility = 0.0 if expected_reference is None else utilities[expected_reference]
        if not math.isclose(reference["utility"], reference_utility, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("supervised-reference utility differs from its exact action outcome")
    elif any(key in outcome_row for key in (
            "exact_oracle_utility", "exact_oracle_action_index",
            "exact_random_forced_expected_utility", "supervised_first_order_reference")):
        raise ValueError("Stage B must omit oracle, random expectation and supervised reference fields")
    return actions, utilities


def audit_raw(action_rows, outcome_rows, registry=None, *, allow_unpaired_scorer=False):
    audit_scorer_rows(action_rows)
    if registry is not None:
        validate_exact_objective_registry(registry)
    outcomes = {row["scorer_record_sha256"]: row for row in outcome_rows}
    if len(outcomes) != len(outcome_rows):
        raise ValueError("duplicate exact-objective evaluator query")
    scorer = {row["record_sha256"]: row for row in action_rows}
    if (len(scorer) != len(action_rows) or not set(outcomes) <= set(scorer) or
            (not allow_unpaired_scorer and set(scorer) != set(outcomes))):
        raise ValueError("evaluator raw records lack matching frozen scorer records")
    if registry is not None:
        for action_row in action_rows:
            provenance = registry["provenance"]
            if (action_row.get("registry_sha256") != registry.get("sha256") or
                    action_row.get("dataset_fingerprint") != registry["dataset_fingerprint"] or
                    action_row.get("registry_source_revision") != registry["source_revision"] or
                    action_row.get("costs", {}).get("action_chunk_size") != provenance["action_chunk_size"] or
                    action_row.get("view_state", {}).get("sha256") != provenance["view_state_sha256"] or
                    action_row.get("parameter_order_sha256") != provenance["parameter_order_sha256"]):
                raise ValueError("scorer provenance differs from the frozen exact-objective registry")
            if (action_row["stage"] == "stage-a" and
                    action_row.get("source_revision") != registry["source_revision"]):
                raise ValueError("Stage-A source revision differs from the frozen registry")
            query = action_row["query"]
            registered = False
            for cell in CELLS:
                expected = {row["timestep"]: row["sample_idx"]
                            for row in registry["cells"][cell]["selected"][action_row["stage"]]}
                registered |= expected.get(query["timestep"]) == query["sample_idx"]
            if not registered:
                raise ValueError("scored identity is outside the frozen exact-objective registry")
    random_groups = {}
    for row in outcome_rows:
        random_groups.setdefault((row["stage"], row["ood_cell"]), []).append(row)
    for members in random_groups.values():
        rng = random.Random(0)
        for draw, outcome in enumerate(sorted(members, key=lambda item: item["query"]["timestep"])):
            if outcome["rng_before_sha256"] != digest(rng.getstate()) or outcome["random_draw"] != draw:
                raise ValueError("random-control RNG sequence provenance mismatch")
            legal_count = scorer[outcome["scorer_record_sha256"]]["legal_swap_count"]
            if outcome["random_index"] != rng.randrange(legal_count):
                raise ValueError("random-forced index differs from the fixed-seed draw")
    rows = []
    for record_sha in sorted(set(scorer) & set(outcomes)):
        action_row, outcome_row = scorer[record_sha], outcomes[record_sha]
        actions, utilities = _verify_pair(action_row, outcome_row)
        if registry is not None:
            cell = outcome_row["ood_cell"]
            stage = action_row["stage"]
            expected = {row["timestep"]: row["sample_idx"]
                        for row in registry["cells"][cell]["selected"][stage]}
            query = action_row["query"]
            if expected.get(query["timestep"]) != query["sample_idx"]:
                raise ValueError("scored identity is outside the frozen exact-objective registry")
        correlations = {}
        if action_row["stage"] == "stage-a":
            for objective in action_row["selections"]:
                score_values = [entry["objective_deltas"][objective] for entry in action_row["actions"][1:]]
                utility_values = [utilities[index] for index in range(len(actions))]
                rho, reason = _rho(score_values, utility_values)
                correlations[objective] = {"rho": rho, "undefined_reason": reason}
        row = {"stage": action_row["stage"], "ood_cell": outcome_row["ood_cell"],
               "timestep": action_row["query"]["timestep"], "sample_idx": action_row["query"]["sample_idx"],
               "block": action_row["query"]["timestep"] // 64, "legal_swap_count": len(actions),
               "outcome": outcome_row, "scorer": action_row}
        if action_row["stage"] == "stage-a":
            row.update(correlations=correlations, action_utilities=utilities)
        rows.append(row)
    return rows


def audit_run_provenance(run_dir, action_rows, outcome_rows, *, expected_protocol_sha256=None):
    """Bind scorer/evaluator records to the committed run, stream and Ramen trace."""
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    state = json.loads((root / "qcgs-exact-state.json").read_text())
    status = json.loads((root / "qcgs-exact-status.json").read_text())
    scan_rows, scan_file = read_jsonl_detailed(root / "qcgs-exact-scan.jsonl")
    trace_rows, trace_file = read_jsonl_detailed(root / "trace.jsonl")
    if scan_file["errors"] or trace_file["errors"]:
        raise ValueError("malformed stream provenance JSONL: " +
                         "; ".join(scan_file["errors"] + trace_file["errors"]))
    reported_scanned = status.get("scanned_rows")
    scanned_reconciled = _reconcile_status_count(status, "scanned_rows", len(scan_rows), multiple_of=100)
    scan_prefix_rows = (len(scan_rows) // 100) * 100
    scan_tail_incomplete = (scan_file["truncated_tail"] or len(scan_rows) % 100 != 0)
    if len(scan_rows) % 100 and reported_scanned > scan_prefix_rows:
        raise ValueError("run status claims an incomplete scan batch as durable")
    trace_complete, trace_prefix_rows = _audit_trace_prefix(
        scan_rows, trace_rows, scan_tail_incomplete=scan_tail_incomplete,
        trace_tail_incomplete=trace_file["truncated_tail"])
    config = manifest["config"]
    args = manifest["args"]
    if manifest["git"]["dirty"] or manifest["git"]["commit"] != config.get("source_revision"):
        raise ValueError("exact-objective run source revision is dirty or differs from registry")
    if action_rows and digest(config) != manifest_config_sha(action_rows):
        raise ValueError("scorer config hash differs from the run manifest")
    stages = {row["stage"] for row in action_rows}
    status_stage = status.get("stage")
    if status_stage == "stage-b":
        frozen_objective = config.get("frozen_objective")
        if (frozen_objective not in OBJECTIVE_NAMES or
                any(row.get("stage_b_objective") != frozen_objective for row in action_rows)):
            raise ValueError("Stage-B scorer objective differs from its frozen run configuration")
    if (len(stages) > 1 or
            (status_stage in ("smoke", "scan") and stages and status_stage not in stages) or
            (status_stage not in ("smoke", "scan") and stages and status_stage not in stages)):
        raise ValueError("exact-objective run status stage differs from its scorer records")
    if status_stage in ("smoke", "scan") and outcome_rows:
        raise ValueError("unscored scan/smoke run must not contain evaluator outcomes")
    for scan in scan_rows:
        if (type(scan.get("timestep")) is not int or
                type(scan.get("sample_idx")) is not int or
                not 0 <= scan["sample_idx"] < 10000 or type(scan.get("eligible")) is not bool or
                any(not isinstance(scan.get(key), str) or len(scan[key]) != 64
                    for key in ("baseline_sha256", "aggregate_sha256", "cache_snapshot_sha256"))):
            raise ValueError("exact scan sidecar contains an invalid row or provenance hash")
    for trace in trace_rows:
        label = trace.get("known_label_or_minus_one")
        if (type(trace.get("prediction")) is not int or not 0 <= trace["prediction"] < 80 or
                type(trace.get("is_ood")) is not bool or type(label) is not int or
                (trace["is_ood"] and label != -1) or
                (not trace["is_ood"] and not 0 <= label < 80) or
                type(trace.get("ground_truth_domain")) is not int or trace["ground_truth_domain"] < 0):
            raise ValueError("ordinary stream trace contains invalid open-set identities")
    requested = len(config.get("query_schedule", []))
    scorer_complete = len(action_rows) == requested
    reported_scored = status.get("scored_queries")
    if status.get("requested_queries") != requested:
        raise ValueError("run status query schedule differs from the locked configuration")
    scored_reconciled = _reconcile_status_count(status, "scored_queries", len(action_rows))
    status_reconciled_fields = []
    if scanned_reconciled:
        status_reconciled_fields.append("scanned_rows")
    if scored_reconciled:
        status_reconciled_fields.append("scored_queries")
    if _reconcile_status_completion(status, "complete", scorer_complete):
        status_reconciled_fields.append("complete")
    is_unscored = status_stage in ("scan", "smoke")
    is_smoke = status_stage == "smoke"
    if is_unscored:
        evaluator_complete = None
    else:
        if len(outcome_rows) > len(action_rows):
            raise ValueError("durable evaluator records exceed frozen scorer records")
        evaluator_complete = (scorer_complete and len(outcome_rows) == requested)
        evaluator_reconciled = _reconcile_status_count(status, "evaluator_queries", len(outcome_rows))
        if evaluator_reconciled:
            status_reconciled_fields.append("evaluator_queries")
        if _reconcile_status_completion(status, "evaluator_complete", evaluator_complete):
            status_reconciled_fields.append("evaluator_complete")
    if state["parameter_order_sha256"] != digest(state["parameter_order"]):
        raise ValueError("adapted parameter ordering hash mismatch")
    protocol_path = Path(__file__).resolve().parents[2] / "docs/research/qcgs-exact-label-free-objective-audit.md"
    expected_protocol_sha256 = expected_protocol_sha256 or file_sha(protocol_path)
    if state["protocol_sha256"] != expected_protocol_sha256:
        raise ValueError("run protocol hash differs from the frozen audit")
    if config.get("protocol_sha256") != state["protocol_sha256"]:
        raise ValueError("run config and state protocol hashes disagree")
    if manifest["artifacts"]["status"] != "verified" or manifest["artifacts"]["dataset"].get("root_digest") != config.get("dataset_fingerprint"):
        raise ValueError("verified dataset fingerprint differs from scorer config")
    if args.get("dataset") != "CIFAR100C" or args.get("model") != "clip_vitbase16" or args.get("batch_size") != 100:
        raise ValueError("run dataset/model/batch differs from the frozen audit")
    if args.get("device") != "cuda" or args.get("stream_mode") != "block" or args.get("stream_block_size") != 64:
        raise ValueError("run device/stream condition differs from the frozen audit")
    split_path = Path(__file__).resolve().parents[2] / "cfg/research/open-set-cifar100-split-v1.json"
    if (not args.get("open_set") or args.get("known_class_split") != "open-set-cifar100-split-v1" or
            args.get("known_class_split_sha256") != file_sha(split_path) or
            args.get("open_set_per_domain_source_budget") != 400 or args.get("seed") != 0 or
            args.get("stream_seed") != 0 or args.get("ood_ratio") not in (0, 0.0, 0.5) or
            args.get("max_eval_samples") % 100):
        raise ValueError("run split/stream seed/OOD fraction/source budget differs from the frozen audit")
    if action_rows and state["view_state"] != action_rows[0]["view_state"]:
        raise ValueError("run fixed-view preprocessing state differs from scorer record")
    config_path = args.get("config_lock_path")
    config_path_sha = args.get("config_lock_sha256")
    if config_path and config_path_sha and Path(config_path).is_file() and file_sha(config_path) != config_path_sha:
        raise ValueError("run config lock file differs from the launch manifest")
    if action_rows:
        configured_schedule = {(row["timestep"], row["sample_idx"])
                               for row in config.get("query_schedule", [])}
        if len(configured_schedule) != len(config.get("query_schedule", [])):
            raise ValueError("run config contains duplicate scheduled query identity")
        if config.get("registry_sha256") != action_rows[0].get("registry_sha256"):
            raise ValueError("run registry hash differs from action record")
        if config.get("run_token") is None:
            raise ValueError("run config lacks an opaque unique run token")
        run_identity = hashlib.sha256(config["run_token"].encode("ascii")).hexdigest()
        for record in action_rows:
            query = record["query"]
            if ((query["timestep"], query["sample_idx"]) not in configured_schedule or
                    record["run_identity_sha256"] != run_identity or
                    record["source_revision"] != manifest["git"]["commit"] or
                    record.get("registry_source_revision") != config.get(
                        "registry_source_revision", config.get("source_revision")) or
                    record["config_sha256"] != digest(config) or
                    record["parameter_order_sha256"] != state["parameter_order_sha256"] or
                    record["protocol_sha256"] != state["protocol_sha256"]):
                raise ValueError("scorer record identity/config/source provenance mismatch")
            timestep = query["timestep"]
            if timestep >= len(scan_rows):
                if scan_tail_incomplete and timestep >= scan_prefix_rows:
                    continue
                raise ValueError("scored query lies beyond preserved scan/trace provenance")
            scan = scan_rows[timestep]
            trace = trace_rows[timestep] if timestep < trace_prefix_rows else None
            if (scan["timestep"] != timestep or scan["sample_idx"] != query["sample_idx"] or
                    scan["baseline_sha256"] != record["baseline_logits_sha256"] or
                    scan["aggregate_sha256"] != record["ramen_aggregate_sha256"] or
                    scan["cache_snapshot_sha256"] != record["cache_snapshot_sha256"] or
                    not scan["eligible"]):
                raise ValueError("Ramen baseline, cache snapshot or stream row mapping mismatch")
            if trace is None:
                continue
            if trace["prediction"] != record["baseline_pseudo_class"]:
                raise ValueError("ordinary Ramen pseudo-class differs from the persisted stream trace")
            batch_start = (timestep // 100) * 100
            if batch_start + 100 > len(trace_rows):
                raise ValueError("caption-bank provenance lacks a complete ordinary Ramen batch")
            expected_caption_classes = [int(item["prediction"])
                                        for item in trace_rows[batch_start:batch_start + 100]]
            if record["caption_pseudo_classes"] != expected_caption_classes:
                raise ValueError("pseudo-caption bank differs from each row's ordinary Ramen prediction")
        outcome_by_hash = {row["scorer_record_sha256"]: row for row in outcome_rows}
        for record in action_rows:
            timestep = record["query"]["timestep"]
            if timestep >= trace_prefix_rows:
                continue
            trace = trace_rows[timestep]
            known_label = trace["known_label_or_minus_one"]
            if (trace["is_ood"] is not False or type(known_label) is not int or
                    not 0 <= known_label < 80):
                raise ValueError("scored query is not a known-ID row in the persisted trace")
            outcome = outcome_by_hash.get(record["record_sha256"])
            if outcome is None:
                continue
            if (trace["known_label_or_minus_one"] != outcome["query"]["known_label"] or
                    trace["prediction"] != outcome["ramen"]["prediction"] or
                    trace["ground_truth_domain"] != outcome["query"]["domain"] or
                    outcome["ood_cell"] != format(float(args["ood_ratio"]), "g")):
                raise ValueError("evaluator outcome differs from run-only labels/domain/cell")
    if is_unscored and outcome_rows:
        raise ValueError("smoke run must not contain evaluator outcomes")
    return {"scorer_complete": scorer_complete, "evaluator_complete": evaluator_complete,
            "trace_complete": trace_complete, "trace_rows": len(trace_rows),
            "trace_prefix_rows": trace_prefix_rows,
            "interrupted_jsonl_artifacts": [name for name, receipt in (
                ("qcgs-exact-scan.jsonl", scan_file), ("trace.jsonl", trace_file),
            ) if receipt["truncated_tail"]],
            "status_reconciled": bool(status_reconciled_fields),
            "status_reconciled_fields": sorted(set(status_reconciled_fields)),
            "scan_rows": len(scan_rows),
            "scan_prefix_rows": scan_prefix_rows,
            "stage": status_stage,
            "ood_cell": format(float(args["ood_ratio"]), "g"),
            "registry_sha256": config.get("registry_sha256"),
            "dataset_fingerprint": config.get("dataset_fingerprint"),
            "protocol_sha256": state["protocol_sha256"],
            "source_revision": manifest["git"]["commit"]}


def _reconcile_status_count(status, field, durable_count, *, multiple_of=None):
    value = status.get(field)
    if (type(value) is not int or value < 0 or value > durable_count or
            (multiple_of is not None and value % multiple_of)):
        raise ValueError(f"run status overstates or misaligns durable {field}")
    return value < durable_count


def _reconcile_status_completion(status, field, durable_complete):
    value = status.get(field)
    if type(value) is not bool or (value and not durable_complete):
        raise ValueError(f"run status claims {field} beyond durable records")
    return value != durable_complete


def _audit_trace_prefix(scan_rows, trace_rows, *, scan_tail_incomplete=False,
                        trace_tail_incomplete=False):
    """Validate the ordinary trace and return its fully flushed batch prefix."""
    if len(trace_rows) > len(scan_rows):
        raise ValueError("ordinary stream trace exceeds the exact scan sidecar")
    if len(scan_rows) % 100 and not scan_tail_incomplete:
        raise ValueError("exact scan sidecar does not contain complete 100-row batches")
    scan_prefix_rows = (len(scan_rows) // 100) * 100
    if len(scan_rows) % 100 and len(trace_rows) != scan_prefix_rows:
        raise ValueError("trace does not end at the verified scan-batch boundary")
    if len(scan_rows) % 100 and not scan_tail_incomplete:
        raise ValueError("partial scan batch lacks an interrupted-write receipt")
    missing_tail_rows = scan_prefix_rows - len(trace_rows)
    if missing_tail_rows < 0 or missing_tail_rows > 100:
        raise ValueError("scan/trace mismatch exceeds one interrupted final batch")
    for timestep, scan in enumerate(scan_rows):
        if scan["timestep"] != timestep:
            raise ValueError("scan sidecar timestep sequence is not contiguous")
    for timestep, trace in enumerate(trace_rows):
        if (trace["timestep"] != timestep or
                trace["sample_idx"] != scan_rows[timestep]["sample_idx"]):
            raise ValueError("scan sidecar and ordinary stream trace identities differ")
    trace_complete = (len(scan_rows) == len(trace_rows) and len(scan_rows) % 100 == 0 and
                      not scan_tail_incomplete and not trace_tail_incomplete)
    return trace_complete, min(scan_prefix_rows, (len(trace_rows) // 100) * 100)


def manifest_config_sha(action_rows):
    hashes = {row["config_sha256"] for row in action_rows}
    if len(hashes) != 1:
        raise ValueError("one run contains multiple scorer config hashes")
    return next(iter(hashes))


def _mean(values):
    return float(np.mean(values)) if values else None


def _selected_metric(row, objective, field):
    choice = row["outcome"]["policies"][objective]
    if choice["index"] is None:
        return row["outcome"]["ramen"][field]
    return next(action[field] for action in row["outcome"]["actions"]
                if action["index"] == choice["index"])


def _summary_cell(rows, stage):
    by_objective = {}
    objectives = OBJECTIVE_NAMES if stage == "stage-a" else sorted({
        row["scorer"]["stage_b_objective"] for row in rows})
    for objective in objectives:
        subset = ([row for row in rows if objective in row["correlations"]] if stage == "stage-a" else
                  [row for row in rows if row["scorer"]["stage_b_objective"] == objective])
        if stage == "stage-a":
            rhos = [row["correlations"][objective]["rho"] for row in subset
                    if row["correlations"][objective]["rho"] is not None]
            selected = [row["outcome"]["policies"][objective]["utility"] for row in subset]
            matched = [row["outcome"]["policies"][objective + "_random_matched"]["utility"] for row in subset]
            positive_delta_fractions = []
            pooled_positive_actions = pooled_helpful_actions = 0
            for row in subset:
                deltas = [action["objective_deltas"][objective]
                          for action in row["scorer"]["actions"][1:]]
                positive = [index for index, delta in enumerate(deltas) if delta > 0]
                if positive:
                    positive_delta_fractions.append(sum(row["action_utilities"][i] > 0 for i in positive) / len(positive))
                    pooled_positive_actions += len(positive)
                    pooled_helpful_actions += sum(row["action_utilities"][i] > 0 for i in positive)
            selected_helpful = [value for value in selected if value > 0]
            selected_harmful = [value for value in selected if value < 0]
            rho_boot_rows = [row for row in subset if row["correlations"][objective]["rho"] is not None]
            rho_boot_values = [row["correlations"][objective]["rho"] for row in rho_boot_rows]
            selected_boot = _block_bootstrap(subset, selected)
            random_expected = [row["outcome"]["exact_random_forced_expected_utility"] for row in subset]
            random_boot = _block_bootstrap(subset, random_expected)
            oracle_values = [row["outcome"]["exact_oracle_utility"] for row in subset]
            oracle_boot = _block_bootstrap(subset, oracle_values)
            by_objective[objective] = {
                "defined_rho": len(rhos), "total_rho": len(subset), "mean_rho": _mean(rhos),
                "median_rho": float(np.median(rhos)) if rhos else None,
                "positive_rho_fraction": sum(value > 0 for value in rhos) / len(rhos) if rhos else None,
                "selected_utility_mean": _mean(selected), "selected_utility_median": float(np.median(selected)) if selected else None,
                "selected_minus_exact_random_forced_mean": _mean([
                    row["outcome"]["policies"][objective]["utility"] - row["outcome"]["exact_random_forced_expected_utility"]
                    for row in subset]),
                "exact_random_forced_expected_mean": _mean([row["outcome"]["exact_random_forced_expected_utility"] for row in subset]),
                "random_forced_draw_mean": _mean([row["outcome"]["policies"]["random_forced"]["utility"] for row in subset]),
                "random_matched_mean": _mean(matched),
                "exact_oracle_mean": _mean([row["outcome"]["exact_oracle_utility"] for row in subset]),
                "replacement_count": sum(row["outcome"]["policies"][objective]["index"] is not None for row in subset),
                "p_utility_positive_given_delta_positive": {
                    "query_balanced": _mean(positive_delta_fractions),
                    "query_balanced_defined_queries": len(positive_delta_fractions),
                    "pooled": (pooled_helpful_actions / pooled_positive_actions if pooled_positive_actions else None),
                    "pooled_positive_action_count": pooled_positive_actions,
                },
                "selected_helpful_rate": sum(value > 0 for value in selected) / len(selected) if selected else None,
                "selected_harmful_rate": sum(value < 0 for value in selected) / len(selected) if selected else None,
                "mean_helpful_magnitude": _mean(selected_helpful),
                "mean_harmful_magnitude": _mean([-value for value in selected_harmful]),
                "mean_baseline_entropy": _mean([row["outcome"]["ramen"]["entropy"] for row in subset]),
                "mean_selected_entropy": _mean([_selected_metric(row, objective, "entropy") for row in subset]),
                "mean_baseline_margin": _mean([row["outcome"]["ramen"]["margin"] for row in subset]),
                "mean_selected_margin": _mean([_selected_metric(row, objective, "margin") for row in subset]),
                "abstention_count": sum(row["outcome"]["policies"][objective]["index"] is None
                                         for row in subset),
                "selected_utility_block_bootstrap": selected_boot,
                "random_expected_block_bootstrap": random_boot,
                "oracle_block_bootstrap": oracle_boot,
                "rho_block_bootstrap": _block_bootstrap(rho_boot_rows, rho_boot_values),
                "costs": {"mean_candidate_action_ms_batch_shared": _mean([
                    row["scorer"]["costs"]["candidate_action_ms"] for row in subset]),
                    "candidate_actions": sum(row["legal_swap_count"] for row in subset),
                    "full_batch_action_forwards": sum(row["legal_swap_count"] for row in subset),
                    "mean_evaluator_ms": _mean([row["outcome"].get("costs", {}).get("evaluator_ms")
                                                 for row in subset
                                                 if row["outcome"].get("costs", {}).get("evaluator_ms") is not None]),
                    "evaluator_action_count": sum(len(row["outcome"]["evaluated_action_indices"])
                                                   for row in subset),
                    "peak_evaluator_device_memory_bytes_max": max(
                        [row["outcome"].get("costs", {}).get("peak_device_memory_bytes") for row in subset
                         if row["outcome"].get("costs", {}).get("peak_device_memory_bytes") is not None],
                        default=None)},
                "per_query_rho": [row["correlations"][objective] for row in subset],
            }
        else:
            selected = [row["outcome"]["policies"][objective]["utility"] for row in subset]
            matched = [row["outcome"]["policies"][objective + "_random_matched"]["utility"] for row in subset]
            forced = [row["outcome"]["policies"]["random_forced"]["utility"] for row in subset]
            helpful = [value for value in selected if value > 0]
            harmful = [value for value in selected if value < 0]
            by_objective[objective] = {"selected_utility_mean": _mean(selected),
                "selected_utility_median": float(np.median(selected)) if selected else None,
                "selected_minus_random_matched_mean": _mean([a - b for a, b in zip(selected, matched)]),
                "selected_minus_random_forced_mean": _mean([a - b for a, b in zip(selected, forced)]),
                "random_matched_mean": _mean(matched), "random_forced_draw_mean": _mean(forced),
                "replacement_count": sum(row["outcome"]["policies"][objective]["index"] is not None for row in subset),
                "abstention_count": sum(row["outcome"]["policies"][objective]["index"] is None for row in subset),
                "mean_baseline_entropy": _mean([row["outcome"]["ramen"]["entropy"] for row in subset]),
                "mean_selected_entropy": _mean([_selected_metric(row, objective, "entropy") for row in subset]),
                "mean_baseline_margin": _mean([row["outcome"]["ramen"]["margin"] for row in subset]),
                "mean_selected_margin": _mean([_selected_metric(row, objective, "margin") for row in subset]),
                "selected_helpful_rate": sum(value > 0 for value in selected) / len(selected) if selected else None,
                "selected_harmful_rate": sum(value < 0 for value in selected) / len(selected) if selected else None,
                "mean_helpful_magnitude": _mean(helpful),
                "mean_harmful_magnitude": _mean([-value for value in harmful]),
                "costs": {"mean_candidate_action_ms_batch_shared": _mean([
                    row["scorer"]["costs"]["candidate_action_ms"] for row in subset]),
                    "candidate_actions_scored": sum(row["legal_swap_count"] for row in subset),
                    "evaluator_actions": sum(len(row["outcome"]["evaluated_action_indices"]) for row in subset),
                    "mean_evaluator_ms": _mean([row["outcome"].get("costs", {}).get("evaluator_ms")
                                                 for row in subset
                                                 if row["outcome"].get("costs", {}).get("evaluator_ms") is not None]),
                    "peak_evaluator_device_memory_bytes_max": max(
                        [row["outcome"].get("costs", {}).get("peak_device_memory_bytes") for row in subset
                         if row["outcome"].get("costs", {}).get("peak_device_memory_bytes") is not None],
                        default=None)}}
    blocks = sorted({row["block"] for row in rows})
    transitions_by_objective = {}
    for objective in objectives:
        transitions = {f"{a}->{b}": 0 for a in (False, True) for b in (False, True)}
        for row in rows:
            if objective not in row["outcome"]["policies"]:
                continue
            baseline = row["outcome"]["ramen"]["correct"]
            choice = row["outcome"]["policies"][objective]
            index = choice["index"]
            if index is None:
                corrected = baseline
            else:
                evaluated = {entry["index"]: entry for entry in row["outcome"]["actions"]}
                corrected = evaluated[index]["correct"]
            transitions[f"{baseline}->{corrected}"] += 1
        transitions_by_objective[objective] = {
            "counts": transitions,
            "net_correction": transitions["False->True"] - transitions["True->False"],
        }
    stage_b_transition = transitions_by_objective.get(objectives[0]) if stage == "stage-b" and objectives else None
    peak_values = [row["scorer"]["costs"].get("peak_device_memory_bytes") for row in rows
                   if row["scorer"]["costs"].get("peak_device_memory_bytes") is not None]
    return {"queries": len(rows), "distinct_blocks": len(blocks), "objectives": by_objective,
            "transitions": stage_b_transition if stage == "stage-b" else transitions_by_objective,
            "net_correction_by_objective": ({name: value["net_correction"]
                                              for name, value in transitions_by_objective.items()}
                                             if stage == "stage-a" else None),
            "net_correction": stage_b_transition["net_correction"] if stage_b_transition else None,
            "peak_device_memory_bytes_max": max(peak_values) if peak_values else None}


def _block_bootstrap(rows, values_by_row, *, draws=10000, seed=917, include_values=False):
    block_ids = sorted({row["block"] for row in rows})
    if not block_ids:
        return {"block_count": 0, "interval_95": None, "raw_one_sided_p": None}
    groups = {block: [values_by_row[i] for i, row in enumerate(rows) if row["block"] == block]
              for block in block_ids}
    block_sums = np.asarray([sum(group) for group in groups.values()], dtype=float)
    block_counts = np.asarray([len(group) for group in groups.values()], dtype=float)
    point = float(block_sums.sum() / block_counts.sum())
    if len(block_ids) < 4:
        return {"block_count": len(block_ids), "mean": point, "interval_95": None,
                "raw_one_sided_p": None, "null_reason": "fewer_than_four_blocks"}
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(block_ids), size=(draws, len(block_ids)))
    sums = block_sums[selected].sum(1)
    counts = block_counts[selected].sum(1)
    boot = sums / counts
    leave_one_out = []
    if len(block_ids) > 1:
        for index in range(len(block_ids)):
            keep = np.arange(len(block_ids)) != index
            leave_one_out.append(float(block_sums[keep].sum() / block_counts[keep].sum()))
    centered_block_sums = block_sums - point * block_counts
    null_sums = centered_block_sums[selected].sum(1)
    null_boot = null_sums / counts
    result = {"block_count": len(block_ids), "mean": point,
              "interval_95": np.percentile(boot, [2.5, 97.5]).tolist(),
              "raw_one_sided_p": float((np.count_nonzero(null_boot >= point) + 1) / (draws + 1)),
              "leave_one_block_out_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None,
              "bootstrap_draws": draws, "seed": seed}
    if include_values:
        result["bootstrap_values"] = boot.tolist()
    return result


def stage_a_gate(raw_rows, *, integrity_errors=()):
    if integrity_errors:
        return {"decision": "INVALID", "integrity_errors": list(integrity_errors)}
    cells = {cell: _summary_cell([row for row in raw_rows if row["stage"] == "stage-a" and row["ood_cell"] == cell], "stage-a")
             for cell in CELLS}
    if any(cells[cell]["queries"] != 16 for cell in CELLS):
        return {"decision": "INCONCLUSIVE", "complete": False, "cells": cells}
    if any(cells[cell]["distinct_blocks"] < 4 for cell in CELLS):
        return {"decision": "INCONCLUSIVE", "complete": True,
                "reason": "fewer_than_four_scored_blocks", "cells": cells}
    passing = []
    for objective in OBJECTIVE_NAMES:
        passed = True
        for cell in CELLS:
            stats = cells[cell]["objectives"][objective]
            passed &= (stats["defined_rho"] >= 8 and stats["mean_rho"] is not None and stats["mean_rho"] > 0 and
                       stats["selected_utility_mean"] > 0 and
                       stats["selected_utility_mean"] > stats["exact_random_forced_expected_mean"] and
                       stats["replacement_count"] >= 3 and stats["exact_oracle_mean"] > 0)
        if passed:
            passing.append(objective)
    winner = None
    pooled_rho_means = {}
    if passing:
        for objective in passing:
            values = [row["correlations"][objective]["rho"] for row in raw_rows
                      if row["stage"] == "stage-a" and row["correlations"][objective]["rho"] is not None]
            pooled_rho_means[objective] = _mean(values)
        winner = max(passing, key=lambda name: (pooled_rho_means[name], -OBJECTIVE_NAMES.index(name)))
    return {"decision": "GO_CONFIRM" if winner else "STOP", "complete": True,
            "winner": winner, "passing_objectives": passing, "cells": cells,
            "pooled_defined_query_mean_rho": pooled_rho_means,
            "stage_b_reserved_ids_remain_unscored": True}


def _holm_adjusted_pvalues(p_values):
    if any(value is None for value in p_values):
        return [None] * len(p_values)
    ordered = sorted((value, index) for index, value in enumerate(p_values))
    adjusted = [None] * len(p_values)
    running = 0.0
    for rank, (value, index) in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * value)
        adjusted[index] = min(1.0, running)
    return adjusted


def stage_b_gate(raw_rows, winner, *, integrity_errors=()):
    if integrity_errors:
        return {"decision": "INVALID", "integrity_errors": list(integrity_errors)}
    if winner not in OBJECTIVE_NAMES:
        return {"decision": "INVALID", "integrity_errors": ["missing frozen Stage-A winner"]}
    cells = {cell: _summary_cell([row for row in raw_rows if row["stage"] == "stage-b" and row["ood_cell"] == cell], "stage-b")
             for cell in CELLS}
    if any(cells[cell]["queries"] != 128 for cell in CELLS):
        return {"decision": "INCONCLUSIVE", "complete": False, "cells": cells}
    metrics = []
    for cell in CELLS:
        rows = [row for row in raw_rows if row["stage"] == "stage-b" and row["ood_cell"] == cell]
        selected = [row["outcome"]["policies"][winner]["utility"] for row in rows]
        matched = [row["outcome"]["policies"][winner + "_random_matched"]["utility"] for row in rows]
        metrics.extend([(cell, "selected", selected), (cell, "paired_random_matched",
                       [a - b for a, b in zip(selected, matched)])])
        positive = sorted((row for row in rows if row["outcome"]["policies"][winner]["utility"] > 0),
                          key=lambda row: (-row["outcome"]["policies"][winner]["utility"], row["sample_idx"]))
        removed = {row["sample_idx"] for row in positive[:2]}
        trimmed = [row["outcome"]["policies"][winner]["utility"] for row in rows
                   if row["sample_idx"] not in removed]
        cells[cell]["objectives"][winner]["drop_two_largest_mean"] = _mean(trimmed)
        cells[cell]["objectives"][winner]["removed_positive_contributor_ids"] = [row["sample_idx"] for row in positive[:2]]
        cells[cell]["objectives"][winner]["trim_tie_rule"] = ANALYSIS["stage_b_tie_rule"]
    inferential = []
    for cell, name, values in metrics:
        rows = [row for row in raw_rows if row["stage"] == "stage-b" and row["ood_cell"] == cell]
        inferential.append({"cell": cell, "metric": name,
                            **_block_bootstrap(rows, values, include_values=True)})
    holm = _holm_adjusted_pvalues([row["raw_one_sided_p"] for row in inferential])
    holm_alpha = [None] * len(inferential)
    if all(item is not None for item in holm):
        ordered = sorted((row["raw_one_sided_p"], i) for i, row in enumerate(inferential))
        for rank, (_, index) in enumerate(ordered):
            holm_alpha[index] = .05 / (len(ordered) - rank)
    for i, item in enumerate(inferential):
        item["holm_adjusted_one_sided_p"] = holm[i]
        alpha = holm_alpha[i]
        item.pop("bootstrap_values", None)
        item["holm_stepdown_alpha"] = alpha
        item["holm_reject_at_fwer_0_05"] = bool(holm[i] is not None and holm[i] <= .05)
    enough_blocks = all(item["block_count"] >= 4 for item in inferential)
    primary_pass = enough_blocks and all(item["holm_reject_at_fwer_0_05"] for item in inferential)
    gates = []
    for cell in CELLS:
        stats = cells[cell]["objectives"][winner]
        gates.append(stats["selected_utility_mean"] > 0 and stats["selected_minus_random_matched_mean"] > 0 and
                     stats["selected_utility_median"] > 0 and
                     stats["drop_two_largest_mean"] > 0 and stats["replacement_count"] >= 3)
    all_queries = all(cells[cell]["queries"] == 128 for cell in CELLS)
    net = sum(cells[cell]["net_correction"] for cell in CELLS)
    decision = "INCONCLUSIVE" if not enough_blocks else "GO_ESTIMATOR" if primary_pass and all(gates) and all_queries and net >= 0 else "STOP"
    return {"decision": decision, "complete": True, "winner": winner,
            "cells": cells, "primary_inference": inferential,
            "net_correction_both_cells": net,
            "selected_vs_random_forced_is_reported_separately": True}


def analyze_raw(action_rows, outcome_rows, registry=None, *, integrity_errors=(),
                allow_unpaired_scorer=False):
    errors = list(integrity_errors)
    try:
        rows = audit_raw(action_rows, outcome_rows, registry,
                         allow_unpaired_scorer=allow_unpaired_scorer)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(str(exc))
        return {"schema_version": SCHEMA_VERSION, "decision": "INVALID", "analysis": ANALYSIS,
                "integrity_errors": errors, "stage_a": {}, "stage_b": None}
    if errors:
        return {"schema_version": SCHEMA_VERSION, "decision": "INVALID", "analysis": ANALYSIS,
                "integrity_errors": errors, "stage_a": {}, "stage_b": None}
    stage_a = stage_a_gate(rows)
    stage_b_rows = [row for row in rows if row["stage"] == "stage-b"]
    if stage_b_rows and stage_a["decision"] != "GO_CONFIRM":
        return {"schema_version": SCHEMA_VERSION, "decision": "INVALID",
                "integrity_errors": ["Stage B exists without a passing Stage-A gate"],
                "stage_a": stage_a, "stage_b": {}}
    stage_b_winner = stage_a.get("winner")
    if stage_b_rows and any(row["scorer"].get("stage_b_objective") != stage_b_winner
                            for row in stage_b_rows):
        return {"schema_version": SCHEMA_VERSION, "decision": "INVALID",
                "integrity_errors": ["Stage-B scorer objective differs from the recomputed Stage-A winner"],
                "stage_a": stage_a, "stage_b": {}}
    stage_b = stage_b_gate(stage_b_rows, stage_a.get("winner")) if stage_b_rows else None
    decision = stage_b["decision"] if stage_b is not None else stage_a["decision"]
    overlap = {}
    for stage in ("stage-a", "stage-b"):
        left = {row["sample_idx"]: row for row in rows if row["stage"] == stage and row["ood_cell"] == "0"}
        right = {row["sample_idx"]: row for row in rows if row["stage"] == stage and row["ood_cell"] == "0.5"}
        common = sorted(set(left) & set(right))
        comparisons = {}
        objective = stage_a.get("winner") if stage == "stage-b" else None
        names = [objective] if objective else (OBJECTIVE_NAMES if stage == "stage-a" else [])
        for name in names:
            differences = [left[index]["outcome"]["policies"][name]["utility"] -
                           right[index]["outcome"]["policies"][name]["utility"] for index in common]
            comparisons[name] = {"paired_base_images": len(common), "mean_ood0_minus_ood05": _mean(differences),
                                 "positive": sum(value > 0 for value in differences),
                                 "negative": sum(value < 0 for value in differences),
                                 "ties": sum(value == 0 for value in differences)}
        overlap[stage] = {"scored_base_image_overlap": len(common), "paired_utility_direction": comparisons}
    return {"schema_version": SCHEMA_VERSION, "decision": decision, "analysis": ANALYSIS,
            "stage_a": stage_a, "stage_b": stage_b,
            "cross_ood_base_image_overlap_and_direction": overlap,
            "stream_exposure_limit": "Scored query disjointness does not certify absence from historical/current cache streams."}


def analyze_directory(evidence_dir, registry=None):
    root = Path(evidence_dir)
    actions = read_jsonl(root / "qcgs-exact-actions.jsonl")
    outcomes = read_jsonl(root / "qcgs-exact-outcomes.jsonl")
    result = analyze_raw(actions, outcomes, registry)
    write_json(root / "qcgs-exact-analysis.json", result)
    return result
