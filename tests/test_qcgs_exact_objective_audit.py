"""Frozen exact-objective formulas, scorer boundary, registry and raw gates."""
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.qcgs_exact_objective_audit import (
    CURRENT_PROTOCOL, LEGACY_PROTOCOL_SHA256, LEGACY_SMOKE_PROTOCOL,
    ExactObjectiveEvaluator, _audit_trace_prefix, _block_bootstrap, _holm_adjusted_pvalues, _rho,
    _reconcile_status_completion, _reconcile_status_count, analyze_raw,
    audit_raw, audit_scorer_rows, stage_a_gate, stage_b_gate, write_evaluator_outputs,
    read_jsonl_detailed, write_scorer_outputs,
)
from methods.OracleSupportUtilityProbe import aggregate_pools, aggregate_swaps, legal_swaps
from methods.QCGSExactObjectiveAudit import IdentityCache, QCGSExactObjectiveAudit
from methods.Ramen import PriorityCache, Ramen
from methods.losses import softmax_entropy
from methods.qcgs_exact_objectives import (
    OBJECTIVE_NAMES, cliptta_image_to_text_entropy, memo_marginal_entropy,
    objective_deltas, objective_values, select_objective_action, single_view_entropy,
)
from models.ModelForBySampleTTA import BySampleLayerNorm, ModelForBySampleTTA
from models.optimizer import SignSGD
import runtime.qcgs_exact_objective_audit as exact_runtime
from runtime.qcgs_exact_objective_audit import build_registry, build_smoke_registry, make_job
from runtime.query_gradient_registry import (
    EXACT_OBJECTIVE_NAMESPACE, assigned_stage, build_exact_objective_registry,
    digest, exact_objective_query_schedule, validate_exact_objective_registry,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PATH = ROOT / "cfg/research/qcgs-exact-objective-audit/historical-exclusions.json"


def registry_provenance(**overrides):
    current_protocol_sha = hashlib.sha256(
        (ROOT / "docs/research/qcgs-exact-label-free-objective-audit.md").read_bytes()).hexdigest()
    value = {
        "active_protocol": CURRENT_PROTOCOL,
        "active_protocol_sha256": current_protocol_sha,
        "peak_memory_policy": "report_only",
        "scan_source_revision": "1" * 40,
        "scan_protocol_sha256": current_protocol_sha,
        "smoke_source_revision": "1" * 40,
        "smoke_protocol_sha256": current_protocol_sha,
        "smoke_peak_device_memory_bytes": {"0": 1024, "0.5": 2048},
        "smoke_peak_device_memory_by_query_bytes": {"0": [1024, 1024], "0.5": [1024, 2048]},
        "action_chunk_size": 1,
        "view_state_sha256": "d" * 64,
        "parameter_order_sha256": "e" * 64,
    }
    value.update(overrides)
    return value


class ObjectiveArithmeticTests(unittest.TestCase):
    def test_three_frozen_objectives_and_improvement_sign(self):
        probs = torch.tensor([0.75, 0.25])
        logits = probs.log()
        expected = -float((probs * probs.log()).sum())
        self.assertAlmostEqual(float(single_view_entropy(logits)), expected, places=6)

        view_logits = torch.stack([logits] * 4 + [logits.flip(0)] * 4)
        self.assertAlmostEqual(float(memo_marginal_entropy(view_logits)), math.log(2), places=6)

        query = torch.tensor([1.0, 0.0])
        bank = torch.zeros(100, 2)
        bank[0, 0] = 1.0
        bank[1:, 1] = 1.0
        actual = float(cliptta_image_to_text_entropy(query, bank, 1.0))
        first = math.e / (math.e + 99)
        other = 1 / (math.e + 99)
        expected = -(first * math.log(first) + 99 * other * math.log(other))
        self.assertAlmostEqual(actual, expected, places=6)

        baseline = {name: 2.0 for name in OBJECTIVE_NAMES}
        after = {name: 1.5 for name in OBJECTIVE_NAMES}
        self.assertEqual(objective_deltas(baseline, after), {name: 0.5 for name in OBJECTIVE_NAMES})
        self.assertEqual(objective_deltas(after, baseline), {name: -0.5 for name in OBJECTIVE_NAMES})

    def test_no_swap_constant_vectors_and_canonical_positive_ties(self):
        self.assertEqual(select_objective_action([]), {"index": None, "score": 0.0, "reason": "no_legal_swaps"})
        self.assertEqual(select_objective_action([0.0, 0.0]), {
            "index": None, "score": 0.0, "reason": "constant_objective_vector"})
        self.assertEqual(select_objective_action([0.4, 0.4, 0.1]), {
            "index": 0, "score": 0.4, "reason": "positive_canonical_argmax"})
        self.assertEqual(select_objective_action([-0.2, 0.0]), {
            "index": None, "score": 0.0, "reason": "zero_maximum_no_swap_wins"})
        self.assertEqual(select_objective_action([-0.2, -0.1]), {
            "index": None, "score": 0.0, "reason": "negative_maximum"})

    def test_average_ties_and_valid_undefined_rank_reasons(self):
        rho, reason = _rho([0.5, 0.5, 0.1], [1.0, 1.0, 2.0])
        self.assertIsNone(reason)
        self.assertAlmostEqual(rho, -1.0)
        self.assertEqual(_rho([0.0, 0.0], [1.0, 2.0]), (None, "constant_objective_vector"))
        self.assertEqual(_rho([0.0, 1.0], [1.0, 1.0]), (None, "constant_utility_vector"))

    def test_fixed_raw_view_corners_and_flip_order(self):
        pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        from methods.query_gradient_multiview import raw_views
        views = raw_views(pixels)
        self.assertEqual(len(views), 8)
        np.testing.assert_array_equal(np.asarray(views[0]), pixels[:30, :30])
        np.testing.assert_array_equal(np.asarray(views[1]), pixels[:30, :30][:, ::-1])
        np.testing.assert_array_equal(np.asarray(views[2]), pixels[:30, 2:32])
        np.testing.assert_array_equal(np.asarray(views[3]), pixels[:30, 2:32][:, ::-1])
        np.testing.assert_array_equal(np.asarray(views[4]), pixels[2:32, :30])
        np.testing.assert_array_equal(np.asarray(views[6]), pixels[2:32, 2:32])


class TinyExactCLIP(ModelForBySampleTTA):
    """Independent-row CLIP-shaped CPU fixture exercising the real scorer path."""

    def __init__(self):
        self.model_type = "vit"
        self.model = BySampleLayerNorm(nn.LayerNorm(2), max_batch_size=100, batch_first=True)
        self.model.eval()
        self.text_feature = torch.zeros(80, 2)
        self.text_feature[0] = torch.tensor([1.0, -1.0]) / math.sqrt(2)
        self.text_feature[1] = torch.tensor([1.0, 1.0]) / math.sqrt(2)
        self.text_feature[2:] = torch.tensor([-1.0, 0.0])
        self.optimizer = SignSGD(self.model.parameters(), lr=0.01)
        self.feat_dim = 2
        self.grad_dim = 4

    def featurize(self, x):
        return F.normalize(self.model(x.float()), dim=1)

    def classify(self, features):
        return 100.0 * features @ self.text_feature.T


def make_batch():
    pixels = np.zeros((100, 32, 32, 3), dtype=np.uint8)
    for row in range(100):
        pixels[row, :, :, 0] = (30 + row) % 255
        pixels[row, :, :, 1] = 3
        pixels[row, :, :, 2] = 7

    def preprocess(image):
        value = np.asarray(image, dtype=np.uint8)
        return torch.tensor([1.0 + float(value[:, :, 0].mean()) / 2550,
                             float(value[:, :, 1].mean()) / 2550], dtype=torch.float32)

    inputs = torch.stack([preprocess(Image.fromarray(pixels[row])) for row in range(100)])
    return inputs, pixels, preprocess


def exact_cfg(mode="stage-a", schedule=None):
    return {"topk": 5, "candidate_m": 10, "max_capacity": 750, "beta": 5.0,
            "optimizer": "signsgd", "lr": 0.01, "qcgs_mode": mode,
            "objective_stage": "all", "ood_cell": "0", "dataset_fingerprint": "a" * 64,
            "source_revision": "unit-test", "registry_sha256": "b" * 64,
            "run_token": "0" * 32,
            "query_schedule": ([{"timestep": i, "sample_idx": i} for i in range(16)]
                               if schedule is None and mode == "stage-a" else schedule or []),
            "max_batch_size": 100, "random_seed": 0, "action_chunk_size": 1,
            "peak_memory_policy": "report_only"}


def make_exact_probe(mode="stage-a", schedule=None):
    cfg = exact_cfg(mode, schedule)
    if mode == "stage-b":
        cfg["frozen_objective"] = OBJECTIVE_NAMES[0]
    probe = object.__new__(QCGSExactObjectiveAudit)
    nn.Module.__init__(probe)
    probe.model = TinyExactCLIP()
    probe.cfg = cfg
    probe.beta = 5.0
    probe.num_classes = 80
    probe.device = torch.device("cpu")
    probe.dtype = torch.half
    probe.feat_dim = 2
    probe.grad_dim = 4
    probe.cache = [IdentityCache(750, 2, 4, "cpu", torch.half) for _ in range(80)]
    probe.counter = 0
    probe.last_diagnostics = {}
    probe.loss_fn = lambda logits: softmax_entropy(logits, reduction="sum")
    probe.clip_logit_scale = 100.0
    probe._initialize_audit(None)
    return probe


def make_ramen_baseline():
    method = object.__new__(Ramen)
    nn.Module.__init__(method)
    method.model = TinyExactCLIP()
    method.cfg = exact_cfg("scan")
    method.beta = 5.0
    method.num_classes = 80
    method.device = torch.device("cpu")
    method.dtype = torch.half
    method.feat_dim = 2
    method.grad_dim = 4
    method.cache = [PriorityCache(750, 2, 4, "cpu", torch.half) for _ in range(80)]
    method.counter = 0
    method.last_diagnostics = {}
    method.loss_fn = lambda logits: softmax_entropy(logits, reduction="sum")
    return method


class ScorerBoundaryAndParityTests(unittest.TestCase):
    def test_audit_constructor_uses_lazy_cache_without_eager_priority_allocations(self):
        cfg = exact_cfg("scan")
        cfg["clip_logit_scale"] = 100.0
        args = SimpleNamespace(config=cfg)
        datasets = SimpleNamespace(num_classes=80, classes=list(range(80)))
        model = nn.Linear(2, 2)

        with patch("methods.Ramen.CLIPModelForBySampleTTA", return_value=TinyExactCLIP()), \
             patch("methods.Ramen.PriorityCache", side_effect=AssertionError("eager cache allocated")):
            audit = QCGSExactObjectiveAudit(model, datasets, args)

        self.assertEqual(len(audit.cache), 80)
        self.assertTrue(all(isinstance(cache, IdentityCache) for cache in audit.cache))
        self.assertTrue(all(cache.keys.shape[0] == 0 for cache in audit.cache))

    def test_identity_cache_grows_lazily_and_matches_priority_cache(self):
        capacity = 750
        device = torch.device("cpu")
        dtype = torch.float16
        cache = IdentityCache(capacity, 3, 5, device, dtype)
        reference = PriorityCache(capacity, 3, 5, device, dtype)
        reference_ids = [None] * capacity

        for index in range(capacity + 1):
            key = torch.tensor([index, index + 1, index + 2], dtype=torch.float32)
            value = torch.arange(5, dtype=torch.float32) + index
            entropy = torch.tensor(index / 1000, dtype=torch.float32)
            priority = torch.tensor(index % 997, dtype=torch.float32)
            identity = {"timestep": index, "sample_idx": index}
            target = reference.size if reference.size < capacity else int(reference.priorities.argmin())
            admitted = (reference.size < capacity or bool(priority > reference.priorities[target]))
            reference.add(key, value, entropy.reshape(1), priority.reshape(1))
            if admitted:
                reference_ids[target] = identity
            cache.add_identity(key, value, entropy, priority, identity)
            if index == 19:
                self.assertLess(cache.keys.shape[0], capacity)

        self.assertEqual(cache.size, reference.size)
        self.assertEqual(cache.identities, reference_ids)
        for name in ("keys", "values", "entropies", "priorities"):
            torch.testing.assert_close(getattr(cache, name)[:cache.size],
                                       getattr(reference, name)[:reference.size], rtol=0, atol=0)
        allocated_capacity = cache.keys.shape[0]
        cache.reset()
        self.assertEqual(cache.size, 0)
        self.assertTrue(all(identity is None for identity in cache.identities))
        self.assertEqual(cache.keys.shape[0], allocated_capacity)

    def test_identity_cache_compares_unrounded_priority_like_priority_cache(self):
        cache = IdentityCache(1, 1, 1, "cpu", torch.float16)
        reference = PriorityCache(1, 1, 1, "cpu", torch.float16)
        entries = ((1.0, "kept-first"), (1.0001, "kept-second"), (1.0, "equal-priority-tie"))
        for key_value, (priority_value, identity) in enumerate(entries):
            key = torch.tensor([key_value], dtype=torch.float32)
            value = torch.tensor([key_value], dtype=torch.float32)
            entropy = torch.tensor([0.0], dtype=torch.float32)
            priority = torch.tensor([priority_value], dtype=torch.float32)
            target = int(reference.priorities.argmin()) if reference.size == 1 else 0
            admitted = reference.size == 0 or bool(priority[0] > reference.priorities[target])
            reference.add(key, value, entropy, priority)
            cache.add_identity(key, value, entropy[0], priority[0], identity)
            if admitted:
                expected_identity = identity

        self.assertEqual(cache.identities, [expected_identity])
        torch.testing.assert_close(cache.keys, reference.keys, rtol=0, atol=0)
        torch.testing.assert_close(cache.values, reference.values, rtol=0, atol=0)
        torch.testing.assert_close(cache.priorities, reference.priorities, rtol=0, atol=0)

    def test_view_logits_use_one_query_per_forward_independent_of_schedule_batch(self):
        x, _, _ = make_batch()
        probe = make_exact_probe()
        view_tensors = [[x[row] for _ in range(8)] for row in (0, 1)]
        observed_batch_sizes = []
        original_featurize = probe.model.featurize

        def record_batch_size(values):
            observed_batch_sizes.append(values.shape[0])
            return original_featurize(values)

        with patch.object(probe.model, "featurize", side_effect=record_batch_size):
            combined = probe._view_logits([0, 1], view_tensors)
        self.assertEqual(observed_batch_sizes, [8, 8])

        individual = []
        for row in range(2):
            with patch.object(probe.model, "featurize", wraps=probe.model.featurize) as featurize:
                individual.append(probe._view_logits([row], [view_tensors[row]]))
                self.assertEqual(featurize.call_count, 1)
                self.assertEqual(featurize.call_args.args[0].shape[0], 8)
        torch.testing.assert_close(combined, torch.cat(individual, dim=0), rtol=0, atol=0)

    def test_real_scorer_exhaustive_actions_labels_isolation_reset_and_cache_parity(self):
        x, pixels, preprocess = make_batch()
        probe, baseline = make_exact_probe(), make_ramen_baseline()
        probe.set_query_identity(torch.arange(100, dtype=torch.long))
        probe.set_query_images(pixels, preprocess)
        expected = baseline(x)
        objective_calls = []

        def independent_objective_replay(logits, view_logits, query_feature, caption_bank, logit_scale):
            expected_bank = probe.model.text_feature[expected.argmax(-1)].float()
            torch.testing.assert_close(caption_bank.float(), expected_bank, rtol=0, atol=0)
            probabilities = F.softmax(logits.float(), dim=-1)
            view_probabilities = F.softmax(view_logits.float(), dim=-1).mean(0)
            caption_logits = float(logit_scale) * (caption_bank.float() @ query_feature.float())

            def entropy(values):
                return float((-torch.special.xlogy(values, values).sum()).detach().cpu())

            direct = {
                OBJECTIVE_NAMES[0]: entropy(probabilities),
                OBJECTIVE_NAMES[1]: entropy(view_probabilities),
                OBJECTIVE_NAMES[2]: entropy(F.softmax(caption_logits, dim=-1)),
            }
            observed = objective_values(logits, view_logits, query_feature, caption_bank, logit_scale)
            for name in OBJECTIVE_NAMES:
                self.assertTrue(math.isclose(observed[name], direct[name], rel_tol=1e-6, abs_tol=1e-6))
            objective_calls.append((logits.detach().clone(), view_logits.detach().clone(),
                                    query_feature.detach().clone(), caption_bank.detach().clone()))
            return observed

        with patch("methods.QCGSExactObjectiveAudit.objective_values",
                   side_effect=independent_objective_replay):
            actual = probe(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(probe.action_rows), 16)
        self.assertEqual(len(objective_calls), sum(row["legal_swap_count"] + 1 for row in probe.action_rows))
        self.assertEqual([row["query"]["batch_row"] for row in probe.action_rows], list(range(16)))
        self.assertTrue(audit_scorer_rows(probe.action_rows))
        for row in probe.action_rows:
            self.assertEqual(row["legal_swap_count"], 25)
            self.assertEqual(len(row["actions"]), row["legal_swap_count"] + 1)
            self.assertEqual(len(row["view_record"]["view_tensor_sha256"]), 8)
            self.assertEqual(len(row["caption_pseudo_classes"]), 100)
            self.assertEqual(row["caption_pseudo_classes"], expected.argmax(-1).tolist())
            self.assertNotIn("known_label", row)
            self.assertNotIn("is_ood", row)
            self.assertNotIn("domain", row)
        self.assertFalse(hasattr(probe, "set_oracle_known_label"))
        self.assertEqual(list(inspect.signature(QCGSExactObjectiveAudit.forward).parameters), ["self", "x"])
        self.assertEqual(len({row["caption_bank_sha256"] for row in probe.action_rows}), 1)
        self.assertEqual(len({row["caption_bank_baseline_logits_sha256"] for row in probe.action_rows}), 1)
        self.assertEqual(len({row["ordinary_ramen_batch_features_sha256"] for row in probe.action_rows}), 1)

        chunked_probe = make_exact_probe()
        chunked_probe.cfg["action_chunk_size"] = 2
        chunked_probe.set_query_identity(torch.arange(100, dtype=torch.long))
        chunked_probe.set_query_images(pixels, preprocess)
        chunked_logits = chunked_probe(x)
        torch.testing.assert_close(chunked_logits, expected, rtol=0, atol=0)
        for left, right in zip(probe.action_rows, chunked_probe.action_rows):
            self.assertEqual(left["legal_actions"], right["legal_actions"])
            self.assertEqual(left["selections"], right["selections"])
            for left_action, right_action in zip(left["actions"], right["actions"]):
                self.assertEqual(left_action["objective_losses"], right_action["objective_losses"])

        for audit_cache, ramen_cache in zip(probe.cache, baseline.cache):
            self.assertEqual(audit_cache.size, ramen_cache.size)
            for name in ("keys", "values", "entropies", "priorities"):
                torch.testing.assert_close(getattr(audit_cache, name)[:audit_cache.size],
                                           getattr(ramen_cache, name)[:ramen_cache.size], rtol=0, atol=0)

        # Every trial starts at theta0; visiting two updates in either order
        # must reproduce each result and leave the admitted cache unchanged.
        cache_before = [(cache.keys[:cache.size].clone(), cache.values[:cache.size].clone())
                        for cache in probe.cache]
        first = torch.zeros(100, 4, dtype=torch.half)
        second = torch.zeros(100, 4, dtype=torch.half)
        first[4, 0], second[4, 3] = 0.2, -0.3
        a1 = probe._temporary_outputs(x, first)[0]
        b1 = probe._temporary_outputs(x, second)[0]
        b2 = probe._temporary_outputs(x, second)[0]
        a2 = probe._temporary_outputs(x, first)[0]
        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
        torch.testing.assert_close(b1, b2, rtol=0, atol=0)
        for (keys, values), cache in zip(cache_before, probe.cache):
            torch.testing.assert_close(keys, cache.keys[:cache.size], rtol=0, atol=0)
            torch.testing.assert_close(values, cache.values[:cache.size], rtol=0, atol=0)

        frozen = probe.pop_evaluator_payload()
        labels = torch.zeros(100, dtype=torch.long)
        is_ood = torch.zeros(100, dtype=torch.bool)
        domains = torch.zeros(100, dtype=torch.long)
        evaluator = ExactObjectiveEvaluator(seed=0)
        outcomes = evaluator.evaluate_batch(frozen, labels, is_ood, domains,
            model=probe.model, batch_images=x, reference_context=frozen["reference_context"], ood_cell="0")
        self.assertEqual(len(outcomes), 16)
        raw = audit_raw(probe.action_rows, outcomes)
        self.assertEqual(len(raw), 16)
        self.assertTrue(all(row["legal_swap_count"] == 25 for row in raw))
        self.assertTrue(all(row["correlations"][OBJECTIVE_NAMES[0]]["rho"] is None or
                            -1 <= row["correlations"][OBJECTIVE_NAMES[0]]["rho"] <= 1 for row in raw))
        frozen_selections = [copy.deepcopy(row["selections"]) for row in probe.action_rows]
        changed_metadata = ExactObjectiveEvaluator(seed=0).evaluate_batch(
            frozen, torch.ones_like(labels), is_ood, torch.full_like(domains, 3),
            model=probe.model, batch_images=x, reference_context=frozen["reference_context"], ood_cell="0")
        self.assertEqual(len(changed_metadata), len(outcomes))
        self.assertEqual(frozen_selections, [row["selections"] for row in probe.action_rows])

    def test_scorer_jsonl_appends_only_new_rows_across_batches(self):
        x, _, _ = make_batch()
        probe = make_exact_probe(mode="scan", schedule=[])
        with tempfile.TemporaryDirectory() as temp_dir:
            for batch in range(2):
                probe.set_query_identity(torch.arange(batch * 100, (batch + 1) * 100, dtype=torch.long))
                probe(x)
                write_scorer_outputs(probe, temp_dir)
                self.assertEqual(probe.pop_evaluator_payload()["queries"], [])
            scan_path = Path(temp_dir) / "qcgs-exact-scan.jsonl"
            rows = [json.loads(line) for line in scan_path.read_text().splitlines()]
            self.assertEqual(len(rows), 200)
            self.assertEqual([row["timestep"] for row in rows], list(range(200)))
            self.assertEqual(len({row["sample_idx"] for row in rows}), 200)

    def test_cost_profile_sidecar_records_component_and_jsonl_timings(self):
        x, pixels, preprocess = make_batch()
        schedule = [{"timestep": 0, "sample_idx": 0}, {"timestep": 1, "sample_idx": 1}]
        probe = make_exact_probe(mode="smoke", schedule=schedule)
        probe.profile_cost_components = True
        probe.cfg["profile_cost_components"] = True
        probe.set_query_identity(torch.arange(100, dtype=torch.long))
        probe.set_query_images(pixels, preprocess)
        probe(x)
        with tempfile.TemporaryDirectory() as temp_dir:
            write_scorer_outputs(probe, temp_dir)
            profile = json.loads((Path(temp_dir) / "qcgs-exact-profile.json").read_text())
        self.assertEqual(profile["scored_query_count"], 2)
        self.assertEqual(profile["batches"][0]["query_count"], 2)
        self.assertGreaterEqual(profile["batches"][0]["jsonl_serialization_ms"], 0)
        self.assertIn("candidate_gradient_aggregation_gpu_ms", profile["batches"][0]["component_timing_ms"])
        self.assertIn("candidate_full_batch_action_forward_gpu_ms", profile["batches"][0]["component_timing_ms"])
        self.assertIn("candidate_view_forward_gpu_ms", profile["batches"][0]["component_timing_ms"])
        self.assertIn("scorer_record_assembly_wall_ms", profile["batches"][0]["component_timing_ms"])
        self.assertGreaterEqual(profile["batches"][0]["record_sha256_ms"], 0)

    def test_startup_status_preserves_stage_b_failure_before_first_query(self):
        schedule = [{"timestep": index, "sample_idx": index} for index in range(128)]
        probe = make_exact_probe(mode="stage-b", schedule=schedule)
        with tempfile.TemporaryDirectory() as temp_dir:
            write_scorer_outputs(probe, temp_dir)
            status = json.loads((Path(temp_dir) / "qcgs-exact-status.json").read_text())
        self.assertEqual(status["stage"], "stage-b")
        self.assertEqual(status["scored_queries"], 0)
        self.assertEqual(status["requested_queries"], 128)
        self.assertFalse(status["complete"])
        self.assertEqual(status["evaluator_queries"], 0)
        self.assertFalse(status["evaluator_complete"])

    def test_evaluator_status_tracks_partial_and_complete_outcomes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "qcgs-exact-status.json"
            status_path.write_text(json.dumps({"schema_version": 1, "stage": "stage-a",
                "scored_queries": 2, "requested_queries": 2, "complete": True,
                "evaluator_queries": 0, "evaluator_complete": False}))
            write_evaluator_outputs([{"scorer_record_sha256": "a" * 64}], temp_dir)
            partial = json.loads(status_path.read_text())
            self.assertEqual(partial["evaluator_queries"], 1)
            self.assertFalse(partial["evaluator_complete"])
            write_evaluator_outputs([{"scorer_record_sha256": "b" * 64}], temp_dir)
            complete = json.loads(status_path.read_text())
            self.assertEqual(complete["evaluator_queries"], 2)
            self.assertTrue(complete["evaluator_complete"])

    def test_smoke_does_not_emit_an_unbound_parameter_delta_hash(self):
        x, pixels, preprocess = make_batch()
        schedule = [{"timestep": 0, "sample_idx": 0}, {"timestep": 1, "sample_idx": 1}]
        probe = make_exact_probe(mode="smoke", schedule=schedule)
        probe.set_query_identity(torch.arange(100, dtype=torch.long))
        probe.set_query_images(pixels, preprocess)
        probe(x)
        self.assertEqual(len(probe.action_rows), 2)
        self.assertTrue(all("parameter_delta_sha256" not in action
                            for row in probe.action_rows for action in row["actions"][1:]))
        self.assertTrue(audit_scorer_rows(probe.action_rows))

    def test_stage_b_evaluates_only_frozen_policy_and_forced_draw(self):
        payload = {"queries": [{
            "stage": "stage-b", "query": {"timestep": 100, "sample_idx": 42, "batch": 1,
                "block": 1, "batch_row": 0},
            "baseline_logits": torch.tensor([3.0, 0.0]),
            "actions": [
                {"action": {"index": 0, "action_id": "swap-0"}, "logits": torch.tensor([4.0, 0.0])},
                {"action": {"index": 1, "action_id": "swap-1"}, "logits": torch.tensor([0.0, 4.0])},
            ],
            "selections": {"memo_marginal_entropy": {"index": 0, "score": 0.1, "reason": "positive_canonical_argmax"}},
            "score_names": ["memo_marginal_entropy"], "record_sha256": "c" * 64,
            "candidate_pool_sha256": "d" * 64, "registry_sha256": "e" * 64,
        }], "reference_context": None}
        labels = torch.zeros(100, dtype=torch.long)
        ood = torch.zeros(100, dtype=torch.bool)
        domains = torch.zeros(100, dtype=torch.long)
        evaluator = ExactObjectiveEvaluator(seed=0)
        outcome = evaluator.evaluate_batch(payload, labels, ood, domains, ood_cell="0")[0]
        self.assertEqual(outcome["random_index"], random.Random(0).randrange(2))
        self.assertEqual(outcome["evaluated_action_indices"], [0, 1])
        self.assertNotIn("exact_oracle_utility", outcome)
        self.assertNotIn("exact_random_forced_expected_utility", outcome)
        self.assertEqual(outcome["policies"]["memo_marginal_entropy_random_matched"]["index"],
                         outcome["random_index"])
        self.assertEqual(outcome["policies"]["random_forced"]["index"], outcome["random_index"])

        abstain = copy.deepcopy(payload)
        abstain["queries"][0]["selections"]["memo_marginal_entropy"] = {
            "index": None, "score": 0.0, "reason": "zero_maximum_no_swap_wins"}
        other = ExactObjectiveEvaluator(seed=0).evaluate_batch(abstain, labels, ood, domains, ood_cell="0")[0]
        self.assertIsNone(other["policies"]["memo_marginal_entropy_random_matched"]["index"])
        self.assertGreater(outcome["policies"]["memo_marginal_entropy"]["utility"],
                           other["policies"]["memo_marginal_entropy"]["utility"])

    def test_scorer_record_rejects_evaluator_metadata(self):
        record, _ = stage_a_raw_fixture()
        for field in ("known_label", "is_ood", "domain", "correct", "utility", "oracle",
                      "supervised_gradient", "corruption", "severity", "evaluator_metadata"):
            with self.subTest(field=field):
                altered = copy.deepcopy(record)
                altered["query"][field] = 0
                altered["record_sha256"] = digest({key: value for key, value in altered.items()
                                                    if key != "record_sha256"})
                with self.assertRaisesRegex(ValueError, "label/metadata/utility"):
                    audit_scorer_rows([altered])


def stage_a_raw_fixture(timestep=100, sample_idx=5000, cell="0", run_identity="6" * 64,
                       rng=None, draw=0):
    rng = random.Random(0) if rng is None else rng
    identities = [
        {"timestep": 10, "sample_idx": 1000, "rank": 0},
        {"timestep": 20, "sample_idx": 2000, "rank": 1},
        {"timestep": 30, "sample_idx": 3000, "rank": 2},
    ]
    pool = {"0": {"k": 1, "values_sha256": "a" * 64, "weights_sha256": "b" * 64,
                  "candidates": identities}}
    legal = [{"class": 0, "out": 0, "in": 1}, {"class": 0, "out": 0, "in": 2}]
    baseline = {OBJECTIVE_NAMES[0]: 1.0, OBJECTIVE_NAMES[1]: 2.0, OBJECTIVE_NAMES[2]: 3.0}
    losses = [
        {OBJECTIVE_NAMES[0]: 0.8, OBJECTIVE_NAMES[1]: 1.8, OBJECTIVE_NAMES[2]: 3.1},
        {OBJECTIVE_NAMES[0]: 1.2, OBJECTIVE_NAMES[1]: 1.7, OBJECTIVE_NAMES[2]: 2.9},
    ]
    action_rows = []
    for index, action in enumerate(legal):
        name = f"swap-{index:05d}"
        action_rows.append({"index": index, "action_id": name, **action,
            "outgoing": {**identities[0], "rank": 0},
            "incoming": {**identities[index + 1], "rank": index + 1},
            "aggregate_gradient_sha256": "a" * 64,
            "objective_losses": losses[index],
            "objective_deltas": {name: baseline[name] - losses[index][name] for name in OBJECTIVE_NAMES}})
    selections = {}
    for name, values in {
        OBJECTIVE_NAMES[0]: [0.2, -0.2],
        OBJECTIVE_NAMES[1]: [0.2, 0.3],
        OBJECTIVE_NAMES[2]: [-0.1, 0.1],
    }.items():
        selections[name] = select_objective_action(values)
    row = {"schema_version": 1, "protocol": CURRENT_PROTOCOL, "stage": "stage-a",
        "query": {"timestep": timestep, "sample_idx": sample_idx, "batch": timestep // 100,
                  "block": timestep // 64, "batch_row": timestep % 100},
        "dataset_fingerprint": "f" * 64, "source_revision": "unit",
        "config_sha256": hashlib.sha256(f"config-{cell}".encode()).hexdigest(),
        "registry_sha256": "b" * 64, "parameter_order_sha256": "c" * 64,
        "run_identity_sha256": run_identity,
        "view_state": {"sha256": "d" * 64},
        "view_record": {"raw_image_sha256": "e" * 64, "input_sha256": "f" * 64,
                        "view_tensor_sha256": ["1" * 64] * 8},
        "caption_bank_sha256": "2" * 64, "caption_pseudo_classes": [0] * 100,
        "caption_bank_source": "per_row_ordinary_ramen_theta_R",
        "caption_bank_pseudo_classes_sha256": digest([0] * 100),
        "caption_bank_baseline_logits_sha256": "8" * 64,
        "ordinary_ramen_batch_features_sha256": "9" * 64,
        "baseline_logits_sha256": "3" * 64, "baseline_feature_sha256": "4" * 64,
        "baseline_pseudo_class": 0, "ramen_aggregate_sha256": "7" * 64,
        "parity": {"serial_aggregate_all_actions": True, "baseline_replay_before_trials": True,
                   "baseline_replay_after_trials": True, "cache_membership_unchanged": True,
                   "candidate_parameters_reset_per_trial": True, "action_chunk_isolation_verified": True,
                   "action_chunk_isolation_scope": "serial_no_grouped_queries",
                   "ordinary_ramen_logits_returned": True},
        "baseline_objectives": baseline, "cache_snapshot_sha256": "5" * 64,
        "active_class_count": 1, "candidate_pool": pool,
        "candidate_pool_sha256": digest(pool), "legal_swap_count": 2, "legal_actions": legal,
        "actions": [{"index": None, "action_id": "no-swap", "objective_losses": baseline,
                     "objective_deltas": {name: 0.0 for name in OBJECTIVE_NAMES}}, *action_rows],
        "selections": selections, "costs": {"candidate_action_ms": 1.0,
                                               "action_chunk_size": 1,
                                               "grouped_isolation_check_forwards_batch": 0,
                                               "peak_device_memory_bytes": None}}
    row["record_sha256"] = digest(row)

    rng_hash = digest(rng.getstate())
    random_index = rng.randrange(2)
    evaluated = [
        {"index": 0, "ce": 1.8, "entropy": 0.2, "margin": 1.0, "prediction": 0, "correct": True, "utility": 0.2},
        {"index": 1, "ce": 2.1, "entropy": 0.3, "margin": -1.0, "prediction": 1, "correct": False, "utility": -0.1},
    ]
    policies = {"no_swap": {"index": None, "utility": 0.0}}
    for name, choice in selections.items():
        policies[name] = {**choice, "utility": 0.2 if choice["index"] == 0 else -0.1 if choice["index"] == 1 else 0.0}
        matched_index = random_index if choice["index"] is not None else None
        policies[name + "_random_matched"] = {"index": matched_index,
            "utility": -0.1 if matched_index == 1 else 0.2 if matched_index == 0 else 0.0}
    policies["random_forced"] = {"index": random_index,
                                  "utility": -0.1 if random_index == 1 else 0.2}
    policies["supervised_first_order_reference"] = {"index": 0, "utility": 0.2,
        "score": 0.1, "scores": [0.1, -0.1], "reason": "positive_canonical_argmax"}
    outcome = {"schema_version": 1, "protocol": CURRENT_PROTOCOL, "stage": "stage-a",
        "ood_cell": cell, "query": {**row["query"], "domain": 0, "known_label": 0, "is_ood": False},
        "scorer_record_sha256": row["record_sha256"], "candidate_pool_sha256": row["candidate_pool_sha256"],
        "registry_sha256": row["registry_sha256"],
        "ramen": {"ce": 2.0, "entropy": 0.1, "margin": 1.0, "prediction": 0, "correct": True, "utility": 0.0},
        "actions": evaluated, "policies": policies, "random_draw": draw, "random_index": random_index,
        "rng_before_sha256": rng_hash, "exact_random_forced_expected_utility": 0.05,
        "exact_oracle_utility": 0.2, "exact_oracle_action_index": 0,
        "evaluated_action_indices": [0, 1], "costs": {"evaluator_ms": 1.0}}
    outcome["outcome_record_sha256"] = digest(outcome)
    return row, outcome


def stage_b_raw_fixture(timestep, sample_idx, cell, run_identity, rng, draw,
                        objective=OBJECTIVE_NAMES[0]):
    base, base_outcome = stage_a_raw_fixture(timestep, sample_idx, cell, run_identity, rng, draw)
    scorer = copy.deepcopy(base)
    scorer["stage"] = "stage-b"
    scorer["stage_b_objective"] = objective
    choice = select_objective_action([action["objective_deltas"][objective]
                                      for action in scorer["actions"][1:]])
    scorer["selections"] = {objective: choice}
    scorer["baseline_objectives"] = {objective: scorer["baseline_objectives"][objective]}
    for action in scorer["actions"]:
        action["objective_losses"] = {objective: action["objective_losses"][objective]}
        action["objective_deltas"] = {objective: action["objective_deltas"][objective]}
    scorer["record_sha256"] = digest({key: value for key, value in scorer.items() if key != "record_sha256"})

    random_index = base_outcome["random_index"]
    action_values = {0: (1.8, 0.2, 1.0, 0, True), 1: (2.1, 0.3, -1.0, 1, False)}
    selected_indices = sorted({random_index, choice["index"] if choice["index"] is not None else random_index})
    evaluated = []
    for index in selected_indices:
        ce, entropy, margin, prediction, correct = action_values[index]
        evaluated.append({"index": index, "ce": ce, "entropy": entropy, "margin": margin,
                          "prediction": prediction, "correct": correct, "utility": 2.0 - ce})
    random_utility = action_values[random_index][0]
    random_utility = 2.0 - random_utility
    selected_utility = 0.2 if choice["index"] == 0 else -0.1 if choice["index"] == 1 else 0.0
    matched_index = random_index if choice["index"] is not None else None
    policies = {"no_swap": {"index": None, "utility": 0.0},
        objective: {**choice, "utility": selected_utility},
        objective + "_random_matched": {"index": matched_index,
                                         "utility": random_utility if matched_index is not None else 0.0},
        "random_forced": {"index": random_index, "utility": random_utility}}
    outcome = {"schema_version": 1, "protocol": CURRENT_PROTOCOL, "stage": "stage-b",
        "ood_cell": cell, "query": {**scorer["query"], "domain": 0, "known_label": 0, "is_ood": False},
        "scorer_record_sha256": scorer["record_sha256"], "candidate_pool_sha256": scorer["candidate_pool_sha256"],
        "registry_sha256": scorer["registry_sha256"],
        "ramen": {"ce": 2.0, "entropy": 0.1, "margin": 1.0, "prediction": 0, "correct": True, "utility": 0.0},
        "actions": evaluated, "policies": policies, "random_draw": draw, "random_index": random_index,
        "rng_before_sha256": base_outcome["rng_before_sha256"],
        "evaluated_action_indices": selected_indices,
        "costs": {"evaluator_ms": 1.0}}
    outcome["outcome_record_sha256"] = digest(outcome)
    return scorer, outcome


class RegistryAndRawAnalysisTests(unittest.TestCase):
    def scans(self):
        return {cell: [{"timestep": i, "sample_idx": i, "domain": 0, "is_ood": False,
                        "eligible": True, "baseline_sha256": "a" * 64,
                        "aggregate_sha256": "b" * 64, "cache_snapshot_sha256": "c" * 64}
                       for i in range(600)] for cell in ("0", "0.5")}

    def test_historical_smoke_exclusions_hash_split_and_shortest_prefix(self):
        exclusions_file = json.loads(HISTORICAL_PATH.read_text())
        indices = exclusions_file["base_image_indices"]
        self.assertEqual(len(indices), 1499)
        self.assertEqual(hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest(),
                         "c6ea2be7074c89ca0ab8d9a6479c2a0a6aa43e4726e2a6a485d527ab22ddb72c")
        scans = self.scans()
        smoke = build_smoke_registry(scans)
        smoke_ids = {row["sample_idx"] for cell in smoke["cells"].values() for row in cell["selected"]}
        self.assertFalse(smoke_ids & set(indices))
        combined = sorted(set(indices) | smoke_ids)
        exclusions = {"base_image_indices": combined,
            "base_image_indices_sha256": digest(combined),
            "historical_base_image_indices": indices,
            "historical_base_image_indices_sha256": digest(indices),
            "smoke_base_image_indices": sorted(smoke_ids),
            "smoke_base_image_indices_sha256": digest(sorted(smoke_ids))}
        registry = build_exact_objective_registry(scans, "a" * 64, exclusions,
            source_revision="1" * 40, provenance=registry_provenance(scan_cap=600))
        validate_exact_objective_registry(registry)
        a = {row["sample_idx"] for row in registry["cells"]["0"]["selected"]["stage-a"]}
        b = {row["sample_idx"] for row in registry["cells"]["0"]["selected"]["stage-b"]}
        self.assertFalse(a & b)
        self.assertFalse((a | b) & set(combined))
        self.assertEqual(assigned_stage("a" * 64, next(iter(a)), EXACT_OBJECTIVE_NAMESPACE), "stage-a")
        for cell in ("0", "0.5"):
            last = max(row["timestep"] for stage in ("stage-a", "stage-b")
                       for row in registry["cells"][cell]["selected"][stage])
            self.assertEqual(registry["cells"][cell]["max_eval_samples"], 100 * (1 + last // 100))
            schedule = exact_objective_query_schedule(registry, "stage-a", cell)
            self.assertEqual(set(schedule[0]), {"timestep", "sample_idx"})
            self.assertFalse({row["sample_idx"] for row in schedule} & smoke_ids)

    def test_smoke_locks_propagate_to_scored_config_and_stage_b_requires_receipt(self):
        scans = self.scans()
        smoke = build_smoke_registry(scans)
        with tempfile.TemporaryDirectory() as temp_dir:
            smoke_runs = {}
            for cell in ("0", "0.5"):
                run_dir = Path(temp_dir) / ("smoke-" + cell.replace(".", ""))
                run_dir.mkdir()
                (run_dir / "manifest.json").write_text(json.dumps({
                    "config": {"qcgs_mode": "smoke"},
                    "args": {"max_eval_samples": smoke["cells"][cell]["max_eval_samples"]}}))
                rows = []
                for index, selected in enumerate(smoke["cells"][cell]["selected"]):
                    row, _ = stage_a_raw_fixture(
                        timestep=selected["timestep"], sample_idx=selected["sample_idx"], cell=cell,
                        run_identity=hashlib.sha256(f"smoke-{cell}-{index}".encode()).hexdigest(),
                        rng=random.Random(0), draw=index)
                    row["stage"] = "smoke"
                    row["dataset_fingerprint"] = "a" * 64
                    row["source_revision"] = "1" * 40
                    row["registry_sha256"] = smoke["sha256"]
                    row["registry_source_revision"] = "1" * 40
                    row["protocol"] = LEGACY_SMOKE_PROTOCOL
                    row["protocol_sha256"] = LEGACY_PROTOCOL_SHA256
                    row["view_state"] = {"sha256": "d" * 64}
                    row["parameter_order_sha256"] = "e" * 64
                    row["costs"]["peak_device_memory_bytes"] = (
                        8358872576 if cell == "0" else 8368574976)
                    row["record_sha256"] = digest({key: value for key, value in row.items()
                                                    if key != "record_sha256"})
                    rows.append(row)
                (run_dir / "qcgs-exact-actions.jsonl").write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
                smoke_runs[cell] = run_dir
            def smoke_receipt(run_dir, *_, **__):
                cell = "0" if Path(run_dir).name == "smoke-0" else "0.5"
                return {"scorer_complete": True, "evaluator_complete": None, "trace_complete": True,
                        "interrupted_jsonl_artifacts": [],
                        "scan_rows": smoke["cells"][cell]["max_eval_samples"],
                        "ood_cell": cell, "registry_sha256": smoke["sha256"],
                        "dataset_fingerprint": "a" * 64, "source_revision": "1" * 40,
                        "protocol_sha256": LEGACY_PROTOCOL_SHA256}

            with patch("evaluation.qcgs_exact_objective_audit.audit_run_provenance",
                       side_effect=smoke_receipt):
                registry = build_registry(
                    scans, smoke, smoke_runs, "a" * 64, "2" * 40,
                    scan_source_revision="1" * 40, smoke_source_revision="1" * 40,
                    scan_protocol_sha256=LEGACY_PROTOCOL_SHA256,
                    smoke_protocol_sha256=LEGACY_PROTOCOL_SHA256)

            self.assertEqual(registry["source_revision"], "2" * 40)
            self.assertEqual(registry["provenance"]["smoke_source_revision"], "1" * 40)
            self.assertEqual(registry["provenance"]["smoke_protocol_sha256"], LEGACY_PROTOCOL_SHA256)
            self.assertEqual(registry["provenance"]["active_protocol"], CURRENT_PROTOCOL)
            self.assertEqual(registry["provenance"]["peak_memory_policy"], "report_only")
            self.assertEqual(registry["provenance"]["smoke_peak_device_memory_bytes"],
                             {"0": 8358872576, "0.5": 8368574976})
            self.assertGreater(registry["provenance"]["smoke_peak_device_memory_bytes"]["0"],
                               7.5 * 1024 ** 3)

            import yaml
            profile_job = make_job(
                "smoke", "0", Path(temp_dir) / "profile-config", data_root=temp_dir,
                evidence_dir=Path(temp_dir) / "profile-evidence", run_id="smoke-profile-0",
                fingerprint="a" * 64, source_revision="1" * 40, smoke_registry=smoke,
                profile_cost_components=True, profile_only=True, profile_query_index=1)
            profile_config = yaml.safe_load(Path(profile_job["config"]).read_text())
            self.assertEqual(len(profile_config["query_schedule"]), 1)
            self.assertEqual(profile_config["query_schedule"][0]["sample_idx"],
                             smoke["cells"]["0"]["selected"][1]["sample_idx"])
            self.assertTrue(profile_config["profile_only"])
            self.assertTrue(profile_config["profile_cost_components"])

            def interrupted_smoke_receipt(run_dir, *_, **__):
                receipt = smoke_receipt(run_dir)
                receipt["trace_complete"] = False
                return receipt

            with patch("evaluation.qcgs_exact_objective_audit.audit_run_provenance",
                       side_effect=interrupted_smoke_receipt):
                with self.assertRaisesRegex(ValueError, "smoke receipt is incomplete"):
                    build_registry(
                        scans, smoke, smoke_runs, "a" * 64, "2" * 40,
                        scan_source_revision="1" * 40, smoke_source_revision="1" * 40,
                        scan_protocol_sha256=LEGACY_PROTOCOL_SHA256,
                        smoke_protocol_sha256=LEGACY_PROTOCOL_SHA256)
            scan_job = make_job("scan", "0", Path(temp_dir) / "scan-config", data_root=temp_dir,
                                evidence_dir=Path(temp_dir) / "scan-evidence", run_id="scan-test",
                                max_eval_samples=600, fingerprint="a" * 64,
                                source_revision="1" * 40)
            job = make_job("stage-a", "0", Path(temp_dir) / "config", data_root=temp_dir,
                           evidence_dir=Path(temp_dir) / "evidence", run_id="stage-a-test",
                           registry=registry, action_chunk_size=1)
            scan_config = yaml.safe_load(Path(scan_job["config"]).read_text())
            locked = yaml.safe_load(Path(job["config"]).read_text())
            self.assertEqual(scan_config["dataset_fingerprint"], "a" * 64)
            self.assertEqual(scan_config["source_revision"], "1" * 40)
            self.assertEqual(locked["view_state_sha256"], "d" * 64)
            self.assertEqual(locked["parameter_order_sha256"], "e" * 64)
            with self.assertRaisesRegex(ValueError, "committed Stage-A decision"):
                make_job("stage-b", "0", Path(temp_dir) / "stage-b-config", data_root=temp_dir,
                         evidence_dir=Path(temp_dir) / "stage-b-evidence", run_id="stage-b-test",
                         registry=registry,
                         analysis={"stage_a": {"decision": "GO_CONFIRM", "winner": OBJECTIVE_NAMES[0]}},
                         action_chunk_size=1)

    def test_stage_b_decision_must_match_committed_raw_recomputation(self):
        decision = {"decision": "GO_CONFIRM", "stage_a": {"decision": "GO_CONFIRM",
                    "winner": OBJECTIVE_NAMES[0]}, "stage_b": None,
                    "registry_sha256": "a" * 64, "source_revision": "1" * 40,
                    "raw_input_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            decision_path = repo / "plans" / "decision.json"
            decision_path.parent.mkdir()
            decision_path.write_text(json.dumps(decision))

            def git_run(command, **_):
                if command[1] == "ls-files":
                    output = "plans/decision.json\n"
                elif command[1] == "rev-parse":
                    output = "2" * 40 + "\n"
                else:
                    output = ""
                return SimpleNamespace(returncode=0, stdout=output, stderr="")

            registry = {"source_revision": "1" * 40}
            with patch.object(exact_runtime, "ROOT", repo), \
                    patch.object(exact_runtime.subprocess, "run", side_effect=git_run), \
                    patch.object(exact_runtime, "analyze_evidence", return_value=decision):
                verified, run_revision = exact_runtime._verify_committed_stage_a_decision(
                    decision, decision_path, repo / "raw", registry)
                self.assertEqual(verified, decision)
                self.assertEqual(run_revision, "2" * 40)

                forged = copy.deepcopy(decision)
                forged["stage_a"]["winner"] = OBJECTIVE_NAMES[1]
                with self.assertRaisesRegex(ValueError, "differs from independent raw recomputation"):
                    exact_runtime._verify_committed_stage_a_decision(
                        forged, decision_path, repo / "raw", registry)

    def test_incomplete_stage_b_preserves_completed_stage_a_decision(self):
        result = {"decision": "GO_ESTIMATOR",
                  "stage_a": {"decision": "GO_CONFIRM", "complete": True},
                  "stage_b": {"decision": "GO_ESTIMATOR", "complete": True}}
        updated = exact_runtime._apply_incomplete_run_status(
            result, [{"stage": "stage-b", "run_dir": "stage-b-ood-0"}])
        self.assertEqual(updated["decision"], "INCONCLUSIVE")
        self.assertEqual(updated["stage_a"], {"decision": "GO_CONFIRM", "complete": True})
        self.assertEqual(updated["stage_b"]["decision"], "INCONCLUSIVE")
        self.assertFalse(updated["stage_b"]["complete"])

    def test_interrupted_trace_uses_only_complete_batches_for_analysis(self):
        scan = [{"timestep": index, "sample_idx": index} for index in range(300)]
        trace = [{"timestep": index, "sample_idx": index} for index in range(250)]
        complete, prefix = _audit_trace_prefix(scan, trace)
        self.assertFalse(complete)
        self.assertEqual(prefix, 200)

        complete, prefix = _audit_trace_prefix(scan, scan)
        self.assertTrue(complete)
        self.assertEqual(prefix, 300)

        too_short = [{"timestep": index, "sample_idx": index} for index in range(199)]
        with self.assertRaisesRegex(ValueError, "exceeds one interrupted final batch"):
            _audit_trace_prefix(scan, too_short)

        mismatched = copy.deepcopy(trace)
        mismatched[-1]["sample_idx"] = -1
        with self.assertRaisesRegex(ValueError, "identities differ"):
            _audit_trace_prefix(scan, mismatched)

        broken_scan = copy.deepcopy(scan)
        broken_scan[250]["timestep"] = 999
        with self.assertRaisesRegex(ValueError, "scan sidecar timestep sequence"):
            _audit_trace_prefix(broken_scan, trace)

        partial_scan = scan[:250]
        partial_trace = trace[:200]
        complete, prefix = _audit_trace_prefix(
            partial_scan, partial_trace, scan_tail_incomplete=True)
        self.assertFalse(complete)
        self.assertEqual(prefix, 200)
        with self.assertRaisesRegex(ValueError, "complete 100-row batches"):
            _audit_trace_prefix(partial_scan, partial_trace)

    def test_partial_jsonl_tail_and_lagging_status_are_recoverable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "actions.jsonl"
            path.write_text('{"complete": 1}\n{"partial":')
            rows, receipt = read_jsonl_detailed(path)
            self.assertEqual(rows, [{"complete": 1}])
            self.assertTrue(receipt["truncated_tail"])
            self.assertFalse(receipt["errors"])
            self.assertEqual(len(receipt["sha256"]), 64)

            path.write_text('{"complete": 1}\n{"broken":}\n')
            rows, receipt = read_jsonl_detailed(path)
            self.assertEqual(rows, [{"complete": 1}])
            self.assertFalse(receipt["truncated_tail"])
            self.assertTrue(receipt["errors"])

        self.assertTrue(_reconcile_status_count({"scored_queries": 1}, "scored_queries", 2))
        self.assertFalse(_reconcile_status_count({"scored_queries": 2}, "scored_queries", 2))
        with self.assertRaisesRegex(ValueError, "overstates"):
            _reconcile_status_count({"scored_queries": 3}, "scored_queries", 2)
        self.assertTrue(_reconcile_status_completion({"complete": False}, "complete", True))
        self.assertFalse(_reconcile_status_completion({"complete": True}, "complete", True))
        with self.assertRaisesRegex(ValueError, "claims"):
            _reconcile_status_completion({"complete": True}, "complete", False)

    def test_campaign_analysis_discovers_prequery_stage_b_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_name, stage in (("stage-a", "stage-a"), ("stage-b", "stage-b"),
                                    ("scan", "scan"), ("smoke", "smoke")):
                run_dir = root / run_name
                run_dir.mkdir()
                config = {"qcgs_mode": stage}
                if stage == "stage-b":
                    config["frozen_objective"] = OBJECTIVE_NAMES[0]
                (run_dir / "manifest.json").write_text(json.dumps({
                    "config": config, "args": {"ood_ratio": 0.0}}))
                (run_dir / "qcgs-exact-status.json").write_text(json.dumps({"stage": stage}))

            def provenance(run_dir, *_):
                stage = Path(run_dir).name
                complete = stage == "stage-a"
                return {"stage": stage, "scorer_complete": complete,
                        "evaluator_complete": complete, "trace_complete": True,
                        "trace_rows": 100, "trace_prefix_rows": 100,
                        "interrupted_jsonl_artifacts": [], "status_reconciled": False,
                        "status_reconciled_fields": []}

            stage_a_decision = {"schema_version": 1, "decision": "GO_CONFIRM",
                                "stage_a": {"decision": "GO_CONFIRM", "complete": True,
                                            "winner": OBJECTIVE_NAMES[0]},
                                "stage_b": None}
            with patch("evaluation.qcgs_exact_objective_audit.audit_run_provenance",
                       side_effect=provenance) as audit, \
                    patch("evaluation.qcgs_exact_objective_audit.analyze_raw",
                          side_effect=lambda *_args, **_kwargs: copy.deepcopy(stage_a_decision)):
                result = exact_runtime.analyze_evidence(root)
                stage_b_manifest = root / "stage-b" / "manifest.json"
                manifest = json.loads(stage_b_manifest.read_text())
                manifest["config"]["frozen_objective"] = OBJECTIVE_NAMES[1]
                stage_b_manifest.write_text(json.dumps(manifest))
                mismatched = exact_runtime.analyze_evidence(root)

        self.assertEqual(audit.call_count, 4)
        self.assertEqual(result["stage_a"]["decision"], "GO_CONFIRM")
        self.assertTrue(result["stage_a"]["complete"])
        self.assertEqual(result["stage_b"]["decision"], "INCONCLUSIVE")
        self.assertFalse(result["stage_b"]["complete"])
        self.assertEqual(result["decision"], "INCONCLUSIVE")
        self.assertEqual([row["stage"] for row in result["incomplete_runs"]], ["stage-b"])
        self.assertEqual(mismatched["decision"], "INVALID")
        self.assertIn("Stage-B run configuration differs from the recomputed Stage-A winner",
                      mismatched["integrity_errors"])

    def test_campaign_analysis_drops_records_from_untraced_final_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "stage-a-run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(json.dumps({
                "config": {"qcgs_mode": "stage-a"}, "args": {"ood_ratio": 0.0}}))
            (run_dir / "qcgs-exact-status.json").write_text(json.dumps({"stage": "stage-a"}))
            rng = random.Random(0)
            first = stage_a_raw_fixture(timestep=0, sample_idx=5000, rng=rng, draw=0)
            tail = stage_a_raw_fixture(timestep=100, sample_idx=5001, rng=rng, draw=1)
            actions = [first[0], tail[0]]
            outcomes = [first[1], tail[1]]
            for filename, rows in (("qcgs-exact-actions.jsonl", actions),
                                   ("qcgs-exact-outcomes.jsonl", outcomes)):
                (run_dir / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
            with (run_dir / "qcgs-exact-actions.jsonl").open("a") as handle:
                handle.write('{"interrupted":')

            audit_status = {"stage": "stage-a", "scorer_complete": False,
                            "evaluator_complete": False, "trace_complete": False,
                            "trace_rows": 250, "trace_prefix_rows": 100,
                            "interrupted_jsonl_artifacts": [], "status_reconciled": False,
                            "status_reconciled_fields": []}
            captured = {}

            def capture_analysis(action_rows, outcome_rows, *args, **kwargs):
                captured["actions"] = action_rows
                captured["outcomes"] = outcome_rows
                return {"schema_version": 1, "decision": "INCONCLUSIVE",
                        "stage_a": {"decision": "INCONCLUSIVE", "complete": False},
                        "stage_b": None}

            with patch("evaluation.qcgs_exact_objective_audit.audit_run_provenance",
                       return_value=audit_status), \
                    patch("evaluation.qcgs_exact_objective_audit.analyze_raw",
                          side_effect=capture_analysis):
                result = exact_runtime.analyze_evidence(temp_dir)

        self.assertEqual([row["query"]["timestep"] for row in captured["actions"]], [0])
        self.assertEqual([row["query"]["timestep"] for row in captured["outcomes"]], [0])
        self.assertEqual(result["decision"], "INCONCLUSIVE")
        self.assertEqual(len(result["incomplete_runs"]), 1)
        self.assertIn("qcgs-exact-actions.jsonl", result["incomplete_runs"][0]["truncated_artifacts"])

    def test_untraced_trailing_records_are_audited_before_metric_filtering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "stage-a-run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(json.dumps({
                "config": {"qcgs_mode": "stage-a"}, "args": {"ood_ratio": 0.0}}))
            (run_dir / "qcgs-exact-status.json").write_text(json.dumps({"stage": "stage-a"}))
            rng = random.Random(0)
            first = stage_a_raw_fixture(timestep=0, sample_idx=5100, rng=rng, draw=0)
            tail = stage_a_raw_fixture(timestep=100, sample_idx=5101, rng=rng, draw=1)
            tail[1]["exact_oracle_utility"] += 0.5
            tail[1]["outcome_record_sha256"] = digest({
                key: value for key, value in tail[1].items() if key != "outcome_record_sha256"})
            for filename, rows in (("qcgs-exact-actions.jsonl", [first[0], tail[0]]),
                                   ("qcgs-exact-outcomes.jsonl", [first[1], tail[1]])):
                (run_dir / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))

            audit_status = {"stage": "stage-a", "scorer_complete": False,
                            "evaluator_complete": False, "trace_complete": False,
                            "trace_rows": 150, "trace_prefix_rows": 100,
                            "interrupted_jsonl_artifacts": [], "status_reconciled": False,
                            "status_reconciled_fields": []}
            with patch("evaluation.qcgs_exact_objective_audit.audit_run_provenance",
                       return_value=audit_status):
                result = exact_runtime.analyze_evidence(temp_dir)

        self.assertEqual(result["decision"], "INVALID")
        self.assertIn("Stage A exact oracle mismatch", result["integrity_errors"][0])

    def test_prequery_stage_b_run_requires_recomputed_stage_a_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "stage-b-run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(json.dumps({
                "config": {"qcgs_mode": "stage-b"}, "args": {"ood_ratio": 0.0}}))
            (run_dir / "qcgs-exact-status.json").write_text(json.dumps({"stage": "stage-b"}))
            audit_status = {"stage": "stage-b", "scorer_complete": False,
                            "evaluator_complete": False, "trace_complete": True,
                            "trace_rows": 0, "trace_prefix_rows": 0,
                            "interrupted_jsonl_artifacts": [], "status_reconciled": False,
                            "status_reconciled_fields": []}
            with patch("evaluation.qcgs_exact_objective_audit.audit_run_provenance",
                       return_value=audit_status):
                result = exact_runtime.analyze_evidence(temp_dir)

        self.assertEqual(result["decision"], "INVALID")
        self.assertEqual(result["stage_b"]["decision"], "INVALID")
        self.assertIn("Stage B run exists without a recomputed Stage-A GO_CONFIRM gate",
                      result["integrity_errors"])

    def test_registry_rejects_outcome_fields_and_tampered_prefix(self):
        scans = self.scans()
        scans["0"][0]["correct"] = True
        smoke = build_smoke_registry(self.scans())
        smoke_ids = sorted({row["sample_idx"] for cell in smoke["cells"].values()
                            for row in cell["selected"]})
        historical = json.loads(HISTORICAL_PATH.read_text())["base_image_indices"]
        combined = sorted(set(historical) | set(smoke_ids))
        exclusions = {"base_image_indices": combined,
            "base_image_indices_sha256": digest(combined),
            "historical_base_image_indices": historical,
            "historical_base_image_indices_sha256": digest(historical),
            "smoke_base_image_indices": smoke_ids,
            "smoke_base_image_indices_sha256": digest(smoke_ids)}
        with self.assertRaisesRegex(ValueError, "evaluator outcomes"):
            build_exact_objective_registry(scans, "a" * 64, exclusions,
                source_revision="1" * 40, provenance=registry_provenance())

        clean_scans = self.scans()
        clean_registry = build_exact_objective_registry(
            clean_scans, "a" * 64, exclusions, source_revision="1" * 40,
            provenance=registry_provenance())
        tampered = copy.deepcopy(clean_registry)
        tampered["cells"]["0"]["max_eval_samples"] += 100
        tampered["sha256"] = digest({key: value for key, value in tampered.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "shortest complete batch prefix"):
            validate_exact_objective_registry(tampered)

        changed_exclusions = copy.deepcopy(clean_registry)
        changed_exclusions["exclusions"]["smoke_base_image_indices"][-1] += 1
        changed_exclusions["sha256"] = digest({key: value for key, value in changed_exclusions.items()
                                                if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "smoke exclusions"):
            validate_exact_objective_registry(changed_exclusions)

        too_long = self.scans()
        too_long["0"] = [{"timestep": i, "sample_idx": i, "domain": 0, "is_ood": False,
                           "eligible": True, "baseline_sha256": "a" * 64,
                           "aggregate_sha256": "b" * 64, "cache_snapshot_sha256": "c" * 64}
                          for i in range(6001)]
        with self.assertRaisesRegex(ValueError, "6000"):
            build_exact_objective_registry(too_long, "a" * 64, exclusions,
                source_revision="1" * 40, provenance=registry_provenance())

    def test_raw_records_must_match_all_registry_provenance_locks(self):
        scans = self.scans()
        smoke = build_smoke_registry(scans)
        smoke_ids = sorted({row["sample_idx"] for cell in smoke["cells"].values()
                            for row in cell["selected"]})
        historical = json.loads(HISTORICAL_PATH.read_text())["base_image_indices"]
        excluded = sorted(set(historical) | set(smoke_ids))
        registry = build_exact_objective_registry(
            scans, "a" * 64,
            {"base_image_indices": excluded,
             "base_image_indices_sha256": digest(excluded),
             "historical_base_image_indices": historical,
             "historical_base_image_indices_sha256": digest(historical),
             "smoke_base_image_indices": smoke_ids,
             "smoke_base_image_indices_sha256": digest(smoke_ids)},
            source_revision="1" * 40,
            provenance=registry_provenance())
        selected = registry["cells"]["0"]["selected"]["stage-a"][0]
        scorer, outcome = stage_a_raw_fixture(
            timestep=selected["timestep"], sample_idx=selected["sample_idx"], cell="0",
            run_identity=hashlib.sha256(b"registry-bound-row").hexdigest(),
            rng=random.Random(0), draw=0)
        scorer.update(registry_sha256=registry["sha256"], dataset_fingerprint=registry["dataset_fingerprint"],
                      source_revision=registry["source_revision"],
                      registry_source_revision=registry["source_revision"],
                      parameter_order_sha256=registry["provenance"]["parameter_order_sha256"],
                      view_state={"sha256": registry["provenance"]["view_state_sha256"]})
        outcome["registry_sha256"] = registry["sha256"]
        scorer["record_sha256"] = digest({key: value for key, value in scorer.items()
                                           if key != "record_sha256"})
        outcome["scorer_record_sha256"] = scorer["record_sha256"]
        outcome["outcome_record_sha256"] = digest({key: value for key, value in outcome.items()
                                                    if key != "outcome_record_sha256"})
        self.assertEqual(len(audit_raw([scorer], [outcome], registry)), 1)

        mutations = (
            (lambda row, _: row.__setitem__("registry_sha256", "f" * 64), "scorer provenance differs"),
            (lambda row, _: row.__setitem__("dataset_fingerprint", "f" * 64), "scorer provenance differs"),
            (lambda row, _: row.__setitem__("source_revision", "f" * 40), "Stage-A source revision differs"),
            (lambda row, _: (row["costs"].__setitem__("action_chunk_size", 2),
                             row["parity"].__setitem__("action_chunk_isolation_scope",
                                                       "single_query_chunk_no_grouping")),
             "scorer provenance differs"),
            (lambda row, _: row["view_state"].__setitem__("sha256", "f" * 64), "scorer provenance differs"),
            (lambda row, _: row.__setitem__("parameter_order_sha256", "f" * 64), "scorer provenance differs"),
        )
        for mutate, expected_error in mutations:
            with self.subTest(mutation=mutate):
                altered, altered_outcome = copy.deepcopy(scorer), copy.deepcopy(outcome)
                mutate(altered, altered_outcome)
                altered["record_sha256"] = digest({key: value for key, value in altered.items()
                                                    if key != "record_sha256"})
                altered_outcome["scorer_record_sha256"] = altered["record_sha256"]
                altered_outcome["registry_sha256"] = altered["registry_sha256"]
                altered_outcome["outcome_record_sha256"] = digest({
                    key: value for key, value in altered_outcome.items() if key != "outcome_record_sha256"})
                with self.assertRaisesRegex(ValueError, expected_error):
                    audit_raw([altered], [altered_outcome], registry)

    def test_independent_raw_recomputation_random_expectation_and_gate(self):
        scorer, outcome = stage_a_raw_fixture()
        audited = audit_raw([scorer], [outcome])
        self.assertAlmostEqual(outcome["exact_random_forced_expected_utility"], 0.05)
        self.assertEqual(outcome["policies"]["random_forced"]["utility"], -0.1)
        self.assertAlmostEqual(audited[0]["correlations"][OBJECTIVE_NAMES[0]]["rho"], 1.0)
        self.assertEqual(analyze_raw([scorer], [outcome])["decision"], "INCONCLUSIVE")

        altered_random = copy.deepcopy(outcome)
        altered_random["policies"]["random_forced"]["utility"] += 0.5
        altered_random["outcome_record_sha256"] = digest({
            key: value for key, value in altered_random.items() if key != "outcome_record_sha256"})
        with self.assertRaisesRegex(ValueError, "random-forced utility mismatch"):
            audit_raw([scorer], [altered_random])

        extra = copy.deepcopy(scorer)
        extra["query"].update(timestep=200, sample_idx=5001, batch=2, block=3, batch_row=0)
        extra["record_sha256"] = digest({key: value for key, value in extra.items()
                                         if key != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "evaluator raw records lack matching frozen scorer records"):
            audit_raw([scorer, extra], [outcome])
        partial = analyze_raw([scorer, extra], [outcome], allow_unpaired_scorer=True)
        self.assertEqual(partial["decision"], "INCONCLUSIVE")

        corrupted_unpaired = copy.deepcopy(scorer)
        corrupted_unpaired["actions"][1]["index"] = 99
        corrupted_unpaired["record_sha256"] = digest({
            key: value for key, value in corrupted_unpaired.items() if key != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "action identity/order"):
            audit_raw([corrupted_unpaired], [], allow_unpaired_scorer=True)

        altered = copy.deepcopy(outcome)
        altered["actions"][0]["utility"] += 0.5
        altered["outcome_record_sha256"] = digest({key: value for key, value in altered.items()
                                                   if key != "outcome_record_sha256"})
        with self.assertRaisesRegex(ValueError, "utility does not recompute"):
            audit_raw([scorer], [altered])

        altered_baseline = copy.deepcopy(outcome)
        altered_baseline["ramen"]["correct"] = False
        altered_baseline["outcome_record_sha256"] = digest({
            key: value for key, value in altered_baseline.items() if key != "outcome_record_sha256"})
        with self.assertRaisesRegex(ValueError, "Ramen baseline correctness"):
            audit_raw([scorer], [altered_baseline])

        altered_scorer = copy.deepcopy(scorer)
        altered_scorer["actions"][1]["objective_deltas"][OBJECTIVE_NAMES[0]] *= -1
        altered_scorer["record_sha256"] = digest({key: value for key, value in altered_scorer.items()
                                                   if key != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "objective action loss/delta arithmetic"):
            audit_scorer_rows([altered_scorer])

    def test_stage_a_dispositions_distinguish_invalid_inconclusive_and_valid_stop(self):
        scorer_rows, outcome_rows = [], []
        for cell in ("0", "0.5"):
            rng = random.Random(0)
            for index in range(16):
                scorer, outcome = stage_a_raw_fixture(
                    timestep=index * 64, sample_idx=7000 + index, cell=cell,
                    run_identity=hashlib.sha256(f"STOP-{cell}-{index}".encode()).hexdigest(),
                    rng=rng, draw=index)
                for objective in OBJECTIVE_NAMES:
                    baseline = scorer["baseline_objectives"][objective]
                    scorer["selections"][objective] = select_objective_action([0.0] * scorer["legal_swap_count"])
                    for action in scorer["actions"]:
                        action["objective_losses"][objective] = baseline
                        action["objective_deltas"][objective] = 0.0
                    outcome["policies"][objective] = {"index": None, "score": 0.0,
                        "reason": "constant_objective_vector", "utility": 0.0}
                    outcome["policies"][objective + "_random_matched"] = {"index": None, "utility": 0.0}
                scorer["record_sha256"] = digest({key: value for key, value in scorer.items()
                                                   if key != "record_sha256"})
                outcome["scorer_record_sha256"] = scorer["record_sha256"]
                outcome["outcome_record_sha256"] = digest({key: value for key, value in outcome.items()
                                                            if key != "outcome_record_sha256"})
                scorer_rows.append(scorer)
                outcome_rows.append(outcome)
        stopped = analyze_raw(scorer_rows, outcome_rows)
        self.assertEqual(stopped["stage_a"]["decision"], "STOP")
        self.assertIsNone(stopped["stage_a"]["winner"])
        self.assertEqual(stopped["stage_a"]["cells"]["0"]["objectives"][OBJECTIVE_NAMES[0]][
            "per_query_rho"][0]["undefined_reason"], "constant_objective_vector")

        scorer_rows, outcome_rows = [], []
        for cell in ("0", "0.5"):
            rng = random.Random(0)
            for index in range(16):
                scorer, outcome = stage_a_raw_fixture(
                    timestep=index, sample_idx=7300 + index, cell=cell,
                    run_identity=hashlib.sha256(f"BLOCK-{cell}-{index}".encode()).hexdigest(),
                    rng=rng, draw=index)
                scorer_rows.append(scorer)
                outcome_rows.append(outcome)
        inconclusive = analyze_raw(scorer_rows, outcome_rows)
        self.assertEqual(inconclusive["stage_a"]["decision"], "INCONCLUSIVE")
        self.assertEqual(inconclusive["stage_a"]["reason"], "fewer_than_four_scored_blocks")

    def test_null_centered_bootstrap_tail_holm_and_ordinary_interval(self):
        rows = [{"block": block} for block in (0, 0, 1, 2, 3)]
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        result = _block_bootstrap(rows, values, draws=10000, seed=917)
        block_sums = np.asarray([4.0, 5.0, 7.0, 9.0])
        block_counts = np.asarray([2.0, 1.0, 1.0, 1.0])
        observed = sum(values) / len(values)
        rng = np.random.default_rng(917)
        selected = rng.integers(0, 4, size=(10000, 4))
        null_means = (block_sums - observed * block_counts)[selected].sum(1) / block_counts[selected].sum(1)
        expected_p = (1 + np.count_nonzero(null_means >= observed)) / 10001
        self.assertEqual(result["raw_one_sided_p"], expected_p)
        effect_means = block_sums[selected].sum(1) / block_counts[selected].sum(1)
        self.assertEqual(result["interval_95"], np.percentile(effect_means, [2.5, 97.5]).tolist())
        self.assertEqual(_holm_adjusted_pvalues([0.01, 0.03, 0.04, 0.5]), [0.04, 0.09, 0.09, 0.5])
        self.assertEqual(_holm_adjusted_pvalues([0.01, None, 0.04, 0.5]), [None] * 4)

    def test_stage_b_too_few_blocks_is_inconclusive(self):
        scorer_rows, outcome_rows = [], []
        for cell in ("0", "0.5"):
            rng = random.Random(0)
            for index in range(128):
                scorer, outcome = stage_b_raw_fixture(
                    timestep=index % 128, sample_idx=8000 + index, cell=cell,
                    run_identity=hashlib.sha256(f"B-BLOCK-{cell}-{index}".encode()).hexdigest(),
                    rng=rng, draw=index)
                scorer_rows.append(scorer)
                outcome_rows.append(outcome)
        rows = audit_raw(scorer_rows, outcome_rows)
        result = stage_b_gate(rows, OBJECTIVE_NAMES[0])
        self.assertEqual(result["decision"], "INCONCLUSIVE")
        self.assertTrue(all(item["raw_one_sided_p"] is None for item in result["primary_inference"]))

    def test_scorer_audit_checks_each_objective_selection(self):
        scorer, _ = stage_a_raw_fixture(
            timestep=0, sample_idx=4000, cell="0",
            run_identity=hashlib.sha256(b"selection-audit").hexdigest(),
            rng=random.Random(0), draw=0)
        for name in OBJECTIVE_NAMES:
            with self.subTest(objective=name):
                altered = copy.deepcopy(scorer)
                selected = altered["selections"][name]["index"]
                altered["selections"][name]["index"] = 1 - selected
                altered["record_sha256"] = digest({key: value for key, value in altered.items()
                                                    if key != "record_sha256"})
                with self.assertRaisesRegex(ValueError, "objective selection/tie/constant rule mismatch"):
                    audit_scorer_rows([altered])

    def test_stage_b_objective_must_match_recomputed_stage_a_winner(self):
        scorer_rows, outcome_rows = [], []
        for cell in ("0", "0.5"):
            rng = random.Random(0)
            for index in range(16):
                scorer, outcome = stage_a_raw_fixture(
                    timestep=index * 64, sample_idx=4000 + index, cell=cell,
                    run_identity=hashlib.sha256(f"A-{cell}-{index}".encode()).hexdigest(),
                    rng=rng, draw=index)
                scorer_rows.append(scorer)
                outcome_rows.append(outcome)
            scorer, outcome = stage_b_raw_fixture(
                timestep=100, sample_idx=7000 + (cell == "0.5"), cell=cell,
                run_identity=hashlib.sha256(f"B-{cell}".encode()).hexdigest(),
                rng=random.Random(0), draw=0, objective=OBJECTIVE_NAMES[1])
            scorer_rows.append(scorer)
            outcome_rows.append(outcome)
        result = analyze_raw(scorer_rows, outcome_rows)
        self.assertEqual(result["decision"], "INVALID")
        self.assertIn("differs from the recomputed Stage-A winner", result["integrity_errors"][0])

    def test_stage_a_winner_uses_query_weighted_defined_rhos(self):
        scorer_rows, outcome_rows = [], []
        for cell in ("0", "0.5"):
            rng = random.Random(0)
            for index in range(16):
                scorer, outcome = stage_a_raw_fixture(
                    timestep=index * 64, sample_idx=4000 + index, cell=cell,
                    run_identity=hashlib.sha256(f"weighted-{cell}-{index}".encode()).hexdigest(),
                    rng=rng, draw=index)
                scorer_rows.append(scorer)
                outcome_rows.append(outcome)
        rows = audit_raw(scorer_rows, outcome_rows)
        for cell in ("0", "0.5"):
            subset = [row for row in rows if row["ood_cell"] == cell]
            for row in subset:
                for objective in OBJECTIVE_NAMES:
                    row["outcome"]["policies"][objective]["utility"] = 0.2
            for index, row in enumerate(subset):
                if cell == "0":
                    row["correlations"][OBJECTIVE_NAMES[0]]["rho"] = 1.0 if index < 8 else None
                    row["correlations"][OBJECTIVE_NAMES[1]]["rho"] = 0.5 if index < 8 else None
                else:
                    row["correlations"][OBJECTIVE_NAMES[0]]["rho"] = 0.1
                    row["correlations"][OBJECTIVE_NAMES[1]]["rho"] = 0.4
                row["correlations"][OBJECTIVE_NAMES[2]]["rho"] = 0.05
        result = stage_a_gate(rows)
        self.assertEqual(result["decision"], "GO_CONFIRM")
        self.assertEqual(result["winner"], OBJECTIVE_NAMES[1])

    def test_stage_b_only_analysis_is_invalid_without_stage_a_decision(self):
        scorer, outcome = stage_b_raw_fixture(
            timestep=100, sample_idx=7000, cell="0",
            run_identity=hashlib.sha256(b"stage-b-only").hexdigest(),
            rng=random.Random(0), draw=0)
        result = analyze_raw([scorer], [outcome])
        self.assertEqual(result["decision"], "INVALID")
        self.assertIn("Stage B exists without a passing Stage-A gate", result["integrity_errors"])

        leaked = copy.deepcopy(outcome)
        leaked["exact_oracle_utility"] = 0.5
        leaked["outcome_record_sha256"] = digest({
            key: value for key, value in leaked.items() if key != "outcome_record_sha256"})
        with self.assertRaisesRegex(ValueError, "forbidden oracle/rank/sign/reference"):
            audit_raw([scorer], [leaked])

        scorer_leak = copy.deepcopy(scorer)
        scorer_leak["correlations"] = {OBJECTIVE_NAMES[0]: 1.0}
        scorer_leak["record_sha256"] = digest({key: value for key, value in scorer_leak.items()
                                                if key != "record_sha256"})
        rejected = analyze_raw([scorer_leak], [outcome])
        self.assertEqual(rejected["decision"], "INVALID")

    def test_complete_stage_a_winner_and_stage_b_holm_gate_recompute(self):
        scorer_rows, outcome_rows = [], []
        for cell in ("0", "0.5"):
            rng = random.Random(0)
            for index in range(16):
                scorer, outcome = stage_a_raw_fixture(
                    timestep=index * 64, sample_idx=4000 + index, cell=cell,
                    run_identity=hashlib.sha256(f"A-{cell}-{index}".encode()).hexdigest(),
                    rng=rng, draw=index)
                scorer_rows.append(scorer)
                outcome_rows.append(outcome)
        for cell in ("0", "0.5"):
            rng = random.Random(0)
            for index in range(128):
                scorer, outcome = stage_b_raw_fixture(
                    timestep=index * 4, sample_idx=6000 + index, cell=cell,
                    run_identity=hashlib.sha256(f"B-{cell}-{index}".encode()).hexdigest(),
                    rng=rng, draw=index)
                scorer_rows.append(scorer)
                outcome_rows.append(outcome)
        result = analyze_raw(scorer_rows, outcome_rows)
        self.assertEqual(result["stage_a"]["decision"], "GO_CONFIRM")
        self.assertEqual(result["stage_a"]["winner"], OBJECTIVE_NAMES[0])
        for cell in ("0", "0.5"):
            transition_rows = result["stage_a"]["cells"][cell]["transitions"]
            self.assertEqual(set(transition_rows), set(OBJECTIVE_NAMES))
            for item in transition_rows.values():
                self.assertEqual(sum(item["counts"].values()), 16)
        self.assertEqual(result["stage_b"]["decision"], "GO_ESTIMATOR")
        self.assertTrue(all(item["raw_one_sided_p"] is not None and
                            item["holm_adjusted_one_sided_p"] <= 0.05 and
                            len(item["interval_95"]) == 2 and
                            "holm_corrected_one_sided_95_interval" not in item
                            for item in result["stage_b"]["primary_inference"]))
        for cell in ("0", "0.5"):
            summary = result["stage_b"]["cells"][cell]["objectives"][OBJECTIVE_NAMES[0]]
            self.assertTrue({"selected_helpful_rate", "selected_harmful_rate",
                             "mean_helpful_magnitude", "mean_harmful_magnitude",
                             "mean_baseline_entropy", "mean_selected_entropy",
                             "mean_baseline_margin", "mean_selected_margin"} <= set(summary))
        self.assertEqual(result["cross_ood_base_image_overlap_and_direction"]["stage-b"][
            "scored_base_image_overlap"], 128)


if __name__ == "__main__":
    unittest.main()
