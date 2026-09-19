"""Label-free scoring only: no evaluator labels or provenance enter this module."""
from __future__ import annotations

import math
import random

import torch


def _fallback(reason, *, invalid=False):
    return {"index": None, "score": 0.0, "reason": reason, "invalid": invalid}


def score_swaps(query_gradient, base_gradient, candidate_gradients, incoming, outgoing, lr):
    """Return fp32 scores without changing baseline aggregation or its sign.

    Rows must already be in canonical legal-action order. Incoming/outgoing
    are unweighted cached gradients; candidate_gradients are exact aggregates.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    result = {"entropy_sign": None, "entropy_cosine": None, "reasons": {}}
    if query_gradient is None:
        reason = "missing_query_gradient"
    elif query_gradient.ndim != 1 or query_gradient.shape != base_gradient.shape:
        reason = "shape_incompatible_query_gradient"
    elif not torch.isfinite(query_gradient).all():
        reason = "nonfinite_query_gradient"
    else:
        reason = None
    if reason:
        result["reasons"] = {name: (reason, True) for name in ("entropy_sign", "entropy_cosine")}
        return result
    if candidate_gradients.ndim != 2 or candidate_gradients.shape[1:] != base_gradient.shape:
        raise ValueError("candidate aggregate shape mismatch")
    if incoming.shape != candidate_gradients.shape or outgoing.shape != incoming.shape:
        raise ValueError("cached gradient shape mismatch")
    if not all(torch.isfinite(t).all() for t in (base_gradient, candidate_gradients, incoming, outgoing)):
        result["reasons"] = {name: ("nonfinite_candidate", True) for name in ("entropy_sign", "entropy_cosine")}
        return result
    h = query_gradient.detach().float()
    if not torch.isfinite(h.norm()):
        reason, invalid = "nonfinite_query_norm", True
    elif h.norm() == 0:
        reason, invalid = "zero_query_gradient", False
    else:
        reason = None
    if reason:
        result["reasons"] = {name: (reason, invalid) for name in ("entropy_sign", "entropy_cosine")}
        return result
    # Take signs before casting: changing aggregate precision can change zeros.
    scores = lr * ((candidate_gradients.sign().float() - base_gradient.sign().float()) * h).sum(1)
    result["entropy_sign"] = scores
    if len(scores) and torch.equal(candidate_gradients.sign(), base_gradient.sign().expand_as(candidate_gradients)):
        result["reasons"]["entropy_sign"] = ("all_aggregate_directions_unchanged", False)
    in_norm, out_norm = incoming.float().norm(dim=1), outgoing.float().norm(dim=1)
    if not torch.isfinite(in_norm).all() or not torch.isfinite(out_norm).all():
        result["reasons"]["entropy_cosine"] = ("nonfinite_candidate_norm", True)
    elif bool((in_norm == 0).any() or (out_norm == 0).any()):
        result["reasons"]["entropy_cosine"] = ("zero_candidate_norm", False)
    else:
        result["entropy_cosine"] = ((incoming.float() * h).sum(1) / (in_norm * h.norm())
                                    - (outgoing.float() * h).sum(1) / (out_norm * h.norm()))
    for name in ("entropy_sign", "entropy_cosine"):
        if result[name] is not None and not torch.isfinite(result[name]).all():
            result[name] = None
            result["reasons"][name] = ("nonfinite_score", True)
    return result


def select_positive(scores, reason=None):
    if reason is not None:
        return _fallback(reason[0], invalid=reason[1])
    if scores is None or scores.numel() == 0:
        return _fallback("no_legal_swaps")
    if not torch.isfinite(scores).all():
        return _fallback("nonfinite_score", invalid=True)
    # torch.argmax returns the first maximum, preserving canonical tie order.
    index = int(scores.argmax())
    score = float(scores[index])
    if score <= 0:
        return _fallback("negative_maximum" if score < 0 else "zero_maximum_no_swap_wins")
    return {"index": index, "score": score, "reason": "positive_score", "invalid": False}


def select_one_swap(query_gradient, base_gradient, candidate_gradients, incoming, outgoing, lr, mode):
    if mode not in ("entropy_sign", "entropy_cosine"):
        raise ValueError("unknown selector mode")
    scores = score_swaps(query_gradient, base_gradient, candidate_gradients, incoming, outgoing, lr)
    return select_positive(scores[mode], scores["reasons"].get(mode))


def random_swap(count, rng: random.Random):
    if count == 0:
        return _fallback("no_legal_swaps")
    return {"index": rng.randrange(count), "score": None, "reason": "uniform_random", "invalid": False}
