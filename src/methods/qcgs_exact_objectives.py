"""Frozen label-free objective arithmetic for the exact QCGS audit."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


OBJECTIVE_NAMES = (
    "single_view_entropy",
    "memo_marginal_entropy",
    "cliptta_image_to_text_entropy",
)


def _entropy(probabilities):
    probabilities = probabilities.float()
    if probabilities.ndim != 1 or not torch.isfinite(probabilities).all():
        raise ValueError("entropy requires one finite probability vector")
    if bool((probabilities < 0).any()):
        raise ValueError("entropy probabilities cannot be negative")
    return -torch.special.xlogy(probabilities, probabilities).sum()


def single_view_entropy(logits):
    """Shannon entropy on the original view and locked known-class logits."""
    logits = logits.float()
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("single-view entropy requires finite class logits")
    return _entropy(F.softmax(logits, dim=-1))


def memo_marginal_entropy(view_logits):
    """MEMO entropy of the equal-weight marginal over exactly eight views."""
    view_logits = view_logits.float()
    if view_logits.ndim != 2 or view_logits.shape[0] != 8 or not torch.isfinite(view_logits).all():
        raise ValueError("MEMO objective requires eight finite view-logit vectors")
    mean_probability = F.softmax(view_logits, dim=-1).mean(0)
    return _entropy(mean_probability)


def cliptta_image_to_text_entropy(query_feature, caption_bank, logit_scale):
    """Entropy of one image row against the frozen batch pseudo-caption bank."""
    query_feature = query_feature.float()
    caption_bank = caption_bank.float()
    if (query_feature.ndim != 1 or caption_bank.ndim != 2 or
            query_feature.shape[0] != caption_bank.shape[1] or caption_bank.shape[0] != 100):
        raise ValueError("CLIPTTA row objective requires one feature and a 100-row caption bank")
    if not torch.isfinite(query_feature).all() or not torch.isfinite(caption_bank).all():
        raise ValueError("CLIPTTA features must be finite")
    if type(logit_scale) not in (int, float) or not math.isfinite(logit_scale) or logit_scale <= 0:
        raise ValueError("CLIP logit scale must be finite and positive")
    logits = float(logit_scale) * (caption_bank @ query_feature)
    return _entropy(F.softmax(logits, dim=-1))


def objective_values(logits, view_logits, query_feature, caption_bank, logit_scale):
    """Return the three frozen losses as host floats, preserving their names."""
    values = {
        OBJECTIVE_NAMES[0]: single_view_entropy(logits),
        OBJECTIVE_NAMES[1]: memo_marginal_entropy(view_logits),
        OBJECTIVE_NAMES[2]: cliptta_image_to_text_entropy(query_feature, caption_bank, logit_scale),
    }
    result = {name: float(value.detach().cpu()) for name, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite exact objective")
    return result


def objective_deltas(baseline, action):
    """Return frozen improvement orientation: baseline loss minus action loss."""
    if set(baseline) != set(OBJECTIVE_NAMES) or set(action) != set(OBJECTIVE_NAMES):
        raise ValueError("objective names differ from the frozen contract")
    result = {name: float(baseline[name] - action[name]) for name in OBJECTIVE_NAMES}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite objective improvement")
    return result


def select_objective_action(action_deltas):
    """Apply the frozen no-swap, constant-vector and canonical tie rules."""
    if not action_deltas:
        return {"index": None, "score": 0.0, "reason": "no_legal_swaps"}
    values = [float(value) for value in action_deltas]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("objective action deltas must be finite")
    if len(set(values)) == 1:
        return {"index": None, "score": 0.0, "reason": "constant_objective_vector"}
    maximum = max(values)
    if maximum <= 0:
        reason = "negative_maximum" if maximum < 0 else "zero_maximum_no_swap_wins"
        return {"index": None, "score": 0.0, "reason": reason}
    # Python's first-max rule preserves the canonical class/out/in order.
    return {"index": values.index(maximum), "score": maximum, "reason": "positive_canonical_argmax"}
