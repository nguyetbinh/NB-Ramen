"""Fixed current-image views and label-free soft-target mathematics."""
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

VIEW_SPEC = {'name': 'four-corners-30-flip-v1', 'input_shape': [32, 32, 3],
             'corners': [[0, 0], [2, 0], [0, 2], [2, 2]], 'crop_size': 30,
             'flip_order': [False, True], 'views': 8, 'teacher_anchor': 'theta_0',
             'gradient_anchor': 'theta_R', 'target': 'mean_probability_fp32',
             'temperature': 1., 'sharpening': False}


def raw_views(pixels):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or list(pixels.shape) != VIEW_SPEC['input_shape']:
        raise ValueError('views require the current corrupted uint8 RGB 32x32 image')
    image = Image.fromarray(pixels, mode='RGB')
    result = []
    for left, top in VIEW_SPEC['corners']:
        crop = image.crop((left, top, left + 30, top + 30))
        result.extend((crop, crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    return result


def ensemble_log_probabilities(view_logits):
    if view_logits.ndim != 3 or view_logits.shape[0] != 8 or not torch.isfinite(view_logits).all():
        raise ValueError('eight finite full-batch view logits are required')
    return torch.logsumexp(F.log_softmax(view_logits.float(), dim=-1), dim=0) - math.log(8)


def soft_target_ce(logits, target):
    if logits.shape != target.shape or not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError('nonfinite or incompatible soft target/logits')
    if target.requires_grad or (target < 0).any() or not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError('teacher target must be detached normalized probabilities')
    return -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1)


def direction_scores(h, base, candidates, lr):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('invalid learning rate')
    if h is None or h.shape != base.shape or candidates.ndim != 2 or candidates.shape[1:] != base.shape:
        raise ValueError('missing or incompatible gradient')
    if not all(torch.isfinite(t).all() for t in (h, base, candidates)):
        raise ValueError('nonfinite gradient')
    h = h.detach().float()
    norm = h.norm()
    if not torch.isfinite(norm):
        raise ValueError('nonfinite query gradient norm')
    if norm == 0:
        return None, ('zero_query_gradient', False)
    scores = lr * ((candidates.sign().float() - base.sign().float()) * h).sum(1)
    if not torch.isfinite(scores).all():
        raise ValueError('nonfinite score')
    reason = ('all_aggregate_directions_unchanged', False) if torch.equal(candidates.sign(), base.sign().expand_as(candidates)) else None
    return scores, reason
