"""Versioned QCGS rescue: frozen multi-view target, gradient at the Ramen anchor."""
import time

import numpy as np
from PIL import Image
import torch

from .QueryGradientUtilityProbe import QueryGradientUtilityProbe, synchronize, tensor_sha
from .OracleSupportUtilityProbe import BySampleLayerNorm, BySampleBatchNorm
from .losses import softmax_entropy
from .query_gradient_selection import random_swap, select_positive
from .query_gradient_multiview import VIEW_SPEC, raw_views, ensemble_log_probabilities, soft_target_ce, direction_scores
from runtime.query_gradient_registry import digest


def preprocess_spec(transform):
    """Stable transform identity, including resize interpolation and antialias."""
    if hasattr(transform, 'transforms'):
        return [preprocess_spec(t) for t in transform.transforms]
    result = {'callable': getattr(transform, '__module__', type(transform).__module__) + '.' +
              getattr(transform, '__qualname__', type(transform).__qualname__)}
    for field in ('size', 'interpolation', 'max_size', 'antialias', 'mean', 'std', 'inplace'):
        if hasattr(transform, field):
            value = getattr(transform, field)
            result[field] = str(value) if field == 'interpolation' else value
    return result


class MultiViewQueryGradientUtilityProbe(QueryGradientUtilityProbe):
    diagnostic_kind = 'qcgs-multiview'
    schema_version = 2
    score_names = ('entropy_sign', 'entropy_ramen', 'mv_target')

    def initialize_diagnostic(self):
        super().initialize_diagnostic()
        if self.cfg.get('view_spec') != VIEW_SPEC:
            raise ValueError('multi-view configuration must match the locked view definition')
        if self.registry is not None and self.registry.get('namespace') != 'qcgs-multiview-rescue-v1':
            raise ValueError('multi-view diagnostic requires its own query registry')
        self.raw_context = None
        self.view_state = {'view_spec': VIEW_SPEC, 'preprocess': None}

    def set_query_images(self, pixels, preprocess):
        if self.raw_context is not None:
            raise RuntimeError('stale raw-image context')
        self.raw_context = (np.array(pixels, copy=True), preprocess)
        state = {'view_spec': VIEW_SPEC, 'preprocess': preprocess_spec(preprocess)}
        state['sha256'] = digest(state)
        if self.view_state['preprocess'] is not None and self.view_state != state:
            raise ValueError('preprocessing changed within the stream')
        if self.cfg.get('view_state_sha256', state['sha256']) != state['sha256']:
            raise ValueError('preprocessing differs from the smoke lock')
        self.view_state = state

    def forward(self, x):
        context, self.raw_context = self.raw_context, None
        try:
            if context is None or len(context[0]) != len(x):
                raise RuntimeError('missing or mismatched current corrupted images')
            self.current_pixels, self.preprocess = context
            return super().forward(x)
        finally:
            self.raw_context = None
            self.identity = None
            self._pending = None
            self.current_pixels = self.preprocess = None
            self.signals = self.target = self.ensemble = self.view_records = None
            self.model.reset_parameters()
            self.model.optimizer.zero_grad()

    def reset(self):
        super().reset()
        self.raw_context = None

    def parameter_hash(self):
        return digest({entry['name']: tensor_sha(dict(self.model.model.named_parameters())[entry['name']])
                       for entry in self.parameter_order})

    def checked_gradient(self, loss, batch_size):
        self.model.optimizer.zero_grad()
        loss.backward()
        for module in self.model.model.modules():
            if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                for name in ('weight_by_sample', 'bias_by_sample'):
                    parameter = getattr(module, name)
                    if parameter.grad is None or parameter.grad.shape != parameter.shape or not torch.isfinite(parameter.grad).all():
                        raise ValueError('missing, incompatible or nonfinite adapted parameter gradient')
        result = self.model.get_by_sample_grad().detach().clone()
        if result.shape != (batch_size, self.parameter_order[-1]['stop']):
            raise ValueError('gradient/parameter ordering mismatch')
        return result

    def prepare_signals(self, x, base, baseline, query_gradients, selected):
        # No labels, domain, OOD flag, or retrieval changes enter this branch.
        synchronize(self.device)
        started = time.perf_counter()
        if self.device.type == 'cuda':
            self.reset_cuda_peak()
            memory_start = torch.cuda.memory_allocated(self.device)
        views = [raw_views(p) for p in self.current_pixels]
        self.view_records = {}
        for b in selected:
            original = self.preprocess(Image.fromarray(self.current_pixels[b], mode='RGB')).to(x)
            if not torch.equal(original, x[b]):
                raise ValueError('raw corrupted pixels do not reproduce the original query tensor')
            self.view_records[b] = {'raw_image_sha256': tensor_sha(torch.from_numpy(self.current_pixels[b])),
                                    'input_sha256': tensor_sha(x[b]), 'view_tensor_sha256': []}
        self.model.reset_parameters()
        self.model.optimizer.zero_grad()
        theta_0 = self.parameter_hash()
        with torch.no_grad():
            teacher_logits = []
            for v in range(8):
                batch = torch.stack([self.preprocess(p[v]) for p in views]).to(x)
                for b in selected:
                    self.view_records[b]['view_tensor_sha256'].append(tensor_sha(batch[b]))
                teacher_logits.append(self.model(batch).detach().float())
            teacher_logits = torch.stack(teacher_logits)
            teacher_logp = ensemble_log_probabilities(teacher_logits)
            self.target = teacher_logits.softmax(-1).mean(0).detach()
            del teacher_logits
        synchronize(self.device)
        teacher_ms = (time.perf_counter() - started) * 1000
        # Install θ_R via the same reset + one-step path used for every trial.
        if not torch.equal(self._temporary_logits(x, base), baseline):
            raise ValueError('multi-view anchor differs from Ramen')
        theta_r = self.parameter_hash()
        synchronize(self.device)
        started = time.perf_counter()
        anchor_logits = self.model(x)
        if not torch.equal(anchor_logits.detach(), baseline):
            raise ValueError('differentiable Ramen anchor changed')
        h_target = self.checked_gradient(soft_target_ce(anchor_logits, self.target)[selected].sum(), len(x))
        del anchor_logits
        synchronize(self.device)
        target_ms = (time.perf_counter() - started) * 1000
        primary_peak = (torch.cuda.max_memory_allocated(self.device) - memory_start if self.device.type == 'cuda' else None)
        started = time.perf_counter()
        anchor_logits = self.model(x)
        h_entropy = self.checked_gradient(softmax_entropy(anchor_logits.float(), reduction='none')[selected].sum(), len(x))
        del anchor_logits
        synchronize(self.device)
        entropy_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        with torch.no_grad():
            view_logits = []
            for v in range(8):
                batch = torch.stack([self.preprocess(p[v]) for p in views]).to(x)
                view_logits.append(self.model(batch).detach().float())
            ramen_logp = ensemble_log_probabilities(torch.stack(view_logits))
        synchronize(self.device)
        ensemble_ms = (time.perf_counter() - started) * 1000
        if self.parameter_hash() != theta_r:
            raise ValueError('side branches changed the Ramen anchor parameters')
        self.signals = {'entropy_sign': query_gradients, 'entropy_ramen': h_entropy, 'mv_target': h_target}
        self.ensemble = {'frozen_mv': teacher_logp, 'ramen_mv': ramen_logp}
        self.side_costs = {'teacher_batch_ms': teacher_ms, 'target_gradient_batch_ms': target_ms,
                           'entropy_anchor_batch_ms': entropy_ms, 'ramen_mv_batch_ms': ensemble_ms,
                           'teacher_full_batch_forwards': 8, 'target_full_batch_forwards': 1,
                           'target_backwards': 1, 'entropy_anchor_forwards': 1, 'entropy_anchor_backwards': 1,
                           'ramen_mv_full_batch_forwards': 8, 'anchor_install_forwards': 1,
                           'supervised_anchor_forwards': 1, 'supervised_reference_forwards': 1,
                           'supervised_reference_backwards': 1, 'baseline_replay_forwards': 1,
                           'primary_side_peak_extra_bytes': primary_peak}
        for b in selected:
            self.view_records[b].update(theta_0_parameter_sha256=theta_0, theta_r_parameter_sha256=theta_r,
                anchor_logits_sha256=tensor_sha(baseline[b]), target_sha256=tensor_sha(self.target[b]),
                target=self.target[b].cpu().tolist(),
                query_signals={name: {'sha256': tensor_sha(g[b]), 'norm': float(g[b].float().norm())}
                               for name, g in self.signals.items()})

    def score_candidates(self, b, query_gradient, base, gradients, incoming, outgoing):
        result = {'reasons': {}}
        for name in self.score_names:
            result[name], reason = direction_scores(self.signals[name][b], base, gradients, self.cfg['lr'])
            if reason is not None:
                result['reasons'][name] = reason
        return result

    def random_choices(self, count, choices):
        forced = random_swap(count, self.rng)
        matched = (dict(forced) if choices['mv_target']['index'] is not None else
                   {'index': None, 'score': 0., 'reason': 'primary_no_swap', 'invalid': False})
        return {'random_forced': forced, 'random_matched': matched}

    def install_reference_anchor(self, x, base):
        self._temporary_logits(x, base)
        self.model.optimizer.zero_grad()

    def reference_choice(self, values):
        return select_positive(torch.tensor(values, dtype=torch.float64))

    def query_metrics(self, logits, label, batch_index):
        return {**self.trial_metrics(logits, label),
                'soft_target_ce': float(soft_target_ce(logits, self.target[batch_index]))}

    def extra_record(self, b, label, baseline, job):
        controls = {}
        for name, logp in self.ensemble.items():
            controls[name] = {**self.query_metrics(logp[b], label, b), 'log_probabilities': logp[b].cpu().tolist()}
        return {'view_state': self.view_state, 'multiview': self.view_records[b],
                'prediction_controls': controls, 'side_costs': self.side_costs,
                'primary_verification_required': job['choices']['mv_target']['index'] is not None}
