"""Evaluator-only one-swap diagnostic; returned predictions remain ordinary Ramen."""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from models.ModelForBySampleTTA import BySampleLayerNorm, BySampleBatchNorm

from .Ramen import Ramen, PriorityCache, cache_distances
from .losses import softmax_entropy


class SupportProvenanceCache(PriorityCache):
    """Same admission policy as Ramen, with evaluator-only support metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = [None] * self.max_capacity

    def add_with_metadata(self, key, value, entropy, priority, metadata):
        target = self.size if self.size < self.max_capacity else int(self.priorities.argmin())
        admitted = self.size < self.max_capacity or bool(priority > self.priorities[target])
        super().add(key, value, entropy.reshape(1), priority.reshape(1))
        if admitted:
            self.metadata[target] = metadata

    def reset(self):
        super().reset()
        self.metadata = [None] * self.max_capacity


def aggregate_pools(pools, template, swap=None):
    """Recompute in baseline dtype/order, including rounding after each class."""
    total = torch.zeros_like(template)
    for c, pool in pools.items():
        indices = list(range(pool['k']))
        if swap is not None and swap['class'] == c:
            indices[swap['out']] = swap['in']
        if 'base_contribution' in pool and (swap is None or swap['class'] != c):
            contribution = pool['base_contribution']
        else:
            contribution = (pool['values'][indices] * pool['weights'][indices, None]).sum(0)
        total += contribution
    return total / len(pools)



def aggregate_swaps(pools, template, swaps):
    """Vectorize independent swaps while retaining class addition/dtype order."""
    total = template.new_zeros((len(swaps), template.numel()))
    for c, pool in pools.items():
        base = pool['base_contribution']
        positions = [i for i, swap in enumerate(swaps) if swap['class'] == c]
        if positions:
            indices = []
            for i in positions:
                chosen = list(range(pool['k']))
                chosen[swaps[i]['out']] = swaps[i]['in']
                indices.append(chosen)
            indices = torch.tensor(indices, device=template.device, dtype=torch.long)
            changed = (pool['values'][indices] * pool['weights'][indices, None]).sum(1)
            contribution = base.expand(len(swaps), -1).clone()
            contribution[positions] = changed
        else:
            contribution = base
        total += contribution
    return total / len(pools)

def legal_swaps(pools, candidate_m):
    if not any(len(pool['values']) >= candidate_m for pool in pools.values()):
        return
    for c, pool in pools.items():
        for outgoing in range(pool['k']):
            for incoming in range(pool['k'], min(candidate_m, len(pool['values']))):
                yield {'class': c, 'out': outgoing, 'in': incoming}


def logit_metrics(logits, label):
    logits = logits.float()
    others = logits.clone()
    others[label] = -torch.inf
    return {'ce': float(F.cross_entropy(logits[None], label.reshape(1))),
            'margin': float(logits[label] - others.max()),
            'correct': bool(logits.argmax() == label)}


class OracleSupportUtilityProbe(Ramen):
    requires_oracle_known_label = True

    def __init__(self, model, datasets, args):
        cfg = args.config
        if cfg.get('optimizer') != 'signsgd':
            raise ValueError('support utility requires state-free signsgd')
        for key in ('topk', 'candidate_m', 'max_capacity', 'probe_queries'):
            if type(cfg.get(key)) is not int or cfg[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if not cfg['topk'] < cfg['candidate_m'] <= cfg['max_capacity']:
            raise ValueError('require topk < candidate_m <= max_capacity')
        if cfg.get('probe_mode') not in ('exhaustive', 'screened'):
            raise ValueError('probe_mode must be exhaustive or screened')
        if cfg.get('oracle_label_source') != 'evaluator_known_label':
            raise ValueError('require evaluator_known_label provenance')
        for key in ('lr', 'beta'):
            value = cfg.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if cfg['lr'] == 0:
            raise ValueError('lr must be positive')
        super().__init__(model, datasets, args)
        self.cache = [SupportProvenanceCache(cfg['max_capacity'], self.feat_dim,
                      self.grad_dim, self.device, self.dtype) for _ in self.cache]
        self._pending = None
        self.rows = []
        self.rng = random.Random(args.seed)
        self.seed = args.seed

    def set_oracle_known_label(self, labels, *, is_ood, domains):
        if self._pending is not None:
            raise RuntimeError('stale oracle label context')
        if not all(isinstance(t, torch.Tensor) and t.ndim == 1 for t in (labels, is_ood, domains)):
            raise ValueError('context must contain one-dimensional tensors')
        if labels.dtype != torch.long or is_ood.dtype != torch.bool or domains.dtype != torch.long:
            raise ValueError('context requires long labels/domains and boolean is_ood')
        if labels.numel() == 0 or labels.shape != is_ood.shape or labels.shape != domains.shape:
            raise ValueError('context batch size mismatch')
        if not torch.equal(labels == -1, is_ood.to(labels.device)):
            raise ValueError('OOD queries must have label -1; cannot score OOD labels')
        if bool(((labels < -1) | (labels >= self.num_classes)).any()):
            raise ValueError('known label out of range')
        self._pending = tuple(t.detach().clone() for t in (labels, is_ood, domains))

    def _consume(self, size):
        context, self._pending = self._pending, None
        if context is None:
            raise RuntimeError('missing oracle label context')
        if context[0].numel() != size:
            raise RuntimeError('oracle label context batch size mismatch')
        return tuple(t.to(self.device) for t in context)

    def _temporary_logits(self, x, gradient):
        # Every adapted normalization layer sees the original batch. Restore its
        # row count directly; a throwaway feature forward is unnecessary.
        self.model.reset_parameters()
        self.model.optimizer.zero_grad()
        with torch.no_grad():
            for module in self.model.model.modules():
                if isinstance(module, (BySampleLayerNorm, BySampleBatchNorm)):
                    module.recent_B = len(x)
            for group in self.model.optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.requires_grad:
                        parameter.grad = torch.zeros_like(parameter)
        self.model.set_by_sample_grad(gradient)
        self.model.step_and_zero_grad()
        with torch.no_grad():
            return self.model(x).detach().clone()

    def forward(self, x):
        labels, is_ood, domains = self._consume(len(x))
        try:
            return self._forward(x, labels, is_ood, domains)
        finally:
            self.model.reset_parameters()
            self.model.optimizer.zero_grad()

    def _forward(self, x, labels, is_ood, domains):
        start = self.counter
        feats = self.model.featurize(x)
        logits = self.model.classify(feats)
        preds = logits.argmax(-1)
        self.last_diagnostics = {
            'pre_adaptation_prediction': preds.detach(),
            'pre_adaptation_ood_score': -torch.logsumexp(logits.detach(), dim=1),
        }
        self.loss_fn(logits).backward()
        grads = self.model.get_by_sample_grad().detach().clone()
        with torch.no_grad():
            priorities = torch.arange(start, start + len(x), device=self.device, dtype=self.dtype)
            entropies = softmax_entropy(logits, reduction='none')
            self.counter += len(x)
            for b in range(len(x)):
                self.cache[int(preds[b])].add_with_metadata(
                    feats[b], grads[b], entropies[b], priorities[b],
                    {'query_index': start+b, 'is_ood': bool(is_ood[b]), 'domain': int(domains[b])})
            self.last_diagnostics['memory_bytes'] = self.memory_bytes
            # Use the ordinary top-k query separately: top-m can break tied
            # distance ordering differently, so its first k is not a safe baseline.
            base = torch.zeros_like(grads)
            active = 0
            for cache in self.cache:
                if not cache.size:
                    continue
                values, _, entropy, distances = cache.query(feats, self.cfg['topk'])
                weights = torch.exp(-entropy) * torch.exp(-self.beta * distances)
                base += (values * weights.unsqueeze(2)).sum(1)
                active += 1
            base /= active
            self.model.set_by_sample_grad(base)
        self.model.step_and_zero_grad()
        with torch.no_grad():
            baseline = self.model(x).detach().clone()
        if (len(self.rows) >= self.cfg['probe_queries'] or not bool((~is_ood).any())
                or not any(cache.size >= self.cfg['candidate_m'] for cache in self.cache)):
            return baseline

        self.model.reset_parameters()
        self.model.optimizer.zero_grad()
        reference = self.model(x)
        F.cross_entropy(reference.float()[~is_ood], labels[~is_ood], reduction='sum').backward()
        query_grads = self.model.get_by_sample_grad().detach().float().clone()
        del reference
        with torch.no_grad():
            # Retrieval uses the entire original batch to preserve cdist kernel
            # selection and floating-point behavior at B=100.
            retrieved = {}
            for c, cache in enumerate(self.cache):
                if not cache.size:
                    continue
                dist = cache_distances(feats.detach().to(cache.dtype), cache.keys[:cache.size])
                k = min(self.cfg['topk'], cache.size)
                base_idx = dist.topk(k, dim=1, largest=False, sorted=True).indices
                top_m = dist.topk(min(self.cfg['candidate_m'], cache.size), dim=1,
                                  largest=False, sorted=True).indices
                retrieved[c] = (dist, base_idx, top_m)
            jobs = []
            for b in range(len(x)):
                if is_ood[b] or len(self.rows) + len(jobs) >= self.cfg['probe_queries']:
                    continue
                pools = {}
                for c, (dist, base_idx, top_m) in retrieved.items():
                    cache = self.cache[c]
                    indices = base_idx[b].tolist()
                    indices += [i for i in top_m[b].tolist() if i not in indices]
                    indices = indices[:self.cfg['candidate_m']]
                    distances = dist[b, indices]
                    entropy = cache.entropies[indices]
                    pools[c] = {'k': len(base_idx[b]), 'values': cache.values[indices],
                                'weights': torch.exp(-entropy) * torch.exp(-self.beta * distances),
                                'distances': distances, 'entropies': entropy,
                                'metadata': [cache.metadata[i] for i in indices]}
                for pool in pools.values():
                    pool['base_contribution'] = (pool['values'][:pool['k']] * pool['weights'][:pool['k'], None]).sum(0)
                    pool['cosines'] = F.cosine_similarity(query_grads[b][None], pool['values'].float(), dim=1).tolist()
                swaps = list(legal_swaps(pools, self.cfg['candidate_m']))
                if not swaps:
                    continue
                recomputed = aggregate_pools(pools, base[b])
                if not torch.equal(recomputed, base[b]):
                    raise RuntimeError('probe aggregation differs from ordinary Ramen')
                rq = query_grads[b]
                gradients = aggregate_swaps(pools, base[b], swaps)
                if not jobs:
                    # Validate every candidate against the original arithmetic
                    # on actual cache data once per evaluator batch.
                    for i, swap in enumerate(swaps):
                        if not torch.equal(gradients[i], aggregate_pools(pools, base[b], swap)):
                            raise RuntimeError('vectorized swap aggregation differs from serial arithmetic')
                directions = []
                for i, swap in enumerate(swaps):
                    gradient = gradients[i]
                    swap['predicted_utility'] = float(self.cfg['lr'] * torch.dot(
                        rq, gradient.sign().float() - base[b].sign().float()))
                    p = pools[swap['class']]
                    cos = p['cosines']
                    swap['cosine_gain'] = cos[swap['in']] - cos[swap['out']]
                    directions.append(gradient.sign().to(torch.int8).cpu().numpy().tobytes())
                    swap['feature_preference'] = float(p['distances'][swap['out']] - p['distances'][swap['in']])
                choices = {'best_predicted': max(range(len(swaps)), key=lambda i: swaps[i]['predicted_utility']),
                           'worst_predicted': min(range(len(swaps)), key=lambda i: swaps[i]['predicted_utility']),
                           'random': self.rng.randrange(len(swaps)),
                           'gradient_cosine': max(range(len(swaps)), key=lambda i: swaps[i]['cosine_gain'])}
                selected = range(len(swaps)) if self.cfg['probe_mode'] == 'exhaustive' else sorted(set(choices.values()))
                base_metrics = logit_metrics(baseline[b], labels[b])
                base_key = base[b].sign().to(torch.int8).cpu().numpy().tobytes()
                records = {}
                for i in selected:
                    record = dict(swaps[i])
                    pool = pools[record['class']]
                    for prefix, rank in [('outgoing', record['out']), ('incoming', record['in'])]:
                        record[prefix] = dict(pool['metadata'][rank], rank=rank+1,
                            feature_distance=float(pool['distances'][rank]), entropy=float(pool['entropies'][rank]),
                            query_gradient_cosine=pool['cosines'][rank])
                    records[i] = record
                jobs.append({'b': b, 'base': base_metrics, 'records': records,
                    'directions': {i: directions[i] for i in selected},
                    'metrics': {base_key: base_metrics}, 'choices': choices,
                    'legal_swap_count': len(swaps),
                    'pending': list(dict.fromkeys(directions[i] for i in selected if directions[i] != base_key))})
                print(f'Oracle ranked query {start+b}: {len(swaps)} swaps, {len(jobs[-1]["pending"])} unique updates', flush=True)
            # Check the optimized parameter installation against the original
            # Ramen forward on the actual backbone before any oracle trials.
            with torch.enable_grad():
                replayed_baseline = self._temporary_logits(x, base)
            if not torch.equal(replayed_baseline, baseline):
                raise RuntimeError('temporary parameter installation differs from Ramen baseline')
            self._verify_jobs(x, base, labels, jobs)
            for job in jobs:
                b = job['b']
                verified = job['records']
                for i, record in verified.items():
                    metrics = job['metrics'][job['directions'][i]]
                    record.update(metrics, exact_utility=job['base']['ce'] - metrics['ce'])
                best = max(verified, key=lambda i: verified[i]['exact_utility'])
                oracle = verified[best] if verified[best]['exact_utility'] > 0 else dict(job['base'], exact_utility=0.)
                self.rows.append({'query_index': start+b, 'true_label': int(labels[b]),
                    'query_domain': int(domains[b]), 'mode': self.cfg['probe_mode'],
                    'legal_swap_count': job['legal_swap_count'], 'verified_swap_count': len(verified),
                    'exact_forward_count': len(job['pending']),
                    'ramen': job['base'], 'oracle': oracle, 'best_forced_swap': verified[best],
                    'baselines': {'feature_similarity': dict(job['base'], exact_utility=0.),
                                  **{name: verified[i] for name, i in job['choices'].items()}},
                    'verified_swaps': list(verified.values())})
                progress = getattr(self, 'probe_progress', None)
                if progress is not None:
                    progress()
        return baseline

    def _verify_jobs(self, x, base, labels, jobs):
        # CLIP ViT has no cross-sample operations: each row owns its adapted
        # LayerNorm parameters. Keep the original batch and its row positions;
        # test one direction per query concurrently. Other backbones stay serial.
        grouped = getattr(self.model, 'model_type', None) == 'vit'
        rounds = max((len(job['pending']) for job in jobs), default=0)
        checked = False
        for step in range(rounds):
            active = [job for job in jobs if step < len(job['pending'])]
            groups = [active] if grouped else [[job] for job in active]
            for group in groups:
                gradient = base.clone()
                for job in group:
                    direction = torch.frombuffer(bytearray(job['pending'][step]), dtype=torch.int8)
                    gradient[job['b']] = direction.to(device=base.device, dtype=base.dtype)
                with torch.enable_grad():
                    logits = self._temporary_logits(x, gradient)
                if grouped and not checked:
                    # Verify actual-backbone equality against isolated full-batch
                    # trials before trusting concurrent query updates.
                    for job in group[:2]:
                        solo = base.clone()
                        solo[job['b']] = gradient[job['b']]
                        with torch.enable_grad():
                            single_logits = self._temporary_logits(x, solo)
                        if not torch.equal(logits[job['b']], single_logits[job['b']]):
                            raise RuntimeError('grouped ViT trial differs from isolated full-batch trial')
                    checked = True
                for job in group:
                    job['metrics'][job['pending'][step]] = logit_metrics(logits[job['b']], labels[job['b']])
            if step == 0 or (step+1) % 10 == 0 or step+1 == rounds:
                print(f'Oracle exact verification: {step+1}/{rounds} rounds, {len(jobs)} queries', flush=True)

    def reset(self):
        super().reset()
        self._pending = None
        self.rows = []
        self.rng = random.Random(self.seed)
