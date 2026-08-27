---
phase: 4
title: "F3: Clean EntropyGatedRamen"
status: pending
priority: P1
effort: "1d"
dependencies: [3]
---

# Phase 4: F3 — Clean EntropyGatedRamen

## Overview

Add the intended prediction-confidence control: ordinary Ramen with only
memory admission gated by normalized entropy `≤ 0.50`; no latent routing.

## Requirements

- [ ] Preserve Ramen's feature/logit/gradient/retrieval/class-balancing/SignSGD/reset behavior and batch-atomic self visibility for admitted rows.
- [ ] Rejected rows never enter the cache; an empty cache yields zero/no update.
- [ ] Expose admission prediction, normalized entropy, admitted flag, retained memory, and pre-adaptation score.

## File ownership / touchpoints

- Create: `src/methods/EntropyGatedRamen.py`, `cfg/CIFAR100C/EntropyGatedRamen.yaml`, `cfg/DomainNet/EntropyGatedRamen.yaml`, `tests/test_entropy_gated_ramen.py`.
- Modify: `src/methods/__init__.py`; only registration/config selection needed by `src/main.py`.
- Read/align with: `src/methods/Ramen.py`, `src/methods/EntropyGatedLatentRamen.py`, `src/evaluation/evidence.py`.

## Implementation Steps

1. Extract/reuse Ramen-compatible cache and temporary-update helpers without importing a router.
2. Validate `max_normalized_entropy: 0.50` as immutable method config and add method diagnostics to the established trace group.
3. Test all-admit equivalence, all-reject empty history, mixed batches, self retrieval, reset, and evaluator-label isolation.
4. Confirm method memory accounting includes only retained ordinary cache state.

## Todo

- [ ] `PYTHONPATH=src python -m unittest tests.test_entropy_gated_ramen tests.test_ramen_memory_bytes tests.test_evidence`
- [ ] `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*ramen*.py'`

## Success Criteria

- [ ] No latent router import/instance exists in `EntropyGatedRamen`.
- [ ] All-admit fixture matches Ramen aggregate/update; rejected current samples cannot self-retrieve.
- [ ] Tests demonstrate reset and label isolation, and evidence validates diagnostics.

## Risks and rollback

Avoid “helpful” alternate fallbacks or per-class entropy behavior: this is a
clean control. Keep it as a new method/config so historical
`EntropyGatedLatentRamen` results remain reproducible.
