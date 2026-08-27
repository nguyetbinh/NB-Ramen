---
phase: 5
title: "F4: Synchronization, matrix identity, and ID-only stability"
status: pending
priority: P1
effort: "1d"
dependencies: [4]
---

# Phase 5: F4 — Synchronization, matrix identity, and ID-only stability

## Overview

Synchronize code, configs, planner identities, and reporting semantics before
freezing the canonical protocol. Fix open-set stability so OOD rows are never
treated as classification failures.

## Requirements

- [ ] Replace `EntropyGatedLatentRamen` with `EntropyGatedRamen` in the exact seven-method CIFAR and DomainNet open-set matrices; retain the former historical method.
- [ ] Keep 252 primary cells and frozen `tau=.2`, `min_consensus_classes=3`, source budget 400, split v1, and explicit CUDA/full-stream/provenance requirements.
- [ ] Compute paired negative adaptation over ordered ID rows only; compute worst-domain ID accuracy and episode recovery with original domain boundaries and minimum-ID-sample rules.
- [ ] Align the thesis report, config comments, analyzer contract, and matrix errors with Bernoulli soft admission and explicit pre/post detection.

## File ownership / touchpoints

- Modify: `src/runtime/experiment_matrix.py`, `src/runtime/open_set_domainnet_matrix.py`, `src/evaluation/open_set_consensus_analysis.py`, `src/evaluation/evidence.py` or a dedicated stability helper.
- Modify configs: `cfg/CIFAR100C/*`, `cfg/DomainNet/*` only as needed for exact selected-method config identity.
- Modify docs at execution time: `docs/research/open-world-gradient-memory-thesis-report.md`, `docs/research/experiment-runtime.md`.
- Test: `tests/test_experiment_matrix.py`, `tests/test_open_set_domainnet_matrix.py`, `tests/test_open_set_consensus_analysis.py`, `tests/test_evidence.py`.

## Implementation Steps

1. Change locked method tuples and expected-config validation atomically; assert exact seven methods, no fallback config, and 252 planned runs.
2. Introduce an ID-filtered paired trace comparator and ID-only recovery helper that preserves episode order rather than filtering domain structure away.
3. Update analyzer output to surface stability/cost rather than generic closed-set values.
4. Update report text only after behavior is test-covered; freeze config hashes and write the matrix planning JSON to the evidence ledger.

## Todo

- [ ] `PYTHONPATH=src python -m unittest tests.test_experiment_matrix tests.test_open_set_domainnet_matrix tests.test_open_set_consensus_analysis tests.test_evidence`
- [ ] `PYTHONPATH=src python -m runtime.experiment_matrix --open-set-consensus --device cuda --artifact-provenance fast --data-root "$RAMEN_DATA_ROOT" --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-cifar100c-canonical" > "$RAMEN_EVIDENCE_ROOT/runtime/open-set-cifar100c-plan.json"`
- [ ] `PYTHONPATH=src python -m runtime.open_set_domainnet_matrix --device cuda --artifact-provenance fast --data-root "$RAMEN_DATA_ROOT" --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-domainnet" > "$RAMEN_EVIDENCE_ROOT/runtime/open-set-domainnet-plan.json"`

## Success Criteria

- [ ] Both planners select the clean entropy method and reject changed canonical identities.
- [ ] Stability fixtures prove that interspersed OOD rows do not alter ID-only windows incorrectly and recovery excludes insufficient-ID episodes.
- [ ] A generated plan/config/report cannot recreate soft magnitude scaling or the latent-router baseline.

## Risks and rollback

Planner updates have public reproducibility consequences. Do not rename prior
evidence; create a new frozen run identity, preserve historical configs, and
halt execution if a method/config hash differs from the approved ledger.
