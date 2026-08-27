---
phase: 9
title: "DomainNet secondary benchmark"
status: blocked
priority: P1
effort: "External data + CUDA"
dependencies: [7]
---

# Phase 9: DomainNet secondary benchmark

## Overview

Execute the natural-domain replication only on the actual six-domain,
345-class dataset, keeping it secondary to the CIFAR primary contract.

## Requirements and ownership

- Modify: `src/runtime/open_set_domainnet_matrix.py` and exact configs/tests
  necessary to select `EntropyGatedRamen`; use existing `OpenSetDomainNet`
  recipe and provenance/preflight contracts.
- Require deep preflight with the real tree and matched taxonomy, then freeze
  an exposure budget appropriate to its six environments before execution.
- Execute the synchronized seven-method open-set CUDA matrix, full streams,
  explicit fast/exact provenance; preserve the fixed semantic name-rank split.

## Implementation Steps

1. Acquire/map DomainNet and archive deep preflight plus taxonomy/split digest.
2. Update/validate planner identity and run paired NoAdapt-first cells on CUDA.
3. Analyse mechanism trends against CIFAR, not headline accuracy alone.

## Todo

- [ ] `PYTHONPATH=src python -m runtime.preflight --data-root "$RAMEN_DATA_ROOT" --dataset DomainNet --deep --json > "$RAMEN_EVIDENCE_ROOT/runtime/domainnet-deep-preflight.json"`
- [ ] `PYTHONPATH=src python -m unittest tests.test_open_set_domainnet tests.test_open_set_domainnet_matrix tests.test_experiment_matrix`

## Success Criteria

- [ ] Completed evidence has real six-domain provenance, full artifacts, and strict paired validation.
- [ ] Results state the benchmark is secondary and report safety/mechanism/stability/cost separately.

## Risks and rollback

**Current blocker:** no DomainNet data and no CUDA runner. A synthetic tree or
pilot prefix may test code only; it cannot satisfy this phase.
