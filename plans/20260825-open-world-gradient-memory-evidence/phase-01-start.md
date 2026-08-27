---
phase: 1
title: "Evidence contract and inventory"
status: pending
priority: P1
effort: "0.5d"
dependencies: []
---

# Phase 1: Evidence contract and inventory

## Overview

Turn the thesis report, runtime contract, and audits into one executable
evidence ledger before changing behavior. This phase establishes the boundary
between code-complete mechanics and evidence-complete science.

## Requirements

- [ ] Record the frozen primary protocol: 80/20 v1 split, ratios `0/.1/.3/.5`,
  `iid_mixed/block/recurring`, seeds `0/1/2`, 400 source examples/domain,
  full streams, seven methods, and CUDA + `fast`/`exact` provenance.
- [ ] Record all external blockers and the noncanonical status of local data.
- [ ] Define evidence acceptance: v2 trace/summary, immutable sibling files,
  exact code/config identity, and equal fingerprints for every paired cell.

## Related code files

- Read only: `docs/research/open-world-gradient-memory-thesis-report.md`,
  `docs/research/experiment-runtime.md`, `src/runtime/experiment_matrix.py`,
  `src/runtime/open_set_domainnet_matrix.py`, audits under
  `plans/20260824-latent-ramen-evidence/reports/`.
- Create during execution: `$RAMEN_EVIDENCE_ROOT/{runtime,open-set-cifar100c-canonical,open-set-split-robustness,open-set-domainnet}/` (not git).

## Implementation Steps

1. Run dependency-light preflight and archive host/runtime facts; do not claim data validity without `--deep`.
2. On a candidate CUDA host, archive `nvidia-smi`, torch/CUDA versions, git revision and config hashes before any run.
3. Acquire only Zenodo record 3555552 CIFAR-100-C; generate sidecar and require exact/deep validation. Require actual six-domain, 345-class DomainNet before its phase.
4. Maintain a cell ledger with planned run ID, baseline reference, artifacts, validation status, and reason for any unavailable result.

## Todo

- [ ] `PYTHONPATH=src python -m runtime.preflight --data-root "$RAMEN_DATA_ROOT" --dataset CIFAR100C --dataset DomainNet --json`
- [ ] `PYTHONPATH=src python -m runtime.preflight --data-root "$RAMEN_DATA_ROOT" --dataset CIFAR100C --dataset DomainNet --deep --json > "$RAMEN_EVIDENCE_ROOT/runtime/deep-preflight.json"`

## Success Criteria

- [ ] The ledger explicitly marks the present Mac as CPU/MPS-only and its CIFAR/DomainNet state as noncanonical/unavailable.
- [ ] No CUDA task starts before official archive/provenance and deep preflight pass.
- [ ] No thesis table accepts generated pilot data, a truncated prefix, MPS, or `artifact-provenance off` as canonical.

## Risks and rollback

Wrong data or an implicit `auto` device invalidates comparability. Stop the
affected run, retain its logs as rejected evidence, and rebuild the evidence
root; do not overwrite a completed run or downgrade validation.
