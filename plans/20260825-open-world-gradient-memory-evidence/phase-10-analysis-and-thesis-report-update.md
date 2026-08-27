---
phase: 10
title: "Analysis and thesis report update"
status: pending
priority: P1
effort: "1d after evidence"
dependencies: [7, 8, 9]
---

# Phase 10: Analysis and thesis report update

## Overview

Validate all completed evidence, generate the prespecified comparisons, and
update the thesis only with claims supported by the frozen protocol.

## Requirements and ownership

- Modify: `src/evaluation/open_set_consensus_analysis.py` only if phase 5
  left an approved reporting gap; update `docs/research/open-world-gradient-memory-thesis-report.md`, `docs/research/experiment-runtime.md`, and an evidence report under this plan's `reports/` directory.
- Required analyses: ratio → GDC/SDR → oracle gap → consensus gain; per-stream
  effect; pre/post safety; ID-only stability; synchronized cost; two splits;
  DomainNet replication; required soft/no-self/tau/min-class ablation outputs.
- Preserve unavailable values and confidence/seed-level distributions; do not
  convert MPS/pilot results to canonical evidence.

## Implementation Steps

1. Strictly validate every run and coverage/fingerprint identity before aggregating.
2. Generate machine-readable report and tables/figures from validated artifacts.
3. Apply the thesis interpretation rules: distinguish OOD-specific benefit,
  general regularization, stream-conditional benefit, or an oracle gap that
  Consensus does not close.
4. Update completion audit and mark code-complete/evidence-complete separately.

## Todo

- [ ] `PYTHONPATH=src python -m evaluation.open_set_consensus_analysis --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-cifar100c-canonical" > "$RAMEN_EVIDENCE_ROOT/reports/cifar100c-analysis.json"`
- [ ] Run the focused analyzer and documentation tests after report formatting changes.

## Success Criteria

- [ ] The primary analyzer returns `canonical_cuda_expected`, all required studies are represented, and every table links to its artifact root/revision.
- [ ] The final report explicitly says whether each thesis condition is supported; no unsupported causal or universal claim remains.

## Risks and rollback

Do not average away missing cells, incompatible fingerprints, or unavailable
detection metrics. Keep rejected evidence out of aggregates and correct the
report rather than relaxing the validator.
