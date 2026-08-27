---
phase: 2
title: "F1: Consensus-versus-OracleID mechanism diagnostics"
status: pending
priority: P1
effort: "1d"
dependencies: [1]
---

# Phase 2: F1 — Consensus-versus-OracleID mechanism diagnostics

## Overview

Measure whether the deployable consensus mask moves an ordinary Ramen update
toward an evaluator-only ID-gradient oracle, without allowing labels into
ConsensusRamen.

## Requirements

- [ ] From the same retrieved supports build `g_ramen`, `g_consensus`, and
  `g_oracle_id`; OracleID still applies only `g_oracle_id`.
- [ ] `g_consensus = m ⊙ g_ramen`, where `m_k = 1[|mean_c sign(h_c,k)| ≥ .2]`;
  fall back to `g_ramen` below three active classes.
- [ ] Add all-or-none oracle trace group: `consensus_vs_oracle_id_cosine`,
  `consensus_vs_oracle_id_sign_disagreement`, `consensus_vs_ramen_cosine`,
  `consensus_diagnostic_mask_rate`, `consensus_diagnostic_applied`.
- [ ] Summarize Ramen/Consensus GDC and SDR plus mean reductions.

## File ownership / touchpoints

- Modify: `src/methods/OracleIDGradientRamen.py`, `src/evaluation/evidence.py`,
  `src/main.py`, `src/evaluation/open_set_consensus_analysis.py`.
- Test: `tests/test_oracle_id_gradient_ramen.py`, `tests/test_evidence.py`,
  `tests/test_open_set_consensus_analysis.py`.
- Do not modify: `src/methods/ConsensusRamen.py` semantics, ordinary `Ramen`,
  or deployable-method evaluator context plumbing.

## Implementation Steps

1. Refactor oracle aggregation to preserve per-class all-support contributions and reuse the locked Consensus aggregation math on those contributions.
2. Keep `is_ood` exclusively in construction of the ID-only contribution; test that changing it cannot affect the consensus mask/direction.
3. Extend strict trace and summary recomputation/validation before exposing analyzer fields.
4. Add synthetic vectors with known cosine/SDR and OOD=0 identity cases; then update post-hoc comparisons.

## Todo

- [ ] `PYTHONPATH=src python -m unittest tests.test_oracle_id_gradient_ramen tests.test_evidence tests.test_open_set_consensus_analysis`
- [ ] `PYTHONPATH=src python -m runtime.experiment_matrix --open-set-consensus --device cuda --artifact-provenance fast --data-root "$RAMEN_DATA_ROOT" --evidence-dir "$RAMEN_EVIDENCE_ROOT/plan-only"` (planning only; no host claim)

## Success Criteria

- [ ] Oracle applied updates are byte/behavior compatible with the prior implementation.
- [ ] In an OOD=0 fixture, Ramen/ID-oracle discrepancy is approximately zero.
- [ ] A synthetic fixture proves positive `GDC_reduction` and `SDR_reduction` when the mask removes conflicted coordinates.
- [ ] Partial diagnostic groups fail strict validation; a completed summary emits the required means.

## Risks and rollback

Sharing logic can accidentally make the diagnostic depend on oracle flags or
change class balancing. Keep separate all/ID sums, use the original active
class divisor, and revert only the isolated refactor if ordinary oracle update
regression tests fail.
