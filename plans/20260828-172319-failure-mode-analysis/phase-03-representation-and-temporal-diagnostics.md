# Phase 3: Representation and Temporal Diagnostics

Status: completed for the diagnostic primitives and report contract. The
four-sample pilot is insufficient for harmful-versus-beneficial or temporal
scientific conclusions.

## Context

- Frozen CLIP feature exports and evaluator labels.
- Paired outcome and mechanism rows from phases 1–2.

## Requirements

- Deterministic linear probes for feature-to-domain, feature-to-class, and class-conditioned domain decodability.
- Stratified harmful-versus-beneficial gradient-conflict summaries.
- Temporal bins for timestep, time since shift, memory occupancy, domain, seed, and stream.
- Include batch size and OOD ratio axes when present.
- Machine-readable paired-panel data for task/failure and mechanism plots.

## Implementation

1. Export frozen CLIP features through a post-hoc replay over the validated immutable stream, not through the adapted forward path. Bind feature rows to stream fingerprint, sample identity, split role, feature dimension/dtype, model artifact, and source fingerprint.
2. Add dependency-light deterministic probe utilities and CLI inputs with fixed split seed and explicit train/validation/test roles.
3. Add temporal/stratified aggregation with explicit counts and `computed`/`insufficient`/`unavailable` states.
4. Produce JSON report sections usable by plotting code without coupling analysis to a plotting backend.

## Validation

- Synthetic separable/non-separable probe fixtures.
- Shift-boundary, empty-stratum, one-class, and deterministic-seed tests.

## Risks and Rollback

- Small strata can mislead: always emit counts and mark insufficient comparisons.
- Analysis/final split leakage: require explicit split role in report metadata.
