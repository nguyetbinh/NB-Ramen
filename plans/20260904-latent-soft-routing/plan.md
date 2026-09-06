# Latent Soft Routing Execution Plan

**Status:** margin-calibrated oracle Gate 1 complete — stop/pivot
**Branch:** `latent-soft-routing`
**Source:** [`docs/research/latent-soft-routing-next-direction.md`](../../docs/research/latent-soft-routing-next-direction.md)

## Phases

1. [x] [Implement controlled oracle-soft retrieval](phase-01-oracle-soft-implementation.md)
2. [x] [Validate and run the minimal reuse-first Gate 1 experiment](phase-02-gate-1-experiment.md)
3. [x] [Profile margins and run calibrated oracle-soft interventions](phase-03-margin-calibrated-oracle-soft.md)
4. [x] Add and execute the missing legacy-Ramen-plus-latent-router ablation.
5. [ ] Implement latent-soft routing — not authorized because calibrated oracle context produced no positive signal.

## Dependencies

- `nb-ramen` for local CPU/MPS validation.
- CIFAR-100-C and the cached CLIP ViT-B/16 checkpoint under `/Users/admin/data`.
- A configured Luna compute endpoint for CUDA execution; none is currently discoverable.

## Acceptance criteria

- Hard-routing behavior remains available through old names and explicit hard aliases.
- `gamma=0` recovers global class-balanced causal retrieval.
- Finite `gamma` never excludes cross-context candidates.
- Required support-composition diagnostics are persisted in evidence.
- Existing canonical CIFAR-100-C block, `n=200`, seed `0` controls are reused
  after fingerprint and summary validation.
- Replacement margins are profiled from the exact gamma-zero memory state.
- Weak, medium, and strong strengths are fixed from Q25/Q50/Q75 before their
  accuracy is read, and exactly those three missing cells are run.
- Latent-soft implementation starts only if nonzero oracle-soft routing beats `CausalRamen` without collapsing support diversity.

## Outcome

All calibrated points preserved support diversity but changed no predictions
and no primary metric. Margin profiling also showed that only 64 support-set
replacements are possible in this prefix; the existing `gamma=0.25` point is
already beyond the maximum measured replacement margin. The branch stops
before `LatentSoftRamen` and should pivot to a richer compatibility signal.
The direct legacy ablation additionally showed that the original prototype
router collapses even when attached to batch-atomic `PriorityCache` Ramen, so
the collapse is not caused by the causal structured implementation. See
[`reports/calibrated-gate1-report.md`](reports/calibrated-gate1-report.md) and
[`reports/legacy-latent-ablation-report.md`](reports/legacy-latent-ablation-report.md).
