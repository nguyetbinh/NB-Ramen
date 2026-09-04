# Latent Soft Routing Execution Plan

**Status:** minimal Gate 1 complete — bounded no-go
**Branch:** `latent-soft-routing`
**Source:** [`docs/research/latent-soft-routing-research-direction.md`](../../docs/research/latent-soft-routing-research-direction.md)

## Phases

1. [x] [Implement controlled oracle-soft retrieval](phase-01-oracle-soft-implementation.md)
2. [x] [Validate and run the minimal reuse-first Gate 1 experiment](phase-02-gate-1-experiment.md)
3. [ ] Implement latent-soft routing — not authorized because the bounded Gate 1 did not pass.

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
- `OracleSoftRankRamen` completes only for `gamma=0` and one nonzero diagnostic
  (`gamma=0.25`).
- Latent-soft implementation starts only if nonzero oracle-soft routing beats `CausalRamen` without collapsing support diversity.

## Outcome

The minimal `gamma=0.25` intervention changed a small fraction of selected
support but did not change any of 200 predictions or any primary accuracy
metric. The branch stops at the oracle-soft diagnostic. See
[`reports/minimal-gate1-report.md`](reports/minimal-gate1-report.md).
