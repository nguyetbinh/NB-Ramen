# Phase 2 — Gate 1 Experiment

**Status:** complete — bounded no-go; see
[`reports/minimal-gate1-report.md`](reports/minimal-gate1-report.md)

## Protocol

- CIFAR-100-C, CLIP ViT-B/16.
- Canonical block stream, prefix 200, seed 0, block size 64.
- Reuse the already validated `CausalRamen` and oracle-hard artifacts on this
  exact stream rather than rerunning those controls.
- Run `OracleSoftRankRamen` only at `gamma=0` and `gamma=0.25`.
- Use `gamma=0` strictly as a baseline-recovery check; use `gamma=0.25` as the
  single nonzero soft-routing diagnostic.
- Identical memory capacity, top-k, beta, optimizer, learning rate, and current-sample policy.
- Pair adapted runs with the identical `NoAdapt` trace.

## Evidence

- Micro, macro-domain, and worst-domain accuracy.
- Negative-adaptation windows.
- Same/cross-domain support ratio.
- Returned support count, active-class count, class coverage, and effective sample size.

## Decision

- This bounded run is a mechanics/pilot check, not a tuned gamma sweep or a
  publication-level benchmark.
- Continue to a broader oracle-soft sweep only if `gamma=0.25` changes support
  selection while preserving diversity and gives a positive accuracy signal.
- Do not implement `LatentSoftRamen` from a null or negative result.

## Runtime

- Use CPU/MPS locally for validation and bounded evidence.
- Luna is unavailable in the current environment, so use the existing MPS
  artifacts and run only the missing bounded soft-routing cell locally.

## Result

- All reused artifacts passed strict resume validation on one stream
  fingerprint.
- `gamma=0` exactly recovered the `CausalRamen` algorithmic trace after
  excluding latency, method-identity routing fields, and the four soft-only
  zero diagnostics.
- `gamma=0.25` changed support selection on 73/200 queries but changed zero
  predictions and produced the same micro, macro-domain, worst-domain, and
  negative-window metrics as `CausalRamen`.
- The minimal Gate 1 therefore did not pass. Do not implement
  `LatentSoftRamen` or run more gamma values without a new explicit decision.
