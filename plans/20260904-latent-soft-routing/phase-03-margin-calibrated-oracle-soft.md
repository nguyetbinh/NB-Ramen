# Phase 3 — Margin-Calibrated Oracle Soft Routing

**Status:** complete — no positive oracle-soft signal

## Protocol

- CIFAR-100-C, CLIP ViT-B/16, canonical block stream, prefix 200, seed 0.
- Reuse the validated NoAdapt, CausalRamen, OracleHardRamen, gamma-zero, and
  gamma-0.25 artifacts on stream fingerprint
  `aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b`.
- Run one gamma-zero profile because historical traces cannot reconstruct
  candidate distances.
- Freeze weak, medium, and strong as the next float32 values above pooled
  replacement-margin Q25, Q50, and Q75 before reading nonzero accuracy.
- Run exactly those three new OracleSoftRankRamen cells. Do not implement a
  latent router or rerun controls.

## Evidence and validation

- Persist per-query replacement margins and reconstruct their distribution
  during strict resume validation.
- Bind generated configs and calibration to content hashes and source
  trace/config/stream identity.
- Pair every adapted run with the external canonical NoAdapt trace and config.
- Record accuracy, negative windows, support composition, class coverage,
  effective sample size, positional selection-change ratio, and rank displacement.

## Result

- Calibration: weak `0.0162578616`, medium `0.0333540924`, strong
  `0.0572425611`.
- All three points retained micro `0.330`, macro-domain `0.39453125`,
  worst-domain `0.28125`, and one negative window out of four.
- All three points changed zero predictions relative to gamma zero.
- Returned support, active classes, and median ESS remained effectively unchanged.
- The profile found only 64 possible support-set replacements across 17,773
  returned support slots. The old `gamma=0.25` exceeds the maximum measured
  margin (`0.1278753`) and already saturates these replacements.

## Decision

Gate B passes, but Gate C fails. Gate A's intended 10/25/50% membership-change
levels are infeasible for exact-domain bonus routing in this bounded stream.
Do not spend more compute increasing gamma and do not implement
`LatentSoftRamen`. Pivot to a compatibility signal that can distinguish more
candidate pairs, such as gradient agreement, reliability, semantic
compatibility, or continuous temporal/context affinity.

See [`reports/calibrated-gate1-report.md`](reports/calibrated-gate1-report.md).
