# Grouped exact verification — 2026-09-07

Purpose: finish exhaustive Pilot A while preserving original Ramen B100 semantics.

For CLIP ViT only, test one candidate direction per query concurrently on the
original full input batch and original row positions. Each row owns independent
LayerNorm adaptation parameters; the inspected ViT has no cross-sample operations.
Other backbones remain serial. Every legal swap is still recorded. Equivalent
SignSGD directions share exact metrics, as the optimizer is state-free.

Temporary installation restores each BySampleLayerNorm/BySampleBatchNorm's known
batch row count directly instead of performing a discarded feature forward.
Before trials, its actual full-batch base output must exactly equal the original
Ramen output. The first two concurrently updated rows are also compared bitwise
to isolated full-batch trials on the actual backbone. Failure aborts the run.

A new CPU test compares all query rows/metrics for grouped and serial exhaustive
execution at B100/k5/m10, 16 ID queries. MPS Half coverage exercises grouped
verification as well. Full suite: 366 tests, 361 passed and 5 skipped; see
`grouped-full-tests.log`.

Independent read-only reviewer: DONE, no concrete correctness blocker. Reviewed
per-sample norm independence, optimizer defaults, same batch/row positions,
complete candidate coverage, reset-on-failure, direct row-count restoration and
the actual-backbone equality guards. The small-network tests do not substitute
for successful official-data execution; the runtime guards cover that run.
