# Historical research note: exact QCGS objective audit under v1.1 resource gate

**Status:** This note preserves the v1.1 resource-gate interpretation. It is superseded by the user-authorized v1.2 report-only peak policy and does not close CASE-QCGS-01.

## Motivation

Earlier QCGS experiments reported a STOP for single-view entropy-sign selection and for a multi-view target update, while a screened oracle suggested headroom. Those reports motivate a narrower question: within a fixed Ramen query and cache snapshot, can exact label-free objective changes rank legal one-support swaps by that query's supervised CE utility? Exact values would test objective information before spending effort on a cheap estimator. They would not by themselves produce a deployable label-free selector.

The literature reopening found that generic test-time action identifiability and selective adapt/skip are already active research areas. The remaining question is a bounded local support-action diagnostic. Its novelty remains provisional.

## Instrument and execution

The reviewed instrument computes original-view entropy, eight-view MEMO marginal entropy, and a CLIPTTA-derived image-to-text entropy for exhaustive legal support swaps. Every action returns to the frozen Ramen state. Labels and utility stay on the evaluator side. The lazy identity cache preserves the 750-support priority semantics and avoids constructing all 80 full-capacity cache tensors at initialization.

The implementation passed 35 focused tests and 443 repository tests (1 skipped); a fresh targeted source review reported no remaining findings. Two outcome-blind 600-row CUDA scans and both separate cost-only smoke cells completed with provenance and coverage receipts. Smoke covered 485 swaps/query in OOD=0 and 850 swaps/query in OOD=0.5. Combined smoke process elapsed was 35:08, below the 60-minute limit.

## Finding

The frozen memory gate failed in every measured cell. The scans peaked at 8,369,873,408 and 8,368,574,976 bytes; the smoke cells peaked at 8,358,872,576 and 8,368,574,976 bytes. Each exceeds the 7.5 GiB cap of 8,053,063,680 bytes. The registry builder audited both smoke receipts and refused to create a scientific registry. No Stage-A utility records were collected.

This is a resource feasibility outcome, not evidence for or against objective-to-utility alignment. It does not reproduce or revise prior scientific STOP reports. Stage A and conditional later stages are not reached.

## Historical decision under v1.1

At the time, this execution cycle was closed as `INCONCLUSIVE — resource gate`. Preserve the scans, smoke runs and registry rejection. That decision was based on the v1.1 memory cap and is superseded; no scientific result was collected.

## Current v1.2 status

High finite peaks are reported costs and do not block a complete run. The v1.2 registry check accepts the existing complete smoke receipts with explicit old source/protocol provenance, but the 600-row scan prefix lacks 8 Stage-A and 51 Stage-B eligible IDs at OOD=0. The next steps are cost-only component profiling on excluded smoke IDs, 1,200-row unscored scans, registry/launch lock, then Stage A within the frozen GPU-hour budgets. See [active resource-policy revision](resource-policy-revision.md).
