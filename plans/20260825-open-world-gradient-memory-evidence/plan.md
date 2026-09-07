---
title: "Open-world Gradient-memory Thesis: Implementation and Evidence"
description: "Finish the frozen implementation contract, then collect and analyse canonical open-set evidence without conflating local pilots with thesis results."
status: in_progress
priority: P1
effort: "External-runner dependent"
tags: [thesis, open-set, ramen, cuda, evidence]
created: 2026-08-25
---

# Open-world Gradient-memory Thesis: Implementation and Evidence

## Overview

Implement F1--F4 from the thesis source of truth, validate mechanics locally,
then execute the prespecified CUDA evidence program. Code completion is not
thesis completion: the final claim requires verified official data, CUDA runs,
robustness evidence, DomainNet evidence, and strict post-hoc analysis.

## Current state and blockers (updated 2026-09-07)

- **Implementation complete:** F1--F4, evaluator-only ID/OOD isolation,
  pre/post-adaptation detection, ID-only stability, `EntropyGatedRamen`, the
  seven-method identities, deterministic v2/v3 split recipes, canonical
  planners, and the ablation/config/split locks are implemented and locally
  validated. This is code-contract completion, not effectiveness evidence.
- **Local evidence:** completed MPS 32-sample paired smoke artifacts are
  recorded in [the final evidence status](./reports/final-evidence-status-20260826.md).
  They validate artifact and reporting paths only; they cannot establish effect
  sizes, CUDA cost, or thesis claims.
- **Canonical blockers (do not reduce scope):** the official CIFAR-100-C
  archive/data is available at `/Users/admin/data`, but this Mac has no CUDA
  runtime. DomainNet is absent. A CUDA runner and actual full-stream execution
  remain required.

- **Pre-full correction:** open-set admission summaries now use known model
  indices for ID accuracy and separate semantic OOD fractions (trace v3,
  summary v4). Current-schema CUDA smoke and causal/package sensitivity are
  pending; see [readiness gates](./reports/pre-canonical-readiness-20260907.md).

## Dependency graph

```text
01 evidence contract/inventory
  └─> 02 F1 diagnostics ─> 03 F2 post-adaptation safety ─> 04 F3 clean entropy baseline
        └────────────────────────────────────────────────────> 05 F4 sync + ID-only stability
05 ─> 06 CPU/MPS mechanics verification
05 ─> 07 official-data CUDA matrix ─> 08 split robustness ─┐
                                      └> 09 DomainNet matrix ─┼─> 10 analysis/report update
06 supplies diagnostic confidence only; it never substitutes for 07--09. ────┘
```

## Goals

| # | Goal | Priority |
|---|------|----------|
| 1 | Preserve the thesis mechanism and evaluator-label isolation invariants | P1 |
| 2 | Complete F1--F4 and freeze exact configs/matrix identities | P1 |
| 3 | Produce canonical CUDA, split-robustness, and DomainNet evidence | P1 |
| 4 | Publish only conclusions supported by validated artifacts | P1 |

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Evidence contract and inventory](./phase-01-start.md) | Complete (implementation ledger and locks) |
| 2 | [F1 mechanism diagnostics](./phase-02-f1-mechanism-diagnostics.md) | Complete (implementation/local validation) |
| 3 | [F2 post-adaptation OOD safety](./phase-03-f2-post-adaptation-ood-safety.md) | Complete (implementation/local validation) |
| 4 | [F3 clean EntropyGatedRamen](./phase-04-f3-clean-entropygatedramen.md) | Complete (implementation/local validation) |
| 5 | [F4 synchronization and ID-only stability](./phase-05-f4-synchronization-and-id-only-stability.md) | Complete (implementation/local validation) |
| 6 | [Local CPU/MPS verification](./phase-06-local-cpu-and-mps-verification.md) | Complete, noncanonical smoke only |
| 7 | [Canonical CIFAR-100-C CUDA matrix](./phase-07-canonical-cifar-100-c-cuda-matrix.md) | Pending external CUDA execution (252 runs) |
| 8 | [Split robustness](./phase-08-split-robustness-study.md) | Pending external CUDA execution (96 runs) |
| 9 | [DomainNet secondary benchmark](./phase-09-domainnet-secondary-benchmark.md) | Pending: data and CUDA |
| 10 | [Analysis and thesis report update](./phase-10-analysis-and-thesis-report-update.md) | Pending canonical evidence |

## Success Criteria

- [x] F1--F4 tests and focused integration tests pass with no regression to ordinary Ramen.
- [x] Historical local artifacts and their original schemas are archived with revision and paired fingerprints where applicable.
- [ ] Regenerate current trace-v3 / summary-v4 CUDA artifacts; historical local runs do not satisfy this gate.
- [ ] Canonical CIFAR-100-C has all 252 CUDA cells (7 methods × 4 ratios × 3 streams × 3 seeds), official provenance, full streams, and strict classification `canonical_cuda_expected`.
- [x] Required ablation identities/config locks and two additional deterministic split recipes are frozen.
- [ ] The actual 72-run ablation study, actual 96-run split-robustness study, and real-data DomainNet secondary benchmark are completed and analysed.
- [ ] Final report labels MPS/CPU results noncanonical and states whether the observed mechanism supports each thesis research question.

## Ownership and evidence rules

- Implementation phases own only the listed source/config/test paths; phase 10 owns report/table changes after data exist. Never silently alter the frozen `tau=0.2`, `min_consensus_classes=3`, seven-method identity, 400-source budget, or primary split.
- Method-visible data are images, predictions, features, gradients, and ordinary caches. `is_ood`, original labels, and known labels remain evaluator-only; only named `Oracle*` hooks may consume `is_ood`.
- Store raw artifacts outside git under the selected evidence root. Preserve `manifest.json`, `stream.json`, `trace.jsonl`, `summary.json`, preflight/provenance JSON, runner/GPU logs, and analysis JSON. Reject rather than repair inconsistent evidence.

<!-- slug: open-world-gradient-memory-evidence -->
