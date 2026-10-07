# Historical CASE-QCGS-01 evidence packet — v1.1 resource-gate snapshot

**Historical decision:** `INCONCLUSIVE — P3 v1.1 memory gate` (superseded)
**Updated:** 2026-10-08
**Protocol for these receipts:** v1.1, SHA-256 `f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860`

The user-authorized v1.2 policy revision removes the precautionary peak threshold. Current status and next gates are in the [active resource-policy revision](resource-policy-revision.md); this packet preserves what was known under v1.1.

## Question and reference

Within a fixed Ramen query/cache state, do the three registered label-free exact objective changes rank legal same-pseudo-class one-support swaps by same-query cross-entropy utility? The motivating distinction is exact-objective information versus estimator fidelity. The synthesis's broad action-identifiability and selective-adaptation framing was narrowed after literature review; the local support-action question remains provisional and is not claimed as novel.

## Decision-relevant evidence

- Source/fix commit: `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93`, authored and committed as `[historical Git identity redacted]`. The `src/` Git archive SHA-256 is `35af0d707d7702af2990e022d34d858b13004c42a45d5b70190c546da805c511`.
- Independent fresh `gpt-6.1-sol`/low review of the lazy-cache factory, dtype-sensitive priority comparisons, reset/identity behavior and regressions reported no remaining findings. The reviewer environment lacked PyTorch; the repository environment ran the validation.
- Focused exact-objective suite: 35 passed. Full repository suite: 443 passed, 1 skipped. `py_compile` and `git diff --check` passed.
- Two fresh unscored scans completed 600 rows and passed source/data/cell/provenance/continuity checks. Allocator peaks were 8,369,873,408 bytes for OOD=0 and 8,368,574,976 bytes for OOD=0.5, each above the locked 7.5 GiB limit (8,053,063,680 bytes).
- The separate outcome-blind smoke registry SHA-256 is `40bd2afacb71482d01ddf65c33672c1a16340fc3fd9650eefdc6c70b5e014a8f`. Both smoke cells completed two registered queries, exhaustive legal swaps plus no-swap, complete trace coverage, finite scorer records, and scorer/parity/cache/reset receipts. OOD=0 had 485 swaps/query and an 8,358,872,576-byte peak; OOD=0.5 had 850 swaps/query and an 8,368,574,976-byte peak. Neither smoke run persisted evaluator outcomes. Process elapsed times were 12:25 and 22:43 (35:08 combined), within the 60-minute time cap; both memory caps failed.
- `build-registry` audited both smoke receipts and rejected scientific-registry construction with `ValueError: smoke must measure allocator peak below the frozen 7.5 GiB cap`. The proposed registry file was not created. The rejection record is preserved at `/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/decisions/registry-build-rejection-20261008.json`. No executable launch lock or Stage-A run exists.

## Interpretation and claim limits

This is a local execution-feasibility result on Python 3.11.15, PyTorch 2.6.0+cu124, and one NVIDIA RTX A2000 Laptop GPU with 8 GiB. The measured peaks do not establish whether any exact objective aligns with utility. Stage A, Stage B, reliability/enrichment, estimator development and confirmation are not reached. The earlier single-view and multi-view STOP reports were not recomputed because their raw archives remain unavailable. Do not describe this cycle as a scientific STOP, evidence of non-identifiability, or a negative objective result.

## Reproduction and artifacts

Deterministic launch settings, config hashes and commands are in [CUDA run reproduction](cuda-run-reproduction.md). The source review, scan/smoke measurements and gate rejection are in [instrument implementation review](instrument-implementation-review.md) and [resource-gate decision](resource-gate-decision.md). Case sampling and smoke IDs are in [case execution readiness](case-execution-readiness.md); claims and permissible wording are in [claim-evidence matrix](claim-evidence-matrix.md). Raw evidence and config locks are Git-ignored under `/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/`.

The failed first scan without `CUBLAS_WORKSPACE_CONFIG` and the two over-cap pre-fix scans remain preserved. All final scans and smokes used `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, the locked dataset fingerprint `115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc`, and the source revision above.

## Next bounded test under the old v1.1 disposition

That historical recommendation required smoke peaks below 7.5 GiB. It has been superseded; the current plan profiles runtime components without a peak gate and proceeds to Stage A if quota, provenance and compute-budget checks pass.
