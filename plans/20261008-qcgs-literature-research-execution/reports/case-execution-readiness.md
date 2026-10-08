# CASE-QCGS-01 — pre-run card and registry readiness

**Prepared:** 2026-10-08

**Experiment:** `EXP-QCGS-001`

**Scientific gates:** preserved from v1.1. **Active resource policy:** [`qcgs-exact-objective-audit-v1.2`](../../../docs/research/qcgs-exact-label-free-objective-audit.md), SHA-256 `397e769d2eda50bad9a4744efbd7d928bf06640b9b15b06745ecd45955a5df03`

**Status:** scientific case remains active. Reviewed v1.1 scans/smokes and final v1.2 cost-only profiles are complete. High peaks are report-only under v1.2. The 1,200-row scans, current registry and launch lock remain outstanding; Stage A has not launched. See [active resource-policy revision](resource-policy-revision.md) for profile hashes and cost forecast.

## Case card

| Field | Frozen value |
|---|---|
| Purpose | Separate exact objective-to-utility alignment from the quality of a cheap gradient estimator before spending more on QCGS scoring engineering. |
| Setting/baseline | CIFAR-100-C severity 5, CLIP ViT-B/16, Ramen with original cache/retrieval/update and OOD stream cells 0 / 0.5; see protocol for all implementation details. |
| Intervention | Exhaustive legal same-pseudo-class one-support substitutions plus no-swap; exact reset and one-step Ramen update for every action. |
| Primary observation | Three frozen label-free objective improvements: original-view Shannon entropy, eight-view MEMO marginal entropy, and query-row CLIPTTA image-to-text batch soft-contrastive entropy. |
| Primary outcome | Per-query Spearman between exact objective improvement and same-query evaluator CE utility; query is the analysis unit. |
| Stage-A prediction | The old entropy-sign and MV-target reports do not establish exact-objective alignment. Exact entropy may remain weak; MEMO and CLIPTTA-derived batch evidence may separate from it if objective mismatch, rather than only first-order approximation, caused the prior failures. This is a directional mechanism prediction, not a fixed numerical effect claim. |
| Falsifier | No objective passes the frozen two-cell Stage-A information gate, or the selected Stage-A candidate fails fresh Stage-B utility/random/harm gates. Either closes only the tested objective/policy in this Ramen state. |
| Competing explanations | Missing cache/history context; strong class imbalance; current Ramen state differs from prior reports; view transforms are not semantic-preserving; local-runtime numerical drift; exact signal exists only in strata/cells; objective computation costs erase any practical value. |
| Controls | Ramen/no-swap, random-forced, random-matched, exact Stage-A oracle, evaluator-only local supervised first-order reference. |
| Sampling | 16 Stage-A and 128 reserved Stage-B known-ID queries per OOD cell; outcome-blind SHA-256 registry; 1,499 historical query exclusions plus separately excluded smoke IDs; max 6,000 scanned stream rows/cell. |
| Validity | Independent action enumeration, Ramen replay/cache/reset, label isolation, objective-value recomputation, registry replay, source/config/data/checkpoint hashes, review and CUDA smoke. Any failure is INVALID, not a scientific negative. |
| Budget | Smoke ≤60 minutes total; Stage A ≤8 GPU-hours/cell and ≤16 total; Stage B ≤24 GPU-hours/cell and ≤48 total only if opened; one GPU and serial execution; peak allocation report-only. |
| Decision | Stage A either freezes one passing candidate for fresh confirmation or stops this selector. Stage B GO opens a separate estimator case; valid failure is STOP for that frozen policy; resource/precision shortfall is INCONCLUSIVE. |

## Registry partition check

An independent outcome-free script applied the frozen SHA-256 assignment formula to the 10,000 CIFAR-100-C base-image IDs using the current P0 root digest `115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc` and the exclusion file. It reproduced 1,499 exclusions (list SHA-256 `c6ea2be7074c89ca0ab8d9a6479c2a0a6aa43e4726e2a6a485d527ab22ddb72c`), leaving 8,501 base IDs partitioned into 926 Stage-A and 7,575 Stage-B hash assignments. Protocol v1.1 keeps registry namespace `qcgs-exact-objective-audit-v1` because the review corrections do not alter query identity, hash assignment, or sampling rules.

This confirms only that the frozen partition has enough *base IDs* in principle. It does not verify ID eligibility, cache candidate eligibility, the earliest selected stream prefix, or absence from historical unscored cache streams. P3 registry replay must validate those from the actual locked scan before scoring. If eligible quotas are short by the 6,000-row cap, the result is INCONCLUSIVE; the hash rule and exclusions remain unchanged.

## Historical P3 CUDA receipts under v1.1 — 2026-10-08

Reviewed source commit: `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93`; source archive SHA-256: `35af0d707d7702af2990e022d34d858b13004c42a45d5b70190c546da805c511`. Both 600-row scans passed their scan/trace provenance checks on the reviewed revision, but allocator peaks were 8,369,873,408 bytes (OOD=0) and 8,368,574,976 bytes (OOD=0.5), above the frozen 7.5 GiB cap.

The separate outcome-blind smoke registry SHA-256 is `40bd2afacb71482d01ddf65c33672c1a16340fc3fd9650eefdc6c70b5e014a8f`. Its OOD=0 run completed two registered queries and 485 legal swaps plus no-swap per query; peak was 8,358,872,576 bytes. OOD=0.5 completed two registered queries and 850 legal swaps plus no-swap per query; peak was 8,368,574,976 bytes. Both smoke scorer/provenance audits passed, and neither run persisted evaluator outcomes. Combined observed process elapsed was 35:08. A v1.1 builder rejected construction at its allocator cap; that policy decision is superseded. The source/protocol of these receipts remains v1.1 and must be recorded separately from active v1.2 registry provenance.

The archived multi-view validation receipt reports 16,500 / 17,965 verified legal swaps across its two 16-query Stage-A cells, with per-query ranges 705–1,230 / 945–1,320. This is a scale reference only: its raw archive is missing and the new fresh-query cache states may yield different action counts. The new runner must emit exact attempted/enumerated/completed counts. The final v1.2 profile estimates Stage-A cost within the frozen budgets under a conservative 20% reserve, but only actual campaign timing can confirm those budgets.

## Run state

- P0 data and CLIP checkpoint identities passed their pinned checks; historical raw result archives are absent, so the old reports are not newly reproduced.
- P1 literature mapping is committed. P2 v1 was committed at `973d81d`; the fresh read-only P2 audit and its four resolved corrections are recorded in [protocol review](protocol-consistency-review.md), with v1.1 hash above.
- Reviewed implementation/fix commit is `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93` on `qcgs-exact-objective-code`; source archive SHA-256 is `35af0d707d7702af2990e022d34d858b13004c42a45d5b70190c546da805c511`. The latest fresh code review found no remaining findings; 35 focused tests and 443 repository tests (1 skipped) passed.
- The 600-row scan prefix is quota-short: the v1.2 builder dry run reported 8 Stage-A and 51 Stage-B eligible IDs missing at OOD=0 and requested 1,200 rows/cell. The high smoke peaks did not trigger rejection. This is a quota extension, not a scientific result.
- Final r3 cost-only profiles on source `9206f08f5d66229a9d1d526667ed8ed9c03e8a65` completed under the required deterministic CUDA environment, passed scorer/provenance audits, and contain no evaluator outcomes. Full-batch candidate forwards dominate measured cost; no exact speed optimization was chosen.
- No current scientific registry or executable launch lock exists. Stage A and later conditional stages are not reached. CASE-QCGS-01 remains active; objective-to-utility is untested. Next: 1,200-row scans, registry replay and launch lock. See [active resource-policy revision](resource-policy-revision.md).
