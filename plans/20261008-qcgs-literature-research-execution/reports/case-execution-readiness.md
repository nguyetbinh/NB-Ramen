# CASE-QCGS-01 — pre-run card and registry readiness

**Prepared:** 2026-10-08

**Experiment:** `EXP-QCGS-001`

**Protocol:** [`qcgs-exact-objective-audit-v1`](../../../docs/research/qcgs-exact-label-free-objective-audit.md), SHA-256 `bd0db538d01cd719bbd7a163e0a470fa14bbf85371a6a8531ce026859ca13221`

**Status:** case frozen; instrument implementation in progress; no smoke or scored run launched.

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
| Budget | Smoke ≤60 minutes total; Stage A ≤8 GPU-hours/cell; Stage B ≤24 GPU-hours/cell only if opened; one GPU and serial execution; peak allocation <7.5 GiB. |
| Decision | Stage A either freezes one passing candidate for fresh confirmation or stops this selector. Stage B GO opens a separate estimator case; valid failure is STOP for that frozen policy; resource/precision shortfall is INCONCLUSIVE. |

## Registry partition check

An independent outcome-free script applied the frozen SHA-256 assignment formula to the 10,000 CIFAR-100-C base-image IDs using the current P0 root digest `115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc` and the exclusion file. It reproduced 1,499 exclusions (list SHA-256 `c6ea2be7074c89ca0ab8d9a6479c2a0a6aa43e4726e2a6a485d527ab22ddb72c`), leaving 8,501 base IDs partitioned into 926 Stage-A and 7,575 Stage-B hash assignments.

This confirms only that the frozen partition has enough *base IDs* in principle. It does not verify ID eligibility, cache candidate eligibility, the earliest selected stream prefix, or absence from historical unscored cache streams. P3 registry replay must validate those from the actual locked scan before scoring. If eligible quotas are short by the 6,000-row cap, the result is INCONCLUSIVE; the hash rule and exclusions remain unchanged.

## Run state

- P0 data and CLIP checkpoint identities passed their pinned checks; historical raw result archives are absent, so the old reports are not newly reproduced.
- P1 source-backed literature map and P2 scientific protocol are committed on `qcgs-objective-audit` at `973d81d`.
- A fresh `gpt-6-luna`/`max` coding context is implementing the instrument on the separate branch `qcgs-exact-objective-code`; the source tree is not yet reviewed.
- Smoke IDs, action chunk size, exact CLI, implementation hashes and run IDs remain P3 launch-lock fields. No inference, CUDA smoke, or scored evaluation has run.
