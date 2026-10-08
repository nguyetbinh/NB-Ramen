# CASE-QCGS-01 — active resource-policy revision

**Updated:** 2026-10-08
**Status:** P3 active; Stage A not launched
**Active protocol:** `qcgs-exact-objective-audit-v1.2`, SHA-256 `397e769d2eda50bad9a4744efbd7d928bf06640b9b15b06745ecd45955a5df03`
**Policy config:** [protocol.json](../../../cfg/research/qcgs-exact-objective-audit/protocol.json), SHA-256 `7cb61cee21407103b0220cb20ab8ed33dd5ccf4928a70185e7f88d41e6272c0b`
**Reviewed policy commit:** `5f486e4ada84fb5b6a1c4794c3a00dc9015654d5`; `src/` archive SHA-256 `bd8404099e12d4094166cfe171f1385305fd0cfee22b85f95288de9273a038ef`

## Current decision

The user's revised plan removes the 7.5 GiB precautionary allocator cap as a pass/fail or launch gate. Peak allocation remains a reported cost. A complete run with a high finite peak is not a STOP or INCONCLUSIVE result. Actual OOM, crash, interruption, missing quota, or exhausted compute budget remains an execution shortfall. Stage-A budgets remain ≤8 GPU-hours per cell and ≤16 total; Stage B remains ≤24 per cell and ≤48 total if opened. Smoke remains ≤60 minutes total.

The old v1.1 builder rejection is preserved at [registry-build-rejection-20261008.json](/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/decisions/registry-build-rejection-20261008.json) as historical evidence. It was caused only by the old memory threshold. CASE-QCGS-01 is active; no objective-to-utility result has been collected.

## Receipt provenance and registry check

The existing 600-row scans and two completed smoke cells remain untouched v1.1 receipts from source `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93`, protocol SHA-256 `f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860`. They must not be described as runs under v1.2. The new registry records active v1.2 source/protocol separately from scan and smoke source/protocol, uses `peak_memory_policy: report_only`, and stores exact per-cell smoke peak measurements.

A temporary builder check in the v1.2 working tree accepted both high-peak smoke receipts and returned `EXTEND_UNSCORED_SCAN`; it did not create a registry. With 600 scan rows, OOD=0 was short by 8 Stage-A and 51 Stage-B eligible IDs. The frozen doubling rule requests 1,200 rows per cell. The original scans and rejection remain preserved; new 1,200-row scans are required before registry freeze.

## Cost profiling and optimization status

Existing smoke timings show `candidate_action_ms` of about 700 seconds at 485 swaps/query and 1,289 seconds at 850 swaps/query; retrieval is about 62 ms. These aggregate values motivate component profiling but do not identify the slowest component. The v1.2 instrument now records per-batch baseline forward/backward, full-batch action forwards, eight-view forwards, gradient aggregation, candidate enumeration/hashing, record assembly, final row-hash time, JSONL serialization, and allocator peak on cost-only smoke runs. Profile rows remain tied to the excluded smoke IDs and contain no evaluator utility.

The first corrected profile pair (`r1`, source `f06094431ebe47258fcb8d4be4a1cdee074caf8f`) exposed batch-shape-dependent baseline MEMO values. The historical v1.1 smokes scored two queries together, while profile-only scored one; baseline `_view_logits` packed scheduled queries but candidate trials used action chunks of one. Candidate pools, hashes, and every candidate objective loss matched v1.1 bitwise, as did baseline CLIP logits and the single-view and CLIPTTA objectives. Baseline `memo_marginal_entropy` differed by `-0.0009608269` (OOD=0) and `+0.0000799298` (OOD=0.5), while selected indices stayed the same. These old values remain historical and are not a parity oracle for the v1.2 one-query profile.

The fix is committed at `9206f08f5d66229a9d1d526667ed8ed9c03e8a65`: `_view_logits` now runs each query's eight views in a separate forward for both baseline and candidate evaluation. The focused exact-objective suite passes 37 tests, including a regression that checks the per-query forward shape and exact output equality across a multi-query fixture. No change was made to actions, views, formulas, full-batch action forwards, or memory policy.

Two `r2` startup attempts failed before stream scoring because the launcher omitted `CUBLAS_WORKSPACE_CONFIG`; PyTorch deterministic algorithms rejected CLIP text-encoder CuBLAS. Their manifests/logs are preserved in [the startup-failure decision](/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/decisions/profile-cublas-workspace-config-missing-20261008.json). They contain no scan/action/status/profile/outcome records and are not profile evidence. The final `r3` pair used `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0`, as recorded in the [CUDA reproduction notes](cuda-run-reproduction.md). Both runs use the clean committed source above and active protocol SHA-256 `397e769d2eda50bad9a4744efbd7d928bf06640b9b15b06745ecd45955a5df03`.

| Cell and profile query | Legal swaps | Candidate scorer wall time | Full-batch action GPU events | Candidate view GPU events | Peak allocation |
|---|---:|---:|---:|---:|---:|
| OOD=0, `profile-ood-0-s0-v12r3-20261008`, sample 4562 at timestep 358 | 485 | 315,741 ms | 285,523 ms | 36,601 ms | 8,358,872,576 bytes |
| OOD=0.5, `profile-ood-05-s0-v12r3-20261008`, sample 5939 at timestep 595 | 850 | 601,279 ms | 543,253 ms | 71,033 ms | 8,368,574,976 bytes |

Both r3 runs completed their requested query, passed scorer and provenance audits, and contain no evaluator outcomes. Their scoring fields exactly match the r1 profile runs under the same one-query schedule; component buckets may overlap, so do not sum them as wall time. Locked YAML SHA-256 values are `ee624b4a815366c4059b82e19efb630f0cb18127bc8b06fe218fb697ac5edbd8` (OOD=0) and `6669ef3a8321e0e912c077f54e9ab840edbbd2933f04930376ef9c70b52aad92` (OOD=0.5). Full raw hashes and status are in [the profile component decision](/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/decisions/profile-component-decision-20261008.json); raw runs are under `/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/runs/profile-only-v12-final-r3/`, with logs under `/home/thinkpad/NB-Ramen/log/qcgs-exact-objective-audit/profile-only-v12-final-r3/`.

Full-batch candidate action forwards are the largest measured component, but the prior target-query-only optimization failed bitwise full-batch parity. No safe speed optimization is selected; the production action path stays full-batch. A conservative linear forecast scales each profile's observed per-swap wall time to the theoretical maximum 2,000 legal swaps/query (80 classes × 25 swaps), 16 queries/cell, and a 20% reserve: 6.94 GPU-hours at OOD=0 and 7.55 at OOD=0.5, 14.49 total. This forecast is not a measured Stage-A guarantee; it fits the frozen 8-hour/cell and 16-hour total budgets, which will be checked against actual Stage-A runtime. Peak allocation remains report-only.

## Next gated actions

1. Run 1,200-row unscored scans in both cells under the committed source and deterministic CUDA environment; preserve earlier prefixes and startup failures.
2. Validate scan quota/provenance, build the scientific registry with explicit v1.1 receipt provenance, and freeze hashes plus unique Stage-A IDs in the launch lock.
3. Run Stage A serially on the single GPU within the frozen budgets and classify only from independently audited raw outcomes.
