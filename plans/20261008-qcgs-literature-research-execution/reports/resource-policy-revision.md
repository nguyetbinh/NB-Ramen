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

The first corrected v1.2 profile pair completed on source `f06094431ebe47258fcb8d4be4a1cdee074caf8f`, protocol SHA-256 `397e769d2eda50bad9a4744efbd7d928bf06640b9b15b06745ecd45955a5df03`. OOD=0 used `profile-ood-0-s0-v12r1-20261008` (config SHA-256 `68ae21e62ea799269e903a89e5d89014e261134f291ab08c0179710a88f2ac45`) at query `(timestep=358, sample_idx=4562)`, with 485 legal swaps. OOD=0.5 used `profile-ood-05-s0-v12r1-20261008` (config SHA-256 `643fda47be89d1586b18242b1fc1e7b5705dd4c63e12b4893f66c36047bdedbc`) at `(595, 5939)`, with 850 legal swaps. Both were complete one-query smoke runs, passed scorer and provenance audits, and contain no evaluator outcomes. Raw run directories are under `/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/runs/profile-only-v12-correct/`; logs are under `/home/thinkpad/NB-Ramen/log/qcgs-exact-objective-audit/profile-only-v12-correct/`.

The measured GPU event buckets identify full-batch candidate action forwards as the largest component: 298,545.6 ms at OOD=0 and 541,871.4 ms at OOD=0.5. Eight-view candidate forwards took 44,657.3 ms and 79,017.5 ms; gradient aggregation took 6,455.3 ms and 13,101.2 ms. The component buckets can overlap, so they are not summed as wall time. Peak allocations were 8,358,872,576 and 8,368,574,976 bytes; they remain reported costs under the active policy. CPU forward-only optimization was already discarded for failing bitwise full-batch parity. No measured, parity-preserving speed optimization is justified, so the production full-batch action path remains unchanged.

The profile-only records matched the historical v1.1 smoke candidate pools, hashes, and every candidate objective loss bitwise. Baseline CLIP logits and the single-view and CLIPTTA objectives also matched. The baseline `memo_marginal_entropy` differed by `-0.0009608269` at OOD=0 and `+0.0000799298` at OOD=0.5; the selected action indices stayed the same. The profile-only schedule scored one query while the historical smoke scored two, and `_view_logits` packed all scheduled queries together for the baseline but candidate trials used action chunks of size one. This is consistent with shape-dependent CUDA rounding in the baseline eight-view forward. It means the baseline objective could depend on neighboring scheduled queries, so the profile is accepted as cost-only evidence but its score is not treated as a scientific result.

The working-tree fix makes `_view_logits` run one query's eight views per forward for both baseline and candidate evaluation. A new CPU regression checks this batch shape, and the focused exact-objective suite passes 37 tests. The corrected profiles must be rerun under the committed fix before final cost receipts and the 1,200-row scans. Any future speed optimization must be selected from measured component costs and compared on the same workload with baseline/objective values, selections, reset/cache behavior, legal-action coverage, label isolation, runtime, and peak allocation. Approximation, action/view reduction, formula changes, and altered batch visibility remain outside scope.

The policy/profile changes are committed at `5f486e4ada84fb5b6a1c4794c3a00dc9015654d5`. The focused exact-objective suite passes 36 tests; syntax compilation, protocol JSON parsing, and `git diff --check` pass. A fresh targeted read-only `gpt-6.1-sol`/low review initially found two profiling issues; both were fixed, and its follow-up review reported no remaining concrete findings. The full repository suite and GPU profile have not been rerun for this policy-only/profiling revision, per the plan.

## Next gated actions

1. Commit the per-query baseline/candidate view-forward fix, then rerun one profile-only query in each OOD cell on the already excluded smoke IDs and preserve per-batch timing scope.
2. Retain the full-batch action path; the measured profile gives no exact speed optimization to apply. Keep peak allocation report-only.
3. Run 1,200-row unscored scans in both cells, validate quota and provenance, then build the scientific registry. Freeze source/config/protocol/data/transform hashes and unique Stage-A run IDs in the launch lock.
4. Run Stage A serially on the single GPU within the frozen budgets and classify only from independently audited raw outcomes.
