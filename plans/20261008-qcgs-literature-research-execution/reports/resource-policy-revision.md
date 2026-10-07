# CASE-QCGS-01 — active resource-policy revision

**Updated:** 2026-10-08
**Status:** P3 active; Stage A not launched
**Active protocol:** `qcgs-exact-objective-audit-v1.2`, SHA-256 `397e769d2eda50bad9a4744efbd7d928bf06640b9b15b06745ecd45955a5df03`
**Policy config:** [protocol.json](../../../cfg/research/qcgs-exact-objective-audit/protocol.json), SHA-256 `7cb61cee21407103b0220cb20ab8ed33dd5ccf4928a70185e7f88d41e6272c0b`

## Current decision

The user's revised plan removes the 7.5 GiB precautionary allocator cap as a pass/fail or launch gate. Peak allocation remains a reported cost. A complete run with a high finite peak is not a STOP or INCONCLUSIVE result. Actual OOM, crash, interruption, missing quota, or exhausted compute budget remains an execution shortfall. Stage-A budgets remain ≤8 GPU-hours per cell and ≤16 total; Stage B remains ≤24 per cell and ≤48 total if opened. Smoke remains ≤60 minutes total.

The old v1.1 builder rejection is preserved at [registry-build-rejection-20261008.json](/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/decisions/registry-build-rejection-20261008.json) as historical evidence. It was caused only by the old memory threshold. CASE-QCGS-01 is active; no objective-to-utility result has been collected.

## Receipt provenance and registry check

The existing 600-row scans and two completed smoke cells remain untouched v1.1 receipts from source `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93`, protocol SHA-256 `f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860`. They must not be described as runs under v1.2. The new registry records active v1.2 source/protocol separately from scan and smoke source/protocol, uses `peak_memory_policy: report_only`, and stores exact per-cell smoke peak measurements.

A temporary builder check in the v1.2 working tree accepted both high-peak smoke receipts and returned `EXTEND_UNSCORED_SCAN`; it did not create a registry. With 600 scan rows, OOD=0 was short by 8 Stage-A and 51 Stage-B eligible IDs. The frozen doubling rule requests 1,200 rows per cell. The original scans and rejection remain preserved; new 1,200-row scans are required before registry freeze.

## Cost profiling and optimization status

Existing smoke timings show `candidate_action_ms` of about 700 seconds at 485 swaps/query and 1,289 seconds at 850 swaps/query; retrieval is about 62 ms. These aggregate values motivate component profiling but do not identify the slowest component. The v1.2 instrument now records per-batch baseline forward/backward, full-batch action forwards, eight-view forwards, gradient aggregation, candidate enumeration/hashing, record assembly, final row-hash time, JSONL serialization, and allocator peak on cost-only smoke runs. Profile rows remain tied to the excluded smoke IDs and contain no evaluator utility.

A targeted CPU optimization attempt that forwarded only the target query did not meet bitwise full-batch parity and was discarded. The production path still forwards the complete Ramen batch. No CUDA profile or speed optimization has run yet. Any later optimization must be selected from measured component costs and compared on the same workload with baseline/objective values, selections, reset/cache behavior, legal-action coverage, label isolation, runtime and peak allocation. Approximation, action/view reduction, formula changes and altered batch visibility remain outside scope.

The focused exact-objective suite passes 36 tests; syntax compilation, protocol JSON parsing, and `git diff --check` pass. A fresh targeted read-only `gpt-6.1-sol`/low review initially found two profiling issues; both were fixed, and its follow-up review reported no remaining concrete findings. The full repository suite and GPU profile have not been rerun for this policy-only/profiling revision, per the plan.

## Next gated actions

1. Commit the v1.2 protocol, config, registry validation, provenance and profiling changes using the configured global Git identity.
2. Run one profile-only query in each OOD cell on already excluded smoke IDs. Preserve per-batch timing scope and cost receipts.
3. Optimize only a measured bottleneck if an exact, parity-preserving change is justified; otherwise retain the reviewed full-batch path.
4. Run 1,200-row unscored scans in both cells, validate quota and provenance, then build the scientific registry. Freeze source/config/protocol/data/transform hashes and unique Stage-A run IDs in the launch lock.
5. Run Stage A serially on the single GPU within the frozen budgets and classify only from independently audited raw outcomes.
