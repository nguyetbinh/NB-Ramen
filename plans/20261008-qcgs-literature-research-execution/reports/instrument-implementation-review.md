# Historical P3 instrument review — v1.1 source

**Status:** Source/review validation below belongs to the v1.1 implementation. The user-authorized v1.2 resource-policy and profiling review is appended at the end; v1.1 peak rejection is no longer current.

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Case | `CASE-QCGS-01` |
| Protocol | `qcgs-exact-objective-audit-v1.1` (`f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860`) |
| Implementation branch/worktree | `qcgs-exact-objective-code` / `/home/thinkpad/NB-Ramen-qcgs-exact-objective-code` |
| Base revision | `d823bd511da74d838e329d0d2533ea9cec048373` |
| Implementation commit | `b33549ad2e4e35e349a42909f7dbd4e32286ce58` |
| Source tree SHA-256 | `116f180b890309cc7a951c9633298053e4c6efa32d565ce96498d81ae2c37e4e` (SHA-256 of `git archive --format=tar <commit> src`) |
| Config bundle SHA-256 | `f2ba05f37e8c8755d7f76ca6ff2d03ee8a1afb33a7f9c821629ad8c5743fd05d` (canonical path→file-SHA manifest) |

## Review outcome

The software implementation and CPU validation are complete enough to proceed to source commit, unscored scans, and the separate CUDA smoke gate. Independent fresh-context reviewers checked the changed code against the frozen protocol. Findings were fixed with regressions; the last targeted review reported no remaining issue in the final changed areas.

Review dispositions:

| Finding | Disposition |
|---|---|
| A scorer/evaluator flush can precede the ordinary trace; partial final batches were previously misclassified. | The analyzer validates every durable scorer/evaluator record, computes metrics only from complete trace batches, and classifies an incomplete run as `INCONCLUSIVE`. A malformed complete JSONL line is `INVALID`; an unterminated final partial write is preserved in the raw-file hash and reported as incomplete. |
| Durable records can be ahead of the atomically updated status counters. | Counters that lag durable, independently validated files are reconciled and reported. Counters or completion flags that overstate durable records remain `INVALID`. |
| A pre-query Stage-B failure could bypass the Stage-A prerequisite. | Any Stage-B manifest requires a recomputed Stage-A `GO_CONFIRM`; its configured objective must also equal the recomputed winner, even if Stage B has no scorer rows. |
| Unpaired trailing scorer rows did not receive all canonical action checks. | The scorer-only audit checks canonical index/order, action identity, outgoing/incoming support rank identity, aggregate hashes, objective arithmetic, and record hashes before metric filtering. |
| Scan rows after the trace prefix were not checked for continuity. | Every durable scan row is checked for contiguous timestep and valid identity/provenance fields; trace identity is checked over its full preserved prefix. |
| A smoke run could freeze a registry without complete trace evidence. | Under v1.1, registry construction required complete scorer, trace, and scan-prefix receipts, no interrupted scan/trace tail, exactly the registered smoke IDs, finite objective records, and peak allocation below 7.5 GiB. The v1.2 validator preserves completeness checks and reports peak without a numeric threshold. |
| The random-forced index was checked but its utility was trusted. | Raw analysis now recomputes random-forced utility from the selected action CE, validates the no-swap control, and rejects a rehashed tampering regression. |
| Grouped isolation evidence was described more broadly than its per-run sample. | Each scorer record now states its isolation-check scope. The per-run GPU check covers the first action in the first query chunk and at most the first two queries; the CPU regression compares chunked and serial objective outputs for every fixture query/action. Campaign configuration defaults to chunk size 1. |

The formula regression independently computes original-view entropy, eight-view marginal entropy, and caption-bank entropy from the actual tensors passed through every baseline/action call of the real scorer path. It checks the 100-row caption bank against ordinary Ramen batch predictions. This establishes implementation behavior on the deterministic CPU fixture; it does not claim CUDA parity or reconstruct all objective tensors from each campaign JSONL row.

Stage B currently computes the three objective values in the scorer path while persisting only the frozen winner. That extra computation is deferred as the plan directs: Stage B cannot run before `GO_CONFIRM`, and any optimization will be considered against the measured smoke cost without changing winner scores or observation access.

## Validation performed

- Focused exact-objective suite: `PYTHONPATH=src ../NB-Ramen/.venv/bin/python -m unittest discover -s tests -p 'test_qcgs_exact_objective_audit.py'` — 32 tests passed.
- Full repository suite: `PYTHONPATH=src ../NB-Ramen/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` — 440 tests passed, 1 skipped.
- Syntax compilation: `python -m py_compile` on the four new exact-objective source modules — passed.
- `git diff --check` — passed after the objective replay and scope-receipt changes.
- Independent review: fresh `gpt-6.1-sol` / low-reasoning review on the changed raw utility, isolation-scope, and replay-test functions — no remaining findings.

## Memory-feasibility follow-up (2026-10-08)

The first two 600-row scans ran on `d4ab6e04edb7f4bd5bb69b9cb78541ab780cb164`. Both had complete scan/trace provenance, but reported an exact CUDA allocator peak of `13,197,609,472` bytes, above the locked 7.5 GiB cap. A separate first attempt failed before its first batch because `CUBLAS_WORKSPACE_CONFIG` was unset; its `execution-failure.json` is preserved. These runs are feasibility attempts, not registry inputs or scientific results.

The memory review found that `IdentityCache` had inherited `PriorityCache`'s eager full-capacity allocation. The first lazy-allocation draft still constructed 80 eager `PriorityCache` instances inside `Ramen.__init__` before replacing them. A fresh `gpt-6.1-sol`/low review also found that converting an incoming priority to FP16 before comparison changes eviction semantics for values such as `1.0001` against a retained FP16 `1.0`.

All three issues are resolved in commit `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93` (`fix(qcgs): allocate exact audit cache lazily`). `Ramen` now creates caches through an overridable factory; ordinary Ramen still creates the same eager `PriorityCache`, while QCGS constructs the lazy `IdentityCache` directly. The audit cache grows geometrically to the same logical cap of 750, compares incoming priorities before the storage cast, and preserves priority/identity alignment and reset semantics. Regressions cover eager-allocation bypass, dtype-sensitive admission, equal-priority ties, growth to capacity/eviction, tensor parity, and identity reset. A new fresh-context `gpt-6.1-sol`/low read-only review found no remaining issue. The CLI reviewer's environment lacked PyTorch, so its attempted tests did not run; the local test environment ran them successfully.

Validation on this fix: focused exact-objective suite — 35 passed; full repository suite — 443 passed, 1 skipped; `py_compile` on the changed source and `git diff --check` — passed. The Git archive SHA-256 for `src/` is `35af0d707d7702af2990e022d34d858b13004c42a45d5b70190c546da805c511`.

Fresh scans on the fixed revision both completed 600 rows and passed `_scan_run` provenance/continuity checks:

| Cell | Run ID | Config SHA-256 | Peak allocator bytes | Gate |
|---|---|---|---:|---|
| OOD=0 | `scan-ood-0-s1-lazycache-20261008` | `94718d74b782034d0d8e6375e77ab469f4dd84af82c0602d6285cfc307d330a8` | 8,369,873,408 | Above 7.5 GiB |
| OOD=0.5 | `scan-ood-05-s1-lazycache-20261008` | `e365858c9af26db327751dfe064dd58cd5826a62293517ffba10ab272a1e3cb6` | 8,368,574,976 | Above 7.5 GiB |

The outcome-blind smoke registry has SHA-256 `40bd2afacb71482d01ddf65c33672c1a16340fc3fd9650eefdc6c70b5e014a8f`. It selected two separately excluded IDs in each cell. OOD=0 smoke completed 400 rows with two complete scorer records and 485 legal swaps plus no-swap (486 actions/query); peak was `8,358,872,576` bytes and process elapsed time 12:25. OOD=0.5 smoke completed 600 rows with two complete records and 850 legal swaps plus no-swap (851 actions/query); peak was `8,368,574,976` bytes and process elapsed time 22:43. The combined observed smoke time was 35:08, within the 60-minute limit. Both scorer-only audits passed action/provenance checks, both traces are complete, and neither run persisted evaluator outcomes. Baseline replay, cache membership, reset, serial evaluation, ordinary Ramen output, and declared isolation scope are preserved in the receipts.

The v1.1 registry builder audited the scans and both smoke receipts, then rejected scientific-registry construction with `ValueError: smoke must measure allocator peak below the frozen 7.5 GiB cap`. The requested scientific-registry output was not created. These memory readings do not imply anything about objective alignment or utility. Preserve the rejection and all attempts; do not reduce views, actions, quotas, or scientific gates.

## v1.2 resource-policy and profiling update

The active protocol is `qcgs-exact-objective-audit-v1.2`, SHA-256 `397e769d2eda50bad9a4744efbd7d928bf06640b9b15b06745ecd45955a5df03`. The registry records `peak_memory_policy: report_only`, active protocol hash, per-cell peak measurements, and separate scan/smoke source and protocol hashes. Existing v1.1 receipts are reused without editing or relabeling them. A targeted regression builds a valid registry from complete smoke records whose peaks exceed 7.5 GiB and verifies that those exact values remain in provenance.

The policy and profiling changes are committed at `5f486e4ada84fb5b6a1c4794c3a00dc9015654d5`; the `src/` archive SHA-256 is `bd8404099e12d4094166cfe171f1385305fd0cfee22b85f95288de9273a038ef`.

Cost-only profiling now separates baseline forward/backward, full-batch action forwards, view forwards, gradient aggregation, candidate hashing/enumeration, record assembly, final row hash, and JSONL serialization with per-batch query identity. Profiling is restricted in both job generation and method initialization to smoke mode. The first targeted read-only review found those two profiling issues; both were fixed and the follow-up review reported no remaining concrete finding. At review time, the focused exact-objective suite passed 36 tests; py_compile, protocol JSON parsing and `git diff --check` passed, and no CUDA profile had yet run. Later profile receipts and the per-query view-parity fix are recorded in [the active resource-policy report](resource-policy-revision.md).

A v1.2 builder dry run accepted the old high-peak receipts, then requested 1,200-row scans because the 600-row prefix is short by 8 Stage-A and 51 Stage-B eligible IDs at OOD=0. The scientific registry/launch lock has not been created and Stage A has not run. The attempted CPU target-row-only forward optimization was discarded because it did not preserve bitwise equality with the existing full-batch forward; the reviewed full-batch path remains active. Current steps are in [resource-policy revision](resource-policy-revision.md).
