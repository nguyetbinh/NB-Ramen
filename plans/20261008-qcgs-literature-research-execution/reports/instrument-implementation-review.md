# P3 instrument implementation review

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Case | `CASE-QCGS-01` |
| Protocol | `qcgs-exact-objective-audit-v1.1` (`f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860`) |
| Implementation branch/worktree | `qcgs-exact-objective-code` / `/home/thinkpad/NB-Ramen-qcgs-exact-objective-code` |
| Base revision | `d823bd511da74d838e329d0d2533ea9cec048373` |

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
| A smoke run could freeze a registry without complete trace evidence. | Registry construction requires complete scorer, trace, and scan-prefix receipts, no interrupted scan/trace tail, exactly the registered smoke IDs, finite objective records, and peak allocation below 7.5 GiB. |
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

## Remaining P3 evidence gates

No CUDA scan, CUDA smoke, scored inference, or registry has run. CPU tests do not establish the CLIP/CUDA implementation's baseline parity, view mapping, cache/reset behavior, peak allocation, or cost. The next steps are to commit the reviewed source, run outcome-blind scans and smoke on separately excluded IDs, measure actual wall time/forwards/peak memory, then freeze the launch lock. Stage A remains closed until those receipts and hashes exist.
