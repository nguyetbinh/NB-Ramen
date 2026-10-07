# P2 protocol consistency review

**Reviewed:** 2026-10-08

**Reviewer context:** fresh `gpt-6.1-sol`, reasoning `low`, read-only Codex CLI worktree on `qcgs-objective-audit`

**Result:** no blocking equation/sign error; four concrete ambiguities were corrected in protocol v1.1 before any scored run.

## Findings and disposition

| Finding | Protocol correction |
|---|---|
| Stage-B evaluator access could be read as permitting general all-action or supervised-reference outputs. | Explicitly restrict oracle, exact expected random, supervised first-order reference, per-action utility, ranking and sign metrics to Stage A. Stage B may evaluate selected action, no-swap and one random draw only; derive random-matched from the frozen abstention decision. |
| Bootstrap p-values and Holm-adjusted intervals were underspecified. | Freeze the stream-block resampling unit, query-weighted treatment of unequal blocks, null-centering, one-sided tail/count formula, 10,000 draws/seed, Holm step-down/adjusted-p formula and descriptive status of percentile effect intervals. |
| CLIPTTA pseudo-caption state was ambiguous under Ramen by-sample parameters. | Define one fixed caption bank per current stream batch using each row's own ordinary Ramen output/parameter slot; candidate query changes only its own visual row and leaves other rows and the bank fixed. |
| Stage-A gate failure did not distinguish scientific STOP from insufficient evidence. | Add ordered `INVALID → INCONCLUSIVE → valid scientific STOP / GO_CONFIRM` dispositions. Constant objective scores or fewer than 8 defined rho values are valid no-information for that objective; missing/non-finite values are INVALID; quota/resource or fewer than four scored blocks is INCONCLUSIVE. |

The reviewer found no further decision-relevant mismatch in objective directions, update/reset/cache parity, random/no-swap semantics, registry separation or bounded claims. Its source check confirmed the CLIPTTA image-to-text row entropy mapping; that is source verification for the formula, not validation of local implementation or empirical result.

The review was read-only: no repository edits, tests or inference were performed by the reviewer. The revised protocol SHA-256 and launch eligibility are recorded in the updated [case readiness card](case-execution-readiness.md). The separate required fresh code review remains pending until the coder produces a committed diff.
