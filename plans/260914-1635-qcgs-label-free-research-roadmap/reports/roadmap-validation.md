# Roadmap validation — 2026-09-14

## Scope

Prepared a proposed five-phase QCGS roadmap in response to the user's request for next research steps. This turn adds planning documents only; no QCGS method or experiments have run.

## Evidence checked

- Read the existing oracle runtime contract and completed Pilot B report: supervised query gradients still drive the oracle ranking; improvement is not evidence of a deployable label-free selector.
- Read Ramen forward: per-query entropy gradients already exist before cache retrieval. Reusing them avoids an extra query backward for the single-view proposal, but ranking and temporary workspace still cost compute/memory.
- Read oracle swap ranking: predicted utility uses the change in aggregate SignSGD direction. The roadmap preserves that structure and separates predicted entropy utility from actual supervised utility.
- Read active evidence/Kaggle plans: existing canonical matrix stays independent. There is no new cross-plan blocking dependency; shared code changes must preserve current contracts and artifact provenance.
- Consulted the primary Ramen and TPT abstracts linked in phase 1. A complete related-work/novelty review remains planned, not completed.

## Validation

- `ak plan validate` returned `valid: true`.
- `ak plan status` discovered all five phases, all incomplete, as intended.
- Every local Markdown target resolves; no generated stub placeholders remain.
- Verified existing Ramen test filenames and corrected the proposed regression command.
- Reviewed all phase content for label isolation, development/test separation, self-support and batch visibility, SignSGD aggregation, screened-oracle limitations, and compute accounting.
- The CLI's `add-phase` creates the phase files without expanding its starter overview table. The overview retains that generated table and explicitly links phases 2–5 immediately below it; CLI discovery confirms all five.

## Remaining decisions

Phase 1 must freeze the independent data split, effect/harm criteria, pilot sampling budget and runtime budget before new results are inspected. The proposal recommends one-swap v0; full M-to-k reranking is a later experiment. No efficacy, novelty or completion-time guarantee is made.
