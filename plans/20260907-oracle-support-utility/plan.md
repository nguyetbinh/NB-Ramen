# Oracle support utility probe

Status: Complete — Pilot A 16 exhaustive queries; both Pilot B cells 128 screened queries each, audited (2026-09-07).

Source: user-provided `oracle_support_utility_small_experiment_report.md`.

1. Implement evaluator-only one-swap probe and isolated output contract.
2. Verify hook isolation, baseline parity, legal swaps, exact state reset and summaries.
3. Execute official-data Pilot A (16 ID queries, exhaustive, OOD=.5), then Pilot B (128 ID queries each at OOD=0/.5, screened), subject to measured runtime.
4. Record measured evidence and remaining limitations.

Acceptance: preserve legacy Ramen batch=100/cache/weights/SignSGD; k=5,m=10; no canonical-matrix changes; no supervised labels in cache; report exhaustive versus screened headroom separately and include no-swap fallback. OOD queries are never scored. Save JSONL query rows, summary and provenance through a separate diagnostic sidecar.

Decisions: supervised reference gradient is at pretrained parameters. Pilot B verifies best/worst/random plus gradient-cosine baseline; similarity baseline retains Ramen. Eligibility requires one class with at least m entries; other classes may contribute any available ranks k+1 through m. Real full-batch forwards preserve batch semantics for every temporary update. No numerical GO threshold is invented for the qualitative research criteria.


## Results

- Implementation and verification complete: 367 tests, 362 passed and 5 skipped.
- Pilot A complete: first 16 eligible ID queries at timesteps 100–126; 110 legal swaps/query, all 1,760 verified.
- Mean/median CE gain: 0.214335 / 0.173319; headroom 16/16; accuracy remains 8/16.
- Pilot B complete: 128 queries/cell, both audits PASS. OOD=0: gain 0.046677, headroom 114/128, correct 70→72. OOD=.5: gain 0.331788, headroom 120/128, correct 62→64.
- GO for investigating label-free utility prediction; no causal OOD claim or deployable selector claim. See [Pilot B report](reports/pilot-b-completed.md).
- The diagnostic target is complete. The generic 600-sample evaluator was stopped after target completion; it has 200 complete trace rows and is not a full 600-sample benchmark.

See [Pilot A report](reports/pilot-a-completed.md), [artifact audit](reports/pilot-a-audit.json),
[grouped verification review](reports/grouped-verification-review.md),
[runtime contract](../../docs/research/oracle-support-utility-probe.md), and
[previous bounded attempt](reports/execution-status.md).
