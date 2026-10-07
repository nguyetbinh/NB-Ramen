# P2a instrument and baseline-parity audit

**Status:** completed by read-only source inspection on 2026-10-08. No tests, inference, downloads, or source edits were run by the auditor. Historical raw evidence was absent, so this is not a new parity validation.

## Reuse map

| Component | Reuse |
|---|---|
| `src/methods/Ramen.py` | Cache admission, retrieval distances, support weighting, adaptation update and reset semantics |
| `src/methods/OracleSupportUtilityProbe.py` | `legal_swaps`, aggregate arithmetic, reset-and-update trials, grouped verification and evaluator metrics |
| `src/methods/QueryGradientUtilityProbe.py` | Outcome-blind scan, registry replay, canonical action records, dedicated RNG, baseline replay and exhaustive Stage-A verification |
| `src/methods/MultiViewQueryGradientUtilityProbe.py`, `src/evaluation/query_gradient_multiview.py` | Fixed image-view generation, detached teacher/anchor checks, exact per-action soft-target evaluation and controls |
| `src/runtime/query_gradient_registry.py` | Exclusion-aware earliest-query selection, identity hashing and disjoint stages |
| `src/runtime/query_gradient_campaign.py`, `src/evaluation/query_gradient_utility.py` | Existing provenance, action/RNG audits, gates and summaries; these schemas are tied to the old single-view/MV diagnostics |

## Baseline behavior to preserve

- Compute original-image entropy gradients at pretrained θ₀. Admit the complete evaluator batch before retrieval, including OOD, unscored rows and self-support. Cache pseudo-class comes from θ₀ argmax; cached values are historical gradients.
- Each pseudo-class has capacity 750. Admission priority increases with timestep; when full, replace the minimum only for a strictly greater priority.
- Retrieve the nearest `k=min(5, class_cache_size)` from every nonempty pseudo-class. Weight by `exp(-entropy) * exp(-5*distance)`, sum within each class, add classes in canonical order, then divide by active-class count. Preserve current dtype and arithmetic order.
- Start each candidate pool from baseline top-k, add top-M neighbors not already present, and cap at M=10. Eligibility requires at least one class with ten cached supports; partially populated classes can still contribute legal replacements.
- Enumerate each same-pseudo-class one-swap in class/outgoing-rank/incoming-rank order. Keep support cardinality and retrieval weights; always include no-swap.
- Reset parameters before every trial, clear gradients, preserve original batch positions, install that candidate aggregate, and take one state-free SignSGD step at lr=.01, momentum=0, weight decay=0. A trial does not change cache state; ordinary stream predictions remain Ramen's.

Historical controls differ and cannot be silently conflated: single-view's supervised reference forces a best legal swap at θ₀, while the MV reference at θ_R permits no-swap for nonpositive scores. Random-forced samples a legal swap; random-matched shares that draw but preserves primary abstention. Neither equals expected utility of a random action.

## Leakage and validity risks

- `src/main.py:464` sends evaluator labels, OOD flags and domains into the probe. Existing score helpers receive gradients only and freeze label-free choices before supervised evaluation, but the probe object retains label/domain metadata and candidate records serialize OOD/domain support metadata. New observation and scoring code must not accept those fields or records as inputs.
- The current `historical-exclusions.json` omits the rescue's own 27 distinct Stage-A query IDs. Exclusions also do not mean those examples were absent from the memory stream; report scored identity and stream exposure separately.
- Retrieval uses `torch.topk` without an explicit secondary key for exact-distance ties. Stable action ordering begins only after retrieval membership is fixed.
- Existing grouped ViT verification compares the first two active queries with isolated full-batch trials, not every action for all queries or another backbone.
- Sign-direction deduplication is valid for this state-free SignSGD outcome only. It cannot stand in for evaluation of objectives tied to support identity or observation-channel enrichment.

## Existing checks and uncovered cases

Existing tests that can be reused, subject to execution in P3:

- `tests/test_oracle_support_utility.py`: baseline B=100/k=5/M=10 parity, serial/vectorized arithmetic, grouped/isolated trials, cache/reset/eviction, partial pseudo-class actions and exception cleanup.
- `tests/test_query_gradient_selection.py`: manual scores, tie order, zero/non-finite fallback, FP16 sign handling, RNG, metadata isolation and exhaustive/subset distinction.
- `tests/test_query_gradient_multiview.py`: view order, detached targets, θ₀/θ_R anchors, label isolation, trajectory parity, finite differences and exhaustive records.
- `tests/test_query_gradient_analysis.py`, `tests/test_multiview_analysis.py`: registry tampering, quotas, ties/constant ranks, incomplete raw evidence, locked ledgers and ordered gates.

New known cases are needed for each objective's loss/reward orientation, constant and undefined objective vectors, no-swap/empty-positive behavior, random expectation versus one draw, trial-order permutation and full coverage of each declared observation channel.

## Decision-relevant unknowns

1. Full multi-view exclusion IDs are absent with the raw archive. Current evidence supports a report-backed count of 27, not an exact locally auditable list.
2. No exact-objective protocol, registry or executable schema is frozen. Objective formulas/access, query/action quotas, validity rules, thresholds, multiplicity, uncertainty and budget remain unspecified.
3. Current CUDA parity has not been checked. This local PyTorch/CUDA/device differs from historical T4 evidence.
4. No raw records are present for independent recomputation of historical cache trajectories or result rows.
5. Historical evaluator peak memory was reported near 7.24 GiB, leaving little headroom on the local 8 GiB GPU; exact audit batching and cost remain unknown.

These unknowns do not invalidate historical report claims by themselves. They do prevent calling a new frozen scored audit ready.
