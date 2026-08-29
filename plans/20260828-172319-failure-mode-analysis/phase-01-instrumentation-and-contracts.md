# Phase 1: Instrumentation, Replay Sidecar, and Contracts

Status: completed. The implemented runtime contract is summarized in
[`docs/research/ramen-failure-mode-analysis-runtime.md`](../../docs/research/ramen-failure-mode-analysis-runtime.md).

## Context

- `src/memory/structured_memory.py`
- `src/methods/LatentRamen.py`
- `src/methods/SupportAblations.py`
- `src/main.py`
- `src/evaluation/evidence.py`
- `src/runtime/experiment_matrix.py`
- New `src/evaluation/failure_analysis_artifacts.py`

## Requirements

- Add an opt-in failure-analysis profile with per-query support IDs, predicted classes, distances, entropies, weights, recencies, support counts, and gradient consensus summaries.
- Compute the unchanged production aggregate and, separately, normalized class-local gradients `h[q,c]` plus pairwise sign/cosine summaries with correct masking and empty/single-class behavior.
- Preserve numerical adaptation behavior when diagnostics are enabled or disabled.
- Keep full raw gradients out of JSON traces.
- Add a bounded `replay_v1` sidecar for exact legal-memory and counterfactual analysis.

## Contract Decisions

- Public option: `failure_analysis_profile = off | trace_v1 | replay_v1`; default `off`. The selected value is recorded in config/manifest identity and may not change during a run.
- Evaluator CLI contract: `src/main.py` and `runtime.experiment_matrix` expose `--failure-analysis-profile`, `--failure-analysis-max-samples`, `--failure-analysis-max-bytes`, and `--failure-counterfactual-thresholds`. Thresholds default to the preregistered tuple `0.50,0.75,1.00`, must be finite unique values in `[0,1]`, and are immutable run identity.
- Trace compatibility: retain trace schema v2. Add one optional nested `failure_analysis` family validated as complete-or-absent per adapted run. `NoAdapt` rows legitimately omit it. Existing v2 evidence remains readable.
- `trace_v1` contains model-visible scalar/support provenance only: query item/timestep identity, support item IDs, predicted classes, distances, entropies, weights, recencies, valid masks/counts, production aggregate summary, normalized class-local consensus summaries, memory occupancy, batch position, future-support count, and future-support weight fraction.
- `replay_v1` adds `failure-analysis/metadata.json`, append-only `items.jsonl`, `queries.jsonl`, `features.bin`, and `gradients.bin`. Each admitted/candidate item vector is written once with byte offset, length, shape, exact torch dtype, predicted class, context, entropy, recency, query timestep, batch position, and admission flag. Each query records legal-candidate IDs before ranking, retrieved IDs/weights, schedule, and current item ID.
- `replay_v1` also performs bounded evaluator-only counterfactual forwards after the actual prediction. For threshold `t`, it holds retrieval and the production aggregate fixed and evaluates `g_t = g_actual * 1[consensus_strength >= t]`. The actual output is captured first; the model is reset to the same pretrained per-sample state before every alternative and again before returning. Counterfactual predictions/entropies are analysis artifacts only and never select or modify the returned prediction.
- The sidecar is bounded by explicit maximum samples/bytes; exceeding either closes the artifact as `insufficient` rather than silently truncating a valid replay.
- `metadata.json` records run ID, profile/schema version, manifest SHA-256, stream fingerprint, source fingerprint, config, dimensions/dtypes, limits, completion status, file sizes, row counts, and SHA-256 for every sidecar file. Readers verify all bindings/checksums before yielding data.
- Memory item IDs are method-local. Sidecar rows also carry producer query timestep, segment/reset index, and evaluator sample identity supplied only by the evaluator after the method returns. Joins never assume `item_id == sample_idx`.

## Implementation

1. Implement replay writer/reader, checksum/identity validation, size limits, and interrupted-artifact states.
2. Add exact legal-candidate snapshot APIs that are read-only and invoked only in analysis profiles.
3. Compute support metadata and diagnostic `h[q,c]` from the exact retrieval result used for adaptation without modifying production aggregation.
4. Add a controlled-update evaluator that reuses the exact aggregate and normalized consensus mask, validates reset state around every counterfactual forward, and exposes predictions only through the opaque replay payload.
5. Thread per-sample summaries and opaque replay payloads through method/evaluator boundaries.
6. Coordinate trace validation in writer, reference verification, completed-run validation, runtime matrix, and fixtures.
7. Add numerical, alignment, serialization, identity, interruption, and no-behavior-change tests.

## Validation

- Focused method, memory, main, and evidence tests.
- CPU synthetic comparison of adapted outputs with profile on versus off.
- Counterfactual parity test proving the returned actual logits are bitwise/numerically identical with replay disabled/enabled and every variant begins/ends at the same reset state.
- Bit-exact round-trip for stored tensor bytes and rejection of changed manifest/stream/checksum.

## Risks and Rollback

- Trace size growth: retain scalar/support metadata only, not raw gradients.
- Replay size growth: opt-in only, bounded limits, append each item vector once, checksum on close.
- Leakage: method payload contains model-visible fields only; ground truth joins happen offline.
- Roll back by disabling the opt-in analysis profile; ordinary trace behavior remains supported.
