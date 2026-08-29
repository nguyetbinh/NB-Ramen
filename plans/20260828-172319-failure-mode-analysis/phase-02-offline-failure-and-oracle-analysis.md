# Phase 2: Offline Failure and Oracle Analysis

Status: completed for the implemented offline evaluator and bounded pilot;
unavailable/insufficient metric states are intentionally preserved in its JSON.

## Context

- Paired `NoAdapt` and adapted evidence traces.
- Failure-analysis trace extension from phase 1.

## Requirements

- Exact F0 outcome decomposition into safe, beneficial, harmful, and unresolved.
- Strict pairing and identity validation.
- Closed-set legal-memory/retrieval oracle metrics with explicitly named oracle definitions.
- Entropy admission influence grouped by entropy and pseudo-label correctness.
- Open-set ID-only aggregate diagnostics: GDC and SDR.
- F4 controlled optimizer/update alternatives with supports/aggregation fixed, consuming preregistered counterfactual predictions produced by `replay_v1`.
- Explicit `computed`, `insufficient`, and `unavailable` states for every oracle family.

## Implementation

1. Add a pure analysis module and CLI producing canonical JSON, refusing unverified sidecars or mismatched run identities.
2. Join evaluator-only labels to support IDs after method execution.
3. Reuse strict paired identity `(timestep, sample_idx, ground_truth_domain, ground_truth_class)` plus manifest, stream fingerprint, config, device, source, and reference bindings; reject reordered/incomplete pairs.
4. Implement all-legal versus retrieved oracle gaps from replayed legal candidates.
5. Implement entropy storage rate, retrieval frequency, total downstream weight, mean retrieved distance, and local gradient sign/cosine grouping.
6. Implement GDC/SDR and fixed-support/fixed-production-aggregate counterfactual analysis for the immutable thresholds `0.50,0.75,1.00`; report each variant separately and a clearly labeled best-of-preregistered evaluator-only oracle upper bound.
7. Implement F5 batch position, future-support count/weight, and atomic-versus-causal paired summaries.

## Validation

- Deterministic fixtures for all outcome categories and edge cases.
- Rejection tests for mismatched fingerprints, IDs, order, missing supports, and non-finite metrics.

## Risks and Rollback

- Oracle ambiguity: report the exact oracle family/version in every output.
- Raw-gradient requirement: expose pure tensor primitives for bounded replay without serializing gradients by default.
- F4 recovery is computed only when the counterfactual artifact is complete, checksum-valid, bound to the same manifest/checkpoint/source/stream/schedule, and contains reset-state verification for every query. Direction-only results remain separate and are never labeled prediction recovery.
