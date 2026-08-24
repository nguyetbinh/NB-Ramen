# Phase 02 — Oracle Routing Diagnostic and LatentRamen-v0

Status: in progress

## Context

- [Research roadmap](../../docs/research/ramen-thesis-research-roadmap.md)
- [Phase 01](phase-01-reproducible-streams-and-evidence.md)
- [LatentRamen-v0](../../src/methods/LatentRamen.py)

## Requirements

- [x] Add a sequential online cosine-prototype router with hard assignments.
- [x] Add a bounded structured gradient memory indexed by `(predicted_class, inferred_context)`.
- [x] Preserve Ramen's entropy, feature-distance, and active-class balancing behavior.
- [x] Make current-sample inclusion an explicit ablation.
- [x] Prevent future items in the same batch from entering a query's support set.
- [x] Emit causal per-sample context, memory-size, and active-context diagnostics.
- [x] Register `LatentRamen` and validate dataset-specific configurations.
- [x] Verify router, memory, aggregation, causality, reset, dtype, and device contracts with focused tests.
- [x] Execute the actual CLIP-backed `LatentRamen.forward` path on MPS.
- [x] Implement and independently review a separately named oracle-domain routing variant.
- [x] Compute NMI, ARI, context purity, discovered contexts, and assignment churn in run summaries.
- [ ] Run the full comparison grid on CIFAR100C and DomainNet.
- [ ] Run and report equal-compute/equal-memory comparisons across methods.
- [ ] Evaluate the Phase 02 go/no-go criteria from the research roadmap.

## Validation

- Run focused Torch unit tests after every router/memory integration change.
- Run the complete suite and syntax compilation after review fixes.
- Exercise the actual CLIP-backed method on CPU or MPS before starting long benchmark runs.
- Use exact stream fingerprints to pair NoAdapt, Ramen, oracle, and LatentRamen traces.

## Risks and rollback

- Batch execution must preserve strict stream causality even if feature gradients are computed together.
- Domain labels must never enter the unsupervised router; the oracle method must be explicit in manifests and method names.
- Per-bucket capacity can increase total memory with the number of contexts, so equal-memory comparisons must report actual retained bytes.
- If oracle routing gives no material recovery on structured streams, stop expanding the router and move to reliability-aware memory.
