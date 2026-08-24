# Phase 04 — Causal Retrieval Profiling

Status: in progress

## Decision basis

- [Research roadmap](../../docs/research/ramen-thesis-research-roadmap.md)
- [Phase 02](phase-02-latent-ramen-v0.md)
- [CIFAR-100-C MPS block n=200 pilot](reports/cifar100c-mps-block-n200-pilot.md)
- [Entropy-gated MPS evidence](reports/entropy-gated-latent-ramen-mps-smoke.md)

The block pilot measured `LatentRamen` at 1.6685 times Ramen's synchronized
forward latency, and the recurrent cost-limited smoke exposed larger MPS
stalls. Those runs prove an end-to-end latency problem, but not that retrieval
or retained-gradient storage causes it. Existing summaries therefore correctly
mark retrieval latency unavailable. Compression is not yet evidence-motivated.

## Preregistered mechanism

Add an opt-in `causal_sync_v1` profiling mode to `LatentRamen`. For every
causal one-item structured-memory query, synchronize the selected device
immediately before and after the query, measure only that interval, and record
the live/eligible/returned support counts. Profiling must not alter routing,
insertion, support selection, aggregation, optimizer behavior, prediction, or
reset semantics.

The profiling run is identified by its resolved config and config hash. Normal
`LatentRamen` configs remain `off`; a dedicated research config root carries
`retrieval_profile: causal_sync_v1`. Original `Ramen` remains untouched.

## Requirements

- [ ] Validate only `off` and `causal_sync_v1` profiling modes.
- [ ] Emit an all-or-none per-sample profiling extension with mode, synchronized
  query milliseconds, live candidates, eligible candidates, returned supports,
  and active classes.
- [ ] Preserve strict item-order causality for batches larger than one.
- [ ] Prove profile-on and profile-off return identical supports, gradients,
  memory timelines, and predictions on deterministic fixtures.
- [ ] Recompute p50, p95, maximum, and total query latency plus candidate/support
  distributions from the trace during strict resume; reject tampering or mixed
  optional rows.
- [ ] Keep existing schema-v2 evidence valid and ordinary retrieval latency
  explicitly unavailable.
- [ ] Run paired profile-off/profile-on CIFAR100C MPS evidence with an identical
  stream fingerprint and verify strict resume.
- [ ] Repeat a cost-limited DomainNet profile on fixed CUDA before selecting any
  compression mechanism.

## Evidence fields

Profiled traces add the complete optional set:

```text
retrieval_profile
retrieval_elapsed_ms
retrieval_candidate_count
retrieval_eligible_candidate_count
retrieval_returned_support_count
retrieval_active_class_count
```

`retrieval_elapsed_ms` excludes feature extraction, logits, backward, routing,
insertion, aggregation, optimizer step, and final prediction. The summary must
state that device synchronization perturbs execution and that profile-on
end-to-end latency is not comparable to ordinary runs.

## Decision gate

Do not implement compression unless fixed-CUDA evidence shows both:

1. retrieval accounts for at least 50% of the profiled adapted-query time; and
2. p95 eligible candidates or p95 query time grows materially with retained
   memory.

If either condition fails, preserve the negative result and profile the next
measured segment. Any later compression comparison must use normal unprofiled
latency, equal retained-byte budgets, identical stream fingerprints, three
seeds, and both CIFAR100C and DomainNet.

## Risks

- Synchronization is intentionally intrusive; it is diagnostic, not an
  optimization benchmark.
- Candidate counters must come only from model-derived memory state and query
  output. Ground-truth class/domain remain evaluator-only.
- MPS results characterize the local path but cannot satisfy the CUDA gate.
