# CIFAR-100-C MPS causal retrieval profile — 2026-08-24

## Scope

This is a Phase 04 instrumentation pilot, not an accuracy benchmark. It uses
official CIFAR-100-C, CLIP ViT-B/16, MPS, seed 0, a cost-limited 32-sample
`recurring` prefix, and explicit `stream_block_size=8`. Profile-off and
`causal_sync_v1` runs use the same LatentRamen benchmark hyperparameters; the
profile config differs only by the explicit profiling key.

Raw evidence:

```text
/Users/admin/Documents/NB-Ramen/evidence/cifar100c-phase04-profile-off-mps-recurring-b8-n32
/Users/admin/Documents/NB-Ramen/evidence/cifar100c-phase04-profile-on-mps-recurring-b8-n32
```

Both adapted runs share stream fingerprint:

```text
e4c2efef13afe75d3b7ad8014ad9f2560d57b080b8b7c78e99bb007c33357ed7
```

Each matrix was rerun with identical arguments plus `--execute --resume`;
strict validation skipped both NoAdapt and LatentRamen artifacts in both roots.

## Instrumentation integrity

The implementation passed 215 tests, `compileall`, `git diff --check`, and an
independent review. A real one-item MPS helper run also verified that elapsed
timestamps remain CPU float64 instead of attempting unsupported MPS float64
allocation. Strict resume causally replays class/context buckets, FIFO
eviction, current-item inclusion, returned supports, and active classes. It
rejects inconsistent counters, memory sizes, admission rows, or recomputed
summary values.

Profile-off and profile-on traces were identical for:

- timestep, sample identity, domain, and ground-truth class;
- prediction, correctness, and predicted entropy;
- inferred context and active-context timeline;
- memory size and retained-byte timeline; and
- admission prediction, normalized entropy, and decision.

Deterministic unit fixtures additionally prove identical retrieved supports,
aggregated gradients, and memory behavior. Original `src/methods/Ramen.py`
remains unchanged.

## Measurements

| Metric | Profile off | `causal_sync_v1` |
|---|---:|---:|
| Samples | 32 | 32 |
| Micro accuracy | 0.4375 | 0.4375 |
| Macro domain accuracy | 0.4375 | 0.4375 |
| Worst-domain accuracy | 0.2500 | 0.2500 |
| Max retained bytes | 2,589,440 | 2,589,440 |
| Synchronized forward total | 3,640.9 ms | 4,523.6 ms |
| Profiled retrieval total | unavailable | 547.3 ms |
| Retrieval share of profiled forward | unavailable | 12.1% |
| Retrieval p50 / p95 / max | unavailable | 13.5 / 35.6 / 49.1 ms |
| Eligible candidates p50 / p95 / max | unavailable | 16.5 / 30.45 / 32 |
| Returned supports p50 / p95 / max | unavailable | 16.5 / 30.45 / 32 |
| Active classes p50 / p95 / max | unavailable | 14 / 23 / 23 |

The cache was still small enough that every eligible item was returned. Query
time increased with candidate growth: median query time was 6.84 ms in the
first eight samples and 21.24 ms in the final eight, with exploratory Pearson
correlation 0.607 between eligible count and query milliseconds. The first
query was a 34.6 ms cold-start outlier, so these growth figures are descriptive
only.

## Decision

This local pilot fails the preregistered first compression gate: retrieval was
only 12.1% of profiled forward time, far below the required 50%. Candidate
growth does affect query time, but both conditions are required. Therefore no
gradient compression mechanism is justified from this evidence.

The profile synchronizes every query and perturbs execution. Its profile-on
forward time must not be compared as ordinary method latency, and MPS timing is
not portable to CUDA. Phase 04 remains open for a cost-limited DomainNet CUDA
profile; compression stays deferred unless that fixed-hardware run satisfies
both preregistered gates.
