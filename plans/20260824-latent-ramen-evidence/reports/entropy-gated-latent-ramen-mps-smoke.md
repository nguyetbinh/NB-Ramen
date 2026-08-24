# Entropy-Gated LatentRamen MPS smoke — 2026-08-24

## Scope

This report validates the first Phase 3 mechanism with the official
`CIFAR100C` wrapper and real CLIP ViT-B/16 on MPS. It is mechanics and bounded
pilot evidence, not a benchmark result. The preregistered gate is:

```text
admit_to_memory = pre_update_normalized_entropy <= 0.50
```

Ground-truth classes are joined only by the evaluator after the method returns.
They are never provided to routing, admission, retrieval, or adaptation.

The implementation and evidence contract are covered by the current 215-test
dependency-complete suite. Independent review fixed strict-resume validation so a persisted
gate decision must agree exactly with the configured threshold. Original
`src/methods/Ramen.py` remains unchanged.

## Canonical-block n=8 mechanics smoke

Raw evidence:

```text
/Users/admin/Documents/NB-Ramen/evidence/cifar100c-entropy-gate-smoke-mps-block-n8
```

NoAdapt, LatentRamen, and EntropyGatedLatentRamen each emitted eight rows with
the same stream fingerprint:

```text
a3ce48af8cc8995c37ccec4fc1b28ffd17134a57a3571171c10bd25012c6a170
```

| Metric | NoAdapt | LatentRamen | Entropy-gated |
|---|---:|---:|---:|
| Micro accuracy | 0.250 | 0.375 | 0.250 |
| Admitted samples | unavailable | 8/8 | 3/8 |
| Admitted pseudo-label accuracy | unavailable | 0.250 | 0.667 |
| Admitted contamination | unavailable | 0.750 | 0.333 |
| Max retained bytes | unavailable | 647,360 | 242,760 |
| Synchronized forward total | 0.321 s | 1.639 s | 1.572 s |

The gate rejected five high-entropy samples and reduced retained bytes by
62.5%. Its admitted subset was cleaner, but final accuracy fell 12.5
percentage points from ungated LatentRamen and matched NoAdapt. With eight
samples this is only a successful mechanism/evidence smoke and an early
negative accuracy signal.

## Noncanonical genuine-recurrence n=128 smoke

Raw evidence:

```text
/Users/admin/Documents/NB-Ramen/evidence/cifar100c-entropy-gate-smoke-mps-recurring-b8-n128
```

This cost-limited stream explicitly uses `stream_block_size=8`; the run IDs
contain `n128-blk-8`, strict resume checks the same manifest argument and stream
metadata, and all three runs share fingerprint:

```text
6ddff2abe4db91f636a61dea0d8ae9ac394fadcd8cd08f3562a3a522a9091577
```

The prefix contains all 15 CIFAR corruption domains in eight-sample episodes,
then returns to the first domain at timestep 120. It therefore exercises a
real recurrence while remaining explicitly noncanonical and cost-limited.

| Metric | NoAdapt | LatentRamen | Entropy-gated |
|---|---:|---:|---:|
| Micro accuracy | 0.3047 | 0.3594 | 0.3359 |
| Macro domain accuracy | 0.3083 | 0.3542 | 0.3375 |
| Negative windows | reference | 0/2 | 1/2 |
| Admission rate | unavailable | 1.000 | 0.2969 |
| Admitted pseudo-label accuracy | unavailable | 0.3047 | 0.5789 |
| Admitted contamination | unavailable | 0.6953 | 0.4211 |
| Rejected pseudo-label accuracy | unavailable | unavailable | 0.1889 |
| Max retained bytes | unavailable | 9,386,720 | 3,074,960 |
| Forward total | 3.814 s | 688.085 s | 442.371 s |
| Inferred routing contexts | unavailable | 1 | 1 |

The gate retained 38 of 128 samples. Relative to ungated LatentRamen, it
reduced admitted contamination by 27.4 percentage points and retained bytes by
67.2%, while losing 2.34 accuracy points and producing one negative window.
It remained 3.13 points above NoAdapt. The router still collapsed to one
context, so this behavior comes from admission rather than improved routing.

The synchronized MPS timings include severe unified-memory/kernel stalls.
They show that the gate reduced observed total time in this run, but they are
not a portable latency estimate and do not satisfy an equal-compute CUDA gate.
The zero worst-domain values arise from only eight samples per domain and are
not stable worst-domain estimates.

## Integrity and interpretation

Both matrices were rerun with their exact arguments plus `--execute --resume`;
strict validation skipped all three runs in each matrix. The gated validator
requires complete admission evidence on every row, recomputes its summary, and
rejects any decision that differs from `normalized_entropy <= 0.50`.

The mechanism succeeds at selecting a cleaner, smaller memory, but these two
smokes do not support its accuracy hypothesis. The preregistered benchmark
threshold must remain unchanged for the planned three-seed CUDA comparison.
If the accuracy loss persists there, the fixed entropy gate should be rejected
rather than tuned on final test streams.

## Benchmark-config block n=200 follow-up

The preregistered benchmark config was subsequently added to the existing
Phase 2 block pilot root:

```text
/Users/admin/Documents/NB-Ramen/evidence/cifar100c-pilot-mps-block-n200
```

The gated run and the existing NoAdapt, Ramen, and LatentRamen runs share the
same fingerprint:

```text
aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b
```

| Metric | NoAdapt | Ramen | LatentRamen | Entropy-gated |
|---|---:|---:|---:|---:|
| Micro accuracy | 0.305 | 0.315 | 0.330 | 0.315 |
| Macro domain accuracy | 0.3203 | 0.3828 | 0.3945 | 0.3555 |
| Worst-domain accuracy | 0.2344 | 0.2812 | 0.2812 | 0.2344 |
| Negative windows | reference | 3/4 | 1/4 | 2/4 |
| Max retained bytes | unavailable | unavailable | 16,184,000 | 2,427,600 |
| Synchronized forward total | 6.473 s | 204.206 s | 340.728 s | 404.876 s |

The gate admitted 30/200 samples (15%). Its admitted pseudo-label accuracy was
0.7667, admitted contamination was 0.2333, and rejected pseudo-label accuracy
was 0.2235. It reduced retained bytes by 85.0% relative to LatentRamen, but
lost 1.5 micro-accuracy points, lost 3.9 macro-domain points, regressed the
worst-domain result, and doubled negative windows from one to two. Strict
resume recomputed and accepted both the paired NoAdapt run and gated run.

This larger single-seed signal is still not a CUDA benchmark, but it strengthens
the bounded negative conclusion: the fixed entropy gate produces a much
cleaner and smaller cache without preserving LatentRamen's accuracy. The
threshold remains frozen; it must not be tuned on this test stream.
