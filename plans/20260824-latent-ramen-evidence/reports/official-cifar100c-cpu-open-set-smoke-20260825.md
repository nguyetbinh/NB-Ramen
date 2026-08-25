# Official CIFAR-100-C CPU open-set smoke

## Status

Completed noncanonical CPU decision signal, not a benchmark result. The
official Zenodo CIFAR-100-C archive was MD5-verified and its extracted 21-file
inventory was exact-SHA-256 verified in a separate pre-run command. Run
manifests record fast provenance verification (not a second per-file rehash).

## Protocol

- Dataset: official CIFAR-100-C, fixed `open-set-cifar100-split-v1`.
- Stream: block; requested OOD ratio 0.50; seed 0; 64 samples; batch size 16.
- Device: CPU; CLIP ViT-B/16; artifact provenance `fast` at run start.
- Paired stream fingerprint:
  `fb6d1e2f81bb037f48bbecbf3e82ed536f319bfd0ef6e41bae53fc876169130b`.

The realized stream has 36 ID and 28 OOD samples (OOD ratio 0.4375). Evidence
is in `evidence/official-cifar100c-cpu-smoke/`.

## Three-seed results

Every method was paired to its same-seed NoAdapt trace. ID accuracy below is
the evaluator-only ID metric; each 64-sample prefix has a different realized
ID count, so the percentages should not be pooled as samples.

The original seed-0 64-sample Ramen attempt is intentionally excluded: it
stopped before producing a summary due to the CPU-half `cdist` defect. The
table uses the successful `-r1` retry after that defect was fixed; it has the
same paired fingerprint as seed-0 NoAdapt.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD | Δ vs NoAdapt mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoAdapt | 44.44% | 66.67% | 20.51% | 43.87 ± 23.08pp | — |
| Ramen | 52.78% | 66.67% | 23.08% | 47.51 ± 22.27pp | +3.64pp |
| OracleIDGradientRamen | 52.78% | 66.67% | 35.90% | 51.78 ± 15.41pp | +7.91pp |
| ConsensusRamen | 52.78% | 69.70% | 28.21% | 50.23 ± 20.86pp | +6.36pp |

Seed 0's oracle-vs-Ramen directional diagnostics average GDC 0.1977,
coordinate sign disagreement 0.2045, and retrieved OOD weight fraction
0.4570. Consensus applied on all 64 seed-0 samples; its mean agreement was
0.2613, retained-coordinate mask rate 0.5470, and mean active-class count
24.25.

## Larger paired prefix

A second seed-0 CPU prefix uses the same stream parameters but 200 samples
(13 batches of 16), with a separate paired fingerprint and its own NoAdapt
reference. It reduces prefix variance but remains noncanonical and is not
combined with the 64-sample results.

| Method | ID accuracy |
| --- | ---: |
| NoAdapt | 49.02% |
| Ramen | 51.96% |
| OracleIDGradientRamen | 53.92% |
| ConsensusRamen | 53.92% |

Consensus again reaches the oracle result in this one larger cell. It applied
on all 200 samples; mean agreement was 0.2206, mask rate 0.4675, and mean
active-class count 43.0. Its final retained support memory was 16.18 MB;
OracleID retained 32.36 MB. The CPU sequence also exposes unfavorable scaling:
the Consensus run used several CPU minutes at 200 samples, so CPU is useful
for correctness and decision signals but not a substitute for CUDA profiling.

## Decision and limits

This supports continuing the Oracle/Consensus direction: the three-seed mean
shows both Oracle and label-free Consensus above NoAdapt and Ramen, while the
seed-0 diagnostic establishes a concrete all-vs-ID gradient difference with
substantial retrieved OOD weight. It does **not** establish a general
advantage: each seed is a different 64-sample CPU prefix in only the block
stream at one OOD ratio, and dispersion is large. The remaining requirement is
the preregistered multi-seed CUDA matrix, with DomainNet as secondary evidence.

## Runtime correction

The first CPU Ramen attempt exposed `torch.cdist` lacking CPU float16 support.
`PriorityCache.query` now promotes only CPU-half operands for the distance
calculation; cache storage and CUDA/MPS paths are unchanged. The focused
regression suite (CPU-half retrieval, memory diagnostics, and Consensus) passed
20 tests before the successful rerun.
