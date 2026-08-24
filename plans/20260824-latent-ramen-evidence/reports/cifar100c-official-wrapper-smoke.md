# Official CIFAR-100-C wrapper smoke — 2026-08-24

## Scope

This report records a cost-limited execution check through the repository's
real `CIFAR100C` wrapper, official OpenAI CLIP ViT-B/16 checkpoint, experiment
matrix, evidence writer, direct NoAdapt pairing, and strict resume validator.
It is mechanics evidence, not a benchmark result: the MPS prefix contains 8
samples and the CPU prefix contains 4 samples, both shorter than the default
64-sample block.

Raw evidence is intentionally gitignored and remains at:

- `/Users/admin/Documents/NB-Ramen/evidence/cifar100c-smoke-mps-block-n8`
- `/Users/admin/Documents/NB-Ramen/evidence/cifar100c-smoke-cpu-block-n4`

## Acquisition and provenance

The official archive was downloaded from the pinned Zenodo record, assembled
without overwriting the partial source, and accepted only after its complete
size and two independent MD5 checks matched:

```text
publisher=Zenodo
record_id=3555552
doi=10.5281/zenodo.3555552
size_bytes=2918473216
expected_md5=11f0ed0f1191edbf9fa23466ae6021d3
actual_md5=11f0ed0f1191edbf9fa23466ae6021d3
```

The tar preflight found one canonical root, 21 regular files, one directory,
no duplicate member names, no links or special files, and no absolute or
parent-traversing paths. After extraction, deep preflight passed all 15 default
corruptions: every array is `(50000, 32, 32, 3)` `uint8`; labels are
`(50000,)` `uint8`, range from 0 to 99, and repeat consistently across all
five severity blocks.

The generated unsigned canonical local inventory was verified once in fast
mode and once in exact mode:

```text
file_count=21
content_algorithm=sha256
root_digest=115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc
sidecar_sha256=1aacb02612a634db5aebaf92332393c4f6f56ab84cbe72a2efb7216c0f0fdbc1
exact_verification=true
```

The run-selected checkpoint also matched the in-code OpenAI trust anchor:

```text
checkpoint=/Users/admin/.cache/clip/ViT-B-16.pt
size_bytes=350837078
sha256=5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
```

The redundant 2.19 GB partial archive and four verified range fragments were
removed only after the complete archive, extraction, deep preflight, and exact
dataset inventory had all passed. The complete official archive and extracted
dataset remain available.

## MPS matrix smoke

The matrix ran all ten methods on the same `block`, seed-0, 8-sample prefix,
using `cfg/smoke`, explicit `--device mps`, and `--artifact-provenance fast`.
NoAdapt was executed first; the remaining nine runs verified and used its
canonical trace. All ten traces contain 8 rows and share stream fingerprint
`a3ce48af8cc8995c37ccec4fc1b28ffd17134a57a3571171c10bd25012c6a170`.

| Method | Accuracy | Negative adaptation | Method-memory max | Forward ms/sample | Sampled MPS bytes |
|---|---:|---:|---:|---:|---:|
| NoAdapt | 0.250 | reference required | unavailable | 58.088 | 594,734,592 |
| Tent | 0.250 | 0.000 | unavailable | 224.266 | 594,710,528 |
| Ramen | 0.375 | 0.000 | unavailable | 395.071 | 444,043,008 |
| CausalRamen | 0.375 | 0.000 | 647,360 | 212.943 | 392,345,600 |
| RandomMemoryRamen | 0.250 | 0.000 | 647,360 | 163.782 | 392,687,360 |
| SameClassRamen | 0.375 | 0.000 | 647,360 | 127.917 | 392,652,544 |
| GlobalNearestRamen | 0.375 | 0.000 | 647,360 | 135.917 | 392,625,920 |
| ContextOnlyRamen | 0.375 | 0.000 | 647,360 | 151.578 | 392,636,672 |
| OracleLatentRamen | 0.375 | 0.000 | 647,360 | 156.200 | 392,424,960 |
| LatentRamen | 0.375 | 0.000 | 647,360 | 167.944 | 392,447,744 |

MPS memory is a synchronized post-batch sample, not an allocator-exact peak.
The latency and accuracy values are retained only to prove field population and
validator recomputation. One tiny single-domain prefix, one accuracy window,
and smoke capacities cannot support method ranking, recovery, or routing
claims.

Rerunning the identical 10-method command with `--resume` strictly validated
and skipped 10/10 runs.

## CPU paired smoke

The CPU smoke ran NoAdapt followed by LatentRamen on the same 4-sample prefix.
Both traces share fingerprint
`d22db1db2a8d25c0901d56a31fa48988c2a43ab1a5866d2913df3d6d079fe00c`.

| Method | Samples | Accuracy | Negative adaptation | Method-memory max | Forward ms/sample | Device memory |
|---|---:|---:|---:|---:|---:|---|
| NoAdapt | 4 | 0.250 | reference required | unavailable | 68.828 | not applicable |
| LatentRamen | 4 | 0.500 | 0.000 | 647,264 | 249.593 | not applicable |

The identical CPU command with `--resume` strictly validated and skipped both
runs.

## Independent validation

A fresh read-only Terra review reconstructed both grids and called the current
strict validator on all artifacts. All 10 MPS and both CPU runs passed; strict
resume skipped all 12 without invoking a runner. The review confirmed every
manifest, stream, trace, summary, and legacy CSV; exact row counts; shared
fingerprints; paired reference paths; recomputed negative-adaptation values;
resolved device evidence; live checkpoint/dataset provenance; and causal
per-sample diagnostics. It found no P0, P1, or P2 evidence-integrity issue.

## Conclusion and boundary

This closes the Phase 1 supported-wrapper smoke criterion: official
CIFAR-100-C bytes, the real wrapper, exact checkpoint path, NoAdapt pairing,
all MPS method entrypoints, CPU LatentRamen, full evidence lifecycle, and
fail-closed resume all executed successfully.

It does not close the Phase 1 baseline matrix or Phase 2 go/no-go criteria.
Those require full CIFAR-100-C and DomainNet streams, three seeds, benchmark
configs/capacities, equal-compute/equal-memory comparisons, and a CUDA runner.
