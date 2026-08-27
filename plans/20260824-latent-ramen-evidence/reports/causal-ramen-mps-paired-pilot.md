# CausalRamen paired MPS pilot — 2026-08-27

## Question and scope

This report isolates the experimental role of
[`CausalRamen`](../../../src/methods/SupportAblations.py): a strictly
stream-causal, class-balanced Ramen control with one fixed context and no
latent router. It asks whether the repository's legacy batch-atomic
[`Ramen`](../../../src/methods/Ramen.py) behaves differently from a causal
implementation under otherwise matched benchmark hyperparameters.

This is a paired, cost-limited MPS pilot, not a full benchmark or a
single-factor causal claim. It combines three evidence slices:

1. the existing 8-sample official-wrapper smoke, used only for mechanics;
2. a new three-seed, 64-sample `block` prefix with noncanonical block size 8;
3. a new seed-0, 200-sample `block` prefix with the canonical block size 64.

Raw evidence remains outside git at:

- `/Users/admin/Documents/NB-Ramen/evidence/cifar100c-smoke-mps-block-n8`
- `/Users/admin/Documents/NB-Ramen/evidence/cifar100c-phase03-mps-block-b8-n64-seeds012`
- `/Users/admin/Documents/NB-Ramen/evidence/cifar100c-pilot-mps-block-n200`

No CUDA, DomainNet, full-stream, or publication-level significance claim is
made here.

## What CausalRamen controls

Both CIFAR-100-C benchmark configs use `max_capacity=750`, `topk=5`,
`beta=5.0`, SignSGD, and `lr=0.01`:

- [`cfg/CIFAR100C/Ramen.yaml`](../../../cfg/CIFAR100C/Ramen.yaml)
- [`cfg/CIFAR100C/CausalRamen.yaml`](../../../cfg/CIFAR100C/CausalRamen.yaml)

`CausalRamen` additionally makes the intended contracts explicit with
`capacity_scope=per_class` and `include_current=true`. It retains Ramen's
class-balanced retrieval and entropy/feature-distance weighting, fixes the
context to zero, and processes each sample in stream order: insert current
sample, query, then advance. A later sample in the same evaluator batch is
therefore never visible to an earlier query.

Legacy Ramen intentionally remains the untouched reference implementation.
For a forward batch of size greater than one, it inserts every batch item
before issuing any query. Consequently, its query for sample `i` can use
gradients from samples `i+1...B-1`. With batch size 100, the 64-sample pilot is
one batch, while the 200-sample pilot is two batches. The comparison is thus
scientifically relevant: the two methods do not receive the same online
information set.

This is not, however, a perfect scheduling-only ablation. Legacy Ramen uses
`PriorityCache`, half-precision distance computation on MPS, and its original
top-k behavior. CausalRamen uses `StructuredGradientMemory`, float32 ranking
metadata, stable sorting, and exact causal diagnostics. Differences can
therefore include numeric, kernel, and tie-breaking effects in addition to
future-within-batch visibility. The defensible comparison is **legacy
batch-atomic Ramen versus the causal structured implementation**, not an
estimate of the isolated effect of one scheduling line.

## Mechanics smoke

The previously reported official-wrapper smoke used seed 0, `block`, 8
samples, `cfg/smoke`, explicit MPS, and fast artifact provenance. All ten
methods shared fingerprint
`a3ce48af8cc8995c37ccec4fc1b28ffd17134a57a3571171c10bd25012c6a170`.

| Method | Micro accuracy | Negative adaptation | Retained bytes | Forward ms/sample |
|---|---:|---:|---:|---:|
| NoAdapt | 0.250 | reference | unavailable | 58.088 |
| Ramen | 0.375 | 0.000 | unavailable | 395.071 |
| CausalRamen | 0.375 | 0.000 | 647,360 | 212.943 |

This proves only that construction, forward execution, evidence emission,
NoAdapt pairing, and strict resume work. Eight samples cannot rank methods.

## Three-seed 64-sample paired pilot

### Protocol

- dataset/model: official CIFAR-100-C wrapper and pinned OpenAI CLIP ViT-B/16;
- methods analyzed: NoAdapt, legacy Ramen, CausalRamen, and LatentRamen;
- seeds: 0, 1, 2;
- stream: `block`, noncanonical `stream_block_size=8`;
- prefix: 64 samples;
- evaluator batch size: 100, so the complete prefix is one forward batch;
- config root: benchmark `cfg/`, not `cfg/smoke`;
- device/provenance: explicit MPS and fast artifact verification;
- metric window/stride: 50/50.

Within each seed, all four methods have the exact same exported stream
fingerprint:

```text
seed=0  86acdcf5267636f5be4219b7cb8dd5efc144dcef29b0578640c1ce918bd63d27
seed=1  ee1ff7da7a5c35983e75fa8b681a6b95219c8d8b6aff2b60d01c8ec62ef3a902
seed=2  dbb17f58a23b02ce70a77aa640ccd8c2b6c9390da99c1b0e581c1afdbcd3aaac
```

The CausalRamen config was locked to raw SHA-256
`491d767378054775891e42dbfd0392b77c70dbec0aaf1456cd97192f622ea68b`.
Every adapted trace references the canonical NoAdapt trace for its own seed.

### Per-seed results

| Seed | Method | Micro | Macro domain | Worst domain | Negative-window rate | Forward ms/sample |
|---:|---|---:|---:|---:|---:|---:|
| 0 | NoAdapt | 0.2500 | 0.2292 | 0.000 | reference | 29.26 |
| 0 | Ramen | 0.2656 | 0.2396 | 0.000 | 0.000 | 382.35 |
| 0 | CausalRamen | 0.2656 | 0.2500 | 0.000 | 0.000 | 644.01 |
| 0 | LatentRamen | 0.2656 | 0.2500 | 0.000 | 0.000 | 166.55 |
| 1 | NoAdapt | 0.3438 | 0.3438 | 0.000 | reference | 29.33 |
| 1 | Ramen | 0.2812 | 0.2812 | 0.125 | 1.000 | 433.81 |
| 1 | CausalRamen | 0.3438 | 0.3438 | 0.125 | 0.000 | 212.18 |
| 1 | LatentRamen | 0.3438 | 0.3438 | 0.125 | 0.000 | 180.97 |
| 2 | NoAdapt | 0.4531 | 0.4286 | 0.000 | reference | 29.55 |
| 2 | Ramen | 0.4688 | 0.4375 | 0.125 | 0.000 | 459.54 |
| 2 | CausalRamen | 0.5000 | 0.4643 | 0.000 | 0.000 | 170.01 |
| 2 | LatentRamen | 0.5000 | 0.4643 | 0.000 | 0.000 | 119.24 |

### Aggregate description

Values below are mean ± population standard deviation across the three fixed
seeds. They describe this pilot; three short prefixes are not a confidence
interval for the full benchmark.

| Method | Micro | Macro domain | Worst domain | Negative-window rate | Forward ms/sample |
|---|---:|---:|---:|---:|---:|
| NoAdapt | 0.3490 ± 0.0830 | 0.3338 ± 0.0817 | 0.0000 ± 0.0000 | reference | 29.38 ± 0.12 |
| Ramen | 0.3385 ± 0.0923 | 0.3194 ± 0.0852 | 0.0833 ± 0.0589 | 0.3333 ± 0.4714 | 425.23 ± 32.09 |
| CausalRamen | 0.3698 ± 0.0974 | 0.3527 ± 0.0877 | 0.0417 ± 0.0589 | 0.0000 ± 0.0000 | 342.07 ± 214.20 |
| LatentRamen | 0.3698 ± 0.0974 | 0.3527 ± 0.0877 | 0.0417 ± 0.0589 | 0.0000 ± 0.0000 | 155.59 ± 26.37 |

On this exact paired prefix, CausalRamen is +0.03125 micro and +0.03323 macro
above legacy Ramen, and +0.02083 micro above NoAdapt. Its mean worst-domain
accuracy is 0.04167 below Ramen, so the pilot is not uniformly better across
metrics. All three CausalRamen runs have zero negative windows; legacy Ramen
has one negative window in seed 1.

CausalRamen retained exactly 5,178,880 bytes after 64 samples. Legacy Ramen's
older artifact does not expose comparable logical retained bytes; its sampled
device value must not be presented as a support-memory measurement.

The latency is unstable enough to reject a speed claim: CausalRamen ranges
from 170 to 644 ms/sample across these runs. The first seed is an outlier, and
the aggregate even reverses the ordering observed in the 200-sample pilot.
MPS latency is retained only as descriptive feasibility evidence.

## Canonical-block 200-sample paired pilot

### Protocol and results

This slice uses seed 0, canonical `stream_block_size=64`, 200 samples, batch
size 100, the benchmark configs, MPS, and fast provenance. The stream contains
three complete domain episodes and an 8-sample final episode:

```text
pixelate=64, gaussian_noise=64, glass_blur=64, shot_noise=8
shift_timesteps=[64,128,192]
stream_fingerprint=aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b
```

| Method | Micro | Macro domain | Worst domain | Negative windows | Retained bytes | Forward total | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| NoAdapt | 0.305 | 0.3203 | 0.2344 | reference | unavailable | 6.473 s | 32.37 |
| Ramen | 0.315 | 0.3828 | 0.2812 | 3/4 | unavailable | 204.206 s | 1021.03 |
| CausalRamen | 0.330 | 0.3945 | 0.2812 | 1/4 | 16,184,000 | 372.820 s | 1864.10 |
| LatentRamen | 0.330 | 0.3945 | 0.2812 | 1/4 | 16,184,000 | 340.728 s | 1703.64 |

CausalRamen is +0.015 micro and +0.01172 macro above legacy Ramen, with equal
worst-domain accuracy and two fewer negative windows. It is +0.025 micro,
+0.07422 macro, and +0.04688 worst-domain accuracy above NoAdapt on this one
prefix.

For the two evaluable shifts, CausalRamen recovers 10 samples after the first
shift and does not recover after the second. Legacy Ramen does not recover
after the first and recovers 14 samples after the second. The final shift has
only eight samples and is correctly marked `insufficient_episode`. These two
episodes are descriptive, not a stable recovery estimate.

CausalRamen is 1.83× slower than legacy Ramen and 1.09× slower than
LatentRamen in synchronized forward time on this MPS run. This fails any claim
that the causal implementation is already an efficiency improvement. The
sampled device-memory values are likewise not allocator-exact peaks and are
not comparable to logical retained bytes.

## Relation to LatentRamen

LatentRamen discovered only one context in all four stronger pilot cells: all
three 64-sample seeds and the 200-sample seed-0 run. After projecting each
trace to timestep, sample identity, ground-truth domain/class, prediction,
correctness, inferred context, memory size, and retained bytes, its trace is
identical to CausalRamen in every cell.

This is useful negative evidence. On these bounded block prefixes, the latent
router collapsed to the same fixed-context behavior implemented explicitly by
CausalRamen and produced no change in predictions. The result supports using
CausalRamen as the causal no-routing control; it does not support a benefit
from latent routing on these cells.

## Evidence validation

The exact three-seed 64-sample matrix invocation was rerun with `--resume` and
strictly skipped all six planned NoAdapt/CausalRamen runs. The exact
200-sample invocation was rerun separately and strictly skipped both planned
runs. These checks revalidated current schemas, row counts, stream identity,
config locks, paired NoAdapt references, derived accuracy/recovery/negative
adaptation metrics, efficiency summaries, and artifact provenance without
launching the model.

The CausalRamen traces contain causal memory timelines: size advances from 1
to 64 or 200 and retained bytes advance per item. The generic summary wording
about a repeated post-forward snapshot is stale for this method; the reported
final/max byte values are still directly recomputable from the trace. This
report describes the actual per-item evidence rather than repeating that
generic wording.

## Conclusion and next evidence

The missing dedicated CausalRamen experiment record is now closed at pilot
level. Across the paired evidence available on this MPS host, CausalRamen:

- executes with strict sample causality and validated evidence lifecycle;
- matches or exceeds legacy Ramen in micro/macro accuracy on the tested
  prefixes, but is not uniformly better in worst-domain accuracy;
- avoids the one observed three-seed negative-adaptation event from legacy
  Ramen;
- is not an efficiency win in the longer pilot;
- exactly matches LatentRamen's projected behavior when the router collapses
  to one context.

The next decision-bearing comparison remains paired full-stream Ramen versus
CausalRamen on fixed CUDA for three seeds, CIFAR-100-C and DomainNet, and the
planned `iid_mixed`, `block`, `gradual`, `recurring`, and `imbalanced`
streams. It must preserve exact fingerprints/config locks and report
micro/macro/worst accuracy, negative adaptation, recovery where applicable,
and like-for-like memory/latency evidence. Until then, the findings above are
descriptive pilot evidence rather than a benchmark claim.
