# Open-set MPS directional pilot — 2026-08-25

## Decision signal

**Conditional go for a ConsensusRamen-v0 implementation.** This is not
benchmark or thesis evidence. It is a small, deliberately non-canonical,
paired MPS pilot that establishes the prerequisite directional signal: OOD
supports materially change Ramen's update direction and an ID-gradient oracle
is non-worse than Ramen on every tested stream, improving mean ID accuracy.

The next claim-bearing step remains the report's canonical CIFAR-100-C CUDA
paired experiment. Do not cite these pilot numbers as CIFAR-100-C results.

## Protocol

- Device: Apple M2 MPS; model: CLIP ViT-B/32.
- Open-set split: `open-set-cifar100-split-v1` (80 known / 20 unknown).
- OOD ratio: 0.50, selected exactly per domain.
- Stream: `block`, block size 8, prefix 128 samples, batch size 8.
- Seeds: 0, 1, 2. Every method pair for a seed had the same stream
  fingerprint.
- Methods: NoAdapt, Ramen, OracleIDGradientRamen, and OracleDropOODRamen.
- Artifact: `scripts/build-cifar100c-pilot.py --source hf-rows
  --samples-per-severity 64 --seed 20260825` generated a loader-compatible
  local artifact from real CIFAR-100 test images. Its `README.json` records
  that its transformations are lightweight approximations, **not** official
  CIFAR-100-C. Runs therefore used `--artifact-provenance off`.

Evidence directories are under
`evidence/open-set-mps-pilot/`; each contains manifest, stream, trace, and
summary JSON.

## Paired results

| Seed | Fingerprint | NoAdapt ID ACC | Ramen ID ACC | OracleID ID ACC | OracleDrop ID ACC | OracleID − Ramen |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `172f4d9bb640d4c7382653f51363b87ff4850721836f2fe113e1873149f596bf` | 16.13% | 17.74% | 19.35% | 19.35% | +1.61 pp |
| 1 | `bb308ff2bca00adb0679cf0da92ad34140a2f9b05408c9449775ed562235e676` | 18.75% | 23.44% | 25.00% | 25.00% | +1.56 pp |
| 2 | `770253cf803616b29370bbdc4d11f1b77af6ffef3db8824066c6d0f9d0721277` | 20.63% | 25.40% | 25.40% | 25.40% | +0.00 pp |
| Mean | — | 18.50% | 22.19% | 23.25% | 23.25% | **+1.06 pp** |

Ramen exceeded NoAdapt by +1.61, +4.69, and +4.76 pp. OracleID exceeded or
matched Ramen in every seed. OracleDrop exactly matched OracleID in this tiny
pilot, so retaining OOD embeddings did not show an additional benefit here.

## Contamination diagnostics (OracleID)

| Seed | GDC (1 − cosine) | Sign disagreement | Retrieved OOD fraction | Retrieved OOD weight fraction |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.1903 | 0.1977 | 0.4283 | 0.4282 |
| 1 | 0.2026 | 0.2098 | 0.4808 | 0.4111 |
| 2 | 0.1442 | 0.1727 | 0.5208 | 0.3775 |

All 128 directions were defined in every oracle run. These are nontrivial
direction changes accompanied by substantial retrieved OOD weight; they are
the mechanism signal motivating a consensus experiment.

AUROC/FPR95 remain descriptive only in this pilot: they use the fixed
pre-adaptation score and are therefore the same across paired methods for a
given stream. No negative-adaptation rate was computed because non-canonical
pilot inputs cannot meet the repository's verified-provenance reference-trace
contract.

## Runtime and repair discovered by the pilot

The 128-sample Ramen and oracle runs completed on MPS (approximately four to
six seconds in the adaptation loop after model initialization). The
OracleDropOOD control initially exposed an MPS numerical defect: a cosine of
an identical direction could round just above 1 and be rejected by the strict
trace schema. `_direction_diagnostics` now computes in fp32, treats zero or
non-finite directions as undefined, and clamps finite cosine values to
`[-1, 1]`. Unit tests cover empty support, fp16 reduction overflow, and the
schema-safe unit-cosine case; the rerun completed for all three seeds.

## Limits and next gate

This artifact has only 64 source examples per severity/domain, generated
transformations, a 128-sample prefix, one stream mode, MPS rather than CUDA,
and unverified provenance. It cannot establish effect size, generalisation,
or publishable open-set performance.

Before reporting a claim-bearing Consensus result, run the preregistered canonical matrix on
official CIFAR-100-C with Linux NVIDIA CUDA: OOD 0/10/30/50%, iid/block/
recurring streams, at least three seeds, and exact paired fingerprints. Then
compare Ramen, OracleIDGradientRamen, OracleDropOODRamen, and ConsensusRamen
against the same NoAdapt traces.

## ConsensusRamen-v0 follow-up

The Phase-C implementation retained Ramen's unnormalised per-class weighted
contribution and batch-atomic retrieval, then applied the hard coordinate mask
after the ordinary class-balanced aggregate. This compatibility choice is
intentional: the report's normalised `h[q,c]` equation conflicts with its
requirement to preserve Ramen aggregation, so the latter takes precedence.

The suggested `tau=0.6` setting was evaluated only on this isolated pilot. It
retained 3.7--4.9% of coordinates and was harmful:

| Method / pilot-only setting | Mean ID ACC | Mean mask rate |
| --- | ---: | ---: |
| Ramen | 22.19% | — |
| ConsensusRamen `tau=0.6` | 19.56% | 4.4% |
| ConsensusRamen `tau=0.2` | **24.85%** | 45.3% |

At `tau=0.2`, ID-accuracy differences from Ramen were +4.84, +1.56, and
+1.59 pp for seeds 0--2. This ablation was run with a separate config under
`plans/20260824-latent-ramen-evidence/pilot-configs/`, before any canonical
stream was evaluated. The canonical `cfg/CIFAR100C/ConsensusRamen.yaml` is
therefore locked to `tau=0.2`, `min_consensus_classes=3`, and `hard_mask` for
the upcoming canonical matrix; `tau=0.6` remains a documented negative
ablation.

The MPS wall-clock comparison is descriptive only because first-run MPS
compilation/initialisation varied: mean synchronized forward cost was 32.9 ms
per sample for Ramen and 38.8 ms for the `tau=0.2` consensus variant. No
extra model forward or backward pass was introduced. Canonical CUDA profiling
must measure this overhead again.

The final schema sanity rerun (`pilot128v3-s0-consensus-applied`) retained
10,355,200 bytes at the end of the 128-sample stream. The hard mask applied on
all 128 samples for this run; evidence now records that fact explicitly so a
future fallback cannot be mistaken for consensus retention.

## Known-only control

An additional paired MPS block control was run after the directional pilot:
OOD ratio `0`, the same 128-example prefix, and the same generated artifact.

| Seed | Ramen ID ACC | OracleID ID ACC | Same support count at every timestep | Mean GDC | SDR / retrieved OOD |
| --- | ---: | ---: | --- | ---: | ---: |
| 0 | 17.97% (23/128) | 17.97% (23/128) | yes | `2.33e-8` | 0 / 0 |
| 1 | 25.00% (32/128) | 25.00% (32/128) | yes | `1.82e-8` | 0 / 0 |
| 2 | 22.66% (29/128) | 22.66% (29/128) | yes | `1.77e-8` | 0 / 0 |

The exact paired fingerprints were `a70a75…cb157d`, `7bec4f…1e9e6`, and
`9db94c…eaa6a`. Seed 0 also had identical predictions and correctness flags;
all three seeds had identical ID accuracy, support counts, and effectively
zero directional discrepancy. This is the expected null mechanism control:
the OracleID implementation agrees with ordinary Ramen when no OOD support
exists. It is still a noncanonical pilot, so it validates the control path
rather than estimating a performance effect.

## Withdrawn pre-fix soft-weight pilot

The first `ConsensusRamenSoft` prototype used a positive linear gradient scale
before the existing SignSGD optimizer. A later code review established that
SignSGD discards that magnitude, so all nonzero-scaled coordinates had the
same update as ordinary Ramen. The historical one-cell soft run (same
fingerprint `1c850b…ff2c1`, 17.74% ID accuracy) is retained for provenance but
is **withdrawn as a soft-consensus result**. It must not be compared with the
hard mask or used to choose v1. The implementation is being replaced with an
effective SignSGD-compatible soft admission rule before any replacement pilot
or canonical ablation is interpreted.

The corrected v1 uses deterministic seeded Bernoulli coordinate admission with
probability `q**gamma` (`gamma=1`), so its expected SignSGD update is
agreement-weighted while every admitted coordinate retains Ramen's sign. A
fresh replacement seed-0 run on the same paired block/OOD-0.5 stream produced
20.97% ID accuracy (13/62): `+3.23 pp` over Ramen (17.74%) and `-1.61 pp`
below hard v0 (22.58%). Its observed admitted-coordinate rate was 23.10%,
matching its 23.11% mean agreement. This proves the corrected mechanism runs
and has a real update effect; it is one noncanonical seeded cell, so it is not
a v1 effectiveness conclusion. A multi-seed paired CUDA comparison remains
required.

## Current-self retrieval ablation

The `ConsensusRamenNoSelf` causal-history variant was also run in that exact
seed-0 block/OOD-0.5 cell (`1c850b…ff2c1`). It produced 14.52% ID accuracy
(9/62), below both Ramen/soft (17.74%) and v0 hard mask (22.58%). Its first
8-sample batch correctly used an ordinary fallback because no historical
classes existed, and consensus applied on 120/128 samples thereafter. This
one-cell negative result supports retaining v0's explicitly documented
batch-atomic current-support behavior; it does not eliminate the required
multi-seed CUDA self-retrieval ablation.

## Minimum-active-class sensitivity

In the same seed-0 block/OOD-0.5 cell, the preregistered `C_min=2` and
`C_min=4` hard-mask variants both exactly matched v0's 22.58% ID accuracy and
47.69% mean mask rate. The stream had 29.94 active class caches on average,
so neither threshold was binding after the earliest updates. This is a useful
mechanics sanity check, not evidence that the threshold is irrelevant in
smaller or more concentrated class support; the fixed paired multi-seed CUDA
ablation planner remains the decision-bearing test.

## Three-stream follow-up

The same small noncanonical artifact was also evaluated at OOD ratio 0.50 over
all three preregistered stream modes, with NoAdapt, Ramen, and the already
locked `tau=0.2` ConsensusRamen. Every seed/method cell used the exact same
stream fingerprint within its comparison; all runs remain 128-sample MPS
pilots with provenance disabled.

| Stream | NoAdapt mean ID ACC | Ramen mean ID ACC | Consensus mean ID ACC | Consensus − Ramen |
| --- | ---: | ---: | ---: | ---: |
| `iid_mixed` | 19.42% | 22.75% | 22.23% | −0.52 pp |
| `block` | 18.50% | 22.19% | 24.85% | +2.66 pp |
| `recurring` | 18.60% | 22.11% | 22.64% | +0.54 pp |
| Mean across 9 paired cells | 18.84% | 22.35% | 23.24% | +0.89 pp |

This changes the decision signal from a broad performance claim to a narrower
one: consensus is promising under the block shifts that motivated the
gradient-conflict hypothesis, neutral-to-weakly-positive in recurring shifts,
and not beneficial on this iid pilot. It is still worth the canonical matrix,
because the directional oracle evidence establishes a real contamination
mechanism; however, a final thesis result must report the iid null/negative
case rather than collapsing all stream modes into one favorable headline.

## Entropy-gated negative ablation

The pre-existing `EntropyGatedLatentRamen` ran successfully with the same
open-set evaluator contract on the three block seeds. Its ID accuracies were
12.90%, 28.12%, and 22.22% (mean 21.08%), versus Ramen's 22.19% mean. Its
memory-admission rates were only 9.4%, 11.7%, and 16.4%. This small pilot is
not a test of latent routing, but it is a sufficient compatibility and
negative-ablation signal to include the preregistered entropy-gated baseline
in the canonical matrix rather than retuning or dropping it.
