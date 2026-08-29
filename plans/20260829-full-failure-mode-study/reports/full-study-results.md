# Full Ramen Failure-Mode Study Result

Date: 2026-08-30

## Decision

The preregistered decision is **`INSUFFICIENT`**. The evidence does not justify
implementing `ConsensusRamen`.

All reports below were recomputed from completed, checksum-validated,
manifest-bound `replay_v1` artifacts. Model execution never receives evaluator
ground truth, beneficial/harmful outcomes, or ID/OOD membership.

All eight primary cells share source fingerprint
`4399d97db7b040d3d0fd2a8aa09e672657366e0bed1d58daed45e32eaa84c220`.
The study aggregator additionally requires one common source, model artifact,
dataset artifact, method configuration, and non-stream evaluator contract
across cells before it evaluates the consensus gate.

## Closed-set primary matrix

Apple MPS, CIFAR-100-C, exact provenance, batch size 1, 64 samples per cell:

| Stream | Seed | NoAdapt | CausalRamen | Beneficial | Harmful | F3 harmful − beneficial conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| block | 0 | 0.2500 | 0.2813 | 4 | 2 | +0.1622 |
| block | 1 | 0.2500 | 0.3125 | 5 | 1 | +0.1369 |
| recurring | 0 | 0.4219 | 0.5469 | 8 | 0 | insufficient |
| recurring | 1 | 0.4531 | 0.5000 | 5 | 2 | −0.0475 |

F0–F2 and entropy diagnostics were computed in all four cells. F3 and F4 were
eligible in three cells. Conflict moved in the hypothesized direction in both
block seeds but reversed in recurring seed 1, so the association is not stable
across streams.

Entropy groups did not supply a consistent trust signal. For example,
high-wrong support gradients had higher mean cosine than high-correct gradients
in both block seeds (`0.1610` vs `0.1098`, `0.2046` vs `0.1859`), while the
ordering reversed in recurring streams. The exact metric is
`support_gradient_vs_current_query_raw_gradient_v1`.

## Fixed-threshold replay oracle

The exact thresholds were fixed at `0.50`, `0.75`, and `1.00`. In eligible
cells, replay recovered 50–100% of observed harmful events. It did not improve
accuracy in any cell: the best delta versus the actual adapted prediction was
`0.0000`, `−0.0156`, `−0.0781`, and `−0.0156`, because recovered harmful cases
were offset by new harm. This is an evaluator-only upper-bound diagnostic, not
a deployable router.

## Schedule-only comparison

The CPU batch-size-four comparison used matched `StructuredAtomicRamen` and
`CausalRamen` configurations:

| Stream | Atomic accuracy | Causal accuracy | Causal − atomic | Atomic future supports/query | Causal future supports/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| block | 0.2813 | 0.3438 | +0.0625 | 1.5 | 0.0 |
| recurring | 0.5625 | 0.5625 | 0.0000 | 1.5 | 0.0 |

The result confirms that causal scheduling removes within-batch future support.
It does not provide a stable positive schedule effect across streams.

## Domain probes

Verified frozen features from both MPS block seeds predicted domain with test
accuracy `1.0`. Feature-to-class test accuracy was `0.0` and `0.1818`.
Class-conditioned probes were data-limited at 64 samples (2 computed classes
per seed), so domain decodability is established for these bounded streams but
class-conditional conclusions remain insufficient.

## Semantic open-set oracle

The model vocabulary contained only the fixed 80 known CIFAR-100 classes;
held-out labels and ID/OOD membership were joined only after prediction.

| OOD ratio | NoAdapt | CausalRamen | Mean retrieved-OOD count | Mean retrieved-OOD weight | Mean GDC | Mean SDR | Harmful ID count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.3594 | 0.4219 | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| 0.1 | 0.4063 | 0.4375 | 0.115 | 0.041 | 0.004 | 0.024 | 3 |
| 0.3 | 0.2813 | 0.2969 | 0.340 | 0.180 | 0.036 | 0.082 | 0 |
| 0.5 | 0.2031 | 0.2500 | 0.566 | 0.327 | 0.076 | 0.123 | 0 |

OOD support contamination and the all-versus-ID gradient gap increase with the
requested ratio. Harmful ID events are too sparse to claim that this gap causes
negative adaptation.

## Device and verification coverage

- MPS primary: 8/8 closed-set cells complete and resume-valid.
- MPS open-set: 8/8 cells complete and resume-valid with no fallback.
- CPU validation: block `+0.0625`, recurring `+0.15625` paired accuracy delta.
- CPU F5: 6/6 schedule cells complete and resume-valid.
- CUDA: PyTorch reported no CUDA device; an explicit CUDA cell failed closed
  without creating fallback evidence.
- Tests: 330 passed; `compileall` and `git diff --check` passed.

The machine-readable aggregate is
[`mps-primary/study-aggregate.json`](mps-primary/study-aggregate.json). It has
three eligible consensus cells and rejects the gate because conflict direction
is not stable across structured streams and the recurring stream does not have
two eligible harmful-event seeds. Raw evidence remains under
`evidence/full-study-20260829/` and is intentionally not committed.
