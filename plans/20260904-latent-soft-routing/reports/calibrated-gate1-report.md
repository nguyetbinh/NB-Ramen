# Margin-Calibrated Oracle Soft Routing Gate 1 — 2026-09-04

## Outcome

The calibrated oracle-soft experiment is complete. It found no positive
performance signal and explains why increasing the absolute domain bonus
cannot create the intended 10/25/50% support-membership interventions in this
bounded stream. `LatentSoftRamen` remains unauthorized.

## Reuse-first protocol

- Dataset/backbone: CIFAR-100-C, CLIP ViT-B/16.
- Stream: canonical block, block size 64, prefix 200, seed 0.
- Device/provenance: MPS, fast verified artifact inventory.
- Stream fingerprint:
  `aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b`.
- Reused: canonical NoAdapt, CausalRamen, OracleHardRamen, gamma-zero, and
  gamma-0.25 artifacts.
- Retained evidence-producing model work: one gamma-zero margin profile and
  exactly three calibrated OracleSoftRankRamen cells. No newly generated
  control was used in the analysis.
- Calibration ID:
  `d61cb595c19ddda5b92d4a463fbe6a0e1c82fa61dc3867498b9531f9225ced68`.

Luna was not discoverable in this environment, so the bounded cells ran on
the locally available MPS device.

During backward-compatibility checking, the legacy Gate 1 runner resolved a
changed config hash and mistakenly launched a disposable NoAdapt run plus a
partial gamma-zero run instead of reusing the canonical pair. They were
stopped/quarantined under `/tmp/nb-ramen-accidental-gate1-resume-20260904/`
and are excluded from every result in this report. The calibrated runner uses
an explicit external baseline config/trace and does not have that behavior.

## Preregistered calibration

The gamma-zero profile produced 64 exact replacement boundaries on 37 of 200
queries. Strengths were fixed from the margin distribution before reading any
nonzero result:

| Strength | Rule | Gamma |
|---|---|---:|
| weak | next float32 above Q25 | 0.0162578616 |
| medium | next float32 above Q50 | 0.0333540924 |
| strong | next float32 above Q75 | 0.0572425611 |

Margin quantiles were Q10 `0.00968862`, Q25 `0.01625786`, Q50 `0.03335409`,
Q75 `0.05724256`, and Q90 `0.09568086`. The maximum observed margin was
`0.12787533`.

## Results

| Method | Micro | Macro-domain | Worst-domain | Negative windows | Prediction changes vs gamma zero |
|---|---:|---:|---:|---:|---:|
| NoAdapt | 0.305 | 0.3203 | 0.2344 | reference | — |
| CausalRamen / gamma zero | 0.330 | 0.3945 | 0.2812 | 1/4 | 0 |
| OracleHardRamen | 0.305 | 0.3750 | 0.2188 | 2/4 | — |
| OracleSoft weak | 0.330 | 0.3945 | 0.2812 | 1/4 | 0/200 |
| OracleSoft medium | 0.330 | 0.3945 | 0.2812 | 1/4 | 0/200 |
| OracleSoft strong | 0.330 | 0.3945 | 0.2812 | 1/4 | 0/200 |
| OracleSoft gamma 0.25, reused | 0.330 | 0.3945 | 0.2812 | 1/4 | 0/200 |

All calibrated points preserve the global support mechanism:

| Strength | Positional slot changes, mean | Same-domain ratio, mean | ESS, mean | Returned support, mean | Active classes, mean |
|---|---:|---:|---:|---:|---:|
| gamma zero | 0.000% | 49.1708% | 23.8288 | 88.865 | 42.105 |
| weak | 0.883% | 49.2254% | 23.8011 | 88.865 | 42.105 |
| medium | 1.491% | 49.2789% | 23.7986 | 88.865 | 42.105 |
| strong | 2.026% | 49.3335% | 23.8015 | 88.865 | 42.105 |
| gamma 0.25, reused | 2.469% | 49.3888% | 23.7998 | 88.865 | 42.105 |

`selection_change_ratio` is a positional diagnostic: it counts a different
item ID at the same class/rank slot, including reorderings that leave the
selected set unchanged. Replacement margins measure actual set membership.
The latter gives the decisive saturation result:

| Strength | Boundaries crossed | Aggregate membership fraction | Mean per-query membership fraction |
|---|---:|---:|---:|
| weak | 16/64 | 0.0900% | 0.0546% |
| medium | 32/64 | 0.1800% | 0.1082% |
| strong | 48/64 | 0.2701% | 0.1627% |
| gamma 0.25 | 64/64 | 0.3601% | 0.2180% |

There are 17,773 returned support slots in the profile. Because gamma 0.25 is
larger than every measured replacement margin, raising gamma further can only
reorder already-selected same-domain items; it cannot add more same-domain
members under these memory/context semantics.

## Gate decision

- **Gate A — intervention validity:** fails as a feasibility condition. The
  intended 10/25/50% membership changes are unavailable; even saturation is
  only 0.3601% of aggregate support membership.
- **Gate B — diversity preservation:** passes. Support count, active classes,
  class coverage, and ESS remain healthy and far from OracleHardRamen.
- **Gate C — oracle-soft value:** fails. No calibrated point changes a
  prediction or improves any primary metric.
- **Gate D — stop/pivot:** stop exact-domain bonus routing as the primary axis
  for this bounded setting. More gamma is not informative, and a learned
  latent replica has no oracle benefit to recover.

This is a bounded mechanism decision, not a broad benchmark claim. It uses
one seed, one deterministic 200-sample prefix, MPS, and fast provenance.

## Evidence locations

Raw profile, calibration, generated configs, and three calibrated runs remain
outside git at:

```text
/Users/admin/Documents/NB-Ramen/plans/20260904-latent-soft-routing/reports/gate1-calibrated-mps-n200
```

Run IDs:

- profile: `cifar100c-block-seed-0-oraclesoftrankramen-dev-mps-n200-cfg-95a4521bb101-prov-fast-data-6dbea801cbad`;
- weak: `cifar100c-block-seed-0-oraclesoftrankramen-dev-mps-n200-cfg-30c62e1d09c4-prov-fast-data-6dbea801cbad`;
- medium: `cifar100c-block-seed-0-oraclesoftrankramen-dev-mps-n200-cfg-4f6f9b547ec0-prov-fast-data-6dbea801cbad`;
- strong: `cifar100c-block-seed-0-oraclesoftrankramen-dev-mps-n200-cfg-5a1b11867d90-prov-fast-data-6dbea801cbad`.
