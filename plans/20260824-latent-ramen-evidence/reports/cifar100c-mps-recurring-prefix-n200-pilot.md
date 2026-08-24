# CIFAR-100-C MPS recurring-prefix pilot — 2026-08-24

## Scope and semantic limit

This cost-limited run uses the official `CIFAR100C` wrapper, benchmark configs
in `cfg/`, seed 0, explicit MPS, `recurring`, batch size 100, a 200-sample
prefix, 50-sample metric windows, and fast provenance. Raw evidence is at:

```text
/Users/admin/Documents/NB-Ramen/evidence/cifar100c-pilot-mps-recurring-n200
```

The four runs are NoAdapt, legacy batch-atomic Ramen, OracleLatentRamen, and
LatentRamen. Every trace contains 200 rows and all runs share stream
fingerprint:

```text
84a2fff8b96925cc04aa5459d9a77baeef03c7a0175a5bd75b1a777d43d160aa
```

This prefix does **not** contain a recurrence. With 15 domains and the
canonical 64-sample block size, the first repeated domain begins only at
timestep 960. The observed prefix contains four previously unseen episodes:

```text
shot_noise=64, brightness=64, impulse_noise=64, fog=8
shift_timesteps=[64,128,192]
```

The artifact is therefore a valid structured-shift and runtime pilot, but it
must not support a claim about reuse after domain recurrence. A future
noncanonical cost-limited recurrence pilot must use an explicit, identity-bound
smaller block size and be labeled separately from the canonical full grid.

## Strictly validated summaries

| Method | Micro | Macro domain | Worst domain | Negative windows | Routing NMI / contexts | Method memory max | Forward total | MPS sampled bytes |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| NoAdapt | 0.385 | 0.3828 | 0.2500 | reference required | unavailable | unavailable | 6.222 s | 650,809,856 |
| Ramen | 0.445 | 0.4297 | 0.2969 | 0/4 | unavailable | unavailable | 225.746 s | 6,542,988,032 |
| OracleLatentRamen | 0.435 | 0.4219 | 0.2812 | 0/4 | 1.000 / 4 | 16,184,000 B | 432.869 s | 494,846,208 |
| LatentRamen | 0.435 | 0.4219 | 0.2812 | 0/4 | 0.000 / 1 | 16,184,000 B | 503.061 s | 495,892,992 |

The MPS device-memory values are synchronized post-batch samples, not exact
allocator peaks. Legacy Ramen still lacks logical retained-memory evidence, so
the device sample cannot be treated as an equal-memory support-cache measure.

For the first complete shift at timestep 64, all four methods met their own
pre-shift baseline immediately. None recovered after the complete timestep-128
shift. The final 8-sample episode is explicitly `insufficient_episode` for a
50-sample recovery window. These outcomes describe two usable shifts in one
prefix; they do not estimate recurrence behavior.

## Analyzer outcome

The strict analyzer reconstructed all four methods and returned:

```text
latent_vs_ramen_micro_gain=-0.010
latent_vs_noadapt_micro_gain=+0.050
oracle_recovery_accuracy_gain=-0.010
latent_vs_oracle_accuracy_gap=0.000
routing_nmi=0.000
class_context_nmi=0.000
forward_latency_ratio_latent_over_ramen=2.2284
sampled_device_memory_ratio_latent_over_ramen=0.0758
go_no_go=insufficient_evidence
```

The LatentRamen latency ratio again fails the configured 1.25 gate. Oracle
routing is 1.0 percentage point below Ramen and the unsupervised router again
collapses to one context. This second bounded cell reinforces the decision to
test the roadmap's reliability-aware memory axis next; it does not establish
that entropy admission will help.

Repeated-run tolerance, structured degradation, router closure,
natural-domain gain, and routing/accuracy association remain insufficient.
The memory ratio passes only as sampled-device evidence and is not an
equal-logical-memory comparison.

## Integrity checks

Rerunning the exact matrix command with `--execute --resume` revalidated and
skipped all four runs without launching a model. The adapted manifests point
to the exact NoAdapt trace, the four stream exports have the same fingerprint,
and the analyzer accepted the complete artifacts before returning exit 1 for
the scientific `insufficient_evidence` outcome.
