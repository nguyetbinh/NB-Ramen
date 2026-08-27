# CIFAR-100-C MPS block pilot — 2026-08-24

## Scope

This is a cost-limited feasibility and evidence-contract pilot, not a Phase 2
result. It uses the real official `CIFAR100C` wrapper, the benchmark configs in
`cfg/`, seed 0, explicit MPS, `block`, batch size 100, a 200-sample prefix,
50-sample metric windows, and fast run provenance after one-time exact dataset
verification.

Raw evidence remains at
`/Users/admin/Documents/NB-Ramen/evidence/cifar100c-pilot-mps-block-n200`.
The five runs analyzed in this report are NoAdapt, Tent, legacy batch-atomic Ramen,
OracleLatentRamen, and LatentRamen. All contain 200 trace rows and share
exported stream fingerprint
`aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b`.
The prefix contains three complete
64-sample episodes and an 8-sample final episode:

```text
pixelate=64, gaussian_noise=64, glass_blur=64, shot_noise=8
shift_timesteps=[64,128,192]
```

A later paired CausalRamen run on the same fingerprint is analyzed separately
in the [dedicated CausalRamen report](causal-ramen-mps-paired-pilot.md).

## Strictly validated summaries

| Method | Micro | Macro domain | Worst domain | Negative windows | Routing NMI / contexts | Method memory max | Forward total | MPS sampled bytes |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| NoAdapt | 0.305 | 0.3203 | 0.2344 | reference required | unavailable | unavailable | 6.473 s | 650,809,856 |
| Tent | 0.305 | 0.3203 | 0.2344 | 0/4 | unavailable | unavailable | 317.459 s | 650,124,800 |
| Ramen | 0.315 | 0.3828 | 0.2812 | 3/4 | unavailable | unavailable | 204.206 s | 6,543,125,248 |
| OracleLatentRamen | 0.305 | 0.3750 | 0.2188 | 2/4 | 1.000 / 4 | 16,184,000 B | 273.262 s | 494,565,376 |
| LatentRamen | 0.330 | 0.3945 | 0.2812 | 1/4 | 0.000 / 1 | 16,184,000 B | 340.728 s | 494,789,376 |

The synchronized MPS memory values are post-batch samples, not exact allocator
peaks. Legacy Ramen does not expose logical retained-memory bytes, so its high
sampled device value cannot be substituted for a like-for-like support-memory
measurement. Forward totals exclude model construction, dataset loading, and
provenance verification.

The persistent-episode recovery evidence is non-vacuous for the first two
shifts; the final 8-sample episode is explicitly insufficient for a 50-sample
window:

- NoAdapt and Tent did not recover on either complete shift.
- Ramen did not recover at timestep 64 and recovered 14 samples after the
  timestep-128 shift.
- OracleLatentRamen did not recover on either complete shift.
- LatentRamen recovered 10 samples after the timestep-64 shift and did not
  recover after the timestep-128 shift.

These individual episode outcomes are descriptive only. Two usable shifts in
one deterministic prefix are not a stable recovery estimate.

## Analyzer outcome

The strict post-hoc analyzer reconstructed and revalidated NoAdapt, Ramen,
OracleLatentRamen, and LatentRamen. For this single cell it reported:

```text
latent_vs_ramen_micro_gain=0.015
latent_vs_noadapt_micro_gain=0.025
oracle_recovery_accuracy_gain=-0.010
routing_nmi=0.000
class_context_nmi=0.000
forward_latency_ratio_latent_over_ramen=1.6685
sampled_device_memory_ratio_latent_over_ramen=0.0756
go_no_go=insufficient_evidence
```

The configured 1.25 latency-ratio gate fails on this MPS pilot. The sampled
device-memory ratio passes numerically but is not an equal-logical-memory
result. Oracle recovery is negative, so oracle-gap closure is undefined. The
unsupervised router discovered only one context even though four ground-truth
domains occur; this is pilot evidence of routing collapse, not support for the
router hypothesis.

The analyzer correctly left repeated-run tolerance, structured degradation,
router closure, natural-domain gain, and routing/accuracy association as
`insufficient_evidence`. The complete gate still requires three seeds, all
selected structured streams, DomainNet, and the fixed CUDA contract.

## Independent validation

A fresh read-only Terra review reconstructed the exact five-run matrix, called
the current strict validator, and reran the thresholded analyzer. All five
runs passed and `--resume` skipped them without launching a model. The review
confirmed 200 trace and stream rows per run, the shared fingerprint, exact
NoAdapt reference paths, live CLIP and dataset artifact identity, recovery,
routing, memory, and latency evidence. It found no evidence-integrity defect;
it independently reproduced the 1.6685 latency-ratio failure, routing collapse,
and `insufficient_evidence` result.

## Feasibility observation

Measured synchronized forward time for only 200 samples was approximately
5.7 minutes for LatentRamen, 4.6 minutes for OracleLatentRamen, 3.4 minutes for
Ramen, and 5.3 minutes for Tent. This empirically confirms that the M2/MPS host
is suitable for mechanics and bounded pilots but not the 300-run full matrix.
The CUDA/DomainNet runbook must use measured pilots before promising duration.

Rerunning the identical matrix invocation with `--resume` strictly validated
and skipped all five runs.
