# Failure-Mode Analysis Implementation and Pilot Summary

Date: 2026-08-28

## Delivered

The runtime supports opt-in `trace_v1` and bounded `replay_v1` profiles,
immutable counterfactual thresholds (`0.50,0.75,1.00`), checksummed replay
sidecars, strict paired evidence validation, offline F0–F5 reporting, and
explicit `computed`, `insufficient`, or `unavailable` states. See the
[runtime record](../../../docs/research/ramen-failure-mode-analysis-runtime.md)
for exact CLI contracts and commands.

Ground-truth class/domain and all oracle/outcome labels are evaluator-only
joins after a method returns. They never participate in deployable method
execution. Controlled counterfactual outputs are evaluator-only as well.

## Bounded verification pilot

The fresh `luna-runs-v4` pilot used CIFAR-100-C, block stream, seed 0, batch
size 1, block size 4, and four samples. Both CPU and MPS completed a NoAdapt
baseline and CausalRamen `replay_v1` run; the adapted sidecars are completed
and verified.

| Metric | CPU | MPS |
| --- | ---: | ---: |
| NoAdapt accuracy | 0.25 | 0.25 |
| CausalRamen accuracy | 0.50 | 0.50 |
| Beneficial / harmful flips | 1 / 0 | 1 / 0 |
| `Acc_adapted - Acc_base = H - A` | +0.25 | +0.25 |
| Legal / retrieved pseudo-label oracle rate | 0.50 / 0.50 | 0.50 / 0.50 |
| Counterfactual accuracy delta (0.50, 0.75, 1.00) | 0.00 each | 0.00 each |

Sources: [CPU report](luna-runs-v4/cpu/failure-mode-analysis.json) and
[MPS report](luna-runs-v4/mps/failure-mode-analysis.json).

## Decision and limitations

The results demonstrate CPU/MPS runtime parity for this bounded mechanics
pilot. They are not final scientific evidence. No harmful events occurred, so
the conflict comparison and harmful-event recovery are insufficient; open-set
GDC/SDR is unavailable; and no atomic/causal F5 pair was supplied. The
individual production traces do record all four adapted queries as causal.

`ConsensusRamen` remains unimplemented. Its report decision is `INSUFFICIENT`,
because no stable conflict association across fixed seeds and structured
streams, nor evaluator-only recovery of harmful updates, has been shown.

CUDA was unavailable on the Apple host. Execution/resume requires an explicit
device; there is no CUDA fallback to CPU or MPS. A CUDA-grade follow-up must
run the recorded matrix with `--device cuda` on a Linux NVIDIA host.
