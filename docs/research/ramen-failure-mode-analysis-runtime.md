# Ramen Failure-Mode Analysis Runtime and Pilot Record

Date: 2026-08-28

## Status and scope

The failure-mode analysis runtime is implemented and its bounded diagnostic pilot has completed on CPU and Apple MPS. It records and validates evidence for existing methods; it neither changes returned predictions nor introduces `ConsensusRamen`.

The scientific framework and go/no-go criteria remain in [the failure-mode error-analysis framework](ramen-failure-mode-error-analysis-framework.md). The implementation/pilot record is in [the plan report](../../plans/20260828-172319-failure-mode-analysis/reports/implementation-and-pilot-summary.md).

## Runtime contract

`src/main.py` and `src/runtime/experiment_matrix.py` provide these immutable run-identity options:

```text
--failure-analysis-profile off|trace_v1|replay_v1     # default: off
--failure-analysis-max-samples N                      # default: 1000
--failure-analysis-max-bytes N                        # default: 268435456
--failure-counterfactual-thresholds 0.50,0.75,1.00
```

The threshold tuple is fixed to the preregistered `(0.50, 0.75, 1.00)` values; noncanonical tuples are rejected at planning, runtime, method, and sidecar boundaries. The runtime records the selected profile, limits, and tuple in the manifest identity. `trace_v1` adds compact model-visible per-query provenance/diagnostics to trace schema v2; it never puts raw gradients in JSONL. `NoAdapt` legitimately has no method failure-analysis payload.

`replay_v1` additionally creates `failure-analysis/` with `metadata.json`, `items.jsonl`, `queries.jsonl`, `features.bin`, and `gradients.bin`. The append-only sidecar stores each vector once and is accepted only when completed, checksum-valid, and bound to the same run ID, manifest SHA-256, stream fingerprint, and source fingerprint. A sample/byte limit overrun is `insufficient`, never a silently complete artifact.

The post-hoc evaluator is invoked as:

```bash
python -m src.evaluation.failure_mode_analysis \
  --baseline-run-dir <noadapt-run-dir> \
  --adapted-run-dir <adapted-replay-v1-run-dir> \
  --output <failure-mode-analysis.json>
```

Verified mode refuses unpaired/mismatched evidence. It strictly pairs `(timestep, sample_idx, ground_truth_domain)` and requires matching ground-truth class, device, evaluator configuration, model and dataset artifact digests, stream fingerprint, source fingerprint, and the declared baseline-reference binding. Optional `--atomic-run-dir` plus `--causal-run-dir` adds F5 scheduling evidence under the same compatibility contract while allowing the intentional method and schedule difference.

## Model-visible data versus evaluator-only labels

The deployable path sees only model-visible retrieval/update evidence. Ground-truth class/domain, pseudo-label correctness, ID/OOD state, beneficial/harmful outcomes, and oracle membership are evaluator-only joins made after the actual prediction returns. Fixed-support counterfactual variants are evaluator-only too; they must verify reset state and never select or alter the returned prediction.

## Exact bounded-pilot commands

The checked pilot used the smoke configuration, `CIFAR100C`, `block`, seed 0, batch size 1, block size 4, and four samples:

```bash
python src/runtime/experiment_matrix.py \
  --dataset CIFAR100C --stream block --method NoAdapt --method CausalRamen \
  --seed 0 --device <cpu|mps> --config-dir cfg/smoke \
  --data-root <data-root> --evidence-dir <report-root> \
  --max-eval-samples 4 --batch-size 1 --stream-block-size 4 \
  --failure-analysis-profile replay_v1 \
  --failure-analysis-max-samples 4 --failure-analysis-max-bytes 268435456 \
  --failure-counterfactual-thresholds 0.50,0.75,1.00 --execute
```

The matrix requires an explicit `--device` for `--execute` or `--resume`; it has no silent fallback. CUDA was unavailable on the Apple host, so there is no CUDA result and no CPU/MPS result is a CUDA claim. Run the same explicit command with `--device cuda` on a Linux NVIDIA host for CUDA evidence.

## Verified pilot outcome

CPU and MPS reports are independently provenance-verified and agree on every reported aggregate for this four-sample pilot:

| Evidence | CPU | MPS |
| --- | ---: | ---: |
| NoAdapt accuracy | 0.25 (1/4) | 0.25 (1/4) |
| CausalRamen accuracy | 0.50 (2/4) | 0.50 (2/4) |
| Exact delta `H - A` | +0.25 | +0.25 |
| Beneficial / harmful | 1 / 0 | 1 / 0 |
| Legal oracle / retrieved oracle | 0.50 / 0.50 | 0.50 / 0.50 |
| Counterfactual delta, each threshold | 0.00 | 0.00 |

The full outputs are [CPU JSON](../../plans/20260828-172319-failure-mode-analysis/reports/luna-runs-v4/cpu/failure-mode-analysis.json) and [MPS JSON](../../plans/20260828-172319-failure-mode-analysis/reports/luna-runs-v4/mps/failure-mode-analysis.json). F3 is insufficient because neither run contains harmful events, open-set GDC/SDR is unavailable, and no atomic/causal F5 pair was supplied. The individual production traces nevertheless identify all four adapted queries as causal. F4 is complete but has no harmful event to recover and no accuracy change.

`ConsensusRamen` remains unimplemented. The decision is `INSUFFICIENT`: the pilot has neither a stable harmful-versus-beneficial conflict direction across fixed seeds/structured streams nor oracle recovery evidence for harmful updates. These CPU/MPS runs verify runtime mechanics/device parity only, not final scientific evidence or threshold-tuning data.
