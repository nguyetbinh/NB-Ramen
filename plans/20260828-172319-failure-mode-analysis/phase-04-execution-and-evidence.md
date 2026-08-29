# Phase 4: Execution and Evidence

Status: completed for bounded CPU/MPS mechanics evidence. CUDA is unavailable
on this Apple host and remains an explicit, no-fallback follow-up on NVIDIA.

## Context

- `src/runtime/experiment_matrix.py`
- `docs/research/experiment-runtime.md`
- `cfg/smoke/`

## Requirements

- Use a fresh Luna execution task for CPU and local accelerator runs.
- Run deep preflight before execution.
- Produce paired NoAdapt/adapted bounded evidence on at least two structured stream shapes where runtime permits.
- Record current-host limitations honestly; no silent CUDA fallback.

## Implementation

1. Run focused/full unit tests in a Luna execution task.
2. Preregister the local analysis-pilot matrix: methods `NoAdapt`, `CausalRamen`, `StructuredAtomicRamen`; streams `block`, `recurring`; seeds `0,1,2`; batch sizes `1,4`; bounded CIFAR-100-C smoke config; explicit `analysis` split role; profile `replay_v1` for adapted methods; fixed counterfactual thresholds `0.50,0.75,1.00`.
3. Launch CPU mechanics matrix with persistent evidence output.
4. Launch MPS analysis-pilot matrix as the available local accelerator, reducing only the declared sample budget if preflight estimates exceed the bounded run budget.
5. Analyze valid paired evidence with the new CLI.
6. Record CUDA command/runbook for an NVIDIA host if CUDA is unavailable.

The analysis CLI must emit per-threshold F4 helpful/harmful flips, harmful-event recovery, newly introduced harm, accuracy delta, and the best-of-preregistered oracle upper bound. Missing or incomplete counterfactual rows force F4 and the overall ConsensusRamen go gate to `INSUFFICIENT`.

## Validation

- Runtime preflight passes.
- Evidence manifests, fingerprints, trace schemas, and pairing validate.
- Analysis JSON identities reconcile with source summaries.
- Local CPU/MPS results are labeled diagnostic pilot evidence; ConsensusRamen decision remains `INSUFFICIENT` without preregistered final/CUDA-grade evidence and evaluator-only oracle recovery.

## Risks and Rollback

- Dataset/model availability or runtime cost can block empirical pilots; unit-level mechanics and exact blockers remain reportable.
- Interrupted matrix runs use strict `--resume`, never partial hand-edited artifacts.
