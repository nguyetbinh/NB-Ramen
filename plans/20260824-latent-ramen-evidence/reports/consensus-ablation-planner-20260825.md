# Preregistered Consensus ablation planner — 2026-08-25

## Status

**Planned, not executed.** This is a separate held-out protocol for thesis
§26. It neither modifies the locked seven-method primary matrix nor supplies
adaptation-effectiveness evidence.

## Fixed identities

For one caller-selected open-set stream, OOD ratio, and seed set, the planner
always schedules these seven adapted identities plus one exact paired NoAdapt
trace per seed:

| §26 purpose | method/config identity |
| --- | --- |
| ordinary aggregation | `Ramen` |
| hard consensus v0 | `ConsensusRamen` (`tau=.2`, `C_min=3`) |
| soft consensus v1 | `ConsensusRamenSoft` |
| causal no-current support | `ConsensusRamenNoSelf` |
| threshold sensitivity | `ConsensusRamenTau060` (`tau=.6`) |
| active-class sensitivity | `ConsensusRamenMin2` (`C_min=2`) |
| active-class sensitivity | `ConsensusRamenMin4` (`C_min=4`) |

All configs specify their complete fixed surface, are read and hashed during
planning, and their hash is bound into the run ID/manifest. The planner rejects
any missing selected config rather than issuing a run with a `missing` hash.
This is a preregistered comparison surface: no variant is selected or retuned
from results in the held-out cell.

## Invocation

Canonical plans require an explicit CUDA identity. For example, this creates
`8 × 3 = 24` planned (but not executed) runs:

```shell
PYTHONPATH=src python -m runtime.consensus_ablation_matrix \
  --stream block --ood-ratio 0.5 --seed 0 --seed 1 --seed 2 \
  --device cuda --data-root "$RAMEN_DATA_ROOT" \
  --evidence-dir "$RAMEN_EVIDENCE_DIR"
```

An MPS/CPU smoke plan is permitted only as an explicitly named noncanonical
pilot; it must never be mixed with canonical evidence:

```shell
PYTHONPATH=src python -m runtime.consensus_ablation_matrix \
  --stream block --ood-ratio 0.5 --seed 0 --device mps \
  --noncanonical-pilot --pilot-name ablation-mps-smoke \
  --max-eval-samples 128 --data-root "$RAMEN_DATA_ROOT"
```

The planner prints `planned_not_executed`; it has no execute flag. Actual
canonical execution still requires verified official CIFAR-100-C and an NVIDIA
CUDA runner. The selected ratio carries the same fixed 400-source-per-domain
budget as the primary protocol, binds it to the stream/run identity, and
requires that budget be divisible by the ratio denominator.

## Validation

`tests/test_consensus_ablation_matrix.py` asserts the fixed config identities
and distinct hashes, exact same-seed NoAdapt reference traces, CUDA rejection
for canonical planning, mandatory named pilot opt-in for MPS, ratio/budget
validation, and plan-only CLI output. This validation proves scheduling
integrity, not an experimental outcome.
