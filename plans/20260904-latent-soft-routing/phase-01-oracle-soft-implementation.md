# Phase 1 — Oracle Soft Implementation

**Status:** complete

## Requirements

- Preserve `LatentRamen` and `OracleLatentRamen` compatibility.
- Add `LatentHardRamen` and `OracleHardRamen` aliases.
- Add a global per-class memory query ranked by feature distance plus a binary same-context bonus.
- Add `OracleSoftRankRamen` with strict per-item causal insertion/query order.
- Persist support count, active classes, class coverage, same/cross-domain ratios, effective sample size, and context strength.

## Files

- `src/memory/structured_memory.py`
- `src/methods/SoftRoutingRamen.py`
- `src/methods/__init__.py`
- `src/main.py`
- `src/evaluation/evidence.py`
- `src/runtime/experiment_matrix.py`
- `cfg/CIFAR100C/*.yaml`
- focused tests under `tests/`

## Validation

- Run focused memory, method, evidence, and matrix tests.
- Run the full unit suite after integration.

## Risks and rollback

- Keep the existing hard query untouched so rollback is removal of additive APIs/classes.
- Keep context metadata diagnostic-only outside the explicitly named oracle method.
