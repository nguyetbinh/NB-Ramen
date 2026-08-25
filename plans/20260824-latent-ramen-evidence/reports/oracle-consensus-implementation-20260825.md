# OracleConsensusRamen implementation — 2026-08-25

## Status

Implemented and locally validated in the pinned `nb-ramen` environment.

## Scope

`OracleConsensusRamen` is the explicitly named evaluator-only upper bound
listed in thesis report §10 and §25.  It consumes a boolean `is_ood` vector
only through the existing single-use `OracleOODContextHook`, fails closed when
the evaluator has not supplied a matching batch, and clears pending context on
reset.

The method retains only evaluator-known ID samples in the usual per-predicted-
class `PriorityCache` instances.  It otherwise preserves `ConsensusRamen-v0`:
batch-atomic current-batch visibility among admitted supports, nearest-support
retrieval, entropy/distance weights, class balancing, hard consensus masking,
temporary SignSGD update, and parameter reset.  It emits pre-adaptation energy
OOD scores and retained-memory bytes for the direct open-set evaluator.

## Diagnostic interpretation

This method does **not** emit `ramen_vs_oracle_id_cosine` or
`ramen_vs_oracle_id_sign_disagreement`.  Unlike `OracleIDGradientRamen`, it
does not construct both the same all-support Ramen direction and its ID-filtered
counterpart; it changes the support cache before consensus.  A trace should
therefore retain its consensus diagnostics, energy score, retained-memory
evidence, and oracle provenance, while leaving the all-vs-ID direction fields
unset rather than presenting them as comparable contamination measurements.

This keeps Phase B's directional oracle evidence semantically separate from
the §25 oracle-consensus upper bound.

## Validation

```text
PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest \
  tests.test_oracle_consensus_ramen tests.test_consensus_ramen \
  tests.test_oracle_id_gradient_ramen -v

/Users/admin/miniconda3/envs/nb-ramen/bin/python -m compileall -q \
  src/methods tests/test_oracle_consensus_ramen.py
```

Result: 25 focused tests passed; compilation completed successfully.

Covered mechanics: config provenance, single-use fail-closed hook, ID-only
admission from a mixed evaluator-labelled batch, ordinary ConsensusRamen label
isolation, reset behavior, config loading, and adjacent consensus/oracle
regression checks.
