# Open-set Consensus/Oracle evidence matrix implementation — 2026-08-25

## Delivered

`build_open_set_evidence_matrix()` is the concrete, separate canonical plan
for open-set CIFAR-100-C. Its fixed grid contains 252 runs:

- methods: `NoAdapt`, `Ramen`, `EntropyGatedLatentRamen`,
  `OracleDropOODRamen`, `OracleIDGradientRamen`, `ConsensusRamen`,
  `OracleConsensusRamen`;
- OOD ratios: `0`, `0.1`, `0.3`, `0.5`;
- streams: `iid_mixed`, `block`, `recurring`;
- seeds: `0`, `1`, `2`.

It requires CUDA, uses the fixed 400-source-example-per-domain protocol by
default, and binds both the OOD ratio and source budget into each run ID.
Every cell schedules `NoAdapt` first; the six adapted runs point
to its same-ratio/stream/seed trace. Commands include the versioned open-set
split and ratio. Oracle evaluator context is checked by the separate analyzer
and is permitted only for the three names explicitly marked as oracle controls.

`evaluation.open_set_consensus_analysis` is deliberately not an extension of
the legacy latent-router gate. It strictly loads completed evidence, checks the
seven-method coverage and identical stream fingerprints for every cell, reports
ID/open-set outcomes plus Oracle and Consensus diagnostics, and labels only a
complete full-stream CUDA plan as `canonical_cuda_expected`. It has no
Consensus certification/go-no-go outcome.

## Evidence boundary

The existing local MPS directional results remain `noncanonical_pilot`:
they are cost-limited, use generated approximate inputs, and have unverified
provenance. They are not consumed as canonical CUDA evidence by the new
analyzer. Canonical evidence still requires the verified official CIFAR-100-C
artifact and a Linux NVIDIA CUDA runner.

## Validation

Focused pinned-environment commands:

```shell
PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest \
  tests.test_experiment_matrix tests.test_open_set_consensus_analysis -v
PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m compileall -q src tests
git diff --check
```

All checks passed: 51 focused matrix/analyzer tests, bytecode compilation, and
`git diff --check`.

Status: DONE
