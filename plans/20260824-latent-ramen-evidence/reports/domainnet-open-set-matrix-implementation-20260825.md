# DomainNet open-set canonical matrix implementation

Date: 2026-08-25

## Delivered planning contract

`src/runtime/open_set_domainnet_matrix.py` now emits a separate, execution-free
secondary-benchmark plan. It locks 252 CUDA runs:

- seven Phase-E baselines: NoAdapt, Ramen, EntropyGatedLatentRamen,
  OracleDropOODRamen, OracleIDGradientRamen, ConsensusRamen, and
  OracleConsensusRamen;
- OOD ratios 0, 0.1, 0.3, 0.5;
- `iid_mixed`, `block`, `recurring`; and
- seeds 0, 1, 2.

Each adapted run points to the `trace.jsonl` of the exact NoAdapt
ratio/stream/seed cell. The run IDs and commands include the OOD ratio and
fixed source budget. The planner rejects non-CUDA identities, partial grids,
cost-limited prefixes, unverified (`off`) provenance, and any budget other
than 690.

The versioned `open-set-domainnet-name-rank-v1` split is passed through every
command. Execution uses the existing `OpenSetDomainNet` and ordinary
`build_open_set_stream` path, which materializes and binds the validated
taxonomy SHA-256 and split fingerprint into stream/manifest evidence.

## Source-exposure decision

The primary CIFAR-100-C contract fixes 400 samples per corruption domain.
DomainNet uses 690 per environment instead: 690 is divisible by all locked
ratio denominators and at OOD=10% allocates 69 unknown examples, one for each
of the split's 69 semantic unknown classes. This avoids a natural-domain
secondary benchmark whose low-OOD condition represents only a random subset
of novelty classes. The six DomainNet environments therefore contribute 4,140
source examples before stream scheduling.

## Configurations

The planner fails closed unless the following DomainNet-specific configs are
present and pinned by content hash: `Ramen`, `EntropyGatedLatentRamen`,
`OracleDropOODRamen`, `OracleIDGradientRamen`, `ConsensusRamen`, and
`OracleConsensusRamen`. The new configs preserve DomainNet/Ramen's capacity
300, top-k 10, beta 5, SignSGD learning rate 0.01. Consensus uses the already
locked v0 `hard_mask`, threshold 0.2, and minimum three active classes.

## Validation

```text
PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest \
  tests/test_open_set_domainnet_matrix.py
python -m compileall -q src/runtime/open_set_domainnet_matrix.py
git diff --check
```

These checks validate only planning, config hashing, exact pairing, commands,
and canonical-constraint rejection. They are not DomainNet execution evidence.
At this date the repository still has no verified DomainNet six-environment
data tree and this host has no NVIDIA CUDA runtime; no accuracy, OOD metric,
or thesis claim is reported here.
