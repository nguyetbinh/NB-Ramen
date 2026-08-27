---
phase: 8
title: "Split robustness study"
status: pending
priority: P1
effort: "CUDA follow-on"
dependencies: [7]
---

# Phase 8: Split robustness study

## Overview

Test whether the CIFAR finding depends on the contiguous v1 semantic split.

## Requirements and ownership

- Create two fixed name-rank 80/20 recipes (`v2`, `v3`) using SHA256 of a
  versioned salt plus class name; never select by observed accuracy.
- Modify: split config/loader and a reduced robustness planner; test split
  determinism, disjoint 80/20 partition, identity/fingerprint binding, and
  matrix coverage.
- Run ratios `.3/.5`, `block/recurring`, seeds `0/1/2`: 12 paired cells per
  split, each with one NoAdapt baseline plus Ramen, ConsensusRamen, and
  OracleIDGradientRamen. This schedules 48 runs per split (12 NoAdapt + 36
  adapted), 96 runs total, on verified CUDA data.

## Implementation Steps

1. Materialize and commit versioned recipes before execution; preserve v1 unchanged.
2. Plan/validate full paired cells, then run after phase 7 is validated.
3. Compare GDC/SDR, oracle gap, consensus gain, post safety, and ID stability by split.

## Todo

- [ ] `PYTHONPATH=src python -m unittest tests.test_open_set tests.test_stream_builders tests.test_experiment_matrix`
- [ ] Archive each completed artifact quartet and split recipe/hash under `$RAMEN_EVIDENCE_ROOT/open-set-split-robustness/`.

## Success Criteria

- [ ] Both new deterministic splits complete with paired fingerprints and no retuning.
- [ ] Final synthesis reports heterogeneity rather than pooling away a split failure.

## Risks and rollback

The split generator is a protocol change: do not overwrite v1 or introduce a
random seed without a recipe/version. Reject incomplete split matrices.
