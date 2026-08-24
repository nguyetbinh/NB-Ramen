# Phase 01 — Reproducible Streams and Evidence

Status: in progress

## Context

- [Research roadmap](../../docs/research/ramen-thesis-research-roadmap.md)
- [Current evaluation entrypoint](../../src/main.py)
- [Current dataset wrapper](../../src/datasets/utils.py)

## Requirements

- [x] Add deterministic schedules for all eight roadmap stream types.
- [x] Preserve `iid_mixed` as the original randomized-mixture comparison, with its order fixed by seed.
- [x] Save the exact stream order metadata and a stable, independently verifiable fingerprint.
- [x] Emit a versioned JSONL trace containing the roadmap fields for every evaluated sample.
- [x] Emit a run manifest with code, arguments, config, runtime, device, dataset, and stream metadata.
- [x] Define primary online metrics precisely and compute them when their required evidence is available.
- [x] Keep the stream/evidence utilities testable without Torch or datasets.
- [x] Integrate the utilities into mixed- and single-domain entrypoints without changing reset semantics.
- [x] Provide separately resolvable CPU/MPS and Linux CUDA environment manifests.
- [x] Run the real CLIP model and LatentRamen method on MPS with decoded image files.
- [x] Run an end-to-end smoke through a supported benchmark dataset wrapper.
- [ ] Run the Phase 1 baseline matrix on DomainNet and CIFAR100C.

## Files

- Create `src/streams/`.
- Create `src/evaluation/`.
- Modify `src/main.py`.
- Add focused tests under `tests/`.
- Add a pinned runtime specification and dataset validation in a later Phase 01 slice.

## Validation

- Run dependency-free unit tests for schedules, schemas, and metrics.
- Run syntax compilation across `src/`.
- Run a synthetic end-to-end evaluation once the dependency-light fixture is available.
- Run dataset/model smoke tests through the Luna task or another verified Torch runner.

## Risks and rollback

- Stateful methods make order part of the experiment contract; never silently shuffle after constructing a schedule.
- Subsampling modes must record dropped samples and effective domain counts.
- Ground-truth domain metadata must not leak into LatentRamen routing. Oracle variants must remain separately named.
- If integration affects baseline results, retain a legacy-order reproduction option until the difference is explained empirically.
