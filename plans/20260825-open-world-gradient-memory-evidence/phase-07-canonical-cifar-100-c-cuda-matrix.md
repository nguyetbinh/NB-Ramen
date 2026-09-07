---
phase: 7
title: "Canonical CIFAR-100-C CUDA matrix"
status: blocked
priority: P1
effort: "External GPU job"
dependencies: [5]
---

# Phase 7: Canonical CIFAR-100-C CUDA matrix

## Overview

Collect the primary canonical evidence without changing the preregistered
method/config/protocol after viewing outcomes.

## Requirements

- [ ] Use Linux x86_64 NVIDIA CUDA 12.1-compatible host, ≥24 GiB VRAM and persistent evidence/data storage; install `environment-cuda.yml` unchanged.
- [ ] Require official CIFAR-100-C Zenodo 3555552, MD5 `11f0ed0f1191edbf9fa23466ae6021d3`, verified model/data provenance and deep preflight.
- [ ] Execute 252 ordered cells: seven methods × four ratios × three streams × three seeds. Each cell runs NoAdapt before every adapted method and shares exactly one fingerprint.

## File ownership / touchpoints

- No planned source changes. Execute `environment-cuda.yml`, `src/runtime/preflight.py`, `src/runtime/experiment_matrix.py`, `src/main.py` and frozen `cfg/`.
- Create non-git: `$RAMEN_EVIDENCE_ROOT/open-set-cifar100c-canonical/{run_id}/` plus `runtime/{preflight,provenance,nvidia-smi,scheduler}.json|log`.

## Implementation Steps

1. Provision the runner and archive hardware/runtime snapshot; fail if `torch.cuda.is_available()` is false.
2. Acquire/verify data and checkpoint; run deep preflight and archive its output.
3. Pass focused/full tests and run all seven methods on the official CUDA v1
   smoke: OOD=.3, block, seed=0, block size=64, 400 source/domain, prefix=256,
   batch=100, CLIP ViT-B/16, fast provenance, trace v3 / summary v4. Use direct
   main.py runs, frozen config locks, and paired NoAdapt references.
4. Run Ramen B=100 / Ramen B=1 / CausalRamen B=100 sensitivity; create a
   separate NoAdapt B=1 reference. Also run paired v2 EntropyGatedRamen and
   OOD=0 OracleID controls. Strictly validate every smoke without tuning.
5. Freeze a clean revision only after all gates pass; verify config digests and
   dry-plan exactly 252 runs. See [readiness report](./reports/pre-canonical-readiness-20260907.md).
6. Launch the fixed matrix with resume, preserving full streams and baseline ordering; validate every completed run before scheduling dependent analyses.

## Todo

- [ ] `conda env create -f environment-cuda.yml && conda activate nb-ramen-cuda`
- [ ] `PYTHONPATH=src python -m runtime.preflight --data-root "$RAMEN_DATA_ROOT" --dataset CIFAR100C --deep --json > "$RAMEN_EVIDENCE_ROOT/runtime/cifar100c-deep-preflight.json"`
- [ ] `PYTHONPATH=src python -m runtime.experiment_matrix --open-set-consensus --device cuda --artifact-provenance fast --data-root "$RAMEN_DATA_ROOT" --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-cifar100c-canonical" --execute --resume`

## Success Criteria

- [ ] All 252 completed runs have manifests, stream, v3 trace, v4 summary, official provenance and exact paired fingerprint.
- [ ] Strict analyzer classifies the full grid `canonical_cuda_expected`.
- [ ] Canonical results include primary utility, pre/post safety, oracle/consensus mechanism, ID stability, synchronized latency/throughput/memory, and paired overhead.

## Risks and rollback

The primary protocol is **legacy-compatible batch-atomic mixed-domain TTA**:
batch=100 may cross block=64 boundaries. Strict causality is a separate
control, never an architecture change to the canonical comparison.

**Current blocker (2026-09-07):** no attached CUDA worker; local PyTorch has
no CUDA. Official CIFAR-100-C at `/Users/admin/data` and the ViT-B/16 checkpoint
pass local deep preflight and fast provenance. Current-schema CUDA smokes,
causal sensitivity, the clean freeze, and full execution remain pending. Do not substitute MPS/CPU or pilot arrays. Resume
only valid completed runs; quarantine any failed/mutated run directory and
re-run it with the same frozen identity.
