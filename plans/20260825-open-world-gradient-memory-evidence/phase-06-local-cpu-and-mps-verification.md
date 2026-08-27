---
phase: 6
title: "Local CPU/MPS mechanics verification"
status: pending
priority: P2
effort: "0.5d"
dependencies: [5]
---

# Phase 6: Local CPU/MPS mechanics verification

## Overview

Exercise the frozen interfaces on available hardware and archive reproducible
noncanonical artifacts. This is a release-quality mechanics gate, never a
substitute for canonical CUDA effectiveness or cost evidence.

## Requirements

- [ ] Verify CPU and, when available, MPS execution, resume/evidence validation, method registration, paired fingerprints, and report generation using the local pilot artifact only.
- [ ] Label every artifact `noncanonical_pilot`: prefix ≤200, provenance off,
  pilot data, and non-CUDA are not allowed in thesis tables.

## File ownership / touchpoints

- No source edits expected. Create: non-git `$RAMEN_EVIDENCE_ROOT/local-{cpu,mps}/` artifacts and `reports/local-mechanics-YYYYMMDD.md` under this plan directory.
- Read: `environment.yml`, `docs/research/experiment-runtime.md`, phase 02--05 contracts.

## Implementation Steps

1. Run focused unit suite, then direct NoAdapt baseline followed by ConsensusRamen and OracleID diagnostic run on identical short streams.
2. Verify v2 artifacts, optional groups, equal fingerprints, pre/post blocks, ID-only stability fields, and analyzer classification.
3. Repeat minimal deterministic test on CPU; use MPS only if it passes numerical/tolerance rules.

## Todo

- [ ] `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*open_set*.py'`
- [ ] Direct baseline-first commands use `--device cpu`/`mps`, `--max_eval_samples 128`, unique local evidence roots, and `--artifact-provenance off` only because the available data are a pilot.
- [ ] `PYTHONPATH=src python -m evaluation.open_set_consensus_analysis --evidence-dir "$RAMEN_EVIDENCE_ROOT/local-mps" > "$RAMEN_EVIDENCE_ROOT/local-mps/analysis.json"`

## Success Criteria

- [ ] All local artifacts validate or are retained as rejected diagnostics with the failure reason.
- [ ] Analyzer returns `noncanonical_pilot`; that status is recorded, not overridden.
- [ ] No local latency, allocator, accuracy, or detection result is copied into a CUDA thesis conclusion.

## Risks and rollback

MPS numerical differences and pilot data can create misleading apparent
effects. Treat deviations as debugging signals; do not retune frozen settings
or overwrite previously archived pilot artifacts.
