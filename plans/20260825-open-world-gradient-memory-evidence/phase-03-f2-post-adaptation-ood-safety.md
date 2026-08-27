---
phase: 3
title: "F2: Post-adaptation OOD safety"
status: pending
priority: P1
effort: "0.75d"
dependencies: [2]
---

# Phase 3: F2 — Post-adaptation OOD safety

## Overview

Separate base-model OOD separability from the safety effect of the temporary
TTA update, using logits already returned by each method.

## Requirements

- [ ] Record `post_adaptation_ood_score = -logsumexp(returned_logits)` in every open-set trace beside the existing pre score.
- [ ] Emit explicit `pre_adaptation_detection` and `post_adaptation_detection` blocks (AUROC, FPR95, H-score, OOD recall at FPR95); OOD=0 remains unavailable.
- [ ] Preserve old top-level pre-detection fields only as a temporary compatibility mapping and label all final tables explicitly.

## File ownership / touchpoints

- Modify: `src/main.py`, `src/evaluation/evidence.py`,
  `src/evaluation/open_set_metrics.py`, `src/evaluation/open_set_consensus_analysis.py`, `src/evaluation/__init__.py` if exports change.
- Test: `tests/test_open_set_metrics.py`, `tests/test_evidence.py`,
  `tests/test_ordered_stream_evidence.py`, `tests/test_open_set_consensus_analysis.py`.

## Implementation Steps

1. Capture logits from the existing method call and calculate the score in the evaluation loop—no second forward.
2. Version/validate the all-or-none trace extension and recompute both detection blocks from trace rows.
3. Make analyzer input strict about status, unavailable reason, and numerical values.
4. Test NoAdapt pre/post equality within tolerance and an adapted fixture with intentionally changed post logits.

## Todo

- [ ] `PYTHONPATH=src python -m unittest tests.test_open_set_metrics tests.test_evidence tests.test_ordered_stream_evidence tests.test_open_set_consensus_analysis`
- [ ] Cost-limited local direct smoke after implementation: `PYTHONPATH=src python src/main.py --dataset CIFAR100C --open_set --known_class_split open-set-cifar100-split-v1 --ood_ratio .3 --tta_mode mixed --stream_mode block --tta_algo NoAdapt --model clip_vitbase16 --seed 0 --device mps --data_root "$RAMEN_DATA_ROOT" --artifact-provenance off --max_eval_samples 128 --evidence_dir "$RAMEN_EVIDENCE_ROOT/local-f2" --run_id local-f2-noadapt`.

## Success Criteria

- [ ] No extra forward appears in call-count tests.
- [ ] NoAdapt pre/post scores match; adapted post detection may differ.
- [ ] OOD=0 has explicit unavailable values, never fabricated metrics.
- [ ] Trace/summary strict validation rejects missing post score or only one detection block.

## Risks and rollback

Score orientation must stay “larger means more OOD.” Validate against existing
metric fixtures; if compatibility needs a transition, keep old fields as a
read-only alias rather than mixing pre/post values.
