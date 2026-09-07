# Kaggle full evidence delivery

Status: implemented and locally validated; full CUDA execution belongs to the
delivered notebook and has not been run on this non-CUDA host.

## Scope

Deliver a standalone Kaggle notebook for the fixed 252-run primary CIFAR-100-C
matrix at validated source `26a7cd7`, retaining full streams, B=100, frozen
config locks, official provenance, and same-cell NoAdapt pairing. Do not alter
the separate in-progress oracle support utility implementation.

## Completed work

- Reused the smoke runtime/dependency setup and official data acquisition.
- Added session-specific logs, frozen campaign identity, strict resume,
  interrupted-run preservation, atomic ZIP checkpoint/restore, and safe
  process-group termination on notebook interruption.
- Added complete-coverage descriptive JSON and per-cell CSV only after all
  252 validated runs; no automatic scientific efficacy certification.
- Added usage/resume documentation and eight real filesystem/subprocess tests.
- Reviewed independently; fixed session device-record location and notebook
  interruption cleanup. Focused re-review found no remaining actionable issue.

## Acceptance evidence

- Eight helper tests pass, including interruption of a child process tree.
- Generated notebook validates with nbformat; code cells and all five embedded
  programs parse. Regeneration is deterministic.
- Plan-only execution against a source snapshot of `26a7cd7` produces exactly
  252 full-stream CUDA runs with matching baseline cells and config locks.
- Original 354 CUDA tests/14 smoke results apply to the pinned experiment
  source; they are not a claim that this new full campaign has executed.

Artifacts: [notebook](../../notebooks/kaggle/kaggle-full-matrix.ipynb),
[guide](../../notebooks/kaggle/full-run-guide.md).
