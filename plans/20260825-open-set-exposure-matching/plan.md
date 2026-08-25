# Open-set exposure matching

Status: approved for implementation (scoped subtask)

## Goal

Add an opt-in, exact per-domain source budget for open-set streams so ratio
comparisons can keep total per-domain exposure constant without changing the
legacy unbudgeted selection behavior.

## Scope

- Update `src/streams/builders.py` and `src/main.py` plumbing.
- Add focused open-set stream tests and a completion report.
- Do not change matrices, datasets, evidence, or project docs.

## Acceptance criteria

- A valid budget yields the same source count in every domain across ratios.
- Per-domain known/OOD counts exactly realize the requested ratio.
- Selection and fingerprint are deterministic and fingerprint metadata records
  requested and realized budget/counts.
- Invalid, non-integral-ratio, and infeasible budgets fail clearly.
- Omitted budget preserves the existing largest-feasible selection behavior.
