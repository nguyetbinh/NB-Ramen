# Open-set exposure matching report

Status: DONE

## Delivered

- Added optional `per_domain_source_budget` to `build_open_set_stream` and
  `--open-set-per-domain-source-budget` to the CLI.
- Budgeted streams allocate exact known/OOD counts from deterministic,
  class-balanced source-pool prefixes per domain.
- Stream fingerprint metadata now binds the requested source budget, realized
  selected source budget per domain, selected-pool counts, and emitted counts.
- The unbudgeted path retains the prior largest-feasible exact-ratio selection.
- Added toy-domain coverage for equal exposure over 0%, 25%, and 50% OOD,
  exact counts, deterministic references/fingerprints, and invalid budgets.

## Validation

- `python -m unittest tests/test_open_set.py` — 6 passed
- `python -m compileall -q src/streams/builders.py src/main.py` — passed
- `git diff --check` — passed
