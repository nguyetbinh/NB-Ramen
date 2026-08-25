# Phase 1: exact budgeted source selection

Modify the open-set builder to accept `per_domain_source_budget`.  Validate it
as a positive integer, require its ratio-derived known/OOD counts to be whole
numbers, and ensure every domain has enough samples.  Deterministically choose
from label-balanced pools, schedule those selected references, and serialize
the requested and realized allocations under `metadata.open_set`.

Update CLI parser/plumbing and add dependency-light tests covering equal
exposure across ratios, exact counts, determinism/fingerprints, and validation.

Validation: `python -m unittest tests/test_open_set.py`.
