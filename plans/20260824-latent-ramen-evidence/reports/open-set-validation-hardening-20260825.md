# Open-set validation hardening — 2026-08-25

Status: DONE

`validate_completed_run` now requires every canonical open-set trace row to
carry complete evaluator-only evidence, validates the original-to-known label
mapping against exported stream metadata, and verifies realized ID/OOD counts
and ratio. It reconstructs `summary.open_set` from the trace, including the
explicit unavailable detection contract when no OOD samples exist.

Named oracle controls must provide complete oracle-gradient evidence and an
exactly matching diagnostic summary. `ConsensusRamen` must provide complete
consensus evidence and an exactly matching applied-only diagnostic summary;
other methods reject those diagnostics.

Validation: `conda run -n nb-ramen python -m unittest tests.test_experiment_matrix -q`
(51 tests passed).
