# Consensus ablation and retained-memory review — 2026-08-25

## Scope and result

Read-only review of `ConsensusRamen` v0, the `ConsensusRamenSoft` and
`ConsensusRamenNoSelf` identities, ordinary `Ramen` retained-memory accounting,
and the open-set analyzer against thesis report §§13–26.  The focused suite

```text
PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest -q \
  tests.test_consensus_ramen tests.test_consensus_ramen_soft_alias \
  tests.test_ramen_memory_bytes tests.test_consensus_evidence \
  tests.test_open_set_consensus_analysis tests.test_oracle_consensus_ramen
```

passed: 36 tests.

The v0 causal ordering, hard-mask arithmetic, below-minimum fallback,
method/evaluator separation, and separately hashed configuration identities are
implemented consistently.  The two findings below should be addressed before
presenting the v1 ablation or retained-memory timeline as final evidence.

## Findings

### P1 — `ConsensusRamenSoft` is observationally the same SignSGD update as Ramen for every nonzero-agreement coordinate

Thesis §17 defines the requested soft weighting, but §17 also warns that the
eventual SignSGD step makes this potentially ineffective.  The implementation
does exactly that multiplication at
`src/methods/ConsensusRamen.py:146-151`; the configured optimizer then takes
the sign of every nonzero gradient at `src/models/optimizer.py:99-103`.

Consequently, for every coordinate with ordinary Ramen gradient `g != 0` and
agreement `q > 0`, `sign(q**gamma * g) == sign(g)`.  With the locked
state-free `signsgd` configuration (no momentum or weight decay), the
parameter update is identical to Ramen.  Only exactly zero agreement can
change the step.  The present test asserts that the soft gradient preserves
sign (`tests/test_consensus_ramen.py:56-74`), but never checks the actual
SignSGD parameter delta; that is precisely the invariant which exposes this
issue.

Impact: a measured difference attributed to continuous soft weighting would
not have the claimed mechanism.  The existing one-cell pilot difference can
only arise from exact-zero coordinates, numerical effects, or another
non-identical run property, not linear attenuation itself.

Recommended disposition: retain the implementation only as an explicitly
"zero-consensus coordinate deletion" ablation, add an optimizer-level
equivalence test, and do not interpret it as a graded soft-weight experiment.
If a genuinely graded v1 test is required, use an optimizer/update rule where
magnitude survives, or define a soft mechanism that changes signs/support
selection before SignSGD.  This is a design correction, not a rationale to
alter the locked v0 method.

### P2 — Per-sample retained-memory timeline is actually a post-batch snapshot

`Ramen.forward` admits an entire batch before retrieving and reports one scalar
`memory_bytes`; `ConsensusRamen` has the same v0 ordering
(`src/methods/ConsensusRamen.py:231-253`).  The runner expands that scalar to
every sample of the batch (`src/main.py:158-171`, `src/main.py:395-408`).
However, the strict summary contract labels the resulting trace as bytes
retained "after each causal sample update"
(`src/runtime/experiment_matrix.py:844-858`).  For batch size greater than
one, early rows instead contain the memory after *all* members of that forward
were admitted.

Impact: final and maximum retained bytes remain correct, but the timeline is
not causal per sample and must not be used to infer sample-level memory growth
or attribute retained memory to an individual support.  Existing memory tests
verify entry-byte arithmetic and the one scalar diagnostic, but not a
multi-sample trace with this timing distinction.

Recommended disposition: either (a) change the evidence definition and
documentation to "post-forward/batch retained bytes, repeated for its rows",
which matches current behavior and requires no method change; or (b) emit a
true per-insertion memory timeline and adjust the v0 batch-atomic contract
deliberately.  Option (a) is the minimally faithful correction.

### P2 — §26's min-class and tau sweeps are not yet executable as named, paired ablation matrices

The validation surface accepts alternate `min_consensus_classes` and
`consensus_threshold` values, and the current v0, soft, and no-self YAMLs are
correctly distinct.  But only `ConsensusRamenSoft` and
`ConsensusRamenNoSelf` have separately selectable configuration identities.
There are no configuration identities or matrix planner inputs for §26(F)
varying minimum active classes and §26(G) varying tau; the canonical planner
intentionally runs only the locked v0 configuration.  The existing completion
audit independently marks these evidence requirements as missing.

Impact: the code has a tunable surface, but it cannot yet produce auditable,
paired evidence for two required ablations without ad-hoc configuration edits.
Those edits would weaken config-hash/run-identity guarantees.

Recommended disposition: preregister explicit YAML identities (or an
immutable, planner-recorded grid) for the chosen values, preserve v0's locked
`.2/3` primary cell, and schedule them outside the canonical primary matrix.

## Verified non-findings

- **Causal ordering:** v0 admits the full current batch before retrieval, as
  ordinary Ramen does.  No-self retrieves history before admitting the current
  batch.  Mechanic tests prove both orders.
- **Hard-mask semantics:** class contributions are unnormalised Ramen
  contributions; vote uses `sign`, zero is neutral, and the hard mask is
  applied to the ordinary class-balanced gradient before SignSGD.  This matches
  §§15–17 and the v0 invariants.
- **Fallback:** fewer than `min_consensus_classes` returns the ordinary Ramen
  aggregate and records `consensus_applied=false`.
- **Configuration identity:** aliases select their own YAML file and therefore
  have distinct path/hash/config fields in the experiment manifest; they only
  share the Python implementation intentionally.
- **Evaluator-label isolation:** `ConsensusRamen` has no oracle OOD hook;
  `main.py` only provides `is_ood` to a method that explicitly declares the
  oracle requirement.  The ordinary open-set analyzer rejects evaluator
  context for ConsensusRamen.  No label-isolation issue was found.
- **Diagnostics:** consensus summary agreement/mask metrics are aggregated on
  `consensus_applied` rows only, while the all-sample active-class mean is
  retained separately.  This correctly avoids representing fallback as a
  hard-mask decision.

## Evidence status

This is a source/mechanics review, not CUDA effectiveness evidence.  The
locked primary v0 matrix remains properly distinct from deferred ablations;
the findings affect interpretation and future ablation scheduling, not the
already implemented v0 causal/hard-mask contract.

## Follow-up resolutions

The controller addressed all three findings after this read-only review:

1. `soft_weight` now uses deterministic seeded Bernoulli coordinate admission
   with probability `q**gamma`, rather than a magnitude scale erased by
   SignSGD. Optimizer-delta tests cover `q=0`, `q=1`, repeatability, and an
   actual update difference. The pre-fix pilot is explicitly withdrawn.
2. The retained-memory summary definition now correctly reads “after each
   completed method forward; a batch snapshot is repeated for its samples.”
   It no longer claims a per-sample causal memory timeline.
3. `ConsensusRamenTau060`, `ConsensusRamenMin2`, and
   `ConsensusRamenMin4` now have immutable YAML identities. The separate
   `runtime.consensus_ablation_matrix` schedules their exact paired held-out
   runs without expanding the locked primary matrix.
