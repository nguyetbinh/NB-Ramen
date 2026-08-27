# Thesis implementation audit — 2026-08-25

## Scope and result

Reviewed the active implementation against
`docs/research/open-world-gradient-memory-thesis-report.md`, concentrating on
the evaluator/deployment boundary, Ramen/Consensus aggregation, open-set
metrics, and the CIFAR-100-C/DomainNet experiment planners.

The deployment-path label boundary is implemented correctly: the stream places
`is_ood` in the evaluator-only tail ([builders.py:66-70](../../../src/streams/builders.py#L66-L70));
the evaluator passes it only to models explicitly declaring the oracle hook
([main.py:205-212](../../../src/main.py#L205-L212)); and `ConsensusRamen` has no such hook
([ConsensusRamen.py:205-316](../../../src/methods/ConsensusRamen.py#L205-L316)).  The
oracle cache intentionally retains provenance only in the named oracle method
([OracleIDGradientRamen.py:82-126](../../../src/methods/OracleIDGradientRamen.py#L82-L126)).

However, the implementation is **not ready for thesis-grade canonical runs**.
The report's F1--F3 requirements remain materially unimplemented, and the
CIFAR planner can emit noncanonical grids while calling itself canonical.

## Prioritized findings

### P0 — required Consensus-vs-oracle mechanism evidence is absent

**Evidence.** `aggregate_oracle_supports()` computes only all-support and
ID-only sums ([OracleIDGradientRamen.py:159-198](../../../src/methods/OracleIDGradientRamen.py#L159-L198));
the method applies the latter ([OracleIDGradientRamen.py:250-256](../../../src/methods/OracleIDGradientRamen.py#L250-L256)).
It neither retains per-class all-support contributions nor derives a
label-free Consensus diagnostic direction.  The only oracle trace fields are
the Ramen-vs-ID cosine and SDR ([evidence.py:68-73](../../../src/evaluation/evidence.py#L68-L73)),
and the summary consequently reports only those two measures
([main.py:752-767](../../../src/main.py#L752-L767)).  The strict matrix
validator mirrors this incomplete contract ([experiment_matrix.py:1086-1121](../../../src/runtime/experiment_matrix.py#L1086-L1121)).

**Impact.** The central thesis claim—Consensus approximates the ID-only
oracle without target labels—cannot be tested.  Existing metrics establish
that OOD can alter Ramen's direction, not that the deployable method moves it
back toward the oracle.

**Narrow fix.** In `OracleIDGradientRamen.aggregate_oracle_supports`, retain
per-class all-support contributions, invoke the same hard-mask calculation as
`ConsensusRamen` using those contributions only, and calculate its cosine/SDR
against the separately evaluator-derived ID vector.  Add the five prescribed
trace fields plus `ramen_gdc_mean`, `consensus_gdc_mean`, `ramen_sdr_mean`,
`consensus_sdr_mean`, and reductions; reject partial groups in both evidence
validators.  Add synthetic tests proving: (1) the diagnostic mask is invariant
to OOD flags when all supports/gradients are held fixed, (2) a known vector
reduces SDR, and (3) the actual OracleID applied gradient is unchanged.

### P0 — post-adaptation OOD safety is not measured

**Evidence.** Methods preserve only a pre-update energy score (for example,
[Ramen.py:145-149](../../../src/methods/Ramen.py#L145-L149)); the evaluator receives the
returned, post-update logits ([main.py:374-385](../../../src/main.py#L374-L385)) but never
calculates `-logsumexp(logits)`.  Open-set trace fields contain only
`pre_adaptation_ood_score` ([evidence.py:60-67](../../../src/evaluation/evidence.py#L60-L67)),
and the summary exclusively builds detection from it
([main.py:627-666](../../../src/main.py#L627-L666)).  Resume validation repeats the same
pre-only computation ([experiment_matrix.py:1014-1084](../../../src/runtime/experiment_matrix.py#L1014-L1084)).

**Impact.** Reported AUROC/FPR95/H-score mostly characterize the shared reset
base model, so they cannot support a claim about whether a TTA update improves
or harms OOD safety.

**Narrow fix.** Calculate and trace `post_adaptation_ood_score` directly from
the returned logits in `ordered_stream_test`; create explicit
`pre_adaptation_detection` and `post_adaptation_detection` summary blocks and
validate both.  Add tests that NoAdapt pre/post agree (within tolerance),
adapted logits change post detection when appropriate, and OOD=0 produces a
well-defined unavailable block.

### P1 — the primary entropy control specified by the thesis does not exist

**Evidence.** The primary CIFAR matrix still names
`EntropyGatedLatentRamen` ([experiment_matrix.py:101-104](../../../src/runtime/experiment_matrix.py#L101-L104));
DomainNet does the same ([open_set_domainnet_matrix.py:43-51](../../../src/runtime/open_set_domainnet_matrix.py#L43-L51)).
`methods/__init__.py` registers no `EntropyGatedRamen`
([__init__.py:1-18](../../../src/methods/__init__.py#L1-L18)).  The available
entropy method subclasses `LatentRamen` and uses `StructuredGradientMemory`
and context routing ([EntropyGatedLatentRamen.py:11-16](../../../src/methods/EntropyGatedLatentRamen.py#L11-L16),
[EntropyGatedLatentRamen.py:69-76](../../../src/methods/EntropyGatedLatentRamen.py#L69-L76)),
which is not the report's ordinary-Ramen-only admission control.

**Impact.** Any comparison labelled “prediction-confidence filtering versus
gradient compatibility” has a confounded baseline.

**Narrow fix.** Add `EntropyGatedRamen` as a direct Ramen-compatible cache
implementation with only normalized-entropy admission changed; add both
dataset configs and replace the two matrix constants.  Cover all-admitted
equivalence to Ramen, empty rejected history/no update, mixed batches,
self-retrieval for admitted current samples, non-retrieval for rejected ones,
reset, and absence of the oracle hook.

### P1 — open-set stability statistics count OOD rows as forced classification failures

**Evidence.** For an OOD row, `label == -1` ([main.py:351-365](../../../src/main.py#L351-L365))
and `correct` is set by `pred.eq(label)` ([main.py:380-389](../../../src/main.py#L380-L389));
with a known-class classifier this is necessarily false.  Those generic
correctness values are appended for every row ([main.py:533-545](../../../src/main.py#L533-L545)),
then fed unchanged to recovery ([main.py:685-695](../../../src/main.py#L685-L695)) and
the paired negative-adaptation comparison ([main.py:701-708](../../../src/main.py#L701-L708)).
`compare_trace_negative_adaptation` likewise windows every aligned row without
checking `is_ood` ([evidence.py:484-510](../../../src/evaluation/evidence.py#L484-L510)).

**Impact.** Negative-adaptation windows and persistent-domain recovery are
partly functions of OOD placement, not ID adaptation stability.  This directly
violates the report's ID-only stability requirement and can hide or exaggerate
method differences when OOD order changes.

**Narrow fix.** Add an ID-filtered paired comparison that verifies both traces'
`is_ood` identity, filters before window construction, and records retained ID
counts.  For recovery, retain episode boundaries but compute baseline/recovery
windows over the ID subsequence within each episode; emit an explicit
insufficient-ID state.  Make the open-set analysis require these ID-only
blocks, while retaining current closed-set metrics for legacy runs.

### P1 — the CIFAR “canonical” planner does not lock the protocol it claims to plan

**Evidence.** Unlike the DomainNet planner, which requires exact streams,
ratios, seeds, methods, CUDA, full streams, and its fixed source budget
([open_set_domainnet_matrix.py:68-89](../../../src/runtime/open_set_domainnet_matrix.py#L68-L89)),
the CIFAR planner accepts arbitrary subsets of streams/ratios/seeds
([experiment_matrix.py:439-451](../../../src/runtime/experiment_matrix.py#L439-L451)),
allows any positive divisible source budget rather than 400
([experiment_matrix.py:454-459](../../../src/runtime/experiment_matrix.py#L454-L459)),
accepts `max_eval_samples`, and returns without asserting 252 runs
([experiment_matrix.py:461-490](../../../src/runtime/experiment_matrix.py#L461-L490)).
It also has no CIFAR counterpart to DomainNet's fixed-config/locked-consensus
validation ([open_set_domainnet_matrix.py:92-120](../../../src/runtime/open_set_domainnet_matrix.py#L92-L120)).

**Impact.** A truncated, altered-exposure, or altered-config run can be
generated from the canonical entry point and mistaken for canonical evidence.
Later analysis labels incomplete coverage as a pilot, but that does not prevent
mis-scheduling or protect the frozen protocol at planning time.

**Narrow fix.** Either split this into an explicitly named flexible pilot
planner and a strict canonical planner, or make this function enforce the
full tuple, `per_domain_source_budget == 400`, `max_eval_samples is None`,
required configs, and locked Consensus values.  Assert 252 planned runs; unit
test every rejected deviation.  Preserve the existing separate pilot/ablation
interfaces for smaller development grids.

## Additional implementation notes

- **Consensus math matches the stated hard-mask mechanics.** Per-class weighted
  contributions are summed, class-balanced by their mean, and agreement is
  based on signs ([ConsensusRamen.py:117-175](../../../src/methods/ConsensusRamen.py#L117-L175)).
  Below the minimum class count, it returns ordinary Ramen
  ([ConsensusRamen.py:141-150](../../../src/methods/ConsensusRamen.py#L141-L150)).
  This part is supported by mechanics tests in
  `tests/test_consensus_ramen.py`.
- **The soft ablation is SignSGD-valid.** It uses seeded Bernoulli coordinate
  admission rather than positive magnitude scaling
  ([ConsensusRamen.py:153-173](../../../src/methods/ConsensusRamen.py#L153-L173)).
- **No leakage defect found in the ordinary deployment method.** The current
  protections are worth preserving in new F1 diagnostics: derive the
  diagnostic consensus mask from all-support gradients before consulting OOD
  flags, and keep `set_oracle_is_ood` unavailable on `ConsensusRamen`.

## Verification performed

Static inspection and targeted test invocation were completed.  The focused
suite command was:

```text
python -m unittest tests.test_oracle_id_gradient_ramen tests.test_consensus_ramen tests.test_open_set_consensus_analysis tests.test_experiment_matrix
```

The non-Torch tests completed, but the two Torch-dependent modules could not
import because the local Python environment is missing
`libtorch_cpu.dylib` (Torch `_C` load failure).  This is an environment blocker
for executing mechanics tests here, not evidence that those tests pass.

## Minimum merge gate before canonical CUDA runs

1. Implement and validate P0 mechanism and post-adaptation detection evidence.
2. Replace the confounded entropy baseline and verify the seven-method matrix.
3. Make stability ID-aware and harden the CIFAR canonical planner.
4. Run the complete focused unit suite in a working Torch environment, then a
   small deterministic open-set smoke run to validate the complete trace and
   summary contracts.

Status: DONE_WITH_CONCERNS

Summary: The evaluator-only OOD boundary and hard Consensus mechanics are sound, but missing F1/F2/F3 evidence and a permissive CIFAR canonical planner block thesis-grade runs.

Concerns/Blockers: Local Torch is broken (`libtorch_cpu.dylib` missing), so Torch-dependent tests could not be executed in this environment.
