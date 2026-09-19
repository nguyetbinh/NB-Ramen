# QCGS Label-Free Utility Diagnostic — Experiment Protocol

**Repository:** `nguyetbinh/NB-Ramen`\
**Source base:** `oracle-hf-workflows`\
**Working branch:** `qcgs-label-free-diagnostic`\
**Pinned base commit:** `3a80623b074f16b8ef87d8d5507427277ea2a54f`\
**Status:** CUDA diagnostic completed and artifacts validated; STOP for the frozen single-view entropy-sign signal (see execution report).\
**Revised:** 2026-09-19\
**Dataset/model:** CIFAR-100-C / CLIP ViT-B/16.

Execution requires a committed implementation, query registry, launch plan,
provenance hashes, and passing preflight checks. No new label-free results are
reported here. The selector, new stage registry, and their tests do not yet exist
at the pinned base. Proposed decisions below are protocol rules, not findings.

## 1. Question and scope

Can a query's unlabeled entropy gradient select a support replacement with
greater supervised query utility than Ramen/no-swap and a random legal swap?
The experiment tests a selection signal inside Ramen's fixed feature-local,
prediction-balanced candidate pool. It does not yet test a complete QCGS method,
unknown-class recognition, adaptive support size, or full top-M reranking.

For an ID query and baseline support set S, define evaluator-only utility:

\[
U_{\mathrm{sup}}(q,S\rightarrow S')
=CE_q(\theta_{\mathrm{Ramen}})-CE_q(\theta_{S'}).
\]

Positive utility means lower true query CE than ordinary Ramen. The selector
must choose without that CE, the true label, or any oracle-derived quantity.
The null hypothesis is that the single-view entropy signal provides no useful
selection advantage; the alternative is that it contains such information.
The pilot decision below does not constitute a statistical rejection of H0.

## 2. Historical evidence, separate from the new experiment

Pilot B used supervised query gradients and screened candidate verification.
Its historical numbers are retained exactly:

| Metric | OOD=0 | OOD=0.5 |
|---|---:|---:|
| Eligible ID queries | 128 | 128 |
| Queries with positive screened headroom | 114/128 (89.06%) | 120/128 (93.75%) |
| Ramen accuracy | 54.69% | 48.44% |
| Screened-oracle accuracy | 56.25% | 50.00% |
| Mean oracle CE gain | 0.046677 | 0.331788 |
| Mean supervised first-order gain | 0.034269 | 0.320014 |
| Mean supervised-gradient cosine gain | 0.015678 | 0.192436 |
| Mean random-swap gain | -0.009044 | -0.002948 |

An offline validation on 2026-09-17 recalculated all 16 numeric cells from
256 raw query records and 1,002 verified swaps; all matched at the displayed
precision. It did not rerun a GPU model or independently reproduce logits.

This is evidence of screened support-selection headroom, not an exhaustive
oracle optimum or a deployable label-free result. The two OOD cells have
different queries/cache histories; their difference is not a causal OOD effect.
The old screened candidates cannot establish unbiased candidate correlation.

## 3. Baseline and isolation contract

Lock the current runner's baseline defaults before new execution:

| Setting | Value |
|---|---|
| Dataset/model | CIFAR-100-C; CLIP ViT-B/16 |
| Open-set split | 80 ID / 20 OOD classes, `open-set-cifar100-split-v1` |
| OOD cells | 0.0 and 0.5 |
| Corruption severity | 5 |
| Stream | `block`, block size 64, stream seed 0 |
| Source budget | 400 samples per domain |
| Maximum stream rows | 600; a larger cap requires preflight justification and locking before scored execution |
| Batch size | 100 |
| Cache capacity | 750 per pseudo-class |
| Retrieval | k=5, M=10, beta=5.0 |
| Optimizer | SignSGD, lr=0.01, momentum=0, weight_decay=0, maximize=False |
| Runner timeout | 3,600 seconds; lock any preflight revision in the launch plan |
| Random-policy RNG | Dedicated generator, seed 0; isolated from model/stream/global RNG |

Keep cache admission, batch-atomic visibility, prediction-balanced retrieval,
entropy/distance weighting, active-class divisor, self-support eligibility,
parameter reset, and baseline aggregate dtype/order unchanged.

Selectors receive only query/cached gradients, weights, distances, pseudo-class
membership, and legal action keys. They cannot receive true labels, ID/OOD flags,
true corruption/domain, supervised gradients/CE, oracle utility, or derived
versions of those variables. Stable provenance is retained by the evaluator;
selectors receive canonical opaque action keys, not evaluator metadata.
Select label-free actions before computing evaluator-only references/utilities.
Only the evaluator filters ID query records for supervised CE. OOD queries are
not assigned artificial known-class CE targets.

## 4. Candidate space, scores, and numerical policy

For each active pseudo-class c, retrieve baseline top-k independently using
Ramen's exact path, with `min(k, class_size)` supports. Retrieve top-M from the
same cache snapshot. Build each pool by keeping the exact baseline indices
first, appending top-M indices not already present, then truncating the combined
pool to M. Extras are the entries after the baseline prefix. This preserves
the current probe's M-candidate cap even when tied distances give different
top-k/top-M memberships. Do not obtain baseline top-k by slicing top-M.
Retain probe eligibility: at least one pseudo-class has M=10 cached entries.

A legal action removes one baseline support and inserts one extra from the
same pseudo-class. Across all classes, a policy changes at most one support
per query; all other supports remain unchanged. Preserve support cardinality,
weights' baseline rules, and the active-class divisor. No-swap is always legal
with score and relative utility zero. Freeze canonical swap ordering from
pseudo-class, outgoing rank, incoming rank, and stable identity tie-breakers.

Copy the already-computed query gradient at pretrained parameters before
support aggregation or an update can overwrite it:

\[
h_q=\nabla_\theta H(p_\theta(y\mid x_q)).
\]

Let G(S) denote Ramen's exact weighted aggregate. The primary entropy-sign
score uses the change in the aggregate SignSGD direction:

\[
\boxed{score_{\mathrm{ent\text{-}sign}}(S\rightarrow S')
=\eta h_q^\top[sign(G(S'))-sign(G(S))],\qquad\eta=0.01.}
\]

The sign follows from comparing updates `theta0 - eta*sign(G)` and
`theta0 - eta*sign(G')`. This approximates entropy improvement relative to
Ramen; it does not guarantee supervised CE improvement.

| Policy | Selection rule |
|---|---|
| Ramen | No-swap |
| Random | Uniform forced choice over all legal swaps; no-swap only if none exist |
| Entropy-cosine | Maximize `cos(h_q,g_in) - cos(h_q,g_out)` |
| Entropy-sign, primary | Maximize the boxed score |
| Supervised first-order reference | Existing oracle scorer, evaluator-only |
| Exact best oracle | Stage A only: maximize measured utility over all legal swaps plus no-swap |

Both entropy policies execute their best swap only when its score is strictly
greater than zero. No-swap wins at zero; canonical ordering resolves positive
ties. Never tune this threshold on Stage A or Stage B.
Random selection is independent of all scores, labels, and verification results.
Record RNG state/draw order so deduplicating verification cannot alter draws.
The supervised first-order reference is not an exact oracle.

Compute dot products, cosine similarities, and norms in float32. Preserve
baseline aggregate dtype and summation order, including before `sign`; casting
the score operands must not silently alter the actual adaptation direction.
Record parameter ordering and the arithmetic comparison tolerance at preflight.

Numerical handling must be explicit and tested:

- Missing, shape-incompatible, or non-finite query gradients: no-swap with a
  distinct reason; flag the integrity failure for repair, not scientific STOP.
- Zero query-gradient norm: valid guarded no-swap, `zero_query_gradient`.
- Non-finite candidate gradients/aggregates or scores: fail closed to no-swap
  for the affected policy/query and flag numerical integrity failure.
- A zero support norm makes cosine undefined: cosine uses no-swap with
  `zero_candidate_norm`; the finite entropy-sign computation remains defined.
- No legal swaps, no positive score, a tied-zero score, and an unchanged
  aggregate sign are valid fallbacks recorded separately from invalid numerics.
- Non-finite baseline logits, CE, entropy, or verified utilities invalidate the
  run. Never remove failed queries and silently replace their observations.

## 5. Two stages and a committed query registry

Stage A evaluates **16 new eligible ID queries per OOD cell**, exhaustively
verifying every legal swap. Its purpose is arithmetic/score-direction checking,
candidate ranking, and exact one-swap headroom/regret. It is development data.

Stage B evaluates **128 new eligible ID queries per OOD cell** after code,
config, registry, and analysis rules are frozen. Verify no-swap and the actions
independently selected by entropy-sign, entropy-cosine, random, and supervised
first-order reference. Deduplicate identical actions only after selection.
Stage B is policy confirmation within this pilot, not independent-run evidence.

The registry must contain the full planned stream mapping and scored-query
assignment, not merely stream positions. Use:

- Base image identity: `(dataset fingerprint, original CIFAR sample_idx)`.
- Observation identity: base image identity plus corruption and severity.
- Execution identity: stage, OOD cell, stream seed, timestep, and batch index.

`query_index` or timestep alone is not an image identity. At this source pin,
the corruption loader preserves the original image index after severity slicing.
Map historical queries through `query_index -> trace.timestep -> trace.sample_idx`.
Historical OOD=0.5 has 128 records but only 126 distinct base images.

Exclude base images used as inspected queries in historical Pilot A/B and smoke
runs from both new stages. Include the new real-model smoke in that exclusion
set before finalizing the scored registry. Group Stage A/B separation by base
image across every corruption, severity, and OOD cell; a seed change is not a
substitute. Allow at most one scored observation per base image per OOD cell.
The same base image may occur in both OOD cells within a stage; report each cell
separately and do not treat cross-cell observations as independent replicates.

Use this prospective, outcome-independent assignment rule: compute SHA-256 of
the UTF-8 string `qcgs-label-free-v1|<dataset_fingerprint>|<sample_idx>` and take
the integer value of the digest modulo 9. Bucket 0 is Stage A; buckets 1-8 are
Stage B, matching the 16:128 query-budget ratio in expectation. The same base
image receives the same stage across both OOD cells. Apply exclusions first.
Within each unchanged stream, take the first 16/128 eligible ID observations
for its stage, skipping subsequent observations of an already selected base
image in that cell. Hashes and decimal sample indices have one canonical
encoding recorded in the registry schema.

Preflight must replay baseline eligibility, resolve the concrete assignments,
verify counts, and commit the resulting registry before utilities are opened.
Eligibility/assignment must not depend on policy score, correctness, supervised
CE, or observed gain. Record the exclusion manifest and source/data/stream
hashes. Do not search partition salts or seeds based on measured utility.
If identities or sufficient counts cannot be established, execution is blocked
until the registry or preflight-locked stream cap is corrected.

Registry assignment only controls which queries are scored. Replay the full
unchanged baseline stream: non-scored rows still enter the cache normally.
Do not remove registry-excluded samples from cache history or create artificial
histories to improve eligibility. Shared support-memory universes are allowed.
Reserve untouched base images for any revised signal or later method study.

## 6. Exact evaluation, ranking, and stage-specific regret

For every trial, start at the same pretrained parameters and cache snapshot,
install the candidate's aggregate, apply the same state-free SignSGD update,
and forward the original complete batch. Score the target query, then reset
parameters and optimizer state in a `finally` path, including on exceptions.
Trials must not mutate the continuing Ramen cache or output trajectory.

Record both `U_sup = CE_Ramen - CE_candidate` and
`U_ent = H_Ramen - H_candidate`. Entropy reduction without supervised utility
is an objective mismatch, not evidence of successful support selection.

**Stage A:** `exact_oracle_utility = max(0, all legal-swap U_sup)` and
`exact_oracle_regret = exact_oracle_utility - policy U_sup`.
These names require complete finite verification of the entire legal space.

**Stage B:** exact oracle utility/regret are **null, not measured**. If useful,
report `best_verified_utility = max(0, U_sup of verified actions)` and
`regret_to_verified_set = best_verified_utility - policy U_sup`.
The former is a lower bound on exhaustive oracle utility; the latter is a
nonnegative lower bound on exhaustive regret, since the policy action is in
the verified set. Neither is an exact oracle estimate.

Candidate-ranking correlation is primary only in Stage A. For each query and
entropy policy, compute Spearman between scores and exact U_sup over **all
legal swaps, excluding no-swap**. Use average ranks for ties. Return null plus
reason for fewer than two candidates, a constant score/utility vector, or
undefined values; never replace undefined correlation by zero.
Report valid/total query counts, mean/median valid rho, and positive rho count
divided by the number of valid correlations. Optional Kendall must state its
tie convention. Do not pool dependent swaps as independent observations.
Stage B score-screened actions do not support headline ranking correlations.

## 7. Metrics, sensitivity, and descriptive uncertainty

For each Stage B cell, let `U = U_sup(entropy_sign)` and let
`D = U_sup(entropy_sign) - U_sup(random)` be paired query-level differences.
Ramen/no-swap utility is zero, so U is already the paired advantage over Ramen.
Use all 128 registered observations, including valid no-swap outcomes.

For every policy report mean, median, P25/P75, helpful/harmful rates, no-swap
rate/reasons, and actual support-replacement count/coverage. Also report exact
accuracy, wrong-to-correct/correct-to-wrong counts, net corrections, mean/median
entropy utility, and the rate of lower entropy but worse supervised CE.
Rates use all registered queries unless another denominator is explicitly named.

For U and D separately in each cell, report:

- Ordinary mean and median.
- Symmetric 10% trimmed mean: sort 128 values, remove `floor(0.10*128)=12`
  from each tail, and average the remaining 104. This is sensitivity only.
- Mean after removing the two largest observations, with identity-based tie
  resolution. This one-sided robustness check is used by the decision gate.

Lock the following uncertainty reporting before execution:

1. Group scored queries by full stream block `floor(timestep/64)` separately
   within each OOD cell; keep all query/policy pairs together.
2. With dedicated analysis RNG seed 917, perform 10,000 draws. Each draw samples
   K nonempty scored blocks with replacement, where K is the observed number
   of such blocks. Recompute query-weighted U and D means as `sum / count`.
3. Report the 2.5th/97.5th percentile interval only when K>=4. Otherwise report
   null with `too_few_blocks`. Four blocks is an operational reporting guard,
   not evidence of independent blocks or adequate statistical power.
4. Always report K, query counts/block, blockwise U/D means, and the range of
   leave-one-block-out means; if undefined, report the reason.

These are conditional, descriptive resampling intervals for one seed/stream.
Shared cache history can create dependence beyond a 64-row block; resampling
does not reproduce new cache trajectories or establish independent-run or
generalization confidence. Do not bootstrap swaps as independent units.
Do not pool OOD cells for an independence-based interval. Interval availability
or exclusion of zero is not an additional GO_PILOT gate.

Stratify utility by baseline correct/wrong, self-support present/absent, query
entropy, query-gradient norm, and legal-swap count. Freeze entropy/norm terciles
and the swap-count median from Stage A before opening Stage B; collapse tied
boundaries deterministically and report actual bins/counts. Report random
comparison restricted to entropy-sign's actual-swap queries as a secondary
analysis of selection versus abstention, without changing the primary gate.

Separate deployable scoring latency/memory from evaluator-only supervised
backward passes and exact verification forwards. Report both cost categories.
Total oracle diagnostic runtime is not QCGS inference latency.

## 8. Ordered decision rule

Apply the first matching rule below. This order makes outcomes mutually
exclusive. All predicates refer to the frozen primary entropy-sign policy.
The rule and thresholds are prospective choices, not measured success.

1. **INVALID:** any label-isolation, baseline/candidate parity, reset, registry,
   provenance, arithmetic, or numerical integrity check fails. Repair and rerun
   into a new evidence directory; make no scientific inference from the run.
2. **INCONCLUSIVE:** registered stage/cell counts or required finite policy
   outcomes are incomplete for a non-integrity reason, such as a stopped run.
   Complete the locked protocol before interpreting its scientific outcome.
3. **GO_PILOT:** all of the following hold:
   - Mean U>0 and mean D>0 separately in both OOD cells.
   - Pooled entropy-sign net corrections across the two cells are >=0.
   - At least three actual entropy-sign support replacements occur per cell.
   - After removing the two largest observations, the remaining mean is >0
     separately for U and D in each cell.
4. **STOP:** mean U<=0 and mean D<=0 in both OOD cells. Stop investment in this
   single-view signal under the tested protocol; this does not erase oracle
   headroom or prove impossibility for every label-free signal.
5. **REVISE:** Stage A mean exact-oracle utility is >0 in both cells, and either
   exactly one Stage B cell has both mean U>0 and mean D>0, or a cell has mean
   entropy-sign U_ent>0 while mean U<=0. Try the one revision below.
6. **INCONCLUSIVE:** all remaining valid outcomes, including insufficient
   replacement coverage, gains that fail the two-observation check, or mixed
   evidence not meeting the preceding rules. Do not tune on these outcomes.

Three replacements is the minimum compatible with positive U surviving removal
of two positive swap observations; it is a protocol floor, not evidence of broad
coverage. Report actual counts and all sensitivity results even when GO passes.
GO_PILOT authorizes investment in implementing QCGSRamen-1Swap. It is an
exploratory pilot decision, not proof of reliable improvement or rejection of H0;
uncertainty crossing zero and the single-stream limitation remain visible.

Positive Stage A ranking correlation strengthens the mechanism description but
is not required for GO_PILOT. STOP takes precedence over REVISE when predicates
overlap. Numerical failures belong to INVALID, never scientific STOP.

## 9. Implementation and required checks

Modify the existing probe/evaluator/runner and their tests:

```text
src/methods/OracleSupportUtilityProbe.py
src/evaluation/oracle_support_utility.py
scripts/run-oracle-support-utility.py
tests/test_oracle_support_utility.py
```

Add pure selectors in `src/methods/query_gradient_selection.py`, focused tests
in `tests/test_query_gradient_selection.py`, and configuration under
`cfg/research/query-gradient-diagnostic/`. Preserve the old oracle contract;
version new sidecar schemas explicitly. The selector returns action, score,
and reason; its signature must exclude evaluator labels and provenance.

Before scored execution, require:

- Label/OOD/domain mutation tests: fixed model/cache produces identical
  label-free actions when evaluator metadata changes.
- No-swap parity: exact baseline aggregation path and logits within the locked
  existing parity tolerance; record that tolerance in preflight.
- Candidate parity: identical cache/top-k/extras, same-class single replacement,
  preserved cardinality/divisor, deterministic ties, and stable action identities.
- Aggregate arithmetic: for the first eligible query in every evaluator batch,
  compare vectorized aggregates with serial `aggregate_pools` for every swap.
- Manual synthetic score checks, including score sign, zero/tied scores,
  unchanged SignSGD direction, no legal swaps, and every numerical fallback.
- Reset/cache-trajectory tests, including exceptions and optimizer state.
- Registry completeness, historical/smoke exclusions, base-image disjointness,
  per-cell uniqueness, reproducible random draws, and artifact/schema checks.

Run focused tests first, then the complete test suite before CUDA evidence:

```bash
pytest -q tests/test_oracle_support_utility.py tests/test_query_gradient_selection.py
pytest -q
```

These are required checks. Consult the execution report and test receipts for
which checks have actually run; notebook generation is not experimental evidence.

## 10. Execution and artifact contract

The research branch is created from the pinned base. In this repository,
continue on that branch and verify the base remains in its history:

```bash
git switch qcgs-label-free-diagnostic
git merge-base --is-ancestor 3a80623b074f16b8ef87d8d5507427277ea2a54f HEAD
git rev-parse HEAD
git status --short
```

For a fresh checkout without this branch, create it directly from the pinned
SHA; do not pull a moving branch tip as the experiment base. Execution requires
a clean committed revision.

Record the initial base SHA, then the new implementation SHA after changes.
Commit the final launch plan, configuration, analysis specification, registry,
exclusions, and preflight report before scored execution. Record content hashes
for those inputs, dataset/split/stream manifests, model weights, environment,
and source revision. Use a clean committed revision and fresh evidence directory.

At the base pin, `scripts/run-oracle-support-utility.py` implements old oracle
A/B probes only and requires `--execute` for actual execution. Its output is
not this experiment. Implement and test the new mode before documenting an
actual launch command; this protocol does not invent runnable CLI flags.

Order: CPU tests/preflight -> tiny real-model smoke -> finalize smoke exclusions
and stage registry -> Stage A -> freeze arithmetic fixes and analysis bins ->
Stage B. A fix that affects scoring or evaluation invalidates affected Stage A
evidence; rerun it under the final implementation before Stage B. If Stage A
prompts a signal change, this protocol no longer confirms
the original score: preregister the revision and use untouched queries.
Never reuse an evidence directory after semantic changes or tune on Stage B.

Keep one raw query sidecar and one concise final pilot report, plus reproducible
run manifests/logs. Raw records must retain:

- Stable base/observation IDs, timestep/batch/block, stage/cell, source/config/
  registry/candidate-pool hashes, eligibility and canonical legal-action order.
- Evaluator-only known label, ID/OOD flag, and corruption/domain metadata,
  stored separately from the selector inputs and used only for verification.
- Baseline prediction/CE/entropy; query gradient norm; active-class/swap counts;
  self-support membership; selected action, score, reason, and RNG provenance.
- Incoming/outgoing stable support identities, pseudo-class/ranks, distances,
  cached entropies, query-gradient cosines, and aggregate-direction change.
- Exact pre/post prediction, CE/entropy and utility for each verified action;
  all legal swaps in Stage A, selected-action union in Stage B; stage-correct
  oracle/verified-set fields; scoring/evaluation cost and memory measurements.

The final report contains integrity/prerequisite status, Stage A ranking and
exact headroom, Stage B policy/paired metrics, transitions and objective mismatch,
uncertainty/sensitivity/strata, costs, and the first matching decision rule.
Report denominators, null reasons, deviations, and limitations. Link raw evidence
and hashes; no empty result-table templates are required.

## 11. Interpretation and next step

After GO_PILOT, the supported statement is that the frozen single-view selector
showed positive sample-average supervised utility and paired advantage over
random in both tested cells, passing the declared pilot checks. Implement the
minimal QCGSRamen-1Swap and seek confirmation on untouched base images and
additional seeds/streams. Only then consider OOD={0,0.1,0.3,0.5}, iid/block/
recurring streams, full top-M reranking, or adaptive support size.

For REVISE, test exactly one cost-matched multi-view entropy/consistency query
gradient. Freeze its compute budget and definition before new confirmation;
do not add several heuristics or tune the score threshold. For INCONCLUSIVE,
predeclare any additional sample/run budget on untouched data. For STOP, keep
the historical oracle conclusion separate from the failed single-view proxy.

Do not claim universal Ramen improvement, supervised-gradient recovery,
causal benefit from OOD, or generalization to untested datasets from this pilot.
Related-work/novelty claims require a separate review.

## Execution entry point

The new mode preserves the historical oracle CLI and uses a separate probe,
evaluator and campaign module. From a clean committed checkout with CUDA:

```bash
python scripts/run-oracle-support-utility.py --diagnostic qcgs --execute \
  --data-root /path/to/data --evidence-dir /path/to/fresh-qcgs-evidence
```

CPU verification without scientific execution:

```bash
python scripts/run-oracle-support-utility.py --diagnostic qcgs --preflight-only \
  --evidence-dir evidence/qcgs-cpu-preflight
```

The Kaggle notebook is `notebooks/kaggle/kaggle-qcgs-label-free.ipynb`. It embeds
a committed source bundle, uses the existing pinned Hugging Face reconstruction,
and exports an atomic `qcgs-label-free-evidence.zip`. Import it with Internet and
GPU enabled. Outputs are empty until it actually runs.

The committed operational spec is
`cfg/research/query-gradient-diagnostic/protocol.json`; it leaves the scientific
score and gates unchanged. Outcome-blind scans start at 600 stream rows and may
double to at most 6,000 only to fill untouched-image quotas. Final scored runs
use the shortest whole-batch prefix covering the registry, locked before A.
Timeout defaults to 3,600 seconds per cell; any override must be supplied before
the initial freeze. A timeout stops later stages and retains partial evidence.

Actual smoke IDs, parameter ordering, registry, scan/provenance, configs and
launch commands are committed to a **separate local Git ledger inside
`evidence/locks`** before Stage A; Stage A bin edges are committed there before
Stage B. This satisfies the pre-result commit requirement while keeping the
source checkout clean. Ledger history is included in the ZIP; no remote push
is required. Checkpoints restart an incomplete cell from its initial model,
cache and dedicated RNG. Completed cells require matching hashes and validation.

After restoring a ZIP at the same execution paths, add `--resume` to execute.
For a downloaded/extracted artifact, `--audit --evidence-dir /path/to/evidence`
rechecks the input ledger, completed-file hashes and raw arithmetic without a
GPU or access to the original Kaggle filesystem. This audit does not independently
recompute model logits. See
[execution status](../../plans/20260919-qcgs-diagnostic/reports/results.md).

## Source basis

Repository-relative references at the pinned base unless described as historical:

- `plans/260914-1635-qcgs-label-free-research-roadmap/phase-01-start.md`
- `plans/260914-1635-qcgs-label-free-research-roadmap/phase-02-label-free-utility-diagnostic.md`
- `plans/20260907-oracle-support-utility/reports/pilot-b-completed.md`
- `evidence/oracle-support-pilot-b-resumed-20260907/` (historical raw query artifacts)
- `docs/research/oracle-support-utility-probe.md`
- `src/methods/Ramen.py` and `src/methods/OracleSupportUtilityProbe.py`
- `src/evaluation/oracle_support_utility.py`
- `scripts/run-oracle-support-utility.py`
- `cfg/research/oracle-support-utility-screened/CIFAR100C/OracleSupportUtilityProbe.yaml`
- `src/datasets/corruption/utils.py` (stable source index after severity slicing)
