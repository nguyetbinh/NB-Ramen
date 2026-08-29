# Failure-Mode Analysis Framework for Ramen-Based Test-Time Adaptation

## Status

This document is the scientific contract for the `failure-analysis` branch.
It replaces the earlier draft with a stricter separation between:

1. completed ablations;
2. research strategy;
3. diagnostic experiments;
4. failure-mode error analysis;
5. criteria for introducing a new method.

The intended order is:

```text
ablation study
    -> strategy
    -> diagnostic experiment
    -> error analysis
    -> only then a new method
```

The methodological inspiration is Galstyan et al., **Failure Modes of Domain Generalization Algorithms** (CVPR 2022). That paper studies static domain generalization, not online test-time adaptation. Its exact error decomposition is therefore not copied literally. What is adopted is the methodology: decompose final error into mechanistically distinct failures, use controlled interventions to isolate them, and do not confuse improvement in a proxy metric with improvement in the task.

The current branch has implemented the diagnostic runtime and a four-sample CPU/MPS mechanics pilot. That pilot validates the evidence pipeline only. It does **not** establish any scientific conclusion about gradient conflict or ConsensusRamen.

---

# 1. Executive summary

The completed NB-Ramen experiments give the following local conclusions.

| Direction | Tested hypothesis | Current conclusion |
| --- | --- | --- |
| Latent context routing | same latent/domain context gives better support | unsupported in current form |
| Oracle domain routing | true domain identity is a useful adaptation partition | not supported by bounded pilots |
| Entropy-gated admission | cleaner pseudo-label memory gives better adaptation | rejected locally as a fixed-gate hypothesis |
| Strict sample causality | removing future-within-batch support improves Ramen | no positive scheduling-only signal in the controlled pilot |
| Retrieval compression | retrieval is the dominant runtime bottleneck | not supported by the bounded profile |
| Gradient compatibility | harmful updates arise from incompatible retrieved gradients | unresolved; this is now the active diagnostic question |

The common lesson is that the previous extensions mainly tried to decide whether a **sample** is trustworthy:

```text
same domain?
low entropy?
strictly historical?
```

Ramen, however, ultimately adapts with cached gradients. The current strategy is therefore:

$$
\boxed{
\text{study gradient usefulness and compatibility before designing another sample-level heuristic}
}
$$

This is a strategy, not yet a result.

---

# Part I — What has already been ablated?

## 2. Baseline mechanism in Ramen

For a query sample, Ramen follows the chain:

```text
query image
  -> CLIP feature and initial prediction
  -> entropy and sample-wise gradient
  -> persistent class-indexed memory
  -> support retrieval
  -> entropy + feature-distance weighting
  -> class-balanced gradient aggregation
  -> temporary SignSGD update
  -> adapted prediction
  -> model-parameter reset
```

The original Ramen study already motivates feature similarity and prediction-balanced support. NB-Ramen has explored additional hypotheses about what makes memory evidence useful.

---

## 3. Ablation A — latent context/domain routing

### Hypothesis

Support should be more useful when it belongs to the same latent deployment context as the query.

### Compared methods

```text
CausalRamen
LatentRamen
OracleLatentRamen
```

- `CausalRamen`: one fixed context;
- `LatentRamen`: online prototype routing from CLIP image features;
- `OracleLatentRamen`: evaluator domain identity used only as a diagnostic oracle.

### Observed evidence

The bounded CIFAR-100-C block pilots show:

- the unsupervised router collapses to one context in the stronger reported cells;
- routing NMI is zero in those cells;
- when the router stays at one context, LatentRamen becomes behaviorally equivalent to the fixed-context causal control;
- oracle-domain routing does not improve the bounded result and can be worse.

### Current conclusion

The defensible statement is:

```text
annotated domain identity != adaptation compatibility
```

The latent-domain path should remain frozen unless a new diagnostic shows that useful domain/context information is present in the representation and the current router alone fails to exploit it.

---

## 4. Ablation B — entropy-gated memory admission

### Hypothesis

Low predictive entropy identifies reliable memory items, so a low-entropy admission rule should improve adaptation.

### Compared methods

```text
LatentRamen
EntropyGatedLatentRamen
```

### Observed evidence

The fixed entropy gate does improve its **proxy objective**:

- admitted pseudo-label accuracy increases;
- admitted contamination decreases;
- retained memory decreases substantially.

But the **task objective** moves in the wrong direction:

- accuracy drops relative to ungated LatentRamen;
- negative-adaptation behavior does not improve;
- the negative paired direction repeats across the bounded three-seed replication.

### Current conclusion

$$
\boxed{
\text{pseudo-label purity} \not\Rightarrow \text{adaptation usefulness}
}
$$

The fixed entropy gate should remain a negative ablation. It must not be retuned on final target streams.

This result is especially important for the failure-analysis design below: **pseudo-label correctness cannot be treated as a ground-truth oracle for gradient usefulness.**

---

## 5. Ablation C — strict sample causality

### Initial hypothesis

Legacy Ramen inserts a whole evaluator batch before retrieval. A strict sample-stream protocol may behave differently because a query cannot use later items from the same evaluator batch.

### Initial comparison

```text
legacy Ramen
vs
CausalRamen
```

Historical bounded pilots favored CausalRamen on some cells, but the comparison was confounded because the implementations differed in memory structure, ranking precision, sorting, and numerical behavior.

### Scheduling-only control

`StructuredAtomicRamen` was introduced so that:

```text
StructuredAtomicRamen
vs
CausalRamen
```

share the structured memory, selection rule, ranking, aggregation, configuration, and optimizer. The intended difference is only scheduling:

```text
StructuredAtomicRamen:
    admit whole evaluator batch
    -> query whole batch

CausalRamen:
    admit x_t
    -> query x_t
    -> advance to x_{t+1}
```

The bounded batch-size sensitivity pilot showed no positive scheduling-only gain.

### Current conclusion

Strict sample causality remains the cleaner **online protocol baseline**, but the current evidence does not support causality itself as an accuracy-improving contribution.

### Important evidence correction

The `failure-analysis` branch fixed a reset-anchor aliasing problem in the by-sample normalization reparameterization that is especially relevant when `max_batch_size == 1`. Historical `B=1` results produced before this fix should therefore be treated as **stale diagnostics** and rerun before they are used in any causal argument.

This does not automatically invalidate larger-batch results, but it removes the old `B=1` evidence from the publication-quality evidence set until it is reproduced with the fixed implementation.

---

## 6. Ablation D — retrieval/compression efficiency

The bounded retrieval profile did not show retrieval dominating the profiled forward path. Compression and approximate retrieval are therefore deferred systems directions rather than the active scientific question.

---

# Part II — Research strategy after the ablations

## 7. Methodological lesson from Galstyan et al.

Galstyan et al. show that a mechanism can improve according to its own proxy while the final task remains poor. In their setting, stronger domain invariance is neither necessary nor sufficient for strong domain generalization.

The analogous lesson for NB-Ramen is:

```text
better domain/context purity     does not imply better adaptation
better pseudo-label/cache purity does not imply better adaptation
```

The response should not be to tune the proxy harder. The response should be to identify which stage of the adaptation pipeline actually produces harmful predictions.

---

## 8. Active research question

The current research question is:

> **When Ramen changes a prediction, which part of the memory/retrieval/update chain made that change beneficial or harmful?**

The active hypothesis is narrower:

> **Do harmful adaptation events contain measurably more gradient-direction conflict than beneficial adaptation events?**

This hypothesis must be diagnosed before any deployable consensus mechanism is implemented.

---

# Part III — Exact outcome decomposition first

## 9. Paired NoAdapt-versus-adapted outcomes

Every analysis must begin with exactly paired samples from `NoAdapt` and the adapted method.

For each query:

| NoAdapt | Adapted | Category |
| --- | --- | --- |
| correct | correct | safe |
| wrong | correct | beneficial |
| correct | wrong | harmful |
| wrong | wrong | unresolved |

Define:

$$
H=P(NoAdapt\ wrong,\ Adapted\ correct)
$$

and:

$$
A=P(NoAdapt\ correct,\ Adapted\ wrong).
$$

On an exactly paired sample set:

$$
\boxed{
Acc_{adapted}-Acc_{NoAdapt}=H-A
}
$$

Also report:

$$
HelpRate=P(Adapted\ correct\mid NoAdapt\ wrong)
$$

and:

$$
HarmRate=P(Adapted\ wrong\mid NoAdapt\ correct).
$$

This decomposition is exact. It should always appear before any proxy, correlation, or oracle analysis.

The key comparison for mechanism analysis is:

```text
beneficial queries
vs
harmful queries
```

---

# Part IV — Corrected Ramen failure taxonomy

The source DG paper uses nested errors for a static representation/classifier pipeline. Ramen is online, query-dependent, memory-dependent, and temporarily updates model parameters. The DG decomposition should therefore not be copied literally.

Instead use controlled TTA-specific failure modes.

---

## 10. F0 — base-model limitation

### Question

Was the query already wrong before adaptation?

### Measurement

The paired NoAdapt trace defines the base error:

$$
E_{base}=1-Acc_{NoAdapt}.
$$

F0 is not automatically a Ramen failure. It is the set of errors that adaptation may repair.

---

## 11. F1 — memory-availability failure

### Scientific question

At query time, does the **legal online memory** contain any evidence that can actually improve this query?

### Critical semantic rule

A correct pseudo-label is **not** sufficient evidence of usefulness. The entropy-gating ablation already shows that pseudo-label purity and adaptation usefulness are different quantities.

Therefore the current correctly-pseudolabeled candidate statistic must be named as a proxy, for example:

```text
correct_pseudolabel_legal_availability
```

It must **not** be presented as a true memory oracle or as `memory_insufficiency_rate` without qualification.

### True F1 intervention

A true F1 diagnostic must ask whether some legal candidate or preregistered legal subset can change the query outcome beneficially under an evaluator-only counterfactual.

Conceptually:

$$
\exists g_j \in M_t^{legal}
\quad\text{s.t.}\quad
Update(g_j)\text{ improves query }q?
$$

The exact intervention family must be preregistered and bounded. Ground-truth labels may only define evaluator-only oracle subsets; they may not influence the production prediction.

### Interpretation

```text
no legal beneficial intervention exists
    -> evidence consistent with memory insufficiency

legal beneficial intervention exists
    -> useful evidence is present; inspect F2/F3
```

---

## 12. F2 — retrieval failure

### Scientific question

Useful evidence exists in legal memory, but does the production retrieval rule fail to expose it to the update?

### Required production diagnostics

For each query, record at least:

```text
support item IDs
support predicted classes
support distances
support entropies
support weights
support recencies
valid masks / support counts
```

Evaluator-only joins may add:

```text
support true class
support true domain/corruption
pseudo-label correctness
ID/OOD state
```

### Correct intervention logic

Compare the same preregistered beneficial-intervention family over:

1. all legal candidates;
2. only the production retrieved support set.

A gap between these two intervention families is evidence for retrieval failure.

A gap between correctly-pseudolabeled availability rates is useful descriptively, but it is only a **pseudo-label retrieval proxy**, not a causal F2 attribution.

---

## 13. F3 — gradient compatibility / aggregation failure

F3 is the central unresolved mechanism.

### F3a — conflict diagnostics

For each active retrieved pseudo-class `c`, compute the weighted class-local gradient:

$$
h_{q,c}
=
\frac{\sum_{j\in S_{q,c}}\alpha_{qj}g_j}
{\sum_{j\in S_{q,c}}\alpha_{qj}+\epsilon}.
$$

Because Ramen ultimately uses SignSGD, define coordinate-wise vote:

$$
v_{q,k}=\frac{1}{C_q}\sum_c\operatorname{sign}(h_{q,c,k})
$$

and consensus strength:

$$
q_{q,k}=|v_{q,k}|.
$$

Useful per-query diagnostics include:

```text
consensus_mean
consensus_p10
consensus_p50
fraction_low_consensus_coordinates
active_support_classes
pairwise class-gradient cosine similarity
pairwise class-gradient sign agreement
```

These quantities are diagnostics only.

### F3b — compatibility counterfactuals

A consensus mask such as:

$$
g_t=g_{actual}\odot\mathbf{1}[q\ge t]
$$

changes the aggregate gradient itself. It therefore belongs to **F3**, not F4.

The current replay thresholds `(0.50, 0.75, 1.00)` should remain labeled as preregistered evaluator-only F3 counterfactuals. They may be used to ask:

- how many harmful events are recovered;
- how many new harmful events are introduced;
- whether accuracy changes relative to the actual adapted prediction.

A positive F3 result requires more than a correlation: a compatible-gradient intervention should recover part of the harmful-update gap without creating comparable new harm.

---

## 14. F4 — optimizer/update sensitivity

F4 should keep the **support set and aggregate gradient fixed** and test only the update rule.

Examples of valid F4 diagnostics include preregistered alternatives such as:

```text
same aggregate, no update
same aggregate, smaller SignSGD step
same aggregate, larger SignSGD step
```

or another explicitly controlled optimizer intervention.

The key invariant is:

```text
same retrieved supports
same aggregate direction
only optimizer/update semantics change
```

The current consensus-mask counterfactual does not satisfy this invariant and should not be described as F4 evidence.

F4 is secondary unless F3 diagnostics indicate that the aggregate itself is not the dominant failure.

---

## 15. F5 — temporal/scheduling failure

### Question

Does the result depend materially on batch-atomic versus strict sample-causal memory scheduling?

### Primary control

```text
StructuredAtomicRamen
vs
CausalRamen
```

Potential diagnostics include:

```text
batch position
future-support count
future-support weight fraction
time to / since a domain transition
```

The current bounded scheduling-only pilot does not show a positive causal gain. F5 remains a protocol dimension, not the active thesis mechanism.

Historical pre-fix `B=1` cells must be rerun before being used as evidence.

---

# Part V — Proxy metrics versus causal/intervention metrics

## 16. Every mechanism needs both kinds of evidence

| Hypothesis | Proxy / mechanism metric | Outcome or intervention metric |
| --- | --- | --- |
| Latent routing | context NMI, domain-probe accuracy | harmful rate, retrieval/usefulness intervention |
| Entropy admission | entropy, pseudo-label purity | downstream influence, removal/usefulness counterfactual |
| Scheduling | future-support fraction | atomic-vs-causal paired flips |
| Gradient compatibility | sign/cosine disagreement | harmful-vs-beneficial contrast, F3 recovery |
| Open-set contamination | retrieved OOD fraction, GDC, SDR | ID harmful adaptation, ID-only intervention |

A proxy is useful only if it explains or predicts a task-level failure component.

---

# Part VI — Retrospective error analysis for completed directions

## 17. LatentRamen: representation versus router failure

The retrospective question is:

```text
does the CLIP representation contain domain information
that the online router fails to exploit?
```

### Required probes

Freeze CLIP image features and train evaluator-only deterministic probes:

```text
feature -> corruption/domain
feature -> semantic class
within semantic class: feature -> corruption/domain
```

### Critical split rule for CIFAR-C

CIFAR-C corruptions reuse the same underlying sample index across corruption domains. A probe split must therefore be **grouped by underlying sample identity**.

For CIFAR-100-C, all corruption views of the same `sample_idx` must belong to the same train/validation/test split.

Do **not** hash `(sample_idx, domain)` independently, because that can place different corrupted views of the same underlying image into both probe train and probe test sets.

Until grouped splitting is implemented and validated, CIFAR-C domain-probe accuracy is not publication-quality evidence.

### Interpretation

```text
high domain decodability + low router NMI
    -> router/formulation failure

low domain decodability + low router NMI
    -> raw CLIP features weak for the routing objective

high domain decodability + no OracleLatent gain
    -> domain identity is decodable but not a useful adaptation partition
```

This analysis is retrospective. It does not reopen router tuning by default.

---

## 18. Entropy gate: confidence versus gradient usefulness

The entropy result should be analyzed through downstream influence rather than cache purity alone.

Group stored/rejected items by:

```text
low entropy + correct pseudo-label
low entropy + wrong pseudo-label
high entropy + correct pseudo-label
high entropy + wrong pseudo-label
```

For each group report:

```text
storage/admission rate
retrieval frequency
total downstream retrieval weight
mean retrieved distance
```

Then add a true usefulness diagnostic.

Preferred options, in order of scientific strength:

1. **leave-one-out support counterfactual**: compare the query update with and without item `j` while holding the rest of the production support fixed;
2. **leave-one-out aggregate agreement**: compare `g_j` with the support aggregate excluding `j`;
3. a clearly labeled weaker proxy such as cosine/sign agreement with a non-self-containing reference gradient.

Do not compare an item only with an aggregate that contains that same item and present the result as independent evidence; that is circular.

The goal is to test the hypothesis:

> the entropy gate discards some high-entropy items that nevertheless have beneficial downstream gradient influence.

---

## 19. Causal scheduling: close attribution, do not rescue the hypothesis

The purpose of CausalRamen now is protocol control.

Required clean-up:

- rerun `B=1` after the reset-anchor alias fix;
- retain `StructuredAtomicRamen vs CausalRamen` as the single-factor scheduling control;
- use F5 only to explain scheduling differences if they recur.

Do not spend large CUDA/DomainNet budget solely to rescue the causal-gain hypothesis unless a new bounded result independently justifies reopening it.

---

# Part VII — Open-set gradient contamination

## 20. Open-set question

The open-set thesis extension remains valid as a later diagnostic, not as a conclusion already established.

Use a fixed known/unknown split where the model-facing vocabulary contains only known classes and unknown status is evaluator-only.

For query `q`, compare:

$$
g_q^{all}=\sum_{j\in S_q}\alpha_{qj}g_j
$$

with evaluator-only ID aggregate:

$$
g_q^{ID}=\sum_{j\in S_q,\;j\in ID}\alpha_{qj}g_j.
$$

Measure:

$$
GDC_q=1-\cos(g_q^{all},g_q^{ID})
$$

and:

$$
SDR_q
=\frac1D\sum_k
\mathbf1[\operatorname{sign}(g_{q,k}^{all})\neq\operatorname{sign}(g_{q,k}^{ID})].
$$

Then relate GDC/SDR to **ID harmful adaptation events** and to an ID-only evaluator counterfactual.

If OOD ratio rises but gradient corruption, harmful adaptation, and ID-only recovery do not rise, semantic gradient contamination should not be promoted as the thesis mechanism.

---

# Part VIII — Temporal analysis

## 21. Why time matters in TTA

The dominant failure can change with stream state. Required analysis axes include:

```text
timestep
time since domain shift
stream type
memory occupancy
batch size
OOD ratio, when applicable
seed
```

A plausible pattern is:

```text
early episode:
    little useful memory is available

mid episode:
    retrieval and aggregation become more informative

after shift:
    stale or conflicting evidence may become important
```

This is a hypothesis to measure, not a narrative to assume.

### Recommended paired panels

Top panel — outcome/failure quantities:

```text
base error
beneficial rate
harmful rate
true F1 intervention gap
true F2 intervention gap
F3 counterfactual recovery
```

Bottom panel — mechanism quantities:

```text
domain decodability / routing NMI
entropy / pseudo-label purity
future-support fraction
gradient consensus / sign disagreement
retrieved OOD fraction
```

The scientific question is whether mechanism changes track the failure component they are supposed to explain.

---

# Part IX — Required implementation corrections before a larger scientific run

## 22. P0 — rename the current pseudo-label “oracle”

The current offline statistic based on “at least one correctly pseudo-labeled candidate exists” is useful, but it is not a true usefulness oracle.

Required naming change:

```text
legal_oracle_rate
    -> correct_pseudolabel_legal_availability

retrieved_oracle_rate
    -> correct_pseudolabel_retrieved_availability

memory_insufficiency_rate
    -> 1 - correct_pseudolabel_legal_availability
       only if explicitly labeled as a pseudo-label proxy

retrieval_gap
    -> correct_pseudolabel_retrieval_gap
```

A separate intervention-based F1/F2 metric must be introduced before claims about memory insufficiency or retrieval failure are made.

---

## 23. P0 — move consensus-mask recovery from F4 to F3

The current fixed-threshold masking experiment modifies the aggregate gradient. It is therefore an aggregation/compatibility intervention.

Required report taxonomy:

```text
F3a conflict diagnostic
F3b consensus-mask counterfactual recovery
F4  same-aggregate optimizer/update sensitivity
```

This is a semantic correction; the existing replay machinery remains useful.

---

## 24. P0 — rerun all decision-relevant B=1 evidence

The by-sample normalization reset-anchor fix changes the trustworthiness of historical `B=1` evidence.

Required action:

```text
rerun B=1 mechanics and causal-control cells with the fixed implementation
mark older B=1 artifacts as historical / non-decision-bearing
```

---

## 25. P1 — grouped representation-probe splits

Implement dataset-aware group splitting so that multiple views of the same underlying sample never cross probe train/test boundaries.

At minimum:

```text
CIFAR-C group key = underlying sample_idx
```

The split policy and group key must be written into probe metadata.

---

## 26. P1 — finish the entropy-to-usefulness bridge

The current report can measure retrieval frequency and downstream weight, but item-level gradient usefulness still needs a non-circular reference or a removal counterfactual.

Required before a strong entropy failure claim:

```text
with-item vs without-item effect
or
item gradient vs leave-one-out aggregate
```

---

## 27. P1 — aggregate decisions across runs

The per-run CLI currently cannot, by itself, establish the cross-seed/cross-stream ConsensusRamen gate.

Add a separate cross-run aggregator that consumes completed per-run reports and checks a fixed coverage matrix.

Minimum decision-bearing coverage should require:

```text
>= 3 fixed seeds
>= 2 structured streams
same analysis protocol and diagnostic version
both beneficial and harmful events represented
complete F3 counterfactual evidence
```

The aggregator should validate the intended Cartesian coverage rather than merely count unique seeds and streams.

No per-run analyzer should claim `GO` for ConsensusRamen.

---

# Part X — Diagnostic experiment plan

## 28. Stage A — implementation validation

Before scientific interpretation:

1. rerun the full unit suite;
2. verify profile-off versus profile-on prediction parity;
3. rerun fixed `B=1` mechanics;
4. verify grouped probe splitting;
5. verify renamed proxy metrics;
6. verify F3/F4 taxonomy in JSON and docs;
7. verify cross-run aggregation on synthetic fixtures.

The four-sample CPU/MPS pilot remains mechanics evidence only.

---

## 29. Stage B — bounded closed-set scientific analysis

Primary methods:

```text
NoAdapt
CausalRamen replay_v1
```

Primary structured streams:

```text
block
recurring
```

Primary seeds:

```text
0, 1, 2
```

Use a **fixed preregistered sample budget** chosen before execution. Do not extend a run merely because it failed to produce harmful events; that would make the analysis outcome-dependent.

Report per cell:

```text
safe / beneficial / harmful / unresolved
HelpRate / HarmRate
conflict metrics
F3 counterfactual recovery
new harm introduced by F3 counterfactuals
pseudo-label availability proxies
memory/retrieval support statistics
```

The scientific objective is not “beat Ramen”. It is:

> determine whether harmful events are consistently associated with gradient conflict and whether a compatibility intervention recovers them.

---

## 30. Stage C — retrospective direction-specific analyses

Run separately rather than creating one oversized matrix.

### LatentRamen retrospective

```text
grouped domain probe
class probe
class-conditioned domain probe
routing NMI
OracleLatent diagnostic
```

### Entropy-gate retrospective

```text
entropy/correctness groups
downstream support influence
leave-one-out or removal usefulness diagnostic
```

### Scheduling retrospective

```text
fixed B=1 rerun
StructuredAtomic vs Causal paired F5 analysis
```

These analyses explain prior directions; they are not method-selection sweeps.

---

## 31. Stage D — open-set analysis

Only after the closed-set failure machinery is stable:

```text
fixed known/unknown split
fixed OOD ratios
GDC / SDR
ID harmful rate
ID-only evaluator counterfactual
```

Do not use target labels or OOD identity in a deployable method.

---

# Part XI — Go / no-go criteria for ConsensusRamen

## 32. GO conditions

ConsensusRamen should be implemented only if all of the following are supported on the preregistered analysis matrix:

1. **Association** — harmful queries have a consistently worse canonical conflict statistic than beneficial queries.
2. **Replication** — the direction repeats across at least three fixed seeds and at least two structured streams.
3. **Intervention** — an evaluator-only F3 compatibility intervention recovers a non-trivial subset of harmful events.
4. **Safety** — the intervention does not introduce a comparable or larger number of new harmful events.
5. **Coverage** — the result is not driven by one domain episode, one batch position, or one seed.

Only after these conditions hold should the smallest deployable consensus mechanism be implemented.

---

## 33. NO-GO / insufficient conditions

Do **not** implement ConsensusRamen if any of the following remains true:

```text
harmful and beneficial conflict distributions strongly overlap
conflict direction changes across seeds/streams
F3 counterfactuals fail to recover harmful events
recovery is offset by comparable new harm
apparent signal depends on evaluator-only labels rather than model-visible diagnostics
```

If evidence is missing rather than negative, report `INSUFFICIENT`, not `NO_GO`.

---

# Part XII — Current runtime status

## 34. Implemented

The branch currently provides:

```text
trace_v1 / replay_v1 profiles
support provenance
bounded gradient/feature sidecars
exact paired H/A decomposition
class-local gradient conflict summaries
preregistered consensus-mask counterfactuals
strict provenance and checksum validation
temporal summaries
representation-probe utilities
semantic open-set evaluator protocol and ID-gradient oracle
verified cross-cell study aggregation
CPU/MPS bounded study and schedule-only F5 comparison
```

The completed primary matrix uses two seeds over block and recurring streams,
with 64 samples in each baseline/adapted cell. All eight cells share one source,
model, dataset, method configuration, and evaluator contract. The separate
semantic open-set matrix also completed all eight requested MPS cells under one
source identity.

`ConsensusRamen` remains unimplemented, which is the correct current state.

---

## 35. Not yet established

The bounded study does **not** yet establish:

```text
a true F1 memory-usefulness gap
a true F2 retrieval-usefulness gap
a replicated harmful-vs-beneficial conflict direction
a safe F3 intervention whose recovery exceeds new harm
a class-conditioned domain-probe result with adequate coverage
that open-set gradient contamination causes harmful ID adaptation
a publication-level CUDA result
```

Entropy/gradient compatibility, grouped domain decodability, and open-set
contamination were measured, but none supplied the stable, adequately covered
causal evidence required for method development. These remain evidence targets
rather than missing runtime capabilities.

---

# Part XIII — Research narrative

The current project should be narrated as a sequence of increasingly specific diagnoses rather than a sequence of failed methods:

```text
Latent routing
    -> annotated domain identity is not sufficient

Entropy gating
    -> predictive confidence and pseudo-label purity are not sufficient

Causal scheduling
    -> strict causality is a protocol property, not an established accuracy gain

Retrieval profiling
    -> retrieval compression is not currently the dominant systems bottleneck

Failure analysis
    -> now ask whether the production gradient update itself is incompatible
       with the query when adaptation becomes harmful
```

The thesis hypothesis remains:

$$
\boxed{
\text{Reliable test-time adaptation may depend more on gradient compatibility than on sample-level trust proxies.}
}
$$

This is still a **research hypothesis**. The purpose of the `failure-analysis` branch is to decide whether it deserves to become a method.

## 36. Bounded study result (2026-08-30)

The diagnostic runtime, verified offline analyzers, and bounded CPU/MPS study
are implemented. The study includes two seeds over block and recurring
streams, entropy/gradient compatibility, fixed-threshold reset replay, an
atomic/causal batch-size-four comparison, frozen-feature domain probes, and a
fixed 80/20 semantic open-set protocol. CUDA was explicitly unavailable on the
host and no fallback result was recorded.

The implementation contract, evaluator-only boundary, exact commands, verified
report locations, and limitations are documented in
[Ramen Failure-Mode Analysis Runtime and Study Record](ramen-failure-mode-analysis-runtime.md).
The complete tables and decision are in the
[Full Ramen Failure-Mode Study Result](../../plans/20260829-full-failure-mode-study/reports/full-study-results.md).

The aggregate decision is `INSUFFICIENT`. Harmful updates had greater conflict
in both block seeds but lower conflict in recurring seed 1; recurring seed 0
had no harmful event. The evaluator-only replay oracle recovered harmful cases
but introduced enough new harm that no threshold improved accuracy. Semantic
OOD contamination increased GDC and SDR, but harmful ID events were too sparse
to establish causality. The GO conditions in Section 32 are therefore not met and
`ConsensusRamen` remains unimplemented.

---

# Reference

Tigran Galstyan, Hrayr Harutyunyan, Hrant Khachatrian, Greg Ver Steeg, and Aram Galstyan. **Failure Modes of Domain Generalization Algorithms.** CVPR 2022.

The source paper contributes the general methodology of failure-mode decomposition, proxy-versus-task analysis, and intervention-based diagnosis. The Ramen-specific taxonomy, diagnostics, oracle definitions, temporal protocol, and experimental sequence in this document are adaptations for NB-Ramen and should not be attributed to the source paper.
