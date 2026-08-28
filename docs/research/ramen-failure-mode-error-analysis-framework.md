# Failure-Mode Error Analysis Framework for Ramen-Based Test-Time Adaptation

## Purpose

This report defines how error analysis should be conducted for the current Ramen research program. It is intentionally organized in the following order:

```text
Ablation study
    -> research strategy
    -> diagnostic experiments
    -> failure-mode error analysis
    -> only then a new method
```

The methodological inspiration is **Galstyan et al., “Failure Modes of Domain Generalization Algorithms,” CVPR 2022**. That paper does not study test-time adaptation or Ramen. The framework below therefore separates:

1. **source-derived principles** from Galstyan et al.; and
2. **our adaptation of those principles** to memory-based, online TTA.

The central lesson taken from the source paper is that reporting final accuracy alone is not enough. A useful analysis should identify qualitatively different failure modes, construct diagnostics or interventions that isolate them, and measure how much each failure contributes to the final error. The source paper also demonstrates that a mechanism metric can improve without improving the task metric: stronger domain invariance can still coincide with worse generalization. This point is directly relevant to the current Ramen experiments.

---

# Part I — Ablation study: what has already been tested?

## 1. Baseline mechanism in Ramen

Ramen performs sample-specific test-time adaptation using a persistent embedding-gradient memory. For a query sample, the main chain is:

```text
query image
  -> CLIP feature / zero-shot prediction
  -> sample entropy and sample-wise gradient
  -> persistent class-indexed memory
  -> support retrieval
  -> entropy + feature-distance weighting
  -> gradient aggregation
  -> temporary SignSGD update
  -> adapted prediction
  -> parameter reset
```

The original Ramen paper already ablates two support-selection principles:

- domain consistency through feature similarity;
- prediction balance through per-predicted-class retrieval.

The NB-Ramen project has tested additional hypotheses about what makes a memory item or support set useful.

---

## 2. Ablation A — latent context/domain routing

### Hypothesis

A support sample should be more useful when it belongs to the same latent deployment context as the query.

### Compared methods

```text
CausalRamen
LatentRamen
OracleLatentRamen
```

- `CausalRamen`: fixed context, no router;
- `LatentRamen`: online prototype routing from CLIP visual features;
- `OracleLatentRamen`: evaluator domain identity used only as an oracle diagnostic.

### Current evidence

The bounded CIFAR-100-C block pilots show:

- the unsupervised router collapses to one context;
- routing NMI is zero in the reported cells;
- LatentRamen becomes behaviorally identical to CausalRamen when the router remains at one context;
- Oracle domain routing does not provide a positive accuracy upper bound and can underperform the non-oracle control.

Relevant reports:

```text
plans/20260824-latent-ramen-evidence/reports/cifar100c-mps-block-n200-pilot.md
plans/20260824-latent-ramen-evidence/reports/causal-ramen-mps-paired-pilot.md
```

### Ablation conclusion

The evidence does **not** support the claim that recovering annotated domain identity is sufficient for better adaptation.

The correct local conclusion is:

```text
annotated domain identity != adaptation compatibility
```

The latent-domain path should therefore remain frozen unless a new diagnostic establishes that the representation contains a useful domain signal that the current router alone fails to exploit.

---

## 3. Ablation B — entropy-gated memory admission

### Hypothesis

Low predictive entropy identifies reliable memory items, so admitting only low-entropy samples should improve adaptation.

### Compared methods

```text
LatentRamen
EntropyGatedLatentRamen
```

### Current evidence

The entropy gate succeeds at its mechanism-level objective:

- admitted pseudo-label accuracy increases;
- admitted contamination decreases;
- retained memory decreases substantially.

However, the task-level outcome moves in the wrong direction:

- accuracy decreases relative to ungated LatentRamen;
- negative-adaptation behavior does not improve;
- the negative accuracy direction repeats across the bounded three-seed replication.

Relevant report:

```text
plans/20260824-latent-ramen-evidence/reports/entropy-gated-latent-ramen-mps-smoke.md
```

### Ablation conclusion

The current evidence supports:

$$
\boxed{
\text{pseudo-label purity} \not\Rightarrow \text{adaptation usefulness}
}
$$

Prediction confidence is therefore not a sufficient proxy for gradient quality. The fixed entropy gate should remain a negative ablation and should not be retuned on final target streams.

---

## 4. Ablation C — strict sample causality

### Initial hypothesis

Legacy Ramen is batch-atomic: for a batch with more than one sample, every batch item is admitted before any query is retrieved. A strict-online implementation may therefore behave differently because it prevents later items in the evaluator batch from influencing earlier queries.

### Initial comparison

```text
legacy Ramen
vs
CausalRamen
```

Historical bounded pilots showed apparent positive gains for CausalRamen. This comparison was confounded because legacy Ramen and CausalRamen used different memory implementations and numerical/ranking behavior.

### Scheduling-only control

The branch `causal-ramen-completion` adds:

```text
StructuredAtomicRamen
```

`StructuredAtomicRamen` and `CausalRamen` share the same structured memory, selection rule, ranking, aggregation, configuration, and optimizer. Their intended experimental difference is scheduling:

```text
StructuredAtomicRamen:
    admit whole evaluator batch
    -> query whole batch

CausalRamen:
    admit x_t
    -> query x_t
    -> advance to x_{t+1}
```

The bounded MPS batch-size sensitivity pilot produced no positive scheduling-only signal. Across batch sizes `1, 2, 5, 10, 20, 50, 100`, the mean `CausalRamen - StructuredAtomicRamen` micro-accuracy delta was negative, and no negative-adaptation advantage was observed.

Relevant material:

```text
docs/research/causal-ramen-completion-report.md
plans/20260827-causal-ramen-completion/reports/local-runtime-and-causal-pilot.md
plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity-extracted.json
```

### Ablation conclusion

Strict causality is still the cleaner **protocol baseline** for sample-by-sample online TTA, but the current evidence does not support promoting causality itself as an accuracy-improving contribution.

The earlier apparent CausalRamen gain cannot be attributed to scheduling alone.

---

## 5. Ablation D — retrieval/compression efficiency

### Hypothesis

Gradient retrieval may be the dominant runtime bottleneck, motivating compression or approximate retrieval.

### Current evidence

The causal retrieval profile measured retrieval at only a minority of profiled forward time in the bounded pilot, below the preregistered compression gate.

### Ablation conclusion

Compression is currently a **deferred systems direction**, not the highest-value scientific question.

---

# Part II — Research strategy after the ablations

## 6. What the failed or weak hypotheses have in common

The tested extensions mostly ask whether a **sample** should be trusted:

```text
same domain?
low entropy?
strictly historical?
```

But Ramen ultimately adapts with cached **gradients**, not with domain labels or confidence values.

The combined evidence therefore motivates a strategy change:

$$
\boxed{
\text{Stop treating sample trustworthiness as the primary object.}
}
$$

Instead ask:

$$
\boxed{
\text{Which retrieved gradients produce an adaptation-compatible update for the current query?}
}
$$

This is a research strategy, not yet a claim that gradient consensus is correct.

---

## 7. Source-derived lesson from Galstyan et al.

Galstyan et al. define several generalization failure modes and use auxiliary classifiers/interventions to isolate them. Two methodological ideas are especially important for NB-Ramen:

1. **Decompose final error into mechanistically distinct components rather than only comparing final accuracies.**
2. **Do not confuse improvement in a mechanism metric with improvement in the task.**

Their experiments show that stronger domain invariance can coexist with worse representation quality or worse generalization. In the current Ramen project, the closest analogues are:

```text
better routing/domain purity       does not imply better adaptation
better pseudo-label/cache purity   does not imply better adaptation
```

The correct response is not to tune the proxy harder. It is to identify which stage of the adaptation pipeline is actually producing the error.

---

# Part III — Ramen failure-mode taxonomy

## 8. Why the DG decomposition cannot be copied literally

The source paper studies a static trained predictor decomposed into a representation and classifier. Ramen is different:

- adaptation occurs during inference;
- the memory state changes with time;
- support sets are query-dependent;
- the update is temporary but memory persists;
- the available information depends on the stream history.

Therefore the exact `e0/e1/e2/e3` decomposition from the DG paper should **not** be copied as if it were mathematically identical.

Instead, NB-Ramen should use the same methodology—nested diagnostics and interventions—but define TTA-specific failure modes.

---

## 9. Failure mode F0 — base-model limitation

### Question

Was the query already misclassified before adaptation?

### Primary measurement

Use the paired NoAdapt trace.

Let:

$$
E_{base} = 1 - Acc_{NoAdapt}.
$$

This is not automatically a Ramen failure: it is the set of errors that adaptation has an opportunity to repair.

---

## 10. Failure mode F1 — memory insufficiency

### Question

At the time of the query, does the legal causal memory contain adaptation evidence that could have helped?

### Diagnostic principle

Use evaluator-only oracle interventions over the **existing legal memory**, without changing what deployable methods observe.

Possible bounded oracle families include:

- ID-only gradients in an open-set experiment;
- correctly pseudo-labeled historical items;
- evaluator-defined beneficial-gradient subsets;
- other preregistered oracle subsets justified by a concrete hypothesis.

The key distinction is:

```text
no useful evidence exists in memory
vs
useful evidence exists but Ramen fails to use it
```

The first case is memory insufficiency, not retrieval failure.

---

## 11. Failure mode F2 — retrieval failure

### Question

Useful evidence exists in memory, but does the Ramen retrieval rule fail to retrieve it?

### Required diagnostics

For each query, record at least:

```text
retrieved support item IDs
support predicted classes
support distances
support entropies
support weights
support recencies
```

Evaluator-only joins may additionally attach:

```text
support true classes
support true domains
support correctness
support ID/OOD state
```

These evaluator fields must never feed back into the method.

### Core analysis

Compare an oracle defined over all legal candidates with the same oracle restricted to Ramen's retrieved support set.

A large gap indicates retrieval failure rather than memory insufficiency.

---

## 12. Failure mode F3 — gradient aggregation / compatibility failure

### Question

The retrieved support set contains useful evidence, but do its gradients conflict or combine into a harmful update?

This is the central unresolved hypothesis.

For each active retrieved pseudo-class `c`, compute a class-level local gradient:

$$
h_{q,c}
=
\frac{\sum_{j\in S_{q,c}}\alpha_{qj}g_j}
{\sum_{j\in S_{q,c}}\alpha_{qj}+\epsilon}.
$$

Because Ramen uses SignSGD, measure coordinate-wise direction agreement:

$$
v_{q,k}
=
\frac{1}{C_q}\sum_c \operatorname{sign}(h_{q,c,k}),
$$

and consensus strength:

$$
q_{q,k}=|v_{q,k}|.
$$

Useful summary diagnostics include:

```text
consensus_mean
consensus_p10
consensus_p50
fraction_low_consensus_coordinates
active_support_classes
pairwise class-gradient cosine similarity
pairwise class-gradient sign agreement
```

At this stage these quantities are **diagnostics only**. They should not modify adaptation until the error analysis shows a stable association with harmful updates.

---

## 13. Failure mode F4 — optimizer/update failure

### Question

Even when the aggregate direction appears useful, does the actual temporary SignSGD step still harm the prediction?

This failure can arise from:

- update magnitude through the fixed learning rate;
- discrete sign changes despite small aggregate margins;
- normalization-layer sensitivity;
- interaction between different parameter coordinates.

A useful oracle diagnostic is to compare the actual update with controlled alternatives on an analysis split, while keeping support and aggregation fixed.

No deployable optimizer change should be introduced before this failure is shown to be material.

---

## 14. Failure mode F5 — temporal/scheduling failure

### Question

Does the result depend materially on whether support is batch-atomic or strictly sample-causal?

### Primary control

```text
StructuredAtomicRamen
vs
CausalRamen
```

The current bounded evidence does not show a positive scheduling-only accuracy gain. Therefore F5 should remain a documented protocol dimension, but it is not the primary active mechanism hypothesis.

Potential retrospective diagnostics include:

```text
batch position
future-support count
future-support weight fraction
distance to domain transition
```

These are secondary unless later evidence reopens the scheduling question.

---

# Part IV — Exact outcome-level error decomposition

## 15. Beneficial and harmful adaptation flips

Before introducing complicated oracles, every paired run should use the simplest exact decomposition available.

For each query, compare NoAdapt and Ramen correctness:

| NoAdapt | Ramen | Category |
|---|---|---|
| correct | correct | safe / unchanged correct |
| wrong | correct | beneficial adaptation |
| correct | wrong | harmful adaptation |
| wrong | wrong | unresolved |

Define:

$$
H = P(NoAdapt\ wrong,\ Ramen\ correct)
$$

and:

$$
A = P(NoAdapt\ correct,\ Ramen\ wrong).
$$

Then on an exactly paired sample set:

$$
\boxed{
Acc_{Ramen}-Acc_{NoAdapt}=H-A
}
$$

This is an exact outcome-level decomposition and should be reported before any more speculative mechanism analysis.

Also report conditional rates:

$$
HelpRate=P(Ramen\ correct\mid NoAdapt\ wrong)
$$

$$
HarmRate=P(Ramen\ wrong\mid NoAdapt\ correct).
$$

The most informative error-analysis contrast is:

```text
beneficial adaptation queries
vs
harmful adaptation queries
```

---

# Part V — Mechanism metrics must be tied to failure metrics

## 16. Do not optimize proxy metrics in isolation

Each hypothesis needs both a mechanism metric and an outcome/failure metric.

| Hypothesis | Mechanism metric | Failure/outcome metric |
|---|---|---|
| Latent routing | context/domain NMI, domain probe accuracy | retrieval failure, harmful adaptation |
| Entropy admission | admitted purity, entropy | harmful adaptation, downstream support influence |
| Causal schedule | future-support fraction | atomic-vs-causal paired error |
| Gradient compatibility | sign/cosine agreement | harmful-adaptation probability |
| Open-set contamination | retrieved OOD fraction, gradient direction corruption | ID accuracy degradation, harmful adaptation |

A proxy is scientifically useful only if its changes are linked to changes in the relevant failure component.

---

# Part VI — Diagnostic experiments

## 17. Experiment 1 — representation/domain decodability

This experiment is the retrospective error analysis for LatentRamen.

### Goal

Distinguish:

```text
representation lacks domain signal
vs
router fails to exploit available signal
```

### Procedure

Freeze CLIP image features collected from an analysis stream and train evaluator-only linear probes:

```text
feature -> domain/corruption
feature -> semantic class
```

Also perform a class-conditioned domain probe where feasible:

```text
within one true semantic class:
feature -> domain/corruption
```

### Interpretation

```text
high domain-probe accuracy + low routing NMI
    -> router/formulation failure

low domain-probe accuracy + low routing NMI
    -> raw feature representation is weak for this routing objective

high domain recoverability but no Oracle-domain gain
    -> domain identity is decodable but not the right adaptation partition
```

No router threshold should be retuned before this analysis is complete.

---

## 18. Experiment 2 — entropy admission failure analysis

### Goal

Explain why a cleaner cache produces worse adaptation.

### Required grouping

Partition candidate/admitted items by:

```text
low entropy + correct pseudo-label
low entropy + wrong pseudo-label
high entropy + correct pseudo-label
high entropy + wrong pseudo-label
```

For each group measure:

```text
storage rate
retrieval frequency
total downstream retrieval weight
mean distance when retrieved
gradient cosine agreement with local query support
gradient sign agreement with local query support
```

Define downstream influence for memory item `j`:

$$
I_j=\sum_q \alpha_{qj},
$$

where the sum is over queries for which item `j` is retrieved.

### Key question

Does the entropy gate reject items that look unreliable as classifiers but have high downstream adaptation influence or gradient compatibility?

If yes, this directly explains why pseudo-label purity is not the correct memory-quality objective.

---

## 19. Experiment 3 — gradient conflict versus harmful adaptation

This is the highest-priority new diagnostic experiment.

### Goal

Test the hypothesis:

> queries harmed by adaptation exhibit systematically more conflict among retrieved gradient directions than queries helped by adaptation.

### Primary comparison

Compare consensus statistics for:

```text
NoAdapt wrong -> Ramen correct   (beneficial)
NoAdapt correct -> Ramen wrong   (harmful)
```

### Required outputs

At minimum:

```text
consensus_mean distribution
consensus_p10 distribution
low-consensus coordinate fraction
active support-class count
pairwise sign-agreement distribution
pairwise cosine-similarity distribution
```

Stratify by:

```text
stream type
time since last domain shift
memory occupancy
domain/corruption
seed
```

### Decision

Do **not** implement ConsensusRamen merely because consensus sounds plausible.

Proceed only if:

1. the conflict/consensus metric separates harmful from beneficial adaptation in a stable direction;
2. the association repeats across fixed seeds and more than one structured stream;
3. an evaluator-only gradient oracle indicates that removing or suppressing conflicting components can recover useful updates.

Exact numerical thresholds should be preregistered after a bounded analysis pilot and before final evaluation.

---

## 20. Experiment 4 — open-set oracle gradient analysis

This experiment belongs to the open-world thesis path and should follow the closed-set gradient diagnostics.

### Goal

Determine whether semantic OOD samples materially corrupt Ramen's update direction.

Use a fixed open-set protocol where the model-facing vocabulary contains known classes only, while the evaluator sees known and held-out unknown samples.

For query `q`, compare:

$$
g_q^{all}
=\sum_{j\in S_q}\alpha_{qj}g_j
$$

with an evaluator-only ID-gradient oracle:

$$
g_q^{ID}
=\sum_{j\in S_q,\;j\in ID}\alpha_{qj}g_j.
$$

Measure:

$$
GDC_q=1-\cos(g_q^{all},g_q^{ID})
$$

and SignSGD-relevant sign disagreement:

$$
SDR_q
=\frac1D\sum_k
\mathbf1[\operatorname{sign}(g_{q,k}^{all})\neq\operatorname{sign}(g_{q,k}^{ID})].
$$

Then relate `GDC` and `SDR` to ID harmful-adaptation events.

If OOD ratio increases but gradient corruption and harmful adaptation do not, semantic gradient contamination should not be promoted as the thesis mechanism.

---

# Part VII — Analysis over time, not only aggregate accuracy

## 21. Why temporal analysis is necessary

The source DG paper shows that the dominant failure mode can change with training epoch, regularization strength, algorithm, and dataset. The TTA analogue is that the dominant failure mode can change with stream state.

Required axes include:

```text
timestep
time since domain shift
stream type
memory occupancy
batch size
OOD ratio, when applicable
seed
```

A plausible sequence is:

```text
early episode:
    memory insufficiency dominates

mid episode:
    retrieval/aggregation becomes more informative

after shift:
    stale or conflicting evidence may dominate
```

This must be measured rather than assumed.

---

## 22. Recommended visualization pattern

Following the spirit of the decomposition figures in Galstyan et al., use paired panels.

### Top panel — task/failure outcomes

Plot over time or experimental axis:

```text
base error
beneficial adaptation rate
harmful adaptation rate
memory-oracle gap
retrieval-oracle gap
aggregation-oracle gap
```

### Bottom panel — mechanism diagnostics

Plot corresponding quantities:

```text
domain decodability / routing NMI
cache purity / entropy
future-support fraction
gradient consensus / sign disagreement
retrieved OOD fraction, when applicable
```

The scientific question is not whether a mechanism metric changes. It is whether that change tracks the failure component the mechanism is supposed to explain.

---

# Part VIII — Evidence discipline and data leakage rules

## 23. Evaluator-only diagnostics

Ground-truth fields may be used for analysis and explicitly named oracle methods only.

Evaluator-only fields include:

```text
true class
true domain/corruption
ID/OOD state
beneficial/harmful counterfactual labels
oracle support membership
```

Deployable methods must not use these quantities.

---

## 24. Analysis split versus final evaluation

The source paper emphasizes careful model selection and avoiding test-domain-driven tuning. The same principle applies here.

Use separate roles:

```text
bounded analysis streams
    -> discover and validate failure metrics

held-out/final streams
    -> evaluate preregistered mechanisms and thresholds
```

Do not tune router thresholds, entropy thresholds, consensus thresholds, or oracle-inspired heuristics directly on final target streams.

---

# Part IX — Priority order

## 25. Immediate priority

The research sequence should now be:

```text
1. Freeze conclusions from completed ablations
   - latent routing: unsupported in current form
   - entropy gate: negative ablation
   - causal scheduling: protocol control, no positive isolated gain
   - compression: deferred

2. Add failure-analysis instrumentation
   - support IDs and metadata
   - support weights
   - per-class aggregate gradients
   - gradient sign/cosine diagnostics

3. Compute exact beneficial/harmful adaptation decomposition

4. Run gradient-conflict diagnostic study

5. Run evaluator-only gradient oracles

6. Decide whether ConsensusRamen is justified

7. Only after positive diagnostic evidence:
   implement the smallest consensus mechanism

8. Run method ablations and final experiments

9. Perform final failure-mode error analysis again
```

---

# Part X — Go/no-go logic for the next method

## 26. When ConsensusRamen is justified

ConsensusRamen should be implemented only if the diagnostic phase supports all of the following qualitatively:

```text
gradient conflict is measurably associated with harmful adaptation
that association is stable across fixed seeds/streams
an oracle compatibility intervention recovers part of the harmful-update gap
```

If the diagnostic distributions overlap strongly and oracle compatibility provides no recovery, the consensus hypothesis should be rejected before method development.

This follows the core methodology learned from Galstyan et al.: **first identify the dominant failure, then design the intervention.**

---

# Part XI — Final research narrative

The current evidence should not be narrated as a sequence of failed ideas. It is a sequence of increasingly specific failure diagnoses:

```text
Latent routing
    -> annotated domain identity is not sufficient

Entropy gating
    -> predictive confidence / pseudo-label purity is not sufficient

Causal scheduling
    -> strict causality is a protocol property but does not explain the previous gain

Retrieval profiling
    -> retrieval compression is not currently the dominant bottleneck

Remaining unresolved question
    -> are harmful updates caused by incompatible retrieved gradients?
```

The next thesis hypothesis should therefore be earned by the failure analysis:

$$
\boxed{
\text{Reliable TTA may depend more on gradient compatibility than on sample-level trust proxies.}
}
$$

This statement is currently a **research hypothesis**, not an established result.

---

# Reference

Tigran Galstyan, Hrayr Harutyunyan, Hrant Khachatrian, Greg Ver Steeg, and Aram Galstyan. **Failure Modes of Domain Generalization Algorithms.** CVPR 2022.

The source paper contributes the general methodology of failure-mode decomposition, invariance diagnostics, and intervention-based analysis. The Ramen-specific taxonomy, metrics, oracle designs, and proposed experimental sequence in this report are adaptations for the NB-Ramen research program and should not be attributed to the source paper.
