# Latent Soft Routing for Ramen

## Research Direction Report

**Branch:** `latent-soft-routing`  
**Base:** `evidence` @ `b054063bf9036e561a9ffc5ef8f21608a2af0dd0`  
**Status:** research proposal / diagnostic plan; not yet experimentally validated

---

## 1. Executive summary

The current `LatentRamen` implementation uses inferred context as a **hard retrieval partition**. A memory item is stored under `(predicted_class, context)`, and a query retrieves support only from the exact matching context bucket.

That design is stricter than the mechanism used by the original Ramen method. Ramen uses embedding similarity to *prefer* domain-consistent samples, while still retaining cross-domain support when it is useful for class balance and gradient stability. The Ramen paper reports that the selected support set is not pure-domain: only about 40.9% of retrieved samples are from the same domain in its retrieval analysis. Therefore, domain consistency in Ramen is a **soft preference**, not a hard exclusivity constraint.

This distinction changes the interpretation of the current negative results for `OracleLatentRamen` and the collapsed `LatentRamen` router.

The current evidence does **not** establish that domain or latent-context information is useless. It establishes a narrower fact:

> Hard partitioning support by context can be too restrictive.

The proposed direction is therefore to replace hard context filtering with **soft context-aware ranking and/or weighting**, while preserving Ramen's global per-class candidate pool and prediction-balanced retrieval.

The core experimental question becomes:

> Does context information improve Ramen when used as an additional relevance signal without excluding cross-context support?

The cleanest first diagnostic is an **oracle soft-routing experiment** using ground-truth domain labels only as a relevance bonus. If oracle soft routing improves over `CausalRamen` while oracle hard routing does not, that would directly support the new hypothesis and explain the earlier negative result mechanistically.

---

## 2. Current evidence that motivates the correction

### 2.1 CausalRamen is currently the correct structured baseline

The current evidence branch has established that the apparent gain of `LatentRamen` cannot be attributed to latent routing because the router collapsed to one context.

The completion report records the following bounded CIFAR-100-C results.

### Three-seed 64-sample pilot

| Method | Micro accuracy |
|---|---:|
| NoAdapt | 34.90% |
| Legacy Ramen | 33.85% |
| CausalRamen | **36.98%** |
| LatentRamen | **36.98%** |

### Canonical 200-sample block pilot

| Method | Micro | Macro | Worst-domain |
|---|---:|---:|---:|
| NoAdapt | 30.5% | 32.03% | 23.44% |
| Legacy Ramen | 31.5% | 38.28% | 28.12% |
| CausalRamen | **33.0%** | **39.45%** | **28.12% |
| LatentRamen | **33.0%** | **39.45%** | **28.12% |

The practical conclusion is:

```text
LatentRamen with one inferred context
≈
CausalRamen
```

Therefore any future routing claim should be measured against `CausalRamen`, not directly against legacy `Ramen`.

### 2.2 There remains a separate causality / implementation confound

The current `CausalRamen` completion report correctly notes that:

```text
CausalRamen - Legacy Ramen
```

still combines:

- scheduling differences;
- structured-memory differences;
- ranking precision / numerical differences.

That issue remains important for claims about strict causality, but it does **not** block the new soft-routing experiment as long as the soft-routing variants and `CausalRamen` share the same structured memory, scheduling, aggregation, optimizer, dtype, and capacity behavior.

For the new direction, the critical controlled comparison is therefore:

```text
CausalRamen
vs
OracleSoftRamen
vs
LatentSoftRamen
```

inside the same structured implementation family.

---

## 3. Problem in the current LatentRamen formulation

### 3.1 Current memory structure

`StructuredGradientMemory` stores items in buckets indexed by:

```text
(predicted_class, inferred_context)
```

The query implementation then performs, for every predicted class:

```python
candidates = self._buckets.get((predicted_class, context), [])
```

and ranks only those candidates.

Therefore, for query item `i` with context `d_i`, a support item `j` is eligible only if:

\[
d_j = d_i.
\]

This is a **hard domain/context filter**.

### 3.2 This is not equivalent to Ramen's domain consistency

Ramen's retrieval principle can be summarized as:

1. preserve prediction balance by retrieving from every predicted-class queue;
2. within each class, prefer samples whose CLIP visual embedding is close to the query;
3. weight retrieved gradients using entropy and feature similarity.

The original mechanism therefore uses domain similarity as a *ranking signal*.

It does not impose:

\[
\text{same domain only}.
\]

The paper's own retrieval analysis reports that only roughly 40.9% of selected support is same-domain. Cross-domain samples therefore remain a substantial part of the effective support set.

The current hard-context implementation changes the method from:

```text
prefer context-compatible support
```

into:

```text
exclude all context-incompatible support
```

These are different algorithms and test different hypotheses.

---

## 4. Why hard filtering can hurt

### 4.1 Reduced support size

For class `c`, standard Ramen can retrieve:

\[
S_i^c = \operatorname{TopK}_{j \in M_c} d(z_i,z_j).
\]

Hard-context Ramen instead retrieves:

\[
S_i^c = \operatorname{TopK}_{j \in M_{c,d_i}} d(z_i,z_j).
\]

Since:

\[
M_{c,d_i} \subseteq M_c,
\]

the hard variant can only preserve or reduce the number of available supports.

This is especially damaging early in a new context or for rare predicted classes.

### 4.2 Reduced class coverage

Ramen explicitly uses class-balanced retrieval. If a query context has not yet accumulated samples for many predicted classes, hard partitioning produces empty `(class, context)` buckets.

The effective active-class count becomes:

\[
A_i
=
\sum_{c=1}^{C}
\mathbf{1}[|M_{c,d_i}|>0].
\]

This may be much smaller than the number of classes represented in the global memory.

Therefore hard domain filtering can accidentally destroy the second core principle of Ramen: **prediction balance**.

### 4.3 Lower effective sample size

Even when support is available, a smaller and more homogeneous support pool may produce a noisier aggregated gradient.

Given normalized support weights `w_j`, effective sample size can be measured by:

\[
ESS
=
\frac{(\sum_j w_j)^2}{\sum_j w_j^2}.
\]

A hard filter may increase domain purity while decreasing ESS and class diversity.

### 4.4 Oracle domain is not an accuracy upper bound

The current `OracleLatentRamen` should not be interpreted as an upper bound on adaptation accuracy.

It answers:

> What happens if retrieval is restricted to the true domain?

It does **not** answer:

> What happens if perfect domain information is optimally used?

A true-domain label can be useful without implying that all cross-domain support should be removed.

Therefore a negative `OracleLatentRamen` result does not falsify the usefulness of domain information.

---

## 5. Revised thesis hypothesis

The previous implicit hypothesis was:

> Test-time adaptation should infer a context and retrieve only support from that context.

The revised hypothesis is:

> Test-time adaptation should infer context compatibility and use it as an additional relevance signal while preserving cross-context candidates, prediction balance, and sufficient gradient diversity.

This changes the architecture from **hard routing** to **soft routing**.

### Old formulation

```text
query
  -> infer context
  -> select matching context bucket
  -> retrieve support inside bucket only
  -> aggregate gradient
```

### New formulation

```text
query
  -> infer context or context posterior
  -> keep global per-class candidate memory
  -> score candidates using
       semantic similarity
       + context compatibility
       + uncertainty / reliability
  -> top-k per class
  -> aggregate gradient
```

The key design requirement is:

> Context changes ranking or weighting, not candidate eligibility.

---

## 6. Proposed method family

The new branch should distinguish hard and soft methods explicitly.

### 6.1 `CausalRamen`

Baseline.

Candidate pool for class `c`:

\[
M_c.
\]

Ranking uses feature distance only.

No context information.

### 6.2 `OracleHardRamen`

Rename / preserve the current oracle behavior for diagnostic comparison.

Candidate pool:

\[
M_{c,d_i}.
\]

This is useful as a deliberately restrictive control.

### 6.3 `OracleSoftRamen`

Uses ground-truth domain only as a relevance bonus.

Candidate pool remains:

\[
M_c.
\]

A simple score is:

\[
s_{ij}
=
-\beta_d d(z_i,z_j)
+
\gamma \mathbf{1}[d_i=d_j].
\]

Equivalent distance form:

\[
\tilde d_{ij}
=
d(z_i,z_j)
-
\lambda \mathbf{1}[d_i=d_j].
\]

Then:

\[
S_i^c
=
\operatorname{TopK}_{j\in M_c} s_{ij}.
\]

No sample is removed solely because it comes from another domain.

### 6.4 `LatentHardRamen`

Current `LatentRamen` semantics.

Keep temporarily as a control rather than as the primary proposed method.

### 6.5 `LatentSoftRamen`

Primary proposed direction.

For each item, the router produces either:

- a discrete context ID;
- a context posterior;
- distances to active prototypes.

Context compatibility is then added to retrieval ranking or gradient weighting.

---

## 7. Soft context compatibility options

The method should be built incrementally, from the least expressive diagnostic to the more general formulation.

### 7.1 Binary same-context bonus

If the router returns discrete context IDs:

\[
a_{ij}
=
\mathbf{1}[\hat d_i=\hat d_j].
\]

Score:

\[
s_{ij}
=
-\beta_d d(z_i,z_j)
+
\gamma a_{ij}.
\]

This is the minimum implementation.

### 7.2 Posterior overlap

If the router produces a context posterior:

\[
q_i \in \Delta^K,
\]

use:

\[
a_{ij}=q_i^\top q_j.
\]

This naturally represents uncertainty around context boundaries.

### 7.3 Prototype-affinity compatibility

If the router exposes distances to prototypes:

\[
r_i(k)=\exp(-d(z_i,\mu_k)/\tau),
\]

normalize `r_i` and use posterior overlap or another similarity metric.

### 7.4 Continuous context embedding

A later extension may learn a context representation `h_i` and use:

\[
a_{ij}=\cos(h_i,h_j).
\]

This should not be attempted until the oracle-soft diagnostic establishes that context information is useful under a soft formulation.

---

## 8. Ranking versus weighting

There are two distinct places where context can enter.

### Option A — context-aware ranking

Use context compatibility to determine which items enter top-k.

\[
s_{ij}
=
-\beta_d d_{ij}
+
\gamma a_{ij}.
\]

Then aggregate with standard Ramen weights.

Advantages:

- directly changes support composition;
- simple diagnostic;
- preserves top-k per class;
- easy to compare with hard routing.

### Option B — context-aware gradient weighting

Keep Ramen's original top-k retrieval, then modify support weights:

\[
w_{ij}
\propto
\exp(
-H_j
-\beta d_{ij}
+\gamma a_{ij}
).
\]

Advantages:

- even less disruptive to Ramen;
- context becomes a pure soft preference;
- cannot reduce class coverage merely by candidate exclusion.

### Recommended order

1. `OracleSoftRankRamen`
2. `OracleSoftWeightRamen`
3. only then latent variants

Do not introduce both ranking and weighting changes simultaneously in the first experiment.

---

## 9. Required invariants

Any soft-routing implementation must preserve the following controls.

### 9.1 Strict causal stream semantics

For query `x_t`, no support from future stream items may be visible.

Use the same sequential insertion/query schedule as `CausalRamen`.

### 9.2 Same memory-capacity semantics

Use:

```text
capacity_scope = per_class
```

for controlled comparison unless an experiment explicitly studies capacity.

Soft routing should not obtain additional effective memory simply because contexts exist.

### 9.3 Global per-class candidate pool

For soft routing, context must not determine eligibility.

Candidate construction should be logically equivalent to:

```text
for predicted_class in classes:
    candidates = all live items with this predicted_class
```

Context metadata should be attached to items but not used as a bucket filter.

### 9.4 Same top-k and aggregation

Keep identical:

- `topk`;
- entropy weighting;
- feature-distance weighting;
- optimizer;
- learning rate;
- inclusion/exclusion of the current sample.

Only the context term should differ.

### 9.5 Recover baseline at zero context strength

A required sanity property is:

\[
\gamma=0
\Rightarrow
\text{SoftRoutingRamen}=\text{CausalRamen}
\]

up to deterministic floating-point equivalence.

This should be unit-tested.

---

## 10. Memory API changes

The current `StructuredGradientMemory.query()` is explicitly context-restricted.

A clean implementation should avoid overloading that method with incompatible semantics.

Recommended API additions:

```python
query_class_balanced_global(
    features,
    topk,
    *,
    context_scores=None,
    query_contexts=None,
    include_current=True,
    current_item_ids=None,
)
```

or a more general scorer interface:

```python
query_class_balanced(
    features,
    topk,
    *,
    candidate_policy="global",
    score_fn=None,
    ...
)
```

For the first implementation, prefer an explicit method rather than a highly generic abstraction.

Every memory item used by soft routing needs context metadata. The existing `_MemoryItem` currently does not store context directly because context is encoded in the bucket key. Soft routing will need either:

1. context stored inside `_MemoryItem`; or
2. global per-class iteration that retains the bucket's context ID while collecting candidates.

The second option is minimally invasive but may be slower. The first option is structurally cleaner if context-aware retrieval becomes a primary research path.

Performance optimization is not a priority until scientific value is established.

---

## 11. Diagnostics required for mechanistic evidence

Accuracy alone is insufficient. The hard-filter hypothesis predicts measurable changes in support composition.

For every query or aggregated window, record:

### 11.1 Returned support count

\[
|S_i|.
\]

### 11.2 Active predicted-class count

Already partially available as `active_classes`.

Measure:

\[
A_i
=
\#\{c: |S_i^c|>0\}.
\]

### 11.3 Class coverage

\[
\text{coverage}_i
=
\frac{A_i}{C}.
\]

### 11.4 Same-domain ratio

For diagnostics where ground-truth domains are available:

\[
r_i^{domain}
=
\frac{\#\{j\in S_i:d_j=d_i\}}{|S_i|}.
\]

This is diagnostic only; latent methods must not use ground-truth domain labels.

### 11.5 Cross-domain support ratio

\[
1-r_i^{domain}.
\]

This is particularly important because Ramen's support set is expected to retain nontrivial cross-domain support.

### 11.6 Effective sample size

Using final aggregation weights:

\[
ESS_i
=
\frac{(\sum_j w_{ij})^2}{\sum_j w_{ij}^2}.
\]

### 11.7 Empty-bucket / missing-class pressure

For hard-routing controls, record the number of class-context combinations with no eligible support for the current query.

### 11.8 Context influence

For soft routing, record:

- fraction of selected items changed relative to `gamma=0`;
- mean context bonus among selected items;
- rank displacement caused by context;
- same-domain ratio as a function of `gamma`.

These metrics allow a mechanistic statement such as:

> Soft routing increased domain-consistent support from X% to Y% while preserving class coverage and ESS, whereas hard routing increased purity but reduced support diversity.

That would be substantially stronger than an aggregate accuracy-only result.

---

## 12. Primary diagnostic experiment

The first experiment should answer only one question:

> Is perfect domain information useful when treated as a soft preference rather than a hard filter?

Use the existing canonical bounded stream so no new dataset confound is introduced.

### Protocol

```text
Dataset: CIFAR-100-C
Backbone: CLIP ViT-B/16
Stream: canonical block
Prefix: 200 samples
Seed: 0
Same stream artifact / fingerprint as current evidence
Same StructuredGradientMemory
Same strict-causal scheduling
Same capacity
Same top-k
Same beta
Same optimizer
Same learning rate
```

### Methods

```text
CausalRamen
OracleHardRamen
OracleSoftRankRamen
```

Optional:

```text
OracleSoftWeightRamen
```

### Context-strength sweep

Use a very small preregistered sweep rather than tuning continuously on the test result.

Example:

```text
gamma in {0, 0.25, 0.5, 1.0}
```

The scale should be normalized relative to the feature-distance term so the interpretation of `gamma` is meaningful.

### Primary quantities

\[
\Delta_{hard}
=
Acc(OracleHard)-Acc(CausalRamen)
\]

\[
\Delta_{soft}
=
Acc(OracleSoft)-Acc(CausalRamen)
\]

### Most informative outcome

If:

\[
\Delta_{hard}<0
\]

but:

\[
\Delta_{soft}>0,
\]

then the new diagnosis is strongly supported:

> Domain information was useful, but hard partitioning destroyed useful support diversity.

This is the highest-value experiment on the branch.

---

## 13. Secondary latent-routing experiment

Only run this if the oracle-soft experiment produces a positive signal.

### Methods

```text
CausalRamen
OracleSoftRamen
LatentHardRamen
LatentSoftRamen
```

### Questions

1. Does soft latent routing outperform hard latent routing?
2. Does the latent router alter retrieval even when its hard context assignment collapses?
3. How much of the oracle-soft gain can latent soft routing recover?
4. Does soft routing improve support composition before it improves accuracy?

### Gap-closure metric

Only if oracle soft routing is better than baseline:

\[
\text{gap closure}
=
\frac{
Acc(LatentSoft)-Acc(Causal)
}{
Acc(OracleSoft)-Acc(Causal)
}.
\]

Do not define this metric when the oracle denominator is non-positive.

---

## 14. Router redesign implications

The current online-prototype router was evaluated primarily through hard context IDs. Under soft routing, its useful output should change.

The router should expose at least one continuous signal:

```text
prototype distances
posterior probabilities
normalized affinities
```

Hard `context_id` can remain for diagnostics but should not be the only adaptation signal.

This may also change the interpretation of the previous one-context collapse.

With hard routing:

```text
one context
=> no context effect
```

With soft routing, a router may still provide useful continuous affinities even when a discrete clustering rule does not spawn multiple contexts.

Therefore future routing diagnostics should separately report:

- number of hard contexts;
- context posterior entropy;
- nearest / second-nearest prototype margin;
- transition-time affinity shifts;
- correlation between affinity and true-domain agreement.

Do not tune the spawn threshold before determining whether the continuous router signal itself contains useful domain/context information.

---

## 15. Natural-domain validation

Synthetic corruption domains are useful for controlled diagnostics but insufficient for a thesis claim about heterogeneous deployment environments.

After the CIFAR diagnostic, validate the same logic on a natural-domain dataset.

Recommended order:

```text
PACS or OfficeHome bounded pilot
then DomainNet bounded pilot
then full DomainNet only if signal persists
```

The important test is not merely whether same-domain ratio increases. It is whether soft context preference improves:

- accuracy;
- worst-domain accuracy;
- negative-adaptation windows;
- support class coverage;
- ESS;
- recurring-domain recovery.

---

## 16. Decision gates

### Gate 1 — Does oracle soft routing help?

If:

\[
OracleSoft \le CausalRamen
\]

across meaningful bounded settings, stop investing in latent-domain routing as the main thesis axis.

Context may still be useful through another signal, but domain-aware support retrieval would lack direct evidence.

### Gate 2 — Is hard routing specifically harmful?

Strong support for the new diagnosis requires a pattern such as:

```text
OracleHard < CausalRamen < OracleSoft
```

plus mechanistic support:

```text
OracleHard:
    higher same-domain purity
    lower class coverage / support count / ESS

OracleSoft:
    higher same-domain preference
    preserved class coverage / support count
```

### Gate 3 — Can latent routing recover oracle-soft benefit?

Proceed to richer router research only if:

\[
LatentSoft > CausalRamen
\]

on more than one controlled setting or seed.

### Gate 4 — Does the result survive natural-domain evaluation?

A thesis-level claim should require a positive signal on at least one natural-domain benchmark.

---

## 17. What not to do yet

Do not add the following before the oracle-soft diagnostic succeeds:

- learned neural routers;
- contrastive context encoders;
- gradient compression;
- FAISS / approximate retrieval;
- complicated reliability gates;
- reinforcement-learning routing policies;
- large hyperparameter searches;
- full ImageNet-C experiments.

The current evidence branch has already shown the value of stopping branches that fail their mechanistic gate. The same discipline should be preserved here.

---

## 18. Minimal implementation plan

### Phase 1 — Preserve hard methods as controls

Rename semantically where practical:

```text
OracleLatentRamen -> OracleHardRamen
LatentRamen       -> LatentHardRamen
```

Backward-compatible aliases may be retained to avoid breaking existing evidence artifacts.

### Phase 2 — Add global per-class soft query

Implement a retrieval path that:

1. gathers all candidates for predicted class `c` across contexts;
2. computes feature distance;
3. reads each candidate's context metadata;
4. computes context compatibility;
5. combines scores;
6. returns top-k per class.

### Phase 3 — Implement oracle soft ranker

Use true domain only in the diagnostic scorer.

Do not use true domain in memory admission, prediction, gradient computation, or final inference.

### Phase 4 — Add support-composition diagnostics

At minimum:

```text
returned_support_count
active_class_count
same_domain_ratio
class_coverage
ESS
```

### Phase 5 — Run canonical 200-sample experiment

No latent router changes yet.

### Phase 6 — Implement latent soft scorer only after Gate 1 passes

Reuse the router's continuous prototype affinity if possible before adding a more complex router.

---

## 19. Suggested class structure

A minimal code structure could be:

```text
src/methods/
├── SupportAblations.py
├── LatentRamen.py               # existing hard method / compatibility
├── OracleLatentRamen.py         # existing hard oracle / compatibility
├── SoftRoutingRamen.py
│   ├── SoftRoutingRamenBase
│   ├── OracleSoftRamen
│   └── LatentSoftRamen
```

and in memory:

```text
src/memory/structured_memory.py
    + query_global_class_balanced(...)
```

Avoid duplicating the causal update loop. The soft methods should reuse the same insertion, gradient aggregation, optimizer, and reset behavior as `CausalRamen` wherever possible.

The research variable should be isolated to **candidate scoring**.

---

## 20. Unit tests required before running evidence

### Baseline recovery

```text
gamma = 0
OracleSoftRamen == CausalRamen
```

for identical deterministic synthetic memory.

### No hard exclusion

A cross-context item must remain eligible under every finite soft-routing strength.

### Same-context preference

If two candidates have equal feature distance but one has higher context compatibility, the compatible candidate must rank first when `gamma > 0`.

### Per-class balance preserved

Soft routing must still return up to `topk` independently for every predicted class.

### Causality preserved

Future item IDs must never appear in earlier query support.

### Oracle isolation

Ground-truth domain may influence only context compatibility in oracle-soft experiments.

### Capacity equivalence

Memory size under `CausalRamen` and soft variants should match under identical admissions and `per_class` capacity.

---

## 21. Expected scientific contributions if successful

A successful result could support three progressively stronger claims.

### Claim 1 — methodological correction

> Domain consistency in mixed-domain TTA should be modeled as a soft preference rather than a hard support partition.

### Claim 2 — mechanism

> Hard routing increases support purity but can reduce prediction balance, class coverage, and effective sample size; soft routing recovers context preference without sacrificing support diversity.

### Claim 3 — latent context adaptation

> Unsupervised context compatibility can improve support selection even when discrete domain discovery is unreliable or does not align perfectly with ground-truth domains.

The third claim is the most thesis-relevant because it moves beyond domain classification toward **adaptation-compatible context estimation**.

---

## 22. Revised thesis framing

The previous framing was close to:

> infer the latent domain, then adapt from that domain.

The revised framing should be:

> infer latent adaptation compatibility and use it to softly reweight the evidence available at test time.

This is more general because:

- adaptation contexts do not need to equal human-defined domains;
- uncertain boundaries can be represented continuously;
- cross-domain support remains available;
- the method naturally reduces to Ramen when context information is uninformative.

A possible thesis-level question is:

> **How should a deployed model identify and weight relevant historical evidence for reliable test-time adaptation under heterogeneous, non-stationary streams?**

Ramen provides semantic similarity and prediction balance. The proposed extension adds **soft latent compatibility** without replacing those principles.

---

## 23. Immediate next experiment

The branch should not start with a new learned router.

The next experiment should be exactly:

```text
CIFAR-100-C canonical block, n=200, seed=0

CausalRamen
OracleHardRamen
OracleSoftRankRamen gamma=0
OracleSoftRankRamen gamma>0 (small preregistered sweep)
```

Primary evidence to collect:

```text
micro accuracy
macro-domain accuracy
worst-domain accuracy
negative-adaptation windows
same-domain support ratio
cross-domain support ratio
returned support count
active-class count
class coverage
ESS
```

### Go condition

Continue to latent soft routing if a nonzero soft context preference improves the controlled baseline and does so without collapsing support diversity.

### Stop / pivot condition

If perfect domain information does not help even as a soft preference, do not invest further in domain-discovery routing. Shift the thesis toward another compatibility signal, such as gradient agreement, reliability, or causal support construction.

---

## 24. Bottom line

The current negative oracle-hard evidence should not be treated as evidence against context-aware adaptation in general.

The current implementation tests:

\[
\boxed{\text{domain exclusivity}}
\]

whereas the original Ramen mechanism motivates:

\[
\boxed{\text{domain/context preference}}
\]

The new `latent-soft-routing` direction therefore makes one controlled correction:

> **Keep the full prediction-balanced per-class support pool and let context influence ranking or weighting softly.**

The first priority is not to improve the router. It is to determine, with an oracle diagnostic, whether soft context information has causal value at all.

If the pattern becomes:

```text
OracleHard < CausalRamen < OracleSoft
```

with preserved class coverage and support ESS, the project will have a substantially cleaner and more defensible research direction than the current hard-routing formulation.
