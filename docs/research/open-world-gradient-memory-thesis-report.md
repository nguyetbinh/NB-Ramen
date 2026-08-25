# Open-World Gradient Memory for Ramen

## Thesis Direction Report

**Base paper:** Ramen: Robust Test-Time Adaptation of Vision-Language Models with Active Sample Selection  
**Repository:** `nguyetbinh/NB-Ramen`  
**Working branch:** `evidence`  
**Research direction:** Open-World / Open-Set mixed-domain Test-Time Adaptation  

---

## 1. Thesis objective

The thesis objective is:

> **Prevent semantic OOD contamination of cached adaptation gradients while preserving useful mixed-domain adaptation.**

The central idea is not to make Ramen's cache merely "cleaner" according to pseudo-label correctness. The goal is to make the gradient memory preserve information that is genuinely useful for adaptation while suppressing gradient components that are harmful under semantic uncertainty.

A concise formulation is:

$$
\boxed{
\text{domain-relevant evidence}
+
\text{adaptation-compatible gradients}
}
$$

instead of:

$$
\boxed{
\text{high-confidence samples only}
}
$$

A possible working title is:

> **Trustworthy Gradient Memory for Open-World Mixed-Domain Test-Time Adaptation**

---

## 2. Why Ramen is a suitable base paper

Ramen addresses mixed-domain TTA. Instead of adapting each query using an undifferentiated mixed stream, it retrieves a query-specific support set from a persistent memory.

For a test sample $x_i$, Ramen computes:

- image embedding $z_i$;
- zero-shot logits;
- pseudo-label $\hat y_i$;
- entropy $H_i$;
- per-sample adaptation gradient $g_i$.

The original implementation then stores the sample in the cache corresponding to the predicted class and later retrieves cached gradients according to feature similarity and class balancing.

Conceptually:

```text
x_i
 ↓
embedding z_i + pseudo-label + entropy + gradient g_i
 ↓
class-partitioned memory
 ↓
retrieve nearby support samples
 ↓
weighted gradient aggregation
 ↓
temporary adaptation
 ↓
prediction
 ↓
reset model parameters
```

The important architectural property is that the model parameters are reset after inference while the memory persists across the stream.

Therefore, the long-lived adaptation state is primarily:

$$
\boxed{\text{the cached feature-gradient memory}}
$$

This makes memory contamination a natural research target.

---

## 3. Original intuition: semantic OOD can contaminate persistent gradient memory

Consider an open-world stream containing both known and unknown semantic classes.

Example:

```text
rainy dog      -> known
rainy cat      -> known
rainy fox      -> unknown
night wolf     -> unknown
sketch car     -> known
```

The CLIP classifier only contains known classes. Therefore an unknown sample such as `fox` may still be assigned a known pseudo-label:

```text
fox -> dog
```

Ramen may then cache:

$$
(z_{\text{fox}}, g_{\text{fox}}, \hat y=\text{dog}).
$$

This is more important than a single bad TTA update because the gradient becomes persistent memory:

```text
unknown sample
      ↓
wrong / unsafe adaptation gradient
      ↓
cached
      ↓
retrieved for future queries
      ↓
reused multiple times
```

Thus a local semantic error can potentially become persistent adaptation contamination.

This motivates the thesis-level question:

> **How should a memory-based TTA system decide which parts of cached adaptation evidence are safe to reuse under semantic OOD?**

---

## 4. First attempted solution: confidence / entropy-based reliability

The first reliability hypothesis was:

$$
\text{low entropy}
\Rightarrow
\text{reliable memory item}.
$$

This led to `EntropyGatedLatentRamen`, where memory admission used normalized predictive entropy:

$$
H_{norm}(x)
=
\frac{H(p(y\mid x))}{\log K}.
$$

A sample was admitted if:

$$
H_{norm}(x) \le 0.50.
$$

This mechanism successfully produced a cleaner and smaller memory.

Observed evidence included:

- substantially lower admitted pseudo-label contamination;
- higher pseudo-label accuracy among admitted samples;
- much lower retained memory usage.

However, adaptation accuracy consistently degraded relative to ungated `LatentRamen` in the bounded pilot experiments.

For example, in the CIFAR-100-C $n=200$ pilot:

- LatentRamen micro accuracy: approximately $0.330$;
- Entropy-gated micro accuracy: approximately $0.315$;
- admitted contamination was greatly reduced;
- retained memory was reduced by approximately $85\%$.

The same negative direction appeared across the three-seed short-prefix replication.

---

## 5. Main finding from the negative result

The entropy-gating experiment provides an important research insight:

$$
\boxed{
\text{pseudo-label correctness}
\neq
\text{adaptation usefulness}
}
$$

and more generally:

$$
\boxed{
\text{cleaner memory}
\neq
\text{better adaptation memory}
}
$$

A sample can be misclassified yet still contain a gradient component useful for adapting to the current domain.

Conversely, a high-confidence sample is not guaranteed to produce a gradient that improves adaptation.

Therefore semantic reliability should not be defined only from:

- entropy;
- maximum probability;
- pseudo-label correctness proxy.

The negative result suggests that the relevant object is not merely the sample prediction, but the **adaptation gradient produced by that sample**.

This changes the research question from:

> "Is this sample trustworthy?"

into:

> **"Is this gradient compatible with the adaptation evidence provided by other relevant samples?"**

---

## 6. Important limitation of current evidence

The current `evidence` branch primarily studies closed-set CIFAR-100-C streams.

The existing "contamination" diagnostics mostly correspond to pseudo-label errors:

$$
\hat y_i \neq y_i.
$$

This is not yet true semantic OOD contamination.

A proper open-set experiment must explicitly divide the semantic label space:

$$
\mathcal Y
=
\mathcal Y_K
\cup
\mathcal Y_U,
$$

with:

$$
\mathcal Y_K \cap \mathcal Y_U = \emptyset.
$$

For CIFAR-100-C, a clean MVP protocol is:

$$
100\text{ classes}
=
80\text{ known}
+
20\text{ unknown}.
$$

The CLIP text classifier should contain prompts only for the $80$ known classes, while the test stream still contains images from all $100$ classes.

Critically, known and unknown classes should experience the **same corruption/domain process**.

Example:

```text
fog / dog       -> known
fog / car       -> known
fog / fox       -> unknown

snow / dog      -> known
snow / car      -> known
snow / fox      -> unknown
```

This separates:

$$
\text{semantic novelty}
$$

from:

$$
\text{domain/style novelty}.
$$

Using a completely different dataset as OOD in the first experiment would make it difficult to determine whether the method detects semantics or merely dataset/domain differences.

---

## 7. Updated thesis hypothesis: adaptation compatibility instead of confidence

The revised hypothesis is:

> **A cached gradient should be trusted according to its compatibility with local adaptation evidence, not only according to prediction confidence.**

The central signal becomes **gradient consensus / gradient agreement**.

Suppose a query has support gradients:

$$
g_1,g_2,\ldots,g_n.
$$

If several domain-relevant samples contain a common adaptation direction, this shared component can be interpreted as local adaptation evidence.

Example:

```text
rainy dog       -> →→→↑
rainy cat       -> →→↑
rainy car       -> →→→↑

rainy fox/OOD   -> ←→↓
```

The first three gradients share a dominant direction. The OOD gradient contains both compatible and conflicting components.

The desired behavior is therefore not necessarily:

```text
OOD -> discard entire gradient
```

but rather:

```text
candidate gradient
       ↓
compare with local gradient consensus
       ↓
compatible component   -> preserve
conflicting component  -> suppress / downweight
```

This directly matches the thesis objective:

$$
\boxed{
\text{prevent harmful contamination}
+
\text{preserve useful adaptation information}
}
$$

---

## 8. Why this direction is especially suitable for Ramen

Ramen already computes per-sample gradients efficiently.

Therefore gradient reliability can be studied without adding a separate backward pass for every memory item.

The implementation also uses SignSGD in the released configurations. The final update is approximately:

$$
\theta'
=
\theta
-
\eta\,\operatorname{sign}(g).
$$

This has an important consequence.

Simply scaling the final aggregated gradient is often insufficient because:

$$
\operatorname{sign}(0.1g)
=
\operatorname{sign}(g).
$$

Reliability must influence **which gradients contribute before aggregation**, so that it changes the final coordinate-wise direction.

This makes gradient agreement particularly natural for Ramen.

---

## 9. Candidate mechanism: Consensus-Aware Gradient Memory

### 9.1 Gradient consensus

For support gradients $g_1,\ldots,g_n$, define an element-wise sign consensus:

$$
s_k
=
\operatorname{sign}
\left(
\sum_i \operatorname{sign}(g_{ik})
\right).
$$

For memory item $j$, define sign agreement:

$$
r_j^{grad}
=
\frac{1}{D}
\sum_{k=1}^{D}
\mathbf 1
\left[
\operatorname{sign}(g_{jk})=s_k
\right].
$$

Then:

$$
0\le r_j^{grad}\le1.
$$

A high value means the sample gradient agrees with the local adaptation consensus.

A low value indicates conflicting adaptation evidence.

### 9.2 Soft gradient suppression

Rather than rejecting a sample completely, a coordinate mask can be used:

$$
m_{jk}
=
\mathbf 1
\left[
\operatorname{sign}(g_{jk})=s_k
\right].
$$

Then:

$$
\tilde g_j
=
m_j\odot g_j.
$$

This keeps the gradient components that agree with consensus while suppressing incompatible components.

### 9.3 Integration with Ramen

Ramen currently has a conceptual weight:

$$
w_{ij}^{Ramen}
=
\exp(-H_j)
\exp(-\beta d(z_i,z_j)).
$$

A consensus-aware form could be:

$$
w_{ij}
=
\exp(-H_j)
\exp(-\beta d(z_i,z_j))
(r_j^{grad})^\gamma.
$$

The final aggregate is:

$$
g_i^*
=
\sum_{j\in S_i}
w_{ij}\tilde g_j.
$$

This preserves Ramen's domain-relevance and prediction-balance structure while introducing adaptation compatibility.

---

## 10. Domain relevance and gradient reliability must remain separate

A central thesis distinction is:

$$
\boxed{
\text{domain relevance}
\neq
\text{gradient reliability}
}
$$

Consider:

```text
query: rainy dog
memory item: rainy fox
```

The sample may be highly relevant to the current domain:

$$
D_{ij}\approx1,
$$

because both images contain the same corruption/environment.

However, its classification/adaptation gradient may be semantically unsafe:

$$
R_j\ll1.
$$

Therefore an open-world memory should ideally retain domain information while controlling gradient influence.

This motivates a longer-term factorized memory view:

```text
                 sample
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   domain evidence      adaptation gradient
         │                   │
         ▼                   ▼
   DOMAIN MEMORY       GRADIENT MEMORY
   broad retention      reliability-aware
         │                   │
         └─────────┬─────────┘
                   ▼
          safe local adaptation
```

This factorization is a possible later-stage contribution. It should not be implemented before the open-set benchmark and gradient-consensus evidence are established.

---

## 11. Oracle experiments required before the final mechanism

The next experimental infrastructure should include true semantic-OOD oracle variants.

| Variant | OOD embedding/context | OOD gradient | Purpose |
|---|---:|---:|---|
| Ramen | yes | yes | original behavior |
| Oracle-Drop-OOD | no | no | upper bound for complete rejection |
| Oracle-No-OOD-Gradient | yes | no | retain domain information but remove semantic gradient |
| Oracle-ID-only | ID only | ID only | trusted-memory reference |

The most informative comparison is:

$$
\boxed{
\text{Oracle-No-OOD-Gradient}
\text{ vs }
\text{Oracle-Drop-OOD}
}
$$

If keeping OOD embeddings/context while removing their gradients performs better than fully dropping OOD samples, this supports the core thesis statement:

> unknown samples can contain useful domain information even when their adaptation gradients are unsafe.

The second important comparison is:

$$
\boxed{
\text{Oracle-No-OOD-Gradient}
\text{ vs }
\text{Ramen}
}
$$

If the oracle improves over Ramen, this directly establishes a semantic gradient-contamination gap.

---

## 12. Metrics for the thesis

The final evaluation should not rely only on classification accuracy.

### ID utility

$$
ACC_{ID}
$$

Measures adaptation performance on known classes.

### OOD detection

Use metrics such as:

$$
AUROC,
\qquad
FPR95,
\qquad
OSCR.
$$

### Cache Contamination Rate

For true semantic OOD:

$$
CCR
=
\frac{\#\text{OOD entries retained in gradient memory}}
{\#\text{gradient-memory entries}}.
$$

### Retrieved Gradient Contamination

Measure the amount of actual adaptation weight contributed by semantic OOD:

$$
RCR_i
=
\frac{
\sum_{j\in S_i}w_{ij}\mathbf 1[j\in OOD]
}{
\sum_{j\in S_i}w_{ij}
}.
$$

### Gradient deviation from oracle-ID adaptation

Construct the oracle ID-only support gradient:

$$
g_i^{ID}
$$

and compare it with the actual aggregate:

$$
g_i^{all}.
$$

Define:

$$
D_i
=
1-\cos(g_i^{all},g_i^{ID}).
$$

This directly measures how much OOD-contaminated memory changes the adaptation direction.

### Efficiency

Retain Ramen's deployment motivation by reporting:

- latency;
- retained memory bytes;
- cache size;
- additional compute caused by the reliability mechanism.

---

## 13. Research questions

The thesis can be organized around three research questions.

### RQ1 — Semantic OOD contamination

> **How does semantic OOD affect persistent gradient memory in mixed-domain Ramen?**

This establishes whether unknown samples create a measurable gradient-contamination problem.

### RQ2 — Adaptation compatibility

> **Can local gradient agreement identify adaptation evidence that is more useful than confidence-based memory admission?**

This directly follows from the negative entropy-gating result.

### RQ3 — Preserve domain information

> **Can useful domain information from semantic-OOD samples be retained without allowing unsafe semantic gradients to dominate adaptation?**

This is the most important long-term research question and creates a path toward a later paper.

---

## 14. Current evidence and decisions

### Established engineering evidence

The current branch already provides:

- reproducible Ramen and NoAdapt runs;
- deterministic structured stream builders;
- block and recurring stream support;
- persistent-memory instrumentation;
- oracle and support-ablation infrastructure;
- latency and memory evidence;
- per-sample trace generation;
- entropy-gated memory admission;
- strict artifact validation and repeatability checks.

### Negative results that should be retained

The following should remain part of the thesis rather than being discarded:

1. The unsupervised latent router collapsed to one context in the bounded pilots.
2. Oracle latent routing did not provide a clear upper bound over Ramen in those pilots.
3. Entropy gating produced cleaner memory but consistently worse adaptation accuracy.
4. Retrieval compression was not justified by the initial bounded profiling evidence.

These results narrow the problem and motivate moving away from latent routing and confidence-only admission toward adaptation-gradient compatibility.

---

## 15. Updated implementation roadmap

### Phase A — Freeze existing evidence

Do not overwrite entropy-gating runs.

Treat `EntropyGatedLatentRamen` as a negative ablation demonstrating:

$$
\text{confidence purity}
\not\Rightarrow
\text{adaptation utility}.
$$

### Phase B — Build true open-set CIFAR-100-C

Implement:

$$
80\text{ known}+20\text{ unknown classes}.
$$

Requirements:

- CLIP text vocabulary contains only known classes;
- stream contains both known and unknown samples;
- same corruption process for known and unknown;
- evaluator records `is_ood` but the method never receives it;
- support multiple OOD ratios.

Recommended ratios:

$$
\rho_{OOD}
\in
\{0.1,0.3,0.5\}.
$$

### Phase C — Oracle gradient-contamination study

Implement:

- `OracleDropOODRamen`;
- `OracleNoOODGradientRamen`;
- `OracleIDOnlyRamen`.

Log:

- OOD cache occupancy;
- OOD retrieval contribution;
- aggregate-gradient deviation;
- ID accuracy;
- OOD metrics.

### Phase D — Gradient-consensus prototype

Start with the simplest mechanism:

$$
r_j^{grad}
=
\text{sign agreement with local support consensus}.
$$

Use it only in aggregation first.

Do not initially add:

- a learned reliability network;
- augmentation consistency;
- latent routing;
- dual memory;
- gradient compression.

### Phase E — Reliability-aware memory lifecycle

If consensus is useful, extend it to:

```text
WRITE
 ↓
should this gradient enter trusted memory?

RETRIEVE
 ↓
is this cached gradient compatible with the query support?

AGGREGATE
 ↓
how much of the gradient should contribute?
```

### Phase F — Factorized domain / gradient memory

Only after the previous phases are understood, investigate:

$$
\boxed{
\text{domain memory}
\neq
\text{gradient memory}
}
$$

This is the most promising path toward a follow-up publication.

---

## 16. Thesis contribution target

A reasonable thesis contribution is not:

> "We introduce a better OOD detector for Ramen."

It is:

> **We study semantic contamination in persistent gradient-memory TTA and introduce an adaptation-compatibility mechanism that preserves domain-relevant evidence while suppressing unsafe gradient influence.**

The contribution can be organized as:

1. **Problem formulation:** semantic OOD contamination in cached sample-level adaptation gradients.
2. **Empirical finding:** confidence-based memory purity is insufficient to characterize adaptation usefulness.
3. **Method:** consensus-aware / compatibility-aware gradient memory.
4. **Evaluation:** mixed-domain + semantic-OOD streams with both predictive and memory-level metrics.

---

## 17. Long-term research trajectory

The thesis should leave a path toward later work:

```text
Ramen
  ↓
Mixed-domain gradient memory
  ↓
Semantic-OOD contamination analysis
  ↓
Consensus-aware trustworthy gradient memory
  ↓
Factorized domain / semantic adaptation evidence
  ↓
Continual Open-World gradient memory
  ↓
Trustworthy Open-World Test-Time Adaptation
```

A particularly promising follow-up question is:

> **Can an OOD sample contribute useful domain adaptation information even when its semantic adaptation gradient is unsafe?**

This goes beyond simple OOD filtering and connects memory-based TTA with open-world adaptation, continual adaptation, and trustworthy model self-update.

---

## 18. North Star

The thesis should remain centered on one distinction:

$$
\boxed{
\textbf{
Prediction confidence is not the same as adaptation-gradient reliability.
}
}
$$

The goal is therefore not to build the cleanest memory according to pseudo-label correctness.

The goal is to build a memory whose stored evidence produces **useful and safe adaptation under mixed-domain semantic uncertainty**.

In one sentence:

> **Retrieve by domain relevance, adapt by gradient compatibility.**
