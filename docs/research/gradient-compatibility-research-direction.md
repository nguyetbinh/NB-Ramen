# Gradient Compatibility Research Direction

## From Context Routing to Consensus-Aware Gradient Memory

**Base method:** Ramen  
**Repository:** `nguyetbinh/NB-Ramen`  
**Working branch:** `open-world-gradient-memory`  
**Status:** research-direction report; context-routing branch is treated as negative exploration, while gradient compatibility is the active direction.

---

## 1. Purpose of this report

This report records the reasoning path that led from the initial context-routing idea to the current gradient-compatibility direction.

The important distinction is:

- the context-routing experiments are **not** the thesis objective;
- they are negative hypothesis tests that helped eliminate an overly strong interpretation of Ramen's domain-consistency principle;
- the active research question is now whether retrieved cached gradients produce a reliable update direction for the current query, especially under semantic OOD contamination.

The current thesis principle is:

\[
\boxed{
\text{retrieve by relevance}
+
\text{adapt by gradient compatibility}
}
\]

---

# 2. Starting point: what Ramen actually assumes

Ramen maintains one cache per predicted class and retrieves the top-\(k\) nearest cached samples from every active class cache using CLIP feature distance.

Its support weighting is:

\[
\alpha_{qj}
=
\exp(-H_j)\exp(-\beta d(z_q,z_j)).
\]

The intended effect is a balance between:

1. **domain consistency** — nearby visual embeddings are more likely to come from the same or a similar domain;
2. **prediction balance** — retrieving from separate predicted-class caches prevents the support set from collapsing toward a few predicted classes.

A key point is that Ramen does **not** require exact same-domain support.

The paper reports that on CIFAR-100-C only about 40.9% of retrieved supports come from the exact same corruption domain as the query. This is much higher than random selection, but it also means useful support remains substantially cross-domain.

Therefore Ramen uses a **soft local neighborhood**, not hard domain isolation.

---

# 3. Initial hypothesis: adapt from samples in the same context

The initial extension hypothesis was:

> If mixed-domain adaptation is difficult because unrelated domains are mixed together, a query should benefit from adapting mainly from cached samples belonging to the same latent deployment context.

This led to `LatentRamen`:

```text
image
  -> CLIP feature z
  -> infer latent context r
  -> predicted class c_hat
  -> store gradient memory under (c_hat, r)
  -> retrieve class-balanced support only from context r
  -> aggregate gradients
  -> temporary TTA update
```

The context router is an unsupervised online prototype router using cosine distance between normalized CLIP features.

This was a reasonable hypothesis to test, but it introduced a stronger assumption than Ramen itself:

\[
\text{Ramen: similar-domain support is preferred}
\]

versus

\[
\text{Latent routing: context membership controls support eligibility}.
\]

---

# 4. Negative result 1: the latent router collapsed

In the bounded CIFAR-100-C pilots, the unsupervised latent router collapsed to a single context even though the stream contained several ground-truth corruption domains.

For example, the block pilot reported:

```text
LatentRamen routing NMI = 0.000
LatentRamen active contexts = 1
```

while the oracle-domain version achieved perfect routing NMI.

This means the observed `LatentRamen` gain over legacy Ramen cannot be interpreted as evidence that useful latent contexts were discovered.

At this point there were two possible explanations:

1. the router itself was poor;
2. exact context separation was not actually the right adaptation criterion.

The oracle experiment was introduced to distinguish them.

Relevant evidence:

- `plans/20260824-latent-ramen-evidence/reports/cifar100c-mps-block-n200-pilot.md`
- `plans/20260824-latent-ramen-evidence/reports/cifar100c-mps-recurring-prefix-n200-pilot.md`

---

# 5. Negative result 2: true-domain oracle routing was still poor

`OracleLatentRamen` removes the routing problem entirely by injecting evaluator-provided true domain IDs.

Its memory is effectively:

\[
M_{c,d},
\]

where \(c\) is predicted class and \(d\) is the true corruption/domain ID.

For query \(q\), retrieval is restricted to:

\[
M_{c,d_q}.
\]

Despite perfect domain assignment, the oracle did not outperform legacy Ramen in the bounded pilots.

For example:

| Pilot | Ramen | OracleLatentRamen |
|---|---:|---:|
| block, 200 samples | 0.315 | 0.305 |
| recurring-prefix, 200 samples | 0.445 | 0.435 |

These are noncanonical development results, but they are sufficient to reject the simple claim:

\[
\boxed{
\text{better domain identification alone} \Rightarrow \text{better adaptation}
}
\]

## 5.1 Why exact-domain routing can hurt

Hard true-domain routing can make the structured memory sparse:

\[
M_c \rightarrow M_{c,d}.
\]

With many classes and relatively few samples per domain episode, many `(class, domain)` buckets are empty or contain very few items.

This can cause:

```text
exact-domain isolation
  -> smaller candidate pool
  -> fewer active predicted classes
  -> weaker prediction balance
  -> noisier aggregate gradients
```

It also removes cross-domain supports that Ramen would normally keep when they are feature-near and potentially useful.

Therefore the oracle failure is compatible with the interpretation that **100% domain consistency is too restrictive**.

However, this explanation alone is not sufficient, because a softer context-routing experiment was also attempted.

---

# 6. Negative result 3: soft context routing still did not solve the problem

A softer router was then explored with the intent to preserve class diversity and cross-domain support while still adding context as a preference signal rather than a hard eligibility rule.

The development result remained poor.

This matters because it weakens the explanation:

> `OracleLatentRamen` failed only because hard partitioning destroyed prediction balance.

The stronger interpretation is:

\[
\boxed{
\text{context relevance itself is not a sufficient adaptation-quality signal}
}
\]

A support sample may be context-relevant but still contribute an undesirable gradient direction. Conversely, a support from another context may provide a useful update direction.

This does **not** make the statement

\[
\text{domain relevance} \neq \text{adaptation compatibility}
\]

itself a thesis contribution. That distinction is conceptually straightforward. The value of these context experiments is instead that they eliminate `better context routing` as the main research objective.

The soft-routing implementation/result should be linked here once its final code and evidence artifact are pushed to the branch.

---

# 7. Important confounder: LatentRamen changed stream semantics

A major implementation issue was discovered after observing that `LatentRamen` sometimes achieved higher micro accuracy than legacy Ramen despite collapsing to one context.

The gain could not plausibly be attributed to context discovery because no useful context separation occurred.

The key difference is **batch visibility**.

## 7.1 Legacy Ramen is batch-atomic

The current legacy `Ramen.forward()` does:

```text
batch = [x1, x2, x3]

compute features/gradients for all samples
        ↓
insert x1, x2, x3 into cache
        ↓
retrieve support for x1, x2, x3
```

Therefore, under an ordered-stream interpretation:

\[
S_1
\]

may contain information from \(x_2\) and \(x_3\).

This is not label leakage, but it is **within-batch future visibility**.

## 7.2 LatentRamen is strictly causal

The structured-memory implementation processes the same batch as:

```text
x1: insert -> retrieve
x2: insert -> retrieve
x3: insert -> retrieve
```

Thus:

\[
S_1 \subseteq \{\text{history},x_1\}
\]

and cannot contain \(x_2\) or \(x_3\).

As a result, the comparison

\[
\text{LatentRamen} > \text{Ramen}
\]

was confounded by two changes:

```text
1. context routing
2. batch-atomic -> causal retrieval
```

If the router collapses to context 0 for every sample, then `LatentRamen` is approximately a **causal Ramen implementation**, not evidence for useful context routing.

---

# 8. Required control for the context-routing branch

The clean experimental design is a 2x2 ablation:

| Retrieval semantics | No context | Latent context |
|---|---|---|
| batch-atomic | `Ramen` | **BatchAtomicLatentRamen** |
| strict causal | `CausalRamen` | `LatentRamen` |

`CausalRamen` already exists in:

```text
src/methods/SupportAblations.py
```

and is explicitly defined as class-balanced Ramen with a fixed context and strictly causal support visibility.

The missing control is `BatchAtomicLatentRamen`: add latent routing while preserving the exact legacy Ramen cache timeline.

The following checks would isolate attribution:

\[
\Delta_{causal}
=
ACC(CausalRamen)-ACC(Ramen)
\]

\[
\Delta_{latent\mid legacy}
=
ACC(BatchAtomicLatentRamen)-ACC(Ramen)
\]

\[
\Delta_{latent\mid causal}
=
ACC(LatentRamen)-ACC(CausalRamen)
\]

If the router remains collapsed, the expected sanity checks are:

\[
ACC(BatchAtomicLatentRamen) \approx ACC(Ramen)
\]

and

\[
ACC(LatentRamen) \approx ACC(CausalRamen).
\]

If these hold, the previous micro gain should be attributed to causalization rather than latent routing.

This control is useful for cleaning up the historical research story, but context routing is no longer the active thesis method.

---

# 9. Refined research question

The context branch suggests that the useful question is not:

> Which domain/context does the query belong to?

The stronger question is:

> Which cached gradient directions are reliable for adapting the model to the current query?

Ramen uses feature similarity to decide which cached samples are relevant, but the optimizer ultimately consumes gradients.

Therefore the active thesis framing is:

\[
\boxed{
\text{retrieval relevance should be separated from update reliability}
}
\]

The intended contribution is **not** to prove the obvious statement that domain similarity and gradient compatibility are different concepts.

The intended contribution is to establish a concrete failure mode:

> Under open-world mixed-domain streams, Ramen's feature-local support can contain semantically OOD cached gradients that materially change the aggregate update direction, and update reliability can be improved by explicitly reasoning about directional support.

---

# 10. Current code support for gradient-compatibility research

The active branch already contains the main infrastructure for this direction.

## 10.1 `OracleIDGradientRamen`

File:

```text
src/methods/OracleIDGradientRamen.py
```

It preserves Ramen's batch-atomic retrieval timeline and retrieves the same support as ordinary Ramen.

Using evaluator-only `is_ood`, it constructs:

\[
g_q^{all}
\]

from all retrieved supports and:

\[
g_q^{ID}
\]

from the same retrieved support set after removing OOD gradient contribution.

It records:

```text
retrieved_ood_fraction
retrieved_ood_weight_fraction
ramen_vs_oracle_id_cosine
ramen_vs_oracle_id_sign_disagreement
```

This is the current oracle diagnostic for gradient contamination.

## 10.2 Existing noncanonical mechanism signal

The current MPS pilot at OOD ratio 0.5 showed approximately:

```text
GDC = 1 - cosine(g_all, g_ID): ~0.14 - 0.20
sign disagreement:             ~0.17 - 0.21
OOD support weight share:      ~0.38 - 0.43
```

`OracleIDGradientRamen` was non-worse than Ramen across the three block seeds in that pilot and improved mean ID accuracy by roughly +1.06 percentage points.

The known-only control at OOD ratio 0 produced effectively zero directional discrepancy.

This is preliminary evidence for:

\[
\boxed{
\text{semantic OOD support}
\rightarrow
\text{gradient direction change}
\rightarrow
\text{oracle removal can help}
}
\]

It is not yet canonical CUDA evidence.

## 10.3 `ConsensusRamen`

File:

```text
src/methods/ConsensusRamen.py
```

`ConsensusRamen` preserves Ramen's:

```text
cache admission
predicted-class partition
feature retrieval
entropy weighting
distance weighting
class balancing
batch-atomic current-support visibility
SignSGD update
parameter reset
```

It changes only the update aggregation.

For each active predicted-class cache:

\[
h_{q,c}
=
\sum_{j\in S_{q,c}}
\alpha_{qj}g_j.
\]

Ordinary Ramen uses:

\[
g_q^{Ramen}
=
\frac{1}{C_q}
\sum_c h_{q,c}.
\]

Consensus computes coordinate-wise sign agreement:

\[
q_{q,k}
=
\left|
\frac{1}{C_q}
\sum_c
\operatorname{sign}(h_{q,c,k})
\right|.
\]

The primary hard-mask mechanism is:

\[
m_{q,k}=\mathbf 1[q_{q,k}\ge\tau]
\]

and:

\[
\boxed{
g_q^{safe}=m_q\odot g_q^{Ramen}}
\]

Current locked development configuration:

```yaml
consensus_threshold: 0.2
min_consensus_classes: 3
consensus_mode: hard_mask
include_current: true
```

The important implementation property is that `ConsensusRamen-v0` retains legacy Ramen's batch-atomic visibility. Therefore any future gain is not silently caused by causalizing the stream semantics.

## 10.4 SignSGD-compatible soft consensus

A positive scalar reweighting such as:

\[
q^\gamma g
\]

cannot produce graded behavior under SignSGD because:

\[
\operatorname{sign}(q^\gamma g)
=
\operatorname{sign}(g)
\]

for positive \(q\).

The implemented soft ablation therefore uses coordinate admission probability:

\[
p_{q,k}=q_{q,k}^{\gamma}
\]

\[
b_{q,k}\sim Bernoulli(p_{q,k})
\]

\[
g_q^{soft}=b_q\odot g_q^{Ramen}.
\]

This remains an ablation rather than the primary method.

## 10.5 `OracleConsensusRamen`

File:

```text
src/methods/OracleConsensusRamen.py
```

This evaluator-only upper bound prevents known OOD items from entering the support cache and then runs ordinary consensus on the retained ID supports.

It answers a different question from `OracleIDGradientRamen`:

- `OracleIDGradientRamen`: how much does OOD contribution alter Ramen's retrieved update?
- `OracleConsensusRamen`: what does consensus look like with oracle-clean support admission?

---

# 11. Current preliminary Consensus result

At OOD ratio 0.5, the noncanonical pilot reported approximately:

| Stream | ConsensusRamen - Ramen ID accuracy |
|---|---:|
| `iid_mixed` | -0.52 pp |
| `block` | +2.66 pp |
| `recurring` | +0.54 pp |
| mean | +0.89 pp |

The correct interpretation is not universal superiority.

The current working hypothesis is narrower:

\[
\boxed{
\text{gradient consensus helps when a coherent local adaptation direction exists}
}
\]

Block streams currently provide the strongest signal; IID mixtures may make consensus over-regularize because a single coherent local direction may not exist.

---

# 12. Main scientific gaps still open

## Gap A — prove that Consensus actually approaches the clean oracle direction

Current diagnostics compare:

\[
g^{Ramen}
\quad\text{vs}\quad
g^{ID}.
\]

The deployable method produces:

\[
g^{Consensus}.
\]

The missing direct mechanism test is:

\[
GDC^{Ramen}
=
1-\cos(g^{Ramen},g^{ID})
\]

versus:

\[
GDC^{Consensus}
=
1-\cos(g^{Consensus},g^{ID}).
\]

Likewise compare sign disagreement:

\[
SDR^{Ramen}
\quad\text{vs}\quad
SDR^{Consensus}.
\]

Primary reductions:

\[
\Delta GDC
=
GDC^{Ramen}-GDC^{Consensus}
\]

\[
\Delta SDR
=
SDR^{Ramen}-SDR^{Consensus}.
\]

Positive values would directly show that Consensus moves the realized update toward the ID-only oracle direction.

This is currently the highest-priority mechanism-validation gap.

---

## Gap B — directly test whether feature similarity predicts gradient compatibility

A stronger failure analysis should quantify the proxy mismatch rather than merely state that the concepts differ.

For retrieved or candidate pairs \((q,j)\), analyze relationships such as:

\[
\cos(z_q,z_j)
\]

versus:

\[
\cos(g_q,g_j)
\]

or SignSGD-relevant agreement:

\[
\frac{1}{D}
\sum_k
\mathbf 1[
\operatorname{sign}(g_{q,k})
=
\operatorname{sign}(g_{j,k})].
\]

The useful scientific question is:

> Does high feature similarity reliably predict a compatible update direction, and how does that relationship change under semantic OOD contamination?

This would establish a concrete proxy mismatch in Ramen rather than relying on the straightforward conceptual statement that domain relevance and gradient compatibility are different.

---

## Gap C — individual support compatibility is not yet solved

Current `ConsensusRamen` filters **gradient coordinates after class-level aggregation**.

It does not yet answer the more granular question:

> For query \(q\), which individual cached sample \(j\) should be used for adaptation?

Current mechanism:

```text
Ramen retrieval
  -> per-class aggregate h_q,c
  -> cross-class sign agreement
  -> coordinate mask
```

A future method could instead define a pairwise compatibility score:

\[
compat(q,j)
\]

and use it for sample selection or support weighting.

This is a possible extension, not a requirement for the current v0 thesis method.

---

## Gap D — canonical evaluation is still pending

Current evidence is development-level and includes CPU/MPS pilots.

Canonical claims still require the locked CUDA evaluation matrix on verified official data, including:

```text
OOD ratios: 0.0 / 0.1 / 0.3 / 0.5
stream modes: iid_mixed / block / recurring
seeds: 0 / 1 / 2
primary dataset: CIFAR-100-C open-set split
secondary dataset: DomainNet open-set split
```

Effect-size and generalization claims should not be made before that matrix is complete.

---

# 13. Recommended research priority

The active order should be:

```text
1. Clean up the historical context-routing attribution
   - run/verify CausalRamen control
   - add BatchAtomicLatentRamen only if needed to close the confounder

2. Complete Consensus-vs-OracleID diagnostics
   - g_ramen
   - g_consensus
   - g_oracle_id
   - GDC / SDR reduction

3. Run direct feature-similarity vs gradient-compatibility analysis
   - pairwise cosine/sign relationships
   - ID vs OOD stratification

4. Freeze method/config

5. Run canonical CUDA matrix

6. Only after mechanism evidence is clear, consider individual sample-level
   compatibility as a larger follow-up extension
```

Do not return to latent context routing as the primary thesis mechanism unless new evidence shows that context information adds value after the causal/batch-atomic confounder is controlled.

---

# 14. Final thesis framing

The context-routing work should appear as a short negative exploration:

> We first explored explicit latent and oracle context routing as a stronger form of Ramen's domain-consistency criterion. Unsupervised routing collapsed, while oracle and soft-routing controls did not establish that better context separation improves adaptation. Further inspection revealed that the initial latent implementation also changed batch visibility from Ramen's batch-atomic semantics to a causal timeline, confounding the observed micro-accuracy gain. These results motivated us to move away from domain identification as the control variable.

The primary story should instead be:

> Ramen retrieves supports through feature-space relevance, but the actual model update is determined by cached gradients. Under open-world contamination, semantically OOD supports can remain feature-local while materially changing the aggregate update direction. We therefore separate retrieval relevance from update reliability and introduce consensus-aware gradient memory to suppress directions unsupported by the retrieved class-balanced memory.

The concise research statement is:

\[
\boxed{
\textbf{Ramen answers: Which samples are relevant?}
}
\]

\[
\boxed{
\textbf{This thesis asks: Which retrieved gradient directions should be trusted?}
}
\]
