# Open-World Gradient Memory for Ramen

## Thesis Direction Report

**Base paper:** *Ramen: Robust Test-Time Adaptation of Vision-Language Models with Active Sample Selection*  
**Repository:** `nguyetbinh/NB-Ramen`  
**Working branch:** `evidence`  
**Research direction:** Open-World / Open-Set Mixed-Domain Test-Time Adaptation  

---

## 1. Thesis objective

The thesis objective is:

> **Prevent semantic OOD contamination of cached adaptation gradients while preserving useful mixed-domain adaptation.**

The central design principle is:

$$
\boxed{
\text{retrieve by domain relevance}
+
\text{adapt by gradient compatibility}
}
$$

The thesis should not define memory quality only through prediction confidence or pseudo-label correctness. A sample may be semantically uncertain or even misclassified while still carrying useful domain-adaptation information. Conversely, a confident sample may produce an adaptation gradient that conflicts with the locally useful adaptation direction.

The final target is therefore not a generic OOD filter. It is a **gradient-memory mechanism that preserves adaptation-compatible evidence and suppresses harmful semantic gradient components**.

A working title is:

> **Consensus-Aware Gradient Memory for Open-World Mixed-Domain Test-Time Adaptation**

---

## 2. Why Ramen is a suitable base

Ramen addresses mixed-domain TTA by constructing a sample-specific support set from cached historical samples. For each test sample, the method:

```text
image
  -> CLIP feature z
  -> zero-shot logits
  -> pseudo-class c_hat
  -> entropy loss
  -> per-sample gradient g
  -> cache (z, g, entropy) under predicted class
  -> retrieve nearby support from class memories
  -> entropy + feature-distance weighting
  -> aggregate cached gradients
  -> temporary adaptation
  -> inference
  -> reset model parameters
```

The important architectural property is:

$$
\boxed{
\text{model parameters are temporary, but gradient memory persists}
}
$$

Therefore, if an unreliable sample enters the cache, its gradient may influence multiple future queries even though the adapted model itself is reset after each prediction.

This makes persistent gradient memory a natural research object for trustworthiness.

---

## 3. Original hypothesis and what the current evidence shows

### 3.1 Initial hypothesis

The first hypothesis was:

$$
\text{unreliable prediction}
\Rightarrow
\text{unreliable memory item}
$$

which motivated entropy-based memory admission.

The implemented `EntropyGatedLatentRamen` uses normalized predictive entropy:

$$
\bar H(x)
=
\frac{H(p(y\mid x))}{\log K}
$$

and admits a sample only when:

$$
\bar H(x)\le0.5.
$$

### 3.2 Observed result

The entropy gate clearly improves pseudo-label purity and reduces retained memory, but it consistently reduces adaptation accuracy in the current pilots.

Representative evidence from the existing branch:

- admitted pseudo-label accuracy improves substantially;
- admitted contamination decreases substantially;
- retained memory decreases strongly;
- final adaptation accuracy is worse than ungated `LatentRamen`;
- the negative direction repeats across the small multi-seed pilot.

Therefore the current evidence supports:

$$
\boxed{
\text{cleaner pseudo-label memory}
\not\Rightarrow
\text{better adaptation memory}
}
$$

and rejects the simple assumption:

$$
\boxed{
\text{low entropy}
\Rightarrow
\text{useful adaptation gradient}
}
$$

The entropy-gated method should remain in the repository as a **negative ablation**, not be tuned further on the final test streams.

---

## 4. Important scope correction: current contamination is not yet semantic OOD

The existing CIFAR-100-C experiments are still closed-set: CLIP receives the full dataset class vocabulary. Current "contamination" mainly means that the pseudo-label differs from the true class.

That is not yet the thesis setting.

The thesis requires a real semantic open-set protocol:

$$
\mathcal Y
=
\mathcal Y_K\cup\mathcal Y_U,
$$

with:

$$
\mathcal Y_K\cap\mathcal Y_U=\varnothing.
$$

For the first benchmark, use a fixed class split on CIFAR-100-C, for example:

$$
100
=
80\text{ known}
+
20\text{ unknown}.
$$

CLIP receives prompts only for the 80 known classes, while the test stream still contains samples from all 100 classes.

Example:

```text
fog / dog       -> known
fog / car       -> known
fog / fox       -> unknown
snow / dog      -> known
snow / fox      -> unknown
```

This design keeps the domain/corruption mechanism controlled while introducing semantic novelty.

The benchmark should support several unknown ratios:

$$
\rho_{OOD}\in\{0,0.1,0.3,0.5\}.
$$

This benchmark must be implemented before making any claim about semantic-OOD gradient contamination.

---

## 5. Revised research hypothesis

The revised hypothesis is not that an OOD sample should be discarded.

Instead:

> **A cached adaptation gradient is useful when its update direction is compatible with the locally shared adaptation direction, regardless of whether the originating sample has a perfectly correct pseudo-label.**

Conceptually, a sample gradient may be decomposed as:

$$
g_i
=
g_i^{domain}
+
g_i^{semantic}
+
\epsilon_i.
$$

For samples sharing the same environment, part of the domain-induced adaptation signal may be common across different semantic classes.

An unknown sample may therefore contain:

$$
\underbrace{g_i^{domain}}_{\text{potentially useful}}
+
\underbrace{g_i^{semantic}}_{\text{potentially harmful}}.
$$

A hard OOD filter removes both components.

The thesis instead aims to retain directions corroborated by the support memory while suppressing unsupported or conflicting directions.

---

## 6. Research pipeline

The next research cycle should be:

$$
\boxed{
\text{Open-Set Benchmark}
\rightarrow
\text{Oracle Gradient Analysis}
\rightarrow
\text{Gradient Consensus}
\rightarrow
\text{Consensus-Aware Ramen}
}
$$

The current latent router is not part of this cycle. Existing evidence shows routing collapse in the bounded pilots and no useful oracle routing upper bound. The thesis should isolate the gradient-memory question first.

---

# Part I — Open-Set Ramen Benchmark

## 7. Open-set dataset wrapper

Create an Open-Set CIFAR-100-C wrapper with a fixed known/unknown split.

Required properties:

```text
known_class_ids
unknown_class_ids
known_classes
is_ood
original_label
known_label_or_minus_one
```

The model-facing class vocabulary must contain only:

$$
\mathcal Y_K.
$$

The stream-facing dataset must still contain:

$$
\mathcal Y_K\cup\mathcal Y_U.
$$

The method must never receive `is_ood` or the original unknown label. These fields are evaluator-only.

Recommended initial split:

```text
CIFAR-100-C
├── 80 known classes
└── 20 held-out unknown classes
```

The exact class split must be versioned and fixed before final evaluation.

---

## 8. Open-set stream construction

Reuse the existing deterministic mixed-domain stream infrastructure.

The first required stream modes are:

```text
iid_mixed
block
recurring
```

Each stream must mix:

```text
domain shift
+
known semantic classes
+
unknown semantic classes
```

The unknown ratio must be explicit in the stream manifest.

Required controls:

```text
--open_set
--known_class_split <version>
--ood_ratio <float>
```

The stream fingerprint must bind the class split, OOD ratio, domain order, and sample identities.

---

## 9. Open-set evaluation metrics

Add the following primary metrics:

### Known-class adaptation

$$
ACC_{ID}
$$

computed only on known samples.

### OOD detection

At minimum:

$$
AUROC,
\qquad
FPR95.
$$

Use a clearly defined pre-adaptation OOD score such as maximum logit, MSP, entropy, or energy. OOD detection is an evaluation/control signal here, not the core contribution.

### Open-set combined performance

Use one combined metric such as OSCR or H-score where appropriate.

### Memory contamination diagnostics

For evaluator-only analysis:

$$
CCR
=
\frac{\#\text{OOD items stored}}{\#\text{stored items}}
$$

and weighted retrieved contamination:

$$
RCR_t
=
\frac{
\sum_{j\in S_t}\alpha_{tj}\mathbf 1[j\in OOD]
}{
\sum_{j\in S_t}\alpha_{tj}
}.
$$

These diagnostics must never affect adaptation.

---

# Part II — Oracle Gradient Analysis

## 10. Why Oracle analysis is required

Before implementing the final method, establish whether semantic OOD gradients actually distort Ramen's adaptation direction.

Oracle methods may use hidden ID/OOD labels only for diagnostic upper bounds. They are not deployable methods.

Required variants:

| Variant | OOD embedding/context | OOD gradient contribution |
|---|---:|---:|
| `Ramen` | yes | yes |
| `OracleDropOODRamen` | no | no |
| `OracleIDGradientRamen` | may remain available for analysis | no |
| `OracleConsensusRamen` | yes | only consensus-compatible component |

The most important comparison is:

$$
\boxed{
\text{Ramen}
\quad vs \quad
\text{OracleIDGradientRamen}
}
$$

because it measures whether OOD gradient contribution is harmful when all other stream conditions are fixed.

---

## 11. Oracle gradient target

For a query $q$, let standard Ramen aggregate:

$$
g_q^{all}
=
\sum_{j\in S_q}\alpha_{qj}g_j.
$$

Using hidden evaluation labels, construct an ID-only oracle:

$$
g_q^{ID}
=
\sum_{j\in S_q,\;j\in ID}\alpha_{qj}g_j.
$$

Measure direction corruption:

$$
GDC_q
=
1-\cos(g_q^{all},g_q^{ID}).
$$

Also record sign disagreement because Ramen uses SignSGD:

$$
SDR_q
=
\frac{1}{D}
\sum_{k=1}^{D}
\mathbf 1
\left[
\operatorname{sign}(g_{q,k}^{all})
\neq
\operatorname{sign}(g_{q,k}^{ID})
\right].
$$

These metrics answer a concrete question:

> **Does semantic OOD change the adaptation direction that Ramen would have taken using only known support gradients?**

---

## 12. Interpretation matrix

The oracle analysis should be interpreted as follows:

| Observation | Interpretation |
|---|---|
| `OracleID > Ramen` | OOD gradients are harmful in aggregate |
| `OracleDropOOD < OracleIDGradient` | OOD/domain evidence may still be useful even if OOD gradients are not |
| `GDC` increases with OOD ratio | semantic contamination changes adaptation direction |
| `GDC` remains near zero | gradient contamination is not the dominant problem |
| entropy purity improves but Oracle gain is absent | prediction correctness is not the correct optimization target |

The thesis remains about reliable gradient memory, but the exact final mechanism must be justified by this oracle structure rather than by confidence alone.

---

# Part III — Gradient Consensus

## 13. Core method: Consensus-Aware Gradient Memory

The proposed deployable method is **ConsensusRamen**.

It preserves Ramen's existing strengths:

- CLIP feature retrieval;
- prediction balance across class memories;
- per-sample cached gradients;
- entropy weighting;
- feature-distance weighting;
- temporary sample-specific adaptation;
- parameter reset after inference.

It changes only how retrieved gradients are combined.

The main idea is:

$$
\boxed{
\text{trust gradient directions corroborated across support groups}
}
$$

rather than:

$$
\boxed{
\text{trust samples with low predictive entropy}
}
$$

---

## 14. Step 1 — Retrieve supports exactly as Ramen

For query feature $z_q$, Ramen retrieves top-$k$ support samples from each active predicted-class cache.

Keep the standard weighting:

$$
\alpha_{qj}
=
\exp(-H_j)
\exp(-\beta d(z_q,z_j)).
$$

For class $c$, let the retrieved support subset be:

$$
S_{q,c}.
$$

---

## 15. Step 2 — Build one local gradient per support class

Instead of immediately summing all retrieved gradients, first build the same
weighted per-class contribution that Ramen uses:

$$
h_{q,c}
=
\sum_{j\in S_{q,c}}\alpha_{qj}g_j
$$

This produces a set:

$$
\mathcal H_q
=
\{h_{q,1},h_{q,2},\ldots,h_{q,C_q}\}
$$

where $C_q$ is the number of active retrieved classes.

The final average across classes is class-balanced.  The per-class contribution
is intentionally **not** normalized by its total weight: renormalizing it
would change Ramen's update magnitude, making ConsensusRamen a different
baseline even when its mask is bypassed.

---

## 16. Step 3 — Coordinate-wise sign consensus

Ramen uses SignSGD, therefore final update magnitude is less important than coordinate-wise direction.

For gradient coordinate $k$, compute:

$$
v_{q,k}
=
\frac{1}{C_q}
\sum_{c=1}^{C_q}
\operatorname{sign}(h_{q,c,k}).
$$

Define consensus strength:

$$
q_{q,k}=|v_{q,k}|.
$$

Thus:

$$
0\le q_{q,k}\le1.
$$

Interpretation:

```text
q ~= 1   -> most support classes agree on the update direction
q ~= 0   -> strong conflict / no stable direction
```

The consensus direction is:

$$
s_{q,k}
=
\operatorname{sign}(v_{q,k}).
$$

---

## 17. Step 4 — Construct the safe gradient

### Hard-mask version: ConsensusRamen-v0

Use threshold $\tau$:

$$
m_{q,k}
=
\mathbf 1[q_{q,k}\ge\tau].
$$

Let ordinary Ramen's aggregated gradient be:

$$
g_q^{Ramen}
=
\frac1{C_q}
\sum_c h_{q,c}.
$$

Then:

$$
\boxed{
g_q^{safe}
=
m_q\odot g_q^{Ramen}
}
$$

Coordinates without sufficient consensus do not update the model.

### Soft version: ConsensusRamen-v1

Instead of a hard threshold:

$$
\boxed{
g_q^{safe}
=
q_q^{\gamma}\odot g_q^{Ramen}
}
$$

where $\gamma\ge0$ controls consensus sharpness.

Because SignSGD eventually takes a sign, soft scaling should be evaluated carefully. The primary mechanism should therefore start with direction masking or coordinate selection, where consensus can actually change which coordinates survive into the SignSGD step.

---

## 18. Optional sample-level gradient compatibility

A secondary diagnostic can score an individual cached gradient against the local consensus:

$$
r_{qj}
=
\frac1D
\sum_{k=1}^{D}
\mathbf 1
\left[
\operatorname{sign}(g_{j,k})
=
s_{q,k}
\right].
$$

This is **not** the first deployable mechanism. It is an analysis signal that can later support:

- reliability-aware retrieval;
- memory eviction;
- delayed admission;
- sample reweighting.

The first thesis method should remain minimal: compute consensus after standard Ramen retrieval and modify the aggregated update direction.

---

## 19. Why consensus is preferable to entropy gating

Entropy asks:

> **Is the model confident about this sample?**

Gradient consensus asks:

> **Is this adaptation direction supported by other locally retrieved evidence?**

These are different quantities.

The current negative entropy-gating result suggests that pseudo-label confidence is not sufficient to determine adaptation usefulness.

Consensus directly operates on the object that ultimately changes the model:

$$
\boxed{g}
$$

and is especially natural in Ramen because sample-wise gradients are already computed and cached.

No additional model, forward pass, or backward pass is required for the initial consensus mechanism.

---

# Part IV — Implementation Contract

## 20. Files to add or modify

Do not modify the original `src/methods/Ramen.py` behavior.

Recommended additions:

```text
src/
├── datasets/
│   └── open_set.py
├── evaluation/
│   └── open_set_metrics.py
├── methods/
│   ├── ConsensusRamen.py
│   ├── OracleIDGradientRamen.py
│   └── OracleConsensusRamen.py
└── memory/
    └── existing structured-memory code reused where possible
```

Expected config additions:

```text
cfg/CIFAR100C/ConsensusRamen.yaml
cfg/CIFAR100C/OracleIDGradientRamen.yaml
cfg/research/open-set-cifar100-split-v1.json
```

---

## 21. ConsensusRamen pseudocode

```text
input: current batch x

1. z       <- CLIP image features
2. logits  <- classify(z) over KNOWN classes only
3. c_hat   <- argmax(logits)
4. H       <- sample entropy
5. g       <- per-sample gradients

6. insert current (z, g, H, c_hat) into standard Ramen memory

7. for each query q:
       for each active predicted-class cache c:
           retrieve top-k nearest supports
           compute Ramen weights alpha
           aggregate class gradient h[q,c]

8. for each gradient coordinate k:
       vote[q,k]      <- mean_c sign(h[q,c,k])
       consensus[q,k] <- abs(vote[q,k])

9. mask[q,k] <- consensus[q,k] >= tau

10. g_ramen[q] <- mean_c h[q,c]
11. g_safe[q]  <- mask[q] * g_ramen[q]

12. set_by_sample_grad(g_safe)
13. SignSGD temporary update
14. inference
15. reset model parameters

16. memory persists
```

This is the primary method contract from which implementation plans should be generated.

---

## 22. Important edge cases

### Too few active support classes

Consensus is not meaningful with one active class.

Define a minimum:

$$
C_q\ge C_{min}.
$$

If fewer than `C_min` classes are available, fallback to ordinary Ramen for that query.

Initial recommendation:

```text
min_consensus_classes: 3
```

### Zero gradient coordinates

`sign(0)=0` should remain neutral and must not count as agreement with either positive or negative votes.

### Empty memory

Fallback to current Ramen empty-support behavior. Do not invent an additional adaptation path.

### Current-sample self-retrieval

Preserve Ramen behavior in the primary method for comparability, but keep an explicit ablation excluding current-sample self-retrieval.

---

# Part V — Evidence and Diagnostics

## 23. Trace fields to add

Evaluator/method diagnostics should expose:

```text
is_ood                          # evaluator-only
open_set_split_version
ood_ratio
retrieved_ood_fraction          # evaluator-only
retrieved_ood_weight_fraction   # evaluator-only
consensus_mean
consensus_p10
consensus_p50
consensus_mask_rate
active_consensus_classes
ramen_vs_oracle_id_cosine       # oracle/evaluator-only
ramen_vs_oracle_id_sign_disagreement
```

The method may use only model-derived quantities. Any field marked evaluator-only must never feed back into retrieval or adaptation.

---

## 24. Core thesis metrics

### Predictive metrics

$$
ACC_{ID},
\quad
AUROC,
\quad
FPR95,
\quad
OSCR/H\text{-score}.
$$

### Gradient-memory metrics

$$
CCR,
\quad
RCR,
\quad
GDC,
\quad
SDR.
$$

### Stability metrics

- negative-adaptation windows;
- worst-domain ID accuracy;
- recovery after domain shifts where applicable.

### Cost metrics

- synchronized forward latency;
- retained memory bytes;
- additional consensus computation time;
- throughput.

ConsensusRamen should not require an additional model forward or backward pass.

---

# Part VI — Required Baselines and Ablations

## 25. Baselines

Primary comparison:

```text
NoAdapt
Ramen
EntropyGatedLatentRamen / entropy-gated Ramen negative ablation
OracleDropOODRamen
OracleIDGradientRamen
ConsensusRamen
OracleConsensusRamen
```

If `LatentRamen` is retained in tables, it should be clearly secondary because latent routing is not the active method hypothesis in this thesis stage.

---

## 26. Consensus ablations

Required ablations:

```text
A. ordinary Ramen aggregation
B. hard coordinate consensus mask
C. soft consensus weighting
D. gradient agreement without open-set OOD
E. exclude current-sample self-retrieval
F. varying minimum active classes
G. varying consensus threshold tau
```

The main method should use a preregistered/default threshold selected without tuning on final benchmark streams.

---

# Part VII — Research Questions

## 27. RQ1 — Semantic OOD contamination

> **Does semantic OOD contamination measurably change the adaptation direction produced by Ramen's persistent gradient memory?**

Primary evidence:

$$
GDC,
\quad
SDR,
\quad
\text{Ramen vs OracleIDGradientRamen}.
$$

---

## 28. RQ2 — Confidence versus adaptation usefulness

> **Is prediction confidence sufficient to identify useful cached adaptation gradients?**

The existing entropy-gating result is already negative preliminary evidence.

The thesis should retain this as motivation for moving from prediction-level reliability to gradient-level compatibility.

---

## 29. RQ3 — Consensus-aware adaptation

> **Can gradient consensus suppress harmful semantic contamination while preserving the domain-adaptation benefit of mixed-domain memory?**

Primary comparison:

$$
\text{ConsensusRamen}
\quad vs \quad
\text{Ramen}
\quad vs \quad
\text{hard OOD filtering}.
$$

The desired outcome is not merely cleaner memory. It is better risk-adjusted adaptation under open-set mixed-domain streams.

---

# Part VIII — Implementation Roadmap

## 30. Phase A — Open-set infrastructure

Deliverables:

```text
OpenSetCIFAR100C wrapper
fixed split v1
OOD-ratio controlled streams
ID/OOD evaluator
open-set trace schema
```

Exit condition:

- known-only setting reproduces the corresponding closed-set baseline closely;
- hidden unknown samples never enter the model class vocabulary;
- stream manifests bind the open-set split and OOD ratio;
- ID/OOD metrics are reproducible.

---

## 31. Phase B — Oracle contamination analysis

Deliverables:

```text
OracleDropOODRamen
OracleIDGradientRamen
GDC metric
SDR metric
retrieved OOD contribution diagnostics
```

Purpose:

> characterize how semantic OOD affects the actual update direction.

This phase establishes the empirical structure that the final method is designed to approximate without labels.

---

## 32. Phase C — ConsensusRamen-v0

Implement:

$$
\boxed{
\text{class-level gradient aggregation}
+
\text{coordinate sign consensus}
+
\text{hard mask}
}
$$

Keep everything else identical to Ramen.

Initial config surface:

```yaml
topk: <Ramen value>
beta: <Ramen value>
optimizer: signsgd
lr: <Ramen value>
consensus_threshold: 0.2
min_consensus_classes: 3
consensus_mode: hard_mask
```

`0.2` was locked after the isolated, explicitly noncanonical MPS development
pilot recorded in `plans/20260824-latent-ramen-evidence/reports/open-set-mps-directional-pilot-20260825.md`.
It must not be retuned on canonical final streams; any alternative threshold
belongs to a separately reported ablation.

---

## 33. Phase D — ConsensusRamen-v1 and ablations

Only after v0 mechanics and evidence are stable, evaluate:

- soft consensus weighting;
- sample-level consensus reliability;
- support reweighting;
- memory admission/eviction based on repeated gradient incompatibility.

These are extensions, not required for the first complete thesis method.

---

## 34. Phase E — Final evaluation

Run at minimum:

```text
OOD ratios: 0%, 10%, 30%, 50%
streams: iid_mixed, block, recurring
seeds: >= 3
primary dataset: CIFAR-100-C open-set split
secondary dataset: DomainNet or another suitable natural-domain benchmark
```

For the primary CIFAR-100-C ratio sweep, keep the selected source exposure
fixed at 400 examples per corruption domain (6,000 examples before an
explicit cost-limited prefix). This count is divisible by every preregistered
ratio denominator. It prevents an OOD-ratio comparison from silently changing
the number of adaptation updates, cache occupancy, or stream duration.

Report both:

$$
\text{adaptation utility}
$$

and:

$$
\text{OOD / gradient-memory safety}.
$$

---

# Part IX — What Is No Longer the Main Thesis Method

## 35. Entropy gate

The entropy gate remains an important negative ablation.

Do not continue treating:

$$
\text{low entropy}
$$

as the primary definition of memory reliability.

Its scientific role is now:

> **show that improved pseudo-label purity and smaller memory do not necessarily improve adaptation.**

---

## 36. Latent router

The current unsupervised latent router is not part of the active thesis method because bounded evidence showed context collapse and the oracle routing upper bound did not establish a useful gain.

Keep the implementation and reports, but do not combine routing + semantic OOD + gradient consensus until the gradient-memory question is resolved independently.

---

# Part X — Thesis Contribution Story

## 37. Contribution 1 — Empirical finding

> **Prediction-level memory cleanliness is not equivalent to adaptation usefulness.**

The entropy-gated experiments provide preliminary evidence that a cleaner pseudo-label memory can reduce adaptation performance.

---

## 38. Contribution 2 — Open-world gradient-memory formulation

> **Semantic OOD in a persistent gradient cache should be evaluated by its effect on adaptation direction, not only by sample classification correctness.**

This is operationalized using oracle ID-only gradients, direction cosine, and sign disagreement.

---

## 39. Contribution 3 — Consensus-aware method

> **ConsensusRamen preserves gradient directions corroborated across class-balanced local support and suppresses conflicted coordinates before SignSGD adaptation.**

The method is label-free at deployment and reuses Ramen's existing per-sample gradients without requiring another model forward/backward pass.

---

# 40. North Star

The thesis should be summarized by:

$$
\boxed{
\textbf{
Retrieve by domain relevance;
adapt by gradient compatibility.
}
}
$$

The final objective remains:

$$
\boxed{
\textbf{
Prevent harmful semantic-OOD contamination of cached adaptation gradients
while preserving useful mixed-domain adaptation.
}
}
$$

The immediate implementation target is therefore **not another confidence gate**. It is:

$$
\boxed{
\textbf{
Open-Set Ramen
\rightarrow
Oracle gradient evidence
\rightarrow
ConsensusRamen-v0
}
}
$$

---

# Appendix A — Plan-generation contract

Any implementation plan generated from this report must preserve the following dependency order:

```text
A. Open-set dataset/evaluation infrastructure
        ↓
B. Oracle gradient diagnostics
        ↓
C. ConsensusRamen-v0
        ↓
D. Consensus ablations/extensions
        ↓
E. final multi-seed evaluation
```

Do **not** generate a plan that starts ConsensusRamen before the open-set evaluator and oracle diagnostics exist.

Each plan should specify:

```text
objective
files touched
new interfaces / config keys
data-flow changes
method equations implemented
evaluator-only vs method-visible fields
tests
smoke command
expected artifacts
exit criteria
```

The implementation invariants are:

1. `src/methods/Ramen.py` remains behaviorally unchanged.
2. Ground-truth ID/OOD information is evaluator-only except in explicitly named `Oracle*` methods.
3. `ConsensusRamen` uses no target labels.
4. `ConsensusRamen-v0` requires no additional model forward/backward pass beyond Ramen.
5. Consensus is computed after standard Ramen support retrieval, not as an OOD detector before retrieval.
6. The primary v0 mechanism modifies the update direction through coordinate masking before SignSGD.
7. If active support classes are fewer than `min_consensus_classes`, the method falls back to ordinary Ramen for that query.
8. Entropy gating remains a preserved negative ablation and is not silently retuned.
9. Latent routing is out of scope for the active method path.
10. Every final comparison must use identical versioned stream fingerprints within a paired experimental cell.

This report is the source document for subsequent implementation-plan files.
