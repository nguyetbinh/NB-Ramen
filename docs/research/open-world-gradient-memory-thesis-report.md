# Open-World Gradient Memory for Ramen

## Thesis Direction and Implementation Specification

**Base paper:** *Ramen: Robust Test-Time Adaptation of Vision-Language Models with Active Sample Selection*  
**Repository:** `nguyetbinh/NB-Ramen`  
**Working branch:** `open-world-gradient-memory`  
**Research direction:** Open-World / Open-Set Mixed-Domain Test-Time Adaptation  
**Status:** active thesis implementation; mechanism is locally de-risked but canonical CUDA evidence is still pending  

---

# 1. Thesis objective

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

The thesis should not define memory quality only through prediction confidence, entropy, or pseudo-label correctness. The current evidence already shows that a cleaner pseudo-label memory can be a worse adaptation memory.

The target is therefore not a generic OOD filter. It is a **gradient-memory mechanism that preserves adaptation-compatible directions and suppresses directions unsupported by the local support memory**.

Working title:

> **Consensus-Aware Gradient Memory for Open-World Mixed-Domain Test-Time Adaptation**

---

# 2. Current scientific status

The active branch has already implemented the main infrastructure required by the thesis:

```text
Open-set dataset wrapper
+ fixed known/unknown split
+ deterministic OOD-ratio-controlled streams
+ evaluator-only ID/OOD metadata
+ open-set metrics
+ OracleDropOODRamen
+ OracleIDGradientRamen
+ ConsensusRamen-v0
+ OracleConsensusRamen
+ consensus ablations
+ canonical CIFAR-100-C matrix planner
+ DomainNet secondary matrix planner
```

The implementation direction is correct. The next work should **not add another large adaptation architecture**. The immediate goal is to improve mechanism validation and evaluation cleanliness before the canonical CUDA run.

---

# 3. What Ramen contributes and where the thesis intervenes

Ramen processes a test sample as:

```text
image
  -> CLIP feature z
  -> known-class logits
  -> pseudo-class c_hat
  -> entropy loss
  -> per-sample gradient g
  -> cache (z, g, entropy) under predicted class
  -> retrieve top-k nearby support from every active class cache
  -> entropy + feature-distance weighting
  -> class-balanced gradient aggregation
  -> temporary SignSGD adaptation
  -> inference
  -> reset model parameters
```

The important architectural fact is:

$$
\boxed{
\text{parameter adaptation is temporary, but gradient memory persists}
}
$$

Therefore a bad support gradient can influence multiple future queries even though adapted parameters are reset after every prediction.

The thesis intervenes at the aggregation stage:

```text
Ramen support retrieval
        ↓
per-class gradient contributions
        ↓
coordinate-wise directional agreement
        ↓
keep supported coordinates
suppress conflicted coordinates
        ↓
SignSGD
```

---

# 4. Negative result that changed the thesis direction

The first reliability hypothesis was:

$$
\text{low predictive entropy}
\Rightarrow
\text{trustworthy memory item}.
$$

The implemented entropy gate reduced pseudo-label contamination and memory size, but adaptation accuracy decreased.

The key conclusion is:

$$
\boxed{
\text{cleaner pseudo-label memory}
\not\Rightarrow
\text{better adaptation memory}
}
$$

and more specifically:

$$
\boxed{
\text{prediction confidence}
\neq
\text{gradient usefulness}
}
$$

Therefore entropy gating remains a negative ablation, but it is no longer the thesis mechanism.

---

# 5. Open-set benchmark definition

The primary benchmark is CIFAR-100-C with a fixed semantic split:

$$
100
=
80\text{ known}
+
20\text{ unknown}.
$$

The model receives prompts only for the known classes:

$$
\mathcal Y_K.
$$

The deployment stream still contains:

$$
\mathcal Y_K\cup\mathcal Y_U.
$$

Unknown samples therefore cannot be predicted as an explicit unknown class by the classifier. They are forced into the known vocabulary, which creates the intended semantic contamination setting.

Required OOD ratios:

$$
\rho_{OOD}\in\{0,0.1,0.3,0.5\}.
$$

Required stream modes:

```text
iid_mixed
block
recurring
```

The primary CIFAR-100-C matrix uses fixed source exposure:

```text
400 selected source examples per corruption domain
6000 source examples before scheduling across 15 corruptions
```

The source count must remain fixed across OOD ratios so that changing the ratio does not silently change stream length, cache opportunity, or number of adaptation steps.

---

# 6. Evaluator-only metadata invariant

Open-set samples may carry evaluator metadata:

```text
original_label
known_label_or_minus_one
is_ood
open_set_split_version
ood_ratio
```

These fields are never inputs to deployable methods.

Only explicitly named `Oracle*` methods may consume `is_ood` through the evaluator hook.

The invariant is:

$$
\boxed{
\text{ConsensusRamen never receives target ID/OOD labels}
}
$$

---

# 7. Oracle gradient analysis

The oracle analysis answers whether semantic OOD actually changes the update Ramen would perform.

For query $q$, Ramen forms:

$$
g_q^{all}
=
\frac{1}{C_q}
\sum_c
h_{q,c}^{all},
$$

where:

$$
h_{q,c}^{all}
=
\sum_{j\in S_{q,c}}
\alpha_{qj}g_j.
$$

Ramen support weight is:

$$
\alpha_{qj}
=
\exp(-H_j)
\exp(-\beta d(z_q,z_j)).
$$

Using evaluator-only labels, construct an ID-only reference:

$$
h_{q,c}^{ID}
=
\sum_{j\in S_{q,c},\;j\in ID}
\alpha_{qj}g_j,
$$

and:

$$
g_q^{ID}
=
\frac{1}{C_q}
\sum_c h_{q,c}^{ID}.
$$

The divisor remains Ramen's active-class count so the oracle changes gradient contribution, not retrieval/class-balancing semantics.

Primary contamination diagnostics are:

$$
GDC_q
=
1-\cos(g_q^{all},g_q^{ID})
$$

and, because the optimizer is SignSGD:

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

Also record:

```text
retrieved_ood_fraction
retrieved_ood_weight_fraction
```

---

# 8. Existing noncanonical mechanism signal

The current MPS pilot is **not canonical benchmark evidence**, but it is sufficient to justify continuing the method.

Observed directional signal in the three block seeds at OOD ratio 0.5:

```text
GDC:              approximately 0.14 - 0.20
sign disagreement approximately 0.17 - 0.21
OOD weight share: approximately 0.38 - 0.43
```

`OracleIDGradientRamen` was non-worse than Ramen in all three block seeds and improved mean ID accuracy by approximately $+1.06$ percentage points.

The known-only control at OOD ratio $0$ produced effectively zero directional discrepancy and matching Ramen/OracleID accuracy.

This supports the mechanism chain:

$$
\boxed{
\text{semantic OOD support}
\rightarrow
\text{gradient direction change}
\rightarrow
\text{oracle removal can help}
}
$$

but does not yet establish canonical effect size or generalization.

---

# 9. ConsensusRamen-v0: deployable method

ConsensusRamen preserves Ramen's:

- cache admission;
- predicted-class memory partition;
- feature retrieval;
- entropy weighting;
- distance weighting;
- class balancing;
- temporary update;
- SignSGD optimizer;
- parameter reset.

It changes only the aggregate gradient support.

## 9.1 Per-class gradient contribution

For every active predicted-class cache $c$:

$$
h_{q,c}
=
\sum_{j\in S_{q,c}}
\alpha_{qj}g_j.
$$

Do **not** normalize this contribution inside each class. Ordinary Ramen also sums the weighted support gradient inside a class before averaging across active classes.

Ordinary Ramen gradient is:

$$
g_q^{Ramen}
=
\frac{1}{C_q}
\sum_{c=1}^{C_q}h_{q,c}.
$$

## 9.2 Coordinate-wise agreement

For coordinate $k$:

$$
v_{q,k}
=
\frac{1}{C_q}
\sum_{c=1}^{C_q}
\operatorname{sign}(h_{q,c,k}).
$$

Define agreement:

$$
q_{q,k}=|v_{q,k}|.
$$

Interpretation:

```text
q ~= 1  -> class support gradients mostly agree on the sign
q ~= 0  -> strong conflict or no stable direction
```

Zero coordinates remain neutral because:

$$
\operatorname{sign}(0)=0.
$$

## 9.3 Hard-mask v0

Primary method uses:

$$
m_{q,k}
=
\mathbf 1[q_{q,k}\ge\tau].
$$

Then:

$$
\boxed{
g_q^{safe}
=
m_q\odot g_q^{Ramen}
}
$$

The locked primary configuration is:

```yaml
consensus_threshold: 0.2
min_consensus_classes: 3
consensus_mode: hard_mask
include_current: true
```

The threshold $0.2$ was selected only in the explicitly noncanonical development pilot and must not be retuned on the final canonical streams.

## 9.4 Fallback

If:

$$
C_q<C_{min},
$$

ConsensusRamen must return ordinary Ramen's gradient for that query.

No new fallback adaptation algorithm is introduced.

---

# 10. Correct SignSGD-compatible soft consensus

The old conceptual formulation:

$$
q_q^\gamma\odot g_q^{Ramen}
$$

is not a valid graded mechanism under SignSGD because for positive $q$:

$$
\operatorname{sign}(q^\gamma g)
=
\operatorname{sign}(g).
$$

Therefore the actual `ConsensusRamenSoft` ablation uses **coordinate admission probability**.

For each coordinate:

$$
p_{q,k}=q_{q,k}^{\gamma}.
$$

Draw:

$$
b_{q,k}\sim\operatorname{Bernoulli}(p_{q,k}).
$$

Then:

$$
\boxed{
g_q^{soft}
=
b_q\odot g_q^{Ramen}}
$$

with a deterministic configured seed for reproducibility.

This mechanism changes which coordinates survive into SignSGD while preserving Ramen's sign on admitted coordinates.

The soft variant remains an ablation, not the primary thesis method.

---

# 11. Current pilot interpretation

At OOD ratio $0.5$, the noncanonical three-stream pilot gave approximately:

| Stream | ConsensusRamen - Ramen ID ACC |
|---|---:|
| `iid_mixed` | $-0.52$ pp |
| `block` | $+2.66$ pp |
| `recurring` | $+0.54$ pp |
| mean over 9 cells | $+0.89$ pp |

This should not be interpreted as universal superiority.

The useful working hypothesis is narrower:

$$
\boxed{
\text{gradient consensus helps when a coherent local adaptation direction exists}
}
$$

Block streams are particularly relevant because consecutive samples share a persistent corruption/domain context. IID mixtures may not provide a single locally coherent direction, so consensus can become over-regularization.

The final thesis must report this heterogeneity rather than collapsing all streams into one favorable mean.

---

# 12. Mechanism-validation gap that must be implemented next

Current oracle diagnostics measure:

$$
g^{Ramen}
\quad\text{vs}\quad
g^{ID}.
$$

The thesis method produces:

$$
g^{Consensus}.
$$

We therefore need direct evidence that Consensus moves the realized update **toward the ID-only oracle direction**.

Required metrics are:

$$
GDC_q^{Ramen}
=
1-\cos(g_q^{Ramen},g_q^{ID})
$$

and:

$$
GDC_q^{Consensus}
=
1-\cos(g_q^{Consensus},g_q^{ID}).
$$

Likewise:

$$
SDR_q^{Ramen}
=
SDR(g_q^{Ramen},g_q^{ID})
$$

and:

$$
SDR_q^{Consensus}
=
SDR(g_q^{Consensus},g_q^{ID}).
$$

The key mechanism quantity is:

$$
\boxed{
\Delta GDC_q
=
GDC_q^{Ramen}
-
GDC_q^{Consensus}
}
$$

and:

$$
\boxed{
\Delta SDR_q
=
SDR_q^{Ramen}
-
SDR_q^{Consensus}
}
$$

Positive values mean Consensus is closer to the ID-only oracle update.

---

# 13. Implementation specification: Consensus-vs-OracleID diagnostics

Do not give `is_ood` to `ConsensusRamen`.

Instead, extend the **oracle analysis path** so that the same retrieved all-support gradients produce three directions:

```text
g_ramen       = ordinary all-support Ramen aggregate
g_consensus   = consensus mask applied using ONLY all-support gradients
g_oracle_id   = evaluator-label ID-only reference
```

The consensus mask must be computed without using `is_ood`.

Evaluator-only labels are used only to construct `g_oracle_id`.

Recommended implementation location:

```text
src/methods/OracleIDGradientRamen.py
```

Refactor `aggregate_oracle_supports()` to retain the per-class all-support contributions:

$$
H_q
=
[h_{q,1},\ldots,h_{q,C_q}].
$$

From $H_q$ compute:

```text
g_ramen
agreement q
g_consensus
```

and separately use OOD flags to compute:

```text
g_oracle_id
```

The behavior of `OracleIDGradientRamen` remains unchanged: it still **applies `g_oracle_id`** as its oracle upper-bound update. The additional consensus direction exists only for diagnostics.

Use the same locked primary Consensus configuration:

```text
consensus_threshold = 0.2
min_consensus_classes = 3
```

Add config keys only if required for strict config identity, for example:

```yaml
diagnostic_consensus_threshold: 0.2
diagnostic_min_consensus_classes: 3
```

Do not tune these independently from the primary v0 method.

---

# 14. New oracle trace fields

Extend the oracle evidence group with:

```text
consensus_vs_oracle_id_cosine
consensus_vs_oracle_id_sign_disagreement
consensus_vs_ramen_cosine
consensus_diagnostic_mask_rate
consensus_diagnostic_applied
```

Existing fields remain:

```text
retrieved_ood_fraction
retrieved_ood_weight_fraction
ramen_vs_oracle_id_cosine
ramen_vs_oracle_id_sign_disagreement
```

Summary output should include:

```text
ramen_gdc_mean
consensus_gdc_mean
ramen_sdr_mean
consensus_sdr_mean
gdc_reduction_mean
sdr_reduction_mean
```

The central mechanism claim is supported only if Consensus reduces oracle-direction discrepancy in the conditions where it improves ID adaptation.

---

# 15. Post-adaptation OOD safety must be measured

The current evaluator mainly uses a pre-adaptation energy score:

$$
E_{pre}(x)
=
-\log\sum_c\exp(l_c^{pre}(x)).
$$

Because methods begin each query from the same reset model, this score is primarily a property of the base model and sample. It does not measure whether the **adaptation step itself** makes OOD behavior safer or worse.

Add a post-adaptation score from returned logits:

$$
\boxed{
E_{post}(x)
=
-\log\sum_c\exp(l_c^{post}(x))
}
$$

No additional model forward is required because the adapted logits are already returned by every method.

---

# 16. Implementation specification: post-adaptation OOD score

Preferred implementation location:

```text
src/main.py
src/evaluation/evidence.py
src/evaluation/open_set_metrics.py
src/evaluation/open_set_consensus_analysis.py
```

In `ordered_stream_test()`:

```text
logits = tta_model(image)
post_adaptation_ood_score = -logsumexp(logits)
```

Add trace field:

```text
post_adaptation_ood_score
```

Keep:

```text
pre_adaptation_ood_score
```

Open-set summary should report two explicit detection blocks:

```text
pre_adaptation_detection
post_adaptation_detection
```

Each block should contain, where defined:

```text
AUROC
FPR95
H-score
OOD recall at FPR95
```

For backward compatibility, existing top-level detection fields may remain mapped to pre-adaptation detection temporarily, but final thesis tables must label pre/post explicitly.

Interpretation:

```text
pre-adaptation detection  -> base model OOD separability
post-adaptation detection -> effect of the TTA action on OOD safety
```

The second one is the relevant safety result for this thesis.

---

# 17. Clean entropy baseline is required

`EntropyGatedLatentRamen` is not a clean control for the thesis because it changes both:

```text
memory admission
+
latent-context routing/memory structure
```

To compare prediction-level reliability with gradient-level compatibility, add:

> **EntropyGatedRamen**

It must be exactly ordinary Ramen except for memory admission.

---

# 18. EntropyGatedRamen method contract

For each sample compute normalized entropy:

$$
\bar H_i
=
\frac{H_i}{\log K}.
$$

Use the preserved threshold:

$$
\tau_H=0.5.
$$

Admission:

$$
a_i
=
\mathbf 1[\bar H_i\le0.5].
$$

Behavior:

```text
compute features/logits/gradient exactly as Ramen
        ↓
compute entropy gate
        ↓
if admitted:
    add current sample to predicted-class Ramen cache
else:
    do not add current sample
        ↓
retrieve from ordinary Ramen class caches
        ↓
use ordinary Ramen aggregation
        ↓
SignSGD temporary update
```

Preserve Ramen batch-atomic semantics for admitted samples: admitted current samples are inserted before retrieval and may self-retrieve. Rejected current samples do not enter memory and therefore cannot self-retrieve.

If no support cache is available, use a zero gradient / no adaptation step rather than inventing another fallback.

The method must not contain a latent router.

---

# 19. Files for EntropyGatedRamen

Add:

```text
src/methods/EntropyGatedRamen.py
cfg/CIFAR100C/EntropyGatedRamen.yaml
cfg/DomainNet/EntropyGatedRamen.yaml
```

Register in:

```text
src/methods/__init__.py
```

Required diagnostics:

```text
admission_prediction
admission_normalized_entropy
admitted_to_memory
memory_bytes
pre_adaptation_ood_score
```

Add tests for:

```text
all admitted -> behavior matches Ramen
all rejected with empty history -> zero/no update
mixed admitted/rejected batch
admitted current sample can self-retrieve
rejected current sample cannot self-retrieve
reset behavior
open-set label isolation
```

---

# 20. Canonical baseline matrix correction

Before running the canonical CIFAR-100-C matrix, replace:

```text
EntropyGatedLatentRamen
```

with:

```text
EntropyGatedRamen
```

in the primary seven-method matrix.

The primary matrix should therefore be:

```text
NoAdapt
Ramen
EntropyGatedRamen
OracleDropOODRamen
OracleIDGradientRamen
ConsensusRamen
OracleConsensusRamen
```

This preserves the current size:

$$
7\text{ methods}
\times
4\text{ OOD ratios}
\times
3\text{ streams}
\times
3\text{ seeds}
=
252\text{ runs}.
$$

`EntropyGatedLatentRamen` remains in the repository and historical reports but is no longer the clean primary entropy baseline.

Apply the same replacement to the DomainNet secondary matrix.

---

# 21. Split robustness

The current primary CIFAR-100 split is:

```text
known:   class IDs 0..79
unknown: class IDs 80..99
```

This is deterministic and suitable as the primary `v1` protocol, but a final scientific claim should not depend entirely on this one contiguous taxonomy slice.

Keep `open-set-cifar100-split-v1` as the primary canonical split.

After the primary matrix, add a smaller split-robustness study with at least two additional deterministic 80/20 splits.

Recommended construction:

```text
rank CIFAR-100 class names by SHA256(salt || class_name)
select first 80 known, remaining 20 unknown
```

Use fixed versioned salts, for example:

```text
open-set-cifar100-name-rank-v2
open-set-cifar100-name-rank-v3
```

Do not choose splits based on final accuracy.

A reduced robustness matrix is sufficient, for example:

```text
OOD ratios: 0.3, 0.5
streams: block, recurring
seeds: 0, 1, 2
methods: Ramen, ConsensusRamen, OracleIDGradientRamen
```

The goal is not another hyperparameter search. The goal is to test whether the mechanism depends on one particular unknown-class subset.

---

# 22. OOD-ratio mechanism test

The most important final trend is not only whether ConsensusRamen beats Ramen.

Measure the relationship:

$$
\rho_{OOD}
\rightarrow
GDC/SDR
\rightarrow
\text{Oracle gap}
\rightarrow
\text{Consensus gain}.
$$

The desired mechanistic pattern is:

$$
\boxed{
\rho_{OOD}\uparrow
\Rightarrow
\text{gradient corruption}\uparrow
\Rightarrow
\text{room for safe aggregation}\uparrow
}
$$

But this is a hypothesis, not an assumption. Report the actual trend.

The OOD=$0$ control is especially important.

Define:

$$
\Delta_0
=
ACC_{ID}(Consensus,0)
-
ACC_{ID}(Ramen,0)
$$

and:

$$
\Delta_{50}
=
ACC_{ID}(Consensus,0.5)
-
ACC_{ID}(Ramen,0.5).
$$

If:

$$
\Delta_0\approx0
$$

and:

$$
\Delta_{50}>0,
$$

then the semantic-contamination story is directly supported.

If Consensus also helps strongly at OOD=$0$, the more accurate interpretation is broader **gradient regularization**, and the thesis claim must be adjusted accordingly.

---

# 23. Stream-structure hypothesis

The current pilot suggests:

```text
block      -> strongest positive signal
recurring  -> small positive signal
iid_mixed  -> null/slightly negative signal
```

Therefore report Consensus performance separately by stream.

A plausible mechanism is:

$$
\boxed{
\text{local domain coherence}
\Rightarrow
\text{shared adaptation direction}
\Rightarrow
\text{consensus becomes informative}
}
$$

This interpretation must be evaluated rather than assumed.

Required analysis:

```text
Consensus gain by stream
mean consensus agreement by stream
retained-coordinate rate by stream
GDC/SDR by stream
Oracle gap by stream
```

If block/recurring consistently show stronger oracle corruption and stronger Consensus recovery than IID, this becomes a strong thesis finding and a possible paper-level extension.

---

# 24. Open-set metrics

## 24.1 Primary utility

$$
ACC_{ID}
$$

and:

```text
worst-domain ID accuracy
```

## 24.2 OOD safety

Report both pre- and post-adaptation:

```text
AUROC
FPR95
H-score
```

## 24.3 Gradient-memory diagnostics

```text
retrieved OOD fraction
retrieved OOD weight fraction
Ramen GDC
Ramen SDR
Consensus GDC
Consensus SDR
GDC reduction
SDR reduction
consensus agreement mean/p10/p50
consensus retained-coordinate rate
active consensus class count
```

## 24.4 Stability

Generic accuracy over all samples is misleading in open-set streams because OOD samples have no valid known-class classification target.

For final thesis reporting, stability metrics must be ID-aware.

At minimum expose:

```text
ID-only negative-adaptation windows
ID-only worst-domain accuracy
ID-only recovery inside persistent-domain episodes
```

Do not interpret a generic correctness sequence where every OOD row is counted as classification failure as an adaptation-stability metric.

## 24.5 Cost

Report:

```text
synchronized forward latency
throughput
max retained method-memory bytes
final retained method-memory bytes
Consensus-vs-Ramen paired overhead
```

ConsensusRamen-v0 should require no extra model forward or backward pass.

---

# 25. ID-only stability implementation

This is lower priority than the gradient mechanism diagnostics, but it must be fixed before final reporting.

Recommended implementation:

```text
src/evaluation/open_set_consensus_analysis.py
src/evaluation/evidence.py or a dedicated open-set stability helper
```

For paired negative-adaptation analysis:

1. verify identical stream fingerprint;
2. select only timesteps where `is_ood == false`;
3. compare adapted method and NoAdapt on exactly those ID timesteps;
4. construct windows over the resulting ordered ID sequence;
5. report negative ID windows.

For persistent-domain recovery:

1. preserve original domain episode boundaries;
2. evaluate only ID rows inside each episode;
3. require a minimum number of ID samples before declaring a recovery statistic defined.

Do not silently reuse closed-set recovery semantics.

---

# 26. Required method comparison

Primary seven-method matrix:

| Method | Purpose |
|---|---|
| `NoAdapt` | base CLIP reference |
| `Ramen` | original mixed-domain gradient-memory baseline |
| `EntropyGatedRamen` | prediction-confidence memory filtering control |
| `OracleDropOODRamen` | upper bound removing OOD memory entirely |
| `OracleIDGradientRamen` | upper bound removing OOD gradient contribution while retaining ordinary support structure |
| `ConsensusRamen` | deployable gradient-compatibility method |
| `OracleConsensusRamen` | ID-only consensus upper bound |

Historical/secondary methods such as `LatentRamen` and `EntropyGatedLatentRamen` should not define the active thesis comparison.

---

# 27. Consensus ablations

Keep the current explicit identities:

```text
ConsensusRamenSoft
ConsensusRamenNoSelf
ConsensusRamenTau060
ConsensusRamenMin2
ConsensusRamenMin4
```

The primary v0 configuration remains immutable.

Ablations answer:

```text
hard mask vs stochastic soft admission
current-sample self-support vs history-only support
tau sensitivity
minimum-active-class sensitivity
```

Do not place all ablations into the primary 252-run matrix.

Use the separate ablation planner on selected held-out cells.

---

# 28. Next implementation cycle

The next implementation work should be done in this order.

## Phase F1 — Mechanism diagnostics

### Objective

Show whether deployable Consensus actually moves Ramen's gradient closer to the ID-only oracle direction.

### Files

```text
src/methods/OracleIDGradientRamen.py
src/evaluation/evidence.py
src/main.py
src/evaluation/open_set_consensus_analysis.py
tests/test_oracle_id_gradient_ramen.py
tests/test_open_set_consensus_analysis.py
```

### Deliverables

```text
consensus_vs_oracle_id_cosine
consensus_vs_oracle_id_sign_disagreement
gdc_reduction
sdr_reduction
```

### Exit criteria

- ordinary OracleID applied update is unchanged;
- consensus diagnostic never uses `is_ood` to construct its mask;
- when OOD=0, Ramen and ID oracle discrepancies remain approximately zero;
- synthetic unit tests prove a consensus mask can reduce SDR to a known oracle vector;
- trace and summary validation reject incomplete diagnostic groups.

---

## Phase F2 — Post-adaptation OOD safety

### Objective

Measure whether adaptation changes OOD separability.

### Files

```text
src/main.py
src/evaluation/evidence.py
src/evaluation/open_set_metrics.py
src/evaluation/open_set_consensus_analysis.py
tests/test_open_set_metrics.py
tests/test_evidence.py
```

### Deliverables

```text
pre_adaptation_ood_score
post_adaptation_ood_score
pre_adaptation_detection
post_adaptation_detection
```

### Exit criteria

- no extra forward pass;
- `NoAdapt` pre/post scores match within numerical tolerance;
- post metrics are method-dependent when adapted logits differ;
- OOD=0 marks detection metrics unavailable rather than fabricating values.

---

## Phase F3 — Clean EntropyGatedRamen

### Objective

Create an apples-to-apples sample-confidence baseline against ConsensusRamen.

### Files

```text
src/methods/EntropyGatedRamen.py
src/methods/__init__.py
cfg/CIFAR100C/EntropyGatedRamen.yaml
cfg/DomainNet/EntropyGatedRamen.yaml
src/runtime/experiment_matrix.py
src/runtime/open_set_domainnet_matrix.py
tests/test_entropy_gated_ramen.py
tests/test_experiment_matrix.py
tests/test_open_set_domainnet_matrix.py
```

### Locked configuration

```yaml
max_normalized_entropy: 0.50
```

### Exit criteria

- no latent router imported or instantiated;
- all-admit fixture reproduces ordinary Ramen aggregation;
- rejected samples are absent from memory;
- primary open-set matrix remains exactly seven methods / 252 runs;
- `EntropyGatedLatentRamen` is removed from the primary matrix but retained as historical code.

---

## Phase F4 — Report/code synchronization

### Objective

Make this document, configs, method docs, and matrix constants describe the same algorithm.

### Required corrections

```text
working branch = open-world-gradient-memory
soft consensus = Bernoulli coordinate admission, not positive magnitude scaling
primary entropy baseline = EntropyGatedRamen
post-adaptation OOD metrics explicitly separated from pre-adaptation metrics
```

### Exit criteria

A new implementation plan generated from this report should not recreate the withdrawn soft-scaling mechanism or latent-router entropy baseline.

---

## Phase G — Canonical CIFAR-100-C CUDA matrix

Run only after F1-F4 are merged and frozen.

Required grid:

```text
methods: 7
OOD ratios: 0, 0.1, 0.3, 0.5
streams: iid_mixed, block, recurring
seeds: 0, 1, 2
device: CUDA
artifact: verified official CIFAR-100-C
prefix: none
source exposure: 400/domain
```

Total:

$$
252\text{ runs}.
$$

Every adapted run must use the exact paired NoAdapt stream fingerprint.

Do not change $\tau=0.2$ after seeing this matrix.

---

## Phase H — Split robustness and DomainNet

After primary CIFAR evidence:

```text
1. run reduced multi-split CIFAR robustness study
2. execute canonical DomainNet secondary matrix
3. compare mechanism trends rather than only headline accuracy
```

Questions:

```text
Does GDC increase with OOD prevalence?
Does Consensus reduce GDC/SDR?
Does ID accuracy gain track GDC reduction?
Is the effect stronger in block/recurring streams?
Does post-adaptation OOD safety improve or degrade?
Does the result survive a different semantic split / natural-domain benchmark?
```

---

# 29. Interpretation rules for final results

## Case A — strongest thesis outcome

Observed:

```text
OOD ratio ↑
GDC/SDR ↑
OracleID - Ramen ↑
Consensus GDC/SDR < Ramen GDC/SDR
Consensus ID ACC > Ramen
Consensus post-OOD safety maintained/improved
```

Interpretation:

> semantic OOD contaminates persistent gradient memory, and consensus approximates the safe ID-only update without target labels.

## Case B — Consensus helps mainly in block/recurring

Interpretation:

> gradient consensus is useful when deployment streams contain locally coherent adaptation structure; IID mixtures do not provide a stable shared direction.

This remains a valid and potentially stronger thesis story than claiming universal improvement.

## Case C — Consensus helps equally when OOD=0

Interpretation:

> the mechanism is a general gradient-conflict regularizer rather than specifically an OOD-contamination mechanism.

The thesis claim must be broadened accordingly.

## Case D — Oracle gap exists but Consensus does not close it

Interpretation:

> semantic OOD contamination is real, but class-wise sign consensus is an insufficient estimator of safe update direction.

The thesis still contains a valid empirical finding and a tested method. Future paper work can move toward learned gradient compatibility, domain-conditioned consensus, or memory repair.

---

# 30. Thesis research questions

## RQ1

> **Does semantic OOD contamination measurably alter Ramen's persistent adaptation gradient?**

Evidence:

$$
GDC,
\quad SDR,
\quad
OracleID-Ramen\text{ gap}.
$$

## RQ2

> **Is prediction confidence sufficient to identify useful memory items?**

Evidence:

```text
EntropyGatedRamen
vs
Ramen
vs
ConsensusRamen
```

## RQ3

> **Can class-wise gradient consensus approximate a safer ID-only adaptation direction without target labels?**

Evidence:

$$
GDC^{Consensus}<GDC^{Ramen}
$$

and:

$$
SDR^{Consensus}<SDR^{Ramen}.
$$

## RQ4

> **Under which stream structures is gradient consensus useful?**

Evidence:

```text
iid_mixed vs block vs recurring
```

## RQ5

> **Does safer gradient aggregation preserve OOD safety after adaptation?**

Evidence:

```text
post-adaptation AUROC/FPR95/H-score
```

---

# 31. Contribution story

## Contribution 1 — Empirical finding

> **Prediction-level memory cleanliness is not equivalent to adaptation usefulness.**

The entropy-gating experiments show that better pseudo-label purity alone can reduce adaptation performance.

## Contribution 2 — Gradient-memory contamination formulation

> **Semantic OOD contamination is characterized by its effect on the actual update direction rather than only by OOD detection or pseudo-label error.**

This is operationalized by all-support vs ID-only oracle gradients, cosine corruption, and SignSGD sign disagreement.

## Contribution 3 — ConsensusRamen

> **ConsensusRamen suppresses update coordinates not corroborated across Ramen's class-balanced local support before SignSGD.**

The deployable method requires no target labels and no additional model forward/backward pass.

## Contribution 4 — Structured evaluation

> **The method is evaluated across OOD prevalence and stream structure, separating base-model OOD separability from post-adaptation OOD safety.**

---

# 32. What is explicitly out of scope for the active thesis method

Do not combine these into the primary method before the canonical matrix:

```text
latent routing
category discovery
new-class learning
large learned risk networks
augmentation-based extra-forward reliability
full gradient compression
continual parameter accumulation
```

They remain possible future paper directions.

---

# 33. Future paper directions opened by the thesis

If ConsensusRamen establishes a meaningful signal, the natural next steps are:

### 33.1 Domain-conditioned consensus

Estimate agreement only among supports likely to share the current domain/context.

### 33.2 Memory repair

Use repeated gradient conflict to:

```text
downweight
relabel
evict
quarantine
```

persistent memory entries.

### 33.3 Useful information from unknowns

Instead of discarding OOD samples, separate:

$$
g_{OOD}
=
g_{domain}
+
g_{semantic}.
$$

Preserve domain-compatible components while suppressing semantic conflict.

### 33.4 Continual open-world gradient memory

Study:

```text
what to remember
what to forget
when to reuse old domain gradients
how recurrence changes memory utility
```

### 33.5 Learned adaptation compatibility

Replace fixed sign consensus by a learned or statistical estimator of whether a cached gradient improves the local adaptation direction.

---

# 34. North Star

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

The active implementation path is now:

$$
\boxed{
\text{Open-Set Ramen}
\rightarrow
\text{Oracle contamination evidence}
\rightarrow
\text{ConsensusRamen-v0}
\rightarrow
\text{mechanism validation}
\rightarrow
\text{canonical evaluation}
}
$$

---

# Appendix A — Plan-generation contract

This file is the source-of-truth for subsequent implementation plans.

Plans must preserve this dependency order:

```text
F1. Consensus-vs-OracleID diagnostics
        ↓
F2. post-adaptation OOD safety
        ↓
F3. clean EntropyGatedRamen baseline
        ↓
F4. docs/config/matrix synchronization
        ↓
G. canonical CIFAR-100-C CUDA matrix
        ↓
H. split robustness + DomainNet
```

Each generated plan must specify:

```text
objective
scientific question
files touched
interfaces / config keys
data-flow changes
exact equations
method-visible fields
evaluator-only fields
unit tests
integration tests
smoke command
expected trace/summary fields
expected artifacts
exit criteria
```

Implementation invariants:

1. Ordinary `Ramen` behavior remains the reference baseline.
2. `ConsensusRamen` never receives evaluator ID/OOD labels.
3. Oracle labels are confined to explicitly named `Oracle*` analysis paths.
4. The consensus mask is constructed entirely from ordinary retrieved gradients.
5. Consensus is applied before SignSGD.
6. `ConsensusRamen-v0` introduces no extra model forward/backward pass.
7. Below `min_consensus_classes`, v0 falls back to ordinary Ramen.
8. Primary $\tau=0.2$ is frozen before canonical evaluation.
9. Soft consensus uses Bernoulli coordinate admission, not positive magnitude scaling.
10. `EntropyGatedRamen` is the clean confidence-filtering baseline; latent routing is excluded from that comparison.
11. Pre- and post-adaptation OOD detection are reported separately.
12. Generic closed-set stability metrics must not be interpreted as open-set ID stability when OOD rows are counted as classification errors.
13. Every paired comparison uses identical versioned stream fingerprints.
14. Pilot/generated CIFAR artifacts are never reported as official CIFAR-100-C results.
15. Canonical effect-size claims require verified official data and CUDA execution.

---

# Appendix B — Immediate implementation checklist

- [ ] Extend oracle aggregation to compute `g_ramen`, `g_consensus`, and `g_oracle_id` from the same retrieved supports.
- [ ] Add Consensus-vs-OracleID cosine/sign diagnostics.
- [ ] Add `post_adaptation_ood_score` to trace output.
- [ ] Add pre/post OOD detection summary blocks.
- [ ] Implement `EntropyGatedRamen` without latent routing.
- [ ] Replace `EntropyGatedLatentRamen` with `EntropyGatedRamen` in both canonical planners.
- [ ] Update tests for seven-method/252-run matrix identity.
- [ ] Add ID-only stability semantics before final reporting.
- [ ] Freeze code/config/report after these changes.
- [ ] Run canonical CIFAR-100-C CUDA matrix.
- [ ] Analyze OOD-ratio and stream-structure trends.
- [ ] Run split robustness study.
- [ ] Execute DomainNet secondary benchmark.
