# QCGS Label-Free Diagnostic — Literature Reopening, Problem Map, and Revised Research Direction

**Repository:** `nguyetbinh/NB-Ramen`  
**Branch:** `qcgs-label-free-diagnostic`  
**Prepared:** 2026-10-08  
**Scope:** Consolidated research note covering the evidence already present in the branch, the two QCGS STOP results, the broader literature survey, candidate gaps, revised research questions, hypotheses, falsification logic, and the recommended next experiment.

---

## 1. Executive summary

The current evidence does **not** justify abandoning the broader research direction of query-conditioned support selection in Ramen. It does, however, justify closing the two specific label-free QCGS signals already tested:

1. the frozen **single-view entropy-sign** signal; and
2. the frozen **multi-view target-\(\theta_R\)** rescue signal.

The key empirical pattern is consistent across the branch:

- the candidate support space still contains substantial **oracle headroom**;
- a supervised first-order reference can rank useful support actions reasonably well;
- the tested label-free entropy-based signals do **not** rank supervised utility reliably;
- lowering entropy is not equivalent to improving supervised query utility;
- multi-view evidence can improve some selected actions while still failing as a stable ranking signal;
- harmful actions can have larger magnitude than helpful ones even when medians look benign;
- the benefit is heterogeneous across query states, especially between queries that Ramen already gets right and those it gets wrong.

Therefore the research bottleneck should be reframed from:

> “Can we design a better entropy-based QCGS score?”

to:

> **Within Ramen’s fixed query-conditioned candidate pool, what unlabeled evidence is sufficient to distinguish beneficial from harmful support-gradient substitutions, and when can that evidence be converted into a reliable deployable ranking?**

This leads to a different scientific order:

1. **Observability / identifiability:** does an unlabeled signal contain information about true support-action utility at all?
2. **Objective alignment:** which label-free objective, if any, has stable action-level ordering with supervised utility?
3. **Reliability / abstention:** can we determine when adaptation should be attempted?
4. **Estimator design:** only after a useful exact signal is found should we approximate it cheaply using gradients, JVPs, finite differences, curvature, or a learned scorer.
5. **Frozen confirmation:** only then should the signal be promoted to a method-level experiment.

The next experiment should therefore be an **Exact Label-Free Objective Audit**, not QCGS-v3.

---

## 2. What was inspected in the repository

The branch contains a fairly complete research chain around Ramen, support utility, QCGS, and the two STOP decisions.

Relevant repository artifacts include:

### Research direction and roadmap

- `docs/research/gradient-compatibility-research-direction.md`
- `docs/research/ramen-thesis-research-roadmap.md`
- `docs/research/query-conditioned-gradient-selection.md`
- `docs/research/qcgs-multiview-rescue-protocol.md`
- `docs/research/oracle-support-utility-probe.md`
- `plans/260914-1635-qcgs-label-free-research-roadmap/`

### Oracle / support utility evidence

- `plans/20260907-oracle-support-utility/`
- `plans/20260907-oracle-support-utility/reports/pilot-b-completed.md`

### QCGS single-view diagnostic

- `plans/20260919-qcgs-diagnostic/`
- `plans/20260919-qcgs-diagnostic/reports/results.md`

### QCGS multi-view rescue

- `plans/20260919-qcgs-multiview-rescue/`
- `plans/20260919-qcgs-multiview-rescue/reports/results.md`
- `plans/20260919-qcgs-multiview-rescue/reports/scientific-validation.json`

The repository also contains dedicated runtime, configs, tests, query registries, open-set splits, and provenance machinery for these experiments.

A requested local skill file,
`.agents/skills/research-orchestrator/references/map-research-literature.md`,
was not present on the inspected branch, so the literature step below follows the contract described by the research workflow rather than pretending that file was available.

---

## 3. Evidence before the two STOPs: support identity really matters

The most important positive result preceding QCGS is the supervised support-utility probe.

From `plans/20260907-oracle-support-utility/reports/pilot-b-completed.md`:

| Metric | OOD=0 | OOD=0.5 |
|---|---:|---:|
| Eligible ID queries | 128 | 128 |
| Queries with positive screened headroom | 114/128 | 120/128 |
| Mean oracle CE gain | +0.046677 | +0.331788 |
| Mean supervised first-order gain | +0.034269 | +0.320014 |
| Mean gradient-cosine gain | +0.015678 | +0.192436 |
| Mean random-swap gain | −0.009044 | −0.002948 |
| Ramen correct → oracle correct | 70 → 72 | 62 → 64 |

The important interpretation is not the exact magnitude. It is that:

\[
\exists a \in A(q)
\quad\text{such that}\quad
U_{\text{sup}}(q,a)>0
\]

for a large fraction of eligible queries.

This establishes that the support-action space is not empty of useful alternatives.

It also means that a failure of label-free QCGS cannot immediately be interpreted as:

> “support identity does not matter.”

The branch already has evidence that support identity does matter.

---

## 4. STOP #1 — single-view entropy-sign QCGS

The frozen single-view diagnostic is documented in:

`plans/20260919-qcgs-diagnostic/reports/results.md`

### 4.1 Stage A

Across 16 queries per OOD cell:

- exact oracle headroom was positive for **16/16 queries** in both cells;
- mean exact oracle CE gain was approximately:
  - `+0.408288` at OOD=0;
  - `+0.120676` at OOD=0.5;
- mean per-query Spearman between entropy-sign score and true utility was only:
  - `+0.018160`;
  - `+0.057525`;
- entropy-cosine mean rho was approximately zero or slightly negative.

Thus:

\[
\text{headroom exists}
\]

but

\[
\text{entropy ranking} \not\approx \text{supervised utility ranking}.
\]

### 4.2 Stage B

On 128 new ID queries per cell, the frozen entropy-sign policy produced negative mean utility relative to both Ramen and random.

| Measure | OOD=0 | OOD=0.5 |
|---|---:|---:|
| Mean utility vs Ramen | −0.018072 | −0.014818 |
| Mean advantage vs random | −0.013258 | −0.011178 |
| Mean entropy reduction | +0.031066 | +0.025518 |
| Helpful / harmful replacements | 70 / 58 | 67 / 61 |
| Lower entropy but worse CE | 27/128 | 36/128 |

This is one of the most important facts in the whole branch:

\[
\Delta H < 0
\]

does **not** imply

\[
\Delta CE < 0.
\]

Entropy can improve while supervised utility worsens.

### 4.3 Interpretation of STOP #1

The STOP is valid for the frozen signal, but it does **not** establish that:

- all label-free signals are impossible;
- all support selection is impossible;
- all gradient compatibility is impossible;
- the candidate space is useless.

It establishes that the particular chain

\[
\nabla H(q)
\rightarrow
\text{entropy-sign first-order score}
\rightarrow
\text{support action}
\]

is not sufficiently informative.

---

## 5. STOP #2 — multi-view rescue

The multi-view rescue is documented in:

`plans/20260919-qcgs-multiview-rescue/reports/results.md`

The protocol stopped at Stage A.

### 5.1 Main result

| Metric | OOD=0 | OOD=0.5 |
|---|---:|---:|
| Queries | 16 | 16 |
| Mean exact-oracle U | +0.326281 | +0.407242 |
| Mean primary U | +0.171246 | +0.123769 |
| Mean D vs random | +0.172130 | +0.120197 |
| Mean E vs Ramen-MV | +0.255355 | +0.021415 |
| Mean per-query Spearman | +0.014331 | **−0.125254** |

The selected action can have positive mean CE utility while the ranking signal itself is still poor.

This distinction matters:

\[
\text{policy mean gain} > 0
\]

does not imply

\[
\text{candidate ordering is reliable}.
\]

### 5.2 Sensitivity

After removing the two largest contributors from each metric, mean utility became negative in important cells.

This shows that the gain is sensitive to a small number of examples and that the harmful tail remains substantial.

The supervised reference still achieved mean rho around:

- `0.9280`
- `0.9098`

while the label-free primary signal achieved:

- `0.0143`
- `−0.1253`

So the action space itself remains rankable when task-relevant supervision is available.

### 5.3 Query-state heterogeneity

The report also showed that primary utility differed dramatically by baseline correctness.

For Ramen-wrong queries:

- mean U was strongly positive.

For Ramen-correct queries:

- mean U was negative.

This suggests that a future method may need a separate decision:

\[
\text{adapt or abstain?}
\]

before attempting full action ranking.

### 5.4 Interpretation of STOP #2

Multi-view information did not rescue the frozen QCGS formulation.

However, this does not prove that multi-view evidence is useless.

A more precise interpretation is:

> More views can reduce variance without necessarily correcting objective bias.

Formally:

\[
\operatorname{Var}(\hat g_{\text{wrong}})
\downarrow
\]

does not imply:

\[
\operatorname{bias}(\hat g_{\text{wrong}},g_{\text{sup}})
\downarrow.
\]

---

# 6. Core problem map after the two STOPs

The problem can be decomposed into distinct failure hypotheses.

| Failure hypothesis | Current evidence | Status |
|---|---|---|
| Candidate pool contains no useful support actions | Oracle headroom is positive and widespread | Mostly rejected |
| OOD contamination is the sole cause | Label-free failure also occurs at OOD=0 | Rejected as complete explanation |
| Feature relevance equals adaptation utility | Context/domain-routing experiments and QCGS disagree with this | Rejected as general principle |
| Entropy is an adequate proxy for supervised query utility | Entropy can decrease while CE worsens | Strongly challenged |
| Multi-view entropy fixes the proxy | Multi-view ranking still fails | Not sufficient |
| First-order approximation is the main problem | Not yet isolated from objective mismatch | Open |
| Some richer unlabeled signal can predict utility | Not yet established | Open |
| Utility may be unidentifiable from current evidence | Consistent with current failures; literature supports possibility | Open |

The central missing distinction is therefore:

\[
\boxed{\text{bad estimator}}
\]

versus

\[
\boxed{\text{insufficient / misaligned evidence}}.
\]

---

# 7. Broader literature map

The survey was broadened well beyond entropy-based TTA.

The useful literature families are:

1. entropy-minimization TTA;
2. reliable / selective adaptation;
3. open-set and noisy-stream TTA;
4. multi-view and augmentation-based adaptation;
5. VLM/CLIP-specific TTA objectives;
6. cache/prototype/memory-based VLM adaptation;
7. gradient influence and data valuation;
8. unlabeled model selection / risk estimation;
9. identifiability of adaptation from unlabeled evidence;
10. decision-theoretic adaptation.

---

## 8. Family A — entropy minimization TTA

### Tent

**Tent: Fully Test-Time Adaptation by Entropy Minimization**  
Wang et al., 2021.

Tent established entropy minimization as a simple and effective fully test-time adaptation objective.

Key idea:

\[
\theta' = \theta - \eta \nabla_\theta H(p_\theta(y|x)).
\]

This is the conceptual ancestor of the entropy-gradient assumption used in QCGS.

Reference:  
https://consensus.app/papers/details/b9e9007c71005ee0b9febfb399327910/?utm_source=chatgpt

### Implication for QCGS

Tent demonstrates that entropy is often useful for adaptation.

It does **not** establish that entropy change is a faithful proxy for:

\[
\Delta CE_y
\]

for every individual support action.

That distinction becomes crucial in QCGS.

---

## 9. Family B — reliable / selective entropy adaptation

### EATA

**Efficient Test-Time Model Adaptation without Forgetting**  
Niu et al., 2022.

Main observation:

- not all test samples contribute equally;
- high-entropy or redundant samples can generate noisy gradients;
- selective adaptation can improve stability.

Reference:  
https://consensus.app/papers/details/ee354940edb25c9a93925f7703d35836/?utm_source=chatgpt

### EATA-C / uncertainty-calibrated adaptation

Later work explicitly notes that plain entropy minimization can make predictions overconfident even when data uncertainty is high.

Reference:  
https://consensus.app/papers/details/f06d12bd29645f70ac554748720b2db0/?utm_source=chatgpt

### SAR

**Towards Stable Test-Time Adaptation in Dynamic Wild World / Adapt in the Wild**

SAR reports that some noisy samples induce large disruptive gradients and proposes sample filtering plus sharpness-aware adaptation.

References:

- https://consensus.app/papers/details/5e1c4e04fae95f83bc7a2b15ceccb8ff/?utm_source=chatgpt
- https://consensus.app/papers/details/ac491864e71c55b3abbfb6dcb07727b0/?utm_source=chatgpt

### Implication for QCGS

This literature strongly supports a distinction between:

\[
\text{Should this query/sample participate in adaptation?}
\]

and

\[
\text{Which exact support action should be selected?}
\]

Most reliable-TTA methods focus on the first question.

QCGS is attempting the second, harder question.

This suggests a possible hierarchy:

1. reliability / abstention;
2. then action ranking.

---

# 10. Family C — open-set and noisy-stream TTA

### Open-set TTA via confidence change

**Towards Open-Set Test-Time Adaptation Utilizing the Wisdom of Crowds in Entropy Minimization**  
Lee et al., ICCV 2023.

The paper observes that noisy/open-set samples can react differently under entropy adaptation and uses confidence change as a filtering signal.

Reference:  
https://consensus.app/papers/details/b3a9f1c439805f1bae37e9a7f0aa45ac/?utm_source=chatgpt

### SoTTA

**SoTTA: Robust Test-Time Adaptation on Noisy Data Streams**

SoTTA uses filtering and robust adaptation to suppress noisy samples.

Reference:  
https://consensus.app/papers/details/e6b1726db7f05261ba47c233590a9d99/?utm_source=chatgpt

### Implication for QCGS

Open-set contamination is a real mechanism.

But it cannot fully explain QCGS:

- the single-view policy failed at OOD=0;
- oracle support headroom also exists at OOD=0.

Therefore OOD filtering may improve reliability but is unlikely to solve the central ranking problem alone.

---

# 11. Family D — entropy is not enough

### DeYO

The important conceptual contribution of DeYO is that absolute predictive entropy can be misleading because it ignores latent factors that make the prediction brittle.

A counterfactual transformation can expose whether confidence actually depends on meaningful object evidence.

This introduces a more general principle:

\[
\text{static confidence}
\neq
\text{reliability}.
\]

A better signal may require observing:

\[
\text{response under intervention / perturbation}.
\]

### Relevance to QCGS

QCGS has mainly used local statistics or gradients derived from the same prediction state.

DeYO suggests that reliability may require a **counterfactual evidence channel**.

That is important because support utility is itself counterfactual:

> what happens to the query if this support action is executed?

---

# 12. Family E — multi-view and augmentation evidence

### MEMO

**MEMO: Test Time Robustness via Adaptation and Augmentation**

MEMO minimizes entropy of the marginal prediction over multiple augmentations.

Reference:  
https://consensus.app/papers/details/f526dab07a30597989d39b3728b62d93/?utm_source=chatgpt

### Test-Time Prompt Tuning

TPT uses multiple augmented views during prompt adaptation for CLIP-like models.

### MTA

**On the Test-Time Zero-Shot Generalization of Vision-Language Models: Do We Really Need Prompt Learning?**

MTA shows that multi-view aggregation itself can be a strong baseline.

Crucially, it does not simply trust every view equally; it estimates an inlierness-like quality variable.

Reference:  
https://consensus.app/papers/details/46771f259a885947a96e681e5fc9115b/?utm_source=chatgpt

### 2026 calibration-oriented TPT objective

Recent work revisits multi-view entropy minimization and argues that standard EM can create overconfidence while ignoring sample-specific uncertainty.

Reference:  
https://consensus.app/papers/details/7f7110e2caaa533ca847ea89c0a8ac04/?utm_source=chatgpt

### Implication for QCGS

Multi-view evidence should not be treated as automatically correct supervision.

It can:

- reduce variance;
- expose inconsistency;
- estimate reliability;
- help form a soft target.

But it does not guarantee that the resulting target is aligned with true query utility.

This is consistent with the multi-view rescue STOP.

---

# 13. Family F — VLM / CLIP objective alignment

This family is particularly relevant because the branch uses CLIP ViT-B/16.

### CLIPTTA

**CLIPTTA: Robust Contrastive Vision-Language Test-Time Adaptation**  
2025.

The core claim is that entropy-based TTA can be fundamentally misaligned with CLIP’s contrastive image-text pretraining objective.

CLIPTTA replaces plain entropy adaptation with a soft contrastive objective more aligned with the pretrained model.

Reference:  
https://consensus.app/papers/details/9c216bca35a958388b418c5adbdfc10b/?utm_source=chatgpt

### ADTE

**Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation**  
2026.

This work argues that Shannon entropy in CLIP-like models can reflect class/pretraining bias poorly and proposes a class-adaptive Tsallis entropy formulation.

Reference:  
https://consensus.app/papers/details/02c59b6a0f3453a8856a443076d077db/?utm_source=chatgpt

### RLCF / CLIP reward

**Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models**

This line of work uses an external/frozen CLIP-like reward signal to avoid blindly reinforcing the current prediction.

Reference:  
https://consensus.app/papers/details/3a075ddcf84e5f608046d5441bba8dce/?utm_source=chatgpt

### Implication for QCGS

The branch currently assumes:

\[
\nabla H(q)
\]

is a useful surrogate for the unknown task gradient.

The VLM-specific literature gives a direct reason to question this assumption.

A stronger candidate is some source-anchored or contrastive signal such as:

- image-text prototype margin;
- soft contrastive consistency;
- source-model agreement;
- text-manifold geometry;
- frozen-teacher or source-anchor reward.

The critical experiment is not yet to build a deployable scorer.

It is first to ask whether **exact post-update changes in those objectives** correlate with supervised support utility.

---

# 14. Family G — cache, prototype, and memory-based adaptation

### Tip-Adapter

Tip-Adapter shows that a key-value cache over visual features and labels/pseudo-label-like evidence can strongly adapt CLIP-like models without heavy training.

Reference family: ECCV 2022.

### TDA / dynamic test-time cache

Recent VLM adaptation methods build dynamic positive/negative caches, prototype structures, and pseudo-label refinement schemes during testing.

### Relevance to Ramen

Ramen already has a strong memory mechanism:

- class-partitioned cache;
- feature-local retrieval;
- prediction balance;
- cached gradients;
- query-specific temporary adaptation.

This means the novelty opportunity is unlikely to be:

> “use a cache.”

The useful distinction is:

\[
\text{relevance}
\neq
\text{trustworthiness}
\neq
\text{gradient compatibility}
\neq
\text{counterfactual utility}.
\]

Ramen already handles relevance fairly well.

QCGS is trying to move toward counterfactual utility.

---

# 15. Family H — gradient influence and data valuation

This family explains why the supervised oracle works.

### TracIn

TracIn evaluates how training examples influence predictions by using gradient similarity across checkpoints / target examples.

### GRAD-MATCH

Gradient matching selects subsets whose gradient approximates a target training or validation gradient.

### GLISTER

GLISTER casts subset selection as a bilevel objective that chooses data improving held-out validation performance.

### General lesson

The usefulness of a support example is normally defined relative to a **target objective**:

\[
U(s \mid q)
\sim
-\langle
\nabla L_q,
\nabla L_s
\rangle
\]

or a richer influence approximation.

This literature supports the empirical behavior of the supervised first-order QCGS reference.

But it also exposes the hard part:

\[
\nabla L_q
\]

is not available in label-free TTA.

Therefore the bottleneck is not necessarily the support-gradient side.

It may be the missing target/query objective.

---

# 16. Theoretical connection — Test-Time Training and gradient alignment

A particularly important conceptual precedent is Test-Time Training.

The self-supervised update helps the main task when the self-supervised and main-task gradients are sufficiently aligned.

The key idea is approximately:

\[
\langle
\nabla L_{\rm main},
\nabla L_{\rm self}
\rangle > 0.
\]

This gives a clean interpretation of the QCGS evidence:

- supervised query gradient:
  \[
  g_q^{sup}
  \]
  can produce a useful ordering;
- entropy gradient:
  \[
  g_q^{ent}
  \]
  largely fails.

Therefore the right question is not:

> “how can entropy-sign be improved?”

but:

> **what unlabeled signal produces a target direction aligned enough with the unknown supervised query gradient to preserve support-action ordering?**

This also means that simply claiming “gradient compatibility” is unlikely to be sufficiently novel on its own, because gradient alignment and gradient-based data utility already have substantial prior literature.

---

# 17. Family I — unlabeled model selection and risk estimation

A separate literature asks whether model quality or adaptation choices can be selected without target labels.

The important general finding is that unlabeled model selection is difficult and many surrogates are unstable across distribution shifts.

This matters because QCGS asks for something even more fine-grained:

\[
\text{choose the best action for this query}.
\]

That is harder than choosing:

\[
\text{the best model / hyperparameter for a distribution}.
\]

### Implication

If distribution-level unlabeled selection is already fragile, per-query action selection should not be assumed possible without testing information sufficiency directly.

---

# 18. Family J — identifiability from unlabeled evidence

The most conceptually important recent direction is explicit **identifiability**.

The key question is:

> Does the observation channel contain enough information to determine which adaptation action is better?

If two deployment situations produce nearly the same accessible evidence but have different oracle action orderings, then no selector using only that evidence can always succeed.

This yields the critical decomposition:

\[
\boxed{\text{estimation error}}
\]

versus:

\[
\boxed{\text{information insufficiency}}.
\]

This is highly relevant to QCGS.

The single-view entropy signal may fail because:

- the estimator is poor;

or because:

- entropy is simply not informative enough about supervised action utility.

Likewise, adding multi-view evidence may still fail if it only refines an observation channel that does not distinguish the action ordering that matters.

---

# 19. Decision-theoretic interpretation

Recent decision-theoretic work on test-time adaptation emphasizes that adaptation should depend on:

- whether a useful signal exists;
- which directions contain useful signal;
- how far to adapt;
- query-specific signal-to-noise.

This reinforces a deeper version of the Ramen thesis question:

> The problem is not only what the query should adapt from, but what evidence is sufficient to justify a particular adaptation direction.

---

# 20. The three-layer decomposition that emerged from the survey

The literature suggests separating the problem into three layers.

## Layer 1 — Reliability

Question:

> Is this query/support evidence trustworthy enough to adapt at all?

Examples:

- entropy threshold;
- confidence change;
- sharpness;
- augmentation agreement;
- OOD detection;
- counterfactual perturbation;
- calibration.

## Layer 2 — Objective alignment

Question:

> If we adapt, does the unlabeled objective move in a direction related to true task utility?

Examples:

- entropy;
- calibrated entropy;
- contrastive image-text objective;
- source-anchor agreement;
- neighborhood consensus;
- multi-view soft target.

## Layer 3 — Action ranking / estimation

Question:

> Given a useful objective, can we cheaply rank support substitutions?

Examples:

- gradient inner product;
- cosine;
- sign-aware first-order score;
- JVP;
- finite difference;
- Hessian / curvature correction;
- learned scorer.

The previous QCGS pipeline effectively jumped directly from Layer 2 assumptions into Layer 3 engineering.

The next research cycle should first test whether Layer 2 is valid.

---

# 21. Candidate research gaps after the expanded survey

## G1 — Objective identifiability before estimation

**Failure explained:** entropy-sign may fail because entropy outcome itself does not order supervised utility.

**New assumption:** at least one observable label-free objective has meaningful action-level alignment with supervised utility.

**Small falsification test:** execute all legal actions and directly measure exact post-update objective change.

This is the highest-priority gap.

---

## G2 — Source / pretraining-aligned support utility

**Failure explained:** entropy may be poorly aligned with CLIP’s contrastive geometry.

**New assumption:** a source-anchored image-text or contrastive signal preserves more task-relevant information.

**Candidate evidence:**

- image-text logit margin;
- soft contrastive loss;
- text-prototype geometry;
- source-model agreement;
- frozen VLM reward.

**Test:** measure exact action-level proxy change before designing a first-order approximation.

---

## G3 — Reliability-conditioned abstention

**Failure explained:** support changes help some queries and harm others; baseline-wrong and baseline-correct queries behave differently.

**New assumption:** an observable reliability variable can identify the subset where adaptation is beneficial.

The target becomes:

\[
\text{adapt vs no-swap}
\]

before:

\[
\text{which swap?}
\]

This is strongly supported by EATA/SAR/DeYO/open-set TTA literature.

---

## G4 — Neighborhood-conditioned evidence

**Failure explained:** a single query observation may be informationally insufficient.

**New assumption:** local memory context contains extra information about whether an action is useful.

Potential evidence:

- neighborhood agreement;
- pseudo-class prototype density;
- temporal consistency;
- source-anchor consistency;
- local view stability;
- consensus among retrieved supports.

---

## G5 — Better action estimator geometry

**Failure explained:** exact label-free objective may be useful but the current first-order/sign approximation may be poor.

**New assumption:** the problem lies in approximation geometry, not objective choice.

Possible estimators:

- exact directional finite difference;
- JVP;
- sign-aware finite-difference approximation;
- HVP / curvature correction;
- local learned ranker.

This gap should only be opened if an exact label-free objective passes the information screen.

---

# 22. Revised central research question

The previous question was too specific:

> Can a query’s unlabeled entropy gradient select a support replacement with greater supervised query utility?

The revised question should be:

> **Within Ramen’s fixed query-conditioned support candidate pool, which unlabeled observation channels contain enough information to distinguish beneficial from harmful support-gradient substitutions, and when can that information be converted into a reliable deployable ranking?**

This question separates:

\[
\boxed{\text{Is utility observable?}}
\]

from:

\[
\boxed{\text{Can utility be estimated cheaply?}}
\]

That separation is the main conceptual outcome of the literature reopening.

---

# 23. Revised hypothesis tree

## H1 — Objective mismatch

Entropy-family outcomes do not preserve stable action-level ordering with supervised utility.

If supported, close entropy-QCGS as a family rather than merely replacing the score formula.

---

## H2 — VLM-aligned evidence

A source/text/contrastive evidence channel produces stronger and more stable action-level alignment than predictive entropy.

This is motivated by VLM-specific TTA work arguing that entropy can be misaligned with contrastive pretraining.

---

## H3 — Reliability precedes ranking

A universal per-query ranking may be too strong.

An observable reliability gate may first identify where support changes are safe or useful.

Possible claim:

\[
E[U_{\rm selector}\mid R(q)=1] > 0
\]

while abstaining otherwise.

---

## H4 — Evidence insufficiency

Even rich unlabeled evidence may fail to identify support-action ordering.

If outcome-equivalent observation states can produce opposite oracle rankings, then the limitation is not estimator quality but lack of information.

This should become an explicit STOP possibility, not an afterthought.

---

# 24. The most important next experiment: Exact Label-Free Objective Audit

The branch currently tests approximately:

\[
\nabla H(q)
\rightarrow
\widehat{\Delta H}(a)
\rightarrow
a^*
\rightarrow
U_{\rm sup}(a).
\]

This conflates two assumptions.

### Assumption A — objective alignment

\[
\Delta H(a)
\leftrightarrow
U_{\rm sup}(a).
\]

### Assumption B — approximation quality

\[
\widehat{\Delta H}(a)
\leftrightarrow
\Delta H(a).
\]

If Assumption A is false, improving Assumption B is scientifically unimportant.

Therefore the next experiment should execute each candidate action exactly and measure both:

\[
U_{\rm sup}(q,a)
\]

and:

\[
\Delta z(q,a)
\]

for a set of label-free objectives \(z\).

No selector is needed at this stage.

---

# 25. Candidate objective families for the audit

The purpose is not a heuristic sweep.

Each family should correspond to a distinct literature-backed information hypothesis.

## 25.1 Entropy / calibration family

- Shannon entropy;
- margin / confidence;
- calibrated entropy;
- Tsallis-style entropy;
- uncertainty from disagreement;
- confidence change after adaptation.

Question:

> Is the entropy family itself informative if measured exactly?

---

## 25.2 Contrastive / source-anchor family

- image-to-text prototype margin;
- soft contrastive loss;
- source model agreement;
- source text manifold distance;
- frozen teacher reward.

Question:

> Does an objective closer to CLIP pretraining preserve support utility better?

---

## 25.3 Multi-view family

- marginal entropy over views;
- view agreement;
- target-distribution cross-entropy;
- inlier-weighted view consensus;
- prediction variance across semantic-preserving augmentations.

Question:

> Do additional views contribute genuinely new information or only reduce noise?

---

## 25.4 Counterfactual reliability family

- DeYO-like response to object-destructive perturbation;
- prediction change under structured corruption;
- confidence response before/after support update;
- robustness of predicted class under intervention.

Question:

> Is an intervention-based signal more informative than a static uncertainty score?

---

## 25.5 Neighborhood / memory family

- neighbor prediction agreement;
- prototype density;
- local class-margin consistency;
- neighborhood entropy;
- support consensus;
- temporal consistency.

Question:

> Is single-query evidence insufficient but local memory evidence sufficient?

---

# 26. Metrics for the information screen

For each query and each exact label-free objective:

### Ranking

\[
\rho_q
=
\operatorname{Spearman}
(
\Delta z(q,\cdot),
U_{\rm sup}(q,\cdot)
).
\]

Report:

- mean rho;
- median rho;
- positive-rho fraction;
- valid/total query count.

### Sign prediction

Estimate:

\[
P(U_{\rm sup}>0 \mid \Delta z>0).
\]

This directly asks whether a positive proxy action is more likely to help than harm.

### Top-action utility

For:

\[
a_z^*=\arg\max_a \Delta z(q,a),
\]

measure:

\[
U_{\rm sup}(q,a_z^*).
\]

Compare with:

- no-swap;
- random;
- supervised first-order;
- exact oracle.

### Regret

\[
R_z(q)
=
U_{\rm oracle}(q)
-
U_{\rm sup}(q,a_z^*).
\]

### Stability

Measure whether the objective retains ordering across:

- OOD=0;
- OOD=0.5;
- corruption groups;
- baseline-correct vs baseline-wrong;
- uncertainty strata;
- memory states.

---

# 27. Falsification ladder

A disciplined next phase should follow this order.

## L0 — Exact objective audit

Use a fresh, outcome-independent Stage-A-style registry.

Execute all legal one-swap actions.

Record:

- exact supervised utility;
- exact post-update label-free objectives;
- action metadata;
- cost.

No deployable selector yet.

---

## L1 — Information screen

For every candidate objective:

- per-query ranking;
- sign precision;
- top-action utility;
- regret;
- cross-cell stability.

Kill entire objective families that fail exact evaluation.

---

## L2 — Reliability screen

Ask whether any observable signal predicts:

\[
\max_a U(q,a) > 0
\]

or whether a chosen action beats no-swap.

This tests whether:

> “know when to adapt”

is easier than:

> “know exactly which action is best.”

---

## L3 — Evidence enrichment

Only if local single-query evidence fails:

- add neighborhood information;
- add temporal evidence;
- add source-anchor evidence;
- add multi-view evidence;
- add one conceptually distinct family at a time.

Do not combine many heuristics before establishing which channel adds information.

---

## L4 — Estimator design

Only for objectives that pass L1/L2:

- gradient dot product;
- sign-aware score;
- finite difference;
- JVP;
- HVP;
- learned ranker.

This is where QCGS-like engineering should resume.

---

## L5 — Frozen confirmation

Lock:

- objective;
- feature/evidence set;
- estimator;
- threshold;
- no-swap rule;
- cost budget;
- Stage B query registry;
- GO/STOP gates.

Then confirm on fresh queries.

---

## L6 — Identifiability STOP

If several conceptually distinct and richer evidence channels all fail exact action-level prediction, and evidence-near queries can exhibit reversed oracle orderings, stop the universal ranking direction.

Possible pivots:

- selective adaptation;
- bounded guarantees;
- population-level selection;
- identifiability characterization;
- explainable failure analysis.

---

# 28. Why “QCGS-v3 with better scoring” should not be the next step

The current evidence does not justify immediately trying:

- another cosine;
- another entropy threshold;
- another number of views;
- a learned MLP scorer;
- Hessian correction;
- larger candidate M;
- adaptive k;
- more reranking.

All of these assume that the current observation channel already contains the relevant signal.

That assumption is exactly what has not been established.

A stronger research sequence is:

\[
\text{information}
\rightarrow
\text{objective}
\rightarrow
\text{reliability}
\rightarrow
\text{estimator}
\rightarrow
\text{method}.
\]

---

# 29. Novelty assessment

A claim such as:

> “select supports using gradient compatibility”

is probably too weak as the main thesis novelty.

Related ideas already exist in:

- Test-Time Training;
- auxiliary-task gradient alignment;
- influence functions;
- TracIn;
- GRAD-MATCH;
- GLISTER;
- supervised gradient-based data valuation.

A more defensible novelty target is:

> **Label-Free Identifiability of Query-Conditioned Support Utility for Gradient-Memory Test-Time Adaptation**

or:

> **What Unlabeled Evidence Predicts Whether a Retrieved Gradient Support Will Help a Test Query?**

The scientific object becomes:

\[
(q,M,A(q))
\rightarrow
z
\rightarrow
\operatorname{rank} A(q)
\]

with hidden evaluator utility:

\[
U(q,a)
=
L_y(\theta_{\rm Ramen})
-
L_y(\theta_a).
\]

The novelty is not the cosine or dot product.

The novelty is the explicit study of **support-action utility under label-free constraints**.

---

# 30. Ranking versus abstention

One of the most important possible simplifications is that full ranking may be unnecessarily strong.

Current target:

\[
\arg\max_{a\in A(q)} U(q,a).
\]

Potentially easier target:

\[
\text{Should we change support at all?}
\]

This leads to a selective policy:

\[
R(q)\in\{0,1\}.
\]

Only if:

\[
R(q)=1
\]

does the method attempt a support modification.

This is well supported by the selective-TTA literature and is consistent with the branch’s observation that:

- some query states benefit strongly;
- others are harmed;
- harmful tails can dominate mean behavior.

A future result of the form

\[
E[U \mid R(q)=1] > 0
\]

with good abstention behavior may be scientifically stronger than a fragile universal ranker.

---

# 31. Evaluation principles for any future method

The literature and branch evidence both argue against relying only on average accuracy or average CE.

A future confirmation should preserve at least:

### Utility

- mean CE utility;
- median;
- trimmed mean;
- harm rate;
- helpful/harmful magnitude.

### Prediction transitions

- wrong → correct;
- correct → wrong;
- unchanged correct;
- unchanged wrong.

### Ranking

- per-query Spearman;
- Kendall if useful;
- top-action regret.

### Reliability

- adaptation coverage;
- abstention quality;
- utility conditional on adaptation.

### Calibration

- entropy;
- confidence;
- ECE or another calibration measure.

### Open-set robustness

- ID/OOD breakdown;
- OOD exposure effects;
- memory contamination.

### Cost

- retrieval;
- scoring;
- evaluator-free deployable latency;
- memory;
- gradient computations.

---

# 32. Literature saturation / survey stopping rule

The survey should not stop after an arbitrary paper count.

The right stopping rule is:

> Stop when new papers mostly fall into already-known mechanism classes and no longer change the candidate research questions.

The current literature clusters are already relatively stable:

1. entropy;
2. calibrated / reliable entropy;
3. selective adaptation;
4. open-set filtering;
5. multi-view aggregation;
6. VLM-aligned objective;
7. cache/prototype memory;
8. gradient influence;
9. unlabeled model selection;
10. identifiability / decision theory.

At this point, additional reading is more likely to refine each cluster than to change the central research question.

Therefore it is reasonable to exit the broad Literature stage and return to Question/Hypothesis design.

---

# 33. Recommended research state transition

The current workflow state should be:

```text
Previous Question
    ↓
Single-view QCGS
    ↓ STOP
Multi-view rescue
    ↓ STOP
Reopen Literature
    ↓
Mechanism-level problem map
    ↓
Revised Question
    ↓
Exact objective audit
```

Do **not** transition directly to another full method.

---

# 34. Recommended revised Question

> **Within Ramen’s fixed query-conditioned support candidate pool, which unlabeled observation channels contain enough information to distinguish beneficial from harmful support-gradient substitutions, and when can that information be converted into a reliable deployable ranking?**

A shorter thesis-facing version:

> **What unlabeled evidence predicts whether a retrieved gradient support will help a test query?**

---

# 35. Recommended near-term experiment name

A clear working name would be:

## Exact Label-Free Support-Utility Objective Audit

Suggested purpose:

> Determine whether candidate label-free objectives contain action-level information about supervised support utility before developing a deployable selector.

Suggested scope:

- CIFAR-100-C first;
- CLIP ViT-B/16;
- same fixed Ramen candidate space;
- OOD=0 and OOD=0.5;
- fresh registry;
- exhaustive one-swap actions;
- no threshold tuning;
- no method promotion;
- no Stage B unless a channel passes a preregistered information gate.

---

# 36. Suggested GO / STOP logic for the next audit

The exact thresholds should be preregistered before execution, but the logical order should be:

### INVALID

- provenance failure;
- candidate-space mismatch;
- label leakage;
- incomplete exhaustive verification;
- non-finite utility/objective.

### INCONCLUSIVE

- insufficient eligible queries;
- objective undefined for too many actions;
- execution budget insufficient.

### STOP OBJECTIVE FAMILY

If exact label-free objective change has:

- no stable positive per-query ranking;
- no positive selected-action utility over no-swap/random;
- or strong cross-cell reversal.

Then do not build a gradient estimator for that objective.

### GO ESTIMATOR

Only if the exact objective itself:

- ranks utility positively;
- selects helpful actions more often than random;
- retains behavior across both OOD cells;
- does not rely on a tiny number of outliers.

Then proceed to approximating it cheaply.

---

# 37. Final research interpretation

The two STOPs should be viewed as productive negative results.

They have already eliminated an overly simple story:

\[
\text{query entropy gradient}
\Rightarrow
\text{support utility}.
\]

The repository now supports a richer statement:

1. support identity affects query adaptation;
2. oracle headroom is real;
3. supervised target gradients can expose that headroom;
4. entropy-based label-free target gradients fail to recover the same action ordering;
5. multi-view information alone does not fix the mismatch;
6. therefore the next scientific problem is **not ranking engineering**;
7. it is **identifying what unlabeled evidence, if any, contains task-relevant information about support utility**.

The most important conceptual pivot is:

\[
\boxed{
\text{from gradient compatibility}
\;\longrightarrow\;
\text{label-free support-utility identifiability}
}
\]

Only after that question is answered should the project return to QCGS-style estimator design.

---

# 38. Representative literature references

The following references were central to the survey and problem map.

1. Wang et al. — **Tent: Fully Test-Time Adaptation by Entropy Minimization** (2021)  
   https://consensus.app/papers/details/b9e9007c71005ee0b9febfb399327910/?utm_source=chatgpt

2. Niu et al. — **Efficient Test-Time Model Adaptation without Forgetting (EATA)** (2022)  
   https://consensus.app/papers/details/ee354940edb25c9a93925f7703d35836/?utm_source=chatgpt

3. Niu et al. — **Towards Stable Test-Time Adaptation in Dynamic Wild World (SAR)** (2023)  
   https://consensus.app/papers/details/5e1c4e04fae95f83bc7a2b15ceccb8ff/?utm_source=chatgpt

4. Niu et al. — **Adapt in the Wild: Test-Time Entropy Minimization with Sharpness and Feature Regularization** (2025)  
   https://consensus.app/papers/details/ac491864e71c55b3abbfb6dcb07727b0/?utm_source=chatgpt

5. Lee et al. — **Towards Open-Set Test-Time Adaptation Utilizing the Wisdom of Crowds in Entropy Minimization** (ICCV 2023)  
   https://consensus.app/papers/details/b3a9f1c439805f1bae37e9a7f0aa45ac/?utm_source=chatgpt

6. Gong et al. — **SoTTA: Robust Test-Time Adaptation on Noisy Data Streams** (2023)  
   https://consensus.app/papers/details/e6b1726db7f05261ba47c233590a9d99/?utm_source=chatgpt

7. MEMO — **Test Time Robustness via Adaptation and Augmentation**  
   https://consensus.app/papers/details/f526dab07a30597989d39b3728b62d93/?utm_source=chatgpt

8. Shu et al. — **Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models**  
   https://consensus.app/papers/details/8e250686647f5dfe92868c814649f743/?utm_source=chatgpt

9. Zanella et al. — **On the Test-Time Zero-Shot Generalization of Vision-Language Models: Do We Really Need Prompt Learning?** (MTA)  
   https://consensus.app/papers/details/46771f259a885947a96e681e5fc9115b/?utm_source=chatgpt

10. **CLIPTTA: Robust Contrastive Vision-Language Test-Time Adaptation** (2025)  
    https://consensus.app/papers/details/9c216bca35a958388b418c5adbdfc10b/?utm_source=chatgpt

11. Zhao et al. — **Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models**  
    https://consensus.app/papers/details/3a075ddcf84e5f608046d5441bba8dce/?utm_source=chatgpt

12. **Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation** (2026)  
    https://consensus.app/papers/details/02c59b6a0f3453a8856a443076d077db/?utm_source=chatgpt

13. **Rethinking the Test-Time Prompt Tuning Objective from the Perspective of Calibration** (2026)  
    https://consensus.app/papers/details/7f7110e2caaa533ca847ea89c0a8ac04/?utm_source=chatgpt

14. Zhou et al. — **Bayesian Test-Time Adaptation for Vision-Language Models** (CVPR 2025)  
    https://consensus.app/papers/details/48626872cdff527490deb5b192d72080/?utm_source=chatgpt

15. Zhang et al. — **Dual Prototype Evolving for Test-Time Generalization of Vision-Language Models**  
    https://consensus.app/papers/details/3725160e0ec352b0a5b1d8406c73ef22/?utm_source=chatgpt

16. Osowiechi et al. — **WATT: Weight Average Test-Time Adaptation of CLIP**  
    https://consensus.app/papers/details/0c2c2964e48a51ad84cc58ccdcee6ac1/?utm_source=chatgpt

17. Sheng et al. — **The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models**  
    https://consensus.app/papers/details/380e9db0b1815fff9a056e9bcc52f888/?utm_source=chatgpt

18. **Test-Time Consistency in Vision Language Models**  
    https://consensus.app/papers/details/66cecb0b1a8059c2aed73041df1c8a43/?utm_source=chatgpt

---

# 39. Immediate next action

The recommended next repository action is **documentation and protocol design**, not implementation.

Create a new research note / plan containing:

1. revised Question;
2. H1–H4 hypothesis tree;
3. exact objective families;
4. fresh query registry rule;
5. information-screen metrics;
6. preregistered GO/STOP gates;
7. explicit prohibition on estimator tuning before objective validation.

Only after that protocol is frozen should the next CUDA evidence campaign be opened.

