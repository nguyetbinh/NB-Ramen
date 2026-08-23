# Ramen to Thesis: Research Roadmap for Structured Test-Time Adaptation

> Status: working research report / thesis planning note  
> Repository: `nguyetbinh/NB-Ramen`  
> Scope: summarize the research discussion around Ramen, audit the current codebase, and define a concrete path from the existing paper to a thesis-level research program.

## 1. Executive summary

Ramen is a strong empirical starting point for mixed-domain test-time adaptation (TTA) of vision-language models. Its main practical idea is to avoid adapting a test sample using an undifferentiated mixed stream. Instead, it builds a sample-specific support set from a class-partitioned memory, favors nearby samples in CLIP feature space, balances retrieval across predicted classes, aggregates cached per-sample gradients, performs a temporary adaptation step, predicts, and then resets the model parameters.

The main thesis opportunity is not to build a slightly more accurate `Ramen++`. The stronger research question is:

> **How can a model adapt reliably and efficiently at test time when the deployment stream is heterogeneous, non-stationary, only partially observable, and contaminated by uncertain pseudo-labels?**

The recommended thesis direction is **Latent Structure-Aware Test-Time Adaptation**. The central hypothesis is that a deployment stream contains local latent structure—environment, corruption, acquisition style, temporal regime, subpopulation, or other context—that should be inferred online and used to determine *what each test sample should adapt from*.

A coherent thesis can therefore progress through four stages:

1. **Ramen baseline:** sample-specific mixed-domain TTA using class-balanced nearest-neighbor support.
2. **Latent context-aware TTA:** infer online latent environments and route samples to context-aware support sets.
3. **Reliable structured memory:** prevent pseudo-label and gradient contamination using uncertainty, neighborhood agreement, temporal consistency, and gradient agreement.
4. **Scalable gradient memory:** compress and organize feature/gradient memories for large vocabularies and larger VLMs.

This direction is feasible in the current codebase because the per-sample gradient machinery is already separated from the cache/retrieval logic. The first research prototype can therefore reuse `ModelForBySampleTTA.py` and focus on stream generation, routing, memory structure, and evaluation.

---

## 2. What Ramen already establishes

### 2.1 Problem framing

Ramen addresses mixed-domain TTA, where test samples do not necessarily come from one stable target domain. A global adaptation state can be harmful when unrelated domains or classes are interleaved. The paper instead uses sample-specific support selection so that each query is adapted using a more relevant subset of previously observed samples.

### 2.2 Core mechanism

The current implementation in [`src/methods/Ramen.py`](../../src/methods/Ramen.py) follows this sequence:

```text
image
  -> CLIP feature z
  -> zero-shot logits
  -> pseudo-class c_hat = argmax(logits)
  -> entropy loss
  -> per-sample gradient g
  -> insert (z, g, entropy, recency) into cache[c_hat]
  -> query every non-empty class cache with z
  -> top-k nearest samples per class
  -> entropy weighting * feature-distance weighting
  -> aggregate retrieved gradients
  -> temporary parameter update
  -> inference
  -> reset adapted parameters
```

Important implementation facts:

- Memory is partitioned **by predicted class**, not true class.
- A sample is admitted immediately to the queue selected by `argmax(logits)`.
- Retrieval uses Euclidean distance through `torch.cdist` over normalized CLIP features.
- Retrieval is performed from every non-empty class cache, which induces prediction balance.
- Retrieved gradients are weighted by sample entropy and feature distance.
- The current sample is inserted before retrieval, so its own feature/gradient can appear in the support set.
- Adaptation is episodic at the parameter level: the model is reset after prediction, while the memory persists through the stream.

### 2.3 Efficient per-sample gradient support

[`src/models/ModelForBySampleTTA.py`](../../src/models/ModelForBySampleTTA.py) implements the expensive part that is worth preserving. Normalization layers are reparameterized with per-sample affine parameters so that one backward pass produces a matrix of sample-wise gradients.

The abstraction exposes:

- `get_by_sample_grad()`
- `set_by_sample_grad()`
- `step_and_zero_grad()`
- `reset_parameters()`

This separation is a strong engineering advantage for future work. New routing, retrieval, reliability, and memory algorithms can operate on feature/gradient tuples without redesigning the differentiation mechanism.

### 2.4 Empirical interpretation from the paper discussion

The paper-reported improvements over the strongest compared baselines were approximately:

| Benchmark | Reported gain |
|---|---:|
| CIFAR-10-C mixed | +1.3 points |
| CIFAR-100-C mixed | +3.4 points |
| ImageNet-C mixed | +2.5 points |
| DomainNet mixed | +0.6 points |

The important research interpretation is that **headline performance is not the obvious weakness of Ramen**. The corruption benchmarks show substantial gains. The more concerning signal is that the improvement on the more natural DomainNet shift is smaller.

Therefore, the working assessment is:

> Ramen is empirically strong, but the larger opportunity is to improve conceptual novelty, realism/breadth of deployment streams, robustness to incorrect memory entries, and scalability—not merely to add another small accuracy gain on the same corruption protocol.

The earlier discussion about CVPR Findings should be treated as an interpretation rather than official reviewer feedback. A plausible explanation is that Ramen combines several established ingredients—memory, nearest-neighbor retrieval, prediction balancing, entropy minimization/weighting, and gradient caching—into a very effective system, but may be perceived as more incremental conceptually than some main-track methods that introduce a new adaptation axis or failure-mode formulation.

---

## 3. Code audit: where the current implementation creates research opportunities

### 3.1 Mixed-domain evaluation is random interleaving

[`src/datasets/utils.py`](../../src/datasets/utils.py) creates `TaggedMultipleDataset` by concatenating all samples into an index map containing `(domain_idx, sample_idx)`.

[`src/main.py`](../../src/main.py) then evaluates mixed-domain TTA using a `DataLoader(..., shuffle=True, ...)`.

This means the current mixed-domain stream is effectively:

```text
D1 U D2 U ... U DM -> random shuffle -> test stream
```

It does **not** explicitly model realistic temporal structure such as:

- persistent domains;
- sudden domain switches;
- gradual transitions;
- recurring domains;
- highly imbalanced domain frequencies;
- new domains appearing halfway through deployment;
- class/domain correlation;
- temporal bursts;
- open-set or partial-label shifts.

This is one of the highest-value extension points because it changes the scientific question from static mixture handling to **online adaptation under structured non-stationarity**.

A useful property is that `TaggedMultipleDataset` already exposes `domain_idx`. A future method should not consume this label, but evaluation can use it to measure whether an unsupervised latent router discovers meaningful domain/context structure.

### 3.2 Hard pseudo-class assignment can contaminate memory

Current admission logic is effectively:

```python
init_preds = logits.argmax(-1)
...
c = init_preds[b]
self.cache[c].add(...)
```

There is no admission gate based on:

- confidence;
- entropy threshold;
- neighbor agreement;
- augmentation consistency;
- temporal consistency;
- predicted-class stability;
- gradient consistency.

If a sample is confidently misclassified, its feature and gradient are stored in the wrong class memory and may affect future queries.

This creates a direct thesis question:

> **How should an online TTA system decide whether a test sample is trustworthy enough to become memory?**

Possible actions include hard rejection, soft admission weights, delayed admission, relabeling, or memory eviction.

### 3.3 Domain consistency is currently a proxy from raw feature distance

Ramen assumes that nearby normalized CLIP features are likely to come from compatible domains/contexts. The implementation queries each class cache with `torch.cdist` and takes the nearest items.

This works empirically, but it leaves an important conceptual gap:

> **Feature similarity is not necessarily the same as adaptation compatibility.**

Two samples may be semantically close but affected by different acquisition conditions. Conversely, two samples may come from the same latent environment but be far apart due to class semantics.

A thesis-level extension should therefore move from raw proximity to **latent context inference** and eventually to **adaptation-utility-aware support selection**.

### 3.4 Retrieval scales with the number of classes

Ramen maintains one `PriorityCache` per class and queries every non-empty class cache for every batch. Each cache preallocates tensors for both features and full gradients:

```text
keys   : max_capacity x feature_dim
values : max_capacity x gradient_dim
```

This has two consequences:

1. GPU memory scales with the number of classes and cache capacity.
2. Retrieval work scales with the number of non-empty class queues.

The repository configuration already suggests practical pressure from large-class settings:

- [`cfg/DomainNet/Ramen.yaml`](../../cfg/DomainNet/Ramen.yaml): `max_capacity=300`, `topk=10`, `beta=5.0`.
- [`cfg/ImageNetC5K/Ramen.yaml`](../../cfg/ImageNetC5K/Ramen.yaml): `max_capacity=75`, `topk=1`, `beta=0.0`.

These settings are not proof of a fundamental limit, but they motivate a serious scalability study.

### 3.5 SignSGD changes what a useful retrieval score should optimize

The shipped Ramen configurations use `signsgd`. [`src/models/optimizer.py`](../../src/models/optimizer.py) applies an elementwise sign to the aggregated gradient before the parameter update:

```text
theta <- theta - eta * sign(g)
```

This suggests that a future learned retrieval method should not only learn scalar importance weights that change gradient magnitude. What matters more is whether selected support samples produce a **useful aggregate gradient direction**.

A stronger compatibility score can combine:

```text
feature similarity
+ gradient alignment
- uncertainty
```

This gives a principled path from nearest-neighbor retrieval to **adaptation-compatibility retrieval**.

### 3.6 Current-sample self-retrieval should be isolated in future ablations

The current sample is inserted into memory before support retrieval. Therefore, its own distance to itself is zero and it is likely to be retrieved from its predicted-class cache, especially when `topk=1`.

A future evaluation should explicitly compare:

- current-sample-only entropy minimization;
- past-memory-only retrieval;
- current sample + historical support;
- historical support excluding same predicted class;
- oracle-domain historical support.

This will quantify how much of the gain actually comes from active memory retrieval versus the current query's own gradient.

---

## 4. Recommended thesis framing

### 4.1 Proposed title family

**Latent Structure-Aware Test-Time Adaptation for Heterogeneous and Non-Stationary Deployment Streams**

Alternative shorter framing:

**Structured Test-Time Adaptation under Heterogeneous Streams**

### 4.2 Thesis statement

> Rather than adapting one global model to an unknown target distribution, test-time adaptation should infer and exploit the latent local structure of the deployment stream so that each sample adapts from relevant, trustworthy, and computationally efficient evidence.

### 4.3 Central research question

> **What should a test sample adapt from?**

Ramen answers:

> nearby samples, balanced across predicted classes.

The thesis-level answer should become:

> samples belonging to a compatible latent context, carrying reliable pseudo-label/gradient information, and contributing a stable adaptation direction under changing deployment conditions.

---

## 5. Primary research direction: Online Latent Context-Aware Ramen

This should be the first major extension because it directly attacks the strongest assumption in Ramen while reusing most of the existing implementation.

### 5.1 Baseline memory

Current structure:

```text
cache[class_0]
cache[class_1]
...
cache[class_C-1]
```

### 5.2 Proposed structured memory

Hard-context version:

```text
class c
  -> context 0
  -> context 1
  -> ...
  -> context M-1
```

Equivalent memory index:

```text
M[class][context]
```

A more flexible implementation uses soft routing `q(context | z)`.

### 5.3 Minimal latent router

Do not start with a large neural network. A thesis prototype should begin with online prototypes:

```text
mu_1, mu_2, ..., mu_M
```

For a query feature `z_i`, assign a soft context posterior using distance to prototypes. A new context can be spawned when the nearest prototype is farther than a threshold or when a change-point score becomes large.

### 5.4 Context-aware support selection

Current conceptual support set:

```text
for each class:
    retrieve top-k nearest samples from class memory
```

Proposed version:

```text
for each class:
    retrieve top-k samples from memories compatible with query context
```

The key is to retain the useful prediction-balance idea from Ramen while making domain/context consistency explicit rather than relying entirely on raw feature distance.

---

## 6. Research axis 2: realistic heterogeneous and non-stationary streams

The new method needs a stronger evaluation protocol than random fully-interleaved mixtures.

### 6.1 Stream types

Implement deterministic stream generators for at least:

| Stream | Description | Research question |
|---|---|---|
| `iid_mixed` | Current random mixture | Can we reproduce Ramen? |
| `block` | Domain persists for a block, then switches | Can the method detect sudden changes? |
| `gradual` | Mixture ratio changes smoothly | Can routing track transitions? |
| `recurring` | A -> B -> C -> A | Does memory reuse help after recurrence? |
| `imbalanced` | Long-tail domain proportions | Does a dominant domain overwrite minority contexts? |
| `novel_domain` | New domain appears midway | Can the router spawn new structure? |
| `class_domain_correlated` | Classes have different domain frequencies | Can the method separate semantic and domain structure? |
| `bursty` | Short local domain bursts | How quickly can adaptation react? |

### 6.2 Important constraint

Ground-truth `domain_idx` may be used for **evaluation only**, not routing or adaptation.

This allows measurements such as:

- Adjusted Rand Index (ARI);
- Normalized Mutual Information (NMI);
- context purity;
- domain routing accuracy after optimal cluster matching;
- number of discovered contexts versus actual environments.

These metrics should be secondary to the real objective: adaptation accuracy and stability.

---

## 7. Research axis 3: reliable structured memory

Once latent context routing works, the next major failure mode is memory contamination.

### 7.1 Reliability-aware admission

Instead of unconditional insertion, use a reliability score from signals such as:

- normalized entropy;
- neighbor agreement;
- temporal consistency;
- augmentation consistency;
- gradient agreement.

Then choose one of:

```text
low reliability    -> do not store
medium reliability -> store with low weight
high reliability   -> store normally
```

### 7.2 Soft class memory

Hard pseudo-label assignment can be replaced by soft class membership using the current class probability distribution. This avoids a single incorrect `argmax` deciding the complete memory destination.

### 7.3 Memory repair

Potential operations:

- relabel an item if neighborhood consensus changes;
- lower the reliability of stale inconsistent items;
- evict items whose gradients repeatedly conflict with useful updates;
- merge duplicate prototypes;
- forget contexts that disappear for long periods;
- retain compact prototypes for recurring contexts.

This creates a second independent paper-level question:

> **How can online TTA memory remain trustworthy when its supervision is generated by the adapting model itself?**

---

## 8. Research axis 4: scalable gradient memory

The existing design stores a full gradient vector for each memory item. This is elegant and fast for aggregation, but expensive as model size, vocabulary size, or cache capacity grows.

### 8.1 Low-rank gradient representation

Approximate each gradient in a low-dimensional basis and store only coefficients. Research questions include whether the basis can be fixed, updated online, and whether sign preservation matters more than exact L2 reconstruction under SignSGD.

### 8.2 Gradient prototypes

Store one or a few representative gradients per `(class, context)`, including a possible weighted sign-vote representation aligned with SignSGD.

### 8.3 Hierarchical retrieval

Avoid querying all class caches:

```text
query
  -> retrieve top candidate contexts
  -> retrieve top candidate classes/prototypes inside contexts
  -> retrieve only local samples
```

This can reduce both search complexity and cross-context interference.

---

## 9. Research axis 5: theory

The strongest theoretical direction is not to reproduce the existing one-step simplified analysis. Instead, theory should target the structural decisions introduced by the thesis.

Possible questions:

1. Under what assumptions does support selected from the same latent context reduce harmful gradient variance?
2. When does class balancing reduce semantic collapse under entropy minimization?
3. How does pseudo-label noise propagate through a persistent gradient memory?
4. What admission error rate can the memory tolerate before adaptation becomes worse than no adaptation?
5. Can online routing/adaptation be analyzed using regret or stability bounds under piecewise-stationary streams?
6. Can gradient alignment provide a sufficient condition for a support sample to improve the query loss locally?

A useful theoretical target would connect:

```text
context purity
    -> gradient compatibility
    -> adaptation stability / error
```

---

## 10. Proposed architecture for the first thesis prototype

```text
                        +----------------------+
image x_t ------------> | CLIP visual encoder  |
                        +----------+-----------+
                                   |
                                   v
                              feature z_t
                                   |
                 +-----------------+-----------------+
                 |                                   |
                 v                                   v
          zero-shot classifier                 LatentRouter
                 |                              q(d | z_t)
                 v                                   |
         class distribution p_t                       |
                 |                                   |
                 +-----------------+-----------------+
                                   v
                         ReliabilityEstimator
                                   |
                                   v
                    StructuredGradientMemory
                         M[class][context]
                                   |
                                   v
                     Compatibility Retrieval
                  feature + context + gradient
                                   |
                                   v
                       weighted gradient vote
                                   |
                                   v
                       temporary TTA update
                                   |
                                   v
                              prediction
                                   |
                                   v
                       reset adapted params

Memory/router state persists across the stream.
```

### Recommended initial modules

```text
src/
  methods/
    Ramen.py
    LatentRamen.py

  memory/
    structured_memory.py
    admission.py
    retrieval.py

  routing/
    online_prototypes.py
    change_detection.py

  streams/
    builders.py
    schedules.py

  evaluation/
    online_metrics.py
```

The exact folder organization can evolve, but research logic should not remain embedded in one large `Ramen.py` file.

---

## 11. Engineering plan: lowest-risk implementation order

### Phase 0 — Reproducibility baseline

Goal: verify the fork can reproduce the existing Ramen protocol before research changes.

Tasks:

- freeze exact environment and package versions;
- reproduce at least one CIFAR corruption experiment and one DomainNet experiment;
- record seed, GPU, runtime, peak memory, accuracy by domain, and total accuracy;
- add deterministic experiment manifests;
- keep original `Ramen` untouched as the reference baseline.

Exit criterion:

> Ramen baseline is reproducible within an agreed tolerance across repeated runs.

### Phase 1 — Stream benchmark infrastructure

Modify evaluation before modifying the method.

Required stream builder API:

```python
stream = build_stream(
    datasets,
    mode="block",
    seed=0,
    domain_weights=None,
    block_size=...,
)
```

Log for every time step or evaluation window:

```text
timestep
sample_idx
ground_truth_domain   # evaluation only
ground_truth_class
prediction
correct
predicted_entropy
inferred_context
memory_size
num_active_contexts
latency_ms
```

Exit criterion:

> Original Ramen can be evaluated under at least IID-mixed, block, gradual, recurring, and imbalanced streams without changing the TTA algorithm itself.

### Phase 2 — Minimal `LatentRamen`

Implement:

- online prototype router;
- hard context assignment first;
- memory indexed by `(predicted_class, context)`;
- context-aware retrieval;
- original entropy and distance weighting;
- no learned neural router yet.

Key comparison:

```text
Ramen raw feature kNN
vs.
LatentRamen prototype context + class balance
```

Exit criterion:

> Improvement is concentrated on structured non-stationary streams and does not materially regress the original IID mixed setting.

### Phase 3 — Reliability-aware memory

Add one mechanism at a time:

1. entropy admission threshold;
2. neighbor label agreement;
3. gradient sign/alignment agreement;
4. soft class assignment;
5. relabel/forget mechanisms.

Avoid introducing all reliability signals simultaneously before ablation evidence exists.

### Phase 4 — Scalability

Measure:

- memory in GB;
- retrieval latency;
- adaptation latency;
- number of stored gradients;
- number of contexts;
- accuracy-memory Pareto frontier.

Then implement gradient prototypes or low-rank/sign sketches.

---

## 12. Experiment matrix

### 12.1 Core datasets

Keep the original corruption datasets for comparability, but increase emphasis on natural multi-environment datasets.

Priority order:

1. DomainNet
2. OfficeHome
3. PACS
4. VLCS
5. TerraIncognita
6. CIFAR-10-C / CIFAR-100-C / ImageNet-C for corruption stress tests

The natural-domain datasets are especially important because the current Ramen advantage on DomainNet is smaller than on corruption mixtures.

### 12.2 Baselines

At minimum:

- No adaptation / zero-shot CLIP;
- Tent;
- original Ramen;
- random memory support;
- same-class-only support;
- global nearest-neighbor support;
- context-only support without class balance;
- class-balanced support without context routing;
- oracle-domain support as an upper-bound diagnostic.

For a publication-quality study, integrate stronger recent VLM-TTA and mixed-domain TTA baselines separately. The current repository only ships a small subset (`NoAdapt`, `Tent`, `Ramen`), so baseline integration is a required engineering workstream rather than something already solved by this fork.

### 12.3 Metrics

Primary:

- average accuracy;
- worst-domain accuracy;
- online/sliding-window accuracy;
- post-shift recovery time;
- negative adaptation rate: fraction of windows where adaptation is worse than zero-shot.

Routing diagnostics:

- ARI;
- NMI;
- context purity;
- number of contexts;
- context churn / assignment stability.

Reliability diagnostics:

- cache pseudo-label accuracy;
- contamination rate;
- fraction of rejected samples;
- gradient agreement of selected supports;
- accuracy versus memory reliability threshold.

Efficiency:

- peak GPU memory;
- average latency per sample/batch;
- retrieval latency;
- stored bytes per memory item;
- throughput;
- accuracy-memory-latency Pareto curves.

---

## 13. Critical ablations

A future paper should make the causal story much tighter than “all components help.”

1. **Self-gradient vs historical memory**
   - query only;
   - historical support only;
   - query + history.

2. **Semantic vs context structure**
   - class balance only;
   - context consistency only;
   - class + context.

3. **Raw similarity vs structured compatibility**
   - feature distance;
   - context posterior;
   - gradient alignment;
   - combinations.

4. **Hard vs soft assignment**
   - hard pseudo-class;
   - soft pseudo-class;
   - hard context;
   - soft context.

5. **Memory contamination**
   - clean oracle memory;
   - natural pseudo-label memory;
   - synthetically injected error rates.

6. **Stream dynamics**
   - IID mixed;
   - sudden shift;
   - gradual shift;
   - recurring shift;
   - long-tail domain mixture.

7. **Model state semantics**
   - reset parameters after each prediction, persistent memory;
   - persistent parameters with safeguards;
   - periodic reset;
   - context-specific lightweight parameter state.

8. **Scalability**
   - full gradients;
   - gradient prototypes;
   - low-rank/sketched gradients;
   - sign-only representations.

---

## 14. Falsifiable hypotheses

### H1 — Latent context helps under structured streams

> Explicit online context inference improves Ramen mainly when the test stream has persistence, recurrence, imbalance, or gradual change; gains should be smaller under fully IID random mixing.

If this fails, latent routing may be unnecessary and raw local feature retrieval may already capture sufficient structure.

### H2 — Context purity predicts adaptation quality

> Higher routing purity should correlate with higher selected-gradient compatibility and better post-adaptation accuracy.

If routing purity improves without accuracy gains, the discovered domains may be visually meaningful but adaptation-irrelevant.

### H3 — Reliable memory matters more under severe shifts

> Admission/reliability mechanisms should help most when zero-shot pseudo-label error is high.

If gains occur only when pseudo-labels are already accurate, the mechanism is unlikely to solve the real contamination problem.

### H4 — Gradient compatibility is more useful than feature similarity alone

> Within a context/class candidate set, support chosen using gradient alignment should produce more stable SignSGD updates than pure feature-distance ranking.

### H5 — Structured compression can preserve useful update directions

> Gradient prototypes/sign sketches can substantially reduce memory while retaining most of the adaptation gain, because exact full-gradient reconstruction is not necessary for sign-based updates.

---

## 15. Risks and failure modes

### Risk 1 — Latent contexts merely rediscover classes

CLIP embeddings strongly encode semantics. A naive feature clustering method may cluster by class rather than domain.

Mitigations:

- route using class-conditioned residual features;
- remove text-class direction from embeddings;
- cluster within predicted class;
- use augmentation/style statistics;
- evaluate class/context mutual information separately.

### Risk 2 — More structure increases fragmentation

A `(class, context)` memory may become sparse, especially for many classes.

Mitigations:

- soft routing;
- hierarchical backoff from `(class, context)` to `context` or `class` memory;
- shared prototypes;
- dynamic context merging.

### Risk 3 — Router drift feeds memory drift

Incorrect context assignments can contaminate the router itself.

Mitigations:

- confidence-weighted prototype updates;
- slow/fast prototype copies;
- change-point thresholds;
- delayed assignment;
- memory replay for recurring contexts.

### Risk 4 — Better domain clustering does not imply better TTA

Ground-truth domain labels are annotations, not necessarily the optimal partition for adaptation.

Mitigation:

> Treat domain-label agreement as a diagnostic, not the training target. Optimize and evaluate adaptation compatibility directly.

### Risk 5 — Gains come from extra compute

A more expensive router/retriever can win unfairly.

Mitigation:

- report equal-memory and equal-latency comparisons;
- include accuracy/latency/memory Pareto frontiers;
- compare against stronger kNN baselines with the same retrieval budget.

---

## 16. Paper/chapter roadmap

### Chapter / Paper 1 — Ramen baseline

**Question:** Can mixed-domain VLM TTA use sample-specific class-balanced local support efficiently?

Role in thesis: establishes the foundation and gradient-memory mechanism.

### Chapter / Paper 2 — Latent Context-Aware TTA

**Question:** Can an online model discover adaptation-relevant latent structure in non-stationary deployment streams?

Expected contribution:

- non-stationary mixed-domain benchmark protocol;
- online latent router;
- structured `(class, context)` memory;
- context-aware support selection;
- diagnostics linking routing and adaptation quality.

This should be the priority next paper.

### Chapter / Paper 3 — Reliable Self-Supervised TTA Memory

**Question:** How can an online adaptation memory remain reliable when its labels and gradients are generated by the adapting model?

Expected contribution:

- reliability-aware admission;
- soft/revisable pseudo-labels;
- gradient-consistency filtering;
- memory repair/forgetting;
- contamination stress tests.

### Chapter / Paper 4 — Scalable Gradient-Memory TTA

**Question:** How can structured TTA memory scale to large vocabularies and larger VLMs?

Expected contribution:

- compressed/prototype/sign gradient memory;
- hierarchical retrieval;
- memory/latency bounds or complexity analysis;
- large-class experiments.

### Thesis synthesis

A coherent final claim becomes:

> **Reliable TTA under heterogeneous deployment requires jointly solving support relevance, latent context, memory trustworthiness, and computational scalability.**

---

## 17. Recommended immediate next milestone

The highest-information, lowest-risk next experiment is **not** a learned neural retrieval model. It is:

> **Online prototype context discovery + deterministic non-stationary stream evaluation, while keeping Ramen's existing per-sample gradient cache unchanged.**

Implementation target:

```text
Ramen
  + StreamBuilder
  + OnlinePrototypeRouter
  + StructuredMemory[class][context]
  = LatentRamen-v0
```

First experimental grid:

```text
Datasets:
  DomainNet
  CIFAR100C

Streams:
  iid_mixed
  block
  gradual
  recurring
  imbalanced

Methods:
  NoAdapt
  Tent
  Ramen
  Ramen + oracle domain routing
  LatentRamen-v0

Metrics:
  average accuracy
  worst-domain accuracy
  sliding-window accuracy
  recovery time after shift
  NMI/ARI of inferred contexts
  peak memory
  latency
```

The oracle-domain variant is particularly important. If oracle routing does not improve over Ramen under structured streams, then latent-domain discovery is unlikely to be the right bottleneck. If oracle routing gives a clear upper-bound gain and the prototype router closes part of that gap, the thesis direction has strong empirical support.

---

## 18. Decision criteria before investing in a full thesis direction

Continue aggressively with latent-structure TTA if the following pattern appears:

1. Original Ramen degrades under at least some realistic non-stationary streams relative to its IID-mixed result.
2. Oracle context/domain routing recovers a meaningful part of that degradation.
3. An unsupervised lightweight router recovers a non-trivial fraction of the oracle gain.
4. The gain persists on a natural-domain dataset such as DomainNet/OfficeHome/PACS, not only synthetic corruptions.
5. Improvements are not explained only by increased memory or compute.
6. Routing diagnostics correlate with adaptation quality or gradient compatibility.

Reconsider the direction if:

- oracle routing gives almost no gain;
- the router mainly clusters by class;
- non-stationary streams do not expose a meaningful weakness in Ramen;
- improvements disappear under equal-compute comparison.

In that case, the next-best thesis axis is **reliable memory under pseudo-label error**, which is also directly motivated by the current implementation.

---

## 19. Working conclusion

The current NB-Ramen codebase is suitable as a thesis foundation because the core research surface is modular enough to extend without replacing the most complex gradient machinery. The strongest gap is not simply final accuracy. It is the assumption that a randomly interleaved mixed stream can be handled using predicted-class queues plus raw feature proximity.

The recommended research program reframes Ramen around a more general principle:

> **Test-time adaptation is a support-selection problem under uncertainty and non-stationarity.**

The next implementation should therefore focus on **stream structure + latent context routing**, then add **memory reliability**, followed by **scalability**. This produces a coherent sequence of falsifiable research questions rather than a collection of independent engineering tweaks.
