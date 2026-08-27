# CausalRamen Completion Report

## 1. Objective

The purpose of `CausalRamen` is to answer one narrow research question before extending Ramen further:

> **Does enforcing strict online causality improve Ramen, or are the observed gains caused by other implementation differences?**

The Ramen paper describes the support set for a test sample as being constructed from previously seen samples. Conceptually, the procedure is:

```text
current sample
  -> compute embedding / pseudo-label / gradient
  -> update memory
  -> retrieve support
  -> aggregate cached gradients
  -> temporary adaptation
  -> inference
  -> reset model parameters
```

However, the released implementation computes gradients for an entire evaluator batch and inserts the whole batch into memory before issuing retrieval queries. For a batch

\[
[x_1,x_2,\ldots,x_B],
\]

the prediction for `x1` can therefore use cached information from `x2 ... xB`.

This is valid under a **batch-arrival protocol**, but not under a **strict sample-by-sample online protocol**.

`CausalRamen` establishes the latter:

```text
x1 -> compute -> insert -> retrieve -> adapt -> predict
x2 -> compute -> insert -> retrieve -> adapt -> predict
x3 -> ...
```

No future item in the evaluator order is visible to an earlier query.

---

## 2. Why CausalRamen became necessary

The initial thesis direction introduced `LatentRamen`, which added online context discovery and memory indexed by predicted class and latent context.

Early pilots produced an apparent accuracy improvement over legacy Ramen. However, the latent router collapsed to one context.

Therefore:

```text
LatentRamen
    |
    +-- inferred contexts = 1
    |
    +-- effectively behaves like
        causal class-balanced structured Ramen
```

The dedicated paired pilot subsequently showed that `LatentRamen` and `CausalRamen` produce identical projected behavior on the stronger tested cells.

Before attributing any gain to latent routing, the project therefore required a no-routing causal control.

That control is `CausalRamen`.

Primary evidence report:

- [`plans/20260824-latent-ramen-evidence/reports/causal-ramen-mps-paired-pilot.md`](../../plans/20260824-latent-ramen-evidence/reports/causal-ramen-mps-paired-pilot.md)

---

## 3. Current implementation

`CausalRamen` is implemented through:

```text
src/methods/SupportAblations.py
```

Its intended behavior is:

```text
query x_t
    |
    v
CLIP embedding z_t
prediction c_t
entropy H_t
gradient g_t
    |
    v
insert only x_t
    |
    v
retrieve class-balanced historical/current support
    |
    v
entropy + similarity weighted aggregation
    |
    v
temporary SignSGD update
    |
    v
prediction
    |
    v
reset model parameters

memory persists
```

The method uses one fixed context:

\[
d_t=0,
\]

so there is no latent-routing mechanism.

Its role is specifically:

> **class-balanced Ramen under strict stream causality.**

---

## 4. Current evidence

### 4.1 Three-seed 64-sample pilot

Protocol:

```text
Dataset: CIFAR-100-C
Backbone: CLIP ViT-B/16
Stream: block
stream_block_size: 8
Prefix: 64 samples
Seeds: 0, 1, 2
Evaluator batch size: 100
```

Mean micro accuracy:

| Method | Micro accuracy |
|---|---:|
| NoAdapt | 34.90% |
| Legacy Ramen | 33.85% |
| CausalRamen | **36.98%** |
| LatentRamen | **36.98%** |

Observed difference:

\[
\Delta_{\text{Causal-Legacy}}
=
36.98-33.85
=
+3.13\text{ pp}.
\]

CausalRamen also had zero negative-adaptation windows in all three seeds, while legacy Ramen had one negative window in seed 1.

This is encouraging but not publication-level evidence because each run contains only 64 samples.

### 4.2 Canonical-block 200-sample pilot

Protocol:

```text
Dataset: CIFAR-100-C
Backbone: CLIP ViT-B/16
Seed: 0
Stream: block
stream_block_size: 64
Prefix: 200 samples
Batch size: 100
```

Results:

| Method | Micro | Macro | Worst-domain | Negative windows |
|---|---:|---:|---:|---:|
| NoAdapt | 30.5% | 32.03% | 23.44% | reference |
| Legacy Ramen | 31.5% | 38.28% | 28.12% | 3/4 |
| CausalRamen | **33.0%** | **39.45%** | **28.12%** | **1/4** |
| LatentRamen | **33.0%** | **39.45%** | **28.12%** | **1/4** |

Thus:

\[
\Delta_{\text{Causal-Legacy}}=+1.5\text{ pp micro}.
\]

Again, CausalRamen and LatentRamen match.

### 4.3 2026-08-27 scheduling-isolation pilot (CPU/MPS)

The results above are historical bounded pilots. The strictly validated
completion runtime pass on 2026-08-27 added the same-memory scheduling
control, `StructuredAtomicRamen`, plus CPU and Apple-MPS checks. The complete
execution record is in the [local runtime and causal pilot report](../../plans/20260827-causal-ramen-completion/reports/local-runtime-and-causal-pilot.md); the canonical sensitivity result is [post-fix-mps-v2-batch-sensitivity.json](../../plans/20260827-causal-ramen-completion/reports/post-fix-mps-v2-batch-sensitivity.json).

CPU B=1 and B=4 mechanics evidence completed and validated after the reviewed
CPU half-precision fix. The strictly paired accuracy deltas between
`StructuredAtomicRamen` and `CausalRamen` were zero for both cells. Those
four-sample CPU cells establish local mechanics only, not a scientific effect.

The informative bounded scheduling test was CIFAR-100-C, block stream,
`n=64`, seed 0, block size 8, on Apple MPS. Every cell in the canonical
sensitivity set was strictly validated. `CausalRamen - StructuredAtomicRamen`
micro-accuracy deltas were:

| Evaluator batch size | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Micro delta | 0 | 0 | -0.015625 | -0.015625 | -0.015625 | -0.03125 | 0 |

Across those seven cells, the mean micro delta was `-0.0111607` (standard
deviation `0.0118114`) and the mean worst-domain delta was `-0.0535714`.
There was no negative-adaptation-rate difference in any cell. At B=1, the
Legacy-Ramen-versus-CausalRamen micro-accuracy delta was also `0` in this
bounded cell.

This is a seed-0, single-stream, 64-sample PILOT. Its Apple-MPS latency values
are descriptive only and do not support CUDA efficiency conclusions. CUDA and
DomainNet were unavailable for this pass.

---

## 5. What has been established

The historical pilots support the limited claim:

> **The strict-causal structured implementation matches or exceeds legacy batch-atomic Ramen on the tested bounded CIFAR-100-C prefixes.**

It also establishes, on these tested cells:

\[
\boxed{
\text{current LatentRamen gain is not caused by latent routing}
}
\]

because the router discovers one context and the projected behavior matches CausalRamen.

This is an important scientific correction to the original latent-routing direction.

The 2026-08-27 completion pass additionally establishes the implementation
control and a bounded B=1 diagnostic: `StructuredAtomicRamen` and
`CausalRamen` have an observed zero micro-accuracy difference at B=1, and the
same-memory scheduling comparison was completed across B=1, 2, 5, 10, 20, 50,
and 100 for seed 0/block/n=64. It does **not** establish a positive scheduling
gain: that signal did not survive this batch-size sensitivity check.

---

## 6. What has NOT yet been established

The following claim is **not yet justified**:

> “Strict causality alone improves Ramen by 1–3 percentage points.”

There is still a confound.

### Legacy Ramen

```text
PriorityCache
half-precision distance behavior
original top-k / tie handling
batch-atomic memory update
```

### CausalRamen

```text
StructuredGradientMemory
float32 ranking metadata
stable sorting
strict sequential memory update
explicit causal diagnostics
```

Therefore:

\[
\text{CausalRamen}-\text{Ramen}
\]

currently measures a combination of:

\[
\text{causal scheduling}
+
\text{memory implementation}
+
\text{numerical/ranking effects}.
\]

The Legacy-versus-Causal comparison therefore remains confounded. The
same-memory `StructuredAtomicRamen` control now isolates scheduling on the
bounded pilot; its result is reported in Section 4.3 and did not show a
positive scheduling gain.

---

# 7. Completion experiments

CausalRamen should not be considered scientifically complete until the following experiments are finished.

## Experiment A — Batch-size-1 diagnostic

### Purpose

Remove future-within-batch visibility from legacy Ramen without modifying its implementation.

When:

\[
B=1,
\]

legacy Ramen cannot see a later item inside the current batch.

Therefore both methods have effectively the same temporal information set.

### Comparison

```text
Legacy Ramen, batch_size=1
vs
CausalRamen, batch_size=1
```

Keep identical:

```text
dataset
stream
sample order
seed
hyperparameters
memory capacity
top-k
beta
learning rate
```

### Interpretation

#### Outcome A1

\[
Ramen_{B=1}\approx CausalRamen_{B=1}
\]

but at larger batch size:

\[
CausalRamen>Ramen.
\]

This strongly supports the hypothesis that future-within-batch visibility is responsible for the difference.

#### Outcome A2

\[
Ramen_{B=1}\neq CausalRamen_{B=1}.
\]

Then implementation/numerical differences remain important and causality is not isolated.

### Minimum pilot

```text
CIFAR-100-C
block stream
n = 64 or 128
seed = 0
```

This should be performed before an expensive DomainNet experiment.

---

## Experiment B — StructuredAtomicRamen

This is the same-memory implementation control used in the 2026-08-27 pilot.
Its original required design was:

```text
StructuredAtomicRamen
```

with exactly the same:

```text
StructuredGradientMemory
dtype
ranking
stable sort
retrieval implementation
aggregation
optimizer
```

as CausalRamen.

The only intended difference should be scheduling.

### StructuredAtomicRamen

```text
batch arrives

insert x1 ... xB
        |
        v
query x1 ... xB
```

### CausalRamen

```text
insert x1 -> query x1
insert x2 -> query x2
...
```

Then:

\[
\boxed{
StructuredAtomicRamen
\quad vs \quad
CausalRamen
}
\]

becomes the primary scheduling ablation.

---

## 8. Ideal 2×2 design

For a clean causal analysis:

| Memory implementation | Batch-atomic | Strict causal |
|---|---|---|
| PriorityCache | Legacy Ramen | PriorityCausalRamen |
| StructuredGradientMemory | StructuredAtomicRamen | CausalRamen |

The minimum required pair is:

```text
StructuredAtomicRamen
vs
CausalRamen
```

The full 2×2 is optional if implementation cost is small.

This allows decomposition into:

\[
\text{scheduling effect}
\]

and:

\[
\text{memory implementation effect}.
\]

---

# 9. Batch-size sensitivity experiment

The Ramen paper reports performance as largely stable across test-time batch sizes. The causal analysis should explicitly examine:

```text
B = 1
B = 2
B = 5
B = 10
B = 20
B = 50
B = 100
```

Compare:

```text
Legacy Ramen
CausalRamen
```

Define:

\[
\Delta(B)
=
Acc_{Causal}(B)-Acc_{Legacy}(B).
\]

If future-within-batch information explains the difference, a plausible pattern is:

\[
\Delta(1)\approx0
\]

with larger divergence as `B` increases.

This would provide a stronger mechanistic result than one aggregate accuracy comparison.

---

# 10. Strict-online versus batch-online terminology

Do not describe legacy Ramen as “data leakage” without qualification.

### Batch-online protocol

A batch arrives jointly:

\[
\{x_t,\ldots,x_{t+B-1}\}.
\]

All batch samples may be available before their individual predictions.

### Strict-online protocol

Samples arrive sequentially:

\[
x_t
\rightarrow
\hat y_t
\rightarrow
x_{t+1}.
\]

Future samples must not affect the prediction of `x_t`.

The research question should therefore be phrased as:

> **How does Ramen behave when converted from a batch-atomic TTA protocol to a strictly causal sample-stream protocol?**

---

# 11. Required metrics

Every paired cell should report at least:

### Accuracy

```text
micro accuracy
macro-domain accuracy
worst-domain accuracy
```

### Stability

```text
negative-adaptation windows
recovery time after shifts
```

### Memory

```text
retained item count
retained bytes
```

### Efficiency

```text
forward latency
retrieval latency if profiling enabled
throughput
```

MPS efficiency remains descriptive only. CUDA should be used for final latency conclusions.

---

# 12. Required stream settings

After causal attribution is isolated on a small pilot, evaluate:

```text
iid_mixed
block
gradual
recurring
imbalanced
```

The main question is not necessarily whether CausalRamen wins everywhere, but **where strict causality matters**.

---

# 13. Dataset progression

Do not begin with the complete expensive grid.

Recommended order:

```text
1. CIFAR-100-C short paired pilot
       ↓
2. CIFAR-100-C full stream
       ↓
3. small natural-domain pilot
       ↓
4. DomainNet full stream
```

DomainNet is important because it tests natural domain styles rather than only corruption types.

---

# 14. Multi-seed requirements

Pilot:

```text
seed = 0
```

Mechanism validation:

```text
seed = 0,1,2
```

Final comparison:

```text
>= 3 fixed seeds
```

Do not select new seeds after inspecting results.

All paired methods must use identical stream fingerprints within each experimental cell.

---

# 15. Go / no-go criteria

CausalRamen should be promoted as a research contribution only if the scheduling-only comparison produces a repeatable effect.

## GO

Continue the causal-Ramen research line if:

\[
Acc(CausalRamen)
>
Acc(StructuredAtomicRamen)
\]

consistently across seeds and at least two meaningful stream conditions, with a practically relevant effect such as approximately:

\[
\ge1\text{ percentage point}
\]

on full or sufficiently large evaluations.

Additional supporting evidence:

```text
negative adaptation decreases
worst-domain accuracy does not systematically regress
effect increases with evaluator batch size
B=1 difference becomes small
```

## WEAK GO

If average accuracy gains are small but strict causal processing substantially improves:

```text
negative adaptation
post-shift failures
worst-case instability
```

then the contribution can be reframed around **online stability** rather than mean accuracy.

## NO-GO

Do not make causality a main thesis contribution if:

```text
StructuredAtomicRamen ~= CausalRamen
```

or the observed gain disappears on full-stream CUDA experiments.

In that case, retain CausalRamen as a protocol-correct control for subsequent research.

---

# 16. Relationship to ConsensusRamen

CausalRamen is not the final thesis mechanism.

Its role is to establish:

\[
\boxed{
\text{which evidence is temporally allowed}
}
\]

ConsensusRamen later addresses:

\[
\boxed{
\text{which allowed gradients should actually influence the update}
}
\]

Conceptual pipeline:

```text
test stream
    |
    v
strict causal support
    |
    v
Ramen retrieval
    |
    v
gradient compatibility / consensus
    |
    v
safe adaptation
```

The causal protocol should therefore be resolved before evaluating gradient consensus.

Otherwise a future sample could participate in the consensus vote for an earlier query, making the experiment difficult to interpret.

---

# 17. Completion sequence and current gate

The implementation/control work listed below is complete. The dated
scheduling-isolation evidence now gates, rather than automatically triggers,
the expensive stages:

1. [x] Run the bounded `Ramen`/`CausalRamen` B=1 diagnostic.
2. [x] Implement `StructuredAtomicRamen` and its configuration-equivalence controls.
3. [x] Run the paired n=64 seed-0 block pilot and batch-size sensitivity.
4. [ ] Escalate to three seeds and full CIFAR-100-C/CUDA only if a revised or independently justified gate warrants it.
5. [ ] Run a bounded natural-domain pilot and DomainNet only after that escalation decision and a verified NVIDIA/DomainNet environment.
6. [ ] Freeze a publication-level causal conclusion after the required coverage exists.

---

# 18. Required tests

At minimum:

### Causality

```text
sample i cannot retrieve item j when j > i
```

### Atomic control

```text
StructuredAtomicRamen can retrieve later items inside the same evaluator batch
```

### Equivalent configuration

Assert identical:

```text
capacity
topk
beta
optimizer
learning rate
dtype
retrieval implementation
aggregation
include_current
```

between StructuredAtomicRamen and CausalRamen.

### B=1 equivalence under the declared paired configuration

On a deterministic synthetic fixture using the preregistered
`include_current=true` configuration:

```text
StructuredAtomicRamen(B=1)
==
CausalRamen(B=1)
```

for:

```text
retrieved support IDs
aggregated gradients
memory state
predictions
```

up to the defined numerical tolerance.

This is one of the strongest correctness tests for the causal experiment.
With historical-only retrieval (`include_current=false`), inserting before
querying can evict history at full capacity even when `B=1`; that scheduling
edge case is expected to differ and must be tested separately rather than
silently treated as equivalent.

---

# 19. Exit criteria for the CausalRamen phase

The CausalRamen investigation is complete when all of the following are answered:

```text
[x] Does B=1 remove the legacy-vs-causal difference on the bounded seed-0 cell? (Observed micro delta: 0.)

[x] Does a same-memory atomic control isolate a scheduling effect on the bounded seed-0 pilot? (It isolates the comparison; no positive scheduling gain survived sensitivity.)

[ ] Does the effect repeat across >=3 seeds?

[ ] Does the effect persist on a full CIFAR-100-C stream?

[ ] Does it generalize to at least one natural-domain dataset?

[x] Does strict causality improve accuracy, stability, or both on the bounded scheduling pilot? (No positive accuracy or negative-adaptation effect was observed.)

[x] Is the bounded scheduling effect sufficiently large to be a contribution? (No; this does not decide the publication-level question.)

[x] Can later methods use CausalRamen as the canonical strict-online baseline? (As an implementation/control baseline, not a promoted causal-gain result.)
```

Until these are answered, describe CausalRamen as:

> **a strict-online implementation/control baseline with historical positive pilots but no positive scheduling-only gain in the dated bounded sensitivity pilot; not yet a standalone method contribution.**

---

# 20. Current decision

Based on the historical pilots and the strictly validated 2026-08-27
implementation/control pass:

\[
\boxed{
\text{CausalRamen is a usable strict-online control; its causal-gain claim is gated}
}
\]

because:

1. it explains the apparent gain previously attributed to LatentRamen;
2. historical bounded pilots contain positive micro/macro signals;
3. the same-memory atomic control and B=1 diagnostic are now implemented and validated;
4. it establishes a cleaner strict-online protocol for future memory research.

However:

\[
\boxed{
\text{the scheduling comparison is isolated, but no positive bounded scheduling effect was observed}
}
\]

because the same-memory control isolates scheduling, while the remaining
evidence is only a seed-0, n=64, single-stream MPS pilot.

The current bounded scheduling gate does not justify escalating to three seeds,
full CUDA, or DomainNet now: the positive scheduling gain did not survive the
seed-0 batch-size sensitivity analysis, the mean micro and worst-domain deltas
are negative, and no negative-adaptation advantage was observed. This is a
bounded `PILOT`/gate decision, **not** a final publication-level `NO_GO`.

CUDA and DomainNet remain unavailable, and MPS latency remains descriptive
only. CausalRamen can serve now as the strict-online experimental foundation
for later work; promotion as a causal-improvement contribution requires a
separate, justified decision to reopen and satisfy the remaining full-coverage
criteria.
