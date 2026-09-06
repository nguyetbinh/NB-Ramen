# Latent Soft Routing — Corrected Next Research Direction

**Branch:** `latent-soft-routing`  
**Based on HEAD:** `2fea7e8004c6531a5aed3d5bf2dd6d5a7e3c5b86`  
**Status:** executed on the canonical bounded cell; stop exact-domain soft routing as the primary axis; do **not** implement `LatentSoftRamen`

> **Outcome update (2026-09-04):** The preregistered margin profile and three
> calibrated cells described below are complete. All nonzero cells preserved
> diversity but changed zero predictions and no primary metric. The profile
> also showed that `gamma=0.25` exceeds every observed replacement margin and
> already saturates the small set of possible membership changes. The earlier
> interpretation of `gamma=0.25` as merely “too weak” is therefore superseded.
> See the [calibrated Gate 1 report](../../plans/20260904-latent-soft-routing/reports/calibrated-gate1-report.md).
>
> **Historical ablation update (2026-09-06):** A direct
> `LegacyLatentRamen` control now keeps legacy `PriorityCache`, float16
> retrieval, batch-atomic scheduling, weighting, and aggregation while adding
> only the original prototype-context eligibility rule. The router again
> discovered one context on the canonical 200-sample prefix. This establishes
> that router collapse is not caused by causal scheduling or structured
> memory. Small MPS accuracy differences are not attributed to routing because
> the eligibility intervention was the identity and repeated legacy MPS runs
> were not prediction-stable. See the
> [legacy latent ablation report](../../plans/20260904-latent-soft-routing/reports/legacy-latent-ablation-report.md).

---

## 1. Executive decision

The completed branch has established three useful facts:

1. **Hard domain/context routing is harmful in the tested cell** because it removes cross-domain support and sharply reduces support count, active-class coverage, and effective sample size.
2. **Exact-domain preference has very little support-set leverage in this
   bounded stream.** Only 64 replacement boundaries exist across 17,773
   support slots, and `gamma=0.25` already crosses all of them.
3. **No tested nonzero oracle-soft point has adaptation value here.** Weak,
   medium, strong, and saturated `gamma=0.25` change no predictions or primary
   metrics while preserving diversity.

Therefore the correct next step is **not**:

```text
increase the exact-domain bonus
```

and it is also **not**:

```text
implement LatentSoftRamen
```

The completed calibration rules out a useful interior point for this
mechanism in the canonical prefix. The next research step is:

> **Pivot to a richer continuous compatibility signal while preserving global class-balanced support.**

This isolates the scientific question:

> Can gradient, reliability, semantic, or temporal affinity influence enough
> support membership to improve adaptation without the diversity collapse
> caused by hard routing?

---

## 2. What the current evidence actually says

Canonical bounded cell:

```text
Dataset: CIFAR-100-C
Backbone: CLIP ViT-B/16
Stream: canonical block
Prefix: 200
Seed: 0
Device: MPS
Fingerprint:
aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b
```

Primary results:

| Method | Micro | Macro-domain | Worst-domain |
|---|---:|---:|---:|
| CausalRamen | 33.0% | 39.45% | 28.12% |
| OracleHardRamen | 30.5% | 37.50% | 21.88% |
| OracleSoftRankRamen, `gamma=0` | 33.0% | 39.45% | 28.12% |
| OracleSoftRankRamen, `gamma=0.25` | 33.0% | 39.45% | 28.12% |

Mechanistic diagnostics:

| Diagnostic | Causal / gamma=0 | Oracle hard | Soft gamma=0.25 |
|---|---:|---:|---:|
| Returned support, p50 | 93 | 31 | 93 |
| Active classes, p50 | 45 | 20 | 45 |
| Class coverage, p50 | 0.45 | 0.20 | 0.45 |
| ESS, p50 | 25.422 | 8.847 | 25.422 |
| Same-domain support, p50 | 0.3560 | 1.0000 | 0.3636 |
| Cross-domain support, p50 | 0.6440 | 0.0000 | 0.6364 |

The hard-routing mechanism is therefore clear:

\[
\text{domain purity}\uparrow
\Rightarrow
\text{support size}\downarrow,
\text{class coverage}\downarrow,
\text{ESS}\downarrow
\Rightarrow
\text{accuracy}\downarrow.
\]

This supports the correction from **domain exclusivity** to **domain preference**.

---

## 3. Why `gamma=0.25` is a saturated but low-leverage intervention

At `gamma=0.25`:

```text
queries with any positional slot change: 73 / 200
mean positional slot changes:           2.47%
mean rank displacement:          0.031
mean same-domain ratio:          49.17% -> 49.39%
mean ESS:                        23.829 -> 23.800
prediction changes:              0 / 200
```

The same-domain ratio increased by only:

\[
49.39 - 49.17 = 0.22\text{ percentage points}.
\]

The positional metric includes rank reorderings that do not change support-set
membership. Exact margin profiling found only 64 possible membership changes,
all with required bonus at most `0.12787533`. Therefore `gamma=0.25` already
saturates the actual membership intervention, even though the resulting
same-domain shift is small.

The correct interpretation is:

> Exact domain equality is too sparse a compatibility signal to alter much
> support membership in this bounded cell, and saturating it does not change
> predictions.

The following conclusion is **not** supported:

> Every possible continuous compatibility signal will fail.

---

## 4. Main methodological issue: absolute gamma has no calibrated meaning

Current oracle-soft ranking uses:

\[
\tilde d_{ij}
=
d_{ij}
-
\gamma\mathbf{1}[d_i=d_j].
\]

`gamma` matters only relative to the distance margin around the class-wise top-k boundary.

For query `i` and predicted class `c`, define the gamma-zero top-k threshold:

\[
t_{ic}=d^{(k)}_{ic}.
\]

For a same-domain candidate `j` currently outside top-k, define its replacement margin:

\[
m_{ij}=d_{ij}-t_{ic}.
\]

That candidate enters the top-k only when approximately:

\[
\gamma > m_{ij}.
\]

Therefore an absolute value such as `0.25` is scientifically uninterpretable until compared with the empirical margin distribution.

---

## 5. Correct next experiment: margin-calibrated oracle soft routing

### Phase A — retrieval-margin profiling

Do **not** perform model adaptation sweeps first.

Using the existing canonical stream and the same memory semantics, collect the gamma-zero retrieval margin distribution.

For each query and predicted class:

1. obtain all eligible candidates from the global per-class pool;
2. compute original feature distances;
3. identify the gamma-zero top-k threshold;
4. find same-domain candidates just outside top-k;
5. record the bonus required for each to enter top-k.

Persist at least:

```text
replacement_margin_p10
replacement_margin_p25
replacement_margin_p50
replacement_margin_p75
replacement_margin_p90
```

Also record per-query distributions because one global gamma may affect contexts unevenly.

### Phase B — define intervention strengths before looking at accuracy

Use the margin distribution to select three preregistered strengths.

Preferred formulation:

```text
weak    = margin quantile causing ~10% support-slot changes
medium  = margin quantile causing ~25% support-slot changes
strong  = margin quantile causing ~50% support-slot changes
```

Alternatively use quantiles directly:

\[
\gamma_{weak}=Q_{25}(m),
\quad
\gamma_{medium}=Q_{50}(m),
\quad
\gamma_{strong}=Q_{75}(m).
\]

The exact mapping must be fixed from retrieval margins, **not selected from test accuracy**.

---

## 6. Why intervention strength should be defined by retrieval change

The experiment is trying to estimate a response curve:

\[
\text{domain preference strength}
\rightarrow
\text{support composition}
\rightarrow
\text{gradient quality}
\rightarrow
\text{accuracy}.
\]

A useful intervention scale is therefore not raw `gamma`, but measurable changes such as:

```text
selection-change ratio
same-domain support ratio
cross-domain support ratio
rank displacement
```

Recommended target levels:

| Level | Approx. support-slot change |
|---|---:|
| baseline | 0% |
| weak | ~10% |
| medium | ~25% |
| strong | ~50% |
| hard oracle control | 100% domain exclusivity, not directly comparable |

This produces an interpretable experiment rather than a hyperparameter sweep.

---

## 7. Primary experiment matrix

Reuse the validated controls when possible.

### Existing reusable controls

```text
CausalRamen
OracleHardRamen
OracleSoftRankRamen gamma=0
```

They need not be rerun merely to reproduce already validated accuracy values.

Rerun them only if:

- retrieval implementation is refactored in a way that may change their trace;
- new diagnostics cannot be reconstructed from existing artifacts;
- the stream fingerprint or configuration changes;
- publication-quality pairing later requires all cells from the same code revision.

### New runs

```text
OracleSoftRankRamen — weak calibrated intervention
OracleSoftRankRamen — medium calibrated intervention
OracleSoftRankRamen — strong calibrated intervention
```

Do **not** implement `LatentSoftRamen` during this phase.

---

## 8. Required evidence per intervention

### Performance

```text
micro accuracy
macro-domain accuracy
worst-domain accuracy
negative-adaptation windows
```

### Retrieval composition

```text
returned support count
active-class count
class coverage
same-domain support ratio
cross-domain support ratio
selection-change ratio
mean rank displacement
```

### Gradient diversity / concentration

```text
ESS
```

If practical, also record:

```text
gradient norm
gradient sign agreement with gamma=0 aggregate
cosine similarity of aggregate gradient vs gamma=0
```

The latter metrics can establish whether changed support actually changes the adaptation direction rather than merely reshuffling near-equivalent samples.

---

## 9. Central hypothesis: the optimum should be intermediate

Current evidence gives two endpoints:

### Low preference

`CausalRamen / gamma=0`:

- high support diversity;
- substantial cross-domain support;
- better accuracy than hard oracle.

### Maximum preference

`OracleHardRamen`:

- 100% same-domain support;
- sharply reduced support count;
- sharply reduced class coverage;
- sharply reduced ESS;
- worse accuracy.

This motivates testing whether an interior optimum exists:

\[
0 < \gamma^* < \infty.
\]

Conceptually:

```text
no domain preference
        |
        v
moderate domain preference  <-- hypothesis: possible optimum
        |
        v
strong domain preference
        |
        v
hard same-domain exclusivity <-- observed harmful endpoint
```

The scientific question is therefore better formulated as:

> **What is the domain-consistency / support-diversity trade-off in class-balanced test-time adaptation?**

This is stronger than asking whether domain routing is simply good or bad.

---

## 10. Decision gates

### Gate A — intervention validity

Before interpreting accuracy, each nonzero point must produce a meaningful retrieval intervention.

Required:

```text
weak   -> clearly > current 2.47% mean slot change
medium -> materially changes support composition
strong -> large but non-degenerate change
```

If a selected gamma fails to create the intended support change, recalibrate from margins; do not interpret it as a negative accuracy result.

### Gate B — diversity preservation

Soft routing remains valid only while it preserves the main Ramen support mechanism.

Watch:

```text
class coverage
returned support count
ESS
```

If these collapse toward `OracleHardRamen`, the intervention is effectively becoming another hard filter.

### Gate C — oracle-soft value

Evidence for continuing the direction requires at least one calibrated nonzero intervention to show a positive signal over `CausalRamen` without diversity collapse.

A strong pattern would be:

```text
OracleHard < CausalRamen < OracleSoft-medium
```

with:

```text
OracleSoft-medium:
    same-domain ratio increased materially
    class coverage preserved
    ESS preserved reasonably
```

### Gate D — stop / pivot

Stop domain-based soft routing as the primary axis if:

1. weak / medium / strong calibrated oracle interventions all materially alter support;
2. class coverage and ESS remain healthy at at least one of them;
3. none provides a repeatable positive performance signal.

Only after this stronger oracle test fails should the project pivot to another compatibility signal such as:

```text
gradient agreement
uncertainty / reliability
semantic-class compatibility
temporal compatibility
```

---

## 11. When to build `LatentSoftRamen`

Do not build it merely because soft routing is conceptually cleaner.

Authorize latent routing only after oracle-soft evidence establishes:

\[
\text{context information has adaptation value when used softly}.
\]

Then the next problem becomes:

> Can an unsupervised router recover enough of that oracle-soft benefit?

At that point use continuous router outputs rather than hard context IDs where possible:

```text
prototype affinities
context posterior overlap
continuous context similarity
```

Possible compatibility:

\[
a_{ij}=q_i^\top q_j
\]

and ranking:

\[
s_{ij}
=
-d(z_i,z_j)+\gamma a_{ij}.
\]

This avoids recreating hard context partitions.

---

## 12. Do not tune router spawn threshold yet

The previous hard router collapsed to one discrete context, but that does not yet justify spawn-threshold tuning.

First determine whether **oracle context information itself** is useful under an adequately strong but soft intervention.

Only after oracle-soft passes should router research examine:

```text
prototype-distance distributions
posterior entropy
nearest/second-nearest margin
transition-time affinity shifts
correlation with ground-truth domains
```

The goal should be adaptation-compatible context affinity, not necessarily recovery of human-labelled domains.

---

## 13. Compute-minimal execution plan

### Step 1 — no expensive model run

Implement / log replacement-margin diagnostics using the same retrieval state.

Target output:

```text
margin distribution
estimated gamma for 10/25/50% retrieval-change targets
```

### Step 2 — only three new bounded model cells

Run:

```text
OracleSoft weak
OracleSoft medium
OracleSoft strong
```

on the existing canonical `n=200`, seed-0 stream.

Reuse validated baseline artifacts.

### Step 3 — decide before expanding compute

If there is no positive oracle-soft signal, stop.

If there is a meaningful positive signal, confirm with:

```text
3 seeds
one additional stream pattern
one natural-domain bounded pilot
```

Only then consider `LatentSoftRamen`.

---

## 14. Natural-domain requirement

CIFAR-100-C is appropriate for mechanism discovery, but a thesis claim about heterogeneous deployment streams ultimately requires natural-domain evidence.

If calibrated oracle-soft routing passes the bounded CIFAR gate, next use:

```text
PACS or OfficeHome bounded pilot
then DomainNet bounded pilot
```

Natural-domain evaluation should ask whether moderate context preference improves:

```text
accuracy
worst-domain accuracy
negative-adaptation windows
class coverage
ESS
recurring-domain recovery
```

Do not move directly to full DomainNet or ImageNet-C before the bounded oracle-soft mechanism is validated.

---

## 15. Correct research framing from this point

Do not frame the project as:

> Discover the domain and retrieve from that domain.

Use:

> **Estimate adaptation compatibility and softly bias support selection while preserving prediction balance and gradient diversity.**

The current domain label is an oracle diagnostic for one form of compatibility, not necessarily the final latent variable.

The larger thesis question is:

> **How should a model trade off local context consistency against support diversity when selecting historical evidence for test-time adaptation under heterogeneous streams?**

This formulation is consistent with all evidence collected so far:

- global mixed support can help;
- hard same-domain exclusivity can hurt;
- exact-domain soft preference has little membership leverage even at saturation;
- the calibrated intermediate regime is neutral in the bounded canonical cell.

---

## 16. Completed action and next decision

The branch now includes the gamma-zero replacement-margin diagnostic and a
two-stage runner that freezes Q25/Q50/Q75 strengths before executing exactly
three missing oracle-soft cells. The resulting response curve is:

\[
\boxed{
\text{context preference strength}
\rightarrow
\text{support composition}
\rightarrow
\text{ESS / class coverage}
\rightarrow
\text{accuracy}
}
\]

Weak, medium, strong, and the reused `gamma=0.25` point all preserve support
diversity, but all retain identical accuracy and predictions. Only 64
membership replacements are possible across 17,773 support slots, and the
maximum replacement margin is `0.12787533`; `gamma=0.25` already crosses all
of them. The positional `selection_change_ratio` is larger because it also
counts rank reorderings that do not change set membership.

Therefore the next action is to stop increasing the exact-domain bonus and
not implement a learned replica of it. Future research should pivot to a
richer continuous compatibility signal—such as gradient agreement,
reliability, semantic compatibility, or temporal affinity—under a separate
preregistered plan.
