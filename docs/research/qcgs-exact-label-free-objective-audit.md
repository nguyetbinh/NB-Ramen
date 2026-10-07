# Exact Label-Free Objective Audit for Ramen Support Actions

**Protocol status:** P2 scientific protocol frozen 2026-10-08; executable command, source/config hashes and safe action chunk size are frozen in the P3 launch lock before any scored query is evaluated.

**Question ID:** `CASE-QCGS-01`

**Protocol version:** `qcgs-exact-objective-audit-v1`

**Related synthesis:** [QCGS literature research synthesis](qcgs-literature-research-synthesis.md)

**Current plan and evidence:** [execution plan](../../plans/20261008-qcgs-literature-research-execution/plan.md)

## 1. Question and claim boundary

For an eligible CIFAR-100-C query in one fixed Ramen stream/cache state, do exact, label-free post-action objective changes rank the legal same-pseudo-class one-support substitutions by that same query's evaluator-only cross-entropy utility? The audit tests three predeclared objective formulations. It separates objective-to-utility alignment from approximation error: every legal action is executed exactly, so the result is an expensive information ceiling, not a deployable selector.

The candidate claim is a bounded diagnostic about within-query support actions. It is not a new general theory of action identifiability, a new selective-adaptation problem, evidence that a label-free selector is deployable, or a proof that other unlabeled channels cannot work. Prior reports are not recomputed here because the referenced raw archives are absent from this checkout; they motivate the question but do not count as new observations.

### Primary hypothesis

At least one frozen objective has positive per-query rank association with exact supervised CE utility in both cache-contamination cells, and a policy choosing its best legal action beats no-swap and uniform-random controls on fresh confirmation queries.

### Alternatives and interpretation

- If exact objective values track utility but the frozen Stage-A sample is too small or fails the investment gate, the result is inconclusive for confirmation; do not build an estimator from a post hoc variant.
- If no objective passes the exact information screen, stop these three formulations in this Ramen state. This does not establish that all label-free evidence is insufficient.
- A positive Stage-A result followed by a failed independent confirmation is a failed policy confirmation, not evidence of general non-identifiability.
- Selective adapt/skip and view consensus are established literature. If later reliability work opens, use the published CAS baseline and make no novelty claim for adapt/skip itself.

## 2. Frozen setting and baseline

| Item | Frozen value |
|---|---|
| Dataset/model | CIFAR-100-C, severity 5; OpenAI CLIP ViT-B/16 |
| Label space | `open-set-cifar100-split-v1`: 80 known classes and 20 held-out classes; only known-class queries are eligible for supervised utility |
| Stream cells | OOD stream fraction 0 and 0.5; seed 0; block mode, block size 64 |
| Data flow | 400 source samples/domain; batch size 100; full batch admitted before retrieval, including OOD rows, unscored rows and the current query as in Ramen |
| Memory/retrieval | Capacity 750 per pseudo-class; nearest `k=min(5, class_size)`; candidate pool `M=10`; `beta=5` |
| Ramen weights | `exp(-cached_entropy) * exp(-5 * distance)`; preserve class-local weights, active-class divisor, dtype and arithmetic order |
| Update | Reset from `theta0`; one state-free SignSGD step at learning rate 0.01, momentum 0, weight decay 0 |
| Candidate eligibility | At least one pseudo-class has 10 cached candidates; query is known-ID according to the locked registry split; no correctness/headroom filter |
| Retrieval ties | Preserve existing `torch.topk` membership behavior. The audit records the exact retrieved identities; it does not silently replace the retrieval algorithm with a new tie-breaker. |
| Runtime | Local Python 3.11 / PyTorch 2.6.0+cu124 / one RTX A2000 Laptop 8 GiB; record actual driver, package and device details in the launch receipt. This differs from prior T4/PyTorch 2.4.1+cu121 reports. |

Keep all Ramen behavior unchanged: prediction and pseudo-class are from the original-image `theta0` pass; cache admission and eviction, batch-atomic visibility, self-support, retrieval, support cardinality, class balance, aggregation, per-sample normalization, reset, output position and ordinary stream predictions remain baseline behavior. A trial changes only one support slot for one query, then applies the same single Ramen update. A trial never mutates cache or carries parameters into another action.

For each query, the action set is no-swap plus every legal same-pseudo-class swap from the frozen top-k/top-M pools. A swap replaces one of the `k` selected supports with one candidate ranked `k..M-1` from that same class; it preserves the number of supports. Enumerate in the existing canonical class/outgoing-rank/incoming-rank order. Use the incoming candidate's already-computed Ramen weight, exactly as the existing serial aggregate helper does. No supervised screening or action pruning is allowed.

## 3. Utility and objective contracts

Let `theta_R(q)` be the ordinary Ramen output for query `q` and `theta_a(q)` the output after action `a` from a clean reset. The evaluator-only utility is

\[
U(q,a)=CE_y(\theta_R(q),x_q)-CE_y(\theta_a(q),x_q),\qquad U(q,\mathrm{no\mbox{-}swap})=0.
\]

Higher utility is better. Only eligible known-ID queries have a known-class CE. True labels, OOD flags, domain/corruption metadata, correctness, supervised gradients, utilities and oracle actions are evaluator-only; none can enter objective construction, score selection, objective records or registry ranking.

All three `Z` values are losses to minimize. For each swap define the improvement score

\[
\Delta Z_j(q,a)=Z_j(\theta_R(q),q)-Z_j(\theta_a(q),q).
\]

Thus positive `Delta Z` means the post-action objective decreased. The no-swap value is exactly zero. Do not reverse sign, temperature, view set or pseudo-caption target after inspecting CE correlations. If an objective vector is constant, its per-query rank is undefined and counts as no discriminating information, not as an invalid run.

### Objective 1 — original-view Shannon entropy (`single_view_entropy`)

\[
Z_1(\theta,q)=-\sum_{c=1}^{80}p_\theta(c\mid x_q)\log p_\theta(c\mid x_q).
\]

Use the locked known-class text prompts and the model's fixed CLIP logits/temperature on the original image only. This is the direct exact post-action objective counterpart to the previously tested entropy-sign approximation; it does not reuse its score to rank actions.

### Objective 2 — MEMO marginal entropy (`memo_marginal_entropy`)

For the fixed set `V(q)` of eight views described below,

\[
Z_2(\theta,q)=-\sum_{c=1}^{80}\bar p_c\log\bar p_c,
\qquad \bar p=\frac{1}{8}\sum_{v\in V(q)}p_\theta(c\mid v).
\]

Use all views with equal weight, no confidence filter and no target from labels. This is the MEMO marginal-prediction entropy objective, evaluated after the same Ramen support intervention. The eight views are the fixed `four-corners-30-flip-v1` set: crop the raw corrupted 32×32 RGB image to 30×30 at `(0,0),(2,0),(0,2),(2,2)`, take unflipped then horizontal-flipped view for each crop, and apply the existing CLIP preprocessing. Crop before resize/normalization. Lock interpolation, antialiasing, Pillow/torchvision versions and tensor hashes in preflight. Do not use clean images, other corruptions, or label-preserving claims beyond this exact deterministic transform.

### Objective 3 — CLIPTTA image-to-text batch soft-contrastive row (`cliptta_image_to_text_entropy`)

This is the image-to-text term of CLIPTTA's soft contrastive loss (Eq. 3; main experiments use this direction when updating only the visual encoder), specialized as a local action measurement rather than a reproduction of CLIPTTA. For the current 100-row stream batch, freeze the pseudo-caption bank from the ordinary Ramen baseline state `theta_R`: each row's caption is the prompt for its `theta_R` argmax among the 80 known classes. For query row `q`, let `v_q(theta)` be its normalized visual feature and `t_j` the frozen text embedding for row `j`'s pseudo-caption. Keep the other 99 rows and all pseudo-captions fixed at their Ramen baseline values. With the pretrained CLIP logit scale and a softmax over the batch rows,

\[
P_{qj}(\theta)=\frac{\exp(s\,v_q(\theta)^\top t_j)}{\sum_{l=1}^{100}\exp(s\,v_q(\theta)^\top t_l)},
\qquad Z_3(\theta,q)=-\sum_{j=1}^{100}P_{qj}(\theta)\log P_{qj}(\theta).
\]

This measures the exact query-row contribution to the source-aligned image-to-text batch objective while changing only the query's candidate update. The pseudo-caption bank is label-free and frozen before action trials. Exclude the text-to-image term, class-diversity regularizer, confident-class memory and OCE: they either depend on other rows' simultaneous updates/state or, for OCE, introduce an extra open-set objective outside this first exact comparison. Do not use true OOD metadata to filter or weight rows; the fixed batch, including pseudo-captioned OOD rows, is visible exactly as in the deployment stream. This is a CLIPTTA-derived local component, not a claim of reproducing the full CLIPTTA method.

The objective choices are frozen before outcomes: direct entropy continuation; published multi-view marginal entropy; and CLIP's batch-aware soft-contrastive component. Tsallis/ADTE, CATE energy/CAS, calibration KL, DeYO perturbations, neighborhood consensus, OCE and other variants are deferred. They are not added as a sweep if the three candidates fail.

## 4. Views, batch access and parity

The eight MEMO views are generated from the same raw corrupted query. They are never admitted to memory and never used to update or choose the candidate support. All objective scoring happens after applying the original one-step Ramen update for each action. The view-generation RNG is fixed, but this crop/flip set itself is deterministic.

For Objective 3, all 100 stream rows and their predictions are visible because Ramen admits the full batch atomically. The scorer receives model outputs, frozen text embeddings, opaque row/action keys and the fixed objective contract only. Labels, OOD/domain fields and any utility-bearing values cross into an evaluator invoked only after all action scores and objective-based choices for that query are frozen. Regression tests must show that changing evaluator metadata cannot alter objective values or selections.

Preserve baseline output: before side evaluations, record ordinary Ramen state, cache digest and query logits; after auxiliary forwards and candidate trials, replay/reset and verify they match the reference. If a side branch changes batch-slot mapping, `theta_R`, cache contents, arithmetic order, or ordinary predictions beyond exact configured tolerance, mark the run INVALID.

## 5. Registry, splits and scan rules

The known-ID registry may use locked split membership and `is_ood` only to establish eligible evaluator queries; neither field is supplied to the scoring code. Selection is earliest-eligible and outcome-blind.

- Use the verified CIFAR-100-C content, checkpoint and current registry data fingerprint. The P0 local root digest was `115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc`; P3 recomputes and records the exact fingerprint used by the implementation.
- Exclude all 1,499 original base-image indices in [historical exclusion audit](../../plans/20261008-qcgs-literature-research-execution/reports/historical-exclusion-audit.json), with list SHA-256 `c6ea2be7074c89ca0ab8d9a6479c2a0a6aa43e4726e2a6a485d527ab22ddb72c`, plus all new smoke IDs. Historical raw archives are unavailable; the 1,499 set conservatively unions all 507 logged IDs with the complete old multi-view Stage-A hash bucket. It cannot certify absence from old unscored cache exposure.
- Registry namespace: `qcgs-exact-objective-audit-v1`. Assign a base-image ID with SHA-256 of UTF-8 `namespace|dataset_fingerprint|sample_idx`; modulo 9 bucket 0 is Stage A and buckets 1–8 are Stage B, matching the prior registry partition rule. Stage A and B base IDs are disjoint. Overlap across OOD cells is allowed and reported; those cells are paired stream conditions, not independent replications.
- Stage A quota: 16 eligible known-ID queries per cell. Stage B reserve: 128 new eligible known-ID queries per cell. Never use Stage-B metrics or labels to choose an objective, score, tie rule, or threshold.
- Start with a 600-row unscored scan prefix per cell and double only if needed, capped at 6,000 rows. Freeze the shortest complete 100-row batch prefix covering both quotas before objective scoring. If quota remains short at cap, mark the campaign INCONCLUSIVE; do not reuse old IDs or tune hash salt.
- Registry assignment and eligibility are validated independently from scoring records. Record scored base IDs separately from all identities in the historical/current cache stream; previous stream exposure is not the same as query re-use.

Before Stage A, generate two smoke queries per cell from a separate registry, exclude those IDs permanently, and use no smoke CE outcomes to alter objectives or gates. The smoke checks mechanics, finite objectives, exact action enumeration, reset, memory, runtime and provenance only.

## 6. Controls and action selection

For each objective, choose `argmax Delta Z` over no-swap plus every legal swap. No-swap wins any nonpositive maximum. For positive ties choose the first action in canonical class/outgoing/incoming order. If the objective vector is constant, select no-swap and mark the ranking undefined. The same rule applies in Stage A and Stage B.

Evaluator-only references:

- **Ramen/no-swap:** utility 0 and the main baseline.
- **Random-forced:** draw uniformly from legal swaps only; no-swap is excluded. In Stage A compute its exact expected utility as the mean `U` over all legal swaps and also record one fixed-seed draw. In Stage B record one fixed-seed legal-swap draw/query.
- **Random-matched:** use the same per-query random draw as random-forced when the objective selects a replacement; use no-swap when the objective abstains. Consume the random draw even on abstention so RNG state does not depend on policy behavior. This preserves the historical distinction between forced and matched random controls.
- **Exact oracle:** `max_a U(q,a)` over no-swap and all legal actions, Stage A only. It measures headroom, not a deployable method.
- **Supervised first-order reference:** after all label-free scores/actions are frozen, compute a single evaluator-only local linear score `-grad_theta CE_y(theta_R,q) · (theta_a-theta_R)` from exact candidate parameter deltas; choose its largest positive action, otherwise no-swap. It is a reference ceiling, not a candidate selector or GO gate.

Do not conflate a random draw with expected random utility, random-forced with random-matched, forced replacement with abstention, or an oracle with the best verified subset.

## 7. Outcomes and analysis

For each query and objective, rank the legal swaps (exclude no-swap, whose utility and objective delta are both fixed at zero) using average ranks for ties:

\[
\rho_{q,j}=\operatorname{Spearman}\big(\Delta Z_j(q,\cdot),U(q,\cdot)\big).
\]

Queries, not swaps or views, are the inferential units. Report per cell/objective: defined/total rho, mean and median rho, fraction positive, all query-level rho values, `P(U>0 | Delta Z>0)` with both query-balanced and pooled denominators, selected-action utility, exact-oracle utility, exact expected random-forced utility, random-forced draw, random-matched utility, selected-minus-random utility, replacement coverage, harmful/helpful magnitude and rates, transitions, confidence/entropy, and runtime/peak memory. Preserve all action-level raw values and undefined reasons so an independent analysis can recompute every summary.

Bootstrap at the stream-block/query level, not the action or view level: block key `floor(timestep/64)`, seed 917, 10,000 paired resamples, percentile 95% intervals. Report both OOD cells separately, cross-cell direction, number of distinct scored blocks and leave-one-block-out sensitivity. If a cell has fewer than four scored blocks, intervals are descriptive only and no confirmation GO is allowed. One seed/stream per cell does not establish cross-stream generalization. OOD fraction is a cache-history condition; differences between cells are not causal effects of OOD alone.

Stage A screens up to three objectives, so it is not a confirmatory significance test. Stage B confirms only the single frozen Stage-A winner; for selected utility and paired advantage over random-matched in each of two cells (four one-sided tests), report paired block-bootstrap p-values with Holm correction at familywise alpha 0.05 and corrected 95% intervals. Do not promote post hoc strata or alternative metrics to primary gates.

### Stage-A investment gate

The full 16-query/cell Stage A is valid only if all registered queries and every enumerated legal action have finite evaluator utility and all three objective values, parity/reset/provenance pass, and action counts independently verify. For an individual objective to pass the information screen, each cell must satisfy all of:

1. At least 8/16 query-level rho values are defined and the cell mean rho is positive.
2. Mean selected-action utility is positive and exceeds exact mean random-forced utility (uniform over legal swaps only).
3. At least 3 of 16 queries select an actual replacement rather than no-swap.
4. Mean exact-oracle utility is positive, establishing some measured headroom in the tested action space.

If several objectives pass, freeze one Stage-B candidate by greatest average per-query rho across the two cells; tie-break in this order: `single_view_entropy`, `memo_marginal_entropy`, `cliptta_image_to_text_entropy`. Stage A is a candidate screen, not evidence of confirmatory significance. If no objective passes in both cells, do not open Stage B for this selector; evaluate whether a separate reliability question is scientifically viable under §9 and record the decision.

### Stage-B confirmation gate

For the single frozen objective, evaluate all label-free candidate scores/actions on 128 fresh queries per cell, then expose evaluator labels only for the chosen action, no-swap baseline and random draw. Do not compute a Stage-B exact oracle or claim Stage-B rank correlation. GO to estimator design only if, in each cell:

1. Mean selected utility and mean paired advantage over random-matched are positive, with Holm-corrected one-sided 95% intervals above zero across both cells and both quantities. Also report selected-versus-random-forced utility separately.
2. Median selected utility is positive; the mean remains positive after removing the two largest positive utility contributors under the stable predeclared query-ID tie rule.
3. At least 3/128 queries select a replacement, all 128 registered outcomes are complete, and transition balance across the two cells (`wrong→correct - correct→wrong`) is nonnegative.

An integrity, provenance, leakage, reset or completeness failure is INVALID. A hard resource cap or too few queries/blocks is INCONCLUSIVE. A complete valid Stage B that misses one or more scientific GO gates is STOP for the frozen objective/policy in this setting; preserve all values and do not tune/re-run on the same registry. A positive GO opens a separate P6 case for approximating the exact objective and a new frozen confirmation, not a method-level claim.

## 8. Reliability and later branches

G3/selective adapt-skip is prior work, notably CAS for episodic VLM TTA, and is not part of the primary exact-objective gate. P5 may open only after Stage A review and a new dated case card. It must name a fixed adaptation policy/action, a nonconstant label-free target, development/confirmation roles, and a fresh use of reserved Stage-B IDs before exposing their outcomes. If Stage A shows that oracle headroom is effectively constant or no fixed action has a plausible nontrivial target, record L2 as not informative and stop; do not fit a gate to evaluator correctness or reuse Stage-B query labels.

If a reliability case opens, CAS from Jiang et al. is the first published baseline: use its cross-augmentation similarity from pre-action predictions on its fixed high-confidence view set, with a threshold chosen only on Stage A/development queries. Compare on fresh IDs to always-adapt, always-no-swap, and a matched random gate; report AUC, risk/utility–coverage, harmful rate and cost. Its episodic TPT result does not guarantee transfer to online Ramen. Any supervised threshold calibration is disclosed as supervised development, even if inference is label-free.

L3 evidence enrichment, L4 estimator geometry and L5 method confirmation remain closed until a dated case explains the new information source or an exact objective passes the gate. A future theory of observation-near states with reversed rankings must be a separate formal result with explicit assumptions; empirical collisions alone do not prove impossibility.

## 9. Validity, cost and execution caps

P3 must add meaningful regressions for: objective direction/loss arithmetic; hand-computable MEMO and pseudo-caption entropy; constant/undefined ranks; no-swap and positive tie behavior; random expectation versus one draw; action enumeration and serial/vectorized aggregate parity; label/metadata isolation; trial-order permutation; reset/cache immutability; fixed views and batch-row mapping; exhaustive Stage-A coverage; registry exclusions and Stage-A/Stage-B disjointness; raw-analysis recomputation. Then run the repository's required unit suite and a CUDA smoke on the separately excluded IDs. CPU success does not validate CUDA parity.

P3 computes and locks an action chunk size using smoke-only peak-memory/time measurements; every action remains enumerated. Keep peak allocation below 7.5 GiB on the 8 GiB GPU. If exact scoring cannot stay within that bound, reduce chunk size before Stage A and rerun the smoke; never drop views/actions or alter formulas. The smoke is limited to 60 minutes total and may be resumed only under the same hashes.

Scored jobs run serially on the single GPU, with independent logs and unique run IDs:

- Stage A hard cap: 8 GPU-hours per cell and 16 GPU-hours total.
- Stage B hard cap if opened: 24 GPU-hours per cell and 48 GPU-hours total.
- Maximum scan prefix: 6,000 stream rows per cell; no stage quota expansion after seeing outcomes.
- Raw evidence and logs go under ignored `evidence/qcgs-exact-objective-audit/<run-id>/` and `log/qcgs-exact-objective-audit/`; reports, hashes, lock files and decisions stay in the plan/report tree.
- On timeout, OOM or interruption, retain partial outputs and terminal status. Resume only an exactly hash-matched incomplete run. Execution failure is not a negative scientific result.

Before Stage A, freeze/record commit, protocol SHA-256, source-tree SHA, config and registry/exclusion SHA, dataset/checkpoint fingerprints, transform hashes, runtime/device, deterministic flags, launch command, unique run IDs, action chunk size, cost estimates and actual smoke receipt. Stage B requires a committed, independently recomputed Stage-A decision and a clean frozen source/config state.

## 10. Source basis and status of claims

- MEMO defines a marginal prediction entropy across augmentations: Zhang et al., NeurIPS 2022, [proceedings](https://papers.neurips.cc/paper_files/paper/2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html).
- CLIPTTA defines a soft contrastive image-to-text/text-to-image entropy over batch pseudo-captions; its main experiments use the image-to-text term when adapting the visual encoder: Lafon et al., NeurIPS 2025, [paper](https://arxiv.org/html/2507.14312) and [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/67e2da8af2476c7441c881fc253042a6-Abstract-Conference.html).
- CAS defines episodic per-sample adapt/skip using cross-augmentation similarity: Jiang et al., arXiv:2609.08367, metadata lists ECCV 2026, [paper](https://arxiv.org/html/2609.08367v1).
- CATE uses energy-anchored consensus and view selection for prompt calibration: Wang et al., *Knowledge-Based Systems* (2026), [publisher](https://www.sciencedirect.com/science/article/pii/S0950705126002595).
- Broad action identifiability already has a direct recent treatment: Jhawar & Wang, arXiv:2609.11235, [paper](https://arxiv.org/html/2609.11235v1). No broad novelty claim is made here.

The literature verification and gap report contain source access, neighboring mechanisms, search coverage and limits. New measurement claims must be based on preserved raw outputs and independent recomputation, not only the runner summary.
