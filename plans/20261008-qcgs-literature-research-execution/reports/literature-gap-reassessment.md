# Literature gap reassessment: unlabeled TTA action identification

**Review date:** 2026-10-08

**Status:** P1 decision record complete for bounded protocol design; broader novelty remains provisional.

## Finding that changes the prior gap

The proposed G1 framing in the synthesis—whether unlabeled evidence identifies which adaptation action is useful—is no longer a safe novelty claim. A primary-source search found Jhawar and Wang, *When Is Test-Time Adaptation Identifiable from Unlabeled Evidence?*, arXiv:2609.11235, submitted 2026-09-10 and marked in its PDF as under review for ICLR 2027 ([arXiv HTML](https://arxiv.org/html/2609.11235v1), [PDF](https://arxiv.org/pdf/2609.11235)).

The paper defines oracle-action identifiability for a specified evidence channel and a set of possible deployment worlds, proves exact and approximate ambiguity results, and develops a finite-batch KEEP-versus-RECENTER result. Its CIFAR-100-C study constructs controlled deployment worlds using five full-stream actions—SOURCE/KEEP, DeYO, Tent, ROID and test-time normalization. In matched image-set experiments, changing stream order can change the oracle action while permutation-invariant evidence stays unchanged. It also tests stream-aware evidence and reports that its advantage depends on the deployment family; its stochastic-deployment result reverses the controlled-family ranking. These details were read in the source's problem setup, theory, experiment design, results and discussion/limitations sections (PDF §§2–7 and appendices A–B).

This is direct conceptual and dataset overlap. The two-point argument and a generic claim that unlabeled evidence may not identify the best TTA action are already in prior work. The synthesis's proposed H4 and L6 language about observation-near deployments with reversed action rankings must therefore be attributed and bounded against that paper rather than presented as a new general contribution.

## Candidate distinction that remains to be verified

The QCGS evidence appears to use a narrower action unit and target:

| Dimension | Jhawar–Wang preprint | Candidate Ramen audit |
|---|---|---|
| Action | Choice among five whole-stream TTA procedures | One of many legal single-support substitutions inside one fixed Ramen update |
| State | Deployment world and stream, with order/class-prior regimes | One query's observed cache snapshot, pseudo-class candidate pool and frozen update |
| Outcome | Prospective target risk / stream-level oracle method | Same-query counterfactual CE change after exact support-gradient replacement |
| Evidence question | Whether stream/global evidence identifies which TTA method wins across worlds | Whether a local observation or exact per-action objective delta orders this query's legal support swaps |
| Main mechanism variation | Order, batch, class priors and deployment family | Support identity and exact local intervention with retrieval/update held fixed |

This distinction is a candidate contribution, not an established gap. The paper itself studies identification of the best member of a fixed menu of data-dependent interventions, so action granularity alone may be too small. Other direct neighbors include retrieval-based TTA for VLMs (TT-RAA, ICCV 2025; RA-TTA, ICLR 2025), few-shot example support curation (SelectNAdapt, ICCV 2023 Workshop), and gradient-informed parameter-subspace support selection (LOFT, arXiv:2605.11872; listed in NeurIPS 2026 downloads). Jiang et al.'s *To Adapt or Not to Adapt? Selective Adaptation for Vision-Language Models* (arXiv:2609.08367; metadata lists ECCV 2026) directly studies per-image binary adapt/skip with Cross-Augmentation Similarity (CAS). CATE (*Knowledge-Based Systems*, 2026) uses energy-anchored view consensus for prompt calibration. These works close broad claims about query-level reliability, selective adaptation and consensus. None of the reviewed primary descriptions changes the narrower candidate action unit, but the difference remains provisional and needs future code/citation review.

The current working question is therefore provisional:

> Under a fixed Ramen query, cache snapshot, candidate pool and exact update, does any prespecified label-free observation channel—or exact post-update objective change—contain useful information for ranking that query's legal one-support substitutions by its own supervised CE utility?

The distinction to test is one query's cache snapshot and exact same-query CE counterfactual after a legal gradient-support substitution. SelectNAdapt uses target examples that are subsequently labeled; LOFT uses supervised calibration gradients to select a parameter subspace; TT-RAA and RA-TTA retrieve class centers or external examples to adjust VLM predictions; CAS selects whether to run episodic adaptation; CATE selects/weights views; Jhawar–Wang select whole-stream TTA methods across deployment worlds. This combination may still be incremental, so importance and practical value need a separate argument.

This question is about a measured local counterfactual setting. It does not assert universal identifiability, impossibility, a new general theory, or a deployable selector. An empirical STOP would close only the exact formulations, actions, candidate pool and environment tested.

## Other high-relevance primary sources found

- [MORPHEUS, ICLR 2026 Test-Time Updates workshop](https://theyoungkwon.github.io/papers/articles/danilowski_morpheus_iclr-ttu26.pdf) predicts which whole TTA method and post-adaptation accuracy to expect from unlabeled source-model entropy and embedding geometry. Its predictor is trained on held-out domains after measuring the candidate methods' labeled adaptation accuracy there. It is a close method-selection precedent with a supervised meta-training budget and a coarse action menu.
- [AETTA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_AETTA_Label-Free_Accuracy_Estimation_for_Test-Time_Adaptation_CVPR_2024_paper.html) estimates adapted-model accuracy from prediction disagreement with dropout inferences and studies recovery. It is relevant to reliability/abstention, not direct support-substitution ranking.
- [Realistic Evaluation of Test-Time Adaptation: Unsupervised Model Selection](https://www.scitepress.org/Papers/2026/143201/143201.pdf) (VISAPP 2026) studies surrogate-based hyperparameter selection. Its 2024 arXiv version ([2407.14231](https://arxiv.org/abs/2407.14231)) reports that some consistently useful strategies require limited labels or source pretraining data. Its outcome/computation budget differs from pre-action Ramen support-swap scoring.
- [Test-Time Correction through Candidate Verification](https://arxiv.org/abs/2610.06147), submitted 2026-10-05, verifies top-k class hypotheses using reconstruction-induced changes in class-subspace divergence. It is a current neighborhood/memory signal for VLM predictions, but its actions are candidate labels and its objective is not Ramen support-gradient utility.
- [TT-RAA, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Test-Time_Retrieval-Augmented_Adaptation_for_Vision-Language_Models_ICCV_2025_paper.html) and [RA-TTA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fa1790d7c3036c691d0b2fb3b9a0ce64-Abstract-Conference.html) are query-conditioned retrieval precedents; inspect their memory/update contracts before freezing neighborhood evidence.
- [SelectNAdapt, ICCV 2023 Workshop](https://openaccess.thecvf.com/content/ICCV2023W/LIMIT/html/Dawoud_SelectNAdapt_Support_Set_Selection_for_Few-Shot_Domain_Adaptation_ICCVW_2023_paper.html) selects target examples with pseudo-labels and self-supervision, then annotates them for few-shot use. [LOFT, arXiv:2605.11872](https://arxiv.org/html/2605.11872v1) uses supervised calibration gradients to choose a low-rank parameter subspace. They are support-selection precedents under different action units and label budgets.
- [CLIPTTA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/67e2da8af2476c7441c881fc253042a6-Abstract-Conference.html) directly motivates contrastive/source-aligned objective candidates; it proposes an adaptation loss, not query-conditioned support utility selection.
- [To Adapt or Not to Adapt? Selective Adaptation for Vision-Language Models](https://arxiv.org/html/2609.08367v1) (Jiang et al., 2026; arXiv metadata lists ECCV 2026) defines per-sample adapt/skip for episodic VLM TTA. CAS uses prediction agreement across augmented views and reports AUC/AEP plus accuracy under high skip ratios. This is the direct selective-adaptation baseline; its action is whole-sample adaptation versus skipping, not which online Ramen support to replace.
- [CATE: Consensus-aware calibration for test-time prompt tuning via energy anchoring](https://www.sciencedirect.com/science/article/pii/S0950705126002595) (Wang et al., *Knowledge-Based Systems*, 2026) uses energy-anchored consensus, view diversity and cross-modal weighting for calibration. It bounds broad consensus/view-selection claims but does not rank support-gradient actions.
- [Adaptive Debiasing Tsallis Entropy, ICLR 2026](https://openreview.net/pdf?id=dHj8hC081K) is a current entropy/calibration alternative for VLM test-time adaptation. Its entropy family should not be treated as exhausted by Shannon entropy alone.
- [Tempora, ICML 2026](https://arxiv.org/abs/2602.06136) reports method ranking instability under latency budgets. It strengthens the need to measure action-selection cost and state the deployment budget, but does not rank support identities.

These sources narrow or constrain objective-family and deployment claims. They do not yet establish whether a local exact per-swap objective is useful.

## P1 completion and bounded decision

The targeted review covered the 18 references and named adjacent work, plus action identifiability, per-sample selective adaptation, current VLM method selection, label-free performance estimation and current calibration/consensus. Search terms and limitations are recorded in [literature verification](literature-verification.md). The candidate local-action study is decision-relevant as a bounded diagnostic because prior QCGS reports leave exact objective alignment versus estimator error unresolved. Continue only with exact, predeclared objective/action measurements and explicit stop rules. Do not sell it as a new selective-adaptation problem or a general identifiability result.

## Evidence and limitations

The identifiability and selective-adaptation metadata and full HTML were checked directly on arXiv; CATE's publisher record and available full-text view were checked on the review date. These checks assess scope overlap, not the correctness or reproducibility of the authors' campaigns. The local QCGS raw archives are missing, so this comparison uses published descriptions and archived reports rather than a replication. The action-granularity distinction remains provisional and can be overturned by future code/citation review.
