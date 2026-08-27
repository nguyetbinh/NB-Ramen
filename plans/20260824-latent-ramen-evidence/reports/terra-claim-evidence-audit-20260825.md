# Claim-to-evidence audit — open-world gradient memory thesis

Date: 2026-08-25
Scope: the active thesis specification, the broader latent-Ramen roadmap, the
evidence-program plan, source/config/planner contracts, and completion-relevant
reports. This is an audit, not a new experiment or an interpretation of a
pilot as a benchmark.

## Bottom line

The active thesis has a credible **mechanics and directional-pilot** basis but
has **no thesis-grade effectiveness evidence**. Its immediate blocker is not
only CUDA execution: four prerequisites in thesis phases F1--F4 have not all
landed, and the generated CIFAR-100-C and DomainNet primary planners still
name `EntropyGatedLatentRamen` where the thesis freezes the clean
`EntropyGatedRamen` control. Consequently, neither current 252-run plan is
the final claim-bearing matrix described by the thesis.

The older `ramen-thesis-research-roadmap.md` describes a broader, prospective
LatentRamen programme. It is useful as a future-work dependency tree, but it
is not evidence for the active ConsensusRamen thesis and its 300-run matrix
must not be merged with the seven-method open-set result.

### Evidence grades used below

| Grade | Meaning |
| --- | --- |
| Implementation | Unit tests, planners, smoke paths, or source review prove a contract, not a scientific effect. |
| Directional pilot | Executed, paired, but prefix-limited and/or non-CUDA; may justify the next experiment only. |
| Thesis-grade | Verified official data, full fixed exposure, CUDA, paired complete multi-seed matrix, and post-hoc validation. None exists yet. |

## Active claim-to-evidence matrix

| ID / claim | Required experiment and acceptance evidence | Current artifact and grade | Missing evidence / exact next action |
| --- | --- | --- | --- |
| C1. Semantic OOD changes persistent Ramen updates (RQ1). | Oracle all-support versus ID-only directions: retrieved OOD/weight fraction, GDC and SDR; OOD=0 null; ratio sweep and OracleID--Ramen gap. | `open-set-mps-directional-pilot-20260825.md`, independently recomputed in `open-set-mps-pilot-independent-audit-20260825.md`: three block/OOD=.5 MPS prefixes show GDC .1442--.2026, SDR .1727--.2098, OOD-weight .3775--.4282, and OracleID--Ramen +1.0589 pp. OOD=0 control is near-zero. **Directional pilot.** `official-cifar100c-cpu-open-set-smoke-20260825.md` is a separately paired, official-data CPU prefix and is also noncanonical. | Full verified CUDA grid must show the trend by ratio, stream, and seed. No claimable effect size, generalisation, or CUDA cost exists. |
| C2. Prediction-confidence cleanliness is not adaptation usefulness (RQ2 / contribution 1). | Clean `EntropyGatedRamen` (`max_normalized_entropy=.50`) versus Ramen and Consensus, with admission/purity and ID utility. | Historical `EntropyGatedLatentRamen` MPS studies consistently make memory cleaner/smaller while reducing accuracy; the directional pilot reports 21.08% versus Ramen 22.19% in block. **Directional negative pilot, confounded by latent routing.** | Implement `src/methods/EntropyGatedRamen.py`; add and hash `cfg/{CIFAR100C,DomainNet}/EntropyGatedRamen.yaml`; use no router. Required fixture: all-admit equals Ramen, rejected samples never enter memory. Replace the stale latent name in both planners/tests before any canonical run. Threshold is frozen; do not retune on final streams. |
| C3. Consensus approximates the safe ID-only direction without labels (RQ3 / contribution 3). | Same retrieved supports must yield `g_ramen`, label-free `g_consensus`, and evaluator-only `g_oracle_id`; require trace cosine/SDR pairs and summary `gdc_reduction_mean`, `sdr_reduction_mean`. Positive reductions are required **in conditions where ID adaptation improves**. | Hard-mask v0 source/config and unit mechanics exist; it uses `tau=.2`, `C_min=3`, all-support class contributions, and pre-SignSGD masking. Consensus MPS results are heterogeneous: iid -0.5208 pp, block +2.6628 pp, recurring +0.5354 pp. **Directional pilot only.** | F1 is unimplemented: source search finds no `consensus_vs_oracle_id_*`, `gdc_reduction`, or `sdr_reduction` fields. Implement in `OracleIDGradientRamen`/evidence/main/analyzer; prove label isolation and synthetic SDR reduction; strict validators must reject partial diagnostic groups. Then collect the full CUDA matrix. |
| C4. Benefit depends on stream structure (RQ4). | Report gain, agreement, retained-coordinate rate, GDC/SDR, and Oracle gap separately for iid/block/recurring. Do not average away an iid null. | The nine MPS cells above provide a useful conditional signal only. **Directional pilot.** | Complete every primary cell and analyze by stream. The evidence programme plan's `gradual` and `imbalanced` streams belong to the older LatentRamen programme, not the active seven-method thesis grid. |
| C5. Safer aggregation preserves OOD safety after adaptation (RQ5). | Per returned adapted logits record `E_post=-logsumexp(logits)` and report pre/post AUROC, FPR95, H-score. NoAdapt pre/post must match within tolerance; OOD=0 detection is unavailable, not invented. | Pre-adaptation energy/open-set infrastructure and mechanics tests exist. Existing pilots correctly state that pre-score is method-invariant. **Implementation only.** | F2 is unimplemented: source search finds no `post_adaptation_ood_score` or pre/post metric groups. Add it without an extra forward, then require/validate it in every canonical trace/summary. |
| C6. Open-set utility/stability/cost is preserved. | ID accuracy, worst-domain ID accuracy, ID-only negative windows, ID-only episode recovery, synchronized forward latency/throughput, retained bytes, and paired Consensus--Ramen overhead. | `open-set-analysis-stability-costs-20260825.md` verifies analyzer extraction/validation. It is contract evidence; the MPS/CPU timings are not CUDA measurements. **Implementation only.** | Populate from complete CUDA evidence. Ensure negative windows select only ID rows and recovery respects ID counts/episode boundaries; do not call generic all-row accuracy a stability result. |
| C7. Results do not depend on the primary contiguous CIFAR taxonomy split. | Reduced robustness study: v1 plus fixed name-rank v2/v3 splits; ratios .3/.5; block/recurring; seeds 0,1,2; Ramen, Consensus, OracleID. | Only v1 wrapper/protocol mechanics are present. **Implementation only.** | Define/version the two salts and materialization artifacts, then run 72 named method cells; also retain paired NoAdapt traces if stability comparisons are reported (making 96 executions if NoAdapt is rerun per split/ratio/stream/seed). Do not select splits after outcomes. |
| C8. The method survives a natural multi-domain benchmark. | Canonical DomainNet semantic open set: 276/69 name-rank v1, 690 sources/environment, 4 ratios, 3 streams, 3 seeds, seven methods, CUDA and verified six-domain tree. | `OpenSetDomainNet`, split fingerprinting, configs, and a 252-run plan exist. **Implementation/planning only.** DomainNet data and CUDA execution are absent. | Repair the stale entropy baseline first; regenerate/validate the seven-method plan, acquire/verify the six-domain tree and provenance sidecar, execute 252 full CUDA runs, and report mechanism trends rather than headline accuracy alone. |
| C9. Consensus design choices are causal rather than accidental (§27). | Held-out, separately reported ablations: Ramen, v0, soft Bernoulli admission, no-self, tau=.6, Cmin=2, Cmin=4; exact same NoAdapt trace and fixed configs. | Corrected soft-v1 implementation, no-self and fixed alias YAMLs plus a planner exist. One-cell MPS checks are mechanics/directional only; pre-fix magnitude-scaled soft pilot is withdrawn. **Implementation + pilot, not acceptance evidence.** | Caller must freeze the held-out cell(s); example preregistered plan is block/.5/seeds 0--2 = 8×3=24 executions including NoAdapt. Run on verified official CUDA evidence, then report separately from the 252 primary cells. |

## Exact canonical experiment contracts

### G. Primary CIFAR-100-C open-set matrix — required after F1--F4

| Locked factor | Required value |
| --- | --- |
| Data / split | Official Zenodo CIFAR-100-C, `open-set-cifar100-split-v1`, 80 known / 20 unknown; known-only prompts. |
| Exposure | 400 selected sources per each of 15 corruptions = 6,000 before scheduling; no prefix. |
| OOD ratios | 0, .1, .3, .5. |
| Streams | `iid_mixed`, `block`, `recurring`; default block size 64 unless explicitly re-frozen. |
| Seeds | 0, 1, 2. |
| Methods | NoAdapt, Ramen, **EntropyGatedRamen**, OracleDropOODRamen, OracleIDGradientRamen, ConsensusRamen, OracleConsensusRamen. |
| Model/config | ViT-B/16; Ramen capacity 750/top-k 5/beta 5/SignSGD lr .01. Consensus hard mask, `tau=.2`, `C_min=3`, `include_current=true`; tau is frozen. Entropy gate `.50`, router-free. |
| Pairing/provenance | NoAdapt first; exact same stream fingerprint per ratio/stream/seed; CUDA; verified official artifact; `fast` or `exact` provenance consistently; run manifest, stream, trace, summary, source revision. |
| Count / acceptance | 7×4×3×3 = **252** executions. Strict analyzer must classify it `canonical_cuda_expected`, and all F1/F2 field groups must validate. |

The current executable planner is close but **not compliant** because
`OPEN_SET_METHODS` schedules `EntropyGatedLatentRamen`. After F1--F4 and the
baseline repair, the intended command shape is:

```shell
PYTHONPATH=src python -m runtime.experiment_matrix \
  --open-set-consensus --device cuda \
  --data-root "$RAMEN_DATA_ROOT" \
  --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-cifar100c" \
  --artifact-provenance fast --execute
```

First run it without `--execute`, archive the planning JSON, and run deep
preflight after exact-sidecar verification. The current CLI exposes streams
and seeds but freezes the OOD sweep and method list; it has no direct source
budget flag, so the regenerated planner must bind the required 400 budget.

### H1. Split robustness — required after G

The thesis specifies two additional deterministic name-ranked 80/20 CIFAR
splits (fixed salts/version IDs v2 and v3), OOD .3/.5, block/recurring,
seeds 0--2, and Ramen/Consensus/OracleID. This is 72 comparison method cells
before accounting for the paired NoAdapt reference. It has no current planner,
salt files, configs, or completed evidence. It is thesis-grade only with the
same official-data/CUDA/provenance/pairing constraints as G.

### H2. Secondary DomainNet open-set matrix — required after G / alongside robustness

| Locked factor | Current planner value |
| --- | --- |
| Split | `open-set-domainnet-name-rank-v1`, 276 known / 69 unknown; taxonomy fingerprinted at materialization. |
| Exposure | 690 per each of six environments = 4,140 before scheduling (divisible at .1 and represents all 69 unknown classes). |
| Grid | 7 methods × ratios 0/.1/.3/.5 × iid/block/recurring × seeds 0/1/2 = **252**. |
| Config | ViT-B/32; capacity 300/top-k 10/beta 5/SignSGD lr .01; locked Consensus v0 as above. |
| Execution | `PYTHONPATH=src python -m runtime.open_set_domainnet_matrix --device cuda --data-root "$RAMEN_DATA_ROOT" --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-domainnet" --artifact-provenance fast` emits plan only; its emitted commands must execute after deep preflight. |

This planner is also currently noncompliant with the active thesis because it
uses `EntropyGatedLatentRamen`; repair and revalidate it before calling it
canonical.

### §26 held-out Consensus ablation — secondary, required

The exact currently discoverable planning command (not execution) is:

```shell
PYTHONPATH=src python -m runtime.consensus_ablation_matrix \
  --stream block --ood-ratio 0.5 --seed 0 --seed 1 --seed 2 \
  --device cuda --data-root "$RAMEN_DATA_ROOT" \
  --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-consensus-ablations" \
  --artifact-provenance fast
```

It fixes the seven adapted identities and an exact NoAdapt trace per seed;
the currently declared example yields 24 planned executions. The protocol is
sound, but the thesis has not yet selected/frozen whether this one held-out
cell is sufficient. A plan is not an executed ablation.

## Broader roadmap experiments: separate, prospective programme

These are explicit in `ramen-thesis-research-roadmap.md`, but are not active
Consensus claims and should be reported as a different research axis.

| Roadmap experiment / claim | Exact available contract | Support | Missing acceptance evidence |
| --- | --- | --- | --- |
| Phase 0 baseline reproducibility | Reproduce one CIFAR corruption and one DomainNet Ramen result; fixed environment/manifests; tolerance must be agreed. | Official CIFAR wrapper/smokes exist; no DomainNet execution. | Define tolerance, run repeated full benchmark cells on CUDA and DomainNet. |
| Phase 1 five-stream infrastructure | IID, block, gradual, recurring, imbalanced without changing Ramen. | Builders/planners and small MPS checks. | Full data/CUDA evidence, including a true recurrence (the early n=200 recurring prefix did not recur). |
| Minimal LatentRamen go/no-go | DomainNet+CIFAR100C; five streams; NoAdapt/Tent/Ramen/oracle-domain/LatentRamen; accuracy, worst domain, sliding/recovery, NMI/ARI, memory, latency. The legacy contract is 2×5×3×10 = 300 runs, batch 100, CUDA. | Planner/mechanics and MPS pilots; observed routing collapse and no qualifying oracle gain in two prefixes. | Complete verified CUDA 300-run grid plus `phase02-go-no-go.json` thresholds: ≥3 repeats, structured degradation ≥.01, oracle recovery ≥.01, router closure ≥.25, natural-domain gain ≥.005, routing/accuracy association ≥.2, latency ≤1.25×, memory ≤1.1×, accuracy SD ≤.01, class-context NMI ≤.8. These are legacy latent-routing gates, not Consensus acceptance thresholds. |
| Reliability-memory variants | Entropy, neighbor agreement, gradient agreement, soft class labels, repair/forgetting one at a time. | Entropy pilot only; active consensus work is a more focused gradient-agreement branch. | Each variant needs a fixed, paired ablation and severe-shift comparison; no complete grid exists. |
| Scalability/compression | Full gradients vs prototypes/low-rank/sign; memory/latency Pareto, large-class experiments. | MPS causal profile found retrieval share 12.1%, below its 50% compression trigger. | Do not add compression yet. A full-capacity fixed-hardware DomainNet CUDA profile must first meet the preregistered retrieval-growth and ≥50% retrieval-share conditions. |

## Dependency graph and no-go conditions

```text
F1 consensus-to-oracle diagnostics ─┐
F2 post-adaptation OOD scores      ├─> F4 docs/config/planner sync ─> CIFAR G (252) ─┬─> split robustness
F3 clean entropy baseline          ┘                                                   └─> DomainNet H2 (252)
                                                                                              │
held-out Consensus ablations (24+; separately paired) ──────────────────────────────────────┘

Broader latent-router programme: reproducibility -> five-stream infrastructure ->
oracle routing -> LatentRamen go/no-go -> reliability / scalability. It does not
block the active Consensus matrix, but it cannot be cited as its validation.
```

Hard no-go conditions:

1. Do not execute the current named 252 plans as final thesis matrices until
   the clean entropy baseline replaces the latent one and F1/F2 schemas are
   implemented and tested.
2. Do not use MPS or CPU prefixes (including official-data CPU smoke) for
   CUDA, effect-size, generalisation, or final latency claims.
3. Do not use generated/approximate-CIFAR MPS artifacts as CIFAR-100-C
   benchmark data; the independent audit confirms this boundary.
4. Do not tune `tau=.2`, entropy `.50`, or choose split/ablation cells after
   inspecting canonical outcomes.
5. Do not collapse iid/block/recurring results to conceal the currently
   observed heterogeneous pilot signal.

## Evidence and environment reconciliation

Some reports are time-stale: `thesis-completion-audit-20260825.md` and CUDA
audit reports say the official CIFAR artifact was unavailable, whereas
`cifar100c-official-wrapper-smoke.md` and
`official-cifar100c-cpu-open-set-smoke-20260825.md` record verified official
data and CPU executions. The filesystem now has the CIFAR tree, but no
DomainNet tree; neither condition changes the CUDA requirement. Treat the
official-data CPU report as a useful paired directional check, and treat all
claims of missing local CIFAR data as superseded by the later provenance report
unless a new deep preflight disproves it.

## Final audit verdict

The defensible present conclusion is narrow: OOD-containing support can alter
the cached update, and hard consensus is sufficiently de-risked to warrant a
proper test. The thesis cannot yet claim that Consensus safely improves
open-world adaptation. Complete F1--F4, regenerate the aligned contracts, run
the 252-cell CIFAR CUDA matrix, execute the separately frozen ablations and
robustness study, and finish the aligned 252-cell DomainNet matrix before
making that claim.

Status: DONE_WITH_CONCERNS

Summary: Mapped every active canonical/secondary thesis experiment and the
broader roadmap programme to its current evidence, exact available commands,
acceptance conditions, and dependencies. Identified two critical pre-run
specification gaps: missing F1/F2 metrics and the stale latent entropy
baseline in both canonical planners.

Concerns/Blockers: No CUDA evidence or DomainNet data; current primary and
secondary planners do not match the thesis method list; split-robustness
planner/salts and execution contract remain absent; pilot evidence is
directional only.
