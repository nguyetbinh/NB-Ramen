# Open-world gradient-memory thesis completion audit

Date: 2026-08-25  
Scope: `docs/research/open-world-gradient-memory-thesis-report.md` against the
current repository and local evidence tree. This is a read-only audit; it does
not reinterpret noncanonical MPS artifacts as benchmark results.

## Verdict

The thesis is **implemented and locally mechanistically de-risked, but not
empirically complete**. The repository now has a reproducible canonical
CIFAR-100-C/CUDA experiment contract, including all seven requested controls,
but there is no completed canonical run on disk. The primary blocking facts
are external to this Mac: no NVIDIA/CUDA runtime and no verified official
CIFAR-100-C archive. The secondary DomainNet benchmark additionally lacks the
real six-domain dataset and has no open-set matrix scheduler yet.

The Apple MPS evidence is useful only for a narrow go/no-go decision: semantic
OOD supports changed the Ramen update direction and the initial hard-consensus
method is worth testing canonically. It is not evidence of final effect size,
generalisation, or compute cost.

## Requirement-by-requirement evidence

| Thesis requirement | Current support | Status | What remains |
| --- | --- | --- | --- |
| Fixed 80/20 semantic CIFAR-100-C open set; known-only model vocabulary; evaluator-only original/OOD fields (§4, §7) | `cfg/research/open-set-cifar100-split-v1.json`, `src/datasets/open_set.py`, `tests/test_open_set.py` | Implemented; local wrapper tests | Execute against official data. |
| Deterministic iid/block/recurring streams, explicit ratios 0/.1/.3/.5, split/ratio/sample IDs in fingerprint (§8) | `src/streams/builders.py`; `src/runtime/experiment_matrix.py` constants; strict validation in `validate_completed_run` | Implemented | Canonical manifests/traces for all cells. |
| Fixed source exposure 400/domain for every ratio (§34) | `OPEN_SET_PER_DOMAIN_SOURCE_BUDGET = 400`; run IDs and stream metadata bind it | Implemented | Verify realized counts in completed canonical streams. |
| ID accuracy, AUROC, FPR95, H-score, CCR/RCR (§9, §24) | `src/evaluation/open_set_metrics.py`, trace schema in `src/evaluation/evidence.py`, recomputation in `src/runtime/experiment_matrix.py` | Implemented | Completed canonical metric outputs. At OOD=0, AUROC/FPR95 are correctly required to be marked unavailable. |
| Directional oracle test: OracleDrop, OracleID, GDC, SDR and retrieved OOD contribution (§10--12, §31) | `src/methods/OracleDropOODRamen.py`, `OracleIDGradientRamen.py`; strict trace/summary checks | Implemented; noncanonical directional evidence exists | Canonical ratio sweep to test trend and effect size. |
| ConsensusRamen-v0: normal Ramen retrieval, unnormalised class contributions, sign consensus, hard mask before SignSGD, fallback below three classes (§13--22, §32) | `src/methods/ConsensusRamen.py`, config `cfg/CIFAR100C/ConsensusRamen.yaml`, unit tests | Implemented and unit-tested | Canonical effectiveness/cost evidence. |
| No label use by deployable ConsensusRamen; oracle labels confined to named controls (Appendix A) | No oracle hook in `ConsensusRamen`; oracle hook present only in named oracle classes; analyzer checks manifest context | Implemented; test-covered | Keep checking completed canonical manifests. |
| OracleConsensus upper bound (§10, §25) | `src/methods/OracleConsensusRamen.py`, its config and focused tests | Implemented and locally validated | Include in completed canonical matrix. It is intentionally not a GDC/SDR oracle. |
| All required §25 baselines | Seven-method `OPEN_SET_METHODS` and 252-run plan; `open-set-matrix-implementation-20260825.md` | Scheduled, not run | Complete all 252 runs. |
| Primary final evaluation: 4 ratios x 3 streams x >=3 seeds, paired fingerprints, official CIFAR-100-C, CUDA (§34) | Planner and post-hoc analyzer: `build_open_set_evidence_matrix`, `evaluation.open_set_consensus_analysis` | **Missing canonical evidence** | Requires verified data + Linux/NVIDIA CUDA. |
| Secondary natural-domain benchmark (§34) | `OpenSetDomainNet`, versioned 276/69 name-rank recipe and direct CLI support; report `domainnet-open-set-protocol-20260825.md` | Protocol implemented only | Acquire/verify DomainNet; add a DomainNet open-set matrix and run multi-seed evaluation. |
| Required ablations: soft weighting, no-open-set agreement, no self retrieval, vary min classes and tau (§26) | A single MPS-only tau .6 negative trial and locked tau .2 are documented | **Missing** | Implement/schedule the four missing ablations, preregister their grids, then run them on a held-out/canonical protocol. Soft v1 is explicitly deferred by §33 but §26 still calls it required. |
| Stability and cost: negative-adaptation windows, worst-domain ID accuracy, recovery, synchronized latency, retained bytes (§24) | Generic runtime emits/verifies all listed summary components; open-set analyzer presently reports worst-domain ID accuracy but not the other blocks | Infrastructure implemented; thesis reporting incomplete | Extend open-set post-hoc report/table to surface stability/cost, then populate from canonical runs. |

## Evidence already obtained on this host (strictly noncanonical)

`plans/20260824-latent-ramen-evidence/reports/open-set-mps-directional-pilot-20260825.md`
and `evidence/open-set-mps-pilot/` provide real executed traces/manifests for
an Apple M2/MPS, CLIP ViT-B/32, 80/20 split, OOD ratio .5, 128-example
approximate-CIFAR pilot.

- In the paired block stream, OracleID minus Ramen was `+1.06 pp` mean ID
  accuracy over seeds 0--2; GDC was `.1442--.2026`, SDR `.1727--.2098`, and
  retrieved OOD weight fraction `.3775--.4282`.
- The locked `tau=.2` ConsensusRamen pilot was `+2.66 pp` over Ramen in block,
  `+0.54 pp` in recurring, and `-0.52 pp` in iid mixed (nine paired cells).
- The entropy-gated negative ablation averaged `21.08%` ID accuracy versus
  Ramen `22.19%` on the three block seeds, with only `9.4--16.4%` admission.

These runs deliberately use an approximate 320-sample artifact made by
`scripts/build-cifar100c-pilot.py`, a 128-sample prefix, MPS, and disabled
official provenance. They support a *conditional go for canonical testing*,
not a thesis claim. They also do not cover ratio sweep, all seven methods,
DomainNet, or the required ablation grid.

## Canonical blockers, verified locally

1. **CUDA is unavailable.** The pinned environment is Apple arm64; PyTorch
   reports `torch.version.cuda is None`, `torch.cuda.is_available() == False`,
   and `nvidia-smi` is absent. See
   `cuda-open-set-runtime-audit-20260825.md`.
2. **The official CIFAR-100-C artifact is unavailable.** The local archive is
   12,516,864 bytes with the wrong MD5, rather than the required Zenodo record
   3555552 archive (2,918,473,216 bytes; MD5
   `11f0ed0f1191edbf9fa23466ae6021d3`). Deep preflight rejects the local
   generated artifact because the official arrays need 50,000 samples.
3. **DomainNet is absent.** Deep preflight has no six-domain tree under the
   configured data root. Even after data acquisition, the open-set matrix is
   CIFAR-100-C-specific today, as the DomainNet protocol report explicitly
   records.

These are genuine execution blockers, not missing code. A CPU/MPS run cannot
substitute for the CUDA requirement in the thesis's final-evaluation contract.

## Prioritized next actions

1. **Highest value, external runner:** use a Linux/NVIDIA Luna environment,
   acquire and verify the exact official CIFAR-100-C archive, generate the
   provenance sidecar, run deep preflight, then execute the already-generated
   252-run matrix with `--open-set-consensus --device cuda --execute`. Use no
   prefix; retain the fixed 400-source/domain exposure. Run NoAdapt first in
   each paired cell as the matrix requires.
2. **Before interpreting CUDA results:** run
   `python -m evaluation.open_set_consensus_analysis` on the evidence
   directory. It requires seven-method coverage and a single exact stream
   fingerprint per ratio/stream/seed cell, and otherwise returns
   `noncanonical_pilot`/fails validation.
3. **Locally solvable code/reporting gap:** add the §26 ablation variants and
   extend the open-set analyzer to publish the already-validated stability and
   cost blocks. This is independent of CUDA, but does not replace its final
   execution.
4. **Secondary benchmark:** add an explicit DomainNet open-set matrix only
   after obtaining the real taxonomy/data and deciding a fixed exposure budget
   appropriate to its six domains; then run its multi-seed CUDA grid.

## Completion criterion

Do not mark the thesis complete until a verified official-data CUDA evidence
tree contains every planned CIFAR-100-C cell, the strict analyzer classifies
it `canonical_cuda_expected`, the required §26 ablations have evidence, and a
secondary natural-domain benchmark has been implemented, executed, and
reported. The present state is therefore **active work with a well-defined
external execution dependency**, not completion.
