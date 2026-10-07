# Historical CASE-QCGS-01 — P3 v1.1 resource-gate decision

**Disposition:** Superseded by the user-authorized v1.2 policy revision. This report preserves the old builder rejection and measurements as history; it is not the current case decision. See [active resource-policy revision](resource-policy-revision.md).

**Updated:** 2026-10-08
**Protocol:** `qcgs-exact-objective-audit-v1.1`, SHA-256 `f4c4ad4b9221da9293915c9b9009e5517c63ee087cbe60cb597db6a04da3d860`
**Reviewed source:** `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93`; `src/` archive SHA-256 `35af0d707d7702af2990e022d34d858b13004c42a45d5b70190c546da805c511`

## Decision

`INCONCLUSIVE — resource gate`. Both reviewed-source scans and both cost-only smoke cells completed with valid provenance and exhaustive registered action coverage, but every run's peak allocator value exceeded the frozen 7.5 GiB cap. Combined smoke process elapsed time was 35:08, below the 60-minute cap. The scientific registry was not created; the validator rejected it at the locked memory check. Do not open the launch lock, Stage A, Stage B, or the conditional estimator/reliability cases from these runs.

This decision describes compute feasibility on the measured local environment. It is not evidence for or against H1–H4, objective-to-utility ranking, selection benefit, or identifiability.

## Measurements

| Run | Cell / coverage | Peak allocator bytes | Locked cap | Status |
|---|---|---:|---:|---|
| `scan-ood-0-s1-lazycache-20261008` | OOD=0; 600 rows; config SHA `94718d74b782034d0d8e6375e77ab469f4dd84af82c0602d6285cfc307d330a8` | 8,369,873,408 | 8,053,063,680 | Complete; over cap |
| `scan-ood-05-s1-lazycache-20261008` | OOD=0.5; 600 rows; config SHA `e365858c9af26db327751dfe064dd58cd5826a62293517ffba10ab272a1e3cb6` | 8,368,574,976 | 8,053,063,680 | Complete; over cap |
| `smoke-ood-0-s1-lazycache-20261008` | OOD=0; 400 rows; 2 queries; 485 swaps + no-swap/query; config SHA `ef3fff0378093cb2b57cf5c9058412a786b9d8da022091627b34bda87aca9955` | 8,358,872,576 | 8,053,063,680 | Complete; over cap |
| `smoke-ood-05-s1-lazycache-20261008` | OOD=0.5; 600 rows; 2 queries; 850 swaps + no-swap/query; config SHA `869e7b8a267e953f6a7936cc8f4474165a594bae0a462a37546cb09b546292ca` | 8,368,574,976 | 8,053,063,680 | Complete; over cap |

Outcome-blind smoke-registry SHA-256: `40bd2afacb71482d01ddf65c33672c1a16340fc3fd9650eefdc6c70b5e014a8f`. Both smoke scorer audits passed; both runs have complete traces and no evaluator outcome file. Smoke utility is excluded from interpretation. The raw runs and config locks are preserved under `/home/thinkpad/NB-Ramen/evidence/qcgs-exact-objective-audit/`; commands and deterministic environment are in [CUDA run reproduction](cuda-run-reproduction.md). The environment was Python 3.11.15, PyTorch 2.6.0+cu124, NVIDIA RTX A2000 Laptop GPU (8 GiB), with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0`.

## Validation and next bounded test

The second smoke completed with two records and 850 legal swaps plus no-swap per query. Process elapsed times were 12:25 (OOD=0) and 22:43 (OOD=0.5), 35:08 combined. `build-registry` audited both smoke receipts and rejected construction with `ValueError: smoke must measure allocator peak below the frozen 7.5 GiB cap`. The proposed scientific-registry output file does not exist. Preserve this rejection and every run. Do not lower action coverage, view count, batch visibility, query quotas, or the memory threshold.

At the time, a future follow-up was framed as memory reduction under the v1.1 cap. The user later removed that precautionary peak cap. Current follow-up is component-level cost profiling and Stage-A preparation under v1.2; the old scans, smokes, cap rejection and raw artifacts remain unchanged.
