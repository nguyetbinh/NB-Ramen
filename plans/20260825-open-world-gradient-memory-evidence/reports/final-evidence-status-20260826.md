# Final evidence status — 2026-08-26

## Verdict

Implementation and local mechanics validation are complete. No canonical
effectiveness conclusion is available. F1--F4, evaluator-only label isolation,
pre/post OOD detection, ID-only stability, clean `EntropyGatedRamen`, the
seven-method planner identity, v2/v3 split recipes, and ablation/config/split
locks are complete and locally validated.

## Completed local smoke

Artifacts are under `/Users/admin/Documents/NB-Ramen/evidence/open-set-mps-post-f1f2-smoke-20260826/`. The
paired NoAdapt, Consensus, and OracleID runs use MPS, `max_eval_samples=32`,
batch/max-batch 8, block stream, seed 0, requested OOD ratio 0.3, fast
provenance, and shared stream fingerprint
`3b426821c8461cc0effbdac160e931d9de75978a5c987eff8a0c9ccdea2688f3`.

The realized stream contains **27 ID and 5 OOD samples (5/32 = 0.15625)**,
all from brightness. It is therefore too small and too unrepresentative to
estimate efficacy or safety. The official CIFAR-100-C archive/data is present
at `/Users/admin/data`, but this is a non-CUDA prefix run and does not meet
the canonical full-stream protocol.

| Run / artifact | Exact smoke metrics |
| --- | --- |
| `open-c100-noadapt-block-s0-mps-n32-w8/summary.json` | micro accuracy 0.65625; ID accuracy 0.77778; pre/post AUROC 0.66667/0.66667; pre/post FPR95 0.62963/0.62963; 30.3049 samples/s |
| `open-c100-consensus-block-s0-mps-n32-w8/summary.json` | micro accuracy 0.65625; ID accuracy 0.77778; pre/post AUROC 0.66667/0.53333; pre/post FPR95 0.62963/0.88889; negative-adaptation rate 0.25 (1/4) and ID-only 0.33333 (1/3); 6.45642 samples/s; retained memory 2,588,800 bytes |
| `open-c100-oracle-id-block-s0-mps-n32-w8-r2/summary.json` | micro accuracy 0.65625; ID accuracy 0.77778; pre/post AUROC 0.66667/0.63704; pre/post FPR95 0.62963/0.85185; GDC 0.05725 (Ramen) versus 0.17534 (Consensus diagnostic); SDR 0.06023 versus 0.41972; GDC reduction −0.11809; SDR reduction −0.35948; 4.68221 samples/s |

The OracleID directional smoke result is negative: the reported Consensus
diagnostic is farther from the ID-only oracle direction on this tiny run. It
is **not** evidence against the thesis mechanism or a basis to retune it:
there are only five realized OOD samples, one seed, one short block prefix,
MPS execution, and no canonical comparison grid. Likewise, the apparent
post-adaptation OOD degradation is a smoke observation, not a safety result.

## Evidence still required

- Canonical CIFAR-100-C: all 252 full-stream CUDA runs, verified official
  provenance, paired NoAdapt-first fingerprints, and strict
  `canonical_cuda_expected` analysis.
- Split robustness: the recipes are frozen, but the actual 96 CUDA runs for
  v2/v3 have not run.
- Ablations: the held-out 72-run grid and config hashes are frozen, but no
  ablation run has executed.
- DomainNet: the actual six-domain, 345-class data is missing and no CUDA
  runner is attached; no DomainNet result exists.

The current Mac has no local CUDA runtime. Do not substitute CPU/MPS, a
truncated prefix, or this local smoke directory for any of the outstanding
studies.
