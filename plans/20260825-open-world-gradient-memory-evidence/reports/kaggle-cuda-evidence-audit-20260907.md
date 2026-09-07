# Kaggle CUDA evidence audit — 2026-09-07

## Verdict

**NO-GO for full execution until the corrected B=1 control is rerun and
reviewed.** All 14 uploaded smoke runs completed and pass strict evidence
validation. The causal control nevertheless exposes a real pretrained-reset
aliasing defect. This supersedes the earlier report's CUDA-not-run status;
it does not certify a canonical matrix or method efficacy.

## Audited artifact

- Input: `nb-ramen-evidence.zip`, 481,528 bytes, 122 ZIP entries.
- SHA-256: `d9cfc37c80e886c9b600939995d807e598e017ebe213c69e6fe9e6926259ebc0`.
- Archived revision: `4eef3564a12515dea137cdba624787afb3032345`; clean Git state.
- Python 3.11.16, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121,
  CUDA 12.1, Tesla T4 (14.56 GiB reported VRAM).
- Uploaded full suite: **350 tests OK, no skips**; focused suite: 140 OK.
  Thus the four real-CUDA FP16 cache regression tests ran successfully.
- Trace schema 3, summary schema 4, all prefixes 256 samples.
- OOD=0 null control passed. v2 NoAdapt/Entropy pairing passed.
- 252 canonical runs were planned; **none were executed**.

The archive was extracted under
`evidence/kaggle-imports/d9cfc37c80e8/raw/` without executing its scripts.
Independent `validate_completed_run` accepted **14/14** runs in a separate
derived validation mirror. Only JSON path prefixes were transported from
the Kaggle repository/evidence directories to local equivalents so configs,
split locks, and reference traces could be reread. Data/model artifact paths
and recorded digests were retained. This is offline evidence validation,
not a fresh hash of the unavailable Kaggle dataset or model cache.

Every archived source inventory exactly matched the local source at
`4eef356` before the fix. Mapping, ZIP digest, and per-run outcomes are in
`evidence/kaggle-imports/d9cfc37c80e8/audit-validation.json`. Raw extracted
artifacts and the original ZIP were not modified.

## Causal anomaly

The v1 OOD=.3 controls share the same ordered stream fingerprint:
`36d57c6137a0fdb9e38a49a95cdb845eae3cc558de91cee3e0b9392adb4ef05f`.
Each has 186 ID and 70 OOD examples.

| Run | ID accuracy |
| --- | ---: |
| NoAdapt B=100 | 51.08% |
| NoAdapt B=1 | 51.08% |
| Ramen B=100 | 53.76% |
| Ramen B=1 | **3.23%** |
| CausalRamen B=100 | 55.91% |

Ramen B=100 versus B=1 differs on **250/256 pre-adaptation predictions**
and **248/256 post-adaptation predictions**. B=1 versus CausalRamen B=100
has the same disagreement counts. NoAdapt B=1 versus B=100 has zero
pre-prediction disagreements. Comparing Ramen to its same-batch NoAdapt
baseline gives 250 pre-prediction disagreements at B=1 and only one at B=100.
These are smoke diagnostics, not statistically supported efficacy estimates.

## Reproduced defect and correction

Both `BySampleLayerNorm` and `BySampleBatchNorm` initialized trainable
sample-specific parameters with `buffer.expand(capacity, -1).contiguous()`.
At capacity 1 this is already contiguous and aliases the pretrained buffer.
Optimizer updates therefore corrupt the reset target. At capacity 100 the
expanded view required a copy, explaining the capacity-specific failure.

A local real-tensor reproduction confirmed mutation of pretrained buffers
and failed reset at capacity 1 for both normalization wrappers, while
capacity 100 reset correctly. New regression tests using actual forward,
backward, and SignSGD steps failed in all three CPU capacity-1 cases before
the change. Replacing `contiguous()` with `clone()` for all four affine
parameter initializations makes the reset state independent at every capacity.
This fixes the demonstrated defect; its end-to-end effect on the Kaggle
accuracy collapse still requires a new CUDA smoke.

After the fix: focused **4 tests OK, 1 CUDA skip**; full **354 tests OK,
5 CUDA skips** on the local non-CUDA host. Configs, thresholds, learning
rates, admission rules, and retrieval semantics were not changed.

## Next execution

Use the refreshed Kaggle notebook pinned to the fix commit and its separate
revision-specific evidence directory. Run all 14 current-schema smoke runs
and 354 tests on CUDA. Inspect the repeated-reset regression test and
same-batch NoAdapt pre-prediction comparisons before reviewing the causal
post-prediction differences. Preserve the uploaded ZIP as the defect record.
Freeze a canonical revision and consider full execution only after that
review; do not tune the methods against this short prefix.
