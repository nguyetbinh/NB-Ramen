# Open-set and oracle-gradient mechanics evidence — 2026-08-25

## Result

Phase A (open-set CIFAR-100-C infrastructure) and Phase B
(`OracleIDGradientRamen` / `OracleDropOODRamen`) passed local mechanics
validation. This is implementation evidence only: this host has neither the
CIFAR-100-C data nor CUDA, so it makes no claim about adaptation utility,
semantic-OOD effect size, or a final thesis result.

## Implemented contract

- `open-set-cifar100-split-v1` fixes 80 known and 20 held-out CIFAR-100
  classes. The model vocabulary contains only the known class names.
- `--open_set --known_class_split open-set-cifar100-split-v1 --ood_ratio R`
  creates deterministic per-domain selections, preserves evaluator-only
  `original_label`, `known_label_or_minus_one`, and `is_ood`, and fingerprints
  the split, requested ratio, realized counts, domain order, and identities.
- Open-set summaries use pre-adaptation energy (`-logsumexp`) for `ACC_ID`,
  AUROC, FPR95, H-score, and worst-domain ID accuracy.
- The two oracle methods require a single-use evaluator OOD hook. The ID
  oracle retains the exact batch-atomic Ramen support set then removes only OOD
  gradient contributions; the drop oracle omits OOD items from memory.
- Oracle traces record retrieved OOD fractions and all-vs-ID direction cosine
  and sign disagreement. The evidence writer rejects partial groups or oracle
  fields lacking an open-set provenance group.

## Commands and observed outcomes

```shell
PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
# exit 0; Ran 231 tests; OK

PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python \
  -m compileall -q src tests
# exit 0

git diff --check
# exit 0
```

Focused coverage includes deterministic/lazy open-set stream construction,
ratio and fingerprint binding, OOD metric tie behavior, evaluator metadata
isolation, oracle hook failure behavior, all-vs-ID gradient arithmetic,
batch-order support visibility, reset, and zero-direction diagnostics.

An in-memory two-domain stream smoke (no benchmark images) ran
`ordered_stream_test` at an exact OOD ratio of 0.25. It emitted a valid
open-set trace and summary with 6 ID plus 2 OOD examples, `ACC_ID=1.0`,
`AUROC=1.0`, `FPR95=0.0`, and `H-score=1.0`; those values validate the
synthetic fixture only.

## Next evidence gate

Run the paired `NoAdapt`, `Ramen`, `OracleIDGradientRamen`, and
`OracleDropOODRamen` commands in
[`docs/research/experiment-runtime.md`](../../../docs/research/experiment-runtime.md)
on the same verified CUDA dataset and stream fingerprints. Only then can GDC,
SDR, contamination, and adaptation comparisons be interpreted as empirical
evidence. The local CUDA/data blocker is recorded separately in
[`cpu-preflight-and-cuda-blocked-20260825.md`](cpu-preflight-and-cuda-blocked-20260825.md).
