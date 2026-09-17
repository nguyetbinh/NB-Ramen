# Hugging Face notebook validation — 2026-09-14

Result: implementation and local checks pass. Kaggle execution is not claimed.

- Patched source commit: `dc3cbf02215cff59046f048e6580dd8f8d40af89`.
- Patch SHA-256: `7489c93a4c32c3a7f882dd7c1fa2227045912d77d12a0be3bce039e832590bea`.
- Dataset commit: `a12f0bcc1da33fa26d8c76ce8c1fb32e6f913bea` in
  `WNJXYK/TTA-CIFAR-100-C`.

## Runtime and provenance tests

Tests ran in `/private/tmp/nb-ramen-huggingface-validation`, a clean clone of
the pinned base with only the bundled acquisition patch applied.

Focused runtime/provenance/matrix suite: 84 tests passed. Full source suite:
360 tests, 355 passed and 5 hardware-dependent skips. Logs:
`/private/tmp/nb-ramen-hf-focused-tests.log` and
`/private/tmp/nb-ramen-hf-full-tests.log`.

## Real data conversion

Downloaded all five brightness Parquet files from the pinned HF commit and
checked their SHA-256 against HF file metadata. Converted these actual files
using the helper with Python 3.11.16, NumPy 1.26.4, Pillow 10.4.0,
PyArrow 18.1.0 and huggingface-hub 0.26.2. The offline cache contained those
verified downloads; no synthetic replacement dataset was used.

- All 50,000 reconstructed brightness images matched original NPY MD5
  `f22d7195aecd6abb541e27fca230c171`.
- Reconstructed 50,000 labels matched original NPY MD5
  `bb4026e9ce52996b95f439544568cdb2`.
- Existing valid output skipped conversion successfully.
- Reordered labels and a truncated one-row Parquet file were rejected.

Output: `/private/tmp/nb-ramen-hf-conversion-verified`.

## Packaging and review

All five code cells, seven embedded Python programs and the Python export
compiled. Static review confirmed embedded helpers/patch match their source
files; the matrix driver differs from the original only in source revision.
A second fresh local clone reproduced the exact commit above and remained
clean. Independent code review found no concrete actionable issues.

## Limits

No full Kaggle notebook execution, complete 19-corruption conversion or CUDA
matrix execution was performed locally. The notebook enforces every original
NPY checksum before starting experiments. Its 252-run completion status remains
dependent on actual successful execution and strict coverage validation.
