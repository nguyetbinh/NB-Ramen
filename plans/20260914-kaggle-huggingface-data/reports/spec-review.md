# Spec compliance

Implementation review: PASS. Local validation is complete; see [validation](validation.md).

- The new notebook acquires CIFAR-100-C from a fixed Hugging Face commit.
- Conversion preserves order/RGB bytes; all twenty original NPY checksums are
  enforced before generating inventory. No archive checksum is fabricated.
- Strict runtime and completed-run verification accept the explicit HF receipt;
  unknown or modified receipts remain rejected. Tests cover this new contract.
- The source patch is applied as a deterministic clean local commit, recorded
  in manifests, with a separate campaign path. Main-checkout scientific source
  and the original Zenodo notebook's experiment revision are preserved.
- Existing model/matrix/configuration and per-run ZIP behavior are preserved.
- The full notebook, rather than only a mismatched data cell, is deliverable.
- Verified one complete real corruption across five severities, pinned helper
  dependencies, deterministic bootstrap regeneration, syntax and final review.
- Full 19-corruption conversion and the 252-run CUDA campaign remain unexecuted
  locally and are not claimed as complete.
