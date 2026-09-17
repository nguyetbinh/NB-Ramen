# Kaggle Hugging Face data support

Status: complete (notebook implementation and local validation; CUDA campaign not run).

Deliver a self-contained Kaggle notebook that runs the existing 252-run matrix
with CIFAR-100-C acquired from Hugging Face. Preserve model, stream and method
configuration. Record the changed acquisition code as a separate reproducible
source commit and campaign; never claim an unmeasured Zenodo archive checksum.

1. Add an explicit pinned Hugging Face acquisition contract and verify all
   reconstructed NumPy files against the published CIFAR-100-C MD5 table.
2. Embed the patch, preparation helper and matching matrix driver in a separate
   notebook. Re-run the patched source's tests on each Kaggle session.
3. Validate conversion with real data, invalid-input rejection, patched runtime
   tests, generated-cell syntax and deterministic source identity.

Acceptance: notebook imports independently, downloads from Hugging Face, fails
closed on altered data, preserves strict matrix validation, and has documented
local validation limits. No changes to the main checkout's scientific source.

Validation and limitations: [validation report](reports/validation.md).
