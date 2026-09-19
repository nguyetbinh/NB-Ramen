# QCGS multi-view rescue diagnostic

Status: local implementation, independent review, CPU preflight and pinned Kaggle handoff complete. CUDA unavailable locally; smoke, scientific registry and Stage A/B remain unrun. See [validation/status report](reports/results.md).

Protocol: [frozen design](../../docs/research/qcgs-multiview-rescue-protocol.md).

1. Implement deterministic raw-image views, gradients at the Ramen anchor, independent controls and raw records while preserving baseline behavior.
2. Add versioned registry, raw audit and Stage A → Stage B gate; verify CPU behavior and regression tests.
3. Commit source; build and validate an embedded-source Kaggle notebook using the existing Hugging Face pipeline.
4. Record actual validation and unrun work in a concise report. Stop at the diagnostic decision.

Acceptance: no label access before choices freeze; exact Ramen/cache parity; reset from θ₀ for every trial; old protocol unchanged; disjoint registered base images; Stage B requires a committed and independently recomputed GO_CONFIRM; atomic evidence ZIP and reproducible pinned source.

Risks: per-sample normalization row mapping, target/anchor mismatch, accidental old-protocol audit reuse. Cover these with real autograd tests, tampered-record tests and an independent review. Rollback is removal of the new diagnostic mode; existing experiment evidence stays untouched.
