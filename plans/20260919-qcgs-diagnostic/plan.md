# QCGS diagnostic execution

Status: implementation and tests complete; notebook packaging in progress; CUDA stages pending. Branch: `qcgs-label-free-diagnostic`.

Protocol: [query-conditioned-gradient-selection](../../docs/research/query-conditioned-gradient-selection.md).

1. Implement pure selectors and evaluator-only probe, retaining ordinary Ramen output/cache.
2. Implement identity registry, exclusions, frozen launch/provenance and raw-record analysis/gates.
3. Verify CPU mechanics, integrity failures, gate decisions and notebook construction; run full tests.
4. Commit source; package a pinned self-contained Kaggle notebook using verified Hugging Face acquisition.
5. Audit available evidence and write one concise results/status report. CUDA stages are unavailable locally and must not be reported as completed.

Environment: existing Python 3.11 / torch 2.4.1 environment, no CUDA.
Unrelated existing logs and notebook metadata are preserved. No `ak` skills.

Acceptance: baseline parity, label isolation, exact Stage A versus subset Stage B, immutable registry/provenance, atomic ZIP checkpoints, replay-safe interrupted runs, no fabricated experimental evidence.
