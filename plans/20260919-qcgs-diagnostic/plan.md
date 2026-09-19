# QCGS diagnostic execution

Status: complete. Kaggle CUDA smoke, Stage A and Stage B finished and their artifacts passed offline validation; diagnostic decision STOP for the frozen single-view entropy-sign signal. Branch: `qcgs-label-free-diagnostic`.

Protocol: [query-conditioned-gradient-selection](../../docs/research/query-conditioned-gradient-selection.md).

1. Implement pure selectors and evaluator-only probe, retaining ordinary Ramen output/cache.
2. Implement identity registry, exclusions, frozen launch/provenance and raw-record analysis/gates.
3. Verify CPU mechanics, integrity failures, gate decisions and notebook construction; run full tests.
4. Commit source; package a pinned self-contained Kaggle notebook using verified Hugging Face acquisition.
5. Completed: audit the supplied Kaggle evidence, independently recalculate primary metrics and publish the ordered STOP decision.

Execution: Kaggle Tesla T4 / torch 2.4.1+cu121. Offline validation: local Python 3.11 / torch 2.4.1, without rerunning CUDA logits.
Unrelated existing logs and notebook metadata are preserved. No `ak` skills.

Acceptance: baseline parity, label isolation, exact Stage A versus subset Stage B, immutable registry/provenance, atomic ZIP checkpoints, replay-safe interrupted runs, no fabricated experimental evidence.

Delivered: [results and limits](reports/results.md). The notebook pins implementation `8298ba434fa3e63710b899500a822a3575ca4884`; its source bundle was restored and the CPU preflight passed on a clean checkout.

Validated artifact: `qcgs-label-free-evidence.zip` (SHA-256 `b7659601e54c3862bd31b4f6917279c2326054ef98884fbf3bfdc6aef51c317c`); 32 Stage A and 256 Stage B query observations. No further experimental expansion.
