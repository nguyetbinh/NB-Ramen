# Research state — exact label-free support-action audit

**Updated:** 2026-10-08

**Research IDs:** `CASE-QCGS-01`; planned experiments `EXP-QCGS-*`

**Repository:** `qcgs-objective-audit`, HEAD `45ea8f4` when P0 began; plan commit by `[historical Git identity redacted]`

## Stage and question

P0–P2 are complete. The implementation fix and v1.1 scans/smokes are historical completed inputs. Active policy is v1.2 with report-only peak allocation. Final deterministic v1.2 component profiles are complete; P3 remains open for quota-complete scans, registry replay and launch-lock construction. P4 Stage A has not started.

The current candidate question is narrower than generic TTA method selection: within one fixed Ramen query, cache snapshot, update rule and legal same-pseudo-class one-support-swap set, do label-free observations or exact post-update objective changes rank that query's supervised CE utility? This is provisional pending the current literature comparison. It does not claim a general theorem about identifying the best TTA action across deployment worlds.

The literature search removed broad novelty claims. Jhawar–Wang (arXiv:2609.11235) directly studies TTA action identifiability and stream-level action reversal on CIFAR-100-C. Jiang et al. (arXiv:2609.08367; metadata lists ECCV 2026) define per-sample adapt/skip with Cross-Augmentation Similarity. CATE (Knowledge-Based Systems, 2026) studies energy-anchored view consensus for prompt calibration. The remaining possible distinction is action granularity and outcome: exhaustive legal support substitutions and same-query counterfactual CE inside a fixed online Ramen state. It is only a candidate diagnostic extension; broader novelty search/code review can still overturn it.

## Observed environment

| Item | Verified observation |
|---|---|
| Main checkout | Clean `qcgs-objective-audit`; plan is committed; no uncommitted source changes at P0 completion |
| Auxiliary worktree | `/home/thinkpad/NB-Ramen-qcgs-audit`, detached at `e7bf615`; not used for scored work |
| Git identity | Global and repository identity `[historical Git identity redacted]` |
| Runtime | Linux WSL2; local `.venv` Python 3.11.15, PyTorch 2.6.0+cu124, torchvision 0.21.0 |
| Accelerator | One NVIDIA RTX A2000 Laptop GPU, 8 GiB; CUDA available |
| Data layout | CIFAR-100-C under `/home/thinkpad/data/corruption/CIFAR-100-C`; `runtime.preflight --deep` passed all required files, shapes and label ranges |
| Dataset content identity | All 20 original `.npy` MD5 values matched the repository's pinned CIFAR-100-C source table. The 2,918,473,216-byte local Zenodo tar also matched the pinned official MD5 `11f0ed0f1191edbf9fa23466ae6021d3` (record 3555552). The extracted directory contains `README.txt`; the Hugging Face helper rejects this extra root file because it expects exactly the 20 `.npy` files. |
| CLIP checkpoint | Cached OpenAI ViT-B/16 file SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`; exact verification passed |
| Historical raw evidence | `evidence/` is absent. Referenced raw query JSONL, validation receipts and archives were not available locally; earlier report numbers remain reported/archived claims, not newly recomputed results. |
| Historical runtime | Prior CUDA reports describe PyTorch 2.4.1+cu121 on Tesla T4. Current local runtime differs and needs a parity/smoke record before new scored evidence. |

Commands completed:

```sh
PYTHONPATH=src uv run --offline python -m runtime.preflight \
  --data-root /home/thinkpad/data --dataset CIFAR100C --deep --json
```

Additional streaming checks compared each local NPY against the 20 expected original-file MD5 values, verified the local Zenodo archive through `verify_official_cifar100c_archive`, and verified the cached checkpoint using `runtime.artifact_provenance.verify_clip_checkpoint`. All passed. The generated preflight JSON is at `/tmp/qcgs-p0-preflight.json` and is a local diagnostic, not yet a campaign receipt.

No unit suite, new inference, CUDA smoke or scored campaign has run in P0.

## Reconstructed prior evidence and limits

The committed reports describe: positive screened-oracle headroom in the oracle pilot; STOP for the frozen single-view entropy-sign diagnostic; and Stage-A STOP for the frozen multi-view target-θ_R rescue. The multi-view report says its offline raw audit reproduced saved arithmetic, while no CUDA rerun occurred. Since the raw files are absent in this checkout, these remain the prior report's validated claims; this execution has not independently reproduced their query metrics or receipts.

`cfg/research/query-gradient-multiview/historical-exclusions.json` contains 507 unique base-image indices: 272 oracle query rows (16+128+128), 271 distinct single-view archive query IDs, and four smoke IDs, with overlap among sources. It does not list the 27 distinct multi-view Stage-A images. The multi-view receipt does expose its registry fingerprint `e316a95d...` and locked modulo-9 namespace; the source code assigns every Stage-A member to that hash bucket. I reconstructed that entire 1,049-index bucket and checked five sample IDs named in the prior report against it. Combining this conservative superset with the 507 known indices yields 1,499 unique exclusions and 8,501 remaining original CIFAR indices. The reproducible list and input hashes are in [historical-exclusion-audit.json](historical-exclusion-audit.json). This resolves query-ID exclusion coverage without recovering raw rows, but it does not establish that those images were absent from earlier *unscored cache streams*; that exposure must be reported separately.

The old oracle/single-view fingerprint recorded in the exclusions file is `115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc`, which matches the current local 21-file directory digest. The multi-view receipt fingerprint is `e316a95d...` and corresponds to the 20 original NPY files; the auxiliary README changes the directory digest. Exact official archive and NPY checksums bind their image indices across these two roots.

## Instrument and resource limits

The read-only P2a audit found a reusable path in `Ramen.py`, `OracleSupportUtilityProbe.py`, the single-view and multi-view probes, their evaluator modules, and `query_gradient_registry.py`. Preserve batch-atomic visibility, full-batch cache admission before retrieval, class-specific top-k support, candidate cardinality, exact SignSGD update, parameter reset, canonical action ordering, outcome-blind selection, and evaluator-only labels/OOD metadata.

The existing probe is not an exact-objective audit: its registry, schemas and gates are specific to previous diagnostics. In particular, sign-direction deduplication only applies when the objective depends on the resulting SignSGD state, not when the objective depends on incoming/outgoing support identity. Historical grouped ViT parity guards cover only the first two active queries, so the new audit needs appropriate known-case coverage.

One report lists evaluator peak memory near 7.24 GiB on an 8 GiB device before accounting for all retained allocations. Final v1.2 one-query cost-only profiles completed full candidate evaluation at 485 / 850 swaps with 8,358,872,576 / 8,368,574,976 peak bytes; these are reported costs, not gates. A conservative maximum-action forecast fits the frozen compute budgets, but actual Stage-A runtime remains unmeasured. Action batching and the frozen action set remain unchanged.

## Next decision

The v1.1 7.5 GiB rejection is preserved as history and superseded by the user's v1.2 revision. The current registry check accepts complete high-peak smoke receipts when their v1.1 source/protocol provenance is explicit. At the 600-row scan prefix, OOD=0 lacks 8 Stage-A and 51 Stage-B eligible IDs, so the frozen doubling rule requests 1,200 rows/cell. Final cost-only profiles on excluded smoke IDs are complete and audited. Next: quota scans, registry replay, launch lock, then Stage A within the frozen budgets. No objective-to-utility outcome exists yet.

## P3 execution state — v1.1 receipts and v1.2 continuation

The exact-objective implementation is committed on `qcgs-exact-objective-code` at `04abdbaab46f8c1d35f304e6cb8dbd1f43554b93`, after a lazy-cache memory fix and a fresh targeted review with no remaining findings. The focused suite passed 35 tests; the repository suite passed 443 tests with 1 skipped. The source archive SHA-256 is `35af0d707d7702af2990e022d34d858b13004c42a45d5b70190c546da805c511`.

Under v1.1, two outcome-blind 600-row CUDA scans passed provenance/continuity checks but exceeded that protocol's 7.5 GiB precautionary cap. Both smoke cells completed with full registered action coverage and parity/provenance receipts. Their peaks were 8,358,872,576 bytes (OOD=0) and 8,368,574,976 bytes (OOD=0.5); combined process elapsed was 35:08. The v1.1 registry builder rejected construction at that gate. These are preserved cost observations, not evidence for or against any research hypothesis; v1.2 supersedes the peak threshold.

The v1.2 protocol/config/validator and profile instrumentation are committed at `5f486e4ada84fb5b6a1c4794c3a00dc9015654d5`; query-invariant per-query view forwards are committed at `9206f08f5d66229a9d1d526667ed8ed9c03e8a65`. The focused suite passes 37 tests. A CPU target-row action-forward optimization failed bitwise batch parity and was discarded; production full-batch action forwards remain. Final r3 cost-only profiles on the frozen queries passed audits with no evaluator outcomes. See [resource-policy revision](resource-policy-revision.md) for hashes, costs, forecast, quota and next actions.
