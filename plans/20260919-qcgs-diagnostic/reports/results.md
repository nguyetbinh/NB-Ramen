# QCGS diagnostic — execution status, 2026-09-19

**Decision: INCONCLUSIVE (ordered gate 2: incomplete experimental evidence).**
No new real-model CUDA query utility has been measured. Historical supervised
oracle probes and small autograd test fixtures are not QCGS experimental results.

Implemented entropy-sign/cosine, independent forced random, Ramen/no-swap and
supervised-reference controls; same-pseudo-class one-swap; ordinary Ramen cache
and returned logits; isolated evaluator labels; exception-safe parameter reset;
finite guards; exhaustive Stage A and selected-union Stage B verification.
The evaluator audits raw utilities, selection/RNG, coverage and provenance,
then calculates paired metrics, transitions, sensitivity, block resampling,
Stage A ranking, frozen strata and the ordered gates without tuning Stage B.

Validation: 405 tests passed, 5 hardware-dependent tests skipped in the complete
suite (12.86 s on the source restored from the notebook). Focused real-autograd tests cover baseline/cache parity,
label/OOD/domain isolation, guards, reset exceptions and Stage A/B action/scope
consistency. Tests also cover registry overlap, gates, bootstrap weighting,
checkpoint/restore and embedded-source round trips. The downloader tests require
localhost binding; they passed outside the restrictive filesystem/network sandbox.
An earlier broad pytest collection included archived source copies; `pytest.ini`
now selects the current repository suite and both Kaggle support test files.

CUDA is unavailable on this Mac. CPU checks are complete; CUDA smoke, actual
smoke exclusions, real query registry, Stage A (16/cell), Stage B (128/cell),
and resulting utility/accuracy/ranking metrics remain **unrun**. No scientific
STOP/REVISE/GO_PILOT gate can be applied before these prerequisites are complete.

The [Kaggle notebook](../../../notebooks/kaggle/kaggle-qcgs-label-free.ipynb)
uses the pinned Hugging Face reconstruction and verified CLIP artifact, embeds
committed source, runs the protocol in sequence and exports
`qcgs-label-free-evidence.zip`. Notebook creation is not experimental evidence.
Runtime config, registry, RNG, source, model/data/split provenance and launch
are committed in the ZIP's separate `locks/` Git ledger before Stage A; bins
are committed before Stage B. Resume validates completed cells and restarts
an interrupted cell from its initial model/cache/RNG state.

Provenance and scope:

- [Protocol](../../../docs/research/query-conditioned-gradient-selection.md)
  and [operational spec](../../../cfg/research/query-gradient-diagnostic/protocol.json).
- [Historical exclusions](../../../cfg/research/query-gradient-diagnostic/historical-exclusions.json):
  236 unique original image indices; source query/trace/manifest SHA-256 recorded.
  The HF reconstruction must pass original NPY checksums to preserve this identity.
- [CPU preflight receipt](../../../evidence/qcgs-cpu-preflight-20260919/preflight.json),
  [focused test log](../../../evidence/qcgs-cpu-preflight-20260919/cpu-tests-0.log),
  [complete test log](../../../evidence/qcgs-cpu-preflight-20260919/cpu-tests-1.log),
  [notebook validation](../../../evidence/qcgs-cpu-preflight-20260919/notebook-validation.json).
  The focused suite passed 22 tests. All six executable cells compile; the
  embedded Git bundle restores a clean source checkout with the required ancestry.
  These local receipts are under ignored `evidence/`, not scientific query data.
- Notebook source: `8298ba434fa3e63710b899500a822a3575ca4884`.
  Source-bundle SHA-256: `a6561ba93b681a1e1ff7571cc0f24539a75fc2a1ec92770f585b916992afa109`.
  Notebook SHA-256: `4c6ece39f83b518ccc2b7175374269527768d9abd07292e1be2d5166032ff0b8`.
  CPU preflight receipt SHA-256: `5e03cfb2b67ecf65ce0cc20c4846a97ccac05f1a619f8b75a74fee75108f13fd`.
  No raw scientific records exist for this diagnostic yet.

When the Kaggle ZIP is available, extract it and run the documented `--audit`
command. It recomputes arithmetic from raw records and verifies committed inputs;
it does not independently regenerate logits. Stage A exact oracle must remain
separate from Stage B best-verified lower bounds. Single seed, shared cache and
cross-cell image overlap limit interpretation; no causal OOD/generalization claim.
No full matrix, full reranking or adaptive support-size experiment was launched.
