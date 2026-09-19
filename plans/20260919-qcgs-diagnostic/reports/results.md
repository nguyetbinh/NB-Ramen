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

Validation: 404 tests passed, 5 hardware-dependent tests skipped in the complete
suite (14.11 s). Focused real-autograd tests cover baseline/cache parity,
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

The [Kaggle notebook](../../../../notebooks/kaggle/kaggle-qcgs-label-free.ipynb)
uses the pinned Hugging Face reconstruction and verified CLIP artifact, embeds
committed source, runs the protocol in sequence and exports
`qcgs-label-free-evidence.zip`. Notebook creation is not experimental evidence.
Runtime config, registry, RNG, source, model/data/split provenance and launch
are committed in the ZIP's separate `locks/` Git ledger before Stage A; bins
are committed before Stage B. Resume validates completed cells and restarts
an interrupted cell from its initial model/cache/RNG state.

Provenance and scope:

- [Protocol](../../../../docs/research/query-conditioned-gradient-selection.md)
  and [operational spec](../../../../cfg/research/query-gradient-diagnostic/protocol.json).
- [Historical exclusions](../../../../cfg/research/query-gradient-diagnostic/historical-exclusions.json):
  236 unique original image indices; source query/trace/manifest SHA-256 recorded.
  The HF reconstruction must pass original NPY checksums to preserve this identity.
- CPU preflight receipt and final notebook/source-bundle validation will be linked
  after packaging. No raw scientific records exist for this diagnostic yet.

When the Kaggle ZIP is available, extract it and run the documented `--audit`
command. It recomputes arithmetic from raw records and verifies committed inputs;
it does not independently regenerate logits. Stage A exact oracle must remain
separate from Stage B best-verified lower bounds. Single seed, shared cache and
cross-cell image overlap limit interpretation; no causal OOD/generalization claim.
No full matrix, full reranking or adaptive support-size experiment was launched.
