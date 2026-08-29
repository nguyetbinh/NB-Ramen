# Failure-Mode Analysis Implementation Plan

Status: completed (diagnostic runtime and bounded CPU/MPS pilot); CUDA/final-evidence extension pending an NVIDIA host

## Objective

Implement and execute the diagnostic framework in `docs/research/ramen-failure-mode-error-analysis-framework.md` without introducing a deployable consensus method before its go/no-go evidence exists.

## Phases

1. [Instrumentation, replay sidecar, and contracts](phase-01-instrumentation-and-contracts.md)
2. [Offline failure and oracle analysis](phase-02-offline-failure-and-oracle-analysis.md)
3. [Representation and temporal diagnostics](phase-03-representation-and-temporal-diagnostics.md)
4. [CPU/MPS execution and evidence](phase-04-execution-and-evidence.md)
5. [Independent review and final validation](phase-05-review-and-validation.md)

## Completion Record

The implementation and bounded pilot outcome are recorded in
[reports/implementation-and-pilot-summary.md](reports/implementation-and-pilot-summary.md).
CPU/MPS machine-readable outputs are under `reports/luna-runs-v4/`.
`ConsensusRamen` remains out of scope and unimplemented: the completed pilot
does not satisfy its evidence gate. CUDA was unavailable on the Apple host;
the runtime requires explicit device selection and has no silent fallback.

## Dependencies

- Phase 1 owns and freezes the diagnostic, replay, identity, and compatibility contracts.
- Pure mathematical helpers and deterministic probe utilities may be developed in parallel, but phases 2 and 3 cannot consume real evidence until phase 1 is merged.
- Phase 2 depends on the finalized phase-1 trace and replay readers.
- Phase 3 depends on the phase-1 feature/replay contract and phase-2 paired outcome contract.
- Phase 4 depends on code and focused tests from phases 1–3.
- Phase 5 depends on all implementation and run evidence.
- CUDA-grade claims require a Linux NVIDIA runner and are not available on the current Apple host.

## Acceptance Criteria

- Structured Ramen can emit legal causal support provenance and gradient-conflict summaries without changing adaptation outputs.
- An opt-in bounded replay sidecar reconstructs exact item vectors, legal candidate sets, retrieved supports, and schedule state, and is cryptographically bound to the run manifest and stream.
- Paired traces produce exact safe/beneficial/harmful/unresolved counts and verify `Acc_adapted - Acc_base = H - A`.
- Offline analyzers cover memory/retrieval oracle gaps, entropy downstream influence, open-set GDC/SDR, domain decodability, and temporal/stratified summaries.
- F4 controlled-update and F5 scheduling fields are explicitly computed or reported with machine-readable `unavailable`/`insufficient` reasons.
- Evaluator-only labels never feed back into deployable method execution.
- Focused and full tests pass.
- Fresh CPU and local accelerator smoke runs produce valid evidence through the existing experiment runtime.
- Independent Terra review finds no unresolved critical correctness, leakage, or public-contract regression.

## Scope Boundary

- Do not implement `ConsensusRamen` unless bounded diagnostics satisfy all three documented go conditions.
- Do not claim CUDA or final scientific evidence from MPS/CPU smoke runs.
- Do not tune thresholds on held-out/final streams.
