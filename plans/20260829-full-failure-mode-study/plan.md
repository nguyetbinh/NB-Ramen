# Full Failure-Mode Study Plan

Status: complete — diagnostic decision `INSUFFICIENT`; `ConsensusRamen` remains unimplemented

## Objective

Turn the bounded mechanics pilot into an executable, provenance-verified study covering the actionable experiments in `docs/research/ramen-failure-mode-error-analysis-framework.md` without implementing `ConsensusRamen` before its evidence gate passes.

## Phases

1. Close verified-analysis wiring gaps: replay feature export, domain probes, temporal panels, full F3 distributions, and cross-cell consensus aggregation.
2. Add a fixed semantic open-set CIFAR-100-C protocol and evaluator-only ID-gradient oracle path.
3. Run focused/full tests and independent Terra review; use Sol only for a hard problem Terra cannot resolve.
4. Use Luna to execute explicit CPU and MPS analysis matrices, plus an explicit CUDA availability probe with no fallback.
5. Aggregate multi-seed/multi-stream evidence, run domain/entropy/F3/F4/F5/open-set reports, and apply the fail-closed `ConsensusRamen` decision.
6. Update the framework/runtime documentation with exact commands, artifacts, limitations, and final study status.

## Result

The verified study completed on CPU and Apple MPS, including two seeds across
two structured streams, a batch-size-four atomic/causal schedule comparison,
domain probes, entropy/gradient compatibility, fixed-threshold replay, and the
semantic open-set oracle matrix. CUDA was explicitly probed and unavailable on
the host without fallback.

The preregistered decision is `INSUFFICIENT`. Gradient conflict was higher for
harmful updates in both block seeds but lower in recurring seed 1, while
recurring seed 0 had no harmful event. The replay oracle recovered harmful
events but never improved accuracy because it introduced new harm. See
[the complete study result](reports/full-study-results.md) and the
[machine-readable aggregate](reports/mps-primary/study-aggregate.json).

## Acceptance Criteria

- Every analysis CLI consumes verified runtime artifacts rather than hand-built JSON.
- Analysis/final split roles and evaluator-only labels are provenance-bound and never enter model execution.
- F3 reports all preregistered conflict distributions and required strata.
- F5 compares a true schedule-only atomic/causal pair at batch size greater than one.
- Open-set runs bind a fixed known/unknown split and OOD ratio while exposing only known-class prompts to the method.
- CPU and MPS runs complete with validated sidecars; CUDA is either completed on real CUDA or explicitly unavailable without fallback.
- Cross-seed/multi-stream aggregation returns `GO` only when every documented condition is met; otherwise it remains `INSUFFICIENT`.
- Full tests, compile checks, and diff checks pass.

## Scope Boundary

- Do not tune thresholds on final-role streams.
- Do not reinterpret domain novelty as semantic OOD.
- Do not implement or alias `ConsensusRamen` unless the completed diagnostic evidence passes every go condition.
