# Open-set runtime implementation

Status: complete

- Added the fixed, versioned CIFAR-100-C 80/20 split and evaluator-safe wrapper.
- Stream fingerprints now include the fixed split, requested OOD ratio, and selected source sample identities.
- The model receives only known prompts and labels in `[0, 79]` or `-1`; original labels and OOD flags are joined only after forward for evidence.
- Added CLI, matrix identity, trace/summary evidence, and deterministic contract tests.
- Completed 8/8 Apple MPS cells for OOD ratios `0`, `0.1`, `0.3`, and `0.5`
  with `NoAdapt` and `CausalRamen`, 64 samples per cell, exact provenance, and
  `replay_v1` sidecars. Resume validation passed for every cell; paired source
  and stream fingerprints match and no device fallback occurred.
- Mean retrieved-OOD count fraction increased from `0.000` at ratio 0 to
  `0.115`, `0.340`, and `0.566`; mean GDC increased from `0.000` to `0.004`,
  `0.036`, and `0.076`. This validates the oracle measurement, but harmful ID
  counts are too small (`1`, `3`, `0`, `0`) for a stable causal conclusion.

Validation: the complete 330-test suite, `compileall`, `git diff --check`, all
eight runtime executions, and four strict resume plans passed under the
`nb-ramen` environment. Reports are in `reports/mps-open-set/`.
