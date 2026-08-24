# Phase 03 — Entropy-Gated Memory Admission

Status: in progress

## Decision basis

- [Research roadmap](../../docs/research/ramen-thesis-research-roadmap.md)
- [Phase 02](phase-02-latent-ramen-v0.md)
- [CIFAR-100-C MPS block pilot](reports/cifar100c-mps-block-n200-pilot.md)

The first real-wrapper Phase 02 pilot did not establish a useful routing
upper bound: oracle routing was 1.0 percentage point below Ramen, while the
unsupervised router discovered one context across four observed domains. The
roadmap says to reconsider latent routing when oracle routing gives almost no
gain and identifies reliable memory under pseudo-label error as the next
research axis. This phase therefore tests one reliability mechanism without
adding another routing hypothesis.

## Preregistered mechanism

Add a separately named `EntropyGatedLatentRamen` method. It is identical to
`LatentRamen` except for memory admission. For each sample, use the model's
pre-update logits to compute:

```text
normalized_entropy = predictive_entropy / log(number_of_classes)
admit_to_memory = normalized_entropy <= 0.50
```

The primary threshold is fixed at `0.50` before benchmark execution. It must
not be selected from final test-stream results. Any later threshold sweep
requires a separately recorded validation protocol and distinct config/run
identities.

An admitted sample keeps LatentRamen's causal insert-then-retrieve behavior.
A rejected sample retrieves only historical memory. This phase does not add a
fallback insertion, reliability weighting, soft labels, relabeling, neighbor
agreement, gradient agreement, or any ground-truth input to the method.

## Requirements

- [x] Add and register `EntropyGatedLatentRamen` without changing original
  `Ramen`, `LatentRamen`, or `OracleLatentRamen` behavior.
- [x] Require a finite `max_normalized_entropy` in `[0, 1]` and ship the
  preregistered `0.50` benchmark configs for CIFAR100C and DomainNet.
- [x] Preserve strict per-sample causality for admitted and rejected samples,
  including batches larger than one.
- [x] Emit the pre-update admission prediction, normalized entropy, and
  admission decision for every gated sample.
- [x] Compute admission rate, rejected count, admitted pseudo-label accuracy,
  rejected pseudo-label accuracy, and contamination rate as evaluation-only
  summary diagnostics.
- [x] Keep older schema-v2 traces and summaries valid; gated optional fields
  must be all present or all absent and validated when present.
- [x] Add the method to explicit matrix selection without changing the Phase
  02 default ten-method grid or analyzer.
- [x] Verify config bounds, empty-history behavior, insert/retrieve ordering,
  batch causality, reset, diagnostics, summary metrics, and strict resume.
- [x] Run a real CIFAR100C MPS mechanics smoke before any comparative claim.
- [ ] Run paired NoAdapt, Ramen, LatentRamen, and EntropyGatedLatentRamen on
  IID-mixed, block, and recurring streams for three seeds on fixed CUDA.
- [ ] Repeat on DomainNet before evaluating the reliability hypothesis.

## Evidence contract

The method receives images and model-derived state only. Ground-truth class
and domain remain evaluator-only. Gated traces add these optional fields:

```text
admission_prediction
admission_normalized_entropy
admitted_to_memory
```

The evaluator may join `admission_prediction` to `ground_truth_class` after
the forward pass to compute pseudo-label quality. This evaluation must never
feed back into routing, admission, retrieval, or adaptation.

Primary comparison: gated versus ungated `LatentRamen` on the exact same
stream fingerprint. NoAdapt and legacy Ramen remain external references.
Every adapted run must use the exact paired NoAdapt trace required by the
existing evidence contract.

## Decision criteria

Support the entropy gate only if it:

1. lowers admitted-memory contamination relative to the ungated stream;
2. improves negative-adaptation behavior under structured shifts;
3. maintains average and worst-domain accuracy; and
4. is most useful where pre-adaptation pseudo-label error is high, as predicted
   by roadmap hypothesis H3.

Reject or revise the mechanism if it merely shrinks memory, produces empty
support for long prefixes, or improves a selected test cell without the
predeclared multi-stream, three-seed replication. Latency and retained bytes
must be reported alongside accuracy; no gain may be attributed to reliability
if compute or memory budgets are not comparable.

## Risks and rollback

- Early high entropy can leave memory empty. The method must produce a safe
  zero update and must not silently insert a rejected sample.
- Post-adaptation entropy is already present in the trace but cannot audit the
  admission decision; the new score must be captured before adaptation.
- Optional evidence fields must not invalidate the completed Phase 01/02
  schema-v2 artifacts.
- If the primary threshold fails, preserve the negative result. Do not change
  the threshold in place or overwrite its run directories.
