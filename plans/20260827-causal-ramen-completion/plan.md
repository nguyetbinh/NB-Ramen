# CausalRamen Completion

## Status

Gated pilot decision on branch `causal-ramen-completion`: local CPU/MPS
implementation/control evidence is complete; final causal attribution remains
pending a justified escalation.

## Phases

1. [x] Audit the completion report, current causal implementation, evidence runtime, and available compute.
2. [x] Implement `StructuredAtomicRamen`, batch-size experiment identity, and causal-completion analysis contracts.
3. [x] Complete integrated tests and independent review.
4. [x] Run local CPU/MPS diagnostics and bounded pilots with validated evidence.
5. [ ] Gated/not executed: run the full CUDA and natural-domain matrix only if a revised or independently justified gate warrants escalation and a verified NVIDIA runner and DomainNet are available.
6. [ ] Final publication-level causal attribution remains pending/gated; the current bounded pilot does not justify escalation.

## Dependencies

- Local: `nb-ramen` Conda environment, official CIFAR-100-C under `/Users/admin/data`, Apple MPS.
- Final evidence: Linux CUDA 12.1 runner, NVIDIA GPU, persistent evidence storage, and DomainNet.

## Acceptance Criteria

- `StructuredAtomicRamen` and `CausalRamen` differ only in scheduling under identical configuration.
- `B=1` and batch-size sensitivity runs have collision-free deterministic identities.
- Every requested cell passes strict evidence validation.
- The completion analyzer reports attribution, legacy diagnostics, coverage, and a fail-closed decision.
- The completion report records which exit criteria are established and which remain blocked.

## Current gate

- The strictly validated CPU/MPS implementation/control pass is complete,
  including the B=1 diagnostic and MPS B=1,2,5,10,20,50,100 sensitivity set.
- The bounded seed-0/block/n=64 MPS sensitivity pilot found no positive
  scheduling gain: Causal-minus-Atomic mean micro delta was `-0.0111607`
  (std `0.0118114`) and mean worst-domain delta was `-0.0535714`; there was no
  negative-adaptation difference.
- This is a `PILOT` gate result, not a final publication-level `NO_GO`.
  Three seeds, full CUDA, and natural-domain/DomainNet execution are not
  justified by this gate now; CUDA and DomainNet are unavailable.

## Reports

- Local run reports belong in [`reports/`](reports/).
- Evidence summary: [local runtime and causal pilot](reports/local-runtime-and-causal-pilot.md).
- Canonical MPS sensitivity analyzer: [post-fix-mps-v2-batch-sensitivity.json](reports/post-fix-mps-v2-batch-sensitivity.json).
