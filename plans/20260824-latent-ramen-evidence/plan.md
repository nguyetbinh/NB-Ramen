# Latent Ramen Evidence Program

Status: in progress

## Phases

1. [Phase 01 — Reproducible streams and evidence](phase-01-reproducible-streams-and-evidence.md)
2. [Phase 02 — Oracle routing diagnostic and `LatentRamen-v0`](phase-02-latent-ramen-v0.md)
3. [Phase 03 — Entropy-gated memory admission](phase-03-entropy-gated-memory.md)
4. [Phase 04 — Causal retrieval profiling](phase-04-causal-retrieval-profiling.md)

## Dependencies

- Phase 01 must provide deterministic orders, trace schemas, and stable metrics before method comparisons.
- Oracle routing is the go/no-go diagnostic for Phase 02.
- Reliability and compression experiments depend on stable structured-memory interfaces.
- Real accuracy evidence requires the external datasets, a working pinned Torch environment, and CUDA hardware sized for the selected backbone.

## Acceptance criteria

- Every claim is linked to a versioned run manifest, per-sample trace, summary, and exact code revision.
- Original `Ramen` remains the reference implementation.
- Ground-truth domains are used only for schedule construction and evaluation, never as input to unsupervised routing/adaptation; oracle routing is explicitly labeled.
- The five immediate stream modes (`iid_mixed`, `block`, `gradual`, `recurring`, `imbalanced`) run for NoAdapt, Tent, Ramen, oracle Ramen, and LatentRamen-v0 on DomainNet and CIFAR100C.
- Results include accuracy, worst-domain accuracy, sliding accuracy, recovery, negative adaptation, routing diagnostics, latency, and memory.
- The final completion audit covers every requirement and go/no-go gate in `docs/research/ramen-thesis-research-roadmap.md`.

## Reports

Runtime probes and experiment reports belong in `reports/`. Generated raw traces and large artifacts remain outside git unless explicitly curated.

- [Official CIFAR-100-C wrapper smoke](reports/cifar100c-official-wrapper-smoke.md)
- [CIFAR-100-C MPS block n=200 pilot](reports/cifar100c-mps-block-n200-pilot.md)
- [CIFAR-100-C MPS recurring-prefix n=200 pilot](reports/cifar100c-mps-recurring-prefix-n200-pilot.md)
- [Entropy-gated LatentRamen MPS smoke](reports/entropy-gated-latent-ramen-mps-smoke.md)
- [Causal retrieval profiling MPS pilot](reports/cifar100c-mps-causal-retrieval-profile.md)
- [CUDA and DomainNet execution strategy](reports/cuda-domainnet-execution-strategy.md)
