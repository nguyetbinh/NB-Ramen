# ConsensusRamen soft-weight v1 implementation — 2026-08-25

## Status

Implemented as the corrected deferred thesis §17 / §26 / §33 ablation. This
is a mechanics artifact, not an adaptation-utility result.

## Preregistered surface

`cfg/CIFAR100C/ConsensusRamenSoft.yaml` fixes:

```text
consensus_mode: soft_weight
consensus_gamma: 1.0
consensus_seed: 1729
min_consensus_classes: 3
```

Let `q = abs(mean_c sign(h_c))`. With `gamma=1`, each coordinate is admitted
to SignSGD independently with probability `q`:

```text
M_j ~ Bernoulli(q_j ** gamma)
g_safe,j = M_j * g_ramen,j
```

This is deliberately an ablation rather than a replacement for v0. SignSGD
discards positive gradient scaling, whereas the Bernoulli support changes the
actual parameter delta: an admitted coordinate has Ramen's sign and a rejected
coordinate produces a zero SignSGD step. Thus its expected coordinate update
is agreement-weighted without changing the locked SignSGD optimizer.

The admission draw is reproducible: a CPU `torch.Generator` is initialized
from the immutable `consensus_seed`, offset once per forward, then its Boolean
mask is moved to the model device. CPU generation makes the configured seed
have the same draw semantics on CPU, CUDA, and MPS for a fixed stream order.
The method resets this forward counter on `reset()`. This stochasticity means
the ablation must be evaluated with the preregistered multi-seed protocol; it
is not a deterministic continuous attenuation method.

## Invariants

- The default `ConsensusRamen.yaml` remains `hard_mask` v0 with its locked
  `tau=0.2` and `C_min=3`.
- Per-class contributions, retrieval, entropy/distance weights, and ordinary
  Ramen aggregation remain unchanged.
- A zero-consensus coordinate has admission probability zero; an unanimous
  coordinate has admission probability one. Zero gradient signs are neutral
  votes.
- Fewer than three active class caches uses the exact ordinary-Ramen fallback.
- `consensus_gamma` must be finite and positive, and `consensus_seed` must be
  a non-negative integer. A zero exponent would make all nonzero agreement
  probabilities one and silently collapse the ablation toward ordinary Ramen,
  so it is rejected rather than treated as a valid setting.

For soft mode, `consensus_mask_rate` now denotes the realized sampled admission
rate for that forward, while `consensus_mean_agreement` retains the underlying
mean probability before sampling.

## Invalidated pre-fix pilot

The historical one-cell MPS `ConsensusRamenSoft` pilot is **pre-fix and not
evidence for this implementation**. It used `g_safe = q * g_ramen`, which is
observationally identical to ordinary Ramen under the locked no-momentum,
no-weight-decay SignSGD update whenever `q > 0`. Its reported accuracy must
not be compared to hard-mask v0, used for model selection, or cited as an
effect of graded consensus. The provenance record remains in
`open-set-mps-directional-pilot-20260825.md`; a fresh paired pilot is required
before interpreting soft-v1 results.

## Validation

```text
PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python \
  -m unittest tests/test_consensus_ramen.py -v
# Ran 15 tests; OK

/Users/admin/miniconda3/envs/nb-ramen/bin/python \
  -m compileall -q src/methods/ConsensusRamen.py

git diff --check
```

The focused tests cover v0 arithmetic and fallback unchanged; deterministic
Bernoulli admission; guaranteed `q=0` exclusion and `q=1` inclusion; and an
actual SignSGD parameter-delta difference from ordinary Ramen under a
controlled seed. They also cover neutral-zero consensus, below-minimum
fallback, and invalid mode/gamma/seed rejection.

## Remaining evidence gate

The runtime matrix/analyser must explicitly schedule this configuration as a
separate `ConsensusRamen-v1-soft` ablation before canonical CUDA evidence can
compare it against v0.  This implementation intentionally does not alter that
matrix or its final-benchmark claims.
