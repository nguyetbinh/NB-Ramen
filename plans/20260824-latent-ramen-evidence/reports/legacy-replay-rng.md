# Legacy replay RNG verification

The legacy mixed-order stream now reproduces the exact CPU
`DataLoader(..., shuffle=True, num_workers=0, generator=None)` seed sequence
at the historical call site after method construction: the iterator's global
base-seed draw, followed by RandomSampler's global seed draw and local-
generator `randperm`.  It consumes the two global draws and records them in
stream metadata, so both sample order and first-forward global RNG match the
old evaluator.

The exact replay contract is scoped to `num_workers=0`; the CLI rejects
`--legacy_mixed_order` with any nonzero worker count.

`--legacy_mixed_order` requires `--stream_seed == --seed`, but constructor RNG
can make order method-dependent.  Therefore the exported fingerprint is the
authoritative identity, and legacy parity is not suitable for research-matrix
or negative-adaptation pairing.  Normal seeded `iid_mixed` is the fair paired
protocol.  The second ordered evaluation loader uses a private generator so it
does not add a third global draw.  Normal stream modes retain their existing
loader generator behavior.

Validation covers empirical parity against the actual historical DataLoader
with zero and nonzero constructor draws, exact first-forward stochastic values,
method-dependent fingerprints, and deterministic prefix truncation.
