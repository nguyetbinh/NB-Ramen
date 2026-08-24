# Local Runtime Smoke Evidence — 2026-08-24

## Installed environment

- Environment: `/Users/admin/miniconda3/envs/nb-ramen`
- Platform: `macOS-15.7.7-arm64-arm-64bit`
- Python: 3.11.16
- PyTorch: 2.4.1
- torchvision: 0.19.1
- OpenAI CLIP package: 1.0, pinned by `environment.yml`
- CUDA available: false
- MPS available: true

The pinned CPU/MPS manifest installed successfully. The separately pinned Linux CUDA manifest resolves in Conda dry-run; it cannot execute on this Apple host.

## Device execution

The same deterministic matrix multiplication ran on CPU and MPS:

```text
cpu_matmul=[[19.0, 22.0], [43.0, 50.0]]
mps_matmul=[[19.0, 22.0], [43.0, 50.0]]
```

## Code validation

Command:

```shell
/Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Final result after the schema-v2 evidence lifecycle, strict artifact/reference
provenance, semantic dataset preflight, analysis gates, legacy-order parity,
support ablations, entropy-gated admission, stream-identity controls, and
runtime integration: 215 tests passed. This includes
causal batch-order regressions, exact historical DataLoader/RNG parity checks,
and adversarial evidence-file and artifact replacement checks.
An independent Luna run reproduced 49/49 at an earlier causal-integration
checkpoint; the complete current suite was then reproduced locally and by a
fresh independent review.

Additional gates passed:

- `python -m compileall -q src tests`
- `git diff --check`
- `python src/main.py --help`
- missing-data preflight returned a non-zero status and structured JSON listing every absent CIFAR100C requirement

## Current evidence boundary

The official CLIP ViT-B/32 checkpoint loaded and executed a real image forward on MPS:

```text
model_dtype=torch.float16
feature_shape=(1, 512)
finite=true
forward_ms=1642.015
```

The actual `LatentRamen.forward` path then ran on two decoded macOS landscape images with three CLIP text classes:

```text
logits_shape=(2, 3)
finite=true
predictions=[0, 1]
memory_size_timeline=[1, 2]
active_contexts_timeline=[1, 1]
inferred_contexts=[0, 0]
retained_memory_bytes=161840
elapsed_ms=5195.775
after_reset_memory_size=0
```

The exact ViT-B/16 backbone selected by the CIFAR100C experiment matrix was
downloaded through the pinned OpenAI CLIP package and verified against the
SHA-256 embedded in its official model URL:

```text
checkpoint=/Users/admin/.cache/clip/ViT-B-16.pt
size_bytes=350837078
sha256=5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
```

Two images from the official CIFAR-100 clean test archive were then processed
with the real ViT-B/16 + LatentRamen path on MPS using all 100 class prompts.
The archive was verified before extraction with the torchvision-published MD5
`eb9058c3a382ffc7106e4002c42a8d85`.

```text
input_shape=(2,3,224,224)
ground_truth=[49,33]  # mountain, forest
logits_shape=(2,100)
finite=true
predictions=[49,33]
memory_size_timeline=[1,2]
active_contexts_timeline=[1,1]
inferred_contexts=[0,0]
retained_memory_bytes_timeline=[80920,161840]
elapsed_ms=1174.041
after_reset_memory_size=0
after_reset_contexts=0
after_reset_memory_bytes=0
```

This clean-set check verifies the exact CIFAR100C model/classification mechanics,
but it is not corruption-benchmark accuracy and does not satisfy the supported
CIFAR100C wrapper exit criterion by itself.

An independent Luna execution repeated the mechanics smoke on explicit CPU with the same cached checkpoint and a decoded Sonoma image:

```text
clip_feature_shape=(1, 512)
clip_feature_finite=true
clip_forward_ms=64.386
latent_ramen_logits_shape=(1, 3)
latent_ramen_logits_finite=true
latent_ramen_forward_ms=154.525
memory_size_timeline=[1]
active_contexts_timeline=[1]
retained_memory_bytes=161816
after_reset_memory_bytes=0
```

Luna also independently repeated the new ViT-B/16 check on explicit CPU with
the two official CIFAR-100 samples and all 100 class prompts. Both artifact
hashes matched before execution, the then-current full suite passed 143/143,
and no files were edited by that run. The current 215-test suite result above
covers the later provenance, deep-preflight, analysis-gate, support-ablation,
and entropy-gating work.

```text
input_shape=(2,3,224,224)
input_dtype=float32
ground_truth=[49,33]  # mountain, forest
logits_shape=(2,100)
logits_dtype=float32
finite=true
predictions=[49,33]
memory_size_timeline=[1,2]
active_contexts_timeline=[1,1]
inferred_contexts=[0,0]
final_retained_memory_bytes=323632
elapsed_ms=496.666
after_reset_memory_size=0
after_reset_contexts=0
after_reset_memory_bytes=0
```

The CPU and MPS retained-byte values intentionally differ because the pinned
CLIP loader uses float32 visual weights on CPU and float16 visual weights on
MPS; the evidence records the actual retained tensor bytes on each backend.

The explicit oracle path also ran on MPS with two distinct evaluator contexts. It emitted `inferred_context=[4,9]`, causal memory sizes `[1,2]`, active contexts `[1,2]`, finite `(2,3)` logits, and rejected a second forward without a newly supplied oracle context.

The official CIFAR-100-C archive subsequently passed its pinned Zenodo size and
MD5, deep semantic preflight, generated-inventory fast verification, and full
exact SHA-256 verification. The supported `CIFAR100C` wrapper then completed a
10-method, 8-sample MPS matrix smoke plus a paired NoAdapt/LatentRamen,
4-sample CPU smoke. Strict resume validated and skipped all 12 artifacts. See
[Official CIFAR-100-C wrapper smoke](cifar100c-official-wrapper-smoke.md).

Together these checks prove that the pinned runtime, both local compute
backends, CLI, stream/evidence utilities, official CIFAR-100-C wrapper, CLIP
checkpoint, and the complete LatentRamen adaptation path execute. They do not
prove benchmark-scale accuracy, CUDA behavior, DomainNet execution, or the
Phase 2 go/no-go gates.

A later benchmark-config MPS pilot evaluated 200 `block` samples across four
domain episodes with NoAdapt, Tent, Ramen, OracleLatentRamen, and LatentRamen.
It produced non-vacuous shift recovery, routing, negative-adaptation, memory,
and latency evidence while correctly remaining `insufficient_evidence` at the
Phase 2 gate. See [CIFAR-100-C MPS block n=200 pilot](cifar100c-mps-block-n200-pilot.md).

A second benchmark-config prefix used the `recurring` scheduler. Its 200
samples do not yet contain an actual repeated domain at the canonical
64-sample block size, so it is reported only as structured-shift/runtime
evidence in the
[recurring-prefix pilot](cifar100c-mps-recurring-prefix-n200-pilot.md).

Phase 3 then added the separately named entropy-admission ablation and ran it
through the same official wrapper on MPS. A canonical n=8 smoke and an
identity-bound `recurring`, block-size-8, n=128 smoke both passed strict
resume. The gate selected cleaner and smaller memory but underperformed
ungated LatentRamen in both bounded samples; see the
[entropy-gated MPS smoke](entropy-gated-latent-ramen-mps-smoke.md).

Phase 4 added opt-in causal retrieval profiling. A paired 32-sample MPS pilot
preserved all adaptation outputs and measured retrieval at 12.1% of profiled
forward time, below the preregistered 50% compression gate. See the
[causal retrieval profile](cifar100c-mps-causal-retrieval-profile.md).

A separate Phase 3 replication ran NoAdapt, Ramen, LatentRamen, and the
entropy-gated ablation for seeds 0/1/2 on 64-sample, block-size-8 prefixes.
Strict resume skipped all 12 artifacts. The gate reduced admitted-cache
contamination and retained bytes in every seed but underperformed LatentRamen
accuracy and produced a negative window in every seed; the detailed table is
in the [entropy-gated report](entropy-gated-latent-ramen-mps-smoke.md).
