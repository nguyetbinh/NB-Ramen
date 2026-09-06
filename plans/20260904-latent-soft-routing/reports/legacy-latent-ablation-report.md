# Legacy Ramen + Latent Routing Ablation — 2026-09-06

## Question

Does the original online prototype router help when it is added to legacy
batch-atomic Ramen without replacing Ramen's cache, schedule, weighting, or
update rule?

## Intervention

`LegacyLatentRamen` retains:

- one legacy `PriorityCache`-compatible cache per predicted class;
- float16 keys and gradients;
- the original per-class capacity and recency-priority replacement rule;
- full-batch admission before any query;
- entropy and feature-distance weighting;
- Ramen's scalar number-of-nonempty-classes aggregation denominator;
- SignSGD, learning rate, model step, and reset behavior.

The only intended intervention is the original `OnlinePrototypeRouter` plus a
context ID attached to each cache item. Retrieval makes candidates from other
contexts ineligible. If every item is assigned to one context, the method
uses `PriorityCache.query` and the legacy aggregation operations directly.

The benchmark config equals `cfg/CIFAR100C/Ramen.yaml` plus exactly four
router values copied from `LatentRamen`: `spawn_threshold=0.25`,
`max_contexts=8`, `router_temperature=0.1`, and `router_momentum=null`.

## Protocol

- CIFAR-100-C, CLIP ViT-B/16;
- canonical block stream, block size 64;
- deterministic 200-sample prefix, seed 0;
- evaluator batch size 100;
- MPS, fast artifact provenance;
- fingerprint
  `aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b`.

The canonical NoAdapt trace was reused after strict validation. A fresh
legacy Ramen control and the new ablation were run into the same evidence
root. Luna was not resolvable from this environment.

## Results

| Method | Micro | Macro-domain | Worst-domain | Negative windows |
|---|---:|---:|---:|---:|
| NoAdapt, reused | 0.305 | 0.3203 | 0.2344 | reference |
| Ramen, fresh | 0.315 | 0.3828 | 0.2656 | 2/4 |
| LegacyLatentRamen, fresh | 0.310 | 0.3789 | 0.2656 | 3/4 |

The raw fresh difference is -0.5 percentage points micro, -0.39 points
macro-domain, and zero worst-domain difference. It is **not** an estimate of
the value of routing because the router never produced an intervention:

```text
discovered contexts: 1
all inferred contexts: 0
assignment churn: 0
domain/context NMI: 0
domain/context ARI: 0
```

Every support item therefore remained eligible exactly as in Ramen.

## Numerical interpretation

MPS output is not stable enough here to interpret the small fresh-run
difference as a method effect. The older strict-valid Ramen artifact and the
fresh Ramen artifact differ on 26/200 predictions despite identical Ramen
source, config, stream fingerprint, checkpoint, and dataset content. Repeated
LegacyLatent executions whose changed multi-context branch was never reached
also changed predictions.

A direct synthetic MPS-half diagnostic gave exact equality between legacy and
collapsed-router cache contents and aggregate gradients:

```text
torch.equal(legacy_gradient, collapsed_router_gradient) = true
max absolute gradient difference = 0
cache tensors equal = true
```

A CPU end-to-end pair was attempted, but unchanged legacy Ramen fails because
PyTorch CPU does not implement `torch.cdist` for its fixed float16 cache. The
baseline was not modified because doing so would invalidate the one-factor
ablation.

No latency conclusion is made from separate MPS runs.

## Decision

The missing historical ablation is now implemented and executed. Its main
finding is mechanistic, not an accuracy ranking:

> The original prototype router also collapses to one context when placed on
> legacy batch-atomic Ramen. The failure to discover context is therefore not
> caused by CausalRamen or structured memory.

Because the router selected one context, this cell provides no evidence that
latent routing helps or hurts legacy Ramen. A performance test of routing
would require a preregistered router setting that actually produces multiple
contexts and a deterministic supported backend, preferably CUDA.

## Evidence and validation

Final evidence:

```text
plans/20260904-latent-soft-routing/reports/legacy-latent-ablation-mps-n200/
```

The final `LegacyLatentRamen` artifact passes strict resume validation and its
recorded method-source SHA-256 matches the current file. The full unit suite
passes 268 tests. Incomplete CPU evidence, the pre-review MPS artifacts, and
the stale smoke artifact were moved to recoverable `/tmp/nb-ramen-*`
directories and are excluded from the table.
