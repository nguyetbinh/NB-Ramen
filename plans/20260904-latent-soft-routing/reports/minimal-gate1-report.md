# Minimal Latent Soft Routing Gate 1 — 2026-09-04

## Outcome

The minimal oracle-soft routing check is complete and does not pass Gate 1.
At `gamma=0.25`, soft context ranking changed selected support on 73 of 200
queries but changed none of the 200 predictions. Accuracy and
negative-adaptation metrics remained identical to `CausalRamen` and the
`gamma=0` recovery control.

This is a bounded no-go for continuing automatically to `LatentSoftRamen`.
It is not a claim that every context strength or dataset must fail.

## Reuse-first protocol

- Dataset/backbone: CIFAR-100-C, CLIP ViT-B/16.
- Stream: canonical block, block size 64, prefix 200, seed 0.
- Device/provenance: MPS, fast verified artifact inventory.
- Stream fingerprint:
  `aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b`.
- Reused: completed `NoAdapt`, `CausalRamen`, `OracleHardRamen`, and
  `OracleSoftRankRamen gamma=0` evidence.
- Newly completed after reducing scope: the already-started
  `OracleSoftRankRamen gamma=0.25` cell.
- Not run: `gamma=0.5`, `gamma=1.0`, `OracleSoftWeightRamen`, or any latent
  soft router.

The final `--resume` invocation strictly validated and skipped all seven
materialized runs; it launched no model work.

## Primary results

| Method | Micro | Macro-domain | Worst-domain | Negative windows |
|---|---:|---:|---:|---:|
| NoAdapt | 0.305 | 0.3203 | 0.2344 | reference |
| CausalRamen | 0.330 | 0.3945 | 0.2812 | 1/4 |
| OracleHardRamen | 0.305 | 0.3750 | 0.2188 | 2/4 |
| OracleSoftRankRamen, `gamma=0` | 0.330 | 0.3945 | 0.2812 | 1/4 |
| OracleSoftRankRamen, `gamma=0.25` | 0.330 | 0.3945 | 0.2812 | 1/4 |

Thus:

```text
delta_hard = 0.305 - 0.330 = -0.025 micro
delta_soft(gamma=0.25) = 0.330 - 0.330 = 0.000 micro
```

The `gamma=0` trace matches the `CausalRamen` algorithmic trace across all
200 rows after excluding latency and the expected method-identity routing
fields (`inferred_context`, `num_active_contexts`), plus the four soft-only
diagnostic fields that are absent from `CausalRamen`. Those four
soft-influence diagnostics are exactly zero.

## Mechanistic result

| Diagnostic | Causal / `gamma=0` | Oracle hard | Soft `gamma=0.25` |
|---|---:|---:|---:|
| Returned support, p50 | 93 | 31 | 93 |
| Active classes, p50 | 45 | 20 | 45 |
| Class coverage, p50 | 0.45 | 0.20 | 0.45 |
| ESS, p50 | 25.422 | 8.847 | 25.422 |
| Same-domain support, p50 | 0.3560 | 1.0000 | 0.3636 |
| Cross-domain support, p50 | 0.6440 | 0.0000 | 0.6364 |

The hard control confirms the proposed failure mechanism: perfect domain
purity cuts median support count, class coverage, and ESS substantially. The
soft variant preserves those diversity measures, but its intervention is too
small to affect predictions in this cell:

- nonzero selection change: 73/200 queries;
- mean changed support slots: 2.47%;
- mean rank displacement: 0.031;
- mean same-domain ratio: 0.4917 at `gamma=0`, 0.4939 at `gamma=0.25`;
- mean ESS: 23.829 at `gamma=0`, 23.800 at `gamma=0.25`;
- prediction changes: 0/200.

## Reused evidence

Current schema-compatible raw evidence remains outside git at:

```text
/Users/admin/Documents/NB-Ramen/plans/20260904-latent-soft-routing/reports/gate1-mps-n200
```

The reused adapted run IDs are:

- `cifar100c-block-seed-0-causalramen-dev-mps-n200-cfg-491d76737805-prov-fast-data-6dbea801cbad`;
- `cifar100c-block-seed-0-oraclehardramen-dev-mps-n200-cfg-6608a56e0341-prov-fast-data-6dbea801cbad`;
- `cifar100c-block-seed-0-oraclesoftrankramen-dev-mps-n200-cfg-5aeb84ccc194-prov-fast-data-6dbea801cbad` (`gamma=0`);
- `cifar100c-block-seed-0-oraclesoftrankramen-dev-mps-n200-cfg-8bca17b9166c-prov-fast-data-6dbea801cbad` (`gamma=0.25`).

Historical corroboration on the same stream fingerprint:

- [`CausalRamen paired MPS pilot`](../../20260824-latent-ramen-evidence/reports/causal-ramen-mps-paired-pilot.md)
- [`CIFAR-100-C MPS block pilot`](../../20260824-latent-ramen-evidence/reports/cifar100c-mps-block-n200-pilot.md)

The historical evidence reproduces the same Causal and oracle-hard primary
metrics, but it is corroboration rather than a pooled estimate. The primary
comparison remains one deterministic seed and one 200-sample prefix.

## Decision and limits

Gate 1 required a nonzero oracle-soft strength to improve `CausalRamen` while
preserving support diversity. Diversity was preserved, but accuracy was not
improved, so the gate does not pass.

No speed or broad benchmark claim is made. MPS timing is descriptive, fast
provenance is not a per-run content rehash, and the cell is too small to
generalize across seeds, streams, strengths, or natural-domain datasets.
Luna was not available in this environment; the scope reduction made a new
CUDA submission unnecessary.
