# Independent audit — open-set MPS pilot (2026-08-25)

## Scope and verdict

This audit independently read the thesis direction, the pilot report, and the
JSON artifacts under `evidence/open-set-mps-pilot/`. It recomputed ID accuracy
directly as `mean(correct | is_ood == false)` from every relevant
`trace.jsonl`; it did not rely on reported summary values.

**Recommendation: YES — proceed to the preregistered canonical CUDA matrix.**
The traces support the narrow directional and conditional-consensus signals
needed to justify that next experiment. **NO — do not use this pilot as a
benchmark, effect-size, or thesis-performance claim.** Its dataset provenance
is deliberately disabled, its inputs are noncanonical, and each cell contains
only a 128-sample prefix.

## Recomputed evidence

### Directional oracle block pilot

The following rows recompute the table in
`open-set-mps-directional-pilot-20260825.md` from the traces listed below.
Each run has 128 records; the ID denominators are 62, 64, and 63 respectively.

| Seed | Ramen ID ACC | OracleID ID ACC | OracleDrop ID ACC | OracleID − Ramen | Stream fingerprints match? |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 17.7419% (11/62) | 19.3548% (12/62) | 19.3548% (12/62) | +1.6129 pp | yes |
| 1 | 23.4375% (15/64) | 25.0000% (16/64) | 25.0000% (16/64) | +1.5625 pp | yes |
| 2 | 25.3968% (16/63) | 25.3968% (16/63) | 25.3968% (16/63) | +0.0000 pp | yes |
| Mean of seed percentages | 22.1921% | 23.2510% | 23.2510% | +1.0589 pp | — |

Evidence: [Ramen traces](../../../evidence/open-set-mps-pilot/pilot128-s0-ramen/trace.jsonl),
[OracleID traces](../../../evidence/open-set-mps-pilot/pilot128-s0-oracle-id/trace.jsonl),
[OracleDrop repaired traces](../../../evidence/open-set-mps-pilot/pilot128v2-s0-oracle-drop/trace.jsonl),
and the corresponding `s1`/`s2` sibling directories. The fingerprints are,
per seed, `172f4d…596bf`, `bb308f…5e676`, and `770253…1277`; they agree within
each Ramen/OracleID/OracleDrop seed comparison.

For `OracleID`, all 128 diagnostic values are non-null in each seed. Direct
means of trace fields reproduce the reported values exactly to four decimals:

| Seed | 1 − `ramen_vs_oracle_id_cosine` | sign disagreement | retrieved OOD fraction | retrieved OOD weight fraction |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.1903 | 0.1977 | 0.4283 | 0.4282 |
| 1 | 0.2026 | 0.2098 | 0.4808 | 0.4111 |
| 2 | 0.1442 | 0.1727 | 0.5208 | 0.3775 |

The source summaries, including the independently reproducible aggregates,
are [seed 0](../../../evidence/open-set-mps-pilot/pilot128-s0-oracle-id/summary.json),
[seed 1](../../../evidence/open-set-mps-pilot/pilot128-s1-oracle-id/summary.json),
and [seed 2](../../../evidence/open-set-mps-pilot/pilot128-s2-oracle-id/summary.json).

### Three-stream Consensus follow-up

Recomputed means of the three seed-level ID accuracies agree with the report:

| Stream | NoAdapt | Ramen | ConsensusRamen | Consensus − Ramen |
| --- | ---: | ---: | ---: | ---: |
| `iid_mixed` | 19.4160% | 22.7494% | 22.2285% | −0.5208 pp |
| `block` | 18.5047% | 22.1921% | 24.8549% | +2.6628 pp |
| `recurring` | 18.6025% | 22.1089% | 22.6443% | +0.5354 pp |
| Mean of all nine cells | 18.8411% | 22.3501% | 23.2426% | +0.8925 pp |

For every `(stream_mode, seed)` triplet, the NoAdapt, Ramen, and Consensus
manifests have an identical stream fingerprint and 128 trace records. Example
paired block seed 0: [NoAdapt manifest](../../../evidence/open-set-mps-pilot/pilot128v4-block-s0-NoAdapt/manifest.json),
[Ramen manifest](../../../evidence/open-set-mps-pilot/pilot128v4-block-s0-Ramen/manifest.json),
and [Consensus manifest](../../../evidence/open-set-mps-pilot/pilot128v4-block-s0-ConsensusRamen/manifest.json).
The other eight paired cells are the `pilot128v4-{iid_mixed,block,recurring}-s{0,1,2}-*` siblings.

### Other reported pilot checks

- The entropy-gated block traces reproduce 12.9032%, 28.1250%, and 22.2222%
  ID accuracy, mean 21.0835%; their summaries report admission rates 9.4%,
  11.7%, and 16.4%.
- The `tau=0.2` consensus traces reproduce 22.5806%, 25.0000%, and 26.9841%,
  with mean mask rate 45.2950%. The `pilot128v3-s0-consensus-applied` trace
  marks consensus applied on all 128 records and ends at 10,355,200 bytes.
- The reported 32.9 ms (Ramen) and 38.8 ms (`tau=0.2`) descriptive averages
  reproduce from their three `summary.json` forward-latency means: 32.9012 ms
  and 38.7977 ms.

## Finding: one minor numerical correction

The report's table states that `tau=0.6` has a mean mask rate of **4.1%**.
The three authoritative summaries are 4.8736%, 4.7052%, and 3.6660%; their
arithmetic mean is **4.4149%** (4.4% to one decimal), not 4.1%.
See [seed 0 summary](../../../evidence/open-set-mps-pilot/pilot128-s0-consensus-v0/summary.json),
[seed 1 summary](../../../evidence/open-set-mps-pilot/pilot128-s1-consensus-v0/summary.json),
and [seed 2 summary](../../../evidence/open-set-mps-pilot/pilot128-s2-consensus-v0/summary.json).
The same report's stated per-seed range (3.7–4.9%) and its accuracy result are
correct. This is not decision-changing, but the table should be corrected
before reuse.

## Provenance and reproducibility limits

1. Every audited pilot manifest explicitly says `artifact_provenance: off` and
   `artifacts.status: unavailable`; this is correctly disclosed in the pilot
   report. The local artifact [README.json](/Users/admin/data/corruption/CIFAR-100-C/README.json)
   self-identifies as 320 Hugging Face CIFAR-100 test examples with lightweight
   approximations, not official CIFAR-100-C. It is therefore impossible to
   independently establish official-dataset lineage from these artifacts.
2. The manifests preserve a dirty-worktree source fingerprint, but snapshots
   are not globally identical across the historical oracle, ablation, and v4
   follow-up runs. Input pairing is verified within each stated comparison;
   cross-era comparisons should remain descriptive. The v4 three-stream
   triplets do share the same source fingerprint within each cell.
3. The stream realized OOD fraction is not always exactly 0.5 after the
   deterministic 128-sample prefix (e.g. block seed 0 is 66/128 = 0.515625).
   This is represented in manifests and is harmless for same-stream pairing,
   but it reinforces that this is not the fixed-exposure canonical matrix.

## Supported and unsupported claims

Supported: OOD-containing supports changed the measured Ramen update direction
in these traces; OracleID was non-worse than Ramen in the three block seeds;
and ConsensusRamen was heterogeneous across the three small stream modes,
with its favorable mean concentrated in `block`.

Unsupported: any claim of official CIFAR-100-C performance, CUDA performance,
generalization, a reliable effect size, or a universal Consensus improvement.
The pilot report mostly avoids these claims; its only identified factual issue
is the `tau=0.6` 4.1% mask-rate table entry.
