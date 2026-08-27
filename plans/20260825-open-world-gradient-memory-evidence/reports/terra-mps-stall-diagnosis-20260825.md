# Bounded MPS ConsensusRamen partial-run diagnosis — 2026-08-25

> **Superseded operationally by [final evidence status](./final-evidence-status-20260826.md).** This remains the record for the incomplete 100-row run; the later completed 32-sample MPS smoke supersedes it as the local mechanics evidence, not as canonical effectiveness evidence.

## Verdict

The evidence is **insufficient to classify this as a crash or deadlock**. It is most consistent with extremely slow, synchronized MPS execution of the second 100-item `ConsensusRamen` forward. Unified-memory/kernel pressure is plausible but unproven. There is no recorded exception, OS diagnostic, process, or partly written second batch establishing a crash, MPS deadlock, or OOM.

`src/main.py` synchronizes MPS before and after `tta_model(image)`, then writes trace rows only after that forward returns. The 100 rows prove batch 1 completed; the seven-minute wait provides no internal progress point for batch 2. Do not resume, repair, or analyze this incomplete directory as a completed run.

## Artifact and provenance checks

| Check | NoAdapt | ConsensusRamen | Result |
| --- | --- | --- | --- |
| Trace rows | 200 | 100 | NoAdapt completed; Consensus contains exactly one batch |
| Exported stream fingerprint | `c43d9b2ff6faf5537e5bae954bb4ac11fefd39a4abdef0edfb3fe1a872af3eea` | same | paired stream identity holds |
| Trace schema / timesteps | v2 / 0–199 | v2 / 0–99 | internally well-formed prefixes |
| Git source | `cd97063ab37d4fd29d9f5f22601ad471d19eee8c`, clean | same | same source snapshot |
| Dataset provenance | fast verified; root digest `115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc` | same | official inventory bound; fast is not a content rehash |
| CLIP ViT-B/16 | verified SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` | same | same pinned OpenAI checkpoint |

The Consensus manifest names the completed NoAdapt trace as its reference. Its first 100 records exactly match NoAdapt’s first 100 in timestep, sample, domain, original/known label, and OOD flag (74 ID / 26 OOD). The copied `stream.json` files are byte-identical by SHA-256: `d8faad1c731e96bc4a8918dfbe6bd4a945d7176611302d0a84e7afa96687f428`.

Observed file SHA-256 values: NoAdapt manifest `4b562efc61aaafc8d83823daf2b9c8bb38ac561c53bb85b4394eb7274ccd1ce5`, trace `82168bdc41755e2debccb81fd7ee883067311ddb0f351a690803292ea26d9e65`, summary `96d42bb2cea060719722901566709b9821f69231e3d066b39f77e06b1fc2e0ce`; Consensus manifest `4ff1d980983136a469f1856ab23b67a0d0fcdbb3cc4f74095c57b999c893e016`, trace `0d008f7009bc2b492f917b7bddc6d56577c012cc202cf0bdfafbe1f282562c73`. Consensus has no `summary.json`, as expected when it never reached the summary writer.

## What the partial trace establishes

- The first Consensus batch completed with finite v2 fields, 100 stream positions, 57 active class caches, hard masking on all 100 samples, 41.53–42.75% retained coordinates, and 8,090,000 logical retained-support bytes.
- Its completed synchronized forward took 336.906 s (3,369.06 ms/sample). This is a first-batch timing, not completed-run throughput.
- The paired first-prefix counts are 40/100 correct for Consensus and 36/100 for NoAdapt. They validate trace pairing only; they are not a 200-sample, ID-accuracy, AUROC/FPR95, or method-selection result.

NoAdapt completed normally: 200 rows, 5.997 s synchronized forwards total, 33.35 samples/s, and sampled post-batch MPS maximum 641,331,712 bytes. Those metrics are valid for its completed cost-limited prefix only.

## Diagnosis

`ConsensusRamen.forward` performs CLIP feature forward, entropy backward, by-sample gradient extraction, cache admission, per-nonempty-class support queries, a stacked per-class gradient/sign-agreement aggregation, masked gradient copies back to every by-sample LayerNorm, SignSGD step, a post-update CLIP forward, and reset. The by-sample CLIP path creates independent LayerNorm affine parameters up to `max_batch_size`, so 100 is materially costlier than a conventional image batch.

At batch 1 completion, 57 caches are nonempty. Batch 2 can query at least that many caches and perhaps more. Consensus constructs a per-class gradient tensor before agreement/masking; its cost need not resemble batch 1. The 8.09 MB value is logical retained memory, not allocator peak, and cannot rule memory pressure in or out.

The prior official CIFAR-100-C MPS block pilot used ViT-B/16, batch 100, and prefix 200. It measured 204.206 s for legacy Ramen across 200 samples and documented 3.4 minutes for that bounded run; related MPS runs reached 5.7 minutes and explicitly recorded severe unified-memory/kernel stalls. The current first Consensus batch alone took 336.906 s. A seven-minute wait for a larger-cache second batch is unacceptable operationally but still compatible with known severe MPS throughput; it is not affirmative deadlock evidence.

No relevant local `.log`, `.err`, or `.out` existed in the evidence root/repository. No matching Python process remained during inspection, and the available recent system-log query contained no MPS/Metal, Python-kill, memory-pressure, or crash record. These absences leave the cause unconfirmed.

## Safe bounded next run

For a mechanics-only retry use **`--batch_size 8 --max_batch_size 8 --max_eval_samples 32`**, retain every other argument/configuration/provenance setting, and use a new run ID/evidence directory. This preserves the actual ConsensusRamen algorithm, by-sample gradient route, cache admission, hard mask, optimizer, and post-update forward while reducing by-sample and consensus temporary dimensions 12.5× and limiting the job to four forwards.

Batch-atomic support composition changes, so this is a survivability/mechanics check—not a replacement for batch-100 effectiveness evidence. `--max_eval_samples` bounds work but does not impose a wall-clock timeout; use an external operational time limit if required and preserve any partial artifact. If 32 completes, make a separate `8/8/64` run next; do not increase batch and prefix simultaneously.

## Metrics status

Valid: completed NoAdapt prefix metrics; shared stream/provenance identity; and first-batch Consensus mechanics/latency observations. Invalid from the incomplete Consensus artifact: overall/micro/macro/worst and ID accuracy; OOD AUROC/FPR95/H-score; sliding windows, recovery, negative-adaptation rate; full-run consensus aggregates; final/max method memory; sampled MPS device memory; completed-run latency/throughput; and any NoAdapt-versus-Consensus conclusion. The 100-row prefix should not be used as an effectiveness claim because adaptation is batch-dependent and the planned unit is 200 samples.

## Conclusion

Treat this as a paired, provenance-verified but incomplete MPS mechanics artifact. Severe MPS throughput is the leading explanation; memory/kernel pressure is possible; crash/deadlock evidence is absent. A separately identified `8/8/32` retry is the safe next observation. Canonical effectiveness and runtime conclusions still require completed CUDA artifacts.
