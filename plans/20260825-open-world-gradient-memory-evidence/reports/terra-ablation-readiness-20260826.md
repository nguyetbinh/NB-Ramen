# Consensus ablation readiness — 2026-08-26

> **Superseded for current status by [final evidence status](./final-evidence-status-20260826.md).** This pre-results review remains a historical record of the frozen 72-run design; its then-open implementation-lock concerns have since been addressed. No ablation run results exist.

## Status

**Execution set selected, but do not launch canonical evidence until the two fail-closed provenance/hash gaps below are resolved operationally or in code.** This review is pre-results and makes no effectiveness claim.

## Frozen held-out execution set

Run the same contamination level across every thesis-relevant stream shape, not the largest-looking pilot cell. This is the smallest grid that can test the report's stream-structure claim without choosing a stream from outcomes.

| Fixed dimension | Selection |
| --- | --- |
| Dataset/split | `CIFAR100C`, `open-set-cifar100-split-v1` |
| Held-out cells | (`iid_mixed`, 0.3), (`block`, 0.3), (`recurring`, 0.3) |
| Seeds | 0, 1, 2 in every cell |
| Source exposure | 400 per source domain |
| Device/stream | CUDA; full stream (`max_eval_samples` omitted); block size 64 |
| Artifact/data | one verified official CIFAR-100-C acquisition; `--artifact-provenance exact` |
| Evidence root | new, otherwise empty `evidence/open-set-consensus-ablation-heldout-v1` |

Each cell/seed schedules one `NoAdapt` run first and exactly these seven adapted runs against that baseline trace: `Ramen`, `ConsensusRamen`, `ConsensusRamenSoft`, `ConsensusRamenNoSelf`, `ConsensusRamenTau060`, `ConsensusRamenMin2`, and `ConsensusRamenMin4`. The frozen workload is `3 streams × 1 ratio × 3 seeds × (1 baseline + 7 adapted) = 72 runs`: 9 NoAdapt and 63 adapted. Do not add an OOD-ratio sweep, a fourth seed, or a replacement cell after inspecting any result. The primary seven-method 252-run matrix remains separate.

Submit exactly one plan per row (the planner intentionally accepts one cell):

```sh
for stream in iid_mixed block recurring; do
  PYTHONPATH=src python -m runtime.consensus_ablation_matrix \
    --stream "$stream" --ood-ratio 0.3 --seed 0 --seed 1 --seed 2 \
    --device cuda --stream-block-size 64 --artifact-provenance exact \
    --per-domain-source-budget 400 --data-root "$RAMEN_DATA_ROOT" \
    --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-consensus-ablation-heldout-v1"
done
```

Treat the emitted JSON plans and command lists as run manifests to retain with the evidence. Execute `NoAdapt` before the seven methods for each cell/seed; every adapted invocation must retain the generated `--reference_trace` for that same cell/seed. No trace may be shared across stream, ratio, or seed.

## Identity and configuration audit

All five required §27 identities exist as distinct CIFAR-100-C YAML files, are selectable aliases of the tested `ConsensusRamen` implementation, and are included by `CONSENSUS_ABLATION_METHODS`.

| Identity | Required semantics | SHA-256 (full; planner uses first 12) |
| --- | --- | --- |
| `ConsensusRamenSoft` | `soft_weight`, seeded Bernoulli coordinate admission, gamma 1, seed 1729; tau .2/C_min 3/current true retained for shared surface | `5cbd2b2cee48491016d547030fc65fa50ff0e16a775d98d574c141035bf5e356` (`5cbd2b2cee48`) |
| `ConsensusRamenNoSelf` | hard mask, tau .2, C_min 3, history-only (`include_current: false`) | `2abe743078b251e9d3c04cccc334112a71298b269da3c014e246bd752f88c071` (`2abe743078b2`) |
| `ConsensusRamenTau060` | hard mask, tau .6, C_min 3, current true | `a8b9adb2e33697c1652ef2c55f30132daa4a96c08295a0a15468c7380fd30736` (`a8b9adb2e336`) |
| `ConsensusRamenMin2` | hard mask, tau .2, C_min 2, current true | `32ded9874f2a14dbaf8568ab5b1756b84eb96ab977c69bd7bf7378220e7bd1bf` (`32ded9874f2a`) |
| `ConsensusRamenMin4` | hard mask, tau .2, C_min 4, current true | `82a5ba1f498edcc3fa899078b0e35c05d61f833def4b36558140521f8f5fe006` (`82a5ba1f498e`) |

The immutable primary comparator is `ConsensusRamen`: hard mask, tau .2, C_min 3, current true; hash `8a9d6fe4bb663653bf275fef4ecafe4d91ddf7415fc45c7eb3ea4634ac6bb34e` (`8a9d6fe4bb66`). `Ramen` is the ordinary aggregation control (`54c124be79a3c1536a8a95c68f34b41b31d84d40972f54fcd5b2c0016552ef27`, planner prefix `54c124be79a3`). Hash each file immediately before launch and record the full digest plus the repository commit; abort if any differs.

The planner reads and binds the selected config path, parsed data, 12-character content hash, device, full/limited-stream token, provenance mode, data-root hash, OOD ratio, and source budget into the run identity. `build_command()` detects a config changed after in-process planning. Planner tests prove expected aliases, distinct alias hashes, same-seed NoAdapt pairing, CUDA requirement, fixed supported ratio/stream values, compatible source budget, and plan-only CLI behavior.

## Required evidence and analysis fields

For every completed run, retain `manifest.json`, `stream.json`, `trace.jsonl`, `summary.json`, `results.csv`, emitted plan JSON, full config digests, commit/environment/CUDA details, and exact artifact-provenance sidecar. Completion validation must pass, including stream fingerprint, full-trace length, config equality, paired negative-adaptation recomputation, open-set split/realized ratio, and exact-CUDA allocator evidence.

Predeclare descriptive, seed-level paired summaries by identity and stream: ID accuracy (including worst-domain ID accuracy), ID-only negative adaptation, pre/post OOD detection, post-shift ID recovery where applicable, retained memory, synchronized forward latency/throughput, and consensus diagnostics (mean/p10/p50 agreement, retained-coordinate rate, active-class count, and applied fraction). Compare each ablation to v0 and report `Ramen` alongside them; do not select a winner or tune a parameter from these results. The directional oracle fields in §28 F1 are not expected for this ablation set because none of these methods is an oracle directional control.

## Findings and blockers

1. **Canonical does not mean full/exact in this planner (blocking).** `build_consensus_ablation_matrix(canonical=True)` only requires CUDA. It permits `artifact_provenance="fast"`, finite `max_eval_samples`, a non-64 block size, any compatible positive source budget, and arbitrary evidence directory/data root. The command above supplies correct values, but the code does not reject a weakened canonical plan. Add a strict held-out builder/CLI mode (or independently signed scheduler wrapper) before treating runs as canonical thesis evidence.
2. **Hashes are recorded, not preregistered constants (blocking).** The planner rejects missing configs and detects mutation within a plan object, but accepts whatever contents exist at planning time. No expected full digest/commit is asserted in planner code or tests. Freeze the hashes in an immutable manifest/strict test before launch; values above are this review's proposed lock.
3. **Test coverage is incomplete.** `tests/test_consensus_ablation_matrix.py` does not assert the five full config surfaces, expected hash values, exact provenance/full-stream/block-size/budget lock, or output plan IDs for all three frozen cells. Add these contract tests. Alias and mechanics tests cover implementation sharing and soft Bernoulli/SignSGD behavior, but the current local Python cannot load Torch (`libtorch_cpu.dylib` missing), so that focused mechanics suite was not runnable here.
4. **Workspace state must be frozen first.** Relevant runtime and evaluator files are already modified/uncommitted by other work. Record a clean reviewed commit (and do not run against a moving worktree); this audit did not alter those files.

## Validation performed

`PYTHONPATH=src python -m unittest -q tests.test_consensus_ablation_matrix` passed (4 tests). A non-executing planner invocation for the first frozen cell with CUDA/exact/full settings emitted the expected plan. No training, download, or CUDA job was run.

Status: DONE_WITH_CONCERNS

Summary: The five §27 identities and paired planner are present; the outcome-independent 72-run grid above is frozen for execution once strict provenance and configuration-lock enforcement is in place.

Findings: The primary concerns are the planner's permissive canonical mode and non-preregistered config digests.

Blockers: Do not label results canonical until exact provenance/full-stream and expected digest/commit locks are enforced and a stable reviewed revision is selected.
