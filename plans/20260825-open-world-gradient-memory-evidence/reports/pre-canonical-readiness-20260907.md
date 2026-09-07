# Pre-canonical readiness — 2026-09-07

## Verdict

**NO-GO for full execution: local fixes and validation pass, but CUDA gates
have not run.** This macOS host has no CUDA device. No current-schema CUDA
smoke, causal-sensitivity result, or full-matrix measurement is claimed.
No method settings were tuned and no canonical method implementation changed.

This report implements the actionable local corrections from
`NB_Ramen_pre_full_cuda_readiness_fix_report.md`. No `ak` skills were used.

## Revision and runtime identity

- Reviewed/base commit: `2d5d205bbc3672ceacf47cf28a0b8bbbbeeee821`.
- Canonical frozen commit: **pending**. Validation ran on uncommitted changes
  from a clean starting worktree. The implementation can be committed and
  pushed, but the required post-smoke canonical freeze has not occurred.
- Tested executable: `/Users/admin/miniconda3/envs/nb-ramen/bin/python`.
- Host: macOS 15.7.7, arm64; PyTorch 2.4.1; CUDA available: **false**;
  `torch.version.cuda`: **null**; NVIDIA GPU identity: **unavailable**.
- `nvidia-smi` is not installed; its archive explicitly records unavailability.
- Tested executable-source SHA-256 fingerprint:
  `9eaf60d75d4359846bcf6d129fab0fc05bf5102e3c976edf3a883f5b4c71a027`.
  The source inventory is in `runtime/source-state.json`; this supplements,
  but does not replace, a clean frozen commit.

Artifacts are outside version control in
`evidence/pre-full-readiness-20260907/`. Its `runtime/` contains device,
pip-freeze, Git head/status, source identity, preflight, provenance, test logs,
config hashes, canonical plan, smoke plan, and `gate-results.json`.
Plans contain paths for this local checkout; regenerate paths/interpreter and
repeat preflight/provenance on the eventual CUDA runner.

## Completed corrections

- A shared evaluator summary function now compares `admission_prediction`
  with `known_label_or_minus_one` only on ID rows for open-set runs.
  `src/main.py` and the strict matrix validator use that same definition.
- Open-set summaries retain total admitted/rejected counts, admission rate,
  and mean normalized entropy, and add separate admitted/rejected ID and OOD
  counts, ID pseudo-label accuracy, and semantic OOD fractions.
  OOD fractions use all rows in each admission group as their denominator;
  ID accuracy uses only ID rows. Empty denominators produce `null`.
- Open-set summaries omit the old unqualified pseudo-label accuracy and
  `admitted_contamination_rate` fields. Closed-set diagnostics are unchanged.
- Trace schema remains **v3**; summary schema is **v4**. Strict validation
  rejects summary v3 rather than silently reinterpreting historical evidence.
- Regression tests cover original IDs `[5, 10]` mapped to `[0, 1]`, mixed
  ID/OOD admission groups, empty ID groups, mutated summary fields, and v3
  rejection. The ordered evaluator itself is exercised end to end.
- Current runtime/thesis docs and operational plan files now name the primary
  protocol **legacy-compatible batch-atomic mixed-domain TTA**. Entire batches
  are admitted before retrieval; batch=100 can cross block=64 boundaries.
  Strict causality remains a separate sensitivity/control experiment.
  Historical pilot reports and raw artifacts were not rewritten.
- Independent read-only review found no actionable defect. Its default Python
  had a broken PyTorch installation; the pinned `nb-ramen` interpreter above
  successfully ran the full suite, including evidence and evaluator tests.

## Tests and local gates

All commands below ran from the repository root with the tested executable
above selected as `python` and `PYTHONPATH=src`.

| Check | Result | Archive |
| --- | --- | --- |
| Three new focused regression tests | PASS, 3 tests, exit 0 | Initial terminal check |
| Requested focused suite | PASS, 138 tests, exit 0 | `runtime/focused-tests.log` |
| Full unit discovery | PASS, 344 tests, exit 0 | `runtime/full-tests.log` |
| Official CIFAR-100-C deep preflight | PASS, exit 0 | `runtime/cifar100c-deep-preflight.json` |
| Fast dataset provenance and cached model checksum | PASS, exit 0 | `runtime/artifact-provenance.json` |
| Canonical config digests and semantic surfaces | PASS, all six unchanged | `runtime/config-sha256.json` |
| Canonical dry plan | PASS, 252 unique runs, exit 0 | `runtime/canonical-plan.json` |
| CUDA runtime probe | BLOCKED: CUDA false | `runtime/device.json` |
| Current-schema CUDA smoke validation | NOT RUN | No smoke artifacts exist |
| Clean post-smoke freeze | PENDING | `runtime/git-status.txt` is nonempty |

```sh
PYTHONPATH=src python -m unittest \
  tests.test_entropy_gated_ramen tests.test_consensus_ramen \
  tests.test_oracle_id_gradient_ramen tests.test_oracle_consensus_ramen \
  tests.test_open_set tests.test_open_set_metrics \
  tests.test_open_set_consensus_analysis tests.test_ordered_stream_evidence \
  tests.test_experiment_matrix

PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'

PYTHONPATH=src python -m runtime.preflight \
  --data-root /Users/admin/data --dataset CIFAR100C --deep --json

PYTHONPATH=src python -m runtime.experiment_matrix \
  --open-set-consensus --device cuda --artifact-provenance fast \
  --data-root /Users/admin/data \
  --evidence-dir evidence/pre-full-readiness-20260907/open-set-cifar100c-canonical
```

The canonical dry plan validates frozen config hashes and parsed surfaces.
It does not execute model inference or prove a CUDA stream fingerprint.
The separate smoke plan also checks the frozen v2 split JSON byte digest.

## Data and model provenance

- Data root: `/Users/admin/data/corruption/CIFAR-100-C`.
- Fast inventory verification: 21 files; `verified_exact=false`. Deep
  preflight separately validates official-size corruption arrays and labels.
  Fast mode does not rehash every array during this readiness check.
- Inventory content root SHA-256:
  `115529dc4e957b58ac383bc1f7d71470ff1a26af3f3e784718aac7a38a102bbc`.
- Sidecar SHA-256:
  `1aacb02612a634db5aebaf92332393c4f6f56ab84cbe72a2efb7216c0f0fdbc1`.
- Persisted official acquisition anchor: Zenodo 3555552, archive MD5
  `11f0ed0f1191edbf9fa23466ae6021d3`, size 2,918,473,216 bytes.
- Model: `/Users/admin/.cache/clip/ViT-B-16.pt`, size 350,837,078 bytes;
  freshly computed SHA-256 equals the pinned official digest
  `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.

## Prepared CUDA smoke identities — not executed

`runtime/smoke-plan.json` contains 14 explicit direct `src/main.py` commands
and their expected `ExperimentRun` identities for strict validation. These
are **noncanonical prefixes**, separate from the 252-run full plan.
All use CIFAR100C, CLIP ViT-B/16, CUDA, fast provenance, block stream,
seed=0, block=64, source budget=400/domain, prefix=256, and metric
window/stride=50. Default split is v1, OOD=.3, batch=100.

| Purpose | Planned run IDs | Paired NoAdapt |
| --- | --- | --- |
| Seven-method current-schema smoke | `prefull-noadapt`, `prefull-ramen`, `prefull-entropy`, `prefull-oracle-drop`, `prefull-oracle-id`, `prefull-consensus`, `prefull-oracle-consensus` | `prefull-noadapt` |
| Ramen B=1 | `prefull-noadapt-b1`, `prefull-ramen-b1` | `prefull-noadapt-b1` |
| CausalRamen B=100 | `prefull-causal-b100` | `prefull-noadapt` |
| Remapped v2 split | `prefull-v2-noadapt`, `prefull-v2-entropy` | `prefull-v2-noadapt` |
| OOD=0 OracleID control | `prefull-ood0-noadapt`, `prefull-ood0-oracle-id` | `prefull-ood0-noadapt` |

Every adapted command carries the selected config path and SHA-256. The v2
pair additionally carries registered split path/SHA-256 locks. Baselines
precede their adapted runs, and each reference matches evaluator batch size;
the B=1 control cannot reuse the B=100 reference under the strict contract.

Smoke stream fingerprints: **unavailable until execution**. No historical
fingerprint is substituted. Once run, validate each expected identity using
`runtime.experiment_matrix.validate_completed_run`, require matching paired
fingerprints, and archive all four artifacts per run. Recompute v2 ID
pseudo-label accuracy from known model indices. Check all required open-set,
oracle, Consensus, and admission evidence groups through the strict validator.

Causal-sensitivity result: **not measured**. Compare Ramen B=100 versus B=1
for packaging sensitivity, then Ramen B=1 versus CausalRamen B=100 for causal
consistency. Record prediction disagreement, ID accuracy and score differences
on identical sample order; investigate numerical differences before declaring
consistency. In OOD=0, require zero retrieved OOD fractions and zero defined
Ramen/OracleID sign disagreement; paired nonzero directions should have cosine
approximately one. Preserve undefined zero-vector diagnostics as null.

## Remaining launch conditions

1. Attach a configured CUDA runner and repeat runtime/data/model checks there.
2. Execute and strictly validate all prepared CUDA smokes, including v2,
   OOD=0, and causal/package sensitivity. Record real run fingerprints/results.
3. Pass focused/full tests on that runtime, freeze a clean reviewed commit,
   and archive its head plus an empty `git-status.txt`.
4. Regenerate the canonical dry plan on the runner, require exactly 252
   locked runs, then execute/resume the full matrix. No full launch was
   attempted here because the required CUDA gates remain unsatisfied.

Do not retune `tau=0.2`, require Consensus to win a smoke, or include prefixes
in thesis effect-size claims. Treat `g_ID` as an ID-only oracle reference,
not a guaranteed optimal gradient. After full execution, report iid_mixed,
block, and recurring separately, including null/negative outcomes. The
96-run robustness study still requires its v2 smoke gate; DomainNet data and
CUDA evaluation remain outstanding separate studies.
