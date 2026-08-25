# Reproducible experiment runtime

## Evidence schema versions

Evidence files are independently versioned. Run manifests remain at schema
version 1, and exported stream schedules remain at format version 1. Trace
rows use schema version 2: compared with v1, v2 requires `memory_bytes` for
each sample (a non-negative retained-memory byte count or `null` when the
method cannot provide it). Completed-run summaries use schema version 2:
compared with v1, they include explicit device-memory, method-memory,
forward-latency, throughput, and retrieval-latency evidence blocks. Strict
resume and reference-provenance validation accept only these current trace and
summary versions; prior v1 trace or summary artifacts must be regenerated.
Before a trace can be used as a direct-CLI negative-adaptation reference, its
current-schema rows (including predictions and derived correctness) are checked
against the sibling stream, manifest, and summary accuracy/domain/sliding-window
evidence. These three sibling documents are mandatory regular files in the
reference run directory; symlinks and files replaced while being opened are
rejected.
The reference manifest must identify a verified `NoAdapt` run with the same
dataset, model checkpoint, canonical data root, resolved device, evaluation
mode, batch/window settings, and verified dataset/model artifact reports as the
current run. Its config must be the canonical `NoAdapt` config; an adapted
method's config is not a valid substitute. Legacy manifests missing any of
these identity or provenance fields are rejected.

`LatentRamen` and the separately named `EntropyGatedLatentRamen` may extend a
v2 trace with the all-or-none fields `admission_prediction`,
`admission_normalized_entropy`, and `admitted_to_memory`. Their optional
`admission_diagnostics` summary is recomputed from the trace during strict
resume. For the gated method, resume additionally requires every decision to
equal `admission_normalized_entropy <= max_normalized_entropy` from the exact
hashed config. Older v2 traces without these optional extensions remain valid.

`LatentRamen` also supports an opt-in diagnostic config value
`retrieval_profile: causal_sync_v1`. Profiled traces add the complete optional
set `retrieval_profile`, `retrieval_elapsed_ms`,
`retrieval_candidate_count`, `retrieval_eligible_candidate_count`,
`retrieval_returned_support_count`, and `retrieval_active_class_count`.
Ordinary runs default to `off` and keep retrieval latency explicitly
unavailable. The profiled interval synchronizes immediately before and after
each causal one-item memory query; this deliberately perturbs execution, so
profile-on end-to-end latency is diagnostic and is not comparable to normal
method latency. Strict resume replays the memory buckets and recomputes all
counter distributions and timing summaries. Research configs live below
`cfg/research/phase04-causal-retrieval/`.

Post-shift recovery is computed only for `block`, `recurring`, and `bursty`
streams, whose contiguous domain episodes support its full-window definition.
`novel_domain` remains `not_applicable`: it mixes eligible domains before and
after the release event, so release-specific recovery requires a separate
metric.

## Artifact provenance

Direct runs accept `--artifact-provenance {off,fast,exact}`. `off` (the direct
CLI default) records explicit unavailable model and dataset evidence. `fast`
and `exact` fail closed before model or dataset loading, and currently support
only `CIFAR100C` and `DomainNet`: they verify the cached official CLIP
checkpoint against an in-repository table of OpenAI HTTPS URLs and SHA-256
digests, plus the canonical local dataset inventory sidecar. No signature is
claimed for the sidecar: its SHA-256 binds the locally generated inventory,
while `exact` additionally rehashes every dataset file. `fast` validates the
sidecar, complete path/size inventory, and every file through a non-following
descriptor without rereading all file contents.
Direct runs using `--reference_trace` must select `fast` or `exact`; references
are rejected when artifact provenance is disabled.

CIFAR-100-C acquisition evidence is accepted only for Zenodo record `3555552`
(DOI `10.5281/zenodo.3555552`), its API content URL, size `2918473216`, and MD5
`11f0ed0f1191edbf9fa23466ae6021d3`. A generic archive checksum helper remains
available for integrity checks, but its output is not treated as publisher
provenance. The resulting run manifest archives the complete verification
report: expected and actual model digests, trusted URL, checkpoint size and
path, dataset root and sidecar paths/digests, root digest, file count, mode,
and pinned acquisition evidence where applicable.

The experiment matrix defaults to `fast`; its provenance mode is part of the
run identity and command line. Strict matrix resume checks the persisted
artifact evidence and mode, without rehashing the live model or dataset.
Resume validates the complete report schema and its internal trust-anchor and
digest consistency.

## Stream block-size identity

Direct `src/main.py` runs use `--stream_block_size`; the matrix and analyzer
use `--stream-block-size`. The value defaults to 64. The default is canonical: it adds no
run-ID token and matrix commands omit the flag, preserving established run IDs
and commands. A nondefault block size is permitted only with
`--max_eval_samples`; it is explicitly cost-limited and binds the run identity
through a path-safe `blk-N` token. Strict resume requires the manifest argument
and exported stream `metadata.block_size` to equal the planned block size.
Negative-adaptation references require the same block size as the current run.

For verified runs, the same fast/exact checks run immediately before loading
and again after model and dataset construction; the reports must be identical
before the manifest is created. The model loader receives the exact verified
checkpoint pathname rather than a symbolic CLIP model name, so package URL
metadata cannot redirect that load. Matrix identity also binds the canonical
data root (using a short path hash in the run ID), and resume requires both the
manifest argument and dataset report root to match it exactly.

These checks narrow the loader race window but cannot make third-party loaders
transactional: a privileged or concurrent writer could still replace bytes
after the second check. Dataset traversal is pathname-based, so concurrent
mutation of ancestor directories is also outside the verifier guarantee.
Scientific runs should use read-only artifact directories or filesystem
snapshots while executing.

This repository includes separate pinned Conda environments for CPU/MPS and
Linux CUDA, plus a dependency-light preflight check. By default, preflight only
inspects filesystem layout and optional runtime metadata; it does not load image
files or NumPy arrays. Opt in to semantic validation with `--deep` before a
scientific run.

## Create the environment

From the repository root:

```shell
conda env create -f environment.yml
conda activate nb-ramen
```

For a Linux CUDA 12.1 runner:

```shell
conda env create -f environment-cuda.yml
conda activate nb-ramen-cuda
```

Both files pin Python 3.11, PyTorch 2.4.1, torchvision 0.19.1, and the official
OpenAI CLIP repository at commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. The CUDA manifest additionally
pins `pytorch-cuda=12.1`.

## Validate data before an experiment

The current shell scripts run `CIFAR10C`, `CIFAR100C`, `ImageNetC5K`, and `DomainNet`. Validate their expected data directories before starting an experiment:

```shell
PYTHONPATH=src python -m runtime.preflight \
  --data-root ~/data \
  --dataset CIFAR10C \
  --dataset CIFAR100C \
  --dataset ImageNetC5K \
  --dataset DomainNet
```

Use JSON for archival or automated checks:

```shell
PYTHONPATH=src python -m runtime.preflight \
  --data-root ~/data \
  --dataset ImageNetC5K \
  --json > preflight.json
```

The command exits with status `1` when any required file or directory is missing. It includes Python, installed package versions, visible GPU indicators, and Git details when they are available. Passing `--include-extra-corruptions` additionally requires the four extra corruption files/directories; `--severity` selects an ImageNet-C severity from 1 through 5.

For CIFAR-C and DomainNet scientific runs, use deep validation:

```shell
PYTHONPATH=src python -m runtime.preflight \
  --data-root ~/data \
  --dataset CIFAR10C \
  --dataset CIFAR100C \
  --dataset DomainNet \
  --deep \
  --json > preflight-deep.json
```

Deep CIFAR-C validation memory-maps every required array, checks the exact
50,000-sample shapes and `uint8` image dtype, validates integer label bounds,
requires identical label blocks across all five severities, and reads the first
and last samples. Deep DomainNet validation requires the
exact six wrapper environments with an identical 345-class taxonomy, rejects
symlinked traversed paths, includes dot-prefixed entries to match ImageFolder,
counts every image deterministically, requires every
class to be nonempty, and verifies and decodes a sample from each environment
with Pillow. Semantic failures are reported under each result's `deep.errors`
and make the CLI exit with status `1`; `missing` remains the existence-only
result. Other supported datasets currently report deep validation as
`not_supported` while retaining their normal layout checks.

`runtime.experiment_matrix --execute` automatically requests this deep mode
for every selected research dataset before it launches the first model
process. Planning JSON remains data-independent and does not open dataset
contents.

Matrix planning may retain the default `--device auto` for portable inspection,
but scientific execution and resume require an explicit `--device cpu`,
`--device mps`, or `--device cuda`. This prevents the same planned identity
from resolving to different hardware backends on different hosts. The runtime
rejects `auto` before dataset preflight or model launch.

Expected layouts are rooted below `--data-root`:

```text
corruption/CIFAR-10-C/{labels.npy,<corruption>.npy}
corruption/CIFAR-100-C/{labels.npy,<corruption>.npy}
corruption/ImageNet-C/{classnames.txt,<corruption>/<severity>/<class>/}
domainbed/domain_net/{clipart,infograph,painting,quickdraw,real,sketch}/<class>/
```

## Open-set CIFAR-100-C gradient-memory evidence

The active open-world thesis path uses the fixed
`open-set-cifar100-split-v1` partition (80 known classes and 20 held-out
classes). Enable it only with `CIFAR100C` mixed streams; `--ood_ratio` is
selected deterministically per source domain and the final stream fingerprint
binds the split, requested ratio, realized counts, domain order, and sample
identities. The evaluator retains ID/OOD labels only in evidence records; the
ordinary methods receive images only.

Start with a paired, cost-limited CPU or CUDA smoke. Run `NoAdapt` first, then
give its resulting `trace.jsonl` to the adapted method; both commands must
produce the same stream fingerprint.

```shell
PYTHONPATH=src python src/main.py \
  --dataset CIFAR100C --open_set --known_class_split open-set-cifar100-split-v1 \
  --ood_ratio 0.30 --tta_mode mixed --stream_mode block --stream_block_size 64 \
  --tta_algo NoAdapt --model clip_vitbase16 --seed 0 --device cuda \
  --data_root "$RAMEN_DATA_ROOT" --artifact-provenance fast \
  --max_eval_samples 200 --evidence_dir "$RAMEN_EVIDENCE_DIR" \
  --run_id open-c100-noadapt-block-s0

PYTHONPATH=src python src/main.py \
  --dataset CIFAR100C --open_set --known_class_split open-set-cifar100-split-v1 \
  --ood_ratio 0.30 --tta_mode mixed --stream_mode block --stream_block_size 64 \
  --tta_algo OracleIDGradientRamen --model clip_vitbase16 --seed 0 --device cuda \
  --data_root "$RAMEN_DATA_ROOT" --artifact-provenance fast \
  --max_eval_samples 200 --evidence_dir "$RAMEN_EVIDENCE_DIR" \
  --reference_trace "$RAMEN_EVIDENCE_DIR/open-c100-noadapt-block-s0/trace.jsonl" \
  --run_id open-c100-oracle-id-block-s0
```

Open-set traces add the all-or-none evaluator fields `original_label`,
`known_label_or_minus_one`, `is_ood`, split version, ratio, and a
pre-adaptation energy score (`-logsumexp`). Oracle gradient diagnostics add
retrieved OOD fractions plus all-vs-ID direction cosine/sign disagreement.
They are diagnostic upper bounds, not deployable methods. A local CPU/MPS
check is recorded in `plans/20260824-latent-ramen-evidence/reports/`; CUDA
effectiveness evidence still requires a Linux NVIDIA runner and the verified
datasets.

ConsensusRamen traces add a separate all-or-none method-only group: coordinate
agreement mean/p10/p50, retained-coordinate rate, active class count, and
`consensus_applied`. Summary agreement and mask-rate aggregates include only
rows where the hard mask actually ran; empty or below-minimum-class ordinary
Ramen fallbacks are counted separately.

`ConsensusRamenSoft` is a separately selectable, preregistered v1 ablation
identity. It selects `cfg/CIFAR100C/ConsensusRamenSoft.yaml` and reuses the
same method implementation with `consensus_mode: soft_weight`; it is not part
of the locked seven-method v0 canonical matrix.

`ConsensusRamenNoSelf` is likewise a separately selectable causal-history
ablation. It selects `cfg/CIFAR100C/ConsensusRamenNoSelf.yaml`, which retrieves
only supports from previous forwards before admitting the current batch. It is
also outside the locked v0 matrix.

## Canonical open-set Consensus matrix

The current thesis matrix is intentionally separate from the legacy latent
matrix. On a Linux NVIDIA host with verified official CIFAR-100-C, plan its
252 paired runs (or add `--execute --resume` to run/continue them) with:

```shell
PYTHONPATH=src python -m runtime.experiment_matrix \
  --open-set-consensus --device cuda --artifact-provenance fast \
  --data-root "$RAMEN_DATA_ROOT" --evidence-dir "$RAMEN_EVIDENCE_DIR"
```

It fixes NoAdapt, Ramen, EntropyGatedLatentRamen (the preserved negative
ablation), OracleDropOODRamen, OracleIDGradientRamen, ConsensusRamen, and
OracleConsensusRamen
across OOD ratios 0/0.1/0.3/0.5, `iid_mixed`/`block`/`recurring`, and seeds
0/1/2. The planner schedules each NoAdapt trace before its exact
same-cell adapted controls. Each ratio selects exactly 400 source examples per
corruption domain (6,000 before any optional smoke prefix), so ratio results
change OOD prevalence rather than adaptation exposure, stream length, or cache
opportunity. This fixed source budget is bound into the run ID, manifest, and
stream fingerprint. After completion, produce the descriptive report
without invoking the legacy latent gate:

```shell
PYTHONPATH=src python -m evaluation.open_set_consensus_analysis \
  --evidence-dir "$RAMEN_EVIDENCE_DIR" --data-root "$RAMEN_DATA_ROOT" \
  --artifact-provenance fast
```

### Canonical open-set oracle/Consensus matrix

The runnable canonical plan is intentionally separate from the legacy
LatentRamen matrix and its router gate. It plans 252 verified,
fixed-source-exposure CUDA runs: `NoAdapt`, `Ramen`, `EntropyGatedLatentRamen`,
`OracleDropOODRamen`, `OracleIDGradientRamen`, `ConsensusRamen`, and `OracleConsensusRamen` across OOD ratios `0/0.1/0.3/0.5`,
`iid_mixed/block/recurring`, and seeds `0/1/2`. Each adapted run references the
ratio-bound `NoAdapt` trace from its own cell, and every run ID binds the OOD
ratio. Only the three explicitly named Oracle methods may receive evaluator OOD
context; ConsensusRamen remains method-only. OracleConsensusRamen is an
evaluator-only consensus upper bound and intentionally has no all-vs-ID
direction diagnostic, because it filters OOD cache admission before aggregate
directions are formed.

```shell
PYTHONPATH=src python - <<'PY'
import os
from src.runtime.experiment_matrix import build_command, build_open_set_evidence_matrix

runs = build_open_set_evidence_matrix(
    evidence_dir=os.path.join(os.environ["RAMEN_EVIDENCE_ROOT"], "open-set-cifar100c-canonical"),
    data_root=os.environ["RAMEN_DATA_ROOT"],
)
for run in runs:
    print(" ".join(build_command(run)))
PY
```

Run commands in their emitted order (or pass the planned runs to
`execute_matrix`); do not independently launch adapted methods before their
paired baseline is complete. Analyze only with the separate descriptive
contract, never `evaluation.experiment_analysis`:

```shell
PYTHONPATH=src python -m evaluation.open_set_consensus_analysis \
  --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-cifar100c-canonical" \
  --data-root "$RAMEN_DATA_ROOT" --artifact-provenance fast
```

The analyzer validates complete method coverage, exact paired stream
fingerprints, OOD-context confinement, ID/open-set metrics, oracle direction
diagnostics, and Consensus diagnostics. Its `canonical_cuda_expected` label
distinguishes complete full-stream CUDA evidence from an explicitly
`noncanonical_pilot`; it deliberately emits no Consensus certification or
legacy latent-router go/no-go verdict.

## Canonical DomainNet open-set secondary matrix

The required natural-domain secondary benchmark is planned separately from
CIFAR-100-C. It is intentionally **planner-only**: it produces no DomainNet
measurements and does not imply that a local DomainNet tree, its 345-class
taxonomy, or CUDA are available. On a Linux NVIDIA host after verified
six-environment DomainNet acquisition, emit the fixed 252-run plan with:

```shell
PYTHONPATH=src python -m runtime.open_set_domainnet_matrix \
  --device cuda --artifact-provenance fast \
  --data-root "$RAMEN_DATA_ROOT" \
  --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-domainnet-canonical"
```

It locks the `open-set-domainnet-name-rank-v1` 276-known/69-unknown recipe,
all seven Phase-E baselines, OOD ratios `0/0.1/0.3/0.5`,
`iid_mixed/block/recurring`, and seeds `0/1/2`. Each adapted run references
the exact same-cell NoAdapt trace. At runtime the normal open-set stream
builder materializes and fingerprints the actual class taxonomy plus split;
therefore a changed DomainNet directory cannot masquerade as the same
benchmark.

The secondary benchmark fixes **690 source examples per environment** (4,140
before stream scheduling). Unlike CIFAR-100-C's 400 corruption examples, this
is a semantic coverage decision: at 10% OOD it reserves exactly 69 OOD
examples, one for each held-out DomainNet class. It remains divisible by every
preregistered OOD-ratio denominator. Do not replace this with a cost-limited
prefix or a non-CUDA pilot inside the canonical evidence directory.

The preflight also supports `ImageNetC`, `PACS`, `VLCS`, `TerraIncognita`, and `OfficeHome`; use `--all-datasets` to check every supported layout.
