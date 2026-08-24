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

The preflight also supports `ImageNetC`, `PACS`, `VLCS`, `TerraIncognita`, and `OfficeHome`; use `--all-datasets` to check every supported layout.
