# CUDA CIFAR-100-C and DomainNet execution strategy

> Date: 2026-08-24  
> Scope: execution feasibility and a scheduler/cloud-neutral runbook for the
> current repository state. No CUDA benchmark evidence has been produced.

## Decision

The complete experiment matrix must run on a separate Linux NVIDIA host. The
local machine is an Apple M2 MacBook Air with 16 GB unified memory, no CUDA
device or `nvidia-smi`, and about 22 GiB free after retaining the official
CIFAR-100-C archive and extraction. It can plan commands and validate CPU/MPS
mechanics, including the completed official-wrapper smoke, but it cannot
produce CUDA latency, CUDA allocator peaks, benchmark-scale CIFAR-100-C
results, or DomainNet accuracy evidence. CIFAR-100-C is now present and
exactly inventoried locally; DomainNet remains absent.

A 24 GiB CUDA GPU is the recommended starting point for the current batch-100
matrix. A 16 GiB GPU is only a borderline floor for full-capacity DomainNet
memory methods and must not be assumed sufficient without a measured pilot.
The prior concern that structured retrieval itself requires 48--80 GiB was an
overestimate: the causal implementations query one item at a time.

Provision at least 100 GiB of persistent working storage; 150 GiB is preferable
when archives, extracted data, Conda packages, CLIP checkpoints, raw JSONL
traces, and a retry copy must coexist. Evidence size is deliberately not
predicted here: measure it with the timed pilot before sizing the complete
grid.

## Fixed experiment contract

The current default research grid is:

- datasets: `CIFAR100C`, `DomainNet`;
- streams: `iid_mixed`, `block`, `gradual`, `recurring`, `imbalanced`;
- seeds: `0`, `1`, `2`;
- methods: `NoAdapt`, `Tent`, `Ramen`, `CausalRamen`,
  `RandomMemoryRamen`, `SameClassRamen`, `GlobalNearestRamen`,
  `ContextOnlyRamen`, `OracleLatentRamen`, `LatentRamen`;
- device: `cuda`;
- batch size: 100, selected internally for both datasets;
- models: CLIP ViT-B/16 for CIFAR-100-C and ViT-B/32 for DomainNet.

This is 30 dataset/stream/seed cells, 10 ordered runs per cell, and 300 runs in
total. Planning checks verified that every cell contains exactly 10 runs, that
`NoAdapt` is first, and that all nine adapted runs point to that cell's exact
NoAdapt trace. Even if `--method NoAdapt` is omitted, the matrix inserts it
first; it is written explicitly below to make the intended grid auditable.

`--artifact-provenance` and the canonical absolute `--data-root` are part of
run identity. The analyzer must receive the same provenance mode, data root,
device, cost limit, evidence root, and configs used during execution.

## Dataset acquisition and storage

The required tree is:

```text
DATA_ROOT/
  corruption/CIFAR-100-C/
    labels.npy
    gaussian_noise.npy
    ... 14 other main-corruption arrays ...
  domainbed/domain_net/
    clipart/<class>/<image>
    infograph/<class>/<image>
    painting/<class>/<image>
    quickdraw/<class>/<image>
    real/<class>/<image>
    sketch/<class>/<image>
```

CIFAR-100-C is pinned to Zenodo record `3555552`, DOI
`10.5281/zenodo.3555552`, content URL
`https://zenodo.org/api/records/3555552/files/CIFAR-100-C.tar/content`, size
`2918473216` bytes, and MD5 `11f0ed0f1191edbf9fa23466ae6021d3`.
The official archive is about 2.72 GiB. The 15 arrays used by the default
benchmark contain 750,000 RGB images total and occupy about 2.15 GiB of raw
pixel payload; the official extraction also includes four extra corruptions,
so archive plus extraction needs roughly 5.5 GiB while staging.

DomainNet must contain exactly the six named environments, an identical
345-class taxonomy in every environment, and at least one readable image per
class. Public cleaned-distribution figures provide a planning proxy of about
17.2 GiB downloaded and 17.5 GiB materialized across the six domains. These
figures are capacity estimates, not repository trust anchors. DomainNet has no
publisher checksum pinned by this repository; its generated sidecar attests to
the locally inventoried bytes, not to publisher authenticity. Record source
URLs and archive checksums separately when acquiring it.

The local Mac's roughly 22 GiB remaining space is insufficient to stage both
DomainNet archives and extracted data alongside the retained CIFAR-100-C
archive/extraction, environments, models, and evidence.

## Verified GPU-memory arithmetic

Both configured CLIP backbones have feature dimension 512 and per-sample
normalization-gradient dimension 39,936. Cache tensors are fp16. Structured
items additionally retain fp32 entropy and reliability plus int64 recency and
item ID, giving:

```text
structured item bytes = (512 + 39,936) * 2 + 4 + 4 + 8 + 8 = 80,920
original Ramen item bytes = (512 + 39,936) * 2 + 2 + 2 = 80,900
```

| Dataset | Classes | Capacity/class | Top-k | Structured cache maximum | Original Ramen cache | Structured query gradient | Original Ramen query gradient |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-100-C | 100 | 750 | 5 | 5.652 GiB | 5.651 GiB | 0.037 GiB | 0.037 GiB |
| DomainNet | 345 | 300 | 10 | 7.800 GiB | 7.798 GiB | 0.257 GiB | 0.074 GiB |

The structured query is `[1, classes, topk, 39936]`, because
`LatentRamen`, `OracleLatentRamen`, and the support ablations slice each batch
into one causal item before calling `StructuredGradientMemory.query`.
Original Ramen queries the full batch but loops over classes, so its live query
is `[100, topk, 39936]`, not `[100, classes, topk, 39936]`.

These numbers cover retained support memory and the principal retrieval result,
not CLIP weights, autograd activations, transformed input batches, CUDA context,
temporary distance/aggregation tensors, or allocator reservation. Therefore:

- 16 GiB is a test-only lower bound for DomainNet memory methods;
- 24 GiB is the recommended initial runner;
- use emitted `peak_device_memory_bytes` from a full-capacity pilot to decide
  whether the complete grid needs a larger GPU;
- do not reduce batch size ad hoc: batch size is fixed by the matrix and a
  change would alter the experiment contract.

## Prepare a Linux CUDA runner

Use persistent paths and the exact repository snapshot that produced the plan.
Much of the current project work is uncommitted, so copying only the Git commit
is not sufficient until that work has been committed or archived explicitly.

```shell
export RAMEN_REPO=/workspace/NB-Ramen
export RAMEN_DATA_ROOT=/datasets/nb-ramen
export RAMEN_EVIDENCE_ROOT=/evidence/nb-ramen

cd "$RAMEN_REPO"
conda env create -f environment-cuda.yml
conda activate nb-ramen-cuda

nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
git rev-parse HEAD
git status --porcelain
```

The host must provide an NVIDIA driver compatible with the pinned CUDA 12.1
runtime. Environment creation requires the configured Conda channels and Git
access for the pinned OpenAI CLIP package unless the environment is built and
transported beforehand. Pre-stage the official `ViT-B-16.pt` and
`ViT-B-32.pt` checkpoints in the CLIP cache when compute nodes lack egress.

## Build and verify provenance sidecars

Generate each canonical sidecar once after acquisition and before making the
dataset tree read-only. Generation hashes every file. The CIFAR command must
carry the exact pinned acquisition record:

```shell
cd "$RAMEN_REPO"

PYTHONPATH=src python -m runtime.artifact_provenance generate cifar100c \
  "$RAMEN_DATA_ROOT/corruption/CIFAR-100-C" \
  --acquisition-json '{"publisher":"Zenodo","record_id":"3555552","doi":"10.5281/zenodo.3555552","url":"https://zenodo.org/api/records/3555552/files/CIFAR-100-C.tar/content","algorithm":"md5","expected_checksum":"11f0ed0f1191edbf9fa23466ae6021d3","actual_checksum":"11f0ed0f1191edbf9fa23466ae6021d3","size_bytes":2918473216}'

PYTHONPATH=src python -m runtime.artifact_provenance generate domainnet \
  "$RAMEN_DATA_ROOT/domainbed/domain_net"

PYTHONPATH=src python -m runtime.artifact_provenance verify cifar100c \
  "$RAMEN_DATA_ROOT/corruption/CIFAR-100-C" --exact

PYTHONPATH=src python -m runtime.artifact_provenance verify domainnet \
  "$RAMEN_DATA_ROOT/domainbed/domain_net" --exact
```

`fast` run provenance verifies the trusted CLIP digest, sidecar digest, exact
path/size inventory, and regular non-symlinked files without rereading every
dataset byte. `exact` additionally rehashes every dataset file before loading
and again after dataset construction for every run. Exact mode gives stronger
per-run content verification but can create substantial repeated DomainNet I/O.
Do not mix modes in one claimed grid: mode changes run IDs and must match the
analyzer.

## Deep preflight

Archive the standalone report before consuming GPU time:

```shell
mkdir -p "$RAMEN_EVIDENCE_ROOT/runtime"

PYTHONPATH=src python -m runtime.preflight \
  --data-root "$RAMEN_DATA_ROOT" \
  --dataset CIFAR100C \
  --dataset DomainNet \
  --deep \
  --json > "$RAMEN_EVIDENCE_ROOT/runtime/preflight-deep.json"
```

Deep CIFAR validation memory-maps every required array, checks exact
`(50000,32,32,3)` uint8 shapes, label bounds and severity consistency, and
reads both ends of each array. Deep DomainNet validation rejects traversed
symlinks, requires the exact environment and 345-class taxonomy, counts all
images, requires nonempty classes, and decodes a sample from every environment.
Matrix `--execute` repeats deep preflight automatically and launches no model
if it fails.

## Staged execution

### 1. Cost-limited smoke

Start with 200 deterministic samples, both datasets, two informative streams,
and the principal baseline/structured methods. `NoAdapt` is paired first in
each cell. Use `fast` after the one-time exact sidecar verification above:

```shell
PYTHONPATH=src python -m runtime.experiment_matrix \
  --dataset CIFAR100C --dataset DomainNet \
  --stream iid_mixed --stream block \
  --method NoAdapt --method Tent --method Ramen \
  --method OracleLatentRamen --method LatentRamen \
  --seed 0 --device cuda \
  --data-root "$RAMEN_DATA_ROOT" \
  --evidence-dir "$RAMEN_EVIDENCE_ROOT/smoke-fast-n200" \
  --max-eval-samples 200 \
  --artifact-provenance fast \
  --execute
```

Inspect every `summary.json`, job log, and CUDA peak. Then verify the smoke is
strictly resumable by rerunning the identical command with `--resume`.

### 2. Timed and storage-measured pilot

Run one complete DomainNet `iid_mixed`, seed-0 cell containing NoAdapt, Ramen,
and LatentRamen. A full cell, rather than a short prefix, exercises cache
growth and representative stream length. Record the scheduler's elapsed time,
GPU utilization, peak VRAM, CPU/RAM, I/O, and resulting evidence bytes. Do not
extrapolate until this succeeds:

```shell
/usr/bin/time -v env PYTHONPATH=src python -m runtime.experiment_matrix \
  --dataset DomainNet \
  --stream iid_mixed \
  --method NoAdapt --method Ramen --method LatentRamen \
  --seed 0 --device cuda \
  --data-root "$RAMEN_DATA_ROOT" \
  --evidence-dir "$RAMEN_EVIDENCE_ROOT/pilot-fast" \
  --artifact-provenance fast \
  --execute

du -sh "$RAMEN_EVIDENCE_ROOT/pilot-fast"
```

Use the observed slowest-method wall time and evidence size to request the full
job limits and storage. No duration is asserted in this report.

### 3. Full grid

The following single invocation is the canonical 300-run `fast` plan. Omit
`--execute` first and archive its planning JSON; then add `--execute` on the
runner:

```shell
PYTHONPATH=src python -m runtime.experiment_matrix \
  --dataset CIFAR100C --dataset DomainNet \
  --stream iid_mixed --stream block --stream gradual \
  --stream recurring --stream imbalanced \
  --method NoAdapt --method Tent --method Ramen \
  --method CausalRamen --method RandomMemoryRamen \
  --method SameClassRamen --method GlobalNearestRamen \
  --method ContextOnlyRamen --method OracleLatentRamen \
  --method LatentRamen \
  --seed 0 --seed 1 --seed 2 \
  --device cuda \
  --data-root "$RAMEN_DATA_ROOT" \
  --evidence-dir "$RAMEN_EVIDENCE_ROOT/full-fast" \
  --artifact-provenance fast
```

For an exact-provenance grid, use a distinct evidence root such as
`full-exact` and replace the final mode with `--artifact-provenance exact`.
Expect every exact-mode run to rehash its selected dataset twice; decide from
the pilot whether that repeated I/O is acceptable.

Scheduler arrays are safe only at dataset/stream/seed granularity: 30 tasks,
one whole cell per task. Each task must invoke the matrix with one dataset, one
stream, one seed, and all ten methods so the matrix runs NoAdapt first and then
validates it before every adapted method. Never array over methods and never
launch an adapted method independently from its paired baseline. Distinct
cells have distinct run IDs and may share the same persistent evidence root.

## Fail-closed resume and recovery

Resume is strict:

```shell
# Append --execute --resume to the exact original cell or grid invocation.
```

A run is skipped only after its manifest, stream, trace, summary, config hash,
artifact mode, data-root identity, metrics, and paired baseline provenance all
validate. A malformed, incomplete, stale, foreign, or tampered run stops the
matrix. There is no within-run checkpoint resume. Preserve an incomplete run
directory and its scheduler log for diagnosis, move it outside the evidence
root, and rerun the identical cell to create a new run directory. Do not edit
evidence files to make resume pass.

## Exact post-hoc analyzer command

The analyzer intentionally reconstructs and validates the four methods used by
the research gates: NoAdapt, Ramen, OracleLatentRamen, and LatentRamen. For the
full `fast` grid above, run:

```shell
PYTHONPATH=src python -m evaluation.experiment_analysis \
  --thresholds cfg/research/phase02-go-no-go.json \
  --evidence-dir "$RAMEN_EVIDENCE_ROOT/full-fast" \
  --config-dir "$RAMEN_REPO/cfg" \
  --dataset CIFAR100C --dataset DomainNet \
  --stream iid_mixed --stream block --stream gradual \
  --stream recurring --stream imbalanced \
  --seed 0 --seed 1 --seed 2 \
  --device cuda \
  --data-root "$RAMEN_DATA_ROOT" \
  --artifact-provenance fast \
  > "$RAMEN_EVIDENCE_ROOT/full-fast-analysis.json"
```

For an exact grid, both `--evidence-dir` and
`--artifact-provenance exact` must match that grid. Exit 0 means `go`; exit 1
means `no_go` or `insufficient_evidence`; invalid evidence produces canonical
JSON and exit 2.

## Artifact return checklist

Return or persist:

- complete run directories (`manifest.json`, `stream.json`, `trace.jsonl`,
  `summary.json`, and `results.csv`);
- the archived planning JSON and `preflight-deep.json`;
- provenance sidecars and acquisition/checksum records;
- Conda manifest/export, Git commit and dirty-state archive, CLIP checkpoint
  digests, `nvidia-smi`, and scheduler logs;
- the analyzer JSON and the exact command used to create it.

Copy evidence only after jobs have stopped writing it. Keep the original
runner copy until the returned archive has been checksummed and verified.

## Remaining feasibility risks

1. Structured methods call a Python-level causal query per sample, and the
   class-balanced query examines all 345 DomainNet classes. Runtime must be
   measured; no defensible duration exists before the pilot.
2. The cache bounds are large enough that a short smoke cannot establish the
   final VRAM peak. The full DomainNet pilot is the sizing authority.
3. Matrix resume is run-granular, not batch-granular. A late interruption
   restarts that run from sample zero.
4. `fast` provenance detects inventory or size changes but does not rehash
   every dataset byte per run. `exact` does, at substantial repeated I/O cost.
5. DomainNet provenance is an unsigned canonical local inventory, not
   publisher-signed acquisition evidence.
6. The current working tree contains uncommitted runtime/configuration work.
   A CUDA runner must receive the exact tree, not merely the current commit.
