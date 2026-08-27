# Terra Luna runtime audit — 2026-08-25

## Decision

There is **no Luna CLI, job-submission endpoint, scheduler configuration, or
attached Luna worker available from this workspace**.  `command -v luna` is
empty and repository search finds only narrative handoffs.  In this project,
as [the earlier runtime probe](runtime-probe.md) states, “Luna” denotes a
Codex execution context/model, not a repository command.  Therefore this
audit cannot submit, query, or reserve a Luna CPU/GPU job.  The correct
handoff is a self-contained Linux/NVIDIA runner specification, to be launched
through whatever Luna interface is supplied to the receiving task.

No expensive job was launched for this audit.

## Local runtime facts (rechecked)

- Host: Darwin arm64, macOS 15.7.7.
- Pinned local environment: Python 3.11.16 and PyTorch 2.4.1.
- `torch.version.cuda` is `None`; CUDA availability is `False`; visible CUDA
  devices: `0`; `nvidia-smi` is not installed.
- MPS is built and available.  It is valid for local mechanics/pilot work, not
  Linux/NVIDIA CUDA evidence.
- `environment-cuda.yml` is the GPU environment contract: Python 3.11,
  PyTorch 2.4.1, torchvision 0.19.1, `pytorch-cuda=12.1`, and CLIP pinned to
  `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.

## How to use a Luna worker

### CPU job (local or CPU-capable Luna worker)

Use only for dependency/data checks and cost-limited mechanics.  It must be
explicitly labeled non-CUDA evidence.  This preflight is dependency-light and
does not load model weights; add `--deep` only after a complete dataset is
available.

```shell
conda env create -f environment.yml
conda activate nb-ramen
PYTHONPATH=src python -m runtime.preflight \
  --data-root "$RAMEN_DATA_ROOT" --dataset CIFAR100C --json \
  > "$RAMEN_EVIDENCE_ROOT/runtime/cifar100c-preflight.json"
```

For a paired CPU mechanics smoke, use the direct CLI contract below with
`--device cpu`, a distinct evidence root, `--max_eval_samples 200`, and the
same baseline-first ordering.  Do not present its latency, allocator values,
or effectiveness as CUDA results.

### CUDA job specification

Request a Linux x86_64 worker with an NVIDIA driver compatible with CUDA 12.1,
one GPU with **at least 24 GiB VRAM** (16 GiB is only a borderline pilot floor
for DomainNet), and 100 GiB persistent storage minimum (150 GiB preferred).
Give it persistent dataset/evidence paths, network or pre-staged Conda/CLIP
dependencies, and the exact workspace snapshot.  The current commit observed
during this audit was `cd97063ab37d4fd29d9f5f22601ad471d19eee8c`; archive the
tree rather than relying only on a commit whenever it contains local changes.

Run this bootstrap before allocating experimental compute:

```shell
export RAMEN_REPO=/workspace/NB-Ramen
export RAMEN_DATA_ROOT=/datasets/nb-ramen
export RAMEN_EVIDENCE_ROOT=/evidence/nb-ramen
cd "$RAMEN_REPO"
conda env create -f environment-cuda.yml
conda activate nb-ramen-cuda
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is not available to this worker'
assert torch.cuda.device_count() >= 1, 'no CUDA device is visible'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

Before a run, acquire the official CIFAR-100-C data (Zenodo record 3555552),
generate its provenance sidecar, verify it with `--exact`, then run deep
preflight.  The full commands are retained in
[the Luna handoff](luna-cuda-handoff-revalidation-20260825.md) and
[the CUDA strategy](cuda-domainnet-execution-strategy.md).  Do not use a
noncanonical tree for this stage.

The smallest paired CUDA compute job is intentionally direct rather than
`experiment_matrix --open-set-consensus --execute`: restricting that matrix to
one seed/stream still expands to 28 runs.  Run the baseline first:

```shell
export RAMEN_SMOKE="$RAMEN_EVIDENCE_ROOT/open-set-cuda-smoke-block-s0"
PYTHONPATH=src python src/main.py \
  --dataset CIFAR100C --model clip_vitbase16 --tta_algo NoAdapt \
  --tta_mode mixed --batch_size 100 --seed 0 --stream_seed 0 \
  --stream_mode block --device cuda --open_set \
  --known_class_split open-set-cifar100-split-v1 --ood_ratio 0.3 \
  --open-set-per-domain-source-budget 400 --data_root "$RAMEN_DATA_ROOT" \
  --config "$RAMEN_REPO/cfg" --artifact-provenance fast --max_eval_samples 200 \
  --evidence_dir "$RAMEN_SMOKE" --run_id cuda-smoke-noadapt-block-s0
```

Then run ConsensusRamen only after the first command has completed and its
`trace.jsonl` exists:

```shell
PYTHONPATH=src python src/main.py \
  --dataset CIFAR100C --model clip_vitbase16 --tta_algo ConsensusRamen \
  --tta_mode mixed --batch_size 100 --seed 0 --stream_seed 0 \
  --stream_mode block --device cuda --open_set \
  --known_class_split open-set-cifar100-split-v1 --ood_ratio 0.3 \
  --open-set-per-domain-source-budget 400 --data_root "$RAMEN_DATA_ROOT" \
  --config "$RAMEN_REPO/cfg" --artifact-provenance fast --max_eval_samples 200 \
  --evidence_dir "$RAMEN_SMOKE" \
  --reference_trace "$RAMEN_SMOKE/cuda-smoke-noadapt-block-s0/trace.jsonl" \
  --run_id cuda-smoke-consensus-block-s0
```

Keep `manifest.json`, `stream.json`, `trace.jsonl`, `summary.json`, the deep
preflight JSON, provenance records, `nvidia-smi`, and scheduler log.  Confirm
equal stream fingerprints before interpretation.  Preserve the 400-source
budget; only reduce the evaluated prefix when declaring a noncanonical smoke.

After the smoke, the canonical open-set command is:

```shell
PYTHONPATH=src python -m runtime.experiment_matrix \
  --open-set-consensus --device cuda --artifact-provenance fast \
  --data-root "$RAMEN_DATA_ROOT" --evidence-dir "$RAMEN_EVIDENCE_ROOT/open-set-cifar100c-canonical" \
  --execute --resume
```

It schedules 252 ordered CUDA runs (four OOD ratios, three stream modes,
three seeds, seven methods), with the cell's NoAdapt reference before every
adapted method.  Do not array over methods or submit an adapted run alone.

## Existing artifact inventory

The checked-in/local `evidence/` tree contains 140 manifests for CIFAR100C:
121 MPS and 19 CPU, with **zero CUDA** manifests.  Of these, 135 have the
complete manifest/stream/trace/summary set.  Five manifests are incomplete:

- `official-cifar100c-cpu-smoke/official-c100-ramen-block-r50-s0-b16-n64`
- `official-cifar100c-mps-smoke/official-c100-ramen-block-r50-s0`
- `open-set-mps-pilot/pilot128-s0-oracle-drop`
- `open-set-mps-pilot/pilot128-s1-oracle-drop`
- `open-set-mps-pilot/pilot128-s2-oracle-drop`

These are local mechanics/pilot artifacts.  They neither provide a reusable
CUDA baseline trace nor satisfy the canonical CUDA contract.

## Blockers and handoff consistency

1. The current host has no NVIDIA CUDA runtime and no attached Luna worker.
2. `/Users/admin/data/corruption/CIFAR-100-C` is the 320-sample pilot, not the
   official 50,000-sample arrays.  `/Users/admin/data/CIFAR-100-C.tar` is
   12,517,376 bytes, not the required 2,918,473,216-byte archive; DomainNet is
   absent.  Neither can support canonical CUDA execution.
3. `cuda-domainnet-execution-strategy.md` says CIFAR-100-C was “present and
   exactly inventoried locally.”  The newer CUDA audit and Luna revalidation
   explicitly supersede that stale statement for this host; they identify the
   local tree/archive as noncanonical.
4. There is no scheduler-neutral Luna command to validate here.  The receiving
   Luna job must provide the Linux/NVIDIA resource selection and execute the
   repository commands above.

Status: DONE_WITH_CONCERNS

Summary: Luna is not locally callable; the project supplies a Linux/NVIDIA
handoff rather than a Luna CLI. Local evidence is CPU/MPS-only, and the report
records safe CPU/CUDA job contracts plus the data and worker blockers.

Concerns/Blockers: A provisioned Luna Linux/NVIDIA worker, complete official
CIFAR-100-C acquisition/provenance, and (for the secondary benchmark)
DomainNet acquisition are required before any canonical CUDA claim.
