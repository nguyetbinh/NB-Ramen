# Luna/CUDA handoff revalidation — 2026-08-25

## Decision

There is no accessible CUDA execution context from this workspace at the time
of this check. The current process is Darwin arm64 (macOS 15.7.7); its pinned
`nb-ramen` PyTorch is `2.4.1`, `torch.version.cuda == None`,
`torch.cuda.is_available() == False`, and zero CUDA devices are visible.
`nvidia-smi` is absent. MPS exists but is not canonical Linux/NVIDIA CUDA
evidence. No live Luna worker is attached to this task tree, so this report
does not claim a remote Luna GPU was contacted, provisioned, or ran a job.

The tree `/Users/admin/data/corruption/CIFAR-100-C` is deliberately the
320-sample pilot. Deep preflight rejected it: labels and all 15 arrays have a
leading dimension of 320 rather than 50,000. It must never feed the commands
below. The historical statement in `cuda-domainnet-execution-strategy.md` that
an official archive was locally present is stale for this host: the runtime
audit found a 12,516,864-byte archive that does not match the pinned official
MD5 or size.

## What was executable here

Planning remains healthy without opening data. In the pinned local
environment, this command emitted 28 runs for one block/seed-0 smoke (four
ratios times seven methods), including `OracleConsensusRamen` and `src-400`:

```shell
PYTHONPATH=src python -m runtime.experiment_matrix \
  --open-set-consensus --device cuda --stream block --seed 0 \
  --max-eval-samples 200 --artifact-provenance fast \
  --data-root /Users/admin/data \
  --evidence-dir /tmp/nb-ramen-open-set-planning-check
```

The complete matrix is 252 runs. Do not use `--open-set-consensus --execute`
as a cheap smoke: selecting one stream and seed still executes every ratio and
method. Use the paired direct-CLI smoke below first.

## Validated Linux/NVIDIA handoff

Copy or archive the current **dirty** workspace, not only commit
`b967cc4288aae71be5447676b774b30973bbeacc`, before starting. Then on a fresh
Linux host with an NVIDIA driver compatible with CUDA 12.1:

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
assert torch.cuda.is_available(), 'CUDA is not available to the runner'
assert torch.cuda.device_count() >= 1, 'no CUDA devices visible'
print({'torch': torch.__version__, 'cuda_build': torch.version.cuda,
       'device': torch.cuda.get_device_name(0)})
PY
git rev-parse HEAD
git status --short
```

Acquire the immutable archive into a new staging directory. `test ! -e`
prevents silently reusing the known noncanonical pilot or a partial download:

```shell
mkdir -p "$RAMEN_DATA_ROOT/downloads" "$RAMEN_DATA_ROOT/corruption"
test ! -e "$RAMEN_DATA_ROOT/downloads/CIFAR-100-C.tar"
curl --fail --location --retry 3 --retry-all-errors \
  --output "$RAMEN_DATA_ROOT/downloads/CIFAR-100-C.tar" \
  'https://zenodo.org/api/records/3555552/files/CIFAR-100-C.tar/content'
PYTHONPATH=src python - <<'PY'
import json, os
from runtime.artifact_provenance import verify_official_cifar100c_archive
print(json.dumps(verify_official_cifar100c_archive(
    os.path.join(os.environ['RAMEN_DATA_ROOT'], 'downloads', 'CIFAR-100-C.tar')
), sort_keys=True))
PY
tar -tf "$RAMEN_DATA_ROOT/downloads/CIFAR-100-C.tar" | head -20
test ! -e "$RAMEN_DATA_ROOT/corruption/CIFAR-100-C"
tar -xf "$RAMEN_DATA_ROOT/downloads/CIFAR-100-C.tar" -C "$RAMEN_DATA_ROOT/corruption"
test -d "$RAMEN_DATA_ROOT/corruption/CIFAR-100-C"
```

Generate the inventory once, then perform exact provenance and deep semantic
checks. The JSON is the repository-pinned Zenodo record, not mirror metadata:

```shell
PYTHONPATH=src python -m runtime.artifact_provenance generate cifar100c \
  "$RAMEN_DATA_ROOT/corruption/CIFAR-100-C" \
  --acquisition-json '{"publisher":"Zenodo","record_id":"3555552","doi":"10.5281/zenodo.3555552","url":"https://zenodo.org/api/records/3555552/files/CIFAR-100-C.tar/content","algorithm":"md5","expected_checksum":"11f0ed0f1191edbf9fa23466ae6021d3","actual_checksum":"11f0ed0f1191edbf9fa23466ae6021d3","size_bytes":2918473216}'
PYTHONPATH=src python -m runtime.artifact_provenance verify cifar100c \
  "$RAMEN_DATA_ROOT/corruption/CIFAR-100-C" --exact
mkdir -p "$RAMEN_EVIDENCE_ROOT/runtime"
PYTHONPATH=src python -m runtime.preflight \
  --data-root "$RAMEN_DATA_ROOT" --dataset CIFAR100C --deep --json \
  > "$RAMEN_EVIDENCE_ROOT/runtime/cifar100c-preflight-deep.json"
```

## Small paired CUDA smoke

This is one deterministic OOD-0.30 block cell with 200 evaluated samples and
the canonical 400 source examples per domain. It runs NoAdapt before
method-only ConsensusRamen; the latter never receives evaluator OOD labels.

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

Retain each run's `manifest.json`, `stream.json`, `trace.jsonl`, and
`summary.json`; verify equal stream fingerprints before interpreting results.
If VRAM fails, record its peak and reduce only the smoke prefix, never the
fixed 400-source canonical protocol without declaring a separate pilot.
