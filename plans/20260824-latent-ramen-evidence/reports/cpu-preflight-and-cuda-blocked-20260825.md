# CPU preflight and CUDA availability evidence — 2026-08-25

## Result

- CPU/MPS runtime checks passed.
- Dependency test suite passed: **215/215**.
- CUDA execution is blocked on this host: it is an Apple M2 macOS host with no NVIDIA GPU, no CUDA runtime, and no `nvidia-smi` executable.
- Dataset-backed deep preflight is blocked by missing data under `/Users/admin/data`; no benchmark or effectiveness claim is made.
- The worktree was clean at the start of this run. By final verification, unrelated concurrent open-set work was present; it was preserved and not modified.

## Host and dependencies

Commands:

```shell
uname -a
sw_vers
sysctl -n hw.model hw.memsize
nvidia-smi
```

Observed:

```text
Darwin ... Darwin Kernel Version 24.6.0 ... arm64
ProductName: macOS
ProductVersion: 15.7.7
Mac14,2
17179869184
zsh: command not found: nvidia-smi
```

The pinned environment probe used `/Users/admin/miniconda3/envs/nb-ramen/bin/python` and reported Python 3.11.16, PyTorch 2.4.1, torchvision 0.19.1, NumPy 1.26.4, Pillow 10.4.0, PyYAML 6.0.2, tqdm 4.66.5, and CLIP import available. It reported `torch.version.cuda=None`, `torch.cuda.is_available()=False`, `torch.backends.mps.is_built()=True`, and `torch.backends.mps.is_available()=True`.

## Checks and statuses

```shell
/Users/admin/miniconda3/envs/nb-ramen/bin/python -m unittest discover -s tests -p 'test_*.py' -v
# exit 0; Ran 215 tests; OK

/Users/admin/miniconda3/envs/nb-ramen/bin/python -m compileall -q src tests
# exit 0

/Users/admin/miniconda3/envs/nb-ramen/bin/python src/main.py --help
# exit 0

PYTHONPATH=src /Users/admin/miniconda3/envs/nb-ramen/bin/python -m runtime.preflight \
  --data-root /Users/admin/data --dataset CIFAR100C --dataset DomainNet --deep --json
# exit 1; CIFAR-100-C and all six DomainNet environment directories are missing

/Users/admin/miniconda3/envs/nb-ramen/bin/python - <<'PY'
import torch
for device in ('cpu', 'mps'):
    if device == 'mps' and not torch.backends.mps.is_available():
        print(device, 'SKIPPED unavailable')
        continue
    a=torch.tensor([[1.,2.],[3.,4.]], device=device)
    b=torch.tensor([[5.,6.],[7.,8.]], device=device)
    c=a @ b
    if device == 'mps': torch.mps.synchronize()
    print(device, c.detach().cpu().tolist())
PY
# exit 0
# cpu [[19.0, 22.0], [43.0, 50.0]]
# mps  [[19.0, 22.0], [43.0, 50.0]]

git diff --check
# exit 0 for the worktree state observed during this run
```

The preflight JSON emitted missing paths including
`/Users/admin/data/corruption/CIFAR-100-C/labels.npy` and
`/Users/admin/data/domainbed/domain_net/{clipart,infograph,painting,quickdraw,real,sketch}`.

## CUDA/Luna blocked status and reproducible command

No GPU smoke was run because CUDA is unavailable locally. The exact Linux CUDA
runner command below is reproducible in a Luna task with an NVIDIA GPU and the
repository checked out at the same revision:

```shell
set -e
export RAMEN_REPO=/workspace/NB-Ramen
export RAMEN_DATA_ROOT=/datasets/nb-ramen
export RAMEN_EVIDENCE_DIR=/evidence/nb-ramen
cd "$RAMEN_REPO"
conda env create -f environment-cuda.yml
conda activate nb-ramen-cuda
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
PYTHONPATH=src python -m runtime.experiment_matrix \
  --dataset CIFAR100C --stream block --method NoAdapt --seed 0 \
  --device cuda --data-root "$RAMEN_DATA_ROOT" \
  --evidence-dir "$RAMEN_EVIDENCE_DIR" \
  --artifact-provenance fast --max-eval-samples 200 --execute
```

This is an execution recipe, not local GPU evidence. The repository’s existing
[CUDA execution strategy](cuda-domainnet-execution-strategy.md) records the
same host limitation and the full-grid contract. The existing
[local runtime smoke](local-runtime-smoke.md) records prior real CPU/MPS CLIP
and LatentRamen mechanics evidence; this report adds the fresh 2026-08-25
checks above.

## Final worktree observation

The final `git status --short` showed these pre-existing/concurrent changes,
which this run did not edit:

```text
 M src/streams/builders.py
?? cfg/research/open-set-cifar100-split-v1.json
?? src/datasets/open_set.py
?? src/evaluation/open_set_metrics.py
?? tests/test_open_set.py
?? tests/test_open_set_metrics.py
?? plans/20260824-latent-ramen-evidence/reports/cpu-preflight-and-cuda-blocked-20260825.md
```
