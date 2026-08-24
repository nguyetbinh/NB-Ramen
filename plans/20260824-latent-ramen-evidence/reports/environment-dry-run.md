# Environment resolution evidence — 2026-08-24

Both reproducibility manifests were resolved with Conda 25.1.1 without installing packages.

## CPU/MPS manifest

Command:

```shell
conda env create --dry-run -f environment.yml --json
```

Result: exit code 0, environment name `nb-ramen`, 76 resolved dependencies, no solver error.

## Linux CUDA manifest

Command:

```shell
conda env create --dry-run --platform linux-64 -f environment-cuda.yml --json
```

Result: exit code 0, environment name `nb-ramen-cuda`, 146 resolved dependencies, no solver error.

The split was necessary because `pytorch-cuda=12.1` is unavailable on the local `osx-arm64` platform. The first combined manifest failed dry-run for that reason and was replaced rather than documented as runnable.
