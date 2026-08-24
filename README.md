# Ramen

## Environment

Create the pinned Python 3.11 / PyTorch 2.4.1 CPU/MPS environment:

```shell
conda env create -f environment.yml
conda activate nb-ramen
```

On a Linux CUDA 12.1 runner, use `environment-cuda.yml` and activate
`nb-ramen-cuda` instead.

OpenAI CLIP is pinned to an exact official repository commit. See
[`docs/research/experiment-runtime.md`](docs/research/experiment-runtime.md)
for the expected dataset layouts and preflight commands.

## Run experiments

Validate the data before launching a run:

```shell
PYTHONPATH=src python -m runtime.preflight \
  --data-root ~/data \
  --dataset CIFAR10C \
  --dataset CIFAR100C \
  --dataset ImageNetC5K \
  --dataset DomainNet
```

```shell
bash shell/run_ramen.sh
```

The mixed-domain entrypoint also supports deterministic non-stationary streams:

```shell
python src/main.py \
  --dataset DomainNet \
  --data_root ~/data \
  --model clip_vitbase32 \
  --tta_algo Ramen \
  --tta_mode mixed \
  --stream_mode recurring \
  --stream_block_size 64 \
  --seed 0 \
  --device cuda
```

Available stream modes are `iid_mixed`, `block`, `gradual`, `recurring`,
`imbalanced`, `novel_domain`, `class_domain_correlated`, and `bursty`. Each run
writes a manifest, exact stream order, per-sample JSONL trace, and summary under
`evidence/<run-id>/`. Reusing a run ID fails instead of mixing evidence.

## Reference GPU memory

The original reported GPU memory usage of Ramen for each setting:

- ViT-B/32 on CIFAR10C: 11,564 MiB,
- ViT-B/16 on CIFAR100C: 14,526 MiB,
- ViT-L/14 on ImageNetC5K: 30,588 MiB,
- ViT-B/32 on DomainNet: 11,568 MiB.
