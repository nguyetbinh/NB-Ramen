# CUDA/open-set runtime audit — 2026-08-25

## Result

Canonical CUDA execution is unavailable on this host. The pinned
`nb-ramen` environment is Apple arm64 with PyTorch 2.4.1, `torch.version.cuda
== None`, `torch.cuda.is_available() == False`, and MPS available; `nvidia-smi`
is not installed.

The only local CIFAR-100-C tree is the deliberately noncanonical 320-sample
pilot made by `scripts/build-cifar100c-pilot.py`. Deep preflight rejects it:
official arrays require 50,000 samples. The local `CIFAR-100-C.tar` is
12,516,864 bytes, truncated, and has MD5
`73d48d18c421a11e793fbfb23cdb2657`, not the required Zenodo record 3555552
value `11f0ed0f1191edbf9fa23466ae6021d3` (2,918,473,216 bytes). No canonical
provenance sidecar exists.

## Verified planner behavior

Planning with the pinned environment succeeds for the fixed CUDA matrix and
emits paired NoAdapt-first commands; requesting a non-CUDA backend is rejected
by design. No CUDA process, canonical artifact, or effectiveness claim was
created.

```shell
PYTHONPATH=src python -m runtime.experiment_matrix \
  --open-set-consensus --device cuda --stream block --seed 0 \
  --max-eval-samples 200 --artifact-provenance fast \
  --data-root "$RAMEN_DATA_ROOT" --evidence-dir "$RAMEN_EVIDENCE_DIR"
```

## CUDA runner recipe

On Linux/NVIDIA with the complete official CIFAR-100-C archive and generated
sidecar, create `environment-cuda.yml`, verify `torch.cuda.is_available()`,
run deep preflight, then add `--execute` to the command above. Remove the
200-sample prefix for the fixed 400-source-example-per-domain canonical
protocol. This report establishes local unavailability and a reproducible
launch recipe only.
