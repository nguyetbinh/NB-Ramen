# Runtime probe — 2026-08-24

## Local host

- macOS on Apple M2 with 16 GB unified memory.
- Default Python: 3.13.0.
- The original default Python had no project runtime dependencies. A dedicated `nb-ramen` Conda environment is now installed and verified; see [local runtime smoke](local-runtime-smoke.md).
- No usable CUDA runtime or NVIDIA GPU is present.
- `src/main.py` now has explicit `auto`, `cpu`, `mps`, and `cuda` selection. An unavailable explicitly requested CUDA backend is an error rather than a silent fallback.
- No external dataset root was found under the common local paths checked.

## Consequence

Unit tests, static validation, and CPU/MPS model smoke tests can run locally. Accuracy evidence still requires the external datasets; CUDA peak-memory and latency evidence require a separate CUDA runner.

## Luna

A fresh Codex Luna task was launched to probe the project runtime. The local shell has no standalone `luna` executable; in this workflow, Luna is the execution context/model rather than a repository CLI.
