# Repository Guidelines

## Research Purpose

NB-Ramen studies CLIP test-time adaptation, open-world gradient memory, and query-conditioned support selection. Progress means reducing scientific uncertainty with trustworthy evidence. Oracle headroom demonstrates an opportunity; it does not establish a deployable label-free selector.

## Research Workflow and State

Use `.agents/skills/research-orchestrator/SKILL.md` for literature mapping, hypotheses, diagnostics, case decisions, and paper evidence. Reconstruct the current stage from protocols, result reports, and available raw artifacts; roadmap status can be stale.

## Project Organization

`src/methods/` contains adaptation algorithms; `src/models/`, `src/datasets/`, `src/streams/`, `src/memory/`, `src/runtime/`, and `src/evaluation/` separate supporting responsibilities. `src/main.py` runs experiments. Configurations and locked splits live in `cfg/`; research protocols in `docs/research/`; plans and reports in `plans/`. Tests are in `tests/` and `notebooks/kaggle/`. Large artifacts belong in Git-ignored `evidence/` and `log/`.

## Evidence Integrity

Keep true labels and OOD metadata outside label-free selection. Preserve baseline parity, cache history, reset semantics, unique run IDs, and source/config/stream provenance. Never alter frozen gates after inspecting outcomes or erase negative runs.

Distinguish planned, failed, completed, and validated runs. Recompute metrics from raw records; disclose missing artifacts and offline-audit limits. Separate CE gains, accuracy gains, ranking correlation, and causal claims. Report sample overlap, uncertainty, costs, and generalization limits.

## Development and Validation

On Linux/NVIDIA, use `uv sync --locked` and `source .venv/bin/activate` for local Python 3.11 / PyTorch 2.6.0 / CUDA 12.4 development. Use `--offline` when dependencies are cached. Historical PyTorch 2.4.1 research environments remain in `environment.yml` and `environment-cuda.yml`; record runtime differences before collecting evidence. Consult `docs/research/experiment-runtime.md` before campaigns. From the repository root:

```sh
PYTHONPATH=src python -m runtime.preflight --data-root ~/data --dataset CIFAR100C
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Use four-space indentation, snake_case helpers, and PascalCase algorithm modules. Follow surrounding style; no formatter is configured. Add meaningful `unittest` regressions for changed behavior; CPU tests do not establish CUDA results.

## Changes and Handoffs

Use `feat:`, `fix:`, or `docs:` commits with optional scopes. Describe the question or defect, changes, validation, and remaining uncertainty. Link protocols and evidence. End research cycles with a justified next decision and its success/stop signals.
