"""Plan the preregistered, separately reported ConsensusRamen ablations.

This planner intentionally does not expand ``experiment_matrix``'s locked
seven-method primary matrix.  It selects one caller-declared held-out open-set
cell and pairs every adaptation method with the exact same NoAdapt trace.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

try:  # Supports ``python -m runtime...`` and ``python -m src.runtime...``.
    from .experiment_matrix import (
        MISSING_CONFIG_HASH,
        OPEN_SET_DATASET,
        OPEN_SET_OOD_RATIOS,
        OPEN_SET_PER_DOMAIN_SOURCE_BUDGET,
        OPEN_SET_SPLIT,
        OPEN_SET_STREAMS,
        REPOSITORY_ROOT,
        SUPPORTED_DEVICES,
        ExperimentRun,
        build_command,
        build_experiment_matrix,
        make_run_id,
    )
except ImportError:  # pragma: no cover - direct-file execution only
    from experiment_matrix import (
        MISSING_CONFIG_HASH,
        OPEN_SET_DATASET,
        OPEN_SET_OOD_RATIOS,
        OPEN_SET_PER_DOMAIN_SOURCE_BUDGET,
        OPEN_SET_SPLIT,
        OPEN_SET_STREAMS,
        REPOSITORY_ROOT,
        SUPPORTED_DEVICES,
        ExperimentRun,
        build_command,
        build_experiment_matrix,
        make_run_id,
    )


# Ramen is retained as §26(A)'s ordinary aggregation control.  The six named
# Consensus identities cover the locked v0, soft v1, causal no-self, and the
# preregistered tau/C_min sensitivity checks.  These are method *identities*,
# not permission to retune their configuration on the selected held-out cell.
CONSENSUS_ABLATION_METHODS = (
    "Ramen",
    "ConsensusRamen",
    "ConsensusRamenSoft",
    "ConsensusRamenNoSelf",
    "ConsensusRamenTau060",
    "ConsensusRamenMin2",
    "ConsensusRamenMin4",
)
REQUIRED_CONSENSUS_ALIASES = frozenset(CONSENSUS_ABLATION_METHODS[2:])
PILOT_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def _validate_pilot_name(pilot_name: str | None) -> str:
    if not isinstance(pilot_name, str) or not pilot_name:
        raise ValueError("noncanonical pilot planning requires a non-empty pilot_name")
    if (pilot_name[0] == "-" or pilot_name[-1] == "-"
            or any(character not in PILOT_NAME_CHARS for character in pilot_name)):
        raise ValueError("pilot_name must use lowercase letters, digits, and internal hyphens only")
    return pilot_name


def _validate_inputs(
    *, stream: str, ood_ratio: float, seeds: tuple[int, ...], device: str,
    canonical: bool, pilot_name: str | None, per_domain_source_budget: int,
) -> None:
    if stream not in OPEN_SET_STREAMS:
        raise ValueError("held-out stream must be one of: " + ", ".join(OPEN_SET_STREAMS))
    if (not isinstance(ood_ratio, (int, float)) or isinstance(ood_ratio, bool)
            or not math.isfinite(ood_ratio) or ood_ratio not in OPEN_SET_OOD_RATIOS):
        raise ValueError("held-out OOD ratio must be one of 0, 0.1, 0.3, 0.5")
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("provide at least one integer seed")
    if device not in SUPPORTED_DEVICES:
        raise ValueError("device must be one of " + ", ".join(SUPPORTED_DEVICES))
    if canonical and device != "cuda":
        raise ValueError("canonical Consensus ablation execution requires device='cuda'")
    if not canonical:
        _validate_pilot_name(pilot_name)
    elif pilot_name is not None:
        raise ValueError("pilot_name is only valid when canonical=False")
    if (not isinstance(per_domain_source_budget, int)
            or isinstance(per_domain_source_budget, bool) or per_domain_source_budget <= 0):
        raise ValueError("per_domain_source_budget must be a positive integer")
    if per_domain_source_budget % Fraction(str(ood_ratio)).denominator:
        raise ValueError("per-domain source budget must support the selected OOD ratio")


def build_consensus_ablation_matrix(
    *,
    stream: str,
    ood_ratio: float,
    seeds: Iterable[int],
    canonical: bool = True,
    pilot_name: str | None = None,
    evidence_dir: str | Path | None = None,
    device: str = "cuda",
    max_eval_samples: int | None = None,
    stream_block_size: int = 64,
    config_dir: str | Path = REPOSITORY_ROOT / "cfg",
    artifact_provenance: str = "fast",
    data_root: str | Path = "~/data",
    per_domain_source_budget: int = OPEN_SET_PER_DOMAIN_SOURCE_BUDGET,
) -> list[ExperimentRun]:
    """Plan one held-out §26 cell, preserving exact NoAdapt pairing.

    ``canonical=True`` is deliberately strict: only an explicit CUDA identity
    is valid.  A local MPS/CPU smoke plan must opt into ``canonical=False`` and
    carry a stable name, which prevents its evidence directory from being
    mistaken for a canonical ablation result.
    """
    seeds = tuple(seeds)
    _validate_inputs(
        stream=stream, ood_ratio=ood_ratio, seeds=seeds, device=device,
        canonical=canonical, pilot_name=pilot_name,
        per_domain_source_budget=per_domain_source_budget,
    )
    if evidence_dir is None:
        suffix = "canonical" if canonical else f"pilot-{_validate_pilot_name(pilot_name)}"
        evidence_dir = REPOSITORY_ROOT / "evidence" / f"open-set-consensus-ablation-{suffix}"

    base_runs = build_experiment_matrix(
        datasets=(OPEN_SET_DATASET,), streams=(stream,), methods=CONSENSUS_ABLATION_METHODS,
        seeds=seeds, evidence_dir=evidence_dir, device=device,
        max_eval_samples=max_eval_samples, stream_block_size=stream_block_size,
        config_dir=config_dir, artifact_provenance=artifact_provenance, data_root=data_root,
        _allowed_methods=CONSENSUS_ABLATION_METHODS,
    )
    missing = [run.method for run in base_runs if run.config_hash == MISSING_CONFIG_HASH]
    if missing:
        raise ValueError("missing fixed Consensus ablation config(s): " + ", ".join(sorted(set(missing))))

    planned: list[ExperimentRun] = []
    baselines: dict[int, Path] = {}
    for base in base_runs:
        run_id = make_run_id(
            base.dataset, base.stream_mode, base.seed, base.method,
            device=base.device, max_eval_samples=base.max_eval_samples,
            stream_block_size=base.stream_block_size, config_hash=base.config_hash,
            artifact_provenance=base.artifact_provenance, data_root=base.data_root,
            open_set_ood_ratio=float(ood_ratio),
            open_set_per_domain_source_budget=per_domain_source_budget,
        )
        run = replace(
            base, run_id=run_id, reference_trace=None, open_set=True,
            known_class_split=OPEN_SET_SPLIT, ood_ratio=float(ood_ratio),
            open_set_per_domain_source_budget=per_domain_source_budget,
        )
        if run.method == "NoAdapt":
            baselines[run.seed] = run.run_dir / "trace.jsonl"
        else:
            run = replace(run, reference_trace=baselines[run.seed])
        planned.append(run)
    return planned


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, choices=OPEN_SET_STREAMS)
    parser.add_argument("--ood-ratio", required=True, type=float, choices=OPEN_SET_OOD_RATIOS)
    parser.add_argument("--seed", action="append", type=int, required=True, help="repeat for each held-out seed")
    parser.add_argument("--device", default="cuda", choices=SUPPORTED_DEVICES)
    parser.add_argument("--noncanonical-pilot", action="store_true")
    parser.add_argument("--pilot-name")
    parser.add_argument("--evidence-dir")
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--config-dir", default=str(REPOSITORY_ROOT / "cfg"))
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--stream-block-size", type=int, default=64)
    parser.add_argument("--artifact-provenance", default="fast")
    parser.add_argument("--per-domain-source-budget", type=int, default=OPEN_SET_PER_DOMAIN_SOURCE_BUDGET)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runs = build_consensus_ablation_matrix(
        stream=args.stream, ood_ratio=args.ood_ratio, seeds=args.seed,
        canonical=not args.noncanonical_pilot, pilot_name=args.pilot_name,
        evidence_dir=args.evidence_dir, device=args.device,
        max_eval_samples=args.max_eval_samples, stream_block_size=args.stream_block_size,
        config_dir=args.config_dir, artifact_provenance=args.artifact_provenance,
        data_root=args.data_root, per_domain_source_budget=args.per_domain_source_budget,
    )
    print(json.dumps({
        "status": "planned_not_executed",
        "canonical": not args.noncanonical_pilot,
        "run_count": len(runs),
        "runs": [run.to_dict() for run in runs],
        "commands": [build_command(run) for run in runs],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
