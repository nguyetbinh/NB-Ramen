"""Plan, but never execute, the canonical DomainNet open-set evidence grid.

This module is intentionally independent from the CIFAR-100-C planner.  It
locks the secondary natural-domain benchmark to its own semantic split and
source-exposure rule so a partial run cannot silently be treated as the Phase
E result.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from typing import Iterable, Sequence

try:  # Supports ``python -m runtime...`` and ``python -m src.runtime...``.
    from .experiment_matrix import (
        MISSING_CONFIG_HASH,
        REPOSITORY_ROOT,
        ExperimentRun,
        build_command,
        build_experiment_matrix,
        make_run_id,
    )
except ImportError:  # pragma: no cover - direct-file execution only
    from experiment_matrix import (
        MISSING_CONFIG_HASH,
        REPOSITORY_ROOT,
        ExperimentRun,
        build_command,
        build_experiment_matrix,
        make_run_id,
    )


DOMAINNET_OPEN_SET_DATASET = "DomainNet"
DOMAINNET_OPEN_SET_SPLIT = "open-set-domainnet-name-rank-v1"
DOMAINNET_OPEN_SET_STREAMS = ("iid_mixed", "block", "recurring")
DOMAINNET_OPEN_SET_OOD_RATIOS = (0.0, 0.1, 0.3, 0.5)
DOMAINNET_OPEN_SET_SEEDS = (0, 1, 2)
DOMAINNET_OPEN_SET_METHODS = (
    "NoAdapt",
    "Ramen",
    "EntropyGatedLatentRamen",
    "OracleDropOODRamen",
    "OracleIDGradientRamen",
    "ConsensusRamen",
    "OracleConsensusRamen",
)

# 690 is the smallest multiple of ten that assigns one example to each of
# DomainNet's 69 held-out classes at the 10% OOD condition.  That prevents the
# natural-domain secondary benchmark from converting semantic novelty into a
# small, arbitrary unknown-class subset.  It also remains a modest 4,140
# source examples across DomainNet's six environments before any stream order.
DOMAINNET_OPEN_SET_PER_DOMAIN_SOURCE_BUDGET = 690


def _exact(value: Iterable[object], expected: tuple[object, ...], label: str) -> tuple[object, ...]:
    result = tuple(value)
    if result != expected:
        raise ValueError(f"canonical DomainNet open-set {label} is fixed to: " + ", ".join(map(str, expected)))
    return result


def _validate_inputs(
    *, streams: Iterable[str], ood_ratios: Iterable[float], seeds: Iterable[int],
    methods: Iterable[str], device: str, max_eval_samples: int | None,
    artifact_provenance: str, per_domain_source_budget: int,
) -> None:
    _exact(streams, DOMAINNET_OPEN_SET_STREAMS, "streams")
    _exact(ood_ratios, DOMAINNET_OPEN_SET_OOD_RATIOS, "OOD ratios")
    _exact(seeds, DOMAINNET_OPEN_SET_SEEDS, "seeds")
    _exact(methods, DOMAINNET_OPEN_SET_METHODS, "methods")
    if device != "cuda":
        raise ValueError("canonical DomainNet open-set matrix requires device='cuda'")
    if max_eval_samples is not None:
        raise ValueError("canonical DomainNet open-set matrix requires full streams (max_eval_samples=None)")
    if artifact_provenance not in {"fast", "exact"}:
        raise ValueError("canonical DomainNet open-set matrix requires artifact_provenance='fast' or 'exact'")
    if per_domain_source_budget != DOMAINNET_OPEN_SET_PER_DOMAIN_SOURCE_BUDGET:
        raise ValueError(
            "canonical DomainNet open-set per-domain source budget is fixed to "
            f"{DOMAINNET_OPEN_SET_PER_DOMAIN_SOURCE_BUDGET}"
        )
    if any(per_domain_source_budget % Fraction(str(ratio)).denominator for ratio in DOMAINNET_OPEN_SET_OOD_RATIOS):
        raise AssertionError("locked DomainNet source budget does not support its locked OOD ratios")


def _validate_configs(runs: Iterable[ExperimentRun]) -> None:
    """Fail closed if a named baseline loses its dataset-specific config."""
    required = set(DOMAINNET_OPEN_SET_METHODS) - {"NoAdapt"}
    seen: set[str] = set()
    for run in runs:
        if run.method not in required:
            continue
        seen.add(run.method)
        if run.config_path is None or run.config_hash == MISSING_CONFIG_HASH:
            raise ValueError(f"missing fixed DomainNet open-set config: {run.method}")
        if run.config_path.parent.name != DOMAINNET_OPEN_SET_DATASET:
            raise ValueError(f"DomainNet open-set method resolved a fallback config: {run.method}")
        required_keys = {"max_capacity", "topk", "beta", "optimizer", "lr"}
        if missing := sorted(required_keys.difference(run.config_data)):
            raise ValueError(f"DomainNet open-set config is incomplete for {run.method}: " + ", ".join(missing))
        if run.method.startswith("Oracle") and run.config_data.get("oracle_ood_source") != "evaluator_is_ood":
            raise ValueError(f"DomainNet oracle config must declare evaluator_is_ood: {run.method}")
        if run.method in {"ConsensusRamen", "OracleConsensusRamen"}:
            locked = {
                "consensus_threshold": 0.2,
                "min_consensus_classes": 3,
                "consensus_mode": "hard_mask",
                "include_current": True,
            }
            if any(run.config_data.get(key) != value for key, value in locked.items()):
                raise ValueError(f"DomainNet ConsensusRamen-v0 config is not locked: {run.method}")
    missing_methods = required.difference(seen)
    if missing_methods:
        raise AssertionError("planner did not resolve config(s): " + ", ".join(sorted(missing_methods)))


def build_domainnet_open_set_evidence_matrix(
    *,
    streams: Iterable[str] = DOMAINNET_OPEN_SET_STREAMS,
    ood_ratios: Iterable[float] = DOMAINNET_OPEN_SET_OOD_RATIOS,
    seeds: Iterable[int] = DOMAINNET_OPEN_SET_SEEDS,
    methods: Iterable[str] = DOMAINNET_OPEN_SET_METHODS,
    evidence_dir: str | Path = REPOSITORY_ROOT / "evidence/open-set-domainnet-canonical",
    device: str = "cuda",
    max_eval_samples: int | None = None,
    stream_block_size: int = 64,
    config_dir: str | Path = REPOSITORY_ROOT / "cfg",
    artifact_provenance: str = "fast",
    data_root: str | Path = "~/data",
    per_domain_source_budget: int = DOMAINNET_OPEN_SET_PER_DOMAIN_SOURCE_BUDGET,
) -> list[ExperimentRun]:
    """Return the full 252-run, CUDA-only Phase E DomainNet plan.

    This is planner-only.  It neither checks for DomainNet files nor launches
    a process; a verified six-environment taxonomy is required later when the
    emitted commands build their streams.  The normal stream builder binds the
    materialized split fingerprint and taxonomy SHA-256 into each manifest and
    stream fingerprint at execution time.
    """
    streams, ratios, seeds, methods = (
        tuple(streams), tuple(ood_ratios), tuple(seeds), tuple(methods)
    )
    _validate_inputs(
        streams=streams, ood_ratios=ratios, seeds=seeds, methods=methods,
        device=device, max_eval_samples=max_eval_samples,
        artifact_provenance=artifact_provenance,
        per_domain_source_budget=per_domain_source_budget,
    )
    base_runs = build_experiment_matrix(
        datasets=(DOMAINNET_OPEN_SET_DATASET,), streams=streams, methods=methods,
        seeds=seeds, evidence_dir=evidence_dir, device=device,
        max_eval_samples=max_eval_samples, stream_block_size=stream_block_size,
        config_dir=config_dir, artifact_provenance=artifact_provenance,
        data_root=data_root, _allowed_methods=methods,
    )
    _validate_configs(base_runs)
    planned: list[ExperimentRun] = []
    for ratio in ratios:
        baselines: dict[tuple[str, int], Path] = {}
        for base in base_runs:
            run_id = make_run_id(
                base.dataset, base.stream_mode, base.seed, base.method,
                device=base.device, max_eval_samples=base.max_eval_samples,
                stream_block_size=base.stream_block_size, config_hash=base.config_hash,
                artifact_provenance=base.artifact_provenance, data_root=base.data_root,
                open_set_ood_ratio=float(ratio),
                open_set_per_domain_source_budget=per_domain_source_budget,
            )
            run = replace(
                base, run_id=run_id, reference_trace=None, open_set=True,
                known_class_split=DOMAINNET_OPEN_SET_SPLIT, ood_ratio=float(ratio),
                open_set_per_domain_source_budget=per_domain_source_budget,
            )
            cell = (run.stream_mode, run.seed)
            if run.method == "NoAdapt":
                baselines[cell] = run.run_dir / "trace.jsonl"
            else:
                run = replace(run, reference_trace=baselines[cell])
            planned.append(run)
    if len(planned) != 252:
        raise AssertionError(f"locked DomainNet matrix must contain 252 runs, got {len(planned)}")
    return planned


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evidence-dir", default=str(REPOSITORY_ROOT / "evidence/open-set-domainnet-canonical"))
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--config-dir", default=str(REPOSITORY_ROOT / "cfg"))
    parser.add_argument("--stream-block-size", type=int, default=64)
    parser.add_argument("--artifact-provenance", default="fast")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runs = build_domainnet_open_set_evidence_matrix(
        device=args.device, evidence_dir=args.evidence_dir, data_root=args.data_root,
        config_dir=args.config_dir, stream_block_size=args.stream_block_size,
        artifact_provenance=args.artifact_provenance,
    )
    print(json.dumps({
        "status": "planned_not_executed",
        "canonical": True,
        "dataset": DOMAINNET_OPEN_SET_DATASET,
        "known_class_split": DOMAINNET_OPEN_SET_SPLIT,
        "run_count": len(runs),
        "runs": [run.to_dict() for run in runs],
        "commands": [build_command(run) for run in runs],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
