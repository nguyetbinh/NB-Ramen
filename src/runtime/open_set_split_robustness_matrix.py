"""Plan the immutable CIFAR-100-C v2/v3 split-robustness study.

This module is plan-only.  Its intentionally closed builder prevents a
partial grid, a config fallback, or a substituted split artifact from being
reported as the preregistered robustness result.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .experiment_matrix import (
        MISSING_CONFIG_HASH, REPOSITORY_ROOT, ExperimentRun, build_command,
        build_experiment_matrix, make_run_id,
    )
except ImportError:  # pragma: no cover - direct-file execution only
    from experiment_matrix import (  # type: ignore
        MISSING_CONFIG_HASH, REPOSITORY_ROOT, ExperimentRun, build_command,
        build_experiment_matrix, make_run_id,
    )


PREREGISTRATION_VERSION = "split-robustness-v1"
DATASET = "CIFAR100C"
SPLITS = (
    "open-set-cifar100-name-rank-v2",
    "open-set-cifar100-name-rank-v3",
)
SPLIT_FILENAMES = {
    SPLITS[0]: "open-set-cifar100-split-v2.json",
    SPLITS[1]: "open-set-cifar100-split-v3.json",
}
# These are SHA-256 values for the *full JSON bytes*, not just their internal
# recipe fingerprints.  Updating a split requires a new preregistration.
SPLIT_SHA256 = {
    SPLITS[0]: "1bfa878fac590ffacfd99595e2eb9ca2f0ee6fabf8415dba46a8e0a3f43eef66",
    SPLITS[1]: "c4e431524f9f4b354905fc0cf43a712b6cf02497777cc5da9a83552f81a68dd8",
}
RATIOS = (0.3, 0.5)
STREAMS = ("block", "recurring")
SEEDS = (0, 1, 2)
METHODS = ("NoAdapt", "Ramen", "ConsensusRamen", "OracleIDGradientRamen")
CONFIG_SHA256 = {
    "Ramen": "54c124be79a3c1536a8a95c68f34b41b31d84d40972f54fcd5b2c0016552ef27",
    "ConsensusRamen": "8a9d6fe4bb663653bf275fef4ecafe4d91ddf7415fc45c7eb3ea4634ac6bb34e",
    "OracleIDGradientRamen": "cd422352be0e640c73a14f6edf671fee4cf89d34c9cd61e444ce27c5516dadb3",
}
CONFIG_SURFACES = {
    "Ramen": {"max_capacity": 750, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01},
    "ConsensusRamen": {"max_capacity": 750, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01,
                       "consensus_threshold": 0.2, "min_consensus_classes": 3,
                       "consensus_mode": "hard_mask", "include_current": True},
    "OracleIDGradientRamen": {"max_capacity": 750, "topk": 5, "beta": 5.0,
                               "oracle_ood_source": "evaluator_is_ood", "optimizer": "signsgd", "lr": 0.01},
}
SOURCE_BUDGET = 400
EVIDENCE_DIR = REPOSITORY_ROOT / "evidence/open-set-cifar100c-split-robustness-v1"


def _split_path(split: str) -> Path:
    return REPOSITORY_ROOT / "cfg" / "research" / SPLIT_FILENAMES[split]


def _validate_split(split: str) -> dict[str, object]:
    path = _split_path(split)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_SHA256[split]:
        raise ValueError(f"preregistered split JSON digest drift: {split}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid preregistered split JSON: {split}") from exc
    if not isinstance(value, dict) or value.get("version") != split:
        raise ValueError(f"preregistered split version drift: {split}")
    if not isinstance(value.get("recipe"), dict) or not isinstance(value.get("fingerprint"), str):
        raise ValueError(f"preregistered split recipe/fingerprint missing: {split}")
    return value


def _validate_configs(runs: Iterable[ExperimentRun], config_dir: str | Path) -> None:
    root = Path(config_dir).expanduser().resolve() / DATASET
    for run in runs:
        if run.method == "NoAdapt":
            continue
        expected = root / f"{run.method}.yaml"
        if run.config_path != expected or run.config_hash == MISSING_CONFIG_HASH or run.config_sha256 is None:
            raise ValueError(f"preregistered method config missing or fallback selected: {run.method}")
        if hashlib.sha256(expected.read_bytes()).hexdigest() != CONFIG_SHA256[run.method]:
            raise ValueError(f"preregistered method config digest drift: {run.method}")
        if run.config_data != CONFIG_SURFACES[run.method]:
            raise ValueError(f"preregistered method config semantic surface drift: {run.method}")


def build_canonical_open_set_split_robustness_matrix(
    *, evidence_dir: str | Path = EVIDENCE_DIR,
    config_dir: str | Path = REPOSITORY_ROOT / "cfg",
    data_root: str | Path = "~/data",
) -> list[ExperimentRun]:
    """Return exactly 96 CUDA/full/exact split-bound runs; never execute."""
    split_metadata = {split: _validate_split(split) for split in SPLITS}
    base_runs = build_experiment_matrix(
        datasets=(DATASET,), streams=STREAMS, methods=METHODS, seeds=SEEDS,
        evidence_dir=evidence_dir, device="cuda", max_eval_samples=None,
        stream_block_size=64, config_dir=config_dir, artifact_provenance="exact",
        data_root=data_root, _allowed_methods=METHODS,
    )
    _validate_configs(base_runs, config_dir)
    planned: list[ExperimentRun] = []
    for split in SPLITS:
        for ratio in RATIOS:
            baselines: dict[tuple[str, int], Path] = {}
            for base in base_runs:
                run = replace(
                    base,
                    run_id=make_run_id(
                        base.dataset, base.stream_mode, base.seed, base.method,
                        device=base.device, max_eval_samples=base.max_eval_samples,
                        stream_block_size=base.stream_block_size, config_hash=base.config_hash,
                        artifact_provenance=base.artifact_provenance, data_root=base.data_root,
                        open_set_ood_ratio=ratio, open_set_per_domain_source_budget=SOURCE_BUDGET,
                        open_set_split_fingerprint=SPLIT_SHA256[split],
                    ),
                    reference_trace=None, open_set=True, known_class_split=split,
                    known_class_split_path=_split_path(split).resolve(),
                    known_class_split_sha256=SPLIT_SHA256[split],
                    ood_ratio=ratio, open_set_per_domain_source_budget=SOURCE_BUDGET,
                    require_config_lock=True,
                )
                cell = (run.stream_mode, run.seed)
                if run.method == "NoAdapt":
                    baselines[cell] = run.run_dir / "trace.jsonl"
                else:
                    run = replace(run, reference_trace=baselines[cell])
                planned.append(run)
    if len(planned) != 96 or len({run.run_id for run in planned}) != 96:
        raise AssertionError("split-robustness-v1 must contain exactly 96 unique runs")
    references: dict[Path, int] = {}
    for run in planned:
        if run.method != "NoAdapt":
            references[run.reference_trace] = references.get(run.reference_trace, 0) + 1
    if len(references) != 24 or set(references.values()) != {3}:
        raise AssertionError("split-robustness-v1 requires 24 baselines referenced by three adaptations")
    return planned


def _payload(runs: list[ExperimentRun]) -> dict[str, object]:
    return {
        "status": "planned_not_executed", "canonical": True,
        "preregistration_version": PREREGISTRATION_VERSION, "run_count": len(runs),
        "split_sha256": SPLIT_SHA256, "config_sha256": CONFIG_SHA256,
        "split_artifacts": {split: str(_split_path(split)) for split in SPLITS},
        "runs": [run.to_dict() for run in runs], "commands": [build_command(run) for run in runs],
        "artifacts": [str(run.run_dir) for run in runs],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", default=str(EVIDENCE_DIR))
    parser.add_argument("--config-dir", default=str(REPOSITORY_ROOT / "cfg"))
    parser.add_argument("--data-root", default="~/data")
    args = parser.parse_args(argv)
    print(json.dumps(_payload(build_canonical_open_set_split_robustness_matrix(
        evidence_dir=args.evidence_dir, config_dir=args.config_dir, data_root=args.data_root,
    )), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
