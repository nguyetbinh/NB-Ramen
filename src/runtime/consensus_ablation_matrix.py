"""Plan the preregistered ConsensusRamen held-out-v1 ablation contract.

The canonical builder is deliberately closed: it has no knobs for selecting
cells, hardware, stream prefixes, provenance, or method configuration.  A
separate explicitly named pilot builder exists for development-only work.
Neither builder executes a run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

try:  # Supports ``python -m runtime...`` and ``python -m src.runtime...``.
    from .experiment_matrix import (
        MISSING_CONFIG_HASH, OPEN_SET_DATASET, OPEN_SET_OOD_RATIOS,
        OPEN_SET_PER_DOMAIN_SOURCE_BUDGET, OPEN_SET_SPLIT, OPEN_SET_STREAMS,
        REPOSITORY_ROOT, SUPPORTED_ARTIFACT_PROVENANCE, SUPPORTED_DEVICES,
        ExperimentRun, build_command, build_experiment_matrix, make_run_id,
    )
except ImportError:  # pragma: no cover - direct-file execution only
    from experiment_matrix import (
        MISSING_CONFIG_HASH, OPEN_SET_DATASET, OPEN_SET_OOD_RATIOS,
        OPEN_SET_PER_DOMAIN_SOURCE_BUDGET, OPEN_SET_SPLIT, OPEN_SET_STREAMS,
        REPOSITORY_ROOT, SUPPORTED_ARTIFACT_PROVENANCE, SUPPORTED_DEVICES,
        ExperimentRun, build_command, build_experiment_matrix, make_run_id,
    )


PREREGISTRATION_VERSION = "heldout-v1"
CANONICAL_STREAMS = ("iid_mixed", "block", "recurring")
CANONICAL_OOD_RATIO = 0.3
CANONICAL_SEEDS = (0, 1, 2)
CANONICAL_DEVICE = "cuda"
CANONICAL_MAX_EVAL_SAMPLES = None
CANONICAL_STREAM_BLOCK_SIZE = 64
CANONICAL_ARTIFACT_PROVENANCE = "exact"
CANONICAL_SOURCE_BUDGET = 400
CANONICAL_EVIDENCE_DIR = REPOSITORY_ROOT / "evidence/open-set-consensus-ablation-heldout-v1"

CONSENSUS_ABLATION_METHODS = (
    "Ramen", "ConsensusRamen", "ConsensusRamenSoft", "ConsensusRamenNoSelf",
    "ConsensusRamenTau060", "ConsensusRamenMin2", "ConsensusRamenMin4",
)
REQUIRED_CONSENSUS_ALIASES = frozenset(CONSENSUS_ABLATION_METHODS[2:])
PILOT_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")

# Full digests (rather than experiment_matrix's abbreviated ID token) are the
# immutable config provenance for this preregistration.
CANONICAL_CONFIG_SHA256 = {
    "Ramen": "54c124be79a3c1536a8a95c68f34b41b31d84d40972f54fcd5b2c0016552ef27",
    "ConsensusRamen": "8a9d6fe4bb663653bf275fef4ecafe4d91ddf7415fc45c7eb3ea4634ac6bb34e",
    "ConsensusRamenSoft": "5cbd2b2cee48491016d547030fc65fa50ff0e16a775d98d574c141035bf5e356",
    "ConsensusRamenNoSelf": "2abe743078b251e9d3c04cccc334112a71298b269da3c014e246bd752f88c071",
    "ConsensusRamenTau060": "a8b9adb2e33697c1652ef2c55f30132daa4a96c08295a0a15468c7380fd30736",
    "ConsensusRamenMin2": "32ded9874f2a14dbaf8568ab5b1756b84eb96ab977c69bd7bf7378220e7bd1bf",
    "ConsensusRamenMin4": "82a5ba1f498edcc3fa899078b0e35c05d61f833def4b36558140521f8f5fe006",
}
_RAMEN_SURFACE = {"max_capacity": 750, "topk": 5, "beta": 5.0, "optimizer": "signsgd", "lr": 0.01}
CANONICAL_CONFIG_SURFACES = {
    "Ramen": _RAMEN_SURFACE,
    "ConsensusRamen": {**_RAMEN_SURFACE, "consensus_threshold": .2, "min_consensus_classes": 3, "consensus_mode": "hard_mask", "include_current": True},
    "ConsensusRamenSoft": {**_RAMEN_SURFACE, "consensus_mode": "soft_weight", "consensus_gamma": 1.0, "consensus_seed": 1729, "consensus_threshold": .2, "min_consensus_classes": 3, "include_current": True},
    "ConsensusRamenNoSelf": {**_RAMEN_SURFACE, "consensus_threshold": .2, "min_consensus_classes": 3, "consensus_mode": "hard_mask", "include_current": False},
    "ConsensusRamenTau060": {**_RAMEN_SURFACE, "consensus_threshold": .6, "min_consensus_classes": 3, "consensus_mode": "hard_mask", "include_current": True},
    "ConsensusRamenMin2": {**_RAMEN_SURFACE, "consensus_threshold": .2, "min_consensus_classes": 2, "consensus_mode": "hard_mask", "include_current": True},
    "ConsensusRamenMin4": {**_RAMEN_SURFACE, "consensus_threshold": .2, "min_consensus_classes": 4, "consensus_mode": "hard_mask", "include_current": True},
}


def _validate_pilot_name(pilot_name: str | None) -> str:
    if not isinstance(pilot_name, str) or not pilot_name:
        raise ValueError("noncanonical pilot planning requires a non-empty pilot_name")
    if (pilot_name[0] == "-" or pilot_name[-1] == "-"
            or any(character not in PILOT_NAME_CHARS for character in pilot_name)):
        raise ValueError("pilot_name must use lowercase letters, digits, and internal hyphens only")
    return pilot_name


def _validate_pilot_inputs(*, stream: str, ood_ratio: float, seeds: tuple[int, ...], device: str,
                           per_domain_source_budget: int, artifact_provenance: str) -> None:
    if stream not in OPEN_SET_STREAMS:
        raise ValueError("held-out stream must be one of: " + ", ".join(OPEN_SET_STREAMS))
    if (not isinstance(ood_ratio, (int, float)) or isinstance(ood_ratio, bool)
            or not math.isfinite(ood_ratio) or ood_ratio not in OPEN_SET_OOD_RATIOS):
        raise ValueError("held-out OOD ratio must be one of 0, 0.1, 0.3, 0.5")
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("provide at least one integer seed")
    if device not in SUPPORTED_DEVICES:
        raise ValueError("device must be one of " + ", ".join(SUPPORTED_DEVICES))
    if artifact_provenance not in SUPPORTED_ARTIFACT_PROVENANCE:
        raise ValueError("artifact_provenance must be one of " + ", ".join(SUPPORTED_ARTIFACT_PROVENANCE))
    if (not isinstance(per_domain_source_budget, int) or isinstance(per_domain_source_budget, bool)
            or per_domain_source_budget <= 0):
        raise ValueError("per_domain_source_budget must be a positive integer")
    if per_domain_source_budget % Fraction(str(ood_ratio)).denominator:
        raise ValueError("per-domain source budget must support the selected OOD ratio")


def _build_cell(*, stream: str, ood_ratio: float, seeds: tuple[int, ...], evidence_dir: str | Path,
                device: str, max_eval_samples: int | None, stream_block_size: int,
                config_dir: str | Path, artifact_provenance: str, data_root: str | Path,
                per_domain_source_budget: int) -> list[ExperimentRun]:
    base_runs = build_experiment_matrix(
        datasets=(OPEN_SET_DATASET,), streams=(stream,), methods=CONSENSUS_ABLATION_METHODS,
        seeds=seeds, evidence_dir=evidence_dir, device=device, max_eval_samples=max_eval_samples,
        stream_block_size=stream_block_size, config_dir=config_dir,
        artifact_provenance=artifact_provenance, data_root=data_root,
        _allowed_methods=CONSENSUS_ABLATION_METHODS,
    )
    planned: list[ExperimentRun] = []
    baselines: dict[int, Path] = {}
    for base in base_runs:
        run = replace(
            base,
            run_id=make_run_id(base.dataset, base.stream_mode, base.seed, base.method,
                device=base.device, max_eval_samples=base.max_eval_samples,
                stream_block_size=base.stream_block_size, config_hash=base.config_hash,
                artifact_provenance=base.artifact_provenance, data_root=base.data_root,
                open_set_ood_ratio=float(ood_ratio),
                open_set_per_domain_source_budget=per_domain_source_budget),
            reference_trace=None, open_set=True, known_class_split=OPEN_SET_SPLIT,
            ood_ratio=float(ood_ratio), open_set_per_domain_source_budget=per_domain_source_budget,
        )
        if run.method == "NoAdapt":
            baselines[run.seed] = run.run_dir / "trace.jsonl"
        else:
            run = replace(run, reference_trace=baselines[run.seed])
        planned.append(run)
    return planned


def _validate_canonical_configs(runs: Iterable[ExperimentRun], config_dir: str | Path) -> None:
    expected_root = Path(config_dir).expanduser().resolve() / OPEN_SET_DATASET
    for run in runs:
        if run.method == "NoAdapt":
            continue
        expected_path = expected_root / f"{run.method}.yaml"
        if run.config_path is None or run.config_hash == MISSING_CONFIG_HASH or run.config_path != expected_path:
            raise ValueError(f"canonical heldout-v1 config is missing or resolved a fallback: {run.method}")
        raw = expected_path.read_bytes()
        actual_digest = hashlib.sha256(raw).hexdigest()
        expected_digest = CANONICAL_CONFIG_SHA256[run.method]
        if actual_digest != expected_digest:
            raise ValueError(f"canonical heldout-v1 config digest drift: {run.method}")
        if run.config_data != CANONICAL_CONFIG_SURFACES[run.method]:
            raise ValueError(f"canonical heldout-v1 config semantic surface drift: {run.method}")
        if run.config_hash != actual_digest[:12]:
            raise ValueError(f"canonical heldout-v1 config hash mismatch: {run.method}")


def _validate_heldout_v1_runs(runs: list[ExperimentRun]) -> None:
    if len(runs) != 72:
        raise ValueError("canonical heldout-v1 plan must contain exactly 72 runs")
    if len({run.run_id for run in runs}) != 72:
        raise ValueError("canonical heldout-v1 plan contains duplicate run IDs")
    baseline_paths: set[Path] = set()
    references: dict[Path, int] = {}
    for stream in CANONICAL_STREAMS:
        for seed in CANONICAL_SEEDS:
            cell = [run for run in runs if run.stream_mode == stream and run.seed == seed]
            if [run.method for run in cell] != ["NoAdapt", *CONSENSUS_ABLATION_METHODS]:
                raise ValueError("canonical heldout-v1 method order/pairing drift")
            baseline = cell[0].run_dir / "trace.jsonl"
            baseline_paths.add(baseline)
            for run in cell[1:]:
                references[run.reference_trace] = references.get(run.reference_trace, 0) + 1
    if len(baseline_paths) != 9 or set(references) != baseline_paths or set(references.values()) != {7}:
        raise ValueError("canonical heldout-v1 requires nine baselines referenced by exactly seven adaptations")


def build_canonical_consensus_ablation_heldout_v1(*, evidence_dir: str | Path = CANONICAL_EVIDENCE_DIR,
                                                  config_dir: str | Path = REPOSITORY_ROOT / "cfg",
                                                  data_root: str | Path = "~/data") -> list[ExperimentRun]:
    """Atomically return the immutable 72-run held-out-v1 plan; never execute it."""
    planned = [run for stream in CANONICAL_STREAMS for run in _build_cell(
        stream=stream, ood_ratio=CANONICAL_OOD_RATIO, seeds=CANONICAL_SEEDS,
        evidence_dir=evidence_dir, device=CANONICAL_DEVICE,
        max_eval_samples=CANONICAL_MAX_EVAL_SAMPLES, stream_block_size=CANONICAL_STREAM_BLOCK_SIZE,
        config_dir=config_dir, artifact_provenance=CANONICAL_ARTIFACT_PROVENANCE,
        data_root=data_root, per_domain_source_budget=CANONICAL_SOURCE_BUDGET)]
    _validate_canonical_configs(planned, config_dir)
    _validate_heldout_v1_runs(planned)
    return planned


def build_noncanonical_consensus_ablation_pilot(*, pilot_name: str, stream: str, ood_ratio: float,
                                                seeds: Iterable[int], evidence_dir: str | Path | None = None,
                                                device: str = "cuda", max_eval_samples: int | None = None,
                                                stream_block_size: int = 64,
                                                config_dir: str | Path = REPOSITORY_ROOT / "cfg",
                                                artifact_provenance: str = "fast", data_root: str | Path = "~/data",
                                                per_domain_source_budget: int = OPEN_SET_PER_DOMAIN_SOURCE_BUDGET) -> list[ExperimentRun]:
    """Plan a named development pilot. Its output is never canonical evidence."""
    pilot_name = _validate_pilot_name(pilot_name)
    seeds = tuple(seeds)
    _validate_pilot_inputs(stream=stream, ood_ratio=ood_ratio, seeds=seeds, device=device,
                           per_domain_source_budget=per_domain_source_budget,
                           artifact_provenance=artifact_provenance)
    if evidence_dir is None:
        evidence_dir = REPOSITORY_ROOT / "evidence" / f"open-set-consensus-ablation-pilot-{pilot_name}"
    return _build_cell(stream=stream, ood_ratio=ood_ratio, seeds=seeds, evidence_dir=evidence_dir,
                       device=device, max_eval_samples=max_eval_samples, stream_block_size=stream_block_size,
                       config_dir=config_dir, artifact_provenance=artifact_provenance, data_root=data_root,
                       per_domain_source_budget=per_domain_source_budget)


def build_consensus_ablation_matrix(**kwargs: object) -> list[ExperimentRun]:
    """Compatibility entry point for pilots; canonical work must use heldout-v1."""
    if kwargs.pop("canonical", False):
        raise ValueError("canonical Consensus ablations require build_canonical_consensus_ablation_heldout_v1")
    return build_noncanonical_consensus_ablation_pilot(**kwargs)  # type: ignore[arg-type]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noncanonical-pilot", action="store_true")
    parser.add_argument("--pilot-name")
    parser.add_argument("--stream", choices=OPEN_SET_STREAMS)
    parser.add_argument("--ood-ratio", type=float, choices=OPEN_SET_OOD_RATIOS)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--evidence-dir")
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--config-dir", default=str(REPOSITORY_ROOT / "cfg"))
    parser.add_argument("--device", choices=SUPPORTED_DEVICES)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--stream-block-size", type=int)
    parser.add_argument("--artifact-provenance", choices=SUPPORTED_ARTIFACT_PROVENANCE)
    parser.add_argument("--per-domain-source-budget", type=int)
    return parser


def _plan_payload(runs: list[ExperimentRun], *, canonical: bool, pilot_name: str | None = None) -> dict[str, object]:
    return {
        "status": "planned_not_executed", "canonical": canonical,
        "preregistration_version": PREREGISTRATION_VERSION if canonical else None,
        "pilot_name": pilot_name, "run_count": len(runs),
        "runs": [run.to_dict() for run in runs], "commands": [build_command(run) for run in runs],
        "artifacts": [str(run.run_dir) for run in runs],
        "config_sha256": CANONICAL_CONFIG_SHA256 if canonical else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.noncanonical_pilot:
        missing = [name for name in ("pilot_name", "stream", "ood_ratio", "seed") if getattr(args, name) in (None, [])]
        if missing:
            _parser().error("--noncanonical-pilot requires --pilot-name, --stream, --ood-ratio, and --seed")
        runs = build_noncanonical_consensus_ablation_pilot(
            pilot_name=args.pilot_name, stream=args.stream, ood_ratio=args.ood_ratio, seeds=args.seed,
            evidence_dir=args.evidence_dir, data_root=args.data_root, config_dir=args.config_dir,
            device=args.device or "cuda", max_eval_samples=args.max_eval_samples,
            stream_block_size=args.stream_block_size or 64, artifact_provenance=args.artifact_provenance or "fast",
            per_domain_source_budget=args.per_domain_source_budget or OPEN_SET_PER_DOMAIN_SOURCE_BUDGET)
        payload = _plan_payload(runs, canonical=False, pilot_name=args.pilot_name)
    else:
        forbidden = (args.stream, args.ood_ratio, args.seed, args.device, args.max_eval_samples,
                     args.stream_block_size, args.artifact_provenance, args.per_domain_source_budget, args.pilot_name)
        if any(value is not None for value in forbidden):
            _parser().error("heldout-v1 is fixed; use --noncanonical-pilot for custom cells")
        runs = build_canonical_consensus_ablation_heldout_v1(
            evidence_dir=args.evidence_dir or CANONICAL_EVIDENCE_DIR,
            config_dir=args.config_dir, data_root=args.data_root)
        payload = _plan_payload(runs, canonical=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
