#!/usr/bin/env python3
"""Run the minimal reuse-first latent-soft-routing Gate 1 check.

The protocol is intentionally fixed to CIFAR-100-C, the canonical block
stream, 200 samples, seed 0, and gamma in {0, 0.25}.  Gamma zero verifies
baseline recovery; gamma 0.25 is the single nonzero diagnostic.  Each gamma
has an immutable config root so the existing matrix runner can validate and
reuse completed evidence with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from runtime.experiment_matrix import (
    SUPPORTED_ARTIFACT_PROVENANCE,
    ExperimentRun,
    build_experiment_matrix,
    execute_matrix,
    validate_completed_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMMA_CONFIGS = (
    (0.0, PROJECT_ROOT / "cfg/research/latent-soft-routing/gamma-000"),
    (0.25, PROJECT_ROOT / "cfg/research/latent-soft-routing/gamma-025"),
)


def build_gate1_groups(
    *,
    evidence_dir: str | Path,
    data_root: str | Path,
    device: str,
    artifact_provenance: str = "fast",
) -> list[tuple[str, list[ExperimentRun]]]:
    """Build control and fixed-gamma groups in deterministic execution order."""
    common = {
        "datasets": ("CIFAR100C",),
        "streams": ("block",),
        "seeds": (0,),
        "evidence_dir": evidence_dir,
        "data_root": data_root,
        "device": device,
        "max_eval_samples": 200,
        "stream_block_size": 64,
        "artifact_provenance": artifact_provenance,
    }
    groups = [(
        "controls",
        build_experiment_matrix(
            methods=("CausalRamen", "OracleHardRamen"),
            config_dir=PROJECT_ROOT / "cfg",
            **common,
        ),
    )]
    for gamma, config_dir in GAMMA_CONFIGS:
        runs = build_experiment_matrix(
            methods=("OracleSoftRankRamen",), config_dir=config_dir, **common,
        )
        soft_run = next(run for run in runs if run.method == "OracleSoftRankRamen")
        actual_gamma = soft_run.config_data.get("gamma")
        if actual_gamma != gamma:
            raise ValueError(
                f"fixed Gate 1 config mismatch: expected gamma={gamma}, got {actual_gamma}"
            )
        groups.append((f"gamma-{gamma:g}", runs))
    run_ids = [run.run_id for _, runs in groups for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Gate 1 config roots must produce unique run IDs")
    return groups


def collect_gate1_results(
    groups: list[tuple[str, list[ExperimentRun]]],
) -> dict[str, object]:
    """Validate completed evidence and require one identical stream fingerprint."""
    results = []
    fingerprints = set()
    for group, runs in groups:
        for run in runs:
            evidence = validate_completed_run(run)
            summary = evidence["summary"]
            fingerprints.add(summary["stream_fingerprint"])
            results.append({
                "group": group,
                "method": run.method,
                "gamma": run.config_data.get("gamma"),
                "run_id": run.run_id,
                "micro_accuracy": summary["micro_accuracy"],
                "macro_domain_accuracy": summary["macro_domain_accuracy"],
                "worst_domain_accuracy": summary["worst_domain_accuracy"],
                "negative_adaptation_rate": summary["negative_adaptation_rate"],
                "support_composition_diagnostics": summary.get(
                    "support_composition_diagnostics"
                ),
                "soft_routing_diagnostics": summary.get("soft_routing_diagnostics"),
                "stream_fingerprint": summary["stream_fingerprint"],
            })
    if len(fingerprints) != 1:
        raise ValueError("Gate 1 runs did not use one identical canonical stream")
    return {"stream_fingerprint": fingerprints.pop(), "results": results}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--artifact-provenance",
        choices=SUPPORTED_ARTIFACT_PROVENANCE,
        default="fast",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    groups = build_gate1_groups(
        evidence_dir=args.evidence_dir,
        data_root=args.data_root,
        device=args.device,
        artifact_provenance=args.artifact_provenance,
    )
    payload: dict[str, object] = {
        "protocol": {
            "dataset": "CIFAR100C",
            "stream": "block",
            "samples": 200,
            "seed": 0,
            "block_size": 64,
            "gamma": [gamma for gamma, _ in GAMMA_CONFIGS],
        },
        "groups": [
            {"name": name, "runs": [run.to_dict() for run in runs]}
            for name, runs in groups
        ],
    }
    if args.execute:
        outcomes = []
        for name, runs in groups:
            outcomes.append({
                "name": name,
                "runs": execute_matrix(
                    runs,
                    python_executable=args.python,
                    data_root=args.data_root,
                    resume=args.resume,
                ),
            })
        payload["outcomes"] = outcomes
        payload["evidence"] = collect_gate1_results(groups)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
