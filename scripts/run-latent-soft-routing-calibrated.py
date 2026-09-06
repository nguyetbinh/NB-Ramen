#!/usr/bin/env python3
"""Run the margin-calibrated, reuse-first oracle soft-routing Gate 1.

The ``profile`` stage runs only gamma zero and writes a calibration before any
nonzero accuracy is read.  The ``run`` stage consumes that immutable
calibration and runs exactly weak, medium, and strong OracleSoftRankRamen
cells, pairing all three with an already validated NoAdapt trace.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence

from runtime.experiment_matrix import (
    SUPPORTED_ARTIFACT_PROVENANCE,
    ExperimentRun,
    build_command,
    build_experiment_matrix,
    validate_completed_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARGIN_PROFILE_CONFIG_ROOT = PROJECT_ROOT / "cfg/research/latent-soft-routing/margin-profile"
CANONICAL = {
    "dataset": "CIFAR100C",
    "stream": "block",
    "seed": 0,
    "samples": 200,
    "block_size": 64,
}
STRENGTHS = (("weak", "replacement_margin_p25", 0.10),
             ("medium", "replacement_margin_p50", 0.25),
             ("strong", "replacement_margin_p75", 0.50))


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float32_next_up(value: object) -> float:
    """Return the next representable finite float32 strictly above ``value``."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("replacement-margin quantiles must be finite numbers")
    packed = struct.pack("!f", float(value))
    rounded = struct.unpack("!f", packed)[0]
    if not math.isfinite(rounded):
        raise ValueError("replacement-margin quantile overflows float32")
    bits = struct.unpack("!I", packed)[0]
    # All valid gamma values are nonnegative.  Canonicalize negative zero to
    # positive zero before advancing it to the smallest positive float32.
    if rounded < 0:
        raise ValueError("replacement-margin quantiles must be nonnegative")
    if bits == 0x80000000:
        bits = 0
    if bits >= 0x7F7FFFFF:
        raise ValueError("replacement-margin quantile has no finite float32 successor")
    result = struct.unpack("!f", struct.pack("!I", bits + 1))[0]
    if not math.isfinite(result) or not result > rounded:
        raise ValueError("cannot advance replacement-margin quantile")
    return result


def _pooled_margins(summary: Mapping[str, object]) -> Mapping[str, object]:
    profile = summary.get("replacement_margin_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("profile summary lacks replacement_margin_profile")
    pooled = {
        key: value for key, value in profile.items()
        if isinstance(key, str) and key.startswith("replacement_margin_p")
    }
    required = {field for _, field, _ in STRENGTHS}
    if not required.issubset(pooled) or any(pooled[field] is None for field in required):
        raise ValueError("replacement_margin_profile pooled quantiles are empty")
    return pooled


def derive_calibration(profile_evidence: Mapping[str, object]) -> dict[str, object]:
    """Create a self-verifiable calibration without consulting accuracy."""
    summary = profile_evidence.get("summary")
    manifest = profile_evidence.get("manifest")
    run = profile_evidence.get("run")
    if not isinstance(summary, Mapping) or not isinstance(manifest, Mapping) or not isinstance(run, ExperimentRun):
        raise ValueError("profile evidence is malformed")
    pooled = _pooled_margins(summary)
    gammas = []
    for name, field, target in STRENGTHS:
        gamma = _float32_next_up(pooled.get(field))
        gammas.append({"name": name, "target_selection_change_ratio": target,
                       "margin_quantile": field, "gamma": gamma})
    values = [entry["gamma"] for entry in gammas]
    if len(set(values)) != len(values) or values != sorted(values) or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("calibrated gamma values must be distinct and strictly ordered")
    trace = run.run_dir / "trace.jsonl"
    if not trace.is_file() or trace.stat().st_size <= 0:
        raise ValueError("profile trace is absent or empty")
    source = {
        "run_id": run.run_id,
        "stream_fingerprint": summary.get("stream_fingerprint"),
        "trace_sha256": _sha256_file(trace),
        "trace_path": str(trace.resolve()),
        "summary_path": str((run.run_dir / "summary.json").resolve()),
        "config_hash": run.config_hash,
        "config_path": str(run.config_path) if run.config_path else None,
    }
    if not isinstance(source["stream_fingerprint"], str) or len(source["stream_fingerprint"]) != 64:
        raise ValueError("profile summary has malformed stream fingerprint")
    calibration = {
        "schema_version": 1,
        "protocol": {**CANONICAL, "method": "OracleSoftRankRamen"},
        "source": source,
        "rules": {
            "selection": "Q25/Q50/Q75 pooled replacement margins; each threshold advances to next finite float32",
            "accuracy_read_before_selection": False,
            "targets": [entry["target_selection_change_ratio"] for entry in gammas],
        },
        "pooled_replacement_margins": dict(pooled),
        "strengths": gammas,
    }
    calibration_id = hashlib.sha256(_canonical_json(calibration)).hexdigest()
    return {**calibration, "calibration_id": calibration_id}


def write_calibration(calibration: Mapping[str, object], path: str | Path) -> Path:
    """Write calibration JSON and verify its content-addressed identity."""
    verified = validate_calibration(calibration)
    destination = _absolute(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(verified, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
        raise ValueError(f"immutable calibration already differs: {destination}")
    destination.write_text(encoded, encoding="utf-8")
    return destination


def validate_calibration(calibration: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration must be an object")
    document = dict(calibration)
    calibration_id = document.pop("calibration_id", None)
    if not isinstance(calibration_id, str) or len(calibration_id) != 64:
        raise ValueError("calibration_id is malformed")
    if hashlib.sha256(_canonical_json(document)).hexdigest() != calibration_id:
        raise ValueError("calibration_id does not match calibration content")
    protocol = document.get("protocol")
    if not isinstance(protocol, Mapping) or any(protocol.get(key) != value for key, value in CANONICAL.items()):
        raise ValueError("calibration protocol does not match the canonical cell")
    strengths = document.get("strengths")
    if not isinstance(strengths, list) or [item.get("name") if isinstance(item, Mapping) else None for item in strengths] != [x[0] for x in STRENGTHS]:
        raise ValueError("calibration must contain weak, medium, and strong strengths")
    gamma_values = []
    for item, (_, field, target) in zip(strengths, STRENGTHS):
        if not isinstance(item, Mapping) or item.get("margin_quantile") != field or item.get("target_selection_change_ratio") != target:
            raise ValueError("calibration strength rules are malformed")
        gamma = item.get("gamma")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or not math.isfinite(float(gamma)) or gamma < 0:
            raise ValueError("calibration gamma is malformed")
        gamma_values.append(float(gamma))
    if any(a >= b for a, b in zip(gamma_values, gamma_values[1:])):
        raise ValueError("calibration gamma values must be strictly ordered")
    source = document.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("trace_sha256"), str) or len(source["trace_sha256"]) != 64:
        raise ValueError("calibration source is malformed")
    for key in ("trace_path", "summary_path", "config_path", "stream_fingerprint", "run_id", "config_hash"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError("calibration source is malformed")
    return dict(calibration)


def load_calibration(path: str | Path) -> dict[str, object]:
    try:
        value = json.loads(_absolute(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid calibration JSON") from exc
    return validate_calibration(value)


def verify_calibration_source(calibration: Mapping[str, object]) -> None:
    """Reject a calibration whose profiled trace/config/stream no longer match."""
    document = validate_calibration(calibration)
    source = document["source"]
    trace = _absolute(source["trace_path"])
    config = _absolute(source["config_path"])
    summary_path = _absolute(source["summary_path"])
    if not trace.is_file() or trace.stat().st_size <= 0 or _sha256_file(trace) != source["trace_sha256"]:
        raise ValueError("calibration source trace identity mismatch")
    if not config.is_file() or hashlib.sha256(config.read_bytes()).hexdigest()[:12] != source["config_hash"]:
        raise ValueError("calibration source config identity mismatch")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("calibration source summary is unavailable") from exc
    if not isinstance(summary, Mapping) or summary.get("run_id") != source["run_id"] or summary.get("stream_fingerprint") != source["stream_fingerprint"]:
        raise ValueError("calibration source stream identity mismatch")


def materialize_configs(calibration: Mapping[str, object], output_root: str | Path) -> dict[str, Path]:
    """Materialize immutable JSON-as-YAML config roots keyed by calibration ID."""
    document = validate_calibration(calibration)
    root = _absolute(output_root) / str(document["calibration_id"])
    paths: dict[str, Path] = {}
    for strength in document["strengths"]:
        name = strength["name"]
        path = root / name / "CIFAR100C" / "OracleSoftRankRamen.yaml"
        payload = {
            "beta": 5.0, "calibration_id": document["calibration_id"],
            "calibration_strength": name, "capacity_scope": "per_class",
            "gamma": strength["gamma"], "include_current": True, "lr": 0.01,
            "max_capacity": 750, "oracle_context_source": "evaluator_domain_idx",
            "optimizer": "signsgd", "profile_replacement_margins": False, "topk": 5,
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable calibrated config already differs: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
        paths[name] = path
    return paths


def _base_runs(*, config_dir: Path, evidence_dir: str | Path, data_root: str | Path, device: str, provenance: str) -> list[ExperimentRun]:
    return build_experiment_matrix(
        datasets=(CANONICAL["dataset"],), streams=(CANONICAL["stream"],),
        methods=("OracleSoftRankRamen",), seeds=(CANONICAL["seed"],), config_dir=config_dir,
        evidence_dir=evidence_dir, data_root=data_root, device=device,
        max_eval_samples=CANONICAL["samples"], stream_block_size=CANONICAL["block_size"],
        artifact_provenance=provenance,
    )


def canonical_baseline(*, evidence_dir: str | Path, data_root: str | Path, device: str, provenance: str) -> ExperimentRun:
    runs = build_experiment_matrix(
        datasets=(CANONICAL["dataset"],), streams=(CANONICAL["stream"],), methods=("NoAdapt",),
        seeds=(CANONICAL["seed"],), config_dir=PROJECT_ROOT / "cfg", evidence_dir=evidence_dir,
        data_root=data_root, device=device, max_eval_samples=CANONICAL["samples"],
        stream_block_size=CANONICAL["block_size"], artifact_provenance=provenance,
    )
    return next(run for run in runs if run.method == "NoAdapt")


def _with_external_baseline(run: ExperimentRun, baseline: ExperimentRun) -> ExperimentRun:
    fields = ExperimentRun.__dataclass_fields__
    if "reference_config_path" not in fields:
        raise RuntimeError("ExperimentRun must support reference_config_path for external baseline reuse")
    return replace(run, reference_trace=baseline.run_dir / "trace.jsonl", reference_config_path=baseline.config_path)


def plan_profile(*, profile_evidence_dir: str | Path, baseline: ExperimentRun, data_root: str | Path, device: str, provenance: str) -> ExperimentRun:
    run = next(run for run in _base_runs(config_dir=MARGIN_PROFILE_CONFIG_ROOT, evidence_dir=profile_evidence_dir,
                                         data_root=data_root, device=device, provenance=provenance)
               if run.method == "OracleSoftRankRamen")
    return _with_external_baseline(run, baseline)


def plan_calibrated_runs(*, calibration: Mapping[str, object], config_root: str | Path, evidence_dir: str | Path,
                         baseline: ExperimentRun, data_root: str | Path, device: str, provenance: str) -> list[ExperimentRun]:
    document = validate_calibration(calibration)
    configs = materialize_configs(document, config_root)
    runs = []
    for name, _, _ in STRENGTHS:
        run = next(run for run in _base_runs(config_dir=configs[name].parents[1], evidence_dir=evidence_dir,
                                             data_root=data_root, device=device, provenance=provenance)
                   if run.method == "OracleSoftRankRamen")
        if run.config_data.get("gamma") != next(item["gamma"] for item in document["strengths"] if item["name"] == name):
            raise ValueError(f"materialized {name} config gamma changed")
        runs.append(_with_external_baseline(run, baseline))
    if len({run.run_id for run in runs}) != 3:
        raise ValueError("calibrated configuration roots must produce three unique runs")
    return runs


def _run_or_resume(run: ExperimentRun, *, python: str, resume: bool) -> str:
    if resume and run.run_dir.exists():
        validate_completed_run(run)
        return "skipped"
    run.run_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(run, python_executable=python), cwd=str(PROJECT_ROOT), check=True)
    validate_completed_run(run)
    return "executed"


def _require_baseline(baseline: ExperimentRun) -> dict[str, object]:
    evidence = validate_completed_run(baseline)
    summary = evidence["summary"]
    if summary.get("stream_fingerprint") != "aa6c94d923ff8024119c10111c8c685f4cd2e72fb70d47fc5978ba593a70020b":
        raise ValueError("external NoAdapt baseline is not the canonical stream")
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("profile", "run"):
        item = sub.add_parser(name)
        item.add_argument("--baseline-evidence-dir", required=True)
        item.add_argument("--data-root", required=True)
        item.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
        item.add_argument("--artifact-provenance", choices=SUPPORTED_ARTIFACT_PROVENANCE, default="fast")
        item.add_argument("--python", default=sys.executable)
        item.add_argument("--execute", action="store_true")
        item.add_argument("--resume", action="store_true")
    profile = sub.choices["profile"]
    profile.add_argument("--profile-evidence-dir", required=True)
    profile.add_argument("--calibration-path", required=True)
    profile.add_argument("--output-config-root", required=True)
    run = sub.choices["run"]
    run.add_argument("--calibration-path", required=True)
    run.add_argument("--output-config-root", required=True)
    run.add_argument("--evidence-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    baseline = canonical_baseline(evidence_dir=args.baseline_evidence_dir, data_root=args.data_root,
                                  device=args.device, provenance=args.artifact_provenance)
    baseline_evidence = _require_baseline(baseline)
    if args.stage == "profile":
        run = plan_profile(profile_evidence_dir=args.profile_evidence_dir, baseline=baseline, data_root=args.data_root,
                           device=args.device, provenance=args.artifact_provenance)
        payload: dict[str, Any] = {"stage": "profile", "baseline": baseline.to_dict(), "run": run.to_dict()}
        if args.execute:
            payload["outcome"] = _run_or_resume(run, python=args.python, resume=args.resume)
            evidence = validate_completed_run(run)
            if evidence["summary"].get("stream_fingerprint") != baseline_evidence["summary"].get("stream_fingerprint"):
                raise ValueError("profile and external baseline stream fingerprints differ")
            calibration = derive_calibration({**evidence, "run": run})
            payload["calibration_path"] = str(write_calibration(calibration, args.calibration_path))
            payload["configs"] = {name: str(path) for name, path in materialize_configs(calibration, args.output_config_root).items()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    calibration = load_calibration(args.calibration_path)
    verify_calibration_source(calibration)
    runs = plan_calibrated_runs(calibration=calibration, config_root=args.output_config_root,
                                evidence_dir=args.evidence_dir, baseline=baseline, data_root=args.data_root,
                                device=args.device, provenance=args.artifact_provenance)
    payload = {"stage": "run", "baseline": baseline.to_dict(), "calibration_id": calibration["calibration_id"],
               "runs": [run.to_dict() for run in runs]}
    if args.execute:
        outcomes = []
        for run in runs:
            outcomes.append({"run_id": run.run_id, "status": _run_or_resume(run, python=args.python, resume=args.resume)})
            evidence = validate_completed_run(run)
            if evidence["summary"].get("stream_fingerprint") != baseline_evidence["summary"].get("stream_fingerprint"):
                raise ValueError("calibrated run and external baseline stream fingerprints differ")
        payload["outcomes"] = outcomes
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
