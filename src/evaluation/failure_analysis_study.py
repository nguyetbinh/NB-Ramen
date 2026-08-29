"""Fail-closed aggregation of artifact-bound failure-analysis reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .failure_mode_analysis import (
    CANONICAL_CONFLICT_METRIC,
    PREREGISTERED_COUNTERFACTUAL_THRESHOLDS,
    _verified_trace_run,
    analyze_verified_run_dirs,
    consensus_ramen_decision,
)


STUDY_SCHEMA_VERSION = 1
_CELL_FIELDS = ("device", "dataset", "stream", "seed", "method", "analysis_role")
_IDENTITY_ALIASES = {"stream": ("stream", "stream_mode")}
_STRUCTURED_STREAMS = frozenset({"block", "gradual", "recurring", "imbalanced", "novel_domain", "class_domain_correlated", "bursty"})
_REPORT_FAMILIES = ("F0", "F1", "F2", "F3", "F4", "F5", "entropy")
_F3_METRICS = ("consensus_mean", "consensus_p10", "consensus_p50", "fraction_low_consensus_coordinates",
               "active_support_classes", "pairwise_sign_agreement_mean", "pairwise_cosine_mean")
_F3_OUTCOMES = ("safe", "beneficial", "harmful", "unresolved")
_CELL_SPECIFIC_EVALUATOR_ARGUMENTS = frozenset({"run_id", "save_to", "reference_trace", "evidence_dir", "seed", "stream_seed", "stream_mode"})


def _state(value: object) -> str:
    return value if value in {"computed", "insufficient", "unavailable"} else "insufficient"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _thresholds(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    parsed = tuple(_number(item) for item in value)
    return None if any(item is None for item in parsed) else tuple(item for item in parsed if item is not None)


def _identity_value(identity: Mapping[str, Any], field: str) -> object:
    for key in _IDENTITY_ALIASES.get(field, (field,)):
        if key in identity:
            return identity[key]
    return None


def _relative_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"cell lacks {label} path")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} path must be relative to the trusted artifact root")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the trusted artifact root") from exc
    return resolved


def _read_json(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("JSON document must be an object")
    return data


def _report_state(report: Mapping[str, Any], family: str) -> str:
    if family == "F0":
        return _state(report.get("paired_outcomes", {}).get("status") if isinstance(report.get("paired_outcomes"), Mapping) else None)
    if family in {"F1", "F2"}:
        return _state(report.get("oracle_gaps", {}).get("status") if isinstance(report.get("oracle_gaps"), Mapping) else None)
    if family == "F3":
        return _state(report.get("gradient_conflict", {}).get("status") if isinstance(report.get("gradient_conflict"), Mapping) else None)
    if family == "F4":
        return "computed" if _f4_evidence(report) is not None else "insufficient"
    if family == "F5":
        temporal = report.get("temporal_schedule")
        paired = temporal.get("paired_schedule_comparison") if isinstance(temporal, Mapping) else None
        return _state(paired.get("status") if isinstance(paired, Mapping) else None)
    return _state(report.get("entropy_admission", {}).get("status") if isinstance(report.get("entropy_admission"), Mapping) else None)


def _f3_direction(report: Mapping[str, Any]) -> int | None:
    conflict = report.get("gradient_conflict")
    value = conflict.get("harmful_minus_beneficial") if isinstance(conflict, Mapping) and conflict.get("status") == "computed" else None
    value = _number(value)
    return None if value is None or value == 0 else (1 if value > 0 else -1)


def _f4_evidence(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    counterfactual = report.get("counterfactual")
    variants = counterfactual.get("variants") if isinstance(counterfactual, Mapping) and counterfactual.get("status") == "computed" else None
    if not isinstance(variants, Mapping):
        return None
    return next((item for item in variants.values() if isinstance(item, Mapping) and item.get("status") == "computed"
                 and item.get("harmful_recovery_status") == "computed"), None)


def _validate_report_shape(report: Mapping[str, Any], errors: list[str]) -> None:
    if report.get("status") not in {"computed", "insufficient", "unavailable"}:
        errors.append("report status is malformed")
    conflict = report.get("gradient_conflict")
    if not isinstance(conflict, Mapping) or conflict.get("metric") != CANONICAL_CONFLICT_METRIC:
        errors.append("report F3 does not declare the canonical conflict metric")
    distributions = report.get("gradient_conflict_distributions")
    outcomes = distributions.get("outcomes") if isinstance(distributions, Mapping) else None
    if not isinstance(outcomes, Mapping) or any(not isinstance(outcomes.get(outcome), Mapping) or
                                                any(metric not in outcomes[outcome] for metric in _F3_METRICS)
                                                for outcome in _F3_OUTCOMES):
        errors.append("report F3 distributions are incomplete or noncanonical")
    counterfactual = report.get("counterfactual")
    variants = counterfactual.get("variants") if isinstance(counterfactual, Mapping) else None
    expected = {f"{threshold:.2f}" for threshold in PREREGISTERED_COUNTERFACTUAL_THRESHOLDS}
    if not isinstance(variants, Mapping) or set(variants) != expected or any(not isinstance(variants[key], Mapping) for key in expected):
        errors.append("report F4 variants do not exactly match preregistered thresholds")


def _run_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    args = manifest.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("run manifest args are malformed")
    role = args.get("analysis_role")
    if role not in {"analysis", "final"}:
        raise ValueError("run manifest args.analysis_role must be 'analysis' or 'final'")
    return {"device": manifest.get("device"), "dataset": args.get("dataset"), "stream": args.get("stream_mode"),
            "seed": args.get("seed"), "method": args.get("tta_algo"), "analysis_role": role}


def _canonical_manifest_value(value: object, *, label: str) -> str:
    """Make a JSON manifest value safe and exact to compare across cells."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"verified manifest {label} is not canonical JSON") from exc


def _verified_study_identity(manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, str]:
    """Return the scientific identity shared by all study cells.

    Streams and seeds are deliberate replication axes, so they are excluded.
    Everything here is recovered from a verified run rather than from the
    study's declarative ``run_identity``.
    """
    args, artifacts, config = manifest.get("args"), manifest.get("artifacts"), manifest.get("config")
    if not isinstance(args, Mapping):
        raise ValueError("verified manifest args are malformed")
    if not isinstance(artifacts, Mapping) or artifacts.get("status") != "verified":
        raise ValueError("verified manifest artifact verification is missing")
    model, dataset = artifacts.get("model"), artifacts.get("dataset")
    if not isinstance(model, Mapping) or model.get("status") != "verified":
        raise ValueError("verified manifest model artifact verification is missing")
    if not isinstance(dataset, Mapping) or dataset.get("status") != "verified":
        raise ValueError("verified manifest dataset artifact verification is missing")
    source, model_digest, dataset_digest = (identity.get("source_fingerprint"), model.get("actual_sha256"),
                                              dataset.get("root_digest"))
    if any(not isinstance(value, str) or not value for value in (source, model_digest, dataset_digest)):
        raise ValueError("verified manifest source/model/dataset identity is missing")
    method = args.get("tta_algo")
    if not isinstance(method, str) or not method or method == "NoAdapt":
        raise ValueError("verified adapted method identity is missing")
    if not isinstance(config, Mapping):
        raise ValueError("verified adapted method config is missing or malformed")
    evaluator = {key: value for key, value in args.items()
                 if key not in _CELL_SPECIFIC_EVALUATOR_ARGUMENTS and not key.startswith("stream_") and key != "tta_algo"}
    return {
        "source fingerprint": source,
        "model artifact digest": model_digest,
        "dataset artifact digest": dataset_digest,
        "method implementation": method,
        "adapted method config": _canonical_manifest_value(config, label="config"),
        "non-stream/non-seed evaluator settings": _canonical_manifest_value(evaluator, label="args"),
    }


def _cell(manifest_cell: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    declared = manifest_cell.get("run_identity", manifest_cell.get("identity"))
    if not isinstance(declared, Mapping):
        return {}, ["cell lacks a run_identity object"]
    values = {field: _identity_value(declared, field) for field in _CELL_FIELDS}
    if any(value is None or value == "" for value in values.values()):
        errors.append("run_identity misses " + ", ".join(field for field, value in values.items() if value is None or value == ""))
    for field in ("device", "dataset", "stream", "method", "analysis_role"):
        if values[field] is not None and not isinstance(values[field], str):
            errors.append(f"run_identity {field} must be a nonempty string")
    # Seed aliases such as 1 and "1" make replication counts ambiguous.
    if isinstance(values["seed"], bool) or not isinstance(values["seed"], int):
        errors.append("run_identity seed must be an integer (string seed aliases are rejected)")
    thresholds = _thresholds(declared.get("failure_counterfactual_thresholds", declared.get("counterfactual_thresholds")))
    if thresholds != PREREGISTERED_COUNTERFACTUAL_THRESHOLDS:
        errors.append("run_identity counterfactual thresholds are not preregistered")
    try:
        report_path = _relative_path(root, manifest_cell.get("report"), label="report")
        baseline_path = _relative_path(root, manifest_cell.get("baseline_run_dir"), label="baseline_run_dir")
        adapted_path = _relative_path(root, manifest_cell.get("adapted_run_dir"), label="adapted_run_dir")
        report = _read_json(report_path)
        base_root, base_manifest, _, base_identity = _verified_trace_run(str(baseline_path))
        adapted_root, adapted_manifest, _, adapted_identity = _verified_trace_run(str(adapted_path))
        recomputed_report = analyze_verified_run_dirs(str(baseline_path), str(adapted_path))
        actual = _run_identity(adapted_manifest)
        study_identity = _verified_study_identity(adapted_manifest, adapted_identity)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, errors + [str(exc)]
    for field in _CELL_FIELDS:
        if values[field] != actual[field]:
            errors.append(f"declared identity {field} disagrees with adapted run manifest")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("verified") is not True:
        errors.append("report is not provenance-verified")
    else:
        for label, actual_identity in (("baseline", base_identity), ("adapted", adapted_identity)):
            bound = provenance.get(label)
            if not isinstance(bound, Mapping) or any(bound.get(key) != actual_identity[key]
                                                    for key in ("run_id", "manifest_sha256", "stream_fingerprint", "source_fingerprint")):
                errors.append(f"report provenance {label} run ID or hashes do not bind the declared run")
    if report != recomputed_report:
        errors.append("report content does not exactly match recomputation from the verified runs")
    _validate_report_shape(report, errors)
    return {"identity": values, "study_identity": study_identity, "report": report, "report_path": str(manifest_cell["report"]),
            "baseline_path": str(manifest_cell["baseline_run_dir"]), "adapted_path": str(manifest_cell["adapted_run_dir"])}, errors


def aggregate_failure_analysis_study(manifest: Mapping[str, Any], *, manifest_root: Path | str = ".") -> dict[str, Any]:
    """Aggregate an artifact-bound study without relaxing canonical GO criteria."""
    result: dict[str, Any] = {"schema_version": STUDY_SCHEMA_VERSION, "status": "INSUFFICIENT"}
    if not isinstance(manifest, Mapping):
        return {**result, "status": "invalid", "errors": ["study manifest must be an object"]}
    if _thresholds(manifest.get("preregistered_counterfactual_thresholds")) != PREREGISTERED_COUNTERFACTUAL_THRESHOLDS:
        return {**result, "status": "invalid", "errors": ["manifest must declare the canonical preregistered thresholds"]}
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        return {**result, "status": "invalid", "errors": ["manifest must declare a nonempty cells list"]}
    root = Path(manifest_root).resolve()
    cells, errors = [], []
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping):
            errors.append(f"cell {index} is not an object")
            continue
        cell, cell_errors = _cell(raw, root)
        errors.extend(f"cell {index}: {error}" for error in cell_errors)
        if cell:
            cells.append(cell)
    identities = [tuple(cell["identity"][field] for field in _CELL_FIELDS) for cell in cells]
    if len(set(identities)) != len(identities): errors.append("duplicate device/dataset/stream/seed/method/analysis-role cell")
    if len({cell["report_path"] for cell in cells}) != len(cells): errors.append("report reuse across cells is forbidden")
    if len({cell["adapted_path"] for cell in cells}) != len(cells): errors.append("adapted run reuse across cells is forbidden")
    if len({cell["baseline_path"] for cell in cells}) != len(cells): errors.append("baseline run reuse across cells is forbidden")
    devices = {cell["identity"]["device"] for cell in cells}
    if len(devices) != 1: errors.append("study must not merge devices")
    if {cell["identity"]["analysis_role"] for cell in cells} != {"analysis"}: errors.append("ConsensusRamen discovery evidence must use analysis-role cells only")
    for field in next(iter(cells))["study_identity"] if cells else ():
        if len({cell["study_identity"][field] for cell in cells}) != 1:
            errors.append(f"study cells disagree on verified {field}")
    if errors: return {**result, "status": "invalid", "errors": errors, "cell_count": len(cells)}
    matrix = {family: {"computed": 0, "insufficient": 0, "unavailable": 0} for family in _REPORT_FAMILIES}
    evidence: list[dict[str, Any]] = []
    for cell in cells:
        report, identity = cell["report"], cell["identity"]
        for family in _REPORT_FAMILIES: matrix[family][_report_state(report, family)] += 1
        direction, f4 = _f3_direction(report), _f4_evidence(report)
        if identity["stream"] in _STRUCTURED_STREAMS and direction is not None and f4 is not None:
            evidence.append({"structured_stream": True, "stream": identity["stream"], "seed": identity["seed"],
                             "harmful_minus_beneficial": direction, "f4": {"status": "computed", "harmful_recovery_status": "computed"}})
    canonical = consensus_ramen_decision(evidence)
    # The canonical helper accepts string seeds and only needs two global seeds.  Studies require
    # two distinct *integer* seeds in every replicated structured stream.
    required_streams = {cell["identity"]["stream"] for cell in cells if cell["identity"]["stream"] in _STRUCTURED_STREAMS}
    complete_streams = {stream for stream in required_streams if len({row["seed"] for row in evidence if row["stream"] == stream}) >= 2}
    replication_missing = len(complete_streams) < 2
    if replication_missing:
        canonical = {**canonical, "status": "INSUFFICIENT", "missing_conditions": [*canonical["missing_conditions"], "two_integer_seeds_within_each_of_two_structured_streams"]}
    result.update(status=canonical["status"], device=next(iter(devices)), cell_count=len(cells),
                  evidence_counts={"declared_cells": len(cells), "structured_cells": sum(c["identity"]["stream"] in _STRUCTURED_STREAMS for c in cells), "eligible_consensus_cells": len(evidence)},
                  status_matrix=matrix, consensus_ramen_decision=canonical,
                  cells=[{"identity": c["identity"], "report": c["report_path"], "baseline_run_dir": c["baseline_path"], "adapted_run_dir": c["adapted_path"], "f3_direction": _f3_direction(c["report"]), "f4_harmful_recovery": _f4_evidence(c["report"]) is not None} for c in cells])
    return result


def _default_artifact_root() -> Path:
    """Return the repository root for the source-tree CLI entry point."""
    return Path(__file__).resolve().parents[2]


def load_and_aggregate_failure_analysis_study(
    manifest_path: str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    # Checked-in study manifests declare artifact paths relative to the repository,
    # allowing one trusted root to bind both plans/reports and evidence run folders.
    # An explicit artifact root remains available for externally located studies.
    root = _default_artifact_root() if artifact_root is None else Path(artifact_root).resolve()
    return aggregate_failure_analysis_study(_read_json(path), manifest_root=root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate a declared verified failure-analysis study")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-root", help="trusted root for relative report/run paths (defaults to repository root)")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try: report = load_and_aggregate_failure_analysis_study(args.manifest, artifact_root=args.artifact_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc: report = {"schema_version": STUDY_SCHEMA_VERSION, "status": "invalid", "errors": [str(exc)]}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output: Path(args.output).write_text(encoded, encoding="utf-8")
    else: print(encoded, end="")
    return 0 if report["status"] != "invalid" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
