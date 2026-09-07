"""Post-hoc descriptive analysis for the canonical open-set Consensus matrix.

This is deliberately separate from ``experiment_analysis``.  It validates the
oracle-gradient evidence and reports paired outcomes, but does not apply the
legacy LatentRamen router gate or certify ConsensusRamen as deployable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Iterable, Mapping, Sequence

try:
    from ..runtime.experiment_matrix import (
        ExperimentRun, IncompleteRunError, OPEN_SET_METHODS, OPEN_SET_OOD_RATIOS,
        OPEN_SET_CONSENSUS_METHODS, OPEN_SET_DIRECTIONAL_ORACLE_METHODS,
        OPEN_SET_EVALUATOR_OOD_METHODS, OPEN_SET_SEEDS, OPEN_SET_STREAMS,
        SUPPORTED_ARTIFACT_PROVENANCE, build_open_set_evidence_matrix, validate_completed_run,
    )
except ImportError:  # pragma: no cover - direct-file support
    source_root = str(Path(__file__).resolve().parents[1])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from runtime.experiment_matrix import (
        ExperimentRun, IncompleteRunError, OPEN_SET_METHODS, OPEN_SET_OOD_RATIOS,
        OPEN_SET_CONSENSUS_METHODS, OPEN_SET_DIRECTIONAL_ORACLE_METHODS,
        OPEN_SET_EVALUATOR_OOD_METHODS, OPEN_SET_SEEDS, OPEN_SET_STREAMS,
        SUPPORTED_ARTIFACT_PROVENANCE, build_open_set_evidence_matrix, validate_completed_run,
    )


REPORT_SCHEMA_VERSION = 3


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def load_open_set_completed_runs(runs: Iterable[ExperimentRun]) -> list[tuple[ExperimentRun, dict[str, object]]]:
    """Strictly validate every planned run before post-hoc inspection."""
    return [(run, validate_completed_run(run)) for run in runs]


def analyse_open_set_completed_runs(
    completed: Iterable[tuple[ExperimentRun, Mapping[str, object]]],
) -> dict[str, object]:
    """Return paired descriptive outcomes without a Consensus go/no-go claim."""
    cells: dict[tuple[float, str, int], dict[str, tuple[ExperimentRun, Mapping[str, object]]]] = defaultdict(dict)
    for run, evidence in completed:
        if not run.open_set or run.dataset != "CIFAR100C" or run.ood_ratio is None:
            raise ValueError(f"not an open-set CIFAR100C run: {run.run_id}")
        if run.method not in OPEN_SET_METHODS:
            raise ValueError(f"unsupported open-set matrix method: {run.method}")
        summary = evidence.get("summary")
        manifest = evidence.get("manifest")
        if not isinstance(summary, Mapping) or not isinstance(manifest, Mapping):
            raise ValueError(f"validated run lacks summary or manifest: {run.run_id}")
        args = manifest.get("args")
        if not isinstance(args, Mapping):
            raise ValueError(f"manifest args missing for {run.run_id}")
        oracle_context = args.get("oracle_ood_contexts")
        if run.method in OPEN_SET_EVALUATOR_OOD_METHODS:
            if oracle_context is not True:
                raise ValueError(f"named oracle lacks evaluator OOD context: {run.run_id}")
        elif oracle_context is not False:
            raise ValueError(f"non-oracle received evaluator OOD context: {run.run_id}")
        key = (float(run.ood_ratio), run.stream_mode, run.seed)
        if run.method in cells[key]:
            raise ValueError(f"duplicate matrix evidence for {key!r}/{run.method}")
        cells[key][run.method] = (run, evidence)

    comparisons = []
    complete = bool(cells)
    for key in sorted(cells):
        methods = cells[key]
        missing = [method for method in OPEN_SET_METHODS if method not in methods]
        if missing:
            complete = False
            comparisons.append({"ood_ratio": key[0], "stream_mode": key[1], "seed": key[2], "status": "incomplete", "missing_methods": missing})
            continue
        fingerprints = {str(evidence["summary"].get("stream_fingerprint")) for _, evidence in methods.values()}
        if len(fingerprints) != 1:
            raise ValueError(f"paired methods have different stream fingerprints: {key!r}")
        metrics = {method: _open_set_metrics(evidence["summary"], method) for method, (_, evidence) in methods.items()}
        comparison = {
            "ood_ratio": key[0], "stream_mode": key[1], "seed": key[2], "status": "complete",
            "stream_fingerprint": next(iter(fingerprints)), "methods": metrics,
            "consensus_vs_ramen_id_accuracy_gain": _difference(metrics["ConsensusRamen"]["id_accuracy"], metrics["Ramen"]["id_accuracy"]),
            "oracle_id_vs_ramen_id_accuracy_gain": _difference(metrics["OracleIDGradientRamen"]["id_accuracy"], metrics["Ramen"]["id_accuracy"]),
            "oracle_drop_vs_ramen_id_accuracy_gain": _difference(metrics["OracleDropOODRamen"]["id_accuracy"], metrics["Ramen"]["id_accuracy"]),
        }
        # This is deliberately an end-to-end paired overhead proxy, not an
        # attribution of all elapsed time to the consensus operation.  The
        # trace contract does not isolate that operation without perturbing
        # the measured path.
        comparison["consensus_vs_ramen_cost_overhead"] = {
            method: _paired_cost_overhead(metrics[method], metrics["Ramen"])
            for method in sorted(OPEN_SET_CONSENSUS_METHODS)
        }
        comparisons.append(comparison)

    expected_cells = {(ratio, stream, seed) for ratio in OPEN_SET_OOD_RATIOS for stream in OPEN_SET_STREAMS for seed in OPEN_SET_SEEDS}
    observed_cells = set(cells)
    canonical_coverage = observed_cells == expected_cells and complete
    runs = [run for methods in cells.values() for run, _ in methods.values()]
    canonical_runtime = bool(runs) and all(
        run.device == "cuda" and run.max_eval_samples is None and run.artifact_provenance in SUPPORTED_ARTIFACT_PROVENANCE
        for run in runs
    )
    classification = "canonical_cuda_expected" if canonical_coverage and canonical_runtime else "noncanonical_pilot"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "analysis_contract": "open_set_consensus_descriptive_v3",
        "classification": classification,
        "consensus_certification": "not_applicable",
        "coverage": {"expected_cell_count": len(expected_cells), "observed_cell_count": len(observed_cells), "complete": canonical_coverage},
        "comparisons": comparisons,
    }


def _open_set_metrics(summary: Mapping[str, object], method: str) -> dict[str, object]:
    block = summary.get("open_set")
    if not isinstance(block, Mapping):
        raise ValueError(f"open-set summary missing for {method}")
    result = {}
    worst_domain_id_accuracy = _number(block.get("worst_domain_id_accuracy"))
    if worst_domain_id_accuracy is None:
        raise ValueError(
            f"open-set summary worst_domain_id_accuracy is missing or non-finite for {method}"
        )
    result["worst_domain_id_accuracy"] = worst_domain_id_accuracy
    pre = block.get("pre_adaptation_detection")
    post = block.get("post_adaptation_detection")
    if not isinstance(pre, Mapping) or not isinstance(post, Mapping):
        raise ValueError(f"open-set summary lacks explicit pre/post adaptation detection for {method}")
    result["pre_adaptation_detection"] = _detection_metrics(pre, method, "pre")
    result["post_adaptation_detection"] = _detection_metrics(post, method, "post")
    # The flat compatibility projection is consistently post-adaptation. The
    # previous runtime combined post ID accuracy with pre-adaptation OOD scores.
    result.update(result["post_adaptation_detection"])
    result["pre_adaptation_id_accuracy"] = result["pre_adaptation_detection"]["id_accuracy"]
    result["post_adaptation_id_accuracy"] = result["post_adaptation_detection"]["id_accuracy"]
    for name in ("auroc", "fpr95", "h_score", "ood_recall_at_fpr95"):
        result[f"pre_adaptation_{name}"] = result["pre_adaptation_detection"][name]
    result["post_adaptation_auroc"] = result["post_adaptation_detection"]["auroc"]
    result["post_adaptation_fpr95"] = result["post_adaptation_detection"]["fpr95"]
    result["post_adaptation_h_score"] = result["post_adaptation_detection"]["h_score"]
    result["post_adaptation_ood_recall_at_fpr95"] = (
        result["post_adaptation_detection"]["ood_recall_at_fpr95"]
    )
    result["stability"] = _stability_metrics(summary, method)
    result["cost"] = _cost_metrics(summary, method)
    if method in OPEN_SET_DIRECTIONAL_ORACLE_METHODS:
        diagnostic = summary.get("oracle_gradient_diagnostics")
        if not isinstance(diagnostic, Mapping):
            raise ValueError(f"oracle diagnostics missing for {method}")
        result["oracle_gradient_direction_corruption"] = _number(diagnostic.get("gradient_direction_corruption_mean"))
        result["oracle_sign_disagreement"] = _number(diagnostic.get("sign_disagreement_mean"))
        for field in (
            "ramen_gdc_mean", "consensus_gdc_mean", "ramen_sdr_mean",
            "consensus_sdr_mean", "gdc_reduction_mean", "sdr_reduction_mean",
            "consensus_wrong_sign_removal_rate",
            "consensus_correct_sign_removal_rate",
        ):
            if field not in diagnostic:
                raise ValueError(f"oracle diagnostic {field} is missing for {method}")
            value = _number(diagnostic.get(field))
            if diagnostic.get(field) is not None and value is None:
                raise ValueError(f"oracle diagnostic {field} is non-finite for {method}")
            result[field] = value
        for field in (
            "consensus_hard_mask_applied_sample_count",
            "consensus_wrong_sign_coordinate_count",
            "consensus_wrong_sign_removed_count",
            "consensus_correct_sign_coordinate_count",
            "consensus_correct_sign_removed_count",
        ):
            value = _nonnegative_integer(diagnostic.get(field))
            if value is None:
                raise ValueError(f"oracle diagnostic {field} is missing or invalid for {method}")
            result[field] = value
    if method in OPEN_SET_CONSENSUS_METHODS:
        diagnostic = summary.get("consensus_diagnostics")
        if not isinstance(diagnostic, Mapping):
            raise ValueError(f"consensus diagnostics missing for {method}")
        result["consensus_applied_sample_fraction"] = _number(diagnostic.get("consensus_applied_sample_fraction"))
        result["consensus_retained_coordinate_rate"] = _number(diagnostic.get("mask_rate"))
    return result


def _detection_metrics(block: Mapping[str, object], method: str, phase: str) -> dict[str, object]:
    detection_status = block.get("status")
    phase_id_accuracy = _number(block.get("id_accuracy"))
    if phase_id_accuracy is None:
        raise ValueError(
            f"{phase}-adaptation summary id_accuracy is missing or non-finite for {method}"
        )
    result = {"id_accuracy": phase_id_accuracy}
    if detection_status == "computed":
        for name in ("auroc", "fpr95", "h_score", "ood_recall_at_fpr95"):
            value = _number(block.get(name))
            if value is None:
                raise ValueError(f"computed {phase}-adaptation summary {name} is missing or non-finite for {method}")
            result[name] = value
    elif detection_status == "unavailable":
        if not isinstance(block.get("reason"), str) or not block["reason"]:
            raise ValueError(f"unavailable {phase}-adaptation summary has no reason for {method}")
        if any(
            block.get(name) is not None
            for name in ("auroc", "fpr95", "h_score", "ood_recall_at_fpr95")
        ):
            raise ValueError(f"unavailable {phase}-adaptation metrics must be null for {method}")
        # OOD=0 is a preregistered canonical cell. Detection statistics are
        # correctly unavailable there; keep that explicit state rather than
        # making the complete canonical matrix impossible to analyse.
        result.update({
            "auroc": None, "fpr95": None, "h_score": None,
            "ood_recall_at_fpr95": None,
        })
    else:
        raise ValueError(f"{phase}-adaptation detection status is invalid for {method}")
    return result


def _stability_metrics(summary: Mapping[str, object], method: str) -> dict[str, object]:
    """Expose open-set stability evidence without treating OOD as errors."""
    negative = _required_mapping(summary, "id_only_negative_adaptation_rate", method)
    negative_status = negative.get("status")
    if negative_status == "computed":
        value = _probability(negative.get("value"))
        retained_id_samples = _nonnegative_integer(negative.get("retained_id_samples"))
        total_windows = _nonnegative_integer(negative.get("total_windows"))
        negative_windows = _nonnegative_integer(negative.get("negative_windows"))
        if value is None:
            raise ValueError(f"negative-adaptation rate is invalid for {method}")
        if (
            retained_id_samples is None or total_windows is None or total_windows == 0
            or negative_windows is None or negative_windows > total_windows
        ):
            raise ValueError(f"ID-only negative-adaptation counts are invalid for {method}")
        negative_output: dict[str, object] = {
            "status": "computed", "rate": value,
            "retained_id_samples": retained_id_samples,
            "total_windows": total_windows,
            "negative_windows": negative_windows,
        }
    elif negative_status == "reference_required":
        if not isinstance(negative.get("reason"), str) or not negative["reason"]:
            raise ValueError(f"negative-adaptation unavailable reason is invalid for {method}")
        negative_output = {"status": "reference_required", "rate": None}
    elif negative_status == "insufficient_id_samples":
        retained_id_samples = _nonnegative_integer(negative.get("retained_id_samples"))
        total_windows = _nonnegative_integer(negative.get("total_windows"))
        negative_windows = _nonnegative_integer(negative.get("negative_windows"))
        if (
            retained_id_samples is None or total_windows != 0 or negative_windows != 0
            or negative.get("value") is not None
        ):
            raise ValueError(f"ID-only negative-adaptation counts are invalid for {method}")
        negative_output = {
            "status": "insufficient_id_samples", "rate": None,
            "retained_id_samples": retained_id_samples,
            "total_windows": total_windows,
            "negative_windows": negative_windows,
        }
    else:
        raise ValueError(f"negative-adaptation status is invalid for {method}")

    recovery = _required_mapping(summary, "id_only_post_shift_recovery_time", method)
    recovery_status = recovery.get("status")
    if recovery_status == "computed":
        shifts = recovery.get("shifts")
        if not isinstance(shifts, list) or any(not isinstance(item, Mapping) for item in shifts):
            raise ValueError(f"ID-only post-shift recovery evidence is invalid for {method}")
        recovery_output: dict[str, object] = {"status": "computed", "shifts": list(shifts)}
    elif recovery_status == "not_applicable":
        if not isinstance(recovery.get("reason"), str) or not recovery["reason"]:
            raise ValueError(f"ID-only post-shift recovery reason is invalid for {method}")
        recovery_output = {"status": "not_applicable", "shifts": []}
    else:
        raise ValueError(f"ID-only post-shift recovery status is invalid for {method}")
    return {
        "negative_adaptation": negative_output,
        "post_shift_recovery": recovery_output,
    }


def _cost_metrics(summary: Mapping[str, object], method: str) -> dict[str, object]:
    """Read summary cost blocks strictly; never synthesize a missing measurement."""
    latency = _required_mapping(summary, "forward_latency", method)
    if latency.get("status") != "computed":
        raise ValueError(f"synchronized forward latency is unavailable for {method}")
    latency_values = {
        "total_ms": _nonnegative(latency.get("total_ms")),
        "mean_per_sample_ms": _nonnegative(latency.get("mean_per_sample_ms")),
        "median_per_sample_ms": _nonnegative(latency.get("median_per_sample_ms")),
    }
    if any(value is None for value in latency_values.values()):
        raise ValueError(f"synchronized forward latency is invalid for {method}")

    throughput = _required_mapping(summary, "throughput", method)
    if throughput.get("status") != "computed":
        raise ValueError(f"throughput is unavailable for {method}")
    throughput_value = throughput.get("samples_per_second")
    if throughput_value is not None and _nonnegative(throughput_value) is None:
        raise ValueError(f"throughput is invalid for {method}")

    memory = _required_mapping(summary, "method_memory", method)
    memory_status = memory.get("status")
    if memory_status == "computed":
        maximum = _nonnegative_integer(memory.get("max_retained_bytes"))
        final = _nonnegative_integer(memory.get("final_retained_bytes"))
        if maximum is None or final is None:
            raise ValueError(f"retained-memory evidence is invalid for {method}")
    elif memory_status == "unavailable":
        if memory.get("max_retained_bytes") is not None or memory.get("final_retained_bytes") is not None:
            raise ValueError(f"unavailable retained-memory evidence is invalid for {method}")
        maximum = final = None
    else:
        raise ValueError(f"retained-memory status is invalid for {method}")
    return {
        "synchronized_forward_latency": {"status": "computed", **latency_values},
        "retained_memory": {
            "status": memory_status,
            "max_retained_bytes": maximum,
            "final_retained_bytes": final,
        },
        "throughput": {"status": "computed", "samples_per_second": _number(throughput_value)},
    }


def _paired_cost_overhead(consensus: Mapping[str, object], ramen: Mapping[str, object]) -> dict[str, object]:
    """Report only paired end-to-end deltas supported by the summary schema."""
    consensus_cost = consensus["cost"]
    ramen_cost = ramen["cost"]
    if not isinstance(consensus_cost, Mapping) or not isinstance(ramen_cost, Mapping):  # defensive API boundary
        raise ValueError("paired cost evidence is malformed")
    consensus_latency = _nested_number(consensus_cost, "synchronized_forward_latency", "total_ms")
    ramen_latency = _nested_number(ramen_cost, "synchronized_forward_latency", "total_ms")
    consensus_throughput = _nested_number(consensus_cost, "throughput", "samples_per_second")
    ramen_throughput = _nested_number(ramen_cost, "throughput", "samples_per_second")
    consensus_memory = _nested_number(consensus_cost, "retained_memory", "max_retained_bytes")
    ramen_memory = _nested_number(ramen_cost, "retained_memory", "max_retained_bytes")
    return {
        "status": "paired_total_path_proxy",
        "definition": "paired difference against Ramen in synchronized end-to-end forward cost; not isolated consensus computation time",
        "forward_latency_total_ms_difference": _difference(consensus_latency, ramen_latency),
        "forward_latency_total_ms_ratio": _ratio(consensus_latency, ramen_latency),
        "throughput_samples_per_second_difference": _difference(consensus_throughput, ramen_throughput),
        "throughput_samples_per_second_ratio": _ratio(consensus_throughput, ramen_throughput),
        "max_retained_memory_bytes_difference": _difference(consensus_memory, ramen_memory),
        "max_retained_memory_bytes_ratio": _ratio(consensus_memory, ramen_memory),
    }


def _required_mapping(summary: Mapping[str, object], name: str, method: str) -> Mapping[str, object]:
    value = summary.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"summary {name} missing for {method}")
    return value


def _nonnegative(value: object) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0.0 else None


def _probability(value: object) -> float | None:
    number = _nonnegative(value)
    return number if number is not None and number <= 1.0 else None


def _nonnegative_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _nested_number(value: Mapping[str, object], block: str, field: str) -> float | None:
    nested = value.get(block)
    return _number(nested.get(field)) if isinstance(nested, Mapping) else None


def _difference(left: object, right: object) -> float | None:
    return left - right if isinstance(left, float) and isinstance(right, float) else None


def _ratio(left: float | None, right: float | None) -> float | None:
    return left / right if left is not None and right is not None and right > 0.0 else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--config-dir", default=Path(__file__).resolve().parents[2] / "cfg")
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--artifact-provenance", choices=SUPPORTED_ARTIFACT_PROVENANCE, default="fast")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        runs = build_open_set_evidence_matrix(evidence_dir=args.evidence_dir, config_dir=args.config_dir,
                                               data_root=args.data_root, artifact_provenance=args.artifact_provenance)
        report = analyse_open_set_completed_runs(load_open_set_completed_runs(runs))
    except (OSError, ValueError, TypeError, IncompleteRunError) as exc:
        print(json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "status": "invalid_evidence", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["classification"] == "canonical_cuda_expected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
