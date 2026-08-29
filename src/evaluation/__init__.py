"""Portable experiment evidence and online evaluation utilities."""

from .evidence import (
    FAILURE_ANALYSIS_REQUIRED_FIELDS,
    SUMMARY_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    JsonlTraceWriter,
    atomic_write_json,
    build_run_manifest,
    compare_trace_negative_adaptation,
    verify_reference_trace_stream_fingerprint,
    write_run_manifest,
    write_summary,
)
from .online_metrics import (
    WindowAccuracy,
    average_accuracy,
    domain_shift_recovery_times,
    domain_accuracies,
    negative_adaptation_rate,
    post_shift_recovery_time,
    sliding_window_accuracy,
    worst_domain_accuracy,
)
from .routing_metrics import (
    RoutingDiagnostics,
    adjusted_rand_index,
    assignment_churn_rate,
    context_purity,
    normalized_mutual_information,
    number_of_discovered_contexts,
    routing_diagnostics,
)
from .failure_analysis_artifacts import (
    ReplayArtifactError,
    ReplaySidecarReader,
    ReplaySidecarWriter,
    parse_counterfactual_thresholds,
)


def __getattr__(name):
    """Lazily expose post-hoc analysis without circular runtime imports."""
    if name in {"AnalysisThresholds", "analyse_completed_runs", "load_completed_runs"}:
        from .experiment_analysis import AnalysisThresholds, analyse_completed_runs, load_completed_runs
        return {
            "AnalysisThresholds": AnalysisThresholds,
            "analyse_completed_runs": analyse_completed_runs,
            "load_completed_runs": load_completed_runs,
        }[name]
    raise AttributeError(name)

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "FAILURE_ANALYSIS_REQUIRED_FIELDS",
    "SUMMARY_SCHEMA_VERSION",
    "AnalysisThresholds",
    "JsonlTraceWriter",
    "ReplayArtifactError",
    "ReplaySidecarReader",
    "ReplaySidecarWriter",
    "RoutingDiagnostics",
    "WindowAccuracy",
    "atomic_write_json",
    "average_accuracy",
    "analyse_completed_runs",
    "adjusted_rand_index",
    "assignment_churn_rate",
    "build_run_manifest",
    "compare_trace_negative_adaptation",
    "verify_reference_trace_stream_fingerprint",
    "context_purity",
    "domain_accuracies",
    "domain_shift_recovery_times",
    "negative_adaptation_rate",
    "load_completed_runs",
    "normalized_mutual_information",
    "number_of_discovered_contexts",
    "post_shift_recovery_time",
    "parse_counterfactual_thresholds",
    "sliding_window_accuracy",
    "routing_diagnostics",
    "worst_domain_accuracy",
    "write_run_manifest",
    "write_summary",
    "aggregate_failure_analysis_study",
    "load_and_aggregate_failure_analysis_study",
]


def aggregate_failure_analysis_study(*args, **kwargs):
    from .failure_analysis_study import aggregate_failure_analysis_study as implementation
    return implementation(*args, **kwargs)


def load_and_aggregate_failure_analysis_study(*args, **kwargs):
    from .failure_analysis_study import load_and_aggregate_failure_analysis_study as implementation
    return implementation(*args, **kwargs)
