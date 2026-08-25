"""Dependency-light runtime checks for reproducible NB-Ramen experiments."""

__all__ = (
    "ExperimentRun", "build_command", "build_experiment_matrix", "build_open_set_evidence_matrix",
    "DeviceMemoryTracker", "collect_hardware_evidence",
    "ProvenanceError", "generate_cifar100c_provenance", "verify_cifar100c_provenance",
    "generate_domainnet_provenance", "verify_domainnet_provenance", "verify_clip_checkpoint",
    "archive_acquisition_record",
)


def __getattr__(name):
    """Expose planner helpers without pre-importing its CLI module."""
    if name in __all__:
        if name in {"ProvenanceError", "generate_cifar100c_provenance", "verify_cifar100c_provenance",
                    "generate_domainnet_provenance", "verify_domainnet_provenance", "verify_clip_checkpoint",
                    "archive_acquisition_record"}:
            from . import artifact_provenance
            return getattr(artifact_provenance, name)
        if name in {"DeviceMemoryTracker", "collect_hardware_evidence"}:
            from . import device_evidence
            return getattr(device_evidence, name)
        from . import experiment_matrix
        return getattr(experiment_matrix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
