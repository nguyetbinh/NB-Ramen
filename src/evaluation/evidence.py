"""Versioned, dependency-free experiment evidence files.

Trace rows deliberately retain evaluation-only ground-truth fields.  Methods
must not use these fields for routing or adaptation; they make later analysis
of heterogeneous streams possible.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import math
from collections import deque
from itertools import zip_longest
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote_from_bytes


# Trace v2 adds required per-sample retained-memory evidence (``memory_bytes``).
TRACE_SCHEMA_VERSION = 2
# Summary v2 adds the device, method-memory, latency, and throughput evidence blocks.
SUMMARY_SCHEMA_VERSION = 2
TRACE_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "timestep",
    "sample_idx",
    "ground_truth_domain",
    "ground_truth_class",
    "prediction",
    "correct",
    "predicted_entropy",
    "inferred_context",
    "memory_size",
    "num_active_contexts",
    "memory_bytes",
    "latency_ms",
)
ADMISSION_TRACE_FIELDS = (
    "admission_prediction",
    "admission_normalized_entropy",
    "admitted_to_memory",
)
RETRIEVAL_PROFILE_TRACE_FIELDS = (
    "retrieval_profile",
    "retrieval_elapsed_ms",
    "retrieval_candidate_count",
    "retrieval_eligible_candidate_count",
    "retrieval_returned_support_count",
    "retrieval_active_class_count",
)
# This is deliberately independent of retrieval timing profiling.  It records
# the composition of the support actually selected by methods that expose it.
# The extension is optional so evidence produced before soft-routing work
# remains resumable under the v2 trace contract.
SUPPORT_COMPOSITION_TRACE_FIELDS = (
    "returned_support_count",
    "active_class_count",
    "class_coverage",
    "same_domain_ratio",
    "cross_domain_ratio",
    "effective_sample_size",
)
SOFT_ROUTING_TRACE_FIELDS = (
    "context_strength",
    "selection_change_ratio",
    "mean_context_bonus",
    "mean_rank_displacement",
)
REFERENCE_IDENTITY_FIELDS = (
    "dataset",
    "model",
    "device",
    "data_root",
    "tta_mode",
    "batch_size",
    "metric_window_size",
    "metric_window_stride",
    "stream_block_size",
    "artifact_provenance",
    "artifacts",
    "reference_config",
    "reference_config_path",
)
_CACHE_PATH_PARTS = frozenset({
    "__pycache__", ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})


def _json_default(value: Any) -> Any:
    """Convert common scalar-like values without importing numerical packages."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalise(value: Any) -> Any:
    """Round-trip through JSON to ensure manifest data is deterministic JSON."""
    return json.loads(json.dumps(value, default=_json_default, sort_keys=True))


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    """Atomically replace *path* with canonical, UTF-8 JSON.

    The temporary file is placed beside its destination so ``os.replace`` is
    atomic on normal local filesystems.  Directory creation is intentional for
    experiment output paths.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _normalise(payload), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _git_metadata(repository: Optional[os.PathLike[str] | str]) -> dict[str, Any]:
    if repository is None:
        return {"available": False}
    cwd = str(Path(repository))

    def command(*arguments: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=cwd, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = command("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    status = command("status", "--porcelain")
    metadata: dict[str, Any] = {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
    }
    root = command("rev-parse", "--show-toplevel")
    if root is not None:
        source = _source_tree_fingerprint(Path(root))
        if source is not None:
            metadata["source"] = source
    return metadata


def _source_tree_fingerprint(repository: Path) -> Optional[dict[str, Any]]:
    """Return an auditable digest of experiment-affecting source files.

    Git supplies the tracked files and non-ignored untracked files, so generated
    evidence and ignored caches are naturally excluded.  Only regular files are
    accepted after an ``lstat`` check: this deliberately avoids following a
    symlink outside the repository.
    """
    pathspecs = ("src", "cfg", "shell", "environment.yml", "environment-cuda.yml")

    def listed_files(*arguments: str) -> Optional[list[bytes]]:
        try:
            completed = subprocess.run(
                ["git", *arguments, "--", *pathspecs],
                cwd=str(repository),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return [item for item in completed.stdout.split(b"\0") if item]

    tracked = listed_files("ls-files", "-z", "--cached")
    untracked = listed_files("ls-files", "-z", "--others", "--exclude-standard")
    if tracked is None or untracked is None:
        return None

    files: dict[str, str] = {}
    for raw_path in sorted(set(tracked + untracked)):
        relative_path = Path(os.fsdecode(raw_path))
        # git paths are repository-relative; reject anything malformed before
        # constructing the filesystem path.
        if relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        if any(part in _CACHE_PATH_PARTS for part in relative_path.parts):
            continue
        if relative_path.suffix in {".pyc", ".pyo"}:
            continue
        candidate = repository / relative_path
        try:
            parent = repository
            has_symlink_parent = False
            for component in relative_path.parts[:-1]:
                parent /= component
                if stat.S_ISLNK(parent.lstat().st_mode):
                    has_symlink_parent = True
                    break
            if has_symlink_parent:
                continue
            if not stat.S_ISREG(candidate.lstat().st_mode):
                continue
            digest = hashlib.sha256()
            with candidate.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            # A concurrent deletion or unreadable source file makes this
            # snapshot incomplete, so omit the fingerprint rather than claim
            # an exact executable state.
            return None
        audit_path = quote_from_bytes(raw_path, safe="/-._")
        files[audit_path] = digest.hexdigest()

    aggregate = hashlib.sha256()
    for relative_path, digest in files.items():
        aggregate.update(relative_path.encode("utf-8", "surrogateescape"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "algorithm": "sha256",
        "path_encoding": "percent-encoded-posix-bytes",
        "fingerprint": aggregate.hexdigest(),
        "files": files,
    }


def _package_versions(package_names: Optional[list[str]]) -> dict[str, str]:
    if package_names is None:
        distributions = metadata.distributions()
        versions = {
            distribution.metadata["Name"].lower(): distribution.version
            for distribution in distributions
            if distribution.metadata.get("Name")
        }
        return dict(sorted(versions.items()))
    versions: dict[str, str] = {}
    for name in sorted(set(package_names), key=str.lower):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_run_manifest(
    *,
    run_id: str,
    args: Mapping[str, Any] | Any,
    config: Optional[Mapping[str, Any]] = None,
    device: Any = None,
    dataset: Optional[Mapping[str, Any]] = None,
    stream: Optional[Mapping[str, Any]] = None,
    hardware: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    repository: Optional[os.PathLike[str] | str] = ".",
    package_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build a canonical run manifest without an implicit timestamp.

    ``args`` may be an argparse namespace or mapping.  Determinism means equal
    inputs and environment state produce equal JSON; callers should add a
    caller-controlled ``started_at`` field separately when wall-clock timing is
    needed.
    """
    args_mapping = vars(args) if hasattr(args, "__dict__") else args
    if not isinstance(args_mapping, Mapping):
        raise TypeError("args must be a mapping or an argparse-style namespace")
    return _normalise({
        "schema_version": 1,
        "run_id": str(run_id),
        "args": dict(args_mapping),
        "config": dict(config or {}),
        "git": _git_metadata(repository),
        "runtime": {
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "packages": _package_versions(package_names),
        "device": str(device) if device is not None else None,
        "hardware": dict(hardware or {}),
        "dataset": dict(dataset or {}),
        "stream": dict(stream or {}),
        # Every manifest makes the availability of artifact evidence explicit.
        # Direct callers that have not opted into verification remain valid,
        # but cannot accidentally look like they supplied verified inputs.
        "artifacts": dict(artifacts or {
            "status": "unavailable",
            "mode": "off",
            "reason": "artifact provenance was not requested",
            "model": {"status": "unavailable"},
            "dataset": {"status": "unavailable"},
        }),
    })


def write_run_manifest(path: os.PathLike[str] | str, **manifest_kwargs: Any) -> dict[str, Any]:
    """Build and atomically write a run manifest, returning the written value."""
    manifest = build_run_manifest(**manifest_kwargs)
    atomic_write_json(path, manifest)
    return manifest


def write_summary(path: os.PathLike[str] | str, summary: Mapping[str, Any]) -> None:
    """Atomically persist a completed-run summary using the canonical JSON form."""
    atomic_write_json(path, summary)


class JsonlTraceWriter:
    """Append validated per-sample evidence rows to a versioned JSONL file.

    Rows are flushed after every write so interrupted experiments retain all
    completed samples.  ``close`` also fsyncs the file.  Use as a context
    manager in normal experiment code.
    """

    def __init__(self, path: os.PathLike[str] | str, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if self._handle.closed:
            raise ValueError("cannot write to a closed trace")
        row = dict(record)
        row.setdefault("schema_version", TRACE_SCHEMA_VERSION)
        row.setdefault("run_id", self.run_id)
        missing = [field for field in TRACE_REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"trace record missing required fields: {', '.join(missing)}")
        if row["schema_version"] != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema version: {row['schema_version']!r}")
        if str(row["run_id"]) != self.run_id:
            raise ValueError("trace record run_id does not match writer run_id")
        if not isinstance(row["timestep"], int) or isinstance(row["timestep"], bool) or row["timestep"] < 0:
            raise ValueError("timestep must be a non-negative integer")
        if not isinstance(row["correct"], bool):
            raise ValueError("correct must be a boolean")
        admission_present = [field in row for field in ADMISSION_TRACE_FIELDS]
        if any(admission_present) and not all(admission_present):
            raise ValueError("admission trace fields must be all present or all absent")
        if all(admission_present):
            if not _is_nonnegative_integer(row["admission_prediction"]):
                raise ValueError("admission_prediction must be a non-negative integer")
            if not _is_finite_number(row["admission_normalized_entropy"], minimum=0.0, maximum=1.0):
                raise ValueError("admission_normalized_entropy must be a finite probability")
            if not isinstance(row["admitted_to_memory"], bool):
                raise ValueError("admitted_to_memory must be a boolean")
        profile_present = [field in row for field in RETRIEVAL_PROFILE_TRACE_FIELDS]
        if any(profile_present) and not all(profile_present):
            raise ValueError("retrieval profile trace fields must be all present or all absent")
        if all(profile_present):
            if row["retrieval_profile"] != "causal_sync_v1":
                raise ValueError("unsupported retrieval_profile")
            if not _is_finite_number(row["retrieval_elapsed_ms"], minimum=0.0):
                raise ValueError("retrieval_elapsed_ms must be a finite non-negative number")
            for field in RETRIEVAL_PROFILE_TRACE_FIELDS[2:]:
                if not _is_nonnegative_integer(row[field]):
                    raise ValueError(f"{field} must be a non-negative integer")
        composition_present = [field in row for field in SUPPORT_COMPOSITION_TRACE_FIELDS]
        if any(composition_present) and not all(composition_present):
            raise ValueError("support composition trace fields must be all present or all absent")
        if all(composition_present):
            for field in ("returned_support_count", "active_class_count"):
                if not _is_nonnegative_integer(row[field]):
                    raise ValueError(f"{field} must be a non-negative integer")
            for field in ("class_coverage", "same_domain_ratio", "cross_domain_ratio"):
                if not _is_finite_number(row[field], minimum=0.0, maximum=1.0):
                    raise ValueError(f"{field} must be a finite probability")
            expected_ratio_sum = 1.0 if row["returned_support_count"] else 0.0
            if not math.isclose(
                row["same_domain_ratio"] + row["cross_domain_ratio"],
                expected_ratio_sum, rel_tol=0.0, abs_tol=1e-6,
            ):
                raise ValueError("same_domain_ratio and cross_domain_ratio disagree with support count")
            if not _is_finite_number(row["effective_sample_size"], minimum=0.0):
                raise ValueError("effective_sample_size must be a finite non-negative number")
        soft_present = [field in row for field in SOFT_ROUTING_TRACE_FIELDS]
        if any(soft_present) and not all(soft_present):
            raise ValueError("soft routing trace fields must be all present or all absent")
        if all(soft_present):
            if not _is_finite_number(row["selection_change_ratio"], minimum=0.0, maximum=1.0):
                raise ValueError("selection_change_ratio must be a finite probability")
            for field in ("context_strength", "mean_context_bonus", "mean_rank_displacement"):
                if not _is_finite_number(row[field], minimum=0.0):
                    raise ValueError(f"{field} must be a finite non-negative number")
        memory_bytes = row["memory_bytes"]
        if memory_bytes is not None and (
            not isinstance(memory_bytes, int)
            or isinstance(memory_bytes, bool)
            or memory_bytes < 0
        ):
            raise ValueError("memory_bytes must be a non-negative integer or null")
        latency_ms = row["latency_ms"]
        if (
            not isinstance(latency_ms, (int, float))
            or isinstance(latency_ms, bool)
            or not math.isfinite(latency_ms)
            or latency_ms < 0
        ):
            raise ValueError("latency_ms must be a finite non-negative number")
        encoded = json.dumps(_normalise(row), ensure_ascii=False, sort_keys=True)
        self._handle.write(encoded + "\n")
        self._handle.flush()
        return row

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def __enter__(self) -> "JsonlTraceWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def compare_trace_negative_adaptation(
    adapted_path: os.PathLike[str] | str,
    reference_path: os.PathLike[str] | str,
    *,
    window_size: int = 50,
    stride: Optional[int] = None,
    _expected_reference_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Compare aligned traces without loading complete runs into memory."""
    if stride is None:
        stride = window_size
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    identity_fields = ("timestep", "sample_idx", "ground_truth_domain", "ground_truth_class")
    adapted_window: deque[bool] = deque(maxlen=window_size)
    reference_window: deque[bool] = deque(maxlen=window_size)
    negative_windows = 0
    total_windows = 0
    row_count = 0
    reference_digest = hashlib.sha256() if _expected_reference_sha256 is not None else None
    with Path(adapted_path).open(encoding="utf-8") as adapted_file, \
            Path(reference_path).open(encoding="utf-8") as reference_file:
        for adapted_line, reference_line in zip_longest(adapted_file, reference_file):
            if adapted_line is None or reference_line is None:
                raise ValueError("adapted and reference traces have different lengths")
            adapted = json.loads(adapted_line)
            reference = json.loads(reference_line)
            if reference_digest is not None:
                reference_digest.update(reference_line.encode("utf-8"))
            if any(adapted.get(field) != reference.get(field) for field in identity_fields):
                raise ValueError(f"trace identity mismatch at row {row_count}")
            if not isinstance(adapted.get("correct"), bool) or not isinstance(reference.get("correct"), bool):
                raise TypeError("trace correct fields must be booleans")
            adapted_window.append(adapted["correct"])
            reference_window.append(reference["correct"])
            window_start = row_count - window_size + 1
            if len(adapted_window) == window_size and window_start % stride == 0:
                total_windows += 1
                negative_windows += int(sum(adapted_window) < sum(reference_window))
            row_count += 1
    if (
        reference_digest is not None
        and reference_digest.hexdigest() != _expected_reference_sha256
    ):
        raise ValueError("reference trace changed after provenance validation")
    if total_windows == 0:
        raise ValueError("at least one full comparison window is required")
    return {
        "status": "computed",
        "value": negative_windows / total_windows,
        "negative_windows": negative_windows,
        "total_windows": total_windows,
        "window_size": window_size,
        "stride": stride,
        "reference_trace": str(reference_path),
    }


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: Any, *, minimum: float | None = None,
                      maximum: float | None = None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (minimum is None or value >= minimum)
        and (maximum is None or value <= maximum)
    )


def _require_matching_probability(value: Any, expected: float, field: str) -> None:
    if not _is_finite_number(value, minimum=0.0, maximum=1.0) or not math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"reference summary {field} disagrees with its trace")


def _open_text_no_follow(path: Path):
    """Open a text file without following its final symlink where supported."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        return os.fdopen(descriptor, "r", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _require_same_regular_path(
    path: Path, expected_stat: os.stat_result, changed_message: str
) -> None:
    """Require the canonical pathname still names the validated regular file."""
    try:
        current_stat = path.lstat()
    except OSError as exc:
        raise ValueError(changed_message) from exc
    if (
        stat.S_ISLNK(current_stat.st_mode)
        or not stat.S_ISREG(current_stat.st_mode)
        or (current_stat.st_dev, current_stat.st_ino) != (
            expected_stat.st_dev, expected_stat.st_ino
        )
    ):
        raise ValueError(changed_message)


def _load_inode_bound_json(
    path: Path, evidence_name: str, expected_parent_stat: os.stat_result
) -> Any:
    """Load one canonical sibling without following links or racing replacement."""
    try:
        current_parent_stat = path.parent.lstat()
        if (
            stat.S_ISLNK(current_parent_stat.st_mode)
            or not stat.S_ISDIR(current_parent_stat.st_mode)
            or (current_parent_stat.st_dev, current_parent_stat.st_ino) != (
                expected_parent_stat.st_dev, expected_parent_stat.st_ino
            )
        ):
            raise ValueError("reference run directory changed during validation")
        expected_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(
            f"reference trace lacks sibling {evidence_name} evidence: {path}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"reference sibling {evidence_name} evidence is invalid: {path}") from exc
    if stat.S_ISLNK(expected_stat.st_mode) or not stat.S_ISREG(expected_stat.st_mode):
        raise ValueError(
            f"reference sibling {evidence_name} evidence must be a regular file, not a symlink"
        )
    try:
        with _open_text_no_follow(path) as evidence_file:
            opened_stat = os.fstat(evidence_file.fileno())
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                expected_stat.st_dev, expected_stat.st_ino
            ):
                raise ValueError(
                    f"reference sibling {evidence_name} evidence changed while it was being opened"
                )
            changed_message = (
                f"reference sibling {evidence_name} evidence changed while it was being read"
            )
            _require_same_regular_path(path, expected_stat, changed_message)
            try:
                result = json.load(evidence_file)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"reference sibling {evidence_name} evidence is invalid: {path}"
                ) from exc
            _require_same_regular_path(path, expected_stat, changed_message)
            return result
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"reference sibling {evidence_name} evidence is invalid: {path}") from exc


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_complete_artifact_evidence(artifacts: Any) -> None:
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "status", "mode", "model", "dataset"
    } or artifacts.get("status") != "verified" \
            or artifacts.get("mode") not in {"fast", "exact"}:
        raise ValueError("reference artifact provenance must be verified in fast or exact mode")
    model = artifacts.get("model")
    dataset = artifacts.get("dataset")
    model_fields = {
        "status", "model", "official_name", "url", "expected_sha256", "filename",
        "publisher", "trust", "path", "actual_sha256", "size_bytes",
    }
    dataset_fields = {
        "status", "schema_version", "dataset", "root", "sidecar", "verified_exact",
        "content_algorithm", "root_digest", "sidecar_sha256", "file_count", "acquisition",
    }
    if not isinstance(model, Mapping) or set(model) != model_fields:
        raise ValueError("reference model artifact evidence is incomplete")
    if not isinstance(dataset, Mapping) or set(dataset) != dataset_fields:
        raise ValueError("reference dataset artifact evidence is incomplete")
    if model.get("status") != "verified" or dataset.get("status") != "verified":
        raise ValueError("reference model and dataset artifacts must be verified")
    if not _is_sha256(model.get("actual_sha256")) \
            or model.get("actual_sha256") != model.get("expected_sha256"):
        raise ValueError("reference model artifact digest is invalid")
    if not _is_sha256(dataset.get("root_digest")) or not _is_sha256(dataset.get("sidecar_sha256")):
        raise ValueError("reference dataset artifact digests are invalid")
    for field in ("model", "official_name", "url", "filename", "publisher", "trust"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise ValueError(f"reference model artifact {field} is invalid")
    if model["publisher"] != "OpenAI" or model["trust"] != "pinned_official" \
            or not _is_nonnegative_integer(model.get("size_bytes")) or model["size_bytes"] == 0:
        raise ValueError("reference model artifact trust or size evidence is invalid")
    if dataset.get("schema_version") != 1 or not isinstance(dataset.get("dataset"), str) \
            or not dataset["dataset"] or not isinstance(dataset.get("verified_exact"), bool) \
            or dataset.get("content_algorithm") != "sha256" \
            or not _is_nonnegative_integer(dataset.get("file_count")) or dataset["file_count"] == 0 \
            or not isinstance(dataset.get("acquisition"), Mapping):
        raise ValueError("reference dataset artifact evidence is malformed")
    if dataset["verified_exact"] != (artifacts["mode"] == "exact"):
        raise ValueError("reference dataset artifact verification mode is inconsistent")
    for value, field in (
        (model.get("path"), "model.path"),
        (dataset.get("root"), "dataset.root"),
        (dataset.get("sidecar"), "dataset.sidecar"),
    ):
        if not isinstance(value, str) or not Path(value).is_absolute() \
                or str(Path(value).resolve()) != value:
            raise ValueError(f"reference artifact {field} must be a canonical absolute path")


def _validate_reference_run_identity(
    manifest: Mapping[str, Any], expected_identity: Mapping[str, Any]
) -> None:
    missing = [field for field in REFERENCE_IDENTITY_FIELDS if field not in expected_identity]
    if missing:
        raise ValueError("expected reference identity is missing: " + ", ".join(missing))
    args = manifest.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("reference manifest args evidence is missing")
    expected_artifacts = expected_identity["artifacts"]
    _validate_complete_artifact_evidence(expected_artifacts)
    _validate_complete_artifact_evidence(manifest.get("artifacts"))
    if manifest["artifacts"] != expected_artifacts:
        raise ValueError("reference artifact evidence does not match the current run")
    if expected_artifacts["mode"] != expected_identity["artifact_provenance"]:
        raise ValueError("expected artifact provenance mode is inconsistent")
    scalar_fields = (
        "dataset", "model", "tta_mode", "batch_size", "metric_window_size",
        "metric_window_stride", "artifact_provenance",
        "stream_block_size",
    )
    for field in scalar_fields:
        if args.get(field) != expected_identity[field]:
            raise ValueError(f"reference manifest {field} does not match the current run")
    if args.get("tta_algo") != "NoAdapt":
        raise ValueError("reference trace must come from a NoAdapt baseline")
    expected_device = expected_identity["device"]
    if not isinstance(expected_device, str) or not expected_device \
            or manifest.get("device") != expected_device or args.get("device") != expected_device:
        raise ValueError("reference resolved device does not match the current run")
    expected_data_root = expected_identity["data_root"]
    if not isinstance(expected_data_root, str) \
            or str(Path(expected_data_root).expanduser().resolve()) != expected_data_root \
            or args.get("data_root") != expected_data_root:
        raise ValueError("reference canonical data_root does not match the current run")
    expected_config_path = expected_identity["reference_config_path"]
    if not isinstance(expected_config_path, str) \
            or str(Path(expected_config_path).expanduser().resolve()) != expected_config_path \
            or args.get("config_path") != expected_config_path:
        raise ValueError("reference NoAdapt config path does not match the current run")
    if manifest.get("config") != expected_identity["reference_config"]:
        raise ValueError("reference NoAdapt config does not match the expected baseline config")
    if not isinstance(expected_identity["reference_config"], Mapping) \
            or Path(expected_config_path).name != "NoAdapt.yaml":
        raise ValueError("expected NoAdapt config identity is invalid")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("name") != expected_identity["dataset"]:
        raise ValueError("reference manifest dataset does not match the current run")
    if manifest["artifacts"]["model"]["model"] != expected_identity["model"] \
            or manifest["artifacts"]["dataset"]["dataset"] != expected_identity["dataset"].lower():
        raise ValueError("reference artifact names do not match the current run")


def verify_reference_trace_stream_fingerprint(
    reference_path: os.PathLike[str] | str,
    expected_fingerprint: str,
    *,
    expected_identity: Optional[Mapping[str, Any]] = None,
) -> str:
    """Require a reference trace's sibling stream export to match a stream.

    Trace-row identity checks are useful, but are not provenance: two distinct
    streams can share the same per-row identities.  The completed reference
    run exports its schedule and completed summary beside ``trace.jsonl``.
    Verify the self-authenticating stream export and its summary fingerprint
    plus an explicit current-run identity and verified artifact report before
    a CLI run uses the trace as a negative-adaptation baseline. The returned
    SHA-256 digest lets the comparator reject a file replacement between
    validation and comparison.
    """
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("expected stream fingerprint must be a non-empty string")
    if not isinstance(expected_identity, Mapping):
        raise ValueError("expected reference identity is required")

    trace_path = Path(reference_path)
    if trace_path.name != "trace.jsonl":
        raise ValueError("reference trace must use the canonical trace.jsonl filename")
    try:
        trace_stat = trace_path.lstat()
        parent_stat = trace_path.parent.lstat()
    except OSError as exc:
        raise ValueError(f"reference trace is unavailable: {trace_path}") from exc
    if stat.S_ISLNK(trace_stat.st_mode) or not stat.S_ISREG(trace_stat.st_mode):
        raise ValueError("reference trace must be a regular file, not a symlink")
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise ValueError("reference trace parent must be a real run directory")

    reference_run_dir = trace_path.parent
    stream_path = reference_run_dir / "stream.json"
    summary_path = reference_run_dir / "summary.json"
    manifest_path = reference_run_dir / "manifest.json"
    stream = _load_inode_bound_json(stream_path, "stream", parent_stat)

    if not isinstance(stream, Mapping):
        raise ValueError("reference sibling stream evidence must be a JSON object")
    metadata = stream.get("metadata")
    references = stream.get("references")
    exported_fingerprint = stream.get("fingerprint")
    if not isinstance(metadata, Mapping) or not isinstance(references, list):
        raise ValueError("reference sibling stream evidence has invalid structure")
    metadata_fingerprint = metadata.get("fingerprint")
    if (
        not isinstance(exported_fingerprint, str)
        or not isinstance(metadata_fingerprint, str)
        or exported_fingerprint != metadata_fingerprint
    ):
        raise ValueError("reference sibling stream evidence has an invalid fingerprint")

    canonical_metadata = dict(metadata)
    canonical_metadata.pop("fingerprint", None)
    try:
        encoded = json.dumps(
            {"metadata": canonical_metadata, "references": references},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("reference sibling stream evidence is not canonical JSON") from exc
    actual_fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if exported_fingerprint != actual_fingerprint:
        raise ValueError("reference sibling stream evidence fingerprint does not verify")
    if actual_fingerprint != expected_fingerprint:
        raise ValueError("reference stream fingerprint does not match the current stream")

    summary = _load_inode_bound_json(summary_path, "summary", parent_stat)
    if (
        not isinstance(summary, Mapping)
        or not isinstance(summary.get("schema_version"), int)
        or isinstance(summary.get("schema_version"), bool)
        or summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or not isinstance(summary.get("run_id"), str)
        or not summary["run_id"]
        or summary.get("stream_fingerprint") != actual_fingerprint
    ):
        raise ValueError("reference sibling summary evidence has an invalid fingerprint")

    summary_run_id = summary["run_id"]
    summary_num_samples = summary.get("num_samples")
    if (
        not isinstance(summary_num_samples, int)
        or isinstance(summary_num_samples, bool)
        or summary_num_samples != len(references)
    ):
        raise ValueError("reference summary sample count does not match its stream")
    metadata_num_samples = metadata.get("num_samples")
    if (
        not isinstance(metadata_num_samples, int)
        or isinstance(metadata_num_samples, bool)
        or metadata_num_samples != len(references)
    ):
        raise ValueError("reference stream sample count is invalid")
    for index, reference in enumerate(references):
        if (
            not isinstance(reference, list)
            or len(reference) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in reference
            )
        ):
            raise ValueError(f"reference stream has an invalid identity at row {index}")

    trace_digest = hashlib.sha256()
    rows: list[Mapping[str, Any]] = []
    row_count = 0
    try:
        with _open_text_no_follow(trace_path) as trace_file:
            opened_stat = os.fstat(trace_file.fileno())
            if (opened_stat.st_dev, opened_stat.st_ino) != (trace_stat.st_dev, trace_stat.st_ino):
                raise ValueError("reference trace changed while it was being opened")
            _require_same_regular_path(
                trace_path, trace_stat, "reference trace changed while it was being read"
            )
            for line_number, line in enumerate(trace_file, 1):
                trace_digest.update(line.encode("utf-8"))
                if not line.strip():
                    raise ValueError(f"reference trace has a blank row at line {line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"reference trace has malformed JSON at line {line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"reference trace row {line_number} is not an object")
                missing = [field for field in TRACE_REQUIRED_FIELDS if field not in row]
                if missing:
                    raise ValueError(
                        f"reference trace row {line_number} is missing: {', '.join(missing)}"
                    )
                if row_count >= len(references):
                    raise ValueError("reference trace row count exceeds its verified stream")
                if (
                    not isinstance(row["schema_version"], int)
                    or isinstance(row["schema_version"], bool)
                    or row["schema_version"] != TRACE_SCHEMA_VERSION
                ):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid schema_version"
                    )
                if not isinstance(row["run_id"], str) or not row["run_id"]:
                    raise ValueError(f"reference trace row {line_number} has an invalid run_id")
                if row["run_id"] != summary_run_id:
                    raise ValueError(f"reference trace row {line_number} has a foreign run_id")
                if (
                    not isinstance(row["timestep"], int)
                    or isinstance(row["timestep"], bool)
                    or row["timestep"] != row_count
                ):
                    raise ValueError(
                        f"reference trace row {line_number} has a non-sequential timestep"
                    )
                expected_domain, expected_sample = references[row_count]
                if (
                    not isinstance(row["sample_idx"], int)
                    or isinstance(row["sample_idx"], bool)
                    or row["sample_idx"] != expected_sample
                ):
                    raise ValueError(
                        f"reference trace row {line_number} sample_idx does not match its stream"
                    )
                if (
                    not isinstance(row["ground_truth_domain"], int)
                    or isinstance(row["ground_truth_domain"], bool)
                    or row["ground_truth_domain"] != expected_domain
                ):
                    raise ValueError(
                        f"reference trace row {line_number} domain does not match its stream"
                    )
                if not _is_nonnegative_integer(row["ground_truth_class"]):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid ground_truth_class"
                    )
                if not _is_nonnegative_integer(row["prediction"]):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid prediction"
                    )
                if not isinstance(row["correct"], bool):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid correct field"
                    )
                admission_present = [field in row for field in ADMISSION_TRACE_FIELDS]
                if any(admission_present) and not all(admission_present):
                    raise ValueError(
                        f"reference trace row {line_number} has partial admission evidence"
                    )
                if all(admission_present):
                    if not _is_nonnegative_integer(row["admission_prediction"]):
                        raise ValueError(
                            f"reference trace row {line_number} has invalid admission_prediction"
                        )
                    if not _is_finite_number(row["admission_normalized_entropy"], minimum=0.0, maximum=1.0):
                        raise ValueError(
                            f"reference trace row {line_number} has invalid admission_normalized_entropy"
                        )
                    if not isinstance(row["admitted_to_memory"], bool):
                        raise ValueError(
                            f"reference trace row {line_number} has invalid admitted_to_memory"
                        )
                if row["correct"] != (row["prediction"] == row["ground_truth_class"]):
                    raise ValueError(
                        f"reference trace row {line_number} correct disagrees with prediction"
                    )
                if not _is_finite_number(row["predicted_entropy"], minimum=0.0):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid predicted_entropy"
                    )
                if not _is_nonnegative_integer(row["memory_size"]):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid memory_size"
                    )
                active_contexts = row["num_active_contexts"]
                if active_contexts is not None and not _is_nonnegative_integer(active_contexts):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid num_active_contexts"
                    )
                inferred_context = row["inferred_context"]
                if inferred_context is not None and not _is_nonnegative_integer(inferred_context):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid inferred_context"
                    )
                memory_bytes = row["memory_bytes"]
                if memory_bytes is not None and not _is_nonnegative_integer(memory_bytes):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid memory_bytes"
                    )
                if not _is_finite_number(row["latency_ms"], minimum=0.0):
                    raise ValueError(
                        f"reference trace row {line_number} has an invalid latency_ms"
                    )
                rows.append(row)
                row_count += 1
            _require_same_regular_path(
                trace_path, trace_stat, "reference trace changed while it was being read"
            )
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"reference trace is unreadable: {trace_path}") from exc
    if row_count != len(references):
        raise ValueError("reference trace row count does not match its verified stream")

    # The direct CLI path does not otherwise load the manifest, but it is the
    # authoritative mapping between numeric trace domains and summary keys.
    manifest = _load_inode_bound_json(manifest_path, "manifest", parent_stat)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != 1
        or manifest.get("run_id") != summary_run_id
        or not isinstance(manifest.get("dataset"), Mapping)
        or not isinstance(manifest["dataset"].get("environments"), list)
        or not manifest["dataset"]["environments"]
        or any(not isinstance(name, str) for name in manifest["dataset"]["environments"])
        or len(set(manifest["dataset"]["environments"])) != len(
            manifest["dataset"]["environments"]
        )
    ):
        raise ValueError("reference sibling manifest evidence is invalid")
    _validate_reference_run_identity(manifest, expected_identity)
    environments = manifest["dataset"]["environments"]
    micro_accuracy = sum(row["correct"] for row in rows) / len(rows)
    _require_matching_probability(summary.get("micro_accuracy"), micro_accuracy, "micro_accuracy")
    counts = summary.get("domain_sample_counts")
    accuracies = summary.get("domain_accuracies")
    if not isinstance(counts, Mapping) or not isinstance(accuracies, Mapping):
        raise ValueError("reference summary domain evidence is invalid")
    if set(counts) != set(environments) or set(accuracies) != set(environments):
        raise ValueError("reference summary domain evidence has invalid keys")
    trace_counts = [0] * len(environments)
    trace_correct = [0] * len(environments)
    for row in rows:
        domain = row["ground_truth_domain"]
        if domain >= len(environments):
            raise ValueError("reference trace domain exceeds manifest environments")
        trace_counts[domain] += 1
        trace_correct[domain] += int(row["correct"])
    active_accuracies = []
    for domain, name in enumerate(environments):
        if not _is_nonnegative_integer(counts[name]) or counts[name] != trace_counts[domain]:
            raise ValueError("reference summary domain sample count disagrees with its trace")
        expected = trace_correct[domain] / trace_counts[domain] if trace_counts[domain] else None
        if expected is None:
            if accuracies[name] is not None:
                raise ValueError("reference summary domain accuracy disagrees with its trace")
        else:
            _require_matching_probability(accuracies[name], expected, f"domain_accuracies.{name}")
            active_accuracies.append(expected)
    _require_matching_probability(
        summary.get("macro_domain_accuracy"), sum(active_accuracies) / len(active_accuracies),
        "macro_domain_accuracy",
    )
    _require_matching_probability(
        summary.get("worst_domain_accuracy"), min(active_accuracies), "worst_domain_accuracy",
    )
    sliding = summary.get("sliding_window")
    if not isinstance(sliding, Mapping) or not _is_nonnegative_integer(sliding.get("window_size")) \
            or sliding["window_size"] == 0 or not _is_nonnegative_integer(sliding.get("stride")) \
            or sliding["stride"] == 0 or not isinstance(sliding.get("values"), list):
        raise ValueError("reference summary sliding-window evidence is invalid")
    expected_windows = []
    for start in range(0, len(rows) - sliding["window_size"] + 1, sliding["stride"]):
        stop = start + sliding["window_size"]
        expected_windows.append({
            "start_timestep": start,
            "end_timestep": stop - 1,
            "accuracy": sum(row["correct"] for row in rows[start:stop]) / sliding["window_size"],
        })
    if sliding["values"] != expected_windows:
        raise ValueError("reference summary sliding-window evidence disagrees with its trace")
    _require_same_regular_path(
        trace_path, trace_stat, "reference trace changed before validation completed"
    )
    return trace_digest.hexdigest()
