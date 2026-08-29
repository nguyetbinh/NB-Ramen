"""Plan and run the deterministic first Latent Ramen experiment matrix.

The module deliberately does not import the training stack.  This keeps
planning, resume checks, and data preflight useful on scheduler/login nodes.
Run it with ``python -m src.runtime.experiment_matrix`` from the repository,
``python -m runtime.experiment_matrix`` from ``src``, or by giving this file
to Python directly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Iterable, Sequence

try:  # Supports both ``python -m runtime...`` and direct-file invocation.
    from .preflight import validate_dataset_layout
    from .artifact_provenance import (
        CIFAR100C_OFFICIAL_ACQUISITION,
        SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
        default_sidecar_path,
        resolve_clip_model,
    )
except ImportError:  # pragma: no cover - exercised only by direct invocation
    from preflight import validate_dataset_layout
    from artifact_provenance import (
        CIFAR100C_OFFICIAL_ACQUISITION,
        SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
        default_sidecar_path,
        resolve_clip_model,
    )

try:
    from ..evaluation.evidence import (
        SUMMARY_SCHEMA_VERSION,
        TRACE_REQUIRED_FIELDS,
        ADMISSION_TRACE_FIELDS,
        RETRIEVAL_PROFILE_TRACE_FIELDS,
        TRACE_SCHEMA_VERSION,
        compare_trace_negative_adaptation,
        validate_failure_analysis,
    )
    from ..evaluation.failure_analysis_artifacts import ReplaySidecarReader, parse_counterfactual_thresholds
    from ..evaluation.online_metrics import domain_shift_recovery_times
    from ..evaluation.routing_metrics import routing_diagnostics
except ImportError:  # ``runtime`` top-level package or direct-file invocation.
    source_root = str(Path(__file__).resolve().parents[1])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from evaluation.evidence import (
        SUMMARY_SCHEMA_VERSION,
        TRACE_REQUIRED_FIELDS,
        ADMISSION_TRACE_FIELDS,
        RETRIEVAL_PROFILE_TRACE_FIELDS,
        TRACE_SCHEMA_VERSION,
        compare_trace_negative_adaptation,
        validate_failure_analysis,
    )
    from evaluation.online_metrics import domain_shift_recovery_times
    from evaluation.routing_metrics import routing_diagnostics
    from evaluation.failure_analysis_artifacts import ReplaySidecarReader, parse_counterfactual_thresholds


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS = ("CIFAR100C", "DomainNet")
DEFAULT_STREAMS = ("iid_mixed", "block", "gradual", "recurring", "imbalanced")
SUPPORTED_STREAMS = DEFAULT_STREAMS + ("novel_domain", "class_domain_correlated", "bursty")
DEFAULT_METHODS = (
    "NoAdapt",
    "Tent",
    "Ramen",
    "CausalRamen",
    "StructuredAtomicRamen",
    "RandomMemoryRamen",
    "SameClassRamen",
    "GlobalNearestRamen",
    "ContextOnlyRamen",
    "OracleLatentRamen",
    "LatentRamen",
)
SUPPORTED_METHODS = DEFAULT_METHODS + ("EntropyGatedLatentRamen",)
MODEL_BY_DATASET = {"CIFAR100C": "clip_vitbase16", "DomainNet": "clip_vitbase32"}
BATCH_SIZE_BY_DATASET = {"CIFAR100C": 100, "DomainNet": 100}
CONFIG_HASH_LENGTH = 12
MISSING_CONFIG_HASH = "missing"
SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")
SUPPORTED_ARTIFACT_PROVENANCE = ("fast", "exact")
OPEN_SET_SPLIT = "open-set-cifar100-split-v1"
OPEN_SET_OOD_RATIOS = (0, 0.1, 0.3, 0.5)
MAX_RUN_ID_LENGTH = 128
RUN_ID_IDENTITY_HASH_LENGTH = 32
SUMMARY_REQUIRED_FIELDS = (
    "schema_version", "run_id", "num_samples", "micro_accuracy",
    "macro_domain_accuracy", "worst_domain_accuracy", "domain_accuracies",
    "domain_sample_counts", "sliding_window", "post_shift_recovery_time",
    "negative_adaptation_rate", "routing_diagnostics", "peak_device_memory_bytes",
    "device_memory", "method_memory", "forward_latency", "throughput",
    "retrieval_latency", "stream_fingerprint",
)
_UNSET = object()


class IncompleteRunError(RuntimeError):
    """A resume target has evidence that is neither complete nor absent."""


@dataclass(frozen=True)
class ExperimentRun:
    dataset: str
    stream_mode: str
    seed: int
    method: str
    run_id: str
    model: str
    batch_size: int
    evidence_dir: Path
    data_root: Path
    device: str
    max_eval_samples: int | None
    stream_block_size: int
    metric_window_size: int
    metric_window_stride: int
    config_dir: Path
    config_path: Path | None
    config_hash: str
    config_data: dict[str, object]
    artifact_provenance: str
    reference_trace: Path | None = None
    failure_analysis_profile: str = "off"
    failure_analysis_max_samples: int = 1000
    failure_analysis_max_bytes: int = 256 * 1024 * 1024
    failure_counterfactual_thresholds: tuple[float, ...] = (0.50, 0.75, 1.00)
    open_set: bool = False
    known_class_split: str | None = None
    ood_ratio: float = 0
    analysis_role: str = "analysis"

    @property
    def run_dir(self) -> Path:
        return self.evidence_dir / self.run_id

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["evidence_dir"] = str(self.evidence_dir)
        result["data_root"] = str(self.data_root)
        result["run_dir"] = str(self.run_dir)
        result["config_dir"] = str(self.config_dir)
        result["config_path"] = str(self.config_path) if self.config_path else None
        result["reference_trace"] = str(self.reference_trace) if self.reference_trace else None
        return result


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _python_executable(path: str | Path) -> str:
    """Keep an explicit PATH command usable while defaulting to absolute Python."""
    text = str(path)
    if "/" not in text and "\\" not in text:
        return text
    return str(_absolute(text))


def _parse_config_scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value[:1] in {'"', "'"}:
        try:
            return json.loads(value) if value.startswith('"') else value[1:-1]
        except (json.JSONDecodeError, IndexError):
            return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_flat_yaml(raw: bytes, path: Path) -> dict[str, object]:
    """Parse the flat scalar YAML subset used by NB-Ramen method configs."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"config must be UTF-8: {path}") from exc
    stripped = text.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"config must contain a mapping: {path}")
        return parsed
    result: dict[str, object] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        if line[:1].isspace() or ":" not in content:
            raise ValueError(f"unsupported non-flat YAML at {path}:{line_number}")
        key, value = content.split(":", 1)
        key = key.strip()
        if not key or key in result:
            raise ValueError(f"invalid or duplicate config key at {path}:{line_number}")
        result[key] = _parse_config_scalar(value)
    return result


def _selected_config(config_dir: Path, dataset: str, method: str) -> tuple[Path | None, str, dict[str, object]]:
    candidates = (config_dir / dataset / f"{method}.yaml", config_dir / "default" / f"{method}.yaml")
    for path in candidates:
        if path.is_file():
            raw = path.read_bytes()
            return path.resolve(), hashlib.sha256(raw).hexdigest()[:CONFIG_HASH_LENGTH], _parse_flat_yaml(raw, path)
    return None, MISSING_CONFIG_HASH, {}


def _seed_token(seed: int) -> str:
    return f"neg{abs(seed)}" if seed < 0 else str(seed)


def _compact_run_id(readable_id: str, identity: dict[str, object]) -> str:
    """Keep short IDs readable and make overlong IDs collision-resistant.

    The readable form remains the compatibility contract for IDs that fit in a
    path segment.  When it does not, retain its descriptive prefix and bind
    the complete semantic identity into a 128-bit SHA-256 prefix.
    """
    if len(readable_id) <= MAX_RUN_ID_LENGTH:
        return readable_id
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    identity_hash = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()[:RUN_ID_IDENTITY_HASH_LENGTH]
    prefix_length = MAX_RUN_ID_LENGTH - len("-h-") - len(identity_hash)
    return readable_id[:prefix_length].rstrip("-") + "-h-" + identity_hash


def make_run_id(
    dataset: str,
    stream_mode: str,
    seed: int,
    method: str,
    *,
    device: str = "auto",
    max_eval_samples: int | None = None,
    stream_block_size: int = 64,
    config_hash: str = MISSING_CONFIG_HASH,
    artifact_provenance: str = "fast",
    data_root: str | Path = "~/data",
    batch_size: int | None = None,
    failure_analysis_profile: str = "off",
    failure_analysis_max_samples: int = 1000,
    failure_analysis_max_bytes: int = 256 * 1024 * 1024,
    failure_counterfactual_thresholds: Sequence[float] = (0.50, 0.75, 1.00),
    open_set: bool = False,
    known_class_split: str = OPEN_SET_SPLIT,
    ood_ratio: float = 0,
    analysis_role: str = "analysis",
) -> str:
    """Return a stable, conservative ID suitable for use as one path segment."""
    thresholds = parse_counterfactual_thresholds(failure_counterfactual_thresholds)
    budget = "full" if max_eval_samples is None else f"n{max_eval_samples}"
    canonical_data_root = str(_absolute(data_root))
    data_root_hash = hashlib.sha256(canonical_data_root.encode("utf-8")).hexdigest()[:CONFIG_HASH_LENGTH]
    if not isinstance(stream_block_size, int) or isinstance(stream_block_size, bool) or stream_block_size <= 0:
        raise ValueError("stream_block_size must be a positive integer")
    if batch_size is not None and (
        not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    default_batch_size = BATCH_SIZE_BY_DATASET.get(dataset)
    analysis_token = "off" if failure_analysis_profile == "off" else hashlib.sha256(
        json.dumps([failure_analysis_profile, failure_analysis_max_samples, failure_analysis_max_bytes,
                    list(thresholds)], separators=(",", ":")).encode()
    ).hexdigest()[:8]
    if analysis_role not in {"analysis", "final"}:
        raise ValueError("analysis_role must be analysis or final")
    if open_set and (known_class_split != OPEN_SET_SPLIT or ood_ratio not in OPEN_SET_OOD_RATIOS):
        raise ValueError("invalid preregistered open-set contract")
    tokens = (dataset, stream_mode, "seed", _seed_token(seed), method, "dev", device, budget,
              *(("bs", batch_size) if batch_size is not None and batch_size != default_batch_size else ()),
              *(("blk", stream_block_size) if stream_block_size != 64 else ()),
              "cfg", config_hash, "prov", artifact_provenance,
              *(("fa", analysis_token) if failure_analysis_profile != "off" else ()),
              *(("os", hashlib.sha256(
                  json.dumps((known_class_split, ood_ratio, analysis_role), separators=(",", ":")).encode()
              ).hexdigest()[:12]) if open_set else ()),
              *(("role", analysis_role) if analysis_role != "analysis" and not open_set else ()),
              "data", data_root_hash)
    normalized = []
    for token in tokens:
        value = "".join(character.lower() if character.isalnum() else "-" for character in str(token))
        value = "-".join(part for part in value.split("-") if part)
        if not value:
            raise ValueError("run ID component cannot be empty")
        normalized.append(value)
    run_id = "-".join(normalized)
    identity = {
        "dataset": dataset,
        "stream_mode": stream_mode,
        "seed": seed,
        "method": method,
        "device": device,
        "max_eval_samples": max_eval_samples,
        "stream_block_size": stream_block_size,
        # Explicitly requesting the dataset default has always been the same
        # planned identity as omitting the option.
        "batch_size_override": (
            batch_size if batch_size is not None and batch_size != default_batch_size else None
        ),
        "config_hash": config_hash,
        "artifact_provenance": artifact_provenance,
        "data_root": canonical_data_root,
        "analysis_role": analysis_role,
        "failure_analysis": (
            None if failure_analysis_profile == "off" else {
                "profile": failure_analysis_profile,
                "max_samples": failure_analysis_max_samples,
                "max_bytes": failure_analysis_max_bytes,
                "counterfactual_thresholds": list(thresholds),
            }
        ),
        "open_set": (
            None if not open_set else {
                "known_class_split": known_class_split,
                "ood_ratio": ood_ratio,
            }
        ),
    }
    return _compact_run_id(run_id, identity)


def build_experiment_matrix(
    datasets: Iterable[str] = DEFAULT_DATASETS,
    streams: Iterable[str] = DEFAULT_STREAMS,
    methods: Iterable[str] = DEFAULT_METHODS,
    seeds: Iterable[int] = (0,),
    evidence_dir: str | Path = REPOSITORY_ROOT / "evidence",
    device: str = "auto",
    max_eval_samples: int | None = None,
    batch_size: int | None = None,
    stream_block_size: int = 64,
    config_dir: str | Path = REPOSITORY_ROOT / "cfg",
    artifact_provenance: str = "fast",
    data_root: str | Path = "~/data",
    failure_analysis_profile: str = "off",
    failure_analysis_max_samples: int = 1000,
    failure_analysis_max_bytes: int = 256 * 1024 * 1024,
    failure_counterfactual_thresholds: Sequence[float] = (0.50, 0.75, 1.00),
    open_set: bool = False,
    known_class_split: str = OPEN_SET_SPLIT,
    ood_ratio: float = 0,
    analysis_role: str = "analysis",
) -> list[ExperimentRun]:
    """Build the grid in execution order, with NoAdapt first in each cell."""
    datasets = tuple(datasets)
    streams = tuple(streams)
    selected_methods = tuple(methods)
    seeds = tuple(seeds)
    unknown_datasets = set(datasets).difference(MODEL_BY_DATASET)
    unknown_streams = set(streams).difference(SUPPORTED_STREAMS)
    unknown_methods = set(selected_methods).difference(SUPPORTED_METHODS)
    if unknown_datasets or unknown_streams or unknown_methods:
        raise ValueError(
            "unsupported matrix value(s): "
            f"datasets={sorted(unknown_datasets)}, streams={sorted(unknown_streams)}, "
            f"methods={sorted(unknown_methods)}"
        )
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("provide at least one integer seed")
    if device not in SUPPORTED_DEVICES:
        raise ValueError("device must be one of " + ", ".join(SUPPORTED_DEVICES))
    if artifact_provenance not in SUPPORTED_ARTIFACT_PROVENANCE:
        raise ValueError("artifact_provenance must be one of " + ", ".join(SUPPORTED_ARTIFACT_PROVENANCE))
    if failure_analysis_profile not in {"off", "trace_v1", "replay_v1"}:
        raise ValueError("failure_analysis_profile must be off, trace_v1, or replay_v1")
    thresholds = parse_counterfactual_thresholds(failure_counterfactual_thresholds)
    if not _is_int(failure_analysis_max_samples, minimum=1) or not _is_int(failure_analysis_max_bytes, minimum=1):
        raise ValueError("failure-analysis limits must be positive integers")
    if max_eval_samples is not None and (
        not isinstance(max_eval_samples, int) or isinstance(max_eval_samples, bool) or max_eval_samples <= 0
    ):
        raise ValueError("max_eval_samples must be a positive integer")
    if batch_size is not None and (
        not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(stream_block_size, int) or isinstance(stream_block_size, bool) or stream_block_size <= 0:
        raise ValueError("stream_block_size must be a positive integer")
    if stream_block_size != 64 and max_eval_samples is None:
        raise ValueError("nondefault stream_block_size requires max_eval_samples")
    if open_set and (datasets != ("CIFAR100C",) or known_class_split != OPEN_SET_SPLIT or ood_ratio not in OPEN_SET_OOD_RATIOS):
        raise ValueError("open-set matrix is fixed to CIFAR100C and preregistered split/ratios")
    if open_set and device == "auto":
        raise ValueError("open-set matrix requires an explicit device")
    if analysis_role not in {"analysis", "final"}:
        raise ValueError("analysis_role must be analysis or final")

    # The paired baseline must execute first even if callers provide methods in
    # another order.  Remaining methods retain their requested relative order.
    # Every adapted run needs a real paired baseline trace, so selecting an
    # adapted subset still schedules NoAdapt first for each stream cell.
    ordered_methods = ("NoAdapt",) + tuple(method for method in selected_methods if method != "NoAdapt")
    root = _absolute(evidence_dir)
    canonical_data_root = _absolute(data_root)
    configs = _absolute(config_dir)
    metric_window = min(50, max_eval_samples) if max_eval_samples is not None else 50
    runs: list[ExperimentRun] = []
    for dataset in datasets:
        dataset_batch_size = BATCH_SIZE_BY_DATASET[dataset] if batch_size is None else batch_size
        for stream_mode in streams:
            for seed in seeds:
                _, baseline_hash, _ = _selected_config(configs, dataset, "NoAdapt")
                baseline_id = make_run_id(
                    dataset, stream_mode, seed, "NoAdapt", device=device,
                    max_eval_samples=max_eval_samples, config_hash=baseline_hash, artifact_provenance=artifact_provenance,
                    data_root=canonical_data_root, stream_block_size=stream_block_size,
                    batch_size=dataset_batch_size,
                    failure_analysis_profile=failure_analysis_profile,
                    failure_analysis_max_samples=failure_analysis_max_samples,
                    failure_analysis_max_bytes=failure_analysis_max_bytes,
                    failure_counterfactual_thresholds=thresholds,
                    open_set=open_set, known_class_split=known_class_split,
                    ood_ratio=ood_ratio, analysis_role=analysis_role,
                )
                baseline_trace = root / baseline_id / "trace.jsonl"
                for method in ordered_methods:
                    config_path, config_hash, config_data = _selected_config(configs, dataset, method)
                    runs.append(ExperimentRun(
                        dataset=dataset,
                        stream_mode=stream_mode,
                        seed=seed,
                        method=method,
                        run_id=make_run_id(
                            dataset, stream_mode, seed, method, device=device,
                            max_eval_samples=max_eval_samples, config_hash=config_hash,
                            artifact_provenance=artifact_provenance,
                            data_root=canonical_data_root, stream_block_size=stream_block_size,
                            batch_size=dataset_batch_size,
                            failure_analysis_profile=failure_analysis_profile,
                            failure_analysis_max_samples=failure_analysis_max_samples,
                            failure_analysis_max_bytes=failure_analysis_max_bytes,
                            failure_counterfactual_thresholds=thresholds,
                            open_set=open_set, known_class_split=known_class_split,
                            ood_ratio=ood_ratio, analysis_role=analysis_role,
                        ),
                        model=MODEL_BY_DATASET[dataset],
                        batch_size=dataset_batch_size,
                        evidence_dir=root,
                        data_root=canonical_data_root,
                        device=device,
                        max_eval_samples=max_eval_samples,
                        stream_block_size=stream_block_size,
                        metric_window_size=metric_window,
                        metric_window_stride=metric_window,
                        config_dir=configs,
                        config_path=config_path,
                        config_hash=config_hash,
                        config_data=config_data,
                        artifact_provenance=artifact_provenance,
                        reference_trace=None if method == "NoAdapt" else baseline_trace,
                        failure_analysis_profile=failure_analysis_profile,
                        failure_analysis_max_samples=failure_analysis_max_samples,
                        failure_analysis_max_bytes=failure_analysis_max_bytes,
                        failure_counterfactual_thresholds=thresholds,
                        open_set=open_set, known_class_split=known_class_split if open_set else None,
                        ood_ratio=ood_ratio, analysis_role=analysis_role,
                    ))
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
        raise ValueError("duplicate generated run ID(s): " + ", ".join(duplicates))
    return runs


# Short alias for use in notebooks and scheduler wrappers.
plan_matrix = build_experiment_matrix


def build_command(
    run: ExperimentRun,
    *,
    python_executable: str | Path = sys.executable,
    data_root: str | Path | None = None,
    device: str | None = None,
    config_dir: str | Path | None = None,
    max_eval_samples: object = _UNSET,
    batch_size: object = _UNSET,
    stream_block_size: object = _UNSET,
) -> list[str]:
    """Build an argv list for one run; callers may pass it to subprocess safely."""
    if device is not None and device != run.device:
        raise ValueError(f"device override contradicts planned identity: {device!r} != {run.device!r}")
    if config_dir is not None and _absolute(config_dir) != run.config_dir:
        raise ValueError("config_dir override contradicts planned identity")
    if max_eval_samples is not _UNSET and max_eval_samples != run.max_eval_samples:
        raise ValueError("max_eval_samples override contradicts planned identity")
    if batch_size is not _UNSET and batch_size != run.batch_size:
        raise ValueError("batch_size override contradicts planned identity")
    if stream_block_size is not _UNSET and stream_block_size != run.stream_block_size:
        raise ValueError("stream_block_size override contradicts planned identity")
    if data_root is not None and _absolute(data_root) != run.data_root:
        raise ValueError("data_root override contradicts planned identity")
    current_path, current_hash, current_data = _selected_config(run.config_dir, run.dataset, run.method)
    if (current_path, current_hash, current_data) != (run.config_path, run.config_hash, run.config_data):
        raise ValueError(f"effective config changed after planning: {run.dataset}/{run.method}")
    command = [
        _python_executable(python_executable), str((REPOSITORY_ROOT / "src" / "main.py").resolve()),
        "--dataset", run.dataset,
        "--model", run.model,
        "--tta_algo", run.method,
        "--tta_mode", "mixed",
        "--batch_size", str(run.batch_size),
        "--seed", str(run.seed),
        "--stream_mode", run.stream_mode,
        "--stream_seed", str(run.seed),
        "--device", run.device,
        "--data_root", str(run.data_root),
        "--config", str(run.config_dir),
        "--evidence_dir", str(run.evidence_dir),
        "--run_id", run.run_id,
        "--save_to", str(run.run_dir / "results.csv"),
        "--metric_window_size", str(run.metric_window_size),
        "--metric_window_stride", str(run.metric_window_stride),
        "--artifact-provenance", run.artifact_provenance,
    ]
    if run.max_eval_samples is not None:
        command.extend(("--max-eval-samples", str(run.max_eval_samples)))
    if run.stream_block_size != 64:
        command.extend(("--stream_block_size", str(run.stream_block_size)))
    if run.reference_trace is not None:
        command.extend(("--reference_trace", str(run.reference_trace)))
    if run.failure_analysis_profile != "off":
        command.extend((
            "--failure-analysis-profile", run.failure_analysis_profile,
            "--failure-analysis-max-samples", str(run.failure_analysis_max_samples),
            "--failure-analysis-max-bytes", str(run.failure_analysis_max_bytes),
            "--failure-counterfactual-thresholds", ",".join(str(value) for value in run.failure_counterfactual_thresholds),
        ))
    # Pass the selected role explicitly for every matrix run.
    command.extend(("--analysis-role", run.analysis_role))
    if run.open_set:
        command.extend(("--open-set", "--known-class-split", str(run.known_class_split),
                        "--ood-ratio", str(run.ood_ratio)))
    return command


def _stream_fingerprint(payload: dict[str, object]) -> str:
    metadata = dict(payload["metadata"])
    metadata.pop("fingerprint", None)
    canonical = {"metadata": metadata, "references": [list(reference) for reference in payload["references"]]}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IncompleteRunError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise IncompleteRunError(f"{label} must be a JSON object: {path}")
    return value


def _require_equal(actual: object, expected: object, field: str, run: ExperimentRun) -> None:
    if actual != expected:
        raise IncompleteRunError(
            f"stale or foreign evidence for {run.run_id}: {field} is {actual!r}, expected {expected!r}"
        )


def _require_fields(value: dict[str, object], fields: Iterable[str], label: str, run: ExperimentRun) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise IncompleteRunError(
            f"{label} missing required field(s) for {run.run_id}: {', '.join(missing)}"
        )


def _is_int(value: object, *, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _is_finite_number(value: object, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        if not math.isfinite(value):
            return False
    except OverflowError:
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _entropy_gate_threshold(run: ExperimentRun) -> float | None:
    """Return the configured gate threshold, validating the gated run contract."""
    if run.method != "EntropyGatedLatentRamen":
        return None
    value = run.config_data.get("max_normalized_entropy")
    if not _is_finite_number(value, minimum=0.0, maximum=1.0):
        raise IncompleteRunError(
            f"EntropyGatedLatentRamen requires max_normalized_entropy in [0, 1]: {run.run_id}"
        )
    return float(value)


def _require_probability(value: object, field: str, run: ExperimentRun) -> float:
    if not _is_finite_number(value, minimum=0.0, maximum=1.0):
        raise IncompleteRunError(f"{field} must be a finite probability: {run.run_id}")
    return float(value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_artifact_evidence(artifacts: object, run: ExperimentRun) -> None:
    if not isinstance(artifacts, dict):
        raise IncompleteRunError(f"manifest.artifacts must be an object: {run.run_id}")
    _require_equal(set(artifacts), {"status", "mode", "model", "dataset"}, "manifest.artifacts fields", run)
    _require_equal(artifacts.get("status"), "verified", "manifest.artifacts.status", run)
    _require_equal(artifacts.get("mode"), run.artifact_provenance, "manifest.artifacts.mode", run)

    model = artifacts.get("model")
    dataset = artifacts.get("dataset")
    if not isinstance(model, dict) or not isinstance(dataset, dict):
        raise IncompleteRunError(f"manifest.artifacts model and dataset must be objects: {run.run_id}")
    resolved = resolve_clip_model(run.model)
    expected_model_fields = {
        "status", "model", "official_name", "url", "expected_sha256", "filename",
        "publisher", "trust", "path", "actual_sha256", "size_bytes",
    }
    _require_equal(set(model), expected_model_fields, "manifest.artifacts.model fields", run)
    _require_equal(model.get("status"), "verified", "manifest.artifacts.model.status", run)
    for field in ("model", "official_name", "url", "expected_sha256", "filename", "publisher", "trust"):
        _require_equal(model.get(field), resolved[field], f"manifest.artifacts.model.{field}", run)
    if not _is_sha256(model.get("actual_sha256")):
        raise IncompleteRunError(f"manifest.artifacts.model.actual_sha256 is malformed: {run.run_id}")
    _require_equal(model.get("actual_sha256"), model.get("expected_sha256"), "manifest.artifacts.model checksum", run)
    if not _is_int(model.get("size_bytes"), minimum=1):
        raise IncompleteRunError(f"manifest.artifacts.model.size_bytes is malformed: {run.run_id}")
    model_path = model.get("path")
    if not isinstance(model_path, str) or not Path(model_path).is_absolute() or Path(model_path).name != resolved["filename"]:
        raise IncompleteRunError(f"manifest.artifacts.model.path is malformed: {run.run_id}")

    expected_dataset_fields = {
        "status", "schema_version", "dataset", "root", "sidecar", "verified_exact",
        "content_algorithm", "root_digest", "sidecar_sha256", "file_count", "acquisition",
    }
    _require_equal(set(dataset), expected_dataset_fields, "manifest.artifacts.dataset fields", run)
    _require_equal(dataset.get("status"), "verified", "manifest.artifacts.dataset.status", run)
    _require_equal(dataset.get("schema_version"), ARTIFACT_SCHEMA_VERSION, "manifest.artifacts.dataset.schema_version", run)
    dataset_key = run.dataset.lower()
    _require_equal(dataset.get("dataset"), dataset_key, "manifest.artifacts.dataset.dataset", run)
    _require_equal(dataset.get("verified_exact"), run.artifact_provenance == "exact", "manifest.artifacts.dataset.verified_exact", run)
    _require_equal(dataset.get("content_algorithm"), "sha256", "manifest.artifacts.dataset.content_algorithm", run)
    for field in ("root_digest", "sidecar_sha256"):
        if not _is_sha256(dataset.get(field)):
            raise IncompleteRunError(f"manifest.artifacts.dataset.{field} is malformed: {run.run_id}")
    if not _is_int(dataset.get("file_count"), minimum=1):
        raise IncompleteRunError(f"manifest.artifacts.dataset.file_count is malformed: {run.run_id}")
    root = dataset.get("root")
    sidecar = dataset.get("sidecar")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise IncompleteRunError(f"manifest.artifacts.dataset.root is malformed: {run.run_id}")
    expected_root = run.data_root / (
        "corruption/CIFAR-100-C" if run.dataset == "CIFAR100C" else "domainbed/domain_net"
    )
    _require_equal(root, str(expected_root), "manifest.artifacts.dataset.root", run)
    expected_sidecar = str(default_sidecar_path(root, dataset_key).absolute())
    _require_equal(sidecar, expected_sidecar, "manifest.artifacts.dataset.sidecar", run)
    expected_acquisition = CIFAR100C_OFFICIAL_ACQUISITION if run.dataset == "CIFAR100C" else {}
    _require_equal(dataset.get("acquisition"), expected_acquisition, "manifest.artifacts.dataset.acquisition", run)


def _validate_summary(
    summary: dict[str, object], manifest: dict[str, object], rows: list[dict[str, object]], run: ExperimentRun,
) -> None:
    _require_fields(summary, SUMMARY_REQUIRED_FIELDS, "summary", run)
    admission_presence = [all(field in row for field in ADMISSION_TRACE_FIELDS) for row in rows]
    profile_presence = [all(field in row for field in RETRIEVAL_PROFILE_TRACE_FIELDS) for row in rows]
    if any(profile_presence) and not all(profile_presence):
        raise IncompleteRunError(f"trace contains mixed retrieval profile availability: {run.run_id}")
    configured_profile = run.config_data.get("retrieval_profile", "off")
    if configured_profile == "causal_sync_v1" and not all(profile_presence):
        raise IncompleteRunError(f"profiled run lacks complete retrieval profile evidence: {run.run_id}")
    if configured_profile != "causal_sync_v1" and any(profile_presence):
        raise IncompleteRunError(f"unprofiled run contains retrieval profile evidence: {run.run_id}")
    if any(admission_presence) and not all(admission_presence):
        raise IncompleteRunError(f"trace contains mixed admission evidence availability: {run.run_id}")
    if all(admission_presence):
        admitted = [row for row in rows if row["admitted_to_memory"]]
        rejected = [row for row in rows if not row["admitted_to_memory"]]
        def pseudo_accuracy(selected):
            return (
                sum(row["admission_prediction"] == row["ground_truth_class"] for row in selected) / len(selected)
                if selected else None
            )
        admitted_accuracy = pseudo_accuracy(admitted)
        expected_admission = {
            "admitted_count": len(admitted),
            "rejected_count": len(rejected),
            "admission_rate": len(admitted) / len(rows),
            "mean_normalized_entropy": sum(row["admission_normalized_entropy"] for row in rows) / len(rows),
            "admitted_pseudo_label_accuracy": admitted_accuracy,
            "rejected_pseudo_label_accuracy": pseudo_accuracy(rejected),
            "admitted_contamination_rate": 1.0 - admitted_accuracy if admitted_accuracy is not None else None,
        }
        _require_equal(summary.get("admission_diagnostics"), expected_admission, "summary.admission_diagnostics", run)
    elif "admission_diagnostics" in summary:
        raise IncompleteRunError(f"summary admission diagnostics without trace evidence: {run.run_id}")
    num_samples = len(rows)
    micro_accuracy = sum(row["correct"] for row in rows) / num_samples
    actual_micro = _require_probability(summary["micro_accuracy"], "summary.micro_accuracy", run)
    if not math.isclose(actual_micro, micro_accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise IncompleteRunError(f"summary.micro_accuracy disagrees with trace: {run.run_id}")

    dataset = manifest["dataset"]
    environments = dataset.get("environments")
    if not isinstance(environments, list) or not environments or any(not isinstance(name, str) for name in environments):
        raise IncompleteRunError(f"manifest.dataset.environments must be a nonempty string list: {run.run_id}")
    counts = summary["domain_sample_counts"]
    accuracies = summary["domain_accuracies"]
    if not isinstance(counts, dict) or not isinstance(accuracies, dict):
        raise IncompleteRunError(f"summary domain metrics must be objects: {run.run_id}")
    _require_equal(set(counts), set(environments), "summary.domain_sample_counts keys", run)
    _require_equal(set(accuracies), set(environments), "summary.domain_accuracies keys", run)
    trace_counts = [0] * len(environments)
    trace_correct = [0] * len(environments)
    for row in rows:
        domain = row["ground_truth_domain"]
        if domain >= len(environments):
            raise IncompleteRunError(f"trace ground_truth_domain exceeds manifest environments: {run.run_id}")
        trace_counts[domain] += 1
        trace_correct[domain] += int(row["correct"])
    active_accuracies = []
    for index, name in enumerate(environments):
        _require_equal(counts[name], trace_counts[index], f"summary.domain_sample_counts.{name}", run)
        expected = trace_correct[index] / trace_counts[index] if trace_counts[index] else None
        actual = accuracies[name]
        if expected is None:
            _require_equal(actual, None, f"summary.domain_accuracies.{name}", run)
        else:
            probability = _require_probability(actual, f"summary.domain_accuracies.{name}", run)
            if not math.isclose(probability, expected, rel_tol=0.0, abs_tol=1e-12):
                raise IncompleteRunError(f"summary domain accuracy disagrees with trace: {name}")
            active_accuracies.append(expected)
    expected_macro = sum(active_accuracies) / len(active_accuracies)
    expected_worst = min(active_accuracies)
    for field, expected in (("macro_domain_accuracy", expected_macro), ("worst_domain_accuracy", expected_worst)):
        actual = _require_probability(summary[field], f"summary.{field}", run)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise IncompleteRunError(f"summary.{field} disagrees with trace: {run.run_id}")

    sliding = summary["sliding_window"]
    if not isinstance(sliding, dict):
        raise IncompleteRunError(f"summary.sliding_window must be an object: {run.run_id}")
    _require_equal(sliding.get("window_size"), run.metric_window_size, "sliding window size", run)
    _require_equal(sliding.get("stride"), run.metric_window_stride, "sliding window stride", run)
    values = sliding.get("values")
    if not isinstance(values, list):
        raise IncompleteRunError(f"summary.sliding_window.values must be a list: {run.run_id}")
    expected_windows = []
    for start in range(0, num_samples - run.metric_window_size + 1, run.metric_window_stride):
        stop = start + run.metric_window_size
        expected_windows.append({
            "start_timestep": start,
            "end_timestep": stop - 1,
            "accuracy": sum(row["correct"] for row in rows[start:stop]) / run.metric_window_size,
        })
    _require_equal(values, expected_windows, "summary.sliding_window.values", run)

    recovery = summary["post_shift_recovery_time"]
    if not isinstance(recovery, dict):
        raise IncompleteRunError(f"summary.post_shift_recovery_time must be an object: {run.run_id}")
    if run.stream_mode in {"block", "recurring", "bursty"}:
        try:
            shifts = domain_shift_recovery_times(
                [row["correct"] for row in rows],
                [row["ground_truth_domain"] for row in rows],
                window_size=run.metric_window_size,
            )
        except (TypeError, ValueError) as exc:
            raise IncompleteRunError(f"cannot recompute post-shift recovery: {run.run_id}") from exc
        expected_recovery = {
            "status": "computed",
            "definition": "full-window recovery within each persistent-domain episode",
            "window_size": run.metric_window_size,
            "shifts": shifts,
        }
    else:
        expected_recovery = {
            "status": "not_applicable",
            "reason": "stream does not define discrete persistent-domain episodes",
        }
    _require_equal(recovery, expected_recovery, "summary.post_shift_recovery_time", run)

    negative = summary["negative_adaptation_rate"]
    if not isinstance(negative, dict):
        raise IncompleteRunError(f"summary.negative_adaptation_rate must be an object: {run.run_id}")
    if run.reference_trace is None:
        expected_negative = {
            "status": "reference_required",
            "reason": "pass --reference_trace from NoAdapt on the identical stream",
        }
    else:
        try:
            expected_negative = compare_trace_negative_adaptation(
                run.run_dir / "trace.jsonl",
                run.reference_trace,
                window_size=run.metric_window_size,
                stride=run.metric_window_stride,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise IncompleteRunError(f"cannot recompute negative adaptation: {run.run_id}") from exc
    _require_equal(negative, expected_negative, "summary.negative_adaptation_rate", run)

    routing = summary["routing_diagnostics"]
    routing_fields = (
        "status", "normalized_mutual_information", "adjusted_rand_index", "context_purity",
        "number_of_discovered_contexts", "assignment_churn_rate",
    )
    if not isinstance(routing, dict):
        raise IncompleteRunError(f"summary.routing_diagnostics must be an object: {run.run_id}")
    _require_fields(routing, routing_fields, "summary.routing_diagnostics", run)
    try:
        expected_routing = asdict(routing_diagnostics(
            [row["ground_truth_domain"] for row in rows],
            [row["inferred_context"] for row in rows],
        ))
    except (TypeError, ValueError) as exc:
        raise IncompleteRunError(f"cannot recompute routing diagnostics: {run.run_id}") from exc
    _require_equal(routing, expected_routing, "summary.routing_diagnostics", run)

    memory_timeline = [row["memory_bytes"] for row in rows]
    memory_availability = {value is not None for value in memory_timeline}
    if len(memory_availability) != 1:
        raise IncompleteRunError(
            f"trace contains mixed memory_bytes availability: {run.run_id}"
        )
    if memory_availability.pop():
        retained_memory = memory_timeline
        expected_method_memory = {
            "status": "computed",
            "definition": (
                "exact bytes retained by the method support memory at the state exposed for each emitted sample; "
                "batch-atomic methods repeat the post-admission batch state"
            ),
            "unit": "bytes",
            "max_retained_bytes": max(retained_memory),
            "final_retained_bytes": retained_memory[-1],
        }
    else:
        expected_method_memory = {
            "status": "unavailable",
            "reason": "method did not expose retained support-memory bytes",
            "unit": "bytes",
            "max_retained_bytes": None,
            "final_retained_bytes": None,
        }
    _require_equal(summary["method_memory"], expected_method_memory, "summary.method_memory", run)

    latencies = [row["latency_ms"] for row in rows]
    total_latency = sum(latencies)
    expected_forward_latency = {
        "status": "computed",
        "definition": "per-sample share of synchronized tta_model forward wall-clock latency; includes adaptation and prediction",
        "unit": "milliseconds",
        "total_ms": total_latency,
        "mean_per_sample_ms": total_latency / len(latencies),
        "median_per_sample_ms": statistics.median(latencies),
    }
    expected_throughput = {
        "status": "computed",
        "definition": "completed samples divided by total synchronized tta_model forward wall-clock latency",
        "unit": "samples_per_second",
        "samples_per_second": len(latencies) * 1000.0 / total_latency if total_latency > 0 else None,
    }
    _require_equal(summary["forward_latency"], expected_forward_latency, "summary.forward_latency", run)
    _require_equal(summary["throughput"], expected_throughput, "summary.throughput", run)
    expected_retrieval_latency = {
        "status": "unavailable",
        "reason": "retrieval is interleaved with causal insertion and adaptation; isolating it would require invasive instrumentation and device synchronization that would perturb the measured path",
    }
    if all(profile_presence):
        if not all(admission_presence):
            raise IncompleteRunError(f"profiled run lacks complete admission evidence: {run.run_id}")
        topk = run.config_data.get("topk")
        capacity = run.config_data.get("max_capacity")
        scope = run.config_data.get("capacity_scope", "per_class_context")
        include_current = run.config_data.get("include_current", True)
        if not _is_int(topk, minimum=1) or not _is_int(capacity, minimum=1) or scope not in {"per_class", "per_class_context"} or not isinstance(include_current, bool):
            raise IncompleteRunError(f"profiled retrieval config is malformed: {run.run_id}")
        buckets: dict[tuple[int, int], list[int]] = {}
        # Replay the method's causal insertions from model-derived diagnostics.
        # Each trace row is one insertion and its timestep is the stable FIFO
        # recency.  This independently derives every profiling counter.
        expected_counters = []
        for timestep, replay_row in enumerate(rows):
            predicted = replay_row["admission_prediction"]
            context = replay_row["inferred_context"]
            if not _is_int(predicted, minimum=0) or not _is_int(context, minimum=0):
                raise IncompleteRunError(f"profiled admission context is malformed: {run.run_id}")
            key = (predicted, context)
            if scope == "per_class":
                class_items = [(item_time, item_key) for item_key, item_times in buckets.items()
                               if item_key[0] == predicted for item_time in item_times]
                if len(class_items) >= capacity:
                    oldest_time, oldest_key = min(class_items)
                    buckets[oldest_key].remove(oldest_time)
                    if not buckets[oldest_key]:
                        del buckets[oldest_key]
            else:
                bucket = buckets.get(key, [])
                if len(bucket) >= capacity:
                    bucket.pop(0)
            buckets.setdefault(key, []).append(timestep)
            live = sum(len(bucket) for bucket in buckets.values())
            eligible_buckets = {
                bucket_key: [item_time for item_time in bucket if include_current or item_time != timestep]
                for bucket_key, bucket in buckets.items() if bucket_key[1] == context
            }
            eligible = sum(len(bucket) for bucket in eligible_buckets.values())
            returned = sum(min(topk, len(bucket)) for bucket in eligible_buckets.values())
            active = sum(bool(bucket) for bucket in eligible_buckets.values())
            expected_counters.append((live, eligible, returned, active))
        for row in rows:
            if row["retrieval_profile"] != "causal_sync_v1":
                raise IncompleteRunError(f"trace retrieval_profile is malformed: {run.run_id}")
            if not _is_finite_number(row["retrieval_elapsed_ms"], minimum=0.0):
                raise IncompleteRunError(f"trace retrieval_elapsed_ms is malformed: {run.run_id}")
            for field in RETRIEVAL_PROFILE_TRACE_FIELDS[2:]:
                if not _is_int(row[field], minimum=0):
                    raise IncompleteRunError(f"trace {field} is malformed: {run.run_id}")
        for row, expected in zip(rows, expected_counters):
            if row["memory_size"] != expected[0]:
                raise IncompleteRunError(f"trace memory_size disagrees with causal replay: {run.run_id}")
            if row["admitted_to_memory"] is not True:
                raise IncompleteRunError(f"profiled LatentRamen trace must admit every item: {run.run_id}")
            actual = tuple(row[field] for field in RETRIEVAL_PROFILE_TRACE_FIELDS[2:])
            if actual != expected:
                raise IncompleteRunError(f"trace retrieval counters disagree with causal replay: {run.run_id}")
        def percentile(data, fraction):
            ordered = sorted(data); position = (len(ordered) - 1) * fraction
            lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        values = [row["retrieval_elapsed_ms"] for row in rows]
        def distribution(field):
            data = [row[field] for row in rows]
            return {"min": min(data), "p50": percentile(data, .5), "p95": percentile(data, .95), "max": max(data)}
        expected_retrieval_latency = {
            "status": "computed", "profile": "causal_sync_v1",
            "definition": "device-synchronized query-only interval after causal insertion; synchronization perturbs execution and is not comparable to ordinary end-to-end latency",
            "unit": "milliseconds", "total_ms": sum(values), "p50_ms": percentile(values, .5),
            "p95_ms": percentile(values, .95), "max_ms": max(values),
            "candidate_count": distribution("retrieval_candidate_count"),
            "eligible_candidate_count": distribution("retrieval_eligible_candidate_count"),
            "returned_support_count": distribution("retrieval_returned_support_count"),
            "active_class_count": distribution("retrieval_active_class_count"),
        }
    _require_equal(summary["retrieval_latency"], expected_retrieval_latency, "summary.retrieval_latency", run)

    peak = summary["peak_device_memory_bytes"]
    if peak is not None and not _is_int(peak, minimum=0):
        raise IncompleteRunError(f"peak_device_memory_bytes is malformed: {run.run_id}")
    device_memory = summary["device_memory"]
    if not isinstance(device_memory, dict) or device_memory.get("status") not in {
        "collected", "sampled", "unavailable", "not_applicable",
    } or not isinstance(device_memory.get("kind"), str):
        raise IncompleteRunError(f"summary.device_memory is malformed: {run.run_id}")
    memory_bytes = device_memory.get("bytes")
    if memory_bytes is not None and not _is_int(memory_bytes, minimum=0):
        raise IncompleteRunError(f"summary.device_memory.bytes is malformed: {run.run_id}")
    if device_memory["kind"] == "exact_cuda_allocator_peak":
        _require_equal(peak, memory_bytes, "peak CUDA device memory", run)
    elif peak is not None:
        raise IncompleteRunError(f"peak_device_memory_bytes is only valid for exact CUDA evidence: {run.run_id}")


def validate_completed_run(run: ExperimentRun) -> dict[str, object]:
    """Strictly validate all evidence needed to regard a run as resumable."""
    if not run.run_dir.is_dir():
        raise IncompleteRunError(f"missing evidence directory: {run.run_dir}")
    current_path, current_hash, current_data = _selected_config(run.config_dir, run.dataset, run.method)
    _require_equal(current_path, run.config_path, "current config path", run)
    _require_equal(current_hash, run.config_hash, "current config hash", run)
    _require_equal(current_data, run.config_data, "current config data", run)
    entropy_gate_threshold = _entropy_gate_threshold(run)

    manifest = _read_json(run.run_dir / "manifest.json", "manifest")
    summary = _read_json(run.run_dir / "summary.json", "summary")
    stream = _read_json(run.run_dir / "stream.json", "stream export")
    trace_path = run.run_dir / "trace.jsonl"
    try:
        if trace_path.stat().st_size <= 0:
            raise IncompleteRunError(f"trace must be nonempty: {trace_path}")
    except OSError as exc:
        raise IncompleteRunError(f"missing trace: {trace_path}") from exc

    _require_equal(manifest.get("schema_version"), 1, "manifest.schema_version", run)
    _require_equal(summary.get("schema_version"), SUMMARY_SCHEMA_VERSION, "summary.schema_version", run)
    for document, name in ((manifest, "manifest"), (summary, "summary")):
        _require_equal(document.get("run_id"), run.run_id, f"{name}.run_id", run)

    args = manifest.get("args")
    if not isinstance(args, dict):
        raise IncompleteRunError(f"manifest args must be an object: {run.run_id}")
    expected_args = {
        "dataset": run.dataset,
        "model": run.model,
        "tta_algo": run.method,
        "tta_mode": "mixed",
        "batch_size": run.batch_size,
        "seed": run.seed,
        "stream_seed": run.seed,
        "max_eval_samples": run.max_eval_samples,
        "stream_block_size": run.stream_block_size,
        "device_request": run.device,
        "metric_window_size": run.metric_window_size,
        "metric_window_stride": run.metric_window_stride,
        "config_path": str(run.config_path) if run.config_path else None,
        "reference_trace": str(run.reference_trace) if run.reference_trace else None,
        "artifact_provenance": run.artifact_provenance,
        "data_root": str(run.data_root),
    }
    expected_args["analysis_role"] = run.analysis_role
    if run.open_set:
        expected_args.update({
            "open_set": True, "known_class_split": run.known_class_split,
            "ood_ratio": run.ood_ratio,
        })
    if run.failure_analysis_profile != "off":
        expected_args.update({
            "failure_analysis_profile": run.failure_analysis_profile,
            "failure_analysis_max_samples": run.failure_analysis_max_samples,
            "failure_analysis_max_bytes": run.failure_analysis_max_bytes,
            "failure_counterfactual_thresholds": list(run.failure_counterfactual_thresholds),
        })
    for key, expected in expected_args.items():
        _require_equal(args.get(key), expected, f"manifest.args.{key}", run)
    _validate_artifact_evidence(manifest.get("artifacts"), run)
    _require_equal(manifest.get("config"), run.config_data, "manifest.config", run)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise IncompleteRunError(f"manifest dataset must be an object: {run.run_id}")
    _require_equal(dataset.get("name"), run.dataset, "manifest.dataset.name", run)
    if run.device != "auto":
        _require_equal(manifest.get("device"), run.device, "manifest.device", run)

    manifest_stream = manifest.get("stream")
    metadata = stream.get("metadata")
    references = stream.get("references")
    if not isinstance(manifest_stream, dict) or not isinstance(metadata, dict) or not isinstance(references, list):
        raise IncompleteRunError(f"invalid stream structures: {run.run_id}")
    _require_equal(metadata.get("format_version"), 1, "stream.metadata.format_version", run)
    _require_equal(metadata.get("mode"), run.stream_mode, "stream.metadata.mode", run)
    _require_equal(metadata.get("seed"), run.seed, "stream.metadata.seed", run)
    _require_equal(metadata.get("block_size"), run.stream_block_size, "stream.metadata.block_size", run)
    _require_equal(manifest_stream, metadata, "manifest.stream", run)
    try:
        computed_fingerprint = _stream_fingerprint(stream)
    except (KeyError, TypeError, ValueError) as exc:
        raise IncompleteRunError(f"cannot verify stream fingerprint: {run.run_id}") from exc
    exported_fingerprint = stream.get("fingerprint") or metadata.get("fingerprint")
    _require_equal(exported_fingerprint, computed_fingerprint, "stream.fingerprint", run)
    _require_equal(metadata.get("fingerprint"), computed_fingerprint, "stream.metadata.fingerprint", run)
    _require_equal(manifest_stream.get("fingerprint"), computed_fingerprint, "manifest.stream.fingerprint", run)
    _require_equal(summary.get("stream_fingerprint"), computed_fingerprint, "summary.stream_fingerprint", run)

    num_samples = summary.get("num_samples")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise IncompleteRunError(f"summary.num_samples must be a positive integer: {run.run_id}")
    _require_equal(len(references), num_samples, "stream reference count", run)
    _require_equal(metadata.get("num_samples"), num_samples, "stream.metadata.num_samples", run)
    budget = metadata.get("evaluation_budget")
    if run.max_eval_samples is None:
        if budget is not None:
            raise IncompleteRunError(f"full run contains cost-limited stream metadata: {run.run_id}")
    else:
        _require_equal(num_samples, run.max_eval_samples, "summary.num_samples budget", run)
        if not isinstance(budget, dict):
            raise IncompleteRunError(f"cost-limited run lacks evaluation budget metadata: {run.run_id}")
        _require_equal(budget.get("retained_sample_count"), num_samples, "retained sample count", run)
        _require_equal(budget.get("cost_limited_evidence"), True, "cost-limited evidence flag", run)

    for index, reference in enumerate(references):
        if not isinstance(reference, list) or len(reference) != 2 \
                or not all(_is_int(value, minimum=0) for value in reference):
            raise IncompleteRunError(f"invalid stream reference at index {index}: {run.run_id}")

    rows: list[dict[str, object]] = []
    failure_analysis_present: bool | None = None
    try:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise IncompleteRunError(f"blank trace row at {trace_path}:{line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("trace row is not an object")
                _require_fields(row, TRACE_REQUIRED_FIELDS, f"trace[{line_number}]", run)
                _require_equal(
                    row.get("schema_version"), TRACE_SCHEMA_VERSION,
                    f"trace[{line_number}].schema_version", run,
                )
                _require_equal(row.get("run_id"), run.run_id, f"trace[{line_number}].run_id", run)
                timestep = row["timestep"]
                if not _is_int(timestep, minimum=0):
                    raise IncompleteRunError(f"trace timestep must be a non-negative integer: row {line_number}")
                _require_equal(timestep, len(rows), f"trace[{line_number}].timestep", run)
                if len(rows) >= len(references):
                    raise IncompleteRunError(f"trace has more rows than the exported stream: {run.run_id}")
                expected_domain, expected_sample = references[len(rows)]
                for field in ("sample_idx", "ground_truth_domain", "prediction"):
                    if not _is_int(row[field], minimum=0):
                        raise IncompleteRunError(f"trace[{line_number}].{field} must be a non-negative integer")
                if run.open_set:
                    open_set = row.get("open_set")
                    if not isinstance(open_set, dict) or set(open_set) != {
                        "original_label", "is_ood", "known_label_or_minus_one", "split_version", "ood_ratio"
                    }:
                        raise IncompleteRunError(f"trace[{line_number}].open_set must be atomic")
                    _require_equal(open_set.get("split_version"), run.known_class_split, "trace open-set split", run)
                    _require_equal(open_set.get("ood_ratio"), run.ood_ratio, "trace open-set ratio", run)
                    _require_equal(row["ground_truth_class"], open_set.get("known_label_or_minus_one"), "trace open-set label", run)
                elif not _is_int(row["ground_truth_class"], minimum=0):
                    raise IncompleteRunError(f"trace[{line_number}].ground_truth_class must be a non-negative integer")
                _require_equal(row["sample_idx"], expected_sample, f"trace[{line_number}].sample_idx", run)
                _require_equal(
                    row["ground_truth_domain"], expected_domain,
                    f"trace[{line_number}].ground_truth_domain", run,
                )
                if not isinstance(row["correct"], bool):
                    raise IncompleteRunError(f"trace[{line_number}].correct must be a boolean")
                expected_correct = row["prediction"] == row["ground_truth_class"]
                _require_equal(row["correct"], expected_correct, f"trace[{line_number}].correct", run)
                if not _is_finite_number(row["predicted_entropy"], minimum=0.0):
                    raise IncompleteRunError(f"trace[{line_number}].predicted_entropy is malformed")
                if not _is_int(row["memory_size"], minimum=0):
                    raise IncompleteRunError(f"trace[{line_number}].memory_size is malformed")
                active_contexts = row["num_active_contexts"]
                if active_contexts is not None and not _is_int(active_contexts, minimum=0):
                    raise IncompleteRunError(f"trace[{line_number}].num_active_contexts is malformed")
                retained_bytes = row["memory_bytes"]
                if retained_bytes is not None and not _is_int(retained_bytes, minimum=0):
                    raise IncompleteRunError(f"trace[{line_number}].memory_bytes is malformed")
                if not _is_finite_number(row["latency_ms"], minimum=0.0):
                    raise IncompleteRunError(f"trace[{line_number}].latency_ms is malformed")
                admission_present = [field in row for field in ADMISSION_TRACE_FIELDS]
                if any(admission_present) and not all(admission_present):
                    raise IncompleteRunError(
                        f"trace[{line_number}] admission fields must be all present or all absent"
                    )
                if all(admission_present):
                    if not _is_int(row["admission_prediction"], minimum=0):
                        raise IncompleteRunError(f"trace[{line_number}].admission_prediction is malformed")
                    if not _is_finite_number(row["admission_normalized_entropy"], minimum=0.0, maximum=1.0):
                        raise IncompleteRunError(
                            f"trace[{line_number}].admission_normalized_entropy is malformed"
                        )
                    if not isinstance(row["admitted_to_memory"], bool):
                        raise IncompleteRunError(f"trace[{line_number}].admitted_to_memory is malformed")
                    if entropy_gate_threshold is not None:
                        expected_admission = row["admission_normalized_entropy"] <= entropy_gate_threshold
                        if row["admitted_to_memory"] is not expected_admission:
                            raise IncompleteRunError(
                                f"trace[{line_number}].admitted_to_memory disagrees with entropy gate"
                            )
                elif entropy_gate_threshold is not None:
                    raise IncompleteRunError(
                        f"EntropyGatedLatentRamen trace lacks complete admission evidence: {run.run_id}"
                    )
                profile_present = [field in row for field in RETRIEVAL_PROFILE_TRACE_FIELDS]
                if any(profile_present) and not all(profile_present):
                    raise IncompleteRunError(
                        f"trace[{line_number}] retrieval profile fields must be all present or all absent"
                    )
                has_failure_analysis = "failure_analysis" in row
                failure = row.get("failure_analysis")
                if has_failure_analysis and (not isinstance(failure, dict) or not failure):
                    raise IncompleteRunError(f"trace[{line_number}].failure_analysis is malformed")
                if isinstance(failure, dict):
                    try:
                        validate_failure_analysis(failure)
                    except ValueError as exc:
                        raise IncompleteRunError(
                            f"trace[{line_number}].failure_analysis is malformed: {exc}"
                        ) from exc
                if failure_analysis_present is None:
                    failure_analysis_present = has_failure_analysis
                elif failure_analysis_present != has_failure_analysis:
                    raise IncompleteRunError(
                        f"trace[{line_number}] failure_analysis fields must be all present or all absent"
                    )
                if run.failure_analysis_profile != "off" and run.method != "NoAdapt" and not has_failure_analysis:
                    raise IncompleteRunError(f"trace[{line_number}] lacks requested failure analysis evidence")
                if run.failure_analysis_profile == "off" and has_failure_analysis:
                    raise IncompleteRunError(f"trace[{line_number}] contains unexpected failure analysis evidence")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise IncompleteRunError(f"invalid trace: {trace_path}") from exc
    _require_equal(len(rows), num_samples, "trace row count", run)
    if run.failure_analysis_profile == "replay_v1" and run.method != "NoAdapt":
        try:
            ReplaySidecarReader(
                run.run_dir / "failure-analysis",
                manifest_sha256=hashlib.sha256((run.run_dir / "manifest.json").read_bytes()).hexdigest(),
                stream_fingerprint=computed_fingerprint,
                source_fingerprint=manifest.get("git", {}).get("source", {}).get("fingerprint"),
                run_id=run.run_id,
            )
        except (OSError, ValueError) as exc:
            raise IncompleteRunError(f"invalid requested replay sidecar: {run.run_id}") from exc
    _validate_summary(summary, manifest, rows, run)
    return {"manifest": manifest, "summary": summary, "stream": stream}


def preflight(
    datasets: Iterable[str], data_root: str | Path, *, deep: bool = True,
) -> list[dict[str, object]]:
    """Validate each selected dataset once before launching scientific runs.

    Matrix execution defaults to semantic validation for the two research
    datasets.  Callers planning on a login node may opt out explicitly, while
    the standalone preflight CLI retains its inexpensive existence-only
    default.
    """
    return [
        validate_dataset_layout(_absolute(data_root), dataset, deep=deep)
        for dataset in dict.fromkeys(datasets)
    ]


def execute_matrix(
    runs: Sequence[ExperimentRun],
    *,
    python_executable: str | Path = sys.executable,
    data_root: str | Path | None = None,
    resume: bool = False,
    runner=subprocess.run,
) -> list[dict[str, object]]:
    """Preflight and execute in order. A failing model command raises immediately."""
    if not runs:
        return []
    selected_data_root = runs[0].data_root if data_root is None else _absolute(data_root)
    for run in runs:
        if run.data_root != selected_data_root:
            raise ValueError("data_root override contradicts planned identity")
    if any(run.device == "auto" for run in runs):
        raise ValueError(
            "scientific matrix execution requires an explicit device; "
            "re-plan with --device cpu, --device mps, or --device cuda"
        )
    checks = preflight((run.dataset for run in runs), selected_data_root)
    failed = [check["dataset"] for check in checks if not check["valid"]]
    if failed:
        raise RuntimeError("dataset preflight failed: " + ", ".join(failed))

    outcomes = []
    baselines = {run.run_dir / "trace.jsonl": run for run in runs if run.method == "NoAdapt"}
    for run in runs:
        baseline_evidence = None
        if run.reference_trace is not None:
            baseline = baselines.get(run.reference_trace)
            if baseline is None:
                raise IncompleteRunError(f"paired NoAdapt run is absent from plan: {run.reference_trace}")
            for field in (
                "dataset", "stream_mode", "seed", "model", "batch_size", "device",
                "max_eval_samples", "stream_block_size", "metric_window_size", "metric_window_stride", "config_dir",
                "artifact_provenance",
                "data_root",
            ):
                _require_equal(getattr(baseline, field), getattr(run, field), f"paired baseline {field}", run)
            baseline_evidence = validate_completed_run(baseline)
        if resume and run.run_dir.exists():
            evidence = validate_completed_run(run)
            if baseline_evidence is not None:
                _require_equal(
                    evidence["summary"].get("stream_fingerprint"),
                    baseline_evidence["summary"].get("stream_fingerprint"),
                    "paired baseline stream fingerprint",
                    run,
                )
            outcomes.append({"run_id": run.run_id, "status": "skipped"})
            continue
        run.run_dir.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(run, python_executable=python_executable, data_root=selected_data_root)
        runner(command, cwd=str(REPOSITORY_ROOT), check=True)
        outcomes.append({"run_id": run.run_id, "status": "executed"})
    return outcomes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DEFAULT_DATASETS, action="append")
    parser.add_argument("--stream", choices=SUPPORTED_STREAMS, action="append")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, action="append")
    parser.add_argument("--seed", type=int, action="append", help="Repeat for multiple deterministic seeds.")
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument("--data-root", default="~/data")
    parser.add_argument("--evidence-dir", default=REPOSITORY_ROOT / "evidence")
    parser.add_argument("--config-dir", default=REPOSITORY_ROOT / "cfg")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--batch-size", type=int,
                        help="Evaluator batch size; omit to use the per-dataset default.")
    parser.add_argument("--stream-block-size", type=int, default=64)
    parser.add_argument("--artifact-provenance", choices=SUPPORTED_ARTIFACT_PROVENANCE, default="fast")
    parser.add_argument("--failure-analysis-profile", choices=("off", "trace_v1", "replay_v1"), default="off")
    parser.add_argument("--failure-analysis-max-samples", type=int, default=1000)
    parser.add_argument("--failure-analysis-max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--failure-counterfactual-thresholds", default="0.50,0.75,1.00")
    parser.add_argument("--open-set", action="store_true")
    parser.add_argument("--known-class-split", default=OPEN_SET_SPLIT)
    parser.add_argument("--ood-ratio", type=float, choices=OPEN_SET_OOD_RATIOS, default=0)
    parser.add_argument("--analysis-role", choices=("analysis", "final"), default="analysis")
    parser.add_argument("--execute", action="store_true", help="Run commands; planning JSON is the default.")
    parser.add_argument("--resume", action="store_true", help="Skip only runs with manifest.json and summary.json.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.execute or args.resume) and args.device == "auto":
        parser.error(
            "--execute/--resume requires an explicit --device cpu, --device mps, or --device cuda"
        )
    try:
        failure_thresholds = parse_counterfactual_thresholds(args.failure_counterfactual_thresholds)
    except ValueError as exc:
        parser.error(str(exc))
    runs = build_experiment_matrix(
        datasets=args.dataset or DEFAULT_DATASETS,
        streams=args.stream or DEFAULT_STREAMS,
        methods=args.method or DEFAULT_METHODS,
        seeds=args.seed or (0,),
        evidence_dir=args.evidence_dir,
        device=args.device,
        max_eval_samples=args.max_eval_samples,
        batch_size=args.batch_size,
        stream_block_size=args.stream_block_size,
        config_dir=args.config_dir,
        artifact_provenance=args.artifact_provenance,
        data_root=args.data_root,
        failure_analysis_profile=args.failure_analysis_profile,
        failure_analysis_max_samples=args.failure_analysis_max_samples,
        failure_analysis_max_bytes=args.failure_analysis_max_bytes,
        failure_counterfactual_thresholds=failure_thresholds,
        open_set=args.open_set, known_class_split=args.known_class_split,
        ood_ratio=args.ood_ratio, analysis_role=args.analysis_role,
    )
    commands = [build_command(
        run, python_executable=args.python_executable, data_root=args.data_root,
    ) for run in runs]
    payload: dict[str, object] = {
        "execute": args.execute,
        "runs": [run.to_dict() for run in runs],
        "commands": commands,
    }
    if args.execute:
        payload["outcomes"] = execute_matrix(
            runs, python_executable=args.python_executable, data_root=args.data_root, resume=args.resume,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
