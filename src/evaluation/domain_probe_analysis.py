"""Deterministic, evaluator-only linear representation probes.

Feature exports are accepted only after their provenance is bound to an
immutable evaluation stream.  This module deliberately has no model imports:
it analyses post-hoc frozen features and never participates in adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPORT_SCHEMA_VERSION = 1
_STATES = frozenset({"computed", "insufficient", "unavailable"})
_IDENTITY = ("timestep", "sample_idx", "ground_truth_domain", "ground_truth_class")
_BINDINGS = ("stream_fingerprint", "source_fingerprint", "model_fingerprint", "model_artifact_fingerprint")


def _state(status: str, **values: Any) -> dict[str, Any]:
    if status not in _STATES:
        raise ValueError("invalid analysis state")
    return {"status": status, **values}


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def validate_feature_artifact_metadata(metadata: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate a post-hoc feature-export manifest and optional expected bindings.

    The accepted aliases make this useful with existing evidence manifests while
    still requiring the semantic bindings needed to safely join feature rows.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("feature artifact metadata must be a mapping")
    stream = metadata.get("stream_fingerprint")
    source = metadata.get("source_fingerprint")
    model = metadata.get("model_fingerprint", metadata.get("model_artifact_fingerprint"))
    dimension = metadata.get("feature_dim", metadata.get("dimension"))
    dtype = metadata.get("feature_dtype", metadata.get("dtype"))
    split = metadata.get("split_role", metadata.get("split_roles"))
    missing = [name for name, value in (("stream_fingerprint", stream), ("source_fingerprint", source),
                                         ("model_fingerprint", model), ("feature_dim", dimension),
                                         ("feature_dtype", dtype), ("split_role", split)) if value is None]
    if missing:
        raise ValueError("feature artifact metadata missing: " + ", ".join(missing))
    if not isinstance(stream, str) or not stream or not isinstance(source, str) or not source or not isinstance(model, str) or not model:
        raise ValueError("feature artifact fingerprints must be non-empty strings")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 or not isinstance(dtype, str) or not dtype:
        raise ValueError("feature dimension/dtype is invalid")
    if not isinstance(split, (str, Mapping)):
        raise ValueError("feature split role must be a string or per-row mapping")
    if expected:
        for name, value in expected.items():
            actual = metadata.get(name)
            if actual is None and name == "model_fingerprint":
                actual = model
            if actual != value:
                raise ValueError(f"feature artifact {name} binding mismatch")
    return {"stream_fingerprint": stream, "source_fingerprint": source, "model_fingerprint": model,
            "feature_dim": dimension, "feature_dtype": dtype, "split_role": split}


def bind_feature_artifact(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate row identity/shape and return a safe immutable-analysis binding."""
    validated = validate_feature_artifact_metadata(metadata, expected=expected)
    identities: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("feature rows must be mappings")
        missing = [name for name in _IDENTITY if name not in row]
        if missing:
            raise ValueError("feature row missing identity: " + ", ".join(missing))
        identity = tuple(row[name] for name in _IDENTITY)
        if identity in identities:
            raise ValueError(f"duplicate feature sample identity at row {index}")
        identities.add(identity)
        feature = row.get("feature", row.get("features"))
        if not isinstance(feature, Sequence) or isinstance(feature, (str, bytes)) or len(feature) != validated["feature_dim"]:
            raise ValueError("feature row dimension does not match metadata")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in feature):
            raise ValueError("feature values must be finite numbers")
        if "split_role" not in row and not isinstance(validated["split_role"], str):
            raise ValueError("per-row feature artifact needs split_role")
    return {"status": "computed", "count": len(rows), "identity_fields": list(_IDENTITY), "metadata": validated}


def _split_roles(rows: Sequence[Mapping[str, Any]], *, seed: int) -> list[str]:
    allowed = {"train", "validation", "test"}
    explicit = [row.get("split_role") for row in rows]
    if any(role is not None for role in explicit):
        if not all(isinstance(role, str) and role in allowed for role in explicit):
            raise ValueError("split_role must be train, validation, or test for every row")
        return list(explicit)
    # Deterministic fallback is intentionally based on row identity, never input order.
    result = []
    for row in rows:
        digest = hashlib.sha256((str(seed) + _key(tuple(row.get(k) for k in _IDENTITY))).encode()).digest()[0]
        result.append("train" if digest < 153 else "validation" if digest < 204 else "test")
    return result


def deterministic_linear_probe(features: Sequence[Sequence[float]], labels: Sequence[Any], split_roles: Sequence[str], *, seed: int = 0, epochs: int = 400, learning_rate: float = 0.1, l2: float = 1e-4) -> dict[str, Any]:
    """Train multinomial logistic regression with train-only normalization.

    Full-batch gradient descent and a fixed zero initialization make results
    deterministic across runs. Validation is reported only; it is never used
    for fitting or early stopping.
    """
    if len(features) != len(labels) or len(labels) != len(split_roles):
        raise ValueError("features, labels, and split_roles must have equal length")
    if not features:
        return _state("insufficient", count=0, reason="no feature rows")
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be rectangular numeric vectors") from exc
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("features must be finite non-empty vectors")
    if not isinstance(seed, int) or isinstance(seed, bool) or epochs <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid deterministic probe parameters")
    classes = sorted({_key(label) for label in labels})
    label_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([label_index[_key(label)] for label in labels], dtype=np.int64)
    roles = np.asarray(split_roles)
    if not set(roles).issubset({"train", "validation", "test"}):
        raise ValueError("invalid split role")
    train = roles == "train"
    if train.sum() < 2 or len(np.unique(y[train])) < 2:
        return _state("insufficient", count=int(train.sum()), reason="training split requires at least two classes")
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    weight = np.zeros((z.shape[1], len(classes)), dtype=np.float64)
    bias = np.zeros(len(classes), dtype=np.float64)
    target = np.eye(len(classes))[y[train]]
    for _ in range(epochs):
        logits = z[train] @ weight + bias
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits); probability /= probability.sum(axis=1, keepdims=True)
        residual = (probability - target) / train.sum()
        weight -= learning_rate * (z[train].T @ residual + l2 * weight)
        bias -= learning_rate * residual.sum(axis=0)
    prediction = (z @ weight + bias).argmax(axis=1)
    metrics: dict[str, Any] = {}
    for role in ("train", "validation", "test"):
        mask = roles == role
        metrics[role] = _state("insufficient", count=0, reason="empty split") if not mask.any() else _state(
            "computed", count=int(mask.sum()), accuracy=float((prediction[mask] == y[mask]).mean()))
    return _state("computed", count=len(labels), classes=[json.loads(item) for item in classes], seed=seed,
                  feature_dim=int(x.shape[1]), train_normalization={"mean": mean.tolist(), "scale": scale.tolist()}, splits=metrics)


def representation_probe_report(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, seed: int = 0, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    binding = bind_feature_artifact(rows, metadata, expected=expected)
    features = [row.get("feature", row.get("features")) for row in rows]
    roles = _split_roles(rows, seed=seed)
    domain = deterministic_linear_probe(features, [row["ground_truth_domain"] for row in rows], roles, seed=seed)
    classes = deterministic_linear_probe(features, [row["ground_truth_class"] for row in rows], roles, seed=seed)
    conditioned: dict[str, Any] = {}
    for value in sorted({row["ground_truth_class"] for row in rows}, key=_key):
        indices = [i for i, row in enumerate(rows) if row["ground_truth_class"] == value]
        conditioned[_key(value)] = deterministic_linear_probe([features[i] for i in indices], [rows[i]["ground_truth_domain"] for i in indices], [roles[i] for i in indices], seed=seed)
    return {"schema_version": REPORT_SCHEMA_VERSION, "status": "computed", "binding": binding,
            "split_seed": seed, "feature_to_domain": domain, "feature_to_class": classes,
            "class_conditioned_feature_to_domain": conditioned}


# Descriptive aliases keep call sites readable without introducing another API.
validate_feature_binding = validate_feature_artifact_metadata
run_representation_probes = representation_probe_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic post-hoc representation probes")
    parser.add_argument("--input", required=True); parser.add_argument("--output"); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8")); report = representation_probe_report(payload["rows"], payload["metadata"], seed=args.seed)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        report = {"schema_version": REPORT_SCHEMA_VERSION, "status": "invalid", "error": str(exc)}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output: Path(args.output).write_text(encoded, encoding="utf-8")
    else: sys.stdout.write(encoded)
    return 0 if report["status"] != "invalid" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
