"""Bounded, checksummed replay artifacts for failure analysis.

The sidecar is deliberately separate from the ordinary JSONL trace: it may
contain exact model vectors, whereas traces remain compact and human-readable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

SCHEMA_VERSION = "replay_v1"
FILES = ("items.jsonl", "queries.jsonl", "features.bin", "gradients.bin")
PREREGISTERED_COUNTERFACTUAL_THRESHOLDS = (0.50, 0.75, 1.00)


class ReplayArtifactError(ValueError):
    pass


def parse_counterfactual_thresholds(value: str | Iterable[float]) -> tuple[float, ...]:
    """Parse the immutable, preregistered consensus thresholds."""
    raw = value.split(",") if isinstance(value, str) else value
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ReplayArtifactError("counterfactual thresholds must be comma-separated numbers") from exc
    if not values or any(not math.isfinite(item) or item < 0 or item > 1 for item in values):
        raise ReplayArtifactError("counterfactual thresholds must be finite values in [0, 1]")
    if len(set(values)) != len(values):
        raise ReplayArtifactError("counterfactual thresholds must be unique")
    if values != PREREGISTERED_COUNTERFACTUAL_THRESHOLDS:
        raise ReplayArtifactError(
            "counterfactual thresholds must equal the preregistered tuple "
            "(0.50, 0.75, 1.00)"
        )
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReplaySidecarWriter:
    """Append-only writer which never presents a bounded partial sidecar as complete."""
    def __init__(self, directory: str | Path, *, run_id: str, manifest_sha256: str,
                 stream_fingerprint: str, source_fingerprint: str | None, config: Mapping[str, Any],
                 max_samples: int, max_bytes: int) -> None:
        if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples <= 0:
            raise ReplayArtifactError("failure-analysis max samples must be a positive integer")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ReplayArtifactError("failure-analysis max bytes must be a positive integer")
        if not _sha(manifest_sha256) or not _sha(stream_fingerprint):
            raise ReplayArtifactError("sidecar requires SHA-256 manifest and stream bindings")
        if source_fingerprint is not None and not _sha(source_fingerprint):
            raise ReplayArtifactError("source fingerprint must be SHA-256 when available")
        if not isinstance(config, Mapping):
            raise ReplayArtifactError("sidecar config must be a mapping")
        try:
            parse_counterfactual_thresholds(config.get("counterfactual_thresholds", ()))
        except ReplayArtifactError as exc:
            raise ReplayArtifactError("sidecar config has invalid counterfactual thresholds: " + str(exc)) from exc
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.run_id, self.manifest_sha256, self.stream_fingerprint = run_id, manifest_sha256, stream_fingerprint
        self.source_fingerprint, self.config = source_fingerprint, _normalise(config)
        self.max_samples, self.max_bytes, self.count, self.status = max_samples, max_bytes, 0, "interrupted"
        self._handles = {name: (self.directory / name).open("xb") for name in FILES}
        self._item_ids: set[tuple[Any, Any]] = set()
        self._item_count = 0
        self._query_count = 0
        self._tensor_formats: set[tuple[str, tuple[int, ...]]] = set()
        # Make an abruptly terminated run distinguishable from an absent
        # sidecar before the first sample is attempted.
        (self.directory / "metadata.json").write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id, "profile": SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256, "stream_fingerprint": self.stream_fingerprint,
            "source_fingerprint": self.source_fingerprint, "config": self.config,
            "limits": {"max_samples": self.max_samples, "max_bytes": self.max_bytes},
            "status": "interrupted",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _used(self) -> int:
        return sum(handle.tell() for handle in self._handles.values())

    def _write_json(self, name: str, value: Mapping[str, Any]) -> None:
        self._handles[name].write(self._encoded_json(value))

    @staticmethod
    def _encoded_json(value: Mapping[str, Any]) -> bytes:
        return (json.dumps(_normalise(value), sort_keys=True, separators=(",", ":")) + "\n").encode()

    @staticmethod
    def _tensor_payload(tensor: torch.Tensor) -> tuple[torch.Tensor, bytes]:
        if not isinstance(tensor, torch.Tensor):
            raise ReplayArtifactError("replay vector must be a torch.Tensor")
        value = tensor.detach().cpu().contiguous()
        if value.dtype == torch.bfloat16:
            raw = value.view(torch.uint8).numpy().tobytes()
        else:
            raw = value.numpy().tobytes(order="C")
        return value, raw

    def _planned_tensor(self, name: str, tensor: torch.Tensor, offset: int) -> tuple[dict[str, Any], bytes]:
        value, raw = self._tensor_payload(tensor)
        return ({"offset": offset, "length": len(raw), "shape": list(value.shape), "dtype": str(value.dtype)}, raw)

    def _write_tensor_bytes(self, name: str, raw: bytes, descriptor: dict[str, Any]) -> None:
        handle = self._handles[name]
        if handle.tell() != descriptor["offset"]:
            raise ReplayArtifactError("replay binary offset changed before commit")
        handle.write(raw)
        self._tensor_formats.add((str(descriptor["dtype"]), tuple(descriptor["shape"])))

    def write(self, *, items: Iterable[Mapping[str, Any]] = (), query: Mapping[str, Any] | None = None,
              gradients: Iterable[torch.Tensor] = ()) -> bool:
        """Write one query. Item dictionaries may include ``feature``/``gradient`` tensors."""
        if self.status != "interrupted":
            return False
        item_rows: list[dict[str, Any]] = []
        binary_writes: list[tuple[str, bytes, dict[str, Any]]] = []
        pending_item_ids: set[tuple[Any, Any]] = set()
        feature_offset = self._handles["features.bin"].tell()
        gradient_offset = self._handles["gradients.bin"].tell()
        for item in items:
            row = dict(item)
            item_id = row.get("item_id")
            segment_index = row.setdefault("segment_index", 0)
            identity = (segment_index, item_id)
            if item_id is not None and (identity in self._item_ids or identity in pending_item_ids):
                continue
            feature = row.pop("feature", row.pop("vector", None))
            gradient = row.pop("gradient", None)
            if feature is not None:
                row["feature"], raw = self._planned_tensor("features.bin", feature, feature_offset)
                feature_offset += len(raw)
                binary_writes.append(("features.bin", raw, row["feature"]))
            if gradient is not None:
                row["gradient"], raw = self._planned_tensor("gradients.bin", gradient, gradient_offset)
                gradient_offset += len(raw)
                binary_writes.append(("gradients.bin", raw, row["gradient"]))
            item_rows.append(row)
            if item_id is not None:
                pending_item_ids.add(identity)
        gradient_rows = []
        for value in gradients:
            descriptor, raw = self._planned_tensor("gradients.bin", value, gradient_offset)
            gradient_offset += len(raw)
            gradient_rows.append(descriptor)
            binary_writes.append(("gradients.bin", raw, descriptor))
        query_row = None
        if query is not None:
            query_row = dict(query)
            query_row.setdefault("segment_index", 0)
            if gradient_rows:
                query_row["gradients"] = gradient_rows
        pending_json_bytes = sum(len(self._encoded_json(row)) for row in item_rows)
        if query_row is not None:
            pending_json_bytes += len(self._encoded_json(query_row))
        pending_binary_bytes = sum(len(raw) for _, raw, _ in binary_writes)
        if self.count >= self.max_samples or self._used() + pending_json_bytes + pending_binary_bytes > self.max_bytes:
            self.status = "insufficient"
            return False
        # The preflight above occurs before any handle is mutated: a rejected
        # query leaves every artifact file byte-for-byte unchanged.
        for name, raw, descriptor in binary_writes:
            self._write_tensor_bytes(name, raw, descriptor)
        for row in item_rows:
            self._write_json("items.jsonl", row)
        self._item_count += len(item_rows)
        if query_row is not None:
            self._write_json("queries.jsonl", query_row)
            self._query_count += 1
        self.count += 1
        self._item_ids.update(pending_item_ids)
        return True

    def close(self, *, completed: bool = True) -> dict[str, Any]:
        if self.status == "interrupted" and completed:
            self.status = "completed"
        for handle in self._handles.values():
            if not handle.closed:
                handle.flush(); os.fsync(handle.fileno()); handle.close()
        files = {name: {"bytes": (self.directory / name).stat().st_size,
                        "sha256": sha256_file(self.directory / name)} for name in FILES}
        metadata = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "profile": SCHEMA_VERSION,
                    "manifest_sha256": self.manifest_sha256, "stream_fingerprint": self.stream_fingerprint,
                    "source_fingerprint": self.source_fingerprint, "config": self.config,
                    "limits": {"max_samples": self.max_samples, "max_bytes": self.max_bytes},
                    "status": self.status, "row_counts": {"items": self._item_count, "queries": self._query_count},
                    "dimensions_dtypes": [{"dtype": dtype, "shape": list(shape)}
                                          for dtype, shape in sorted(self._tensor_formats)], "files": files}
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata


class ReplaySidecarReader:
    def __init__(self, directory: str | Path, *, manifest_sha256: str | None = None,
                 stream_fingerprint: str | None = None, source_fingerprint: str | None = None,
                 run_id: str | None = None, allow_insufficient: bool = False) -> None:
        self.directory = Path(directory)
        try:
            self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayArtifactError("invalid replay metadata") from exc
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ReplayArtifactError("unsupported replay sidecar schema")
        config = self.metadata.get("config")
        if not isinstance(config, Mapping):
            raise ReplayArtifactError("replay sidecar config is missing or malformed")
        try:
            parse_counterfactual_thresholds(config.get("counterfactual_thresholds", ()))
        except ReplayArtifactError as exc:
            raise ReplayArtifactError("replay sidecar has noncanonical counterfactual thresholds: " + str(exc)) from exc
        if not _sha(self.metadata.get("manifest_sha256")) or not _sha(self.metadata.get("stream_fingerprint")):
            raise ReplayArtifactError("replay sidecar has malformed bindings")
        stored_source = self.metadata.get("source_fingerprint")
        if stored_source is not None and not _sha(stored_source):
            raise ReplayArtifactError("replay sidecar has malformed source binding")
        if run_id is not None and self.metadata.get("run_id") != run_id:
            raise ReplayArtifactError("replay sidecar run binding mismatch")
        allowed_statuses = {"completed", "insufficient"} if allow_insufficient else {"completed"}
        if self.metadata.get("status") not in allowed_statuses:
            raise ReplayArtifactError("replay sidecar is not closed")
        for name in FILES:
            expected = self.metadata.get("files", {}).get(name, {})
            path = self.directory / name
            try:
                valid = path.stat().st_size == expected.get("bytes") and sha256_file(path) == expected.get("sha256")
            except OSError as exc:
                raise ReplayArtifactError(f"replay sidecar file is missing: {name}") from exc
            if not valid:
                raise ReplayArtifactError(f"replay sidecar checksum mismatch: {name}")
        for name, actual, expected in (("manifest", self.metadata.get("manifest_sha256"), manifest_sha256),
                                       ("stream", self.metadata.get("stream_fingerprint"), stream_fingerprint),
                                       ("source", self.metadata.get("source_fingerprint"), source_fingerprint)):
            if expected is not None and actual != expected:
                raise ReplayArtifactError(f"replay sidecar {name} binding mismatch")
        counts = self.metadata.get("row_counts")
        if not isinstance(counts, Mapping):
            raise ReplayArtifactError("replay sidecar row counts are missing")
        parsed_rows = {}
        for name in ("items", "queries"):
            rows = self.rows(name)
            parsed_rows[name] = rows
            if counts.get(name) != len(rows):
                raise ReplayArtifactError(f"replay sidecar {name} row count mismatch")
        limits = self.metadata.get("limits", {})
        total_bytes = sum(self.metadata["files"][name]["bytes"] for name in FILES)
        if (
            counts.get("queries", 0) > limits.get("max_samples", -1)
            or total_bytes > limits.get("max_bytes", -1)
        ):
            raise ReplayArtifactError("replay sidecar exceeds its declared limits")
        self._validate_references(parsed_rows["items"], parsed_rows["queries"])

    def _validate_references(self, items: list[dict[str, Any]], queries: list[dict[str, Any]]) -> None:
        item_keys: set[tuple[Any, Any]] = set()
        for row in items:
            for name, descriptor in (("feature", row.get("feature")), ("gradient", row.get("gradient"))):
                if descriptor is not None:
                    self.tensor(descriptor, kind=name)
            if "item_id" not in row:
                continue
            key = (row.get("segment_index", 0), row["item_id"])
            if key in item_keys:
                raise ReplayArtifactError("replay sidecar has duplicate segment-scoped item identity")
            item_keys.add(key)
        for row in queries:
            segment = row.get("segment_index", 0)
            item_id = row.get("item_id")
            if item_id is not None and (segment, item_id) not in item_keys:
                raise ReplayArtifactError("replay query item does not resolve within its segment")
            candidates = row.get("legal_candidates", [])
            if not isinstance(candidates, list):
                raise ReplayArtifactError("replay query candidates must be a list")
            for candidate in candidates:
                candidate_id = candidate.get("item_id") if isinstance(candidate, Mapping) else candidate
                candidate_segment = candidate.get("segment_index", segment) if isinstance(candidate, Mapping) else segment
                if (candidate_segment, candidate_id) not in item_keys:
                    raise ReplayArtifactError("replay query candidate does not resolve within its segment")
            for descriptor in row.get("gradients", []):
                self.tensor(descriptor, kind="gradient")

    def rows(self, name: str) -> list[dict[str, Any]]:
        if name not in {"items", "queries"}:
            raise ReplayArtifactError("rows are available only for items or queries")
        try:
            rows = [json.loads(line) for line in (self.directory / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReplayArtifactError(f"invalid replay {name} rows") from exc
        if any(not isinstance(row, dict) for row in rows):
            raise ReplayArtifactError(f"invalid replay {name} row")
        return rows

    def tensor(self, descriptor: Mapping[str, Any], *, kind: str = "feature") -> torch.Tensor:
        """Return a bit-exact tensor slice described by an item/query row."""
        filename = "features.bin" if kind == "feature" else "gradients.bin"
        try:
            offset, length = int(descriptor["offset"]), int(descriptor["length"])
            dtype = getattr(torch, str(descriptor["dtype"]).removeprefix("torch."))
            shape = tuple(int(value) for value in descriptor["shape"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ReplayArtifactError("invalid tensor descriptor") from exc
        if not isinstance(dtype, torch.dtype) or offset < 0 or length < 0:
            raise ReplayArtifactError("invalid tensor descriptor")
        with (self.directory / filename).open("rb") as handle:
            handle.seek(offset); raw = handle.read(length)
        if len(raw) != length:
            raise ReplayArtifactError("tensor descriptor exceeds sidecar binary")
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if expected != length:
            raise ReplayArtifactError("tensor descriptor shape/dtype does not match byte length")
        # clone detaches the returned tensor from the temporary bytes buffer.
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _normalise(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda item: item.item() if hasattr(item, "item") else str(item), sort_keys=True))
