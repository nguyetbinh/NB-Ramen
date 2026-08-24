"""Hardware provenance and conservative accelerator-memory evidence."""

from __future__ import annotations

import platform
from typing import Any


def _call(value: Any, *args: Any) -> Any:
    """Call an optional runtime API, returning ``None`` when it is unavailable."""
    if not callable(value):
        return None
    try:
        return value(*args)
    except (AttributeError, RuntimeError, TypeError):
        return None


def _torch(torch_module: Any = None) -> Any:
    if torch_module is not None:
        return torch_module
    try:
        import torch
    except ImportError:
        return None
    return torch


def _device_type(device: Any) -> str:
    return str(getattr(device, "type", device)).split(":", 1)[0]


def _device_index(device: Any) -> int | None:
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    text = str(device)
    if ":" in text:
        try:
            return int(text.rsplit(":", 1)[1])
        except ValueError:
            pass
    return 0


def collect_hardware_evidence(
    requested_device: Any,
    resolved_device: Any = None,
    *,
    torch_module: Any = None,
) -> dict[str, Any]:
    """Return JSON-safe facts about the resolved runtime device.

    Optional Torch APIs are deliberately best-effort: this evidence must not
    make a CPU-only build or a partially configured accelerator fail.
    """
    torch = _torch(torch_module)
    resolved_device = requested_device if resolved_device is None else resolved_device
    evidence: dict[str, Any] = {
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "platform": {"machine": platform.machine(), "processor": platform.processor()},
        "torch": {"version": getattr(torch, "__version__", None) if torch else None,
                  "num_threads": _call(getattr(torch, "get_num_threads", None)) if torch else None,
                  "num_interop_threads": _call(getattr(torch, "get_num_interop_threads", None)) if torch else None},
    }
    if torch is None:
        return evidence

    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(_call(getattr(cuda, "is_available", None)))
    cuda_evidence: dict[str, Any] = {
        "available": cuda_available,
        "device_count": _call(getattr(cuda, "device_count", None)),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": _call(getattr(getattr(torch, "backends", None), "cudnn", None) and getattr(torch.backends.cudnn, "version", None)),
    }
    if _device_type(resolved_device) == "cuda" and cuda_available:
        index = _device_index(resolved_device)
        properties = _call(getattr(cuda, "get_device_properties", None), index)
        cuda_evidence.update({
            "device_index": index,
            "name": getattr(properties, "name", None),
            "total_memory_bytes": getattr(properties, "total_memory", None),
            "capability": list(_call(getattr(cuda, "get_device_capability", None), index) or ()) or None,
        })
    evidence["cuda"] = cuda_evidence

    mps = getattr(torch, "mps", None)
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_evidence: dict[str, Any] = {
        "built": bool(_call(getattr(mps_backend, "is_built", None))),
        "available": bool(_call(getattr(mps_backend, "is_available", None))),
    }
    if _device_type(resolved_device) == "mps":
        for field in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
            value = _call(getattr(mps, field, None))
            if value is not None:
                mps_evidence[f"{field}_bytes"] = int(value)
    evidence["mps"] = mps_evidence
    return evidence


class DeviceMemoryTracker:
    """Report only memory measurements whose precision can be stated honestly."""

    def __init__(self, device: Any, *, torch_module: Any = None) -> None:
        self.device = device
        self._device_type = _device_type(device)
        self._torch = _torch(torch_module)
        self._mps_maximum: int | None = None

    def start(self) -> None:
        """Reset CUDA allocator peak immediately before the evaluated stream."""
        if self._device_type == "cuda" and self._torch is not None:
            _call(getattr(getattr(self._torch, "cuda", None), "reset_peak_memory_stats", None), self.device)

    def sample_post_batch(self) -> None:
        """Sample MPS allocation after the caller has synchronized the batch."""
        if self._device_type != "mps" or self._torch is None:
            return
        value = _call(getattr(getattr(self._torch, "mps", None), "current_allocated_memory", None))
        if value is not None:
            self._mps_maximum = max(self._mps_maximum or 0, int(value))

    def summary(self) -> dict[str, Any]:
        if self._device_type == "cuda":
            value = _call(getattr(getattr(self._torch, "cuda", None), "max_memory_allocated", None), self.device) if self._torch else None
            return {"status": "collected" if value is not None else "unavailable",
                    "kind": "exact_cuda_allocator_peak", "bytes": int(value) if value is not None else None}
        if self._device_type == "mps":
            return {"status": "sampled" if self._mps_maximum is not None else "unavailable",
                    "kind": "sampled_post_batch_maximum", "bytes": self._mps_maximum}
        return {"status": "not_applicable", "kind": "unsupported", "bytes": None}
