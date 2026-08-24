"""Deterministic, metadata-only stream construction for TTA evaluation."""

from .builders import (
    StreamDataset,
    build_single_domain_stream,
    build_stream,
    stream_fingerprint,
    truncate_stream,
    verify_stream_fingerprint,
)

__all__ = [
    "StreamDataset",
    "build_single_domain_stream",
    "build_stream",
    "stream_fingerprint",
    "truncate_stream",
    "verify_stream_fingerprint",
]
