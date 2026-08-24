"""Dependency-free verification of model artifacts and local dataset inventories.

Generation deliberately hashes every dataset file once.  Normal run startup
can instead use ``exact=False``: it validates the sidecar, canonical paths,
and the complete file/size inventory without rereading multi-gigabyte arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


SCHEMA_VERSION = 1
SIDECAR_DIRECTORY = ".nb-ramen-provenance"
_MODEL_ALIASES = {
    "clip_vitbase16": "ViT-B/16", "clip_vitbase32": "ViT-B/32",
    "clip_vitlarge14": "ViT-L/14", "clip_rn50": "RN50", "clip_rn101": "RN101",
}
OFFICIAL_CLIP_MODEL_URLS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}
CIFAR100C_OFFICIAL_ACQUISITION = {
    "publisher": "Zenodo",
    "record_id": "3555552",
    "doi": "10.5281/zenodo.3555552",
    "url": "https://zenodo.org/api/records/3555552/files/CIFAR-100-C.tar/content",
    "algorithm": "md5",
    "expected_checksum": "11f0ed0f1191edbf9fa23466ae6021d3",
    "actual_checksum": "11f0ed0f1191edbf9fa23466ae6021d3",
    "size_bytes": 2918473216,
}


class ProvenanceError(ValueError):
    """Artifact evidence is missing, malformed, or does not match the bytes on disk."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ProvenanceError("manifest contains a non-canonical relative path")
    return path


def _reject_symlink_path(path: Path) -> None:
    """Reject a symlink at the supplied artifact path.

    Callers walking a dataset also inspect every descendant directory.  Do not
    reject arbitrary absolute ancestors here: macOS commonly exposes ``/var``
    as a system symlink, which is outside the artifact trust boundary.
    """
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ProvenanceError(f"symlinked artifact path is not allowed: {path}")
    except FileNotFoundError:
        return


def _regular_fd(path: Path) -> tuple[int, os.stat_result]:
    _reject_symlink_path(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ProvenanceError(f"cannot safely open artifact: {path}") from exc
    try:
        info = os.fstat(fd)
    except OSError as exc:
        os.close(fd)
        raise ProvenanceError(f"cannot safely inspect artifact: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ProvenanceError(f"artifact is not a regular file: {path}")
    return fd, info


def checksum_regular_file(path: str | Path, algorithm: str = "sha256") -> dict[str, Any]:
    """Checksum one stable regular file without following symlinks.

    The descriptor, rather than a second pathname open, is streamed.  Inode
    identity and size are checked before and after hashing to reject common
    replacement/truncation races.
    """
    if algorithm not in {"md5", "sha256"}:
        raise ValueError("checksum algorithm must be md5 or sha256")
    artifact = Path(path)
    fd, before = _regular_fd(artifact)
    digest = hashlib.new(algorithm)
    try:
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        named = artifact.lstat()
    except OSError as exc:
        raise ProvenanceError(f"artifact disappeared while hashing: {artifact}") from exc
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size) or \
            (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
        raise ProvenanceError(f"artifact changed while hashing: {artifact}")
    return {"algorithm": algorithm, "checksum": digest.hexdigest(), "size_bytes": before.st_size}


def sha256_regular_file(path: str | Path) -> dict[str, Any]:
    """Hash one stable regular file using SHA-256 without following symlinks."""
    result = checksum_regular_file(path, "sha256")
    return {"algorithm": "sha256", "sha256": result["checksum"], "size_bytes": result["size_bytes"]}


def resolve_clip_model(model: str, *, model_urls: Mapping[str, str] | None = None,
                       clip_source: str | Path | None = None) -> dict[str, Any]:
    """Resolve CLIP metadata from the built-in trust anchor.

    ``model_urls`` exists only for low-level unit tests and is explicitly
    labelled untrusted in the result. Production verification never supplies
    it and therefore never trusts importable package metadata.
    """
    if clip_source is not None:
        raise ProvenanceError("importable CLIP package metadata is not a trusted model source")
    official_name = _MODEL_ALIASES.get(model, model)
    injected = model_urls is not None
    urls = dict(model_urls) if injected else OFFICIAL_CLIP_MODEL_URLS
    url = urls.get(official_name)
    if not url:
        raise ProvenanceError(f"unsupported OpenAI CLIP model: {model}")
    parts = [part for part in urlparse(url).path.split("/") if part]
    expected = next((part.lower() for part in parts if len(part) == 64 and all(c in "0123456789abcdefABCDEF" for c in part)), None)
    if expected is None:
        raise ProvenanceError("official CLIP URL does not carry a SHA-256 path component")
    filename = parts[-1]
    if not filename.endswith(".pt"):
        raise ProvenanceError("official CLIP URL does not name a checkpoint")
    if not injected and (
        urlparse(url).scheme != "https" or urlparse(url).hostname != "openaipublic.azureedge.net"
    ):
        raise ProvenanceError("pinned OpenAI CLIP URL trust anchor is invalid")
    return {
        "model": model, "official_name": official_name, "url": url,
        "expected_sha256": expected, "filename": filename,
        "publisher": "OpenAI" if not injected else None,
        "trust": "pinned_official" if not injected else "untrusted_injected_test_metadata",
    }


def verify_clip_checkpoint(model: str, checkpoint: str | Path) -> dict[str, Any]:
    resolved = resolve_clip_model(model)
    actual = sha256_regular_file(checkpoint)
    if actual["sha256"] != resolved["expected_sha256"]:
        raise ProvenanceError(f"CLIP checkpoint SHA-256 mismatch: {checkpoint}")
    return {**resolved, "path": str(Path(checkpoint)), "actual_sha256": actual["sha256"], "size_bytes": actual["size_bytes"]}


def verify_cached_clip_checkpoint(model: str, cache_root: str | Path) -> dict[str, Any]:
    resolved = resolve_clip_model(model)
    return verify_clip_checkpoint(model, Path(cache_root) / resolved["filename"])


def archive_acquisition_record(archive: str | Path, *, url: str, algorithm: str,
                               expected_checksum: str) -> dict[str, Any]:
    """Generic archive checksum record; this does not prove publisher origin."""
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ProvenanceError("acquisition URL must be an absolute HTTP(S) URL")
    lengths = {"md5": 32, "sha256": 64}
    if algorithm not in lengths:
        raise ProvenanceError("archive checksum algorithm must be md5 or sha256")
    if not isinstance(expected_checksum, str) or len(expected_checksum) != lengths[algorithm] or any(c not in "0123456789abcdefABCDEF" for c in expected_checksum):
        raise ProvenanceError(f"expected archive {algorithm} checksum is malformed")
    actual = checksum_regular_file(archive, algorithm)
    record = {"url": url, "algorithm": algorithm, "expected_checksum": expected_checksum.lower(),
              "actual_checksum": actual["checksum"], "size_bytes": actual["size_bytes"]}
    if record["expected_checksum"] != record["actual_checksum"]:
        raise ProvenanceError(f"archive {algorithm} checksum mismatch")
    return record


def verify_official_cifar100c_archive(archive: str | Path) -> dict[str, Any]:
    """Verify the exact archive published by Zenodo record 3555552."""
    expected = CIFAR100C_OFFICIAL_ACQUISITION
    checked = archive_acquisition_record(
        archive, url=expected["url"], algorithm=expected["algorithm"],
        expected_checksum=expected["expected_checksum"],
    )
    if checked["size_bytes"] != expected["size_bytes"]:
        raise ProvenanceError("official CIFAR-100-C archive size mismatch")
    return dict(expected)


def _walk_regular_files(root: Path) -> list[tuple[str, Path]]:
    _reject_symlink_path(root)
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise ProvenanceError(f"dataset root is not a directory: {root}")
    except OSError as exc:
        raise ProvenanceError(f"cannot inspect dataset root: {root}") from exc
    found: list[tuple[str, Path]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        if current_path == root and SIDECAR_DIRECTORY in directories:
            _reject_symlink_path(root / SIDECAR_DIRECTORY)
        # Only the root's own canonical inventory directory is metadata.
        # A nested class/image subtree with the same name is dataset content.
        directories[:] = sorted(
            directory for directory in directories
            if not (current_path == root and directory == SIDECAR_DIRECTORY)
        )
        for name in directories:
            candidate = current_path / name
            if stat.S_ISLNK(candidate.lstat().st_mode):
                raise ProvenanceError(f"symlinked dataset directory is not allowed: {candidate}")
        for name in sorted(files):
            candidate = current_path / name
            if stat.S_ISLNK(candidate.lstat().st_mode):
                raise ProvenanceError(f"symlinked dataset file is not allowed: {candidate}")
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ProvenanceError(f"dataset contains a non-regular file: {candidate}")
            relative = candidate.relative_to(root).as_posix()
            _safe_relative(relative)
            found.append((relative, candidate))
    return sorted(found)


def _content_digest(files: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item["path"]).encode("utf-8")); digest.update(b"\0")
        digest.update(str(item["size_bytes"]).encode("ascii")); digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii")); digest.update(b"\n")
    return digest.hexdigest()


def _verified_regular_size(path: Path) -> int:
    """Inspect one pathname through a non-following descriptor and bind its identity."""
    fd, before = _regular_fd(path)
    try:
        after = os.fstat(fd)
        named = path.lstat()
    except OSError as exc:
        raise ProvenanceError(f"artifact changed while inspecting: {path}") from exc
    finally:
        os.close(fd)
    if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode) or \
            (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size) or \
            (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
        raise ProvenanceError(f"artifact changed while inspecting: {path}")
    return before.st_size


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(path.parent)
    _reject_symlink_path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_json(payload)); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def default_sidecar_path(dataset_root: str | Path, dataset: str) -> Path:
    if dataset not in {"cifar100c", "domainnet"}:
        raise ValueError("dataset must be cifar100c or domainnet")
    return Path(dataset_root) / SIDECAR_DIRECTORY / f"{dataset}-v{SCHEMA_VERSION}.json"


def _validate_cifar_acquisition(acquisition: Mapping[str, Any]) -> None:
    if dict(acquisition) != CIFAR100C_OFFICIAL_ACQUISITION:
        raise ProvenanceError("CIFAR-100-C acquisition does not match the pinned official Zenodo artifact")


def generate_dataset_provenance(dataset: str, dataset_root: str | Path, *, manifest_path: str | Path | None = None,
                                acquisition: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create a canonical local content inventory. This explicit step may be slow."""
    if dataset not in {"cifar100c", "domainnet"}:
        raise ValueError("dataset must be cifar100c or domainnet")
    root = Path(dataset_root).absolute()
    sidecar = Path(manifest_path) if manifest_path is not None else default_sidecar_path(root, dataset)
    _reject_symlink_path(root / SIDECAR_DIRECTORY)
    if sidecar.absolute().parent != (root / SIDECAR_DIRECTORY).absolute() or sidecar.name != f"{dataset}-v{SCHEMA_VERSION}.json":
        raise ProvenanceError("sidecar must use the canonical dataset-root provenance path")
    records = []
    for relative, file_path in _walk_regular_files(root):
        hashed = sha256_regular_file(file_path)
        records.append({"path": relative, "size_bytes": hashed["size_bytes"], "sha256": hashed["sha256"]})
    if not records:
        raise ProvenanceError("refusing to generate provenance for an empty dataset root")
    acquisition_record = dict(acquisition or {})
    if dataset == "cifar100c":
        _validate_cifar_acquisition(acquisition_record)
        acquisition_record["expected_checksum"] = acquisition_record["expected_checksum"].lower()
        acquisition_record["actual_checksum"] = acquisition_record["actual_checksum"].lower()
    payload = {"schema_version": SCHEMA_VERSION, "dataset": dataset, "root": ".", "sidecar": f"{SIDECAR_DIRECTORY}/{sidecar.name}",
               "acquisition": acquisition_record, "content": {"algorithm": "sha256", "files": records, "root_digest": _content_digest(records)}}
    _atomic_json(sidecar, payload)
    return payload


def _read_manifest(root: Path, dataset: str, manifest_path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    sidecar = Path(manifest_path) if manifest_path is not None else default_sidecar_path(root, dataset)
    _reject_symlink_path(root / SIDECAR_DIRECTORY)
    canonical = default_sidecar_path(root, dataset).absolute()
    if sidecar.absolute() != canonical:
        raise ProvenanceError("provenance sidecar path is not canonical")
    raw = sha256_regular_file(sidecar)  # safely opens and establishes it is a regular non-symlink
    fd, before = _regular_fd(sidecar)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        named = sidecar.lstat()
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size) or \
                (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise ProvenanceError("provenance sidecar changed while reading")
        encoded = b"".join(chunks)
        if hashlib.sha256(encoded).hexdigest() != raw["sha256"]:
            raise ProvenanceError("provenance sidecar changed while reading")
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError("invalid provenance JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION or payload.get("dataset") != dataset or payload.get("root") != "." or payload.get("sidecar") != f"{SIDECAR_DIRECTORY}/{sidecar.name}":
        raise ProvenanceError("provenance sidecar contract is invalid")
    payload["sidecar_sha256"] = raw["sha256"]
    return sidecar, payload


def verify_dataset_provenance(dataset: str, dataset_root: str | Path, *, exact: bool = False,
                              manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Validate generated evidence; ``exact`` additionally streams each file."""
    root = Path(dataset_root).absolute()
    _, payload = _read_manifest(root, dataset, manifest_path)
    content = payload.get("content")
    if not isinstance(content, dict) or content.get("algorithm") != "sha256" or not isinstance(content.get("files"), list):
        raise ProvenanceError("provenance content contract is invalid")
    if dataset == "cifar100c" or payload.get("acquisition"):
        _validate_cifar_acquisition(payload.get("acquisition") if isinstance(payload.get("acquisition"), dict) else {})
    entries = content["files"]
    expected: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ProvenanceError("invalid provenance file entry")
        relative = _safe_relative(str(entry["path"])).as_posix()
        if relative in expected or not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0 or \
                not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64 or \
                any(character not in "0123456789abcdef" for character in entry["sha256"]):
            raise ProvenanceError("invalid provenance file entry")
        expected[relative] = entry
    if not expected or content.get("root_digest") != _content_digest(entries):
        raise ProvenanceError("provenance root digest mismatch")
    present = dict(_walk_regular_files(root))
    if set(present) != set(expected):
        raise ProvenanceError("dataset file inventory is stale")
    for relative, path in present.items():
        record = expected[relative]
        current = _verified_regular_size(path)
        if current != record["size_bytes"]:
            raise ProvenanceError(f"dataset file size changed: {relative}")
        if exact and sha256_regular_file(path)["sha256"] != record["sha256"]:
            raise ProvenanceError(f"dataset file SHA-256 mismatch: {relative}")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "root": str(root),
        "sidecar": str(default_sidecar_path(root, dataset).absolute()),
        "verified_exact": exact,
        "content_algorithm": "sha256",
        "root_digest": content["root_digest"],
        "sidecar_sha256": payload["sidecar_sha256"],
        "file_count": len(expected),
        "acquisition": dict(payload.get("acquisition") or {}),
    }


def generate_cifar100c_provenance(dataset_root: str | Path, *, acquisition: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    return generate_dataset_provenance("cifar100c", dataset_root, acquisition=acquisition, **kwargs)


def verify_cifar100c_provenance(dataset_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    return verify_dataset_provenance("cifar100c", dataset_root, **kwargs)


def generate_domainnet_provenance(dataset_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    return generate_dataset_provenance("domainnet", dataset_root, **kwargs)


def verify_domainnet_provenance(dataset_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    return verify_dataset_provenance("domainnet", dataset_root, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "verify")); parser.add_argument("dataset", choices=("cifar100c", "domainnet")); parser.add_argument("dataset_root")
    parser.add_argument("--exact", action="store_true"); parser.add_argument("--acquisition-json")
    args = parser.parse_args(argv)
    acquisition = json.loads(args.acquisition_json) if args.acquisition_json else None
    result = generate_dataset_provenance(args.dataset, args.dataset_root, acquisition=acquisition) if args.action == "generate" else verify_dataset_provenance(args.dataset, args.dataset_root, exact=args.exact)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
