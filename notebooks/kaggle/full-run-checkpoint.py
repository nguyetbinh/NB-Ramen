"""Atomic ZIP checkpoints and non-overwriting restore for Kaggle campaigns."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def checkpoint(evidence):
    evidence = Path(evidence).resolve()
    if not evidence.is_dir():
        raise FileNotFoundError(evidence)
    archive = evidence.with_suffix(".zip")
    partial = archive.with_suffix(".zip.part")
    # Keep the previous complete ZIP until the replacement is closed successfully.
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            for path in sorted(evidence.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Evidence must not contain symlinks: {path}")
                if path.is_file():
                    output.write(path, path.relative_to(evidence.parent).as_posix())
        partial.replace(archive)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return archive


def restore(archive, evidence):
    archive, evidence = Path(archive), Path(evidence)
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    marker_name = "restored-archive.json"
    if evidence.exists():
        marker = evidence / marker_name
        if marker.is_file() and json.loads(marker.read_text()).get("sha256") == digest:
            return  # Setup rerun: retain all progress since the same restore.
        raise FileExistsError(
            "Evidence already exists. Clear RESUME_ARCHIVE to use it; restore never merges or overwrites runs."
        )
    evidence.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        seen = set()
        for entry in source.infolist():
            parts = PurePosixPath(entry.filename).parts
            mode = entry.external_attr >> 16
            normalized = PurePosixPath(entry.filename).as_posix()
            if (not parts or parts[0] != evidence.name or ".." in parts
                    or "\\" in entry.filename or PurePosixPath(entry.filename).is_absolute()
                    or stat.S_ISLNK(mode) or normalized in seen):
                raise ValueError(f"Invalid checkpoint member: {entry.filename}")
            seen.add(normalized)
        if not seen:
            raise ValueError("Checkpoint is empty")
        required = sum(entry.file_size for entry in source.infolist())
        if required >= shutil.disk_usage(evidence.parent).free:
            raise OSError("Insufficient disk space to restore this checkpoint")
        with tempfile.TemporaryDirectory(prefix="restore-full-", dir=evidence.parent) as tmp:
            # All paths/types were checked before any extraction. CRC errors abort staging.
            source.extractall(tmp)
            staged = Path(tmp) / evidence.name
            if not staged.is_dir():
                raise ValueError("Checkpoint root must be a directory")
            (staged / marker_name).write_text(json.dumps({"sha256": digest}, indent=2) + "\n")
            staged.rename(evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("save", "restore"))
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.action == "restore":
        if args.archive is None:
            parser.error("restore requires --archive")
        restore(args.archive, args.evidence)
        print("Checkpoint restored; each run still requires strict validation.")
    else:
        print(checkpoint(args.evidence))
