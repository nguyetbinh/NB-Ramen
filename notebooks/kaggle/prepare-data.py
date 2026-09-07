"""Prepare the checksum-verified official CIFAR-100-C archive and CLIP model."""
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from runtime.artifact_provenance import (
    CIFAR100C_OFFICIAL_ACQUISITION, verify_official_cifar100c_archive,
    generate_cifar100c_provenance, verify_cifar100c_provenance,
    resolve_clip_model, verify_cached_clip_checkpoint,
)
from evaluation.evidence import atomic_write_json


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--retry-delay", "5",
                        "--output", str(partial), url], check=True)
        partial.replace(path)


def prepare():
    data = Path(os.environ["RAMEN_DATA_ROOT"])
    runtime = Path(os.environ["RAMEN_EVIDENCE_ROOT"]) / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    archive_input = os.environ.get("RAMEN_ARCHIVE_INPUT", "").strip()
    archive = Path(archive_input) if archive_input else data.parent / "CIFAR-100-C.tar"
    if archive_input and not archive.is_file():
        raise FileNotFoundError(f"Attached archive not found: {archive}")
    if not archive_input:
        download(CIFAR100C_OFFICIAL_ACQUISITION["url"], archive)
    print("Verifying the official archive MD5 and size...", flush=True)
    acquisition = verify_official_cifar100c_archive(archive)
    atomic_write_json(runtime / "archive-acquisition.json", acquisition)
    dataset = data / "corruption/CIFAR-100-C"
    if not dataset.exists():
        staging = data / "extract-staging"
        staging.mkdir(parents=True, exist_ok=False)
        # Permit only the expected dataset tree and regular files/directories.
        with tarfile.open(archive) as source:
            members = source.getmembers()
            for member in members:
                parts = PurePosixPath(member.name).parts
                if (not parts or parts[0] != "CIFAR-100-C" or ".." in parts
                        or not (member.isfile() or member.isdir())):
                    raise RuntimeError(f"Unexpected archive entry: {member.name}")
            source.extractall(staging, members=members, filter="data")
        staged = staging / "CIFAR-100-C"
        # Inventory is created only for bytes extracted from the verified archive.
        generate_cifar100c_provenance(staged, acquisition=acquisition)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(dataset)
        staging.rmdir()
    # On resume, never bless an arbitrary existing tree by rebuilding its sidecar.
    dataset_provenance = verify_cifar100c_provenance(dataset, exact=True)
    resolved = resolve_clip_model("clip_vitbase16")
    cache = Path.home() / ".cache/clip"
    download(resolved["url"], cache / resolved["filename"])
    model_provenance = verify_cached_clip_checkpoint("clip_vitbase16", cache)
    atomic_write_json(runtime / "artifact-provenance.json", {
        "dataset": dataset_provenance, "model": model_provenance,
    })
    print("Official data and model verified.", flush=True)


if __name__ == "__main__":
    prepare()
