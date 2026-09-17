"""Prepare the checksum-verified official CIFAR-100-C archive and CLIP model."""
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import time

from runtime.artifact_provenance import (
    CIFAR100C_OFFICIAL_ACQUISITION, verify_official_cifar100c_archive,
    generate_cifar100c_provenance, verify_cifar100c_provenance,
    resolve_clip_model, verify_clip_checkpoint, ProvenanceError,
)
from evaluation.evidence import atomic_write_json


def preserve(path):
    saved = path.with_name(f"{path.name}.rejected-{time.time_ns()}")
    path.rename(saved)
    print(f"Preserved unusable download: {saved}", flush=True)


def download(urls, path, verify, *, log_path=None, attempts=6, retry_delay=30):
    """Resume transfers, rotate endpoints and publish only checksum-verified bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            return verify(path)
        except ProvenanceError:
            preserve(path)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(attempts):
        url = urls[attempt % len(urls)]
        print(f"Download {path.name}: attempt {attempt + 1}/{attempts}: {url}", flush=True)
        result = subprocess.run([
            "curl", "--fail", "--location", "--connect-timeout", "30",
            "--max-time", "3600", "--speed-time", "90", "--speed-limit", "1024",
            "--continue-at", "-", "--output", str(partial),
            "--write-out", "%{http_code}", url,
        ], stdout=subprocess.PIPE, text=True)
        http_status = result.stdout.strip()
        record = {"file": path.name, "url": url, "attempt": attempt + 1,
                  "curl_exit": result.returncode, "http_status": http_status,
                  "verified": False}
        verified = False
        if partial.is_file() and (result.returncode == 0 or http_status == "416"):
            try:
                verify(partial)
            except ProvenanceError as exc:
                record["verification_error"] = str(exc)
                preserve(partial)
            else:
                partial.replace(path)
                verified = record["verified"] = True
        elif result.returncode == 33 and partial.exists():
            # This endpoint cannot resume; make the next attempt a fresh transfer.
            preserve(partial)
        if log_path is not None:
            with Path(log_path).open("a") as log:
                log.write(json.dumps(record) + "\n")
        if verified:
            return verify(path)
        if attempt + 1 < attempts:
            print(f"Download incomplete (HTTP {http_status}); retry in {retry_delay}s.", flush=True)
            time.sleep(retry_delay)
    raise RuntimeError(
        f"Could not obtain verified {path.name} after {attempts} attempts. "
        "Keep the partial file and retry this cell, or attach the official archive via ARCHIVE_INPUT."
    )


def prepare():
    data = Path(os.environ["RAMEN_DATA_ROOT"])
    runtime = Path(os.environ["RAMEN_EVIDENCE_ROOT"]) / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    archive_input = os.environ.get("RAMEN_ARCHIVE_INPUT", "").strip()
    archive = Path(archive_input) if archive_input else data.parent / "CIFAR-100-C.tar"
    if archive_input and not archive.is_file():
        raise FileNotFoundError(f"Attached archive not found: {archive}")
    if not archive_input:
        download([
            "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1",
            CIFAR100C_OFFICIAL_ACQUISITION["url"],
        ], archive, verify_official_cifar100c_archive,
            log_path=runtime / "download-attempts.jsonl")
    print("Verifying the official archive MD5 and size...", flush=True)
    acquisition = verify_official_cifar100c_archive(archive)
    atomic_write_json(runtime / "archive-acquisition.json", acquisition)
    dataset = data / "corruption/CIFAR-100-C"
    if not dataset.exists():
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="extract-staging-", dir=data) as tmp:
            staging = Path(tmp)
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
    # On resume, never bless an arbitrary existing tree by rebuilding its sidecar.
    dataset_provenance = verify_cifar100c_provenance(dataset, exact=True)
    resolved = resolve_clip_model("clip_vitbase16")
    cache = Path.home() / ".cache/clip"
    model_provenance = download(
        [resolved["url"]], cache / resolved["filename"],
        lambda path: verify_clip_checkpoint("clip_vitbase16", path),
        log_path=runtime / "download-attempts.jsonl",
    )
    atomic_write_json(runtime / "artifact-provenance.json", {
        "dataset": dataset_provenance, "model": model_provenance,
    })
    print("Official data and model verified.", flush=True)


if __name__ == "__main__":
    prepare()
