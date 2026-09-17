"""Download pinned HF Parquet files and reconstruct checksum-identical CIFAR100C arrays."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from io import BytesIO
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import hf_hub_download

from evaluation.evidence import atomic_write_json
from runtime.artifact_provenance import (
    ProvenanceError, checksum_regular_file, generate_cifar100c_provenance,
    verify_cifar100c_provenance, resolve_clip_model, verify_clip_checkpoint,
)
from runtime.cifar100c_huggingface import (
    HF_REPOSITORY, HF_REVISION, CIFAR100C_NPY_MD5, CIFAR100C_HF_ACQUISITION,
)


def verified_npy(path):
    if not path.is_file():
        return False
    digest = checksum_regular_file(path, "md5")["checksum"]
    if digest != CIFAR100C_NPY_MD5[path.name]:
        raise ProvenanceError(f"Original CIFAR-100-C checksum mismatch: {path.name}")
    return True


def fetch(corruption, severity, cache):
    filename = f"data/{corruption}/severity_{severity}/data-00000.parquet"
    for attempt in range(1, 7):
        try:
            return Path(hf_hub_download(
                HF_REPOSITORY, filename, repo_type="dataset", revision=HF_REVISION,
                cache_dir=str(cache), token=False,
            ))
        except Exception as exc:
            if attempt == 6:
                raise
            print(f"Retry {filename} ({attempt}/6): {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)


def write_severity(parquet, output, offset, expected_labels=None):
    """Preserve the original row order; refuse wrong counts, labels, shapes or modes."""
    source = pq.ParquetFile(parquet)
    if source.metadata.num_rows != 10000 or set(source.schema_arrow.names) != {"image", "label"}:
        raise ValueError(f"Unexpected CIFAR100C Parquet schema/row count: {parquet}")
    labels, row = np.empty(10000, dtype=np.uint8), 0
    for batch in source.iter_batches(batch_size=256):
        values = batch.to_pydict()
        for encoded, label in zip(values["image"], values["label"]):
            if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < 100:
                raise ValueError(f"Invalid CIFAR100C label at row {row}: {label}")
            if not isinstance(encoded, dict) or not encoded.get("bytes"):
                raise ValueError(f"Missing embedded image bytes at row {row}")
            with Image.open(BytesIO(encoded["bytes"])) as image:
                if image.mode != "RGB" or image.size != (32, 32):
                    raise ValueError(f"Unexpected CIFAR100C image at row {row}")
                output[offset + row] = np.asarray(image, dtype=np.uint8)
            labels[row] = label
            row += 1
    if row != 10000:
        raise ValueError("Incomplete severity split")
    if expected_labels is not None and not np.array_equal(labels, expected_labels):
        raise ValueError("CIFAR100C label order differs across severity/corruption files")
    return labels


def convert_corruption(corruption, destination, cache, reference_labels=None):
    output = destination / f"{corruption}.npy"
    if verified_npy(output):
        return reference_labels
    # The .part file is never treated as a completed array on the next invocation.
    partial = output.with_suffix(".npy.part")
    array = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.uint8, shape=(50000, 32, 32, 3), version=(1, 0),
    )
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            paths = list(pool.map(lambda level: fetch(corruption, level, cache), range(1, 6)))
        for severity, parquet in enumerate(paths, 1):
            print(f"  convert {corruption}, severity {severity}/5", flush=True)
            labels = write_severity(parquet, array, (severity - 1) * 10000, reference_labels)
            if reference_labels is None:
                reference_labels = labels
        array.flush()
    finally:
        del array
    actual = checksum_regular_file(partial, "md5")["checksum"]
    if actual != CIFAR100C_NPY_MD5[output.name]:
        raise ProvenanceError(f"Reconstructed array differs from the original: {output.name}")
    partial.replace(output)
    return reference_labels


def prepare():
    data = Path(os.environ["RAMEN_DATA_ROOT"])
    runtime = Path(os.environ["RAMEN_RUNTIME_ROOT"])
    data.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    dataset = data / "corruption/CIFAR-100-C"
    cache = data.parent / "nb-ramen-hf-cache"
    if dataset.exists():
        report = verify_cifar100c_provenance(dataset, exact=True)
        if report["acquisition"] != CIFAR100C_HF_ACQUISITION:
            raise RuntimeError("This notebook requires its separately verified Hugging Face dataset")
    else:
        staging = data / "cifar100c-hf-staging"
        staging.mkdir(exist_ok=True)
        labels_path = staging / "labels.npy"
        labels = None
        if verified_npy(labels_path):
            labels = np.load(labels_path, allow_pickle=False)[:10000]
        corruptions = sorted(Path(name).stem for name in CIFAR100C_NPY_MD5 if name != "labels.npy")
        for index, corruption in enumerate(corruptions, 1):
            print(f"[{index}/19] {corruption}", flush=True)
            # If conversion completed before a crash but labels were not saved,
            # recover labels from a pinned split instead of accepting missing labels.
            if labels is None and (staging / f"{corruption}.npy").exists():
                raw = pq.read_table(fetch(corruption, 1, cache), columns=["label"])["label"].to_numpy()
                if raw.shape != (10000,) or raw.dtype.kind not in "iu" or np.any((raw < 0) | (raw >= 100)):
                    raise ValueError("Invalid or incomplete label split")
                labels = raw.astype(np.uint8)
            labels = convert_corruption(corruption, staging, cache, labels)
            if not labels_path.exists():
                partial = labels_path.with_suffix(".npy.part")
                with partial.open("wb") as handle:
                    np.save(handle, np.tile(labels, 5), allow_pickle=False)
                if checksum_regular_file(partial, "md5")["checksum"] != CIFAR100C_NPY_MD5["labels.npy"]:
                    raise ProvenanceError("Original CIFAR100C labels/order checksum mismatch")
                partial.replace(labels_path)
        # The patched generator independently rechecks all original file MD5s.
        generate_cifar100c_provenance(staging, acquisition=CIFAR100C_HF_ACQUISITION)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(dataset)
        report = verify_cifar100c_provenance(dataset, exact=True)

    atomic_write_json(runtime / "huggingface-acquisition.json", report["acquisition"])
    support_path = runtime / "download-support.py"
    spec = importlib.util.spec_from_file_location("download_support", support_path)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    resolved = resolve_clip_model("clip_vitbase16")
    model = support.download(
        [resolved["url"]], Path.home() / ".cache/clip" / resolved["filename"],
        lambda path: verify_clip_checkpoint("clip_vitbase16", path),
        log_path=runtime / "download-attempts.jsonl",
    )
    atomic_write_json(runtime / "artifact-provenance.json", {"dataset": report, "model": model})
    print("Hugging Face data matches all original NPY checksums; CLIP verified.", flush=True)


if __name__ == "__main__":
    prepare()
