#!/usr/bin/env python3
"""Build a small, non-canonical CIFAR-100-C-compatible *pilot* layout.

This generator deliberately does not reproduce the CIFAR-100-C benchmark.
It applies lightweight deterministic image transformations to real CIFAR-100
test images solely to make fast loader and runtime-direction checks possible.
Do not use its output for benchmark results or comparisons with CIFAR-100-C.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image


CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)
SEVERITIES = 5
README_NAME = "README.json"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_CIFAR100_DATASET = "uoft-cs/cifar100"
HF_ROWS_PER_REQUEST = 100


def _clip_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, 255).astype(np.uint8)


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """A dependency-free square box blur for tiny CIFAR images."""
    if radius <= 0:
        return image.copy()
    padded = np.pad(image.astype(np.float32), ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(image, dtype=np.float32)
    width = 2 * radius + 1
    for y in range(width):
        for x in range(width):
            result += padded[y:y + image.shape[0], x:x + image.shape[1]]
    return _clip_uint8(result / (width * width))


def _resized_center(image: np.ndarray, scale: float, resample: Image.Resampling) -> np.ndarray:
    """Zoom into an image's centre and restore its original dimensions."""
    height, width = image.shape[:2]
    crop_h = max(1, int(round(height / scale)))
    crop_w = max(1, int(round(width / scale)))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = Image.fromarray(image).crop((left, top, left + crop_w, top + crop_h))
    return np.asarray(cropped.resize((width, height), resample=resample), dtype=np.uint8)


def apply_corruption(image: np.ndarray, corruption: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Apply a lightweight deterministic pilot transformation to one RGB image.

    Names intentionally match CIFAR-100-C only for loader compatibility; these
    operations are not the benchmark's corruption implementations.
    """
    if severity not in range(1, SEVERITIES + 1):
        raise ValueError(f"severity must be in 1..{SEVERITIES}, got {severity}")
    if corruption not in CORRUPTIONS:
        raise ValueError(f"unknown pilot corruption: {corruption}")

    level = severity / SEVERITIES
    pixels = image.astype(np.float32)
    if corruption == "gaussian_noise":
        return _clip_uint8(pixels + rng.normal(0, 8 + 34 * level, image.shape))
    if corruption == "shot_noise":
        scaled = np.clip(pixels / 255 * (60 - 45 * level), 0, None)
        return _clip_uint8(rng.poisson(scaled) / (60 - 45 * level) * 255)
    if corruption == "impulse_noise":
        result = image.copy()
        mask = rng.random(image.shape[:2]) < (0.01 + 0.14 * level)
        result[mask] = rng.integers(0, 2, (int(mask.sum()), 1), dtype=np.uint8) * 255
        return result
    if corruption == "defocus_blur":
        return _box_blur(image, severity)
    if corruption == "glass_blur":
        blurred = _box_blur(image, max(1, severity // 2))
        shift = severity // 2 + 1
        return np.roll(blurred, (int(rng.integers(-shift, shift + 1)), int(rng.integers(-shift, shift + 1))), axis=(0, 1))
    if corruption == "motion_blur":
        shifts = range(-severity, severity + 1)
        return _clip_uint8(sum(np.roll(image.astype(np.float32), x, axis=1) for x in shifts) / len(shifts))
    if corruption == "zoom_blur":
        zooms = np.linspace(1.0, 1.0 + 0.45 * level, severity + 2)
        return _clip_uint8(sum(_resized_center(image, float(zoom), Image.Resampling.BILINEAR).astype(np.float32) for zoom in zooms) / len(zooms))
    if corruption == "snow":
        snow = rng.random(image.shape[:2]) < (0.04 + 0.26 * level)
        result = pixels * (1 - 0.12 * level)
        result[snow] = result[snow] * 0.25 + 255 * 0.75
        return _clip_uint8(result)
    if corruption == "frost":
        tint = np.array((210, 235, 255), dtype=np.float32)
        texture = rng.random(image.shape[:2] + (1,)) * (0.08 + 0.22 * level)
        return _clip_uint8(pixels * (1 - texture) + tint * texture)
    if corruption == "fog":
        fog = _box_blur(rng.integers(0, 256, image.shape, dtype=np.uint8), 2 + severity).astype(np.float32)
        alpha = 0.08 + 0.35 * level
        return _clip_uint8(pixels * (1 - alpha) + (0.7 * fog + 180 * 0.3) * alpha)
    if corruption == "brightness":
        return _clip_uint8(pixels * (0.72 + 0.48 * level))
    if corruption == "contrast":
        return _clip_uint8((pixels - 128) * (1 - 0.72 * level) + 128)
    if corruption == "elastic_transform":
        result = np.empty_like(image)
        max_shift = severity
        for row in range(image.shape[0]):
            result[row] = np.roll(image[row], int(rng.integers(-max_shift, max_shift + 1)), axis=0)
        return result
    if corruption == "pixelate":
        small = max(2, 16 - 2 * severity)
        reduced = Image.fromarray(image).resize((small, small), resample=Image.Resampling.BOX)
        return np.asarray(reduced.resize((image.shape[1], image.shape[0]), resample=Image.Resampling.NEAREST), dtype=np.uint8)

    quality = max(8, 52 - 8 * severity)
    encoded = io.BytesIO()
    Image.fromarray(image).save(encoded, format="JPEG", quality=quality)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def _selected_indices(total: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Select real source examples, repeating only if the requested pilot is large."""
    if count <= total:
        return rng.permutation(total)[:count]
    return rng.choice(total, size=count, replace=True)


def load_huggingface_cifar100_test_rows(count: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fetch a small real-image CIFAR-100 test subset via the public dataset viewer.

    This is intentionally an acquisition convenience for the non-benchmark
    pilot only.  It avoids adding a parquet dependency or downloading the
    full torchvision archive when the local network is bandwidth constrained.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    images: list[np.ndarray] = []
    labels: list[int] = []
    for offset in range(0, count, HF_ROWS_PER_REQUEST):
        request_count = min(HF_ROWS_PER_REQUEST, count - offset)
        query = urlencode({
            "dataset": HF_CIFAR100_DATASET,
            "config": "cifar100",
            "split": "test",
            "offset": offset,
            "length": request_count,
        })
        with urlopen(f"{HF_ROWS_URL}?{query}", timeout=60) as response:  # nosec B310: fixed HTTPS endpoint
            payload = json.load(response)
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != request_count:
            raise RuntimeError("Hugging Face CIFAR-100 row response was incomplete")
        for item in rows:
            row = item.get("row", {})
            label = row.get("fine_label")
            image_source = row.get("img", {}).get("src")
            if not isinstance(label, int) or not isinstance(image_source, str):
                raise RuntimeError("Hugging Face CIFAR-100 row is missing image or fine_label")
            with urlopen(image_source, timeout=60) as response:  # nosec B310: dataset server's signed HTTPS URL
                with Image.open(io.BytesIO(response.read())) as image:
                    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if pixels.shape != (32, 32, 3):
                raise RuntimeError(f"unexpected CIFAR-100 image shape: {pixels.shape}")
            images.append(pixels)
            labels.append(label)
    return np.stack(images), np.asarray(labels, dtype=np.int64), {
        "dataset": "Hugging Face uoft-cs/cifar100 test split",
        "dataset_server": HF_ROWS_URL,
        "dataset_id": HF_CIFAR100_DATASET,
        "source_examples": count,
    }


def build_pilot(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    samples_per_severity: int,
    seed: int,
    source: dict[str, object],
) -> None:
    """Write a loader-compatible five-severity layout from real source images."""
    if samples_per_severity <= 0:
        raise ValueError("samples_per_severity must be positive")
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("images must have shape (N, 32, 32, 3)")
    if len(images) != len(labels):
        raise ValueError("images and labels must have equal length")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    rng = np.random.default_rng(seed)
    indices = _selected_indices(len(images), SEVERITIES * samples_per_severity, rng)
    base_images = images[indices].astype(np.uint8, copy=False)
    base_labels = np.asarray(labels)[indices].astype(np.int64, copy=False)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    try:
        np.save(staging / "labels.npy", base_labels)
        for corruption_index, corruption in enumerate(CORRUPTIONS):
            transformed = np.empty_like(base_images)
            for severity_index in range(SEVERITIES):
                start = severity_index * samples_per_severity
                end = start + samples_per_severity
                transform_rng = np.random.default_rng(seed + 10_000 * corruption_index + severity_index)
                for image_index in range(start, end):
                    transformed[image_index] = apply_corruption(
                        base_images[image_index], corruption, severity_index + 1, transform_rng,
                    )
            np.save(staging / f"{corruption}.npy", transformed)

        metadata = {
            "format": "NB-Ramen CIFAR-100-C pilot layout v1",
            "purpose": "Fast real-image loader/runtime directional checks only.",
            "benchmark_status": "NOT canonical CIFAR-100-C; not valid for benchmark reporting or comparison.",
            "source": source,
            "generator": "scripts/build-cifar100c-pilot.py",
            "parameters": {
                "samples_per_severity": samples_per_severity,
                "severities": SEVERITIES,
                "seed": seed,
                "corruptions": list(CORRUPTIONS),
                "selection": "deterministic source-test selection; repeats only when 5N exceeds source size",
                "transformations": "lightweight deterministic approximations, not CIFAR-100-C implementations",
            },
        }
        (staging / README_NAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing source torchvision data and corruption output.")
    parser.add_argument("--samples-per-severity", type=int, default=256, metavar="N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source", choices=("torchvision", "hf-rows"), default="torchvision",
        help="real CIFAR-100 test image source; hf-rows is suitable for a small bandwidth-limited pilot",
    )
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-100 if absent.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing pilot output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.data_root / "corruption" / "CIFAR-100-C"
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {output_dir}; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    source_count = SEVERITIES * args.samples_per_severity
    if args.source == "hf-rows":
        if args.download:
            raise SystemExit("--download is only valid with --source torchvision")
        images, labels, source = load_huggingface_cifar100_test_rows(source_count)
    else:
        try:
            import torchvision
            from torchvision.datasets import CIFAR100
        except ImportError as error:
            raise SystemExit("torchvision is required to build this pilot dataset.") from error
        dataset = CIFAR100(root=str(args.data_root), train=False, download=args.download)
        images = np.asarray(dataset.data, dtype=np.uint8)
        labels = np.asarray(dataset.targets, dtype=np.int64)
        source = {
            "dataset": "torchvision.datasets.CIFAR100 test split",
            "torchvision_version": torchvision.__version__,
            "download_requested": args.download,
            "source_examples": int(len(images)),
        }
    build_pilot(
        images, labels, output_dir, args.samples_per_severity, args.seed, source=source,
    )
    print(f"Built non-benchmark pilot data at {output_dir}")


if __name__ == "__main__":
    main()
