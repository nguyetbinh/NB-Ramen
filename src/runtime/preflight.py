"""Validate NB-Ramen data layouts and report runtime facts without importing ML packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Iterable


MAIN_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)
EXTRA_CORRUPTIONS = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
DOMAINBED_LAYOUTS = {
    "PACS": ("PACS", ("art_painting", "cartoon", "photo", "sketch")),
    "VLCS": ("VLCS", ("Caltech101", "LabelMe", "SUN09", "VOC2007")),
    "TerraIncognita": (
        "terra_incognita", ("location_100", "location_38", "location_43", "location_46"),
    ),
    "OfficeHome": ("office_home", ("Art", "Clipart", "Product", "Real World")),
    "DomainNet": (
        "domain_net", ("clipart", "infograph", "painting", "quickdraw", "real", "sketch"),
    ),
}
DATASETS = ("CIFAR10C", "CIFAR100C", "ImageNetC", "ImageNetC5K", *DOMAINBED_LAYOUTS)
DOMAINNET_CLASS_COUNT = 345
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp",
}


def _missing(path: Path, expected: str) -> dict[str, str]:
    return {"path": str(path), "expected": expected}


def _check_cifar(root: Path, dataset: str, corruptions: Iterable[str]) -> list[dict[str, str]]:
    directory = "CIFAR-10-C" if dataset == "CIFAR10C" else "CIFAR-100-C"
    base = root / "corruption" / directory
    missing = []
    labels = base / "labels.npy"
    if not labels.is_file():
        missing.append(_missing(labels, "labels.npy file"))
    for corruption in corruptions:
        path = base / (corruption + ".npy")
        if not path.is_file():
            missing.append(_missing(path, "NumPy corruption file"))
    return missing


def _contains_class_directory(path: Path) -> bool:
    """Check ImageFolder structure without opening a class image."""
    try:
        return any(entry.is_dir() for entry in path.iterdir())
    except OSError:
        return False


def _check_imagenet(root: Path, severity: int, corruptions: Iterable[str]) -> list[dict[str, str]]:
    base = root / "corruption" / "ImageNet-C"
    missing = []
    classnames = base / "classnames.txt"
    if not classnames.is_file():
        missing.append(_missing(classnames, "classnames.txt file"))
    for corruption in corruptions:
        path = base / corruption / str(severity)
        if not path.is_dir():
            missing.append(_missing(path, "severity directory"))
        elif not _contains_class_directory(path):
            missing.append(_missing(path, "severity directory containing class directories"))
    return missing


def _check_domainbed(root: Path, dataset: str) -> list[dict[str, str]]:
    directory, environments = DOMAINBED_LAYOUTS[dataset]
    base = root / "domainbed" / directory
    missing = []
    for environment in environments:
        path = base / environment
        if not path.is_dir():
            missing.append(_missing(path, "environment directory"))
        elif not _contains_class_directory(path):
            missing.append(_missing(path, "environment directory containing class directories"))
    return missing


def _deep_error(path: Path, check: str, message: str) -> dict[str, str]:
    return {"path": str(path), "check": check, "message": message}


def _deep_check_cifar(
    base: Path,
    dataset: str,
    corruptions: Iterable[str],
    expected_samples: int = 50_000,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Inspect CIFAR-C arrays with mmap so validation does not load them in RAM."""
    import numpy as np

    errors = []
    details: dict[str, object] = {
        "expected_samples": expected_samples,
        "expected_classes": 10 if dataset == "CIFAR10C" else 100,
        "arrays": {},
    }
    if base.is_symlink():
        return details, [_deep_error(base, "symlink", "dataset root must not be a symlink")]
    labels_path = base / "labels.npy"
    if labels_path.is_symlink():
        errors.append(_deep_error(labels_path, "symlink", "labels artifact must not be a symlink"))
    else:
        try:
            labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
            details["labels"] = {"shape": list(labels.shape), "dtype": str(labels.dtype)}
            if labels.shape != (expected_samples,):
                errors.append(_deep_error(
                    labels_path, "labels_shape", f"expected ({expected_samples},), got {labels.shape}",
                ))
            if not np.issubdtype(labels.dtype, np.integer):
                errors.append(_deep_error(labels_path, "labels_dtype", "labels must use an integer dtype"))
            elif labels.size:
                label_min = int(labels.min())
                label_max = int(labels.max())
                details["labels"].update({"min": label_min, "max": label_max})
                max_class = details["expected_classes"] - 1
                if label_min < 0 or label_max > max_class:
                    errors.append(_deep_error(
                        labels_path, "labels_range", f"expected labels in 0..{max_class}",
                    ))
                if labels.shape == (expected_samples,) and expected_samples % 5 == 0:
                    severity_size = expected_samples // 5
                    first_severity = labels[:severity_size]
                    if any(
                        not np.array_equal(
                            labels[severity * severity_size:(severity + 1) * severity_size],
                            first_severity,
                        )
                        for severity in range(1, 5)
                    ):
                        errors.append(_deep_error(
                            labels_path,
                            "labels_severity_consistency",
                            "label blocks for severities 2..5 must equal severity 1",
                        ))
        except Exception as exc:
            errors.append(_deep_error(labels_path, "labels_readable", f"NumPy load failed: {exc}"))

    expected_shape = (expected_samples, 32, 32, 3)
    for corruption in corruptions:
        path = base / f"{corruption}.npy"
        if path.is_symlink():
            errors.append(_deep_error(path, "symlink", "corruption artifact must not be a symlink"))
            continue
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            details["arrays"][corruption] = {
                "shape": list(array.shape), "dtype": str(array.dtype),
            }
            if array.shape != expected_shape:
                errors.append(_deep_error(
                    path, "corruption_shape", f"expected {expected_shape}, got {array.shape}",
                ))
            if array.dtype != np.uint8:
                errors.append(_deep_error(path, "corruption_dtype", "expected uint8"))
            if array.shape and array.shape[0] > 0:
                # Force mmap reads at both ends rather than trusting only the header.
                _ = int(np.asarray(array[0]).reshape(-1)[0])
                _ = int(np.asarray(array[-1]).reshape(-1)[-1])
        except Exception as exc:
            errors.append(_deep_error(path, "corruption_readable", f"NumPy mmap/read failed: {exc}"))
    return details, errors


def _sorted_entries(path: Path) -> list[Path]:
    """Match ImageFolder by including dot-prefixed entries in sorted traversal."""
    return sorted(path.iterdir(), key=lambda p: p.name)


def _recursive_image_files(
    directory: Path, errors: list[dict[str, str]],
) -> list[Path]:
    """Return sorted recursive image paths without following symlinks."""
    images = []
    try:
        entries = _sorted_entries(directory)
    except OSError as exc:
        errors.append(_deep_error(directory, "directory_readable", str(exc)))
        return images
    for entry in entries:
        if entry.is_symlink():
            errors.append(_deep_error(entry, "symlink", "nested path must not be a symlink"))
        elif entry.is_dir():
            images.extend(_recursive_image_files(entry, errors))
        elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(entry)
    return images


def _deep_check_domainnet(
    base: Path,
    environments: Iterable[str] = DOMAINBED_LAYOUTS["DomainNet"][1],
    expected_class_count: int = DOMAINNET_CLASS_COUNT,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Validate DomainNet's exact wrapper taxonomy and sample readability."""
    from PIL import Image

    environments = tuple(environments)
    errors = []
    details: dict[str, object] = {
        "expected_environments": list(environments),
        "expected_class_count": expected_class_count,
        "environment_image_counts": {},
        "total_images": 0,
    }
    if base.is_symlink():
        errors.append(_deep_error(base, "symlink", "dataset root must not be a symlink"))
        return details, errors
    try:
        actual_environment_entries = _sorted_entries(base)
    except OSError as exc:
        return details, [_deep_error(base, "root_readable", f"directory scan failed: {exc}")]
    actual_environments = [entry.name for entry in actual_environment_entries]
    details["actual_environments"] = actual_environments
    for entry in actual_environment_entries:
        if entry.is_symlink():
            errors.append(_deep_error(entry, "symlink", "environment entry must not be a symlink"))
    if actual_environments != sorted(environments):
        errors.append(_deep_error(
            base, "environments", f"expected exactly {sorted(environments)}, got {actual_environments}",
        ))

    reference_classes: list[str] | None = None
    for environment in environments:
        environment_path = base / environment
        if not environment_path.is_dir():
            continue
        if environment_path.is_symlink():
            errors.append(_deep_error(environment_path, "symlink", "environment must not be a symlink"))
            continue
        try:
            class_entries = _sorted_entries(environment_path)
        except OSError as exc:
            errors.append(_deep_error(environment_path, "environment_readable", str(exc)))
            continue
        for entry in class_entries:
            if entry.is_symlink():
                errors.append(_deep_error(entry, "symlink", "class entry must not be a symlink"))
        class_dirs = [entry for entry in class_entries if entry.is_dir()]
        class_names = [entry.name for entry in class_dirs]
        if len(class_names) != expected_class_count:
            errors.append(_deep_error(
                environment_path,
                "class_count",
                f"expected {expected_class_count} class directories, got {len(class_names)}",
            ))
        if reference_classes is None:
            reference_classes = class_names
            details["classes"] = class_names
        elif class_names != reference_classes:
            errors.append(_deep_error(
                environment_path, "class_taxonomy", "sorted class directories differ across environments",
            ))

        image_count = 0
        first_image: Path | None = None
        for class_path in class_dirs:
            if class_path.is_symlink():
                errors.append(_deep_error(class_path, "symlink", "class directory must not be a symlink"))
                continue
            images = _recursive_image_files(class_path, errors)
            if not images:
                errors.append(_deep_error(class_path, "class_nonempty", "class has no image files"))
            else:
                first_image = first_image or images[0]
                image_count += len(images)
        details["environment_image_counts"][environment] = image_count
        details["total_images"] += image_count
        if first_image is not None:
            try:
                with Image.open(first_image) as image:
                    image.verify()
                with Image.open(first_image) as image:
                    image.convert("RGB").load()
            except Exception as exc:
                errors.append(_deep_error(
                    first_image, "image_decode", f"Pillow verify/decode failed: {exc}",
                ))
    return details, errors


def validate_dataset_layout(
    data_root: str | Path, dataset: str, severity: int = 5, include_extra: bool = False,
    deep: bool = False,
) -> dict[str, object]:
    """Validate the path contract used by a dataset wrapper without loading data."""
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset: {}".format(dataset))
    if severity not in range(1, 6):
        raise ValueError("severity must be an integer from 1 through 5")
    root = Path(data_root).expanduser()
    corruptions = MAIN_CORRUPTIONS + (EXTRA_CORRUPTIONS if include_extra else ())
    if dataset in ("CIFAR10C", "CIFAR100C"):
        missing = _check_cifar(root, dataset, corruptions)
    elif dataset in ("ImageNetC", "ImageNetC5K"):
        missing = _check_imagenet(root, severity, corruptions)
    else:
        missing = _check_domainbed(root, dataset)
    deep_result: dict[str, object] = {
        "requested": deep, "status": "not_requested", "details": {}, "errors": [],
    }
    if deep and missing:
        deep_result["status"] = "skipped_missing"
    elif deep and dataset in ("CIFAR10C", "CIFAR100C"):
        directory = "CIFAR-10-C" if dataset == "CIFAR10C" else "CIFAR-100-C"
        details, errors = _deep_check_cifar(
            root / "corruption" / directory, dataset, corruptions,
        )
        deep_result.update({"status": "passed" if not errors else "failed", "details": details, "errors": errors})
    elif deep and dataset == "DomainNet":
        details, errors = _deep_check_domainnet(root / "domainbed" / "domain_net")
        deep_result.update({"status": "passed" if not errors else "failed", "details": details, "errors": errors})
    elif deep:
        deep_result["status"] = "not_supported"
    return {
        "dataset": dataset,
        "data_root": str(root),
        "severity": severity if dataset.startswith(("CIFAR", "ImageNet")) else None,
        "include_extra_corruptions": include_extra if dataset.startswith(("CIFAR", "ImageNet")) else None,
        "valid": not missing and not deep_result["errors"],
        "missing": missing,
        "deep": deep_result,
    }


def _run(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, check=False, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_facts(repository: str | Path | None = None) -> dict[str, object]:
    """Collect only facts available from the standard library or optional executables."""
    packages = {}
    for package in ("torch", "torchvision", "clip", "numpy", "Pillow", "PyYAML", "tqdm"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    repository_path = Path(repository or Path.cwd())
    git = {"available": shutil.which("git") is not None, "commit": None, "dirty": None}
    if git["available"]:
        git["commit"] = _run(["git", "rev-parse", "HEAD"], repository_path)
        status = _run(["git", "status", "--porcelain"], repository_path)
        if git["commit"] is not None and status is not None:
            git["dirty"] = bool(status)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "device": {
            "cuda_visible_devices": cuda_visible,
            "nvidia_device_files_present": any(Path("/dev").glob("nvidia*")),
            "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        },
        "git": git,
    }


def build_report(
    data_root: str | Path, datasets: Iterable[str], severity: int = 5,
    include_extra: bool = False, repository: str | Path | None = None, deep: bool = False,
) -> dict[str, object]:
    checks = [validate_dataset_layout(data_root, name, severity, include_extra, deep) for name in datasets]
    return {"runtime": runtime_facts(repository), "datasets": checks, "valid": all(check["valid"] for check in checks)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Parent directory containing corruption/ and domainbed/.")
    parser.add_argument("--dataset", choices=DATASETS, action="append", help="Dataset to validate; repeat as needed.")
    parser.add_argument("--all-datasets", action="store_true", help="Validate every supported dataset layout.")
    parser.add_argument("--severity", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--include-extra-corruptions", action="store_true")
    parser.add_argument(
        "--deep", action="store_true",
        help="Open and semantically validate CIFAR-C arrays or the DomainNet taxonomy and samples.",
    )
    parser.add_argument("--repository", default=Path.cwd(), help="Repository used for optional Git facts.")
    parser.add_argument("--json", action="store_true", help="Write the full report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.all_datasets:
        datasets = DATASETS
    elif args.dataset:
        datasets = tuple(dict.fromkeys(args.dataset))
    else:
        _parser().error("provide --dataset at least once or use --all-datasets")
    report = build_report(
        args.data_root, datasets, args.severity, args.include_extra_corruptions,
        args.repository, args.deep,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Runtime: Python {} on {}".format(report["runtime"]["python"], report["runtime"]["platform"]))
        for check in report["datasets"]:
            if check["valid"]:
                suffix = " (deep)" if check["deep"]["status"] == "passed" else ""
                print("PASS {}{}".format(check["dataset"], suffix))
            else:
                print("FAIL {}: {} missing, {} deep error(s)".format(
                    check["dataset"], len(check["missing"]), len(check["deep"]["errors"]),
                ))
                for item in check["missing"]:
                    print("  - {} ({})".format(item["path"], item["expected"]))
                for item in check["deep"]["errors"]:
                    print("  - {} [{}] {}".format(item["path"], item["check"], item["message"]))
        print("Preflight: {}".format("PASS" if report["valid"] else "FAIL"))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
