"""Open-set dataset views with evaluator-only sample metadata.

The wrapper intentionally leaves the source examples in place.  The model
vocabulary is restricted to the known classes while stream construction can
still select examples from both known and held-out classes.  Unknown labels
are never remapped into the model vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:  # Keep split/metadata utilities usable in minimal test environments.
    from .corruption.CIFAR100C import CIFAR100C
except ImportError:  # pragma: no cover - exercised only without dataset dependencies
    class CIFAR100C:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenSetCIFAR100C requires the CIFAR-100-C dataset dependencies")

try:  # DomainNet is optional for dependency-light split validation tests.
    from .domainbed import DomainNet
except ImportError:  # pragma: no cover - exercised only without dataset dependencies
    class DomainNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("OpenSetDomainNet requires the DomainNet dataset dependencies")


DEFAULT_SPLIT_PATH = (
    Path(__file__).resolve().parents[2] / "cfg" / "research" / "open-set-cifar100-split-v1.json"
)
DEFAULT_DOMAINNET_SPLIT_PATH = (
    Path(__file__).resolve().parents[2] / "cfg" / "research" / "open-set-domainnet-split-v1.json"
)


def load_cifar100_open_set_split(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the repository's versioned CIFAR-100 open-set split."""
    split_path = Path(path) if path is not None else DEFAULT_SPLIT_PATH
    with split_path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, Mapping):
        raise ValueError("open-set split must be a JSON object")
    version = split.get("version")
    known = split.get("known_class_ids")
    unknown = split.get("unknown_class_ids")
    if not isinstance(version, str) or not version:
        raise ValueError("open-set split requires a non-empty version")
    if not isinstance(known, list) or not isinstance(unknown, list):
        raise ValueError("open-set split requires known_class_ids and unknown_class_ids lists")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in known + unknown):
        raise ValueError("open-set class IDs must be integers")
    if len(known) != 80 or len(unknown) != 20:
        raise ValueError("CIFAR-100 open-set split must contain 80 known and 20 unknown classes")
    if set(known).intersection(unknown) or set(known).union(unknown) != set(range(100)):
        raise ValueError("open-set class IDs must be a disjoint partition of 0..99")
    return {
        "version": version,
        "known_class_ids": tuple(known),
        "unknown_class_ids": tuple(unknown),
    }


class OpenSetDomainDataset:
    """Delegate image reads while exposing non-model-facing sample metadata."""

    def __init__(self, dataset: Any, known_label_by_original: Mapping[int, int]):
        self.dataset = dataset
        self._known_label_by_original = dict(known_label_by_original)
        labels = getattr(dataset, "Y", None)
        if labels is None:
            labels = getattr(dataset, "targets", None)
        if labels is None or len(labels) != len(dataset):
            raise ValueError("open-set datasets require Y or targets label metadata matching their length")
        self.Y = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        # Preserve the original (source) item contract.  Evaluation-only
        # metadata is supplied separately by StreamDataset's mapping tail.
        return self.dataset[index]

    def sample_metadata(self, index: int) -> dict[str, int | bool]:
        original_label = self.Y[index]
        known_label = self._known_label_by_original.get(original_label, -1)
        return {
            "original_label": original_label,
            "known_label_or_minus_one": known_label,
            "is_ood": known_label == -1,
        }


class OpenSetCIFAR100C(CIFAR100C):
    """CIFAR-100-C with an 80-class model vocabulary and 20 held-out classes."""

    def __init__(self, root, extra=False, severity=5, transform=None, *, split_path=None):
        split = load_cifar100_open_set_split(split_path)
        super().__init__(root, extra=extra, severity=severity, transform=transform)
        all_classes = tuple(self.classes)
        self.open_set_split_version = split["version"]
        self.known_class_ids = split["known_class_ids"]
        self.unknown_class_ids = split["unknown_class_ids"]
        known_label_by_original = {original: known for known, original in enumerate(self.known_class_ids)}
        self.classes = [all_classes[original] for original in self.known_class_ids]
        self.num_classes = len(self.classes)
        self.datasets = [
            OpenSetDomainDataset(dataset, known_label_by_original)
            for dataset in self.datasets
        ]


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("open-set split must be a JSON object")
    return value


def load_domainnet_open_set_split(path: str | Path | None = None) -> dict[str, Any]:
    """Load the versioned DomainNet split recipe without requiring images.

    DomainNet's labels are discovered from class directories at runtime.  The
    recipe therefore pins a name-based ranking algorithm and expected taxonomy
    size instead of embedding an unverified copy of its 345 class names.
    ``materialize_domainnet_open_set_split`` validates and binds the actual
    vocabulary before it is used.
    """
    split_path = Path(path) if path is not None else DEFAULT_DOMAINNET_SPLIT_PATH
    split = _read_json_object(split_path)
    version = split.get("version")
    dataset = split.get("dataset")
    expected_count = split.get("expected_class_count")
    known_count = split.get("known_class_count")
    unknown_count = split.get("unknown_class_count")
    selection = split.get("selection")
    if not isinstance(version, str) or not version:
        raise ValueError("DomainNet open-set split requires a non-empty version")
    if dataset != "DomainNet":
        raise ValueError("DomainNet open-set split requires dataset='DomainNet'")
    counts = (expected_count, known_count, unknown_count)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts):
        raise ValueError("DomainNet open-set split class counts must be positive integers")
    if known_count + unknown_count != expected_count:
        raise ValueError("DomainNet known and unknown counts must partition expected_class_count")
    if not isinstance(selection, Mapping):
        raise ValueError("DomainNet open-set split requires a selection object")
    if selection.get("strategy") != "sha256-name-rank-v1":
        raise ValueError("unsupported DomainNet open-set selection strategy")
    salt = selection.get("salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError("DomainNet open-set split requires a non-empty selection salt")
    return {
        "version": version,
        "dataset": dataset,
        "expected_class_count": expected_count,
        "known_class_count": known_count,
        "unknown_class_count": unknown_count,
        "selection": {"strategy": selection["strategy"], "salt": salt},
    }


def _domainnet_taxonomy_digest(class_names: tuple[str, ...]) -> str:
    encoded = json.dumps(list(class_names), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def materialize_domainnet_open_set_split(
    class_names: Any, path: str | Path | None = None
) -> dict[str, Any]:
    """Create an explicit known/unknown partition for the supplied taxonomy.

    Class names are the stable semantic identifiers; numeric labels are only
    ImageFolder's local encoding.  The output is safe to persist in an
    evaluation manifest because it includes both split and taxonomy digests.
    """
    split = load_domainnet_open_set_split(path)
    names = tuple(class_names)
    if len(names) != split["expected_class_count"]:
        raise ValueError(
            "DomainNet taxonomy size does not match the split recipe: "
            f"expected {split['expected_class_count']}, got {len(names)}"
        )
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("DomainNet taxonomy must contain unique, non-empty class names")
    # ImageFolder supplies a sorted vocabulary.  Sorting here makes the recipe
    # independent of any equivalent source-container ordering.
    canonical_names = tuple(sorted(names))
    salt = split["selection"]["salt"]
    ranked_names = sorted(
        canonical_names,
        key=lambda name: (hashlib.sha256(f"{salt}\0{name}".encode("utf-8")).hexdigest(), name),
    )
    known_names = tuple(ranked_names[: split["known_class_count"]])
    unknown_names = tuple(ranked_names[split["known_class_count"] :])
    original_id_by_name = {name: index for index, name in enumerate(names)}
    known_class_ids = tuple(original_id_by_name[name] for name in known_names)
    unknown_class_ids = tuple(original_id_by_name[name] for name in unknown_names)
    if set(known_class_ids).intersection(unknown_class_ids) or len(known_class_ids) + len(unknown_class_ids) != len(names):
        raise AssertionError("DomainNet open-set partition must cover the supplied taxonomy exactly once")
    taxonomy_digest = _domainnet_taxonomy_digest(canonical_names)
    fingerprint_payload = {
        "version": split["version"],
        "taxonomy_sha256": taxonomy_digest,
        "known_class_names": known_names,
        "unknown_class_names": unknown_names,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **split,
        "taxonomy_sha256": taxonomy_digest,
        "fingerprint": fingerprint,
        "known_class_names": known_names,
        "unknown_class_names": unknown_names,
        "known_class_ids": known_class_ids,
        "unknown_class_ids": unknown_class_ids,
    }


class OpenSetDomainNet(DomainNet):
    """DomainNet with a name-derived known-class model vocabulary.

    The source ImageFolder labels remain untouched.  Only ``classes`` is
    reduced for prompt/model construction; evaluation reads original labels
    and OOD membership from wrapped domain datasets.
    """

    def __init__(self, root, transform=None, *, split_path=None):
        super().__init__(root, transform=transform)
        split = materialize_domainnet_open_set_split(self.classes, split_path)
        self.open_set_split_version = split["version"]
        self.open_set_split_fingerprint = split["fingerprint"]
        self.open_set_taxonomy_sha256 = split["taxonomy_sha256"]
        self.known_class_ids = split["known_class_ids"]
        self.unknown_class_ids = split["unknown_class_ids"]
        self.known_class_names = split["known_class_names"]
        self.unknown_class_names = split["unknown_class_names"]
        known_label_by_original = {
            original: known for known, original in enumerate(self.known_class_ids)
        }
        self.classes = list(self.known_class_names)
        self.num_classes = len(self.classes)
        self.datasets = [
            OpenSetDomainDataset(dataset, known_label_by_original)
            for dataset in self.datasets
        ]
