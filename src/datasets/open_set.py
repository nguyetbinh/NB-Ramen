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

# This is the source-label order set by CIFAR100C.set_classes().  Keep this
# dependency-light copy here because split validation is also used by tools
# which intentionally do not import the image dataset stack.
CIFAR100_CLASS_NAMES = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)
_CIFAR100_TAXONOMY_SHA256 = hashlib.sha256(
    json.dumps(list(CIFAR100_CLASS_NAMES), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
_CIFAR100_NAME_RANK_ALGORITHM = "sha256-utf8-salt-nul-class-name-rank-v1"


def _cifar100_name_ranked_ids(salt: str) -> tuple[int, ...]:
    """Return canonical CIFAR-100 IDs ranked by the frozen name recipe."""
    return tuple(sorted(
        range(len(CIFAR100_CLASS_NAMES)),
        key=lambda class_id: (
            hashlib.sha256(f"{salt}\0{CIFAR100_CLASS_NAMES[class_id]}".encode("utf-8")).hexdigest(),
            CIFAR100_CLASS_NAMES[class_id],
        ),
    ))


def _cifar100_split_fingerprint(split: Mapping[str, Any]) -> str:
    payload = {
        "version": split["version"],
        "dataset": split["dataset"],
        "canonical_class_names": split["canonical_class_names"],
        "taxonomy_sha256": split["taxonomy_sha256"],
        "recipe": split["recipe"],
        "known_class_ids": split["known_class_ids"],
        "unknown_class_ids": split["unknown_class_ids"],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_cifar100_recipe_metadata(split: Mapping[str, Any], known: list[int], unknown: list[int]) -> None:
    """Validate optional, self-auditing metadata on name-ranked split files.

    v1 predates this metadata and remains intentionally loadable.  Once a
    recipe is supplied, however, the materialized IDs must exactly match it.
    """
    recipe = split.get("recipe")
    metadata_keys = ("dataset", "canonical_class_names", "taxonomy_sha256", "fingerprint")
    if recipe is None:
        if any(key in split for key in metadata_keys):
            raise ValueError("CIFAR-100 split audit metadata requires a recipe")
        return
    if not isinstance(recipe, Mapping):
        raise ValueError("CIFAR-100 split recipe must be an object")
    if split.get("dataset") != "CIFAR-100-C":
        raise ValueError("CIFAR-100 split recipe requires dataset='CIFAR-100-C'")
    if tuple(split.get("canonical_class_names", ())) != CIFAR100_CLASS_NAMES:
        raise ValueError("CIFAR-100 split canonical class taxonomy does not match CIFAR100C")
    if split.get("taxonomy_sha256") != _CIFAR100_TAXONOMY_SHA256:
        raise ValueError("CIFAR-100 split taxonomy SHA-256 does not match CIFAR100C")
    expected_recipe = {
        "algorithm": _CIFAR100_NAME_RANK_ALGORITHM,
        "input_encoding": "UTF-8",
        "input_format": "salt + NUL + canonical class name",
        "rank_order": "SHA-256 digest ascending, then class name ascending",
        "known_selection": "first 80 ranked class names",
        "unknown_selection": "remaining 20 ranked class names",
    }
    if any(recipe.get(key) != value for key, value in expected_recipe.items()):
        raise ValueError("unsupported or malformed CIFAR-100 split recipe")
    salt = recipe.get("salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError("CIFAR-100 split recipe requires a non-empty salt")
    if split["version"] != salt:
        raise ValueError("CIFAR-100 name-ranked split version and salt must match")
    ranked_ids = _cifar100_name_ranked_ids(salt)
    if tuple(known) != ranked_ids[:80] or tuple(unknown) != ranked_ids[80:]:
        raise ValueError("CIFAR-100 split IDs do not match the declared name-ranking recipe")
    fingerprint = split.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _cifar100_split_fingerprint(split):
        raise ValueError("CIFAR-100 split fingerprint does not verify")


def load_cifar100_open_set_split(
    path: str | Path | None = None, *, expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and validate the repository's versioned CIFAR-100 open-set split."""
    split_path = Path(path) if path is not None else DEFAULT_SPLIT_PATH
    raw = split_path.read_bytes()
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)):
            raise ValueError("expected CIFAR-100 split SHA-256 must be 64 lowercase hexadecimal characters")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("CIFAR-100 split JSON SHA-256 does not match the planned bytes")
    try:
        split = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid CIFAR-100 split JSON: {split_path}") from exc
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
    _validate_cifar100_recipe_metadata(split, known, unknown)
    result = {
        "version": version,
        "known_class_ids": tuple(known),
        "unknown_class_ids": tuple(unknown),
    }
    # v1 has no audited recipe identifiers.  Preserve its historical return
    # shape while propagating verified identifiers from newer split artifacts.
    if split.get("recipe") is not None:
        result["fingerprint"] = split["fingerprint"]
        result["taxonomy_sha256"] = split["taxonomy_sha256"]
    return result


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

    def __init__(self, root, extra=False, severity=5, transform=None, *, split_path=None, split_sha256=None):
        split = load_cifar100_open_set_split(split_path, expected_sha256=split_sha256)
        super().__init__(root, extra=extra, severity=severity, transform=transform)
        all_classes = tuple(self.classes)
        self.open_set_split_version = split["version"]
        if "fingerprint" in split:
            self.open_set_split_fingerprint = split["fingerprint"]
            self.open_set_taxonomy_sha256 = split["taxonomy_sha256"]
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
