"""Evaluator-safe semantic open-set views for CIFAR-100-C."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPEN_SET_SPLIT = "open-set-cifar100-split-v1"


def load_cifar100_split(version=OPEN_SET_SPLIT):
    if version != OPEN_SET_SPLIT:
        raise ValueError(f"unsupported known-class split: {version}")
    payload = json.loads((PROJECT_ROOT / "cfg/research/open-set-cifar100-split-v1.json").read_text(encoding="utf-8"))
    known, unknown = tuple(payload["known_class_ids"]), tuple(payload["unknown_class_ids"])
    if payload.get("split_version") != version or len(known) != 80 or len(unknown) != 20 \
            or set(known) & set(unknown) or set(known) | set(unknown) != set(range(100)):
        raise ValueError("invalid fixed CIFAR-100 open-set split")
    return payload


class OpenSetDomainDataset:
    """Model-facing labels are contiguous known labels or -1; provenance stays private."""
    def __init__(self, source, known_class_ids):
        self.source = source
        self.targets = [int(label) for label in source.Y]
        known_index = {label: index for index, label in enumerate(known_class_ids)}
        self._known_label = [known_index.get(label, -1) for label in self.targets]
        self._is_ood = [label not in known_index for label in self.targets]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        image, _ = self.source[index]
        return image, self._known_label[index]

    def evaluator_metadata(self, index):
        return {"original_label": self.targets[index], "is_ood": self._is_ood[index],
                "known_label_or_minus_one": self._known_label[index]}


class OpenSetCIFAR100C:
    def __init__(self, source, split_version=OPEN_SET_SPLIT):
        split = load_cifar100_split(split_version)
        self.split_version = split_version
        self.known_class_ids = tuple(split["known_class_ids"])
        self.unknown_class_ids = tuple(split["unknown_class_ids"])
        self.classes = [source.classes[index] for index in self.known_class_ids]
        self.num_classes = len(self.classes)
        # Preserve the repository dataset contract: the CSV writer and stream
        # evidence treat environment names as a mutable list.
        self.environments = list(source.environments)
        self.datasets = [OpenSetDomainDataset(dataset, self.known_class_ids) for dataset in source.datasets]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]
