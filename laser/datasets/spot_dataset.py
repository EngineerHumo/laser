"""Dataset and dataloader utilities for spot grading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

SpotTransform = Callable[[Image.Image], Tensor]


@dataclass
class SpotSample:
    """Container returned by :class:`SpotDataset`."""

    spot_image: Tensor
    global_image: Tensor
    label: int
    spot_name: str
    global_name: str


def _default_spot_transform() -> SpotTransform:
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _default_global_transform() -> SpotTransform:
    return transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SpotDataset(Dataset[SpotSample]):
    """Dataset yielding spot crops alongside down-sampled global context images."""

    def __init__(
        self,
        root: Path | str = Path("data"),
        spot_transform: Optional[SpotTransform] = None,
        global_transform: Optional[SpotTransform] = None,
    ) -> None:
        self.root = Path(root)
        self.spot_dir = self.root / "spot"
        self.global_dir = self.root / "image"
        self.labels_path = self.root / "labels.txt"

        if not self.labels_path.exists():
            raise FileNotFoundError(f"labels.txt not found at {self.labels_path}")

        self.spot_transform = spot_transform or _default_spot_transform()
        self.global_transform = global_transform or _default_global_transform()

        self.entries: List[Tuple[str, int]] = []
        self.class_to_indices: Dict[int, List[int]] = {}

        with self.labels_path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                parts = line.strip().split()
                if len(parts) != 2:
                    LOGGER.warning("Malformed line %d in %s: %s", idx + 1, self.labels_path, line)
                    continue
                spot_name, label_str = parts
                try:
                    label = int(label_str)
                except ValueError:
                    LOGGER.warning("Non-integer label '%s' in %s", label_str, self.labels_path)
                    continue
                self.entries.append((spot_name, label))
                self.class_to_indices.setdefault(label, []).append(len(self.entries) - 1)

        if not self.entries:
            raise RuntimeError(f"No valid entries found in {self.labels_path}")

        self._labels = torch.tensor([label for _, label in self.entries], dtype=torch.long)
        self._num_classes = len(self.class_to_indices)

    @property
    def labels(self) -> Tensor:
        return self._labels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> SpotSample:
        spot_name, label = self.entries[index]
        spot_path = self.spot_dir / f"{spot_name}.png"
        global_name = spot_name.split("_")[0]
        global_path = self.global_dir / f"{global_name}.png"

        if not spot_path.exists():
            raise FileNotFoundError(f"Spot crop not found: {spot_path}")
        if not global_path.exists():
            raise FileNotFoundError(f"Global context image not found: {global_path}")

        with Image.open(spot_path) as spot_img:
            spot_img = spot_img.convert("RGB")
            spot_tensor = self.spot_transform(spot_img)

        with Image.open(global_path) as global_img:
            global_img = global_img.convert("RGB")
            global_tensor = self.global_transform(global_img)

        return SpotSample(
            spot_image=spot_tensor,
            global_image=global_tensor,
            label=label,
            spot_name=spot_name,
            global_name=global_name,
        )


def create_stratified_folds(labels: Sequence[int], num_folds: int) -> List[List[int]]:
    """Split indices into ``num_folds`` stratified folds."""

    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    from collections import defaultdict

    per_class_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        per_class_indices[int(label)].append(idx)

    folds: List[List[int]] = [[] for _ in range(num_folds)]
    for label, indices in per_class_indices.items():
        indices = list(indices)
        import random

        random.Random(label).shuffle(indices)
        for fold_idx, sample_index in enumerate(indices):
            folds[fold_idx % num_folds].append(sample_index)

    return folds
