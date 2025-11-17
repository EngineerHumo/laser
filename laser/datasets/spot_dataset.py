"""Dataset and dataloader utilities for spot grading."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

LOGGER = logging.getLogger(__name__)

@dataclass
class SpotSample:
    """Container returned by :class:`SpotDataset`."""

    spot_image: Tensor
    spot_context_image: Tensor
    global_image: Tensor
    label: int
    spot_name: str
    global_name: str


SPOT_MEAN = [0.485, 0.456, 0.406]
SPOT_STD = [0.229, 0.224, 0.225]


SpotTransform = Callable[[Image.Image], Tuple[Tensor, Tensor]]
GlobalTransform = Callable[[Image.Image], Tensor]
PairTransform = Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]


def build_spot_transform(augment: bool = False) -> SpotTransform:
    """Create the transformation pipeline for spot crops.

    The returned callable resizes the image to ``128x128`` pixels, applies
    optional augmentations and yields both the ``64x64`` centre crop (used for
    training) and the full ``128x128`` crop (used for visualisation).
    """

    resize = transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR)
    augmentation: Optional[transforms.Compose]
    if augment:
        augmentation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=90,
                    interpolation=InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0,
                ),
            ]
        )
    else:
        augmentation = None

    to_tensor = transforms.ToTensor()
    normalise = transforms.Normalize(mean=SPOT_MEAN, std=SPOT_STD)

    def transform(image: Image.Image) -> Tuple[Tensor, Tensor]:
        image = resize(image)
        if augmentation is not None:
            image = augmentation(image)
        context_tensor = normalise(to_tensor(image))
        spot_tensor = TF.center_crop(context_tensor, (64, 64))
        return spot_tensor, context_tensor

    return transform


def _default_spot_transform() -> SpotTransform:
    return build_spot_transform(augment=False)


def build_global_transform(output_size: int = 128) -> Callable[[Image.Image], Tensor]:
    """Create the transformation pipeline for global context images."""

    if output_size <= 0:
        raise ValueError("output_size must be a positive integer")

    return transforms.Compose(
        [
            transforms.Resize((output_size, output_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=SPOT_MEAN, std=SPOT_STD),
        ]
    )


def _default_global_transform() -> GlobalTransform:
    return build_global_transform()


def build_shared_hsv_transform(
    hue_delta: float = 0.02,
    saturation_range: Tuple[float, float] = (0.95, 1.05),
    value_range: Tuple[float, float] = (0.95, 1.05),
    probability: float = 1.0,
) -> PairTransform:
    """Create a transform that applies identical HSV jitter to paired images.

    Args:
        hue_delta: Maximum absolute hue shift applied in the range ``[-hue_delta, hue_delta]``.
        saturation_range: Multiplicative range for saturation changes.
        value_range: Multiplicative range for value (brightness) changes.
        probability: Probability of applying the jitter to a given pair.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in the range [0, 1]")

    def transform(spot_img: Image.Image, global_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() >= probability:
            return spot_img, global_img

        hue_adjust = random.uniform(-hue_delta, hue_delta)
        saturation_factor = random.uniform(*saturation_range)
        value_factor = random.uniform(*value_range)

        def _apply(image: Image.Image) -> Image.Image:
            image = TF.adjust_brightness(image, value_factor)
            image = TF.adjust_saturation(image, saturation_factor)
            return TF.adjust_hue(image, hue_adjust)

        return _apply(spot_img), _apply(global_img)

    return transform


class SpotDataset(Dataset[SpotSample]):
    """Dataset yielding spot crops alongside down-sampled global context images."""

    def __init__(
        self,
        root: Path | str = Path("data"),
        spot_transform: Optional[SpotTransform] = None,
        global_transform: Optional[GlobalTransform] = None,
        pair_transform: Optional[PairTransform] = None,
    ) -> None:
        self.root = Path(root)
        self.spot_dir = self.root / "spot"
        self.global_dir = self.root / "image"
        self.labels_path = self.root / "labels.txt"

        if not self.labels_path.exists():
            raise FileNotFoundError(f"labels.txt not found at {self.labels_path}")

        self.spot_transform = spot_transform or _default_spot_transform()
        self.global_transform = global_transform or _default_global_transform()
        self.pair_transform = pair_transform

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

        # ``len(class_to_indices)`` only counts classes that are present in the
        # current dataset split. Some training runs expect the classifier head
        # to allocate logits for every class defined in ``labels.txt`` even if
        # a particular class is missing from the sampled data.  Relying on the
        # observed class count therefore leads to ``out_features`` being too
        # small when the labels are not contiguous (e.g. classes ``0-5`` with
        # one class absent), which in turn triggers device-side asserts when
        # constructing one-hot encodings.  Using the maximum label value keeps
        # ``num_classes`` stable across runs.
        max_label = int(self._labels.max().item())
        if max_label < 0:
            raise ValueError("labels.txt must contain non-negative class indices")
        self._num_classes = max_label + 1

    @property
    def labels(self) -> Tensor:
        return self._labels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> SpotSample:
        return self.load_sample(index, pair_transform=self.pair_transform)

    def load_sample(
        self,
        index: int,
        *,
        spot_transform: Optional[SpotTransform] = None,
        global_transform: Optional[GlobalTransform] = None,
        pair_transform: Optional[PairTransform] = None,
    ) -> SpotSample:
        spot_transform = spot_transform or self.spot_transform
        global_transform = global_transform or self.global_transform
        pair_transform = pair_transform or self.pair_transform

        spot_name, label = self.entries[index]
        spot_path = self.spot_dir / f"{spot_name}.png"
        global_name = spot_name.split("_")[0]
        global_path = self.global_dir / f"{global_name}.png"

        if not spot_path.exists():
            raise FileNotFoundError(f"Spot crop not found: {spot_path}")
        if not global_path.exists():
            raise FileNotFoundError(f"Global context image not found: {global_path}")

        with Image.open(spot_path) as spot_img, Image.open(global_path) as global_img:
            spot_img = spot_img.convert("RGB")
            global_img = global_img.convert("RGB")

            if pair_transform is not None:
                spot_img, global_img = pair_transform(spot_img, global_img)

            spot_tensor, spot_context_tensor = spot_transform(spot_img)
            global_tensor = global_transform(global_img)

        return SpotSample(
            spot_image=spot_tensor,
            spot_context_image=spot_context_tensor,
            global_image=global_tensor,
            label=label,
            spot_name=spot_name,
            global_name=global_name,
        )


class SpotSubsetDataset(Dataset[SpotSample]):
    """Subset view of :class:`SpotDataset` with dedicated transforms."""

    def __init__(
        self,
        dataset: SpotDataset,
        indices: Sequence[int],
        *,
        spot_transform: Optional[SpotTransform] = None,
        global_transform: Optional[GlobalTransform] = None,
        pair_transform: Optional[PairTransform] = None,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.spot_transform = spot_transform
        self.global_transform = global_transform
        self.pair_transform = pair_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> SpotSample:
        base_index = self.indices[index]
        return self.dataset.load_sample(
            base_index,
            spot_transform=self.spot_transform,
            global_transform=self.global_transform,
            pair_transform=self.pair_transform,
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
