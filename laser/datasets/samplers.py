"""Custom PyTorch data samplers used by the spot grading project."""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional

from torch.utils.data import BatchSampler

from .spot_dataset import SpotSubsetDataset

LOGGER = logging.getLogger(__name__)


class BalancedBatchSampler(BatchSampler[List[int]]):
    """Batch sampler that appends extra samples for specific classes."""

    def __init__(
        self,
        subset: SpotSubsetDataset,
        batch_size: int,
        extra_classes: Iterable[int],
        extras_per_class: int = 2,
        *,
        drop_last: bool = False,
        generator: Optional[random.Random] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if extras_per_class <= 0:
            raise ValueError("extras_per_class must be a positive integer")

        self.subset = subset
        self.batch_size = batch_size
        self.extra_classes: List[int] = list(extra_classes)
        self.extras_per_class = extras_per_class
        self.drop_last = drop_last
        self.rng = generator or random

        labels = subset.dataset.labels
        class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for subset_idx, base_index in enumerate(subset.indices):
            label = int(labels[base_index].item())
            class_to_indices[label].append(subset_idx)

        self.class_to_subset_indices = class_to_indices
        self._warned_missing_classes: Dict[int, bool] = {}

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.subset)))
        self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            base_batch = indices[start : start + self.batch_size]
            if len(base_batch) < self.batch_size and self.drop_last:
                break

            batch = list(base_batch)
            base_set = set(base_batch)
            for cls in self.extra_classes:
                subset_indices = self.class_to_subset_indices.get(cls)
                if not subset_indices:
                    if not self._warned_missing_classes.get(cls):
                        LOGGER.warning(
                            "BalancedBatchSampler: no samples available for class %s", cls
                        )
                        self._warned_missing_classes[cls] = True
                    continue

                available = [idx for idx in subset_indices if idx not in base_set]
                self.rng.shuffle(available)

                chosen: List[int]
                if len(available) >= self.extras_per_class:
                    chosen = available[: self.extras_per_class]
                else:
                    chosen = list(available)
                    while len(chosen) < self.extras_per_class:
                        chosen.append(self.rng.choice(subset_indices))

                batch.extend(chosen)

            yield batch

    def __len__(self) -> int:
        total = len(self.subset)
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)
