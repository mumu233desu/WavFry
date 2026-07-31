"""Epoch-aware deterministic sampler for ordinary shuffled training."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler[int]):
    """Shuffle dataset indices deterministically from seed, fold, and epoch."""

    def __init__(self, dataset, *, seed: int, fold_index: int = 0) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.fold_index = int(fold_index)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.fold_index * 100_000 + self.epoch)
        yield from torch.randperm(len(self.dataset), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.dataset)
