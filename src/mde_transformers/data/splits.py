"""Deterministic train-derived development splits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

_MAX_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class TrainDevSplit:
    """Non-overlapping optimization and development partitions of training indices."""

    optimization: tuple[int, ...]
    validation: tuple[int, ...]


def split_train_dev_indices(
    train_indices: Sequence[int],
    *,
    validation_fraction: float,
    seed: int,
) -> TrainDevSplit:
    """Partition supplied training indices deterministically without touching a test split.

    ``floor(len(train_indices) * validation_fraction)`` items are selected for validation using a
    local seeded PyTorch generator. Both returned partitions retain the input ordering. The
    function requires the requested fraction to yield at least one item in each partition.
    """
    indices = tuple(train_indices)
    if len(indices) < 2:
        raise ValueError("at least two training indices are required")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("train_indices must contain integers")
    if len(set(indices)) != len(indices):
        raise ValueError("train_indices must not contain duplicates")
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, int | float):
        raise TypeError("validation_fraction must be a real number")
    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be finite and lie strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")

    validation_count = int(len(indices) * fraction)
    if validation_count == 0 or validation_count == len(indices):
        raise ValueError("validation_fraction must produce non-empty train and validation sets")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    validation_positions = set(permutation[:validation_count])
    optimization = tuple(
        index for position, index in enumerate(indices) if position not in validation_positions
    )
    validation = tuple(
        index for position, index in enumerate(indices) if position in validation_positions
    )
    return TrainDevSplit(optimization=optimization, validation=validation)
