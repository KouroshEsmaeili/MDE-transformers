from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from mde_transformers.data import DepthSample
from mde_transformers.engine import (
    EpochShuffleSampler,
    NYUExperimentConfig,
    TrainingConfig,
    create_nyu_train_dev_loaders,
)


class _SyntheticNYU(Dataset[DepthSample]):
    def __init__(self, split: str = "train") -> None:
        self.split = split

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> DepthSample:
        image = torch.full((3, 2, 2), index / 10)
        depth = torch.full((1, 2, 2), float(index + 1))
        mask = torch.ones_like(depth, dtype=torch.bool)
        return DepthSample(image, depth, mask, sample_id=str(index))


def test_epoch_shuffle_is_resume_equivalent_without_serialized_generator_state() -> None:
    uninterrupted = EpochShuffleSampler(20, base_seed=2025)
    uninterrupted.set_epoch(7)
    expected_epoch_seven = list(uninterrupted)

    resumed = EpochShuffleSampler(20, base_seed=2025)
    resumed.set_epoch(7)
    resumed_epoch_seven = list(resumed)

    uninterrupted.set_epoch(8)
    assert resumed_epoch_seven == expected_epoch_seven
    assert sorted(resumed_epoch_seven) == list(range(20))
    assert list(uninterrupted) != expected_epoch_seven


def test_epoch_shuffle_repeated_iteration_is_stable_within_epoch() -> None:
    sampler = EpochShuffleSampler(10, base_seed=3)
    sampler.set_epoch(2)
    assert list(sampler) == list(sampler)


def test_train_dev_loaders_partition_only_supplied_training_dataset() -> None:
    loaders = create_nyu_train_dev_loaders(
        _SyntheticNYU(),  # type: ignore[arg-type]
        training_config=TrainingConfig(epochs=2, batch_size=2, seed=9),
        experiment_config=NYUExperimentConfig(
            validation_fraction=0.2,
            image_height=2,
            image_width=2,
        ),
    )
    loaders.train_sampler.set_epoch(1)
    train_ids = {identifier for batch in loaders.train for identifier in batch.sample_ids}
    dev_ids = [identifier for batch in loaders.dev for identifier in batch.sample_ids]

    assert loaders.official_train_samples == 10
    assert loaders.optimization_partition_samples == 8
    assert loaders.dev_partition_samples == 2
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.union(dev_ids) == {str(index) for index in range(10)}
    assert dev_ids == [identifier for batch in loaders.dev for identifier in batch.sample_ids]


def test_training_loader_rejects_official_test_dataset() -> None:
    with pytest.raises(ValueError, match="split='train'"):
        create_nyu_train_dev_loaders(
            _SyntheticNYU(split="test"),  # type: ignore[arg-type]
            training_config=TrainingConfig(epochs=2),
            experiment_config=NYUExperimentConfig(image_height=2, image_width=2),
        )
