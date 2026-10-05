"""Deterministic NYU preprocessing and DataLoader construction."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from mde_transformers.data import (
    DepthBatch,
    DepthSample,
    NYUDepthV2,
    collate_depth_samples,
    crop_depth_sample,
    normalize_imagenet_sample,
    resize_depth_sample,
    split_train_dev_indices,
)
from mde_transformers.engine.config import NYUExperimentConfig, TrainingConfig
from mde_transformers.metrics import (
    NYU_EIGEN_CROP,
    NYU_NATIVE_IMAGE_SIZE,
    NYUCrop,
    nyu_native_crop_mask,
)


@dataclass(frozen=True, slots=True)
class NYUPreprocess:
    """Apply an optional native training crop, synchronized resize, then RGB normalization.

    The ``nyu_eigen`` training option is an explicit implementation choice, not an assertion that
    this standard literature crop is the thesis's unresolved indoor training crop. Cropping occurs
    in native ``480 x 640`` coordinates before resize and updates RGB, depth, validity, and K.
    """

    image_height: int
    image_width: int
    training_crop: NYUCrop = "none"

    def __call__(self, sample: DepthSample) -> DepthSample:
        spatial_sample = sample
        if self.training_crop == "nyu_eigen":
            _require_native_nyu_size(sample)
            spatial_sample = crop_depth_sample(
                sample,
                top=NYU_EIGEN_CROP.top,
                left=NYU_EIGEN_CROP.left,
                height=NYU_EIGEN_CROP.bottom - NYU_EIGEN_CROP.top,
                width=NYU_EIGEN_CROP.right - NYU_EIGEN_CROP.left,
            )
        elif self.training_crop != "none":
            raise ValueError("training_crop must be 'none' or 'nyu_eigen'")
        resized = resize_depth_sample(
            spatial_sample,
            (self.image_height, self.image_width),
        )
        return normalize_imagenet_sample(resized)


@dataclass(frozen=True, slots=True)
class NYUEvaluationPreprocess:
    """Apply an evaluation mask in native target space before deterministic resize.

    Unlike a training crop, ``nyu_eigen`` does not crop the model input. It intersects the
    authoritative validity mask with the exact native-coordinate evaluation mask, then resizes
    that mask alongside depth using the project's nearest-exact geometry. Literal crop constants
    are therefore never applied to a resized image.
    """

    image_height: int
    image_width: int
    crop: NYUCrop = "none"

    def __call__(self, sample: DepthSample) -> DepthSample:
        native_crop_mask = nyu_native_crop_mask(
            self.crop,
            sample.image_size,
            device=sample.valid_mask.device,
        )
        masked_sample = DepthSample(
            image=sample.image,
            depth=sample.depth,
            valid_mask=sample.valid_mask & native_crop_mask.unsqueeze(0),
            intrinsics=sample.intrinsics,
            sample_id=sample.sample_id,
        )
        resized = resize_depth_sample(masked_sample, (self.image_height, self.image_width))
        return normalize_imagenet_sample(resized)


@dataclass(frozen=True, slots=True)
class NYUTrainDevLoaders:
    """Train-derived loaders and auditable partition sizes.

    ``train_sampler`` derives each permutation only from ``base_seed + epoch``. Calling
    ``set_epoch(N)`` therefore gives identical order in uninterrupted and resumed runs.
    """

    train: DataLoader[DepthBatch]
    dev: DataLoader[DepthBatch]
    train_sampler: EpochShuffleSampler
    official_train_samples: int
    optimization_partition_samples: int
    dev_partition_samples: int
    effective_train_samples: int
    effective_dev_samples: int


class EpochShuffleSampler(Sampler[int]):
    """Deterministically shuffle a fixed-size subset from an explicit one-based epoch."""

    def __init__(self, sample_count: int, *, base_seed: int) -> None:
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        self.sample_count = sample_count
        self.base_seed = base_seed
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        """Select the one-based epoch whose permutation will be generated."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        epoch_seed = (self.base_seed + self.epoch) % (2**63)
        generator = torch.Generator(device="cpu").manual_seed(epoch_seed)
        return iter(torch.randperm(self.sample_count, generator=generator).tolist())

    def __len__(self) -> int:
        return self.sample_count


def create_nyu_train_dev_loaders(
    dataset: NYUDepthV2,
    *,
    training_config: TrainingConfig,
    experiment_config: NYUExperimentConfig,
    dev_dataset: NYUDepthV2 | None = None,
) -> NYUTrainDevLoaders:
    """Partition an official NYU training dataset and construct deterministic loaders.

    The caller must explicitly construct ``NYUDepthV2(split="train")``. Diagnostic limits are
    applied only after the full official training split is partitioned.
    """
    if dataset.split != "train":
        raise ValueError("training loaders require NYUDepthV2(split='train')")
    if dev_dataset is not None and dev_dataset.split != "train":
        raise ValueError("dev loader source must also be NYUDepthV2(split='train')")
    dev_source = dataset if dev_dataset is None else dev_dataset
    official_count = len(dataset)
    if len(dev_source) != official_count:
        raise ValueError("training and dev dataset views must cover the same official split")
    partition = split_train_dev_indices(
        tuple(range(official_count)),
        validation_fraction=experiment_config.validation_fraction,
        seed=training_config.seed,
    )
    optimization_indices = partition.optimization
    dev_indices = partition.validation
    effective_train_indices = _limit_indices(
        optimization_indices,
        experiment_config.max_train_samples,
    )
    effective_dev_indices = _limit_indices(dev_indices, experiment_config.max_dev_samples)

    train_sampler = EpochShuffleSampler(
        len(effective_train_indices),
        base_seed=training_config.seed,
    )
    train_loader = cast(
        DataLoader[DepthBatch],
        DataLoader(
            Subset(dataset, effective_train_indices),
            batch_size=training_config.batch_size,
            sampler=train_sampler,
            num_workers=experiment_config.num_workers,
            collate_fn=collate_depth_samples,
            worker_init_fn=seed_data_worker,
            drop_last=False,
            persistent_workers=False,
        ),
    )
    dev_loader = cast(
        DataLoader[DepthBatch],
        DataLoader(
            Subset(dev_source, effective_dev_indices),
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=experiment_config.num_workers,
            collate_fn=collate_depth_samples,
            worker_init_fn=seed_data_worker,
            drop_last=False,
            persistent_workers=False,
        ),
    )
    return NYUTrainDevLoaders(
        train=train_loader,
        dev=dev_loader,
        train_sampler=train_sampler,
        official_train_samples=official_count,
        optimization_partition_samples=len(optimization_indices),
        dev_partition_samples=len(dev_indices),
        effective_train_samples=len(effective_train_indices),
        effective_dev_samples=len(effective_dev_indices),
    )


def create_nyu_evaluation_loader(
    dataset: NYUDepthV2,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader[DepthBatch]:
    """Construct a deterministic, non-shuffled loader for an explicit NYU test dataset."""
    if dataset.split != "test":
        raise ValueError("final evaluation loader requires NYUDepthV2(split='test')")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return cast(
        DataLoader[DepthBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_depth_samples,
            worker_init_fn=seed_data_worker,
            generator=generator,
            drop_last=False,
            persistent_workers=False,
        ),
    )


def seed_data_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the deterministic PyTorch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _limit_indices(indices: tuple[int, ...], limit: int | None) -> tuple[int, ...]:
    if limit is None:
        return indices
    return indices[:limit]


def _require_native_nyu_size(sample: DepthSample) -> None:
    if sample.image_size != NYU_NATIVE_IMAGE_SIZE:
        raise ValueError("nyu_eigen training crop must be applied to native NYU 480x640 samples")
