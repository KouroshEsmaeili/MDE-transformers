from __future__ import annotations

import pytest
import torch

from mde_transformers.data import DepthBatch, DepthSample, collate_depth_samples


def _sample(identifier: str, *, intrinsics: bool = False, width: int = 3) -> DepthSample:
    image = torch.full((3, 2, width), 0.5, dtype=torch.float32)
    depth = torch.full((1, 2, width), 2.0, dtype=torch.float32)
    mask = torch.ones_like(depth, dtype=torch.bool)
    matrix = torch.eye(3) if intrinsics else None
    return DepthSample(image, depth, mask, matrix, sample_id=identifier)


def test_collate_fixed_size_samples_without_moving_or_changing_dtypes() -> None:
    first = _sample("a")
    second = _sample("b")

    batch = collate_depth_samples([first, second])

    assert isinstance(batch, DepthBatch)
    assert batch.image.shape == (2, 3, 2, 3)
    assert batch.depth.shape == (2, 1, 2, 3)
    assert batch.valid_mask.shape == (2, 1, 2, 3)
    assert batch.image.dtype == torch.float32
    assert batch.depth.dtype == torch.float32
    assert batch.valid_mask.dtype == torch.bool
    assert batch.image.device.type == "cpu"
    assert batch.intrinsics is None
    assert batch.sample_ids == ("a", "b")


def test_collate_stacks_intrinsics_when_every_sample_has_them() -> None:
    batch = collate_depth_samples([_sample("a", intrinsics=True), _sample("b", intrinsics=True)])

    assert batch.intrinsics is not None
    assert batch.intrinsics.shape == (2, 3, 3)
    torch.testing.assert_close(batch.intrinsics, torch.eye(3).repeat(2, 1, 1))


def test_collate_rejects_mixed_intrinsics_and_spatial_sizes() -> None:
    with pytest.raises(ValueError, match="present for every"):
        collate_depth_samples([_sample("a", intrinsics=True), _sample("b")])
    with pytest.raises(ValueError, match="matching tensor shapes"):
        collate_depth_samples([_sample("a"), _sample("b", width=4)])


def test_depth_batch_to_returns_new_batch_and_preserves_identifiers() -> None:
    batch = collate_depth_samples([_sample("a")])

    moved = batch.to(torch.device("cpu"))

    assert moved is not batch
    assert moved.sample_ids == batch.sample_ids
    torch.testing.assert_close(moved.image, batch.image)
    torch.testing.assert_close(moved.depth, batch.depth)
    torch.testing.assert_close(moved.valid_mask, batch.valid_mask)
