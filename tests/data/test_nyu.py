from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy.io import savemat

import mde_transformers.data.nyu as nyu_module
from mde_transformers.data import DepthSample, NYUDepthV2, horizontal_flip_depth_sample


@dataclass(frozen=True)
class NYUFixture:
    mat_path: Path
    split_path: Path
    images: np.ndarray
    depths: np.ndarray
    raw_depths: np.ndarray


def _write_splits(path: Path, train: list[int], test: list[int]) -> None:
    savemat(
        path,
        {
            "trainNdxs": np.asarray(train, dtype=np.float64).reshape(-1, 1),
            "testNdxs": np.asarray(test, dtype=np.float64).reshape(-1, 1),
        },
    )


@pytest.fixture
def nyu_fixture(tmp_path: Path) -> NYUFixture:
    sample_count, height, width = 4, 2, 4
    images = np.empty((sample_count, 3, height, width), dtype=np.uint8)
    base_channels = np.asarray(
        [
            [[0, 10, 20, 30], [40, 50, 60, 70]],
            [[80, 90, 100, 110], [120, 130, 140, 150]],
            [[160, 170, 180, 190], [200, 210, 220, 230]],
        ],
        dtype=np.uint8,
    )
    for index in range(sample_count):
        images[index] = base_channels + index

    depths = np.ones((sample_count, height, width), dtype=np.float32)
    depths[0] = np.asarray(
        [[1.0, 0.0, np.nan, 4.0], [5.0, 6.0, 7.0, 12.0]],
        dtype=np.float32,
    )
    depths[1] *= 2.0
    depths[2] *= 3.0
    depths[3] *= 4.0
    raw_depths = depths.copy()
    raw_depths[0] = np.asarray(
        [[0.5, 0.0, 0.0, 2.0], [0.0, 3.0, 0.0, 4.0]],
        dtype=np.float32,
    )

    mat_path = tmp_path / "nyu_depth_v2_labeled.mat"
    with h5py.File(mat_path, "w") as file:
        file.create_dataset("images", data=images.transpose(0, 1, 3, 2))
        file.create_dataset("depths", data=depths.transpose(0, 2, 1))
        file.create_dataset("rawDepths", data=raw_depths.transpose(0, 2, 1))

    split_path = tmp_path / "splits.mat"
    _write_splits(split_path, train=[1, 3], test=[2, 4])
    return NYUFixture(mat_path, split_path, images, depths, raw_depths)


def test_standard_split_lengths_and_matlab_index_conversion(nyu_fixture: NYUFixture) -> None:
    train = NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="train")
    test = NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="test")

    assert len(train) == 2
    assert len(test) == 2
    assert train.source_indices == (0, 2)
    assert test.source_indices == (1, 3)
    assert set(train.source_indices).isdisjoint(test.source_indices)


def test_rgb_order_axis_conversion_and_unit_interval(nyu_fixture: NYUFixture) -> None:
    dataset = NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="train")

    sample = dataset[0]

    expected = torch.from_numpy(nyu_fixture.images[0]).to(torch.float32) / 255.0
    assert isinstance(sample, DepthSample)
    assert sample.image.shape == (3, 2, 4)
    assert sample.image.dtype == torch.float32
    torch.testing.assert_close(sample.image, expected)
    assert sample.image[0, 0, 1].item() == pytest.approx(10.0 / 255.0)
    assert sample.image[1, 0, 1].item() == pytest.approx(90.0 / 255.0)
    assert sample.image[2, 0, 1].item() == pytest.approx(170.0 / 255.0)
    assert 0.0 <= sample.image.min().item() <= sample.image.max().item() <= 1.0
    dataset.close()


def test_depth_remains_metric_and_mask_is_sensor_validity(nyu_fixture: NYUFixture) -> None:
    dataset = NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="train")

    sample = dataset[0]

    expected_depth = torch.from_numpy(nyu_fixture.depths[0]).unsqueeze(0)
    expected_mask = torch.isfinite(expected_depth) & (expected_depth > 0)
    assert sample.depth.shape == (1, 2, 4)
    assert sample.depth.dtype == torch.float32
    torch.testing.assert_close(sample.depth, expected_depth, equal_nan=True)
    torch.testing.assert_close(sample.valid_mask, expected_mask)
    assert sample.depth[0, 1, 3].item() == 12.0
    assert sample.valid_mask[0, 1, 3].item()
    assert sample.intrinsics is None
    dataset.close()


def test_dense_target_can_use_raw_sensor_validity(nyu_fixture: NYUFixture) -> None:
    target_validity = NYUDepthV2(
        nyu_fixture.mat_path,
        nyu_fixture.split_path,
        split="train",
        depth_source="depths",
        validity_source="target",
    )
    raw_validity = NYUDepthV2(
        nyu_fixture.mat_path,
        nyu_fixture.split_path,
        split="train",
        depth_source="depths",
        validity_source="rawDepths",
    )

    dense_sample = target_validity[0]
    sensor_sample = raw_validity[0]

    expected_depth = torch.from_numpy(nyu_fixture.depths[0]).unsqueeze(0)
    expected_raw_mask = torch.from_numpy(nyu_fixture.raw_depths[0]).unsqueeze(0) > 0
    torch.testing.assert_close(dense_sample.depth, expected_depth, equal_nan=True)
    torch.testing.assert_close(sensor_sample.depth, expected_depth, equal_nan=True)
    torch.testing.assert_close(sensor_sample.valid_mask, expected_raw_mask)
    assert dense_sample.valid_mask[0, 1, 0].item()
    assert not sensor_sample.valid_mask[0, 1, 0].item()
    assert sensor_sample.depth[0, 1, 0].item() == 5.0
    target_validity.close()
    raw_validity.close()


def test_raw_depth_target_uses_target_validity(nyu_fixture: NYUFixture) -> None:
    dataset = NYUDepthV2(
        nyu_fixture.mat_path,
        nyu_fixture.split_path,
        split="train",
        depth_source="rawDepths",
        validity_source="target",
    )

    sample = dataset[0]

    expected = torch.from_numpy(nyu_fixture.raw_depths[0]).unsqueeze(0)
    torch.testing.assert_close(sample.depth, expected)
    torch.testing.assert_close(sample.valid_mask, expected > 0)
    dataset.close()


def test_bad_validity_source_is_rejected(nyu_fixture: NYUFixture) -> None:
    with pytest.raises(ValueError, match="validity_source"):
        NYUDepthV2(
            nyu_fixture.mat_path,
            nyu_fixture.split_path,
            split="train",
            validity_source="depths",  # type: ignore[arg-type]
        )


def test_max_samples_is_explicit_deterministic_prefix(nyu_fixture: NYUFixture) -> None:
    complete = NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="train")
    limited = NYUDepthV2(
        nyu_fixture.mat_path,
        nyu_fixture.split_path,
        split="train",
        max_samples=1,
    )

    assert len(complete) == 2
    assert len(limited) == 1
    assert limited.source_indices == complete.source_indices[:1]


@pytest.mark.parametrize("max_samples", [0, -1])
def test_max_samples_must_be_positive(nyu_fixture: NYUFixture, max_samples: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        NYUDepthV2(
            nyu_fixture.mat_path,
            nyu_fixture.split_path,
            split="train",
            max_samples=max_samples,
        )


def test_boolean_max_samples_is_rejected(nyu_fixture: NYUFixture) -> None:
    with pytest.raises(TypeError, match="integer or None"):
        NYUDepthV2(
            nyu_fixture.mat_path,
            nyu_fixture.split_path,
            split="train",
            max_samples=True,
        )


def test_bad_split_name_is_rejected(nyu_fixture: NYUFixture) -> None:
    with pytest.raises(ValueError, match="train.*test"):
        NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="validation")  # type: ignore[arg-type]


def test_negative_index_selects_from_end_of_split(nyu_fixture: NYUFixture) -> None:
    dataset = NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="train")
    assert dataset[-1].sample_id == "nyu_depth_v2/0002"
    with pytest.raises(IndexError):
        dataset[-3]
    dataset.close()


def test_split_index_out_of_bounds_is_rejected(
    nyu_fixture: NYUFixture,
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "out_of_bounds.mat"
    _write_splits(split_path, train=[1, 5], test=[2, 3, 4])
    with pytest.raises(IndexError, match="outside"):
        NYUDepthV2(nyu_fixture.mat_path, split_path, split="train")


def test_duplicate_split_index_is_rejected(nyu_fixture: NYUFixture, tmp_path: Path) -> None:
    split_path = tmp_path / "duplicate.mat"
    _write_splits(split_path, train=[1, 1], test=[2, 3, 4])
    with pytest.raises(ValueError, match="duplicate"):
        NYUDepthV2(nyu_fixture.mat_path, split_path, split="train")


def test_train_test_overlap_is_rejected(nyu_fixture: NYUFixture, tmp_path: Path) -> None:
    split_path = tmp_path / "overlap.mat"
    _write_splits(split_path, train=[1, 2], test=[2, 3, 4])
    with pytest.raises(ValueError, match="overlap"):
        NYUDepthV2(nyu_fixture.mat_path, split_path, split="train")


def test_incomplete_split_union_is_rejected(nyu_fixture: NYUFixture, tmp_path: Path) -> None:
    split_path = tmp_path / "incomplete.mat"
    _write_splits(split_path, train=[1], test=[2, 3])
    with pytest.raises(ValueError, match="does not cover"):
        NYUDepthV2(nyu_fixture.mat_path, split_path, split="train")


def test_official_sample_count_requires_795_train_and_654_test(tmp_path: Path) -> None:
    mat_path = tmp_path / "official_count.mat"
    with h5py.File(mat_path, "w") as file:
        file.create_dataset("images", shape=(1_449, 3, 1, 1), dtype=np.uint8)
        file.create_dataset("depths", shape=(1_449, 1, 1), dtype=np.float32)
    split_path = tmp_path / "wrong_official_counts.mat"
    _write_splits(
        split_path,
        train=list(range(1, 795)),
        test=list(range(795, 1_450)),
    )

    with pytest.raises(ValueError, match="795 train and 654 test"):
        NYUDepthV2(mat_path, split_path, split="train")


def test_optional_transform_is_applied_after_sample_construction(nyu_fixture: NYUFixture) -> None:
    dataset = NYUDepthV2(
        nyu_fixture.mat_path,
        nyu_fixture.split_path,
        split="train",
        transform=horizontal_flip_depth_sample,
    )

    sample = dataset[0]

    expected_depth = torch.from_numpy(nyu_fixture.depths[0]).unsqueeze(0).flip(-1)
    torch.testing.assert_close(sample.depth, expected_depth, equal_nan=True)
    assert sample.sample_id == "nyu_depth_v2/0000"
    dataset.close()


def test_transform_must_return_depth_sample(nyu_fixture: NYUFixture) -> None:
    dataset = NYUDepthV2(
        nyu_fixture.mat_path,
        nyu_fixture.split_path,
        split="train",
        transform=lambda sample: sample.image,  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(TypeError, match="return a DepthSample"):
        dataset[0]
    dataset.close()


def test_hdf5_handle_is_lazy_and_not_pickled(nyu_fixture: NYUFixture) -> None:
    dataset = NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="train")
    assert dataset._h5_file is None

    dataset[0]
    original_handle = dataset._h5_file
    assert original_handle is not None
    assert original_handle.id.valid

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._h5_file is None
    restored[0]
    assert restored._h5_file is not None
    assert restored._h5_file is not original_handle

    restored_handle = restored._h5_file
    dataset.close()
    restored.close()
    assert not original_handle.id.valid
    assert restored_handle is not None
    assert not restored_handle.id.valid


def test_hdf5_handle_is_reopened_when_process_id_changes(
    nyu_fixture: NYUFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = NYUDepthV2(nyu_fixture.mat_path, nyu_fixture.split_path, split="train")
    dataset[0]
    parent_handle = dataset._h5_file
    assert parent_handle is not None
    child_process_id = os.getpid() + 1
    monkeypatch.setattr(nyu_module.os, "getpid", lambda: child_process_id)

    dataset[1]

    assert not parent_handle.id.valid
    assert dataset._h5_file is not None
    assert dataset._h5_file is not parent_handle
    assert dataset._h5_process_id == child_process_id
    dataset.close()
