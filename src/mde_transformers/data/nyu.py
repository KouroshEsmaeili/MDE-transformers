"""NYU Depth V2 labeled-dataset reader.

The official labeled MATLAB file stores logical arrays as ``images: H x W x 3 x N`` and
``depths``/``rawDepths: H x W x N``. MATLAB v7.3/HDF5 dimension order appears reversed through
h5py, so this module validates ``[N, 3, W, H]`` and ``[N, W, H]`` storage and explicitly converts
it to PyTorch ``[3, H, W]`` and ``[1, H, W]`` tensors.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from mde_transformers.data.sample import DepthSample

NYUSplit = Literal["train", "test"]
NYUDepthSource = Literal["depths", "rawDepths"]
NYUValiditySource = Literal["target", "rawDepths"]
DepthTransform = Callable[[DepthSample], DepthSample]

_IMAGE_KEY = "images"
_TRAIN_KEY = "trainNdxs"
_TEST_KEY = "testNdxs"
_OFFICIAL_SAMPLE_COUNT = 1_449
_OFFICIAL_TRAIN_COUNT = 795
_OFFICIAL_TEST_COUNT = 654


class NYUDepthV2(Dataset[DepthSample]):
    """Read the official NYU Depth V2 labeled HDF5/MATLAB dataset.

    Args:
        mat_path: Explicit path to ``nyu_depth_v2_labeled.mat``.
        split_path: Explicit path to the official ``splits.mat`` metadata.
        split: Standard labeled split, either ``"train"`` or ``"test"``.
        depth_source: ``"depths"`` selects the official in-painted dense depths used by the
            labeled benchmark; ``"rawDepths"`` selects projected depths before hole filling.
            Both sources are stored in meters.
        validity_source: ``"target"`` derives validity from the selected depth target;
            ``"rawDepths"`` derives validity from actual projected sensor observations without
            changing the selected target values.
        transform: Optional synchronized transform applied to a constructed :class:`DepthSample`.
        max_samples: Optional deterministic prefix length applied after split selection. ``None``
            keeps the complete split.

    Images are returned as RGB float32 tensors in ``[0, 1]``. Depth is returned as float32 in
    meters without clipping or normalization. The dataset mask is finite positive validity from
    the explicitly selected validity source and does not impose benchmark evaluation bounds.
    Intrinsics remain ``None`` until an authoritative calibration choice for the labeled aligned
    images is incorporated.

    HDF5 handles are opened lazily and per process so DataLoader workers do not share a handle
    inherited from the parent process.
    """

    def __init__(
        self,
        mat_path: str | Path,
        split_path: str | Path,
        *,
        split: NYUSplit,
        depth_source: NYUDepthSource = "depths",
        validity_source: NYUValiditySource = "target",
        transform: DepthTransform | None = None,
        max_samples: int | None = None,
    ) -> None:
        self._h5_file: h5py.File | None = None
        self._h5_process_id: int | None = None
        self.mat_path = Path(mat_path).expanduser()
        self.split_path = Path(split_path).expanduser()
        _validate_file(self.mat_path, "mat_path")
        _validate_file(self.split_path, "split_path")

        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if depth_source not in ("depths", "rawDepths"):
            raise ValueError("depth_source must be 'depths' or 'rawDepths'")
        if validity_source not in ("target", "rawDepths"):
            raise ValueError("validity_source must be 'target' or 'rawDepths'")
        if max_samples is not None:
            if isinstance(max_samples, bool) or not isinstance(max_samples, int):
                raise TypeError("max_samples must be an integer or None")
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
        if transform is not None and not callable(transform):
            raise TypeError("transform must be callable or None")

        self.split: NYUSplit = split
        self.depth_source: NYUDepthSource = depth_source
        self.validity_source: NYUValiditySource = validity_source
        self.transform = transform
        self.total_samples, self.image_shape, self.depth_shape = _inspect_labeled_file(
            self.mat_path,
            depth_source,
            validity_source,
        )
        train_indices, test_indices = _load_and_validate_splits(
            self.split_path,
            self.total_samples,
        )
        selected_indices = train_indices if split == "train" else test_indices
        if max_samples is not None:
            selected_indices = selected_indices[:max_samples]
        self._source_indices = selected_indices

    def __len__(self) -> int:
        """Return the selected split length after any explicit debug limit."""
        return len(self._source_indices)

    def __getitem__(self, index: int) -> DepthSample:
        """Load one sample and optionally apply the configured synchronized transform."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        source_index = self._source_indices[index]
        file = self._get_h5_file()
        images = _require_dataset(file, _IMAGE_KEY)
        depths = _require_dataset(file, self.depth_source)

        stored_image = np.asarray(images[source_index])
        stored_depth = np.asarray(depths[source_index])
        image_array = np.ascontiguousarray(stored_image.transpose(0, 2, 1))
        depth_array = np.ascontiguousarray(stored_depth.transpose(1, 0))

        image = torch.from_numpy(image_array).to(dtype=torch.float32) / 255.0
        depth = torch.from_numpy(depth_array).to(dtype=torch.float32).unsqueeze(0)
        validity_depth = depth
        if self.validity_source == "rawDepths" and self.depth_source != "rawDepths":
            validity_dataset = _require_dataset(file, "rawDepths")
            stored_validity = np.asarray(validity_dataset[source_index])
            validity_array = np.ascontiguousarray(stored_validity.transpose(1, 0))
            validity_depth = torch.from_numpy(validity_array).to(dtype=torch.float32).unsqueeze(0)
        valid_mask = torch.isfinite(validity_depth) & (validity_depth > 0)
        sample = DepthSample(
            image=image,
            depth=depth,
            valid_mask=valid_mask,
            intrinsics=None,
            sample_id=f"nyu_depth_v2/{source_index:04d}",
        )
        if self.transform is None:
            return sample
        transformed = self.transform(sample)
        if not isinstance(transformed, DepthSample):
            raise TypeError("transform must return a DepthSample")
        return transformed

    @property
    def source_indices(self) -> tuple[int, ...]:
        """Return immutable zero-based indices into the 1,449-image labeled file."""
        return self._source_indices

    def close(self) -> None:
        """Close this process's lazily opened HDF5 handle, if present."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._h5_process_id = None

    def __getstate__(self) -> dict[str, object]:
        """Discard open HDF5 state when pickling for a spawned worker."""
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_h5_process_id"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def _get_h5_file(self) -> h5py.File:
        process_id = os.getpid()
        if self._h5_file is not None and self._h5_process_id != process_id:
            self.close()
        if self._h5_file is None:
            self._h5_file = h5py.File(self.mat_path, "r")
            self._h5_process_id = process_id
        return self._h5_file


def _inspect_labeled_file(
    mat_path: Path,
    depth_source: NYUDepthSource,
    validity_source: NYUValiditySource,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    with h5py.File(mat_path, "r") as file:
        images = _require_dataset(file, _IMAGE_KEY)
        depths = _require_dataset(file, depth_source)
        image_shape = tuple(int(size) for size in images.shape)
        depth_shape = tuple(int(size) for size in depths.shape)

        if images.dtype != np.dtype(np.uint8):
            raise TypeError("official NYU images dataset must have uint8 dtype")
        if len(image_shape) != 4 or image_shape[1] != 3:
            raise ValueError("NYU images must have HDF5 shape [N, 3, W, H]")
        if image_shape[0] == 0 or image_shape[2] == 0 or image_shape[3] == 0:
            raise ValueError("NYU datasets must have non-empty sample and spatial dimensions")
        _validate_depth_dataset(depths, depth_source, image_shape)
        if validity_source == "rawDepths" and depth_source != "rawDepths":
            raw_depths = _require_dataset(file, "rawDepths")
            _validate_depth_dataset(raw_depths, "rawDepths", image_shape)
        return image_shape[0], image_shape, depth_shape


def _validate_depth_dataset(
    depths: h5py.Dataset,
    name: str,
    image_shape: tuple[int, ...],
) -> None:
    depth_shape = tuple(int(size) for size in depths.shape)
    if not np.issubdtype(depths.dtype, np.floating):
        raise TypeError(f"NYU {name} dataset must have floating-point dtype")
    if len(depth_shape) != 3:
        raise ValueError(f"NYU {name} must have HDF5 shape [N, W, H]")
    if image_shape[0] != depth_shape[0] or image_shape[2:] != depth_shape[1:]:
        raise ValueError(f"NYU image and {name} sample/spatial dimensions do not agree")


def _load_and_validate_splits(
    split_path: Path,
    total_samples: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    split_data = loadmat(split_path, variable_names=(_TRAIN_KEY, _TEST_KEY))
    missing_keys = [key for key in (_TRAIN_KEY, _TEST_KEY) if key not in split_data]
    if missing_keys:
        raise KeyError(f"split metadata is missing keys: {', '.join(missing_keys)}")

    train_indices = _matlab_indices_to_python(split_data[_TRAIN_KEY], _TRAIN_KEY, total_samples)
    test_indices = _matlab_indices_to_python(split_data[_TEST_KEY], _TEST_KEY, total_samples)
    overlap = set(train_indices).intersection(test_indices)
    if overlap:
        raise ValueError("NYU train and test splits overlap")
    if len(set(train_indices).union(test_indices)) != total_samples:
        raise ValueError("NYU train/test split union does not cover every labeled sample")
    if total_samples == _OFFICIAL_SAMPLE_COUNT and (
        len(train_indices) != _OFFICIAL_TRAIN_COUNT or len(test_indices) != _OFFICIAL_TEST_COUNT
    ):
        raise ValueError("official NYU split must contain 795 train and 654 test samples")
    return train_indices, test_indices


def _matlab_indices_to_python(
    values: object,
    name: str,
    total_samples: int,
) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric indices")

    flat = array.reshape(-1)
    if not np.isfinite(flat).all() or not np.equal(flat, np.floor(flat)).all():
        raise ValueError(f"{name} must contain finite integer-valued indices")
    one_based = flat.astype(np.int64)
    if (one_based < 1).any() or (one_based > total_samples).any():
        raise IndexError(f"{name} contains an index outside the labeled dataset")

    zero_based = tuple(int(value) for value in one_based - 1)
    if len(set(zero_based)) != len(zero_based):
        raise ValueError(f"{name} contains duplicate indices")
    return zero_based


def _require_dataset(file: h5py.File, key: str) -> h5py.Dataset:
    if key not in file:
        raise KeyError(f"NYU labeled file is missing '{key}'")
    value = file[key]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"NYU labeled key '{key}' is not an HDF5 dataset")
    return value


def _validate_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
