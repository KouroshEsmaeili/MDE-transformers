"""Read-only verification for local official NYU Depth V2 labeled files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.io import whosmat

from mde_transformers.data import DepthSample, NYUDepthV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-path", required=True, type=Path)
    parser.add_argument("--split-path", required=True, type=Path)
    parser.add_argument("--depth-source", choices=("depths", "rawDepths"), default="depths")
    parser.add_argument("--validity-source", choices=("target", "rawDepths"), default="target")
    return parser.parse_args()


def summarize_depth(
    target_dataset: h5py.Dataset,
    validity_dataset: h5py.Dataset,
) -> tuple[float, float, float]:
    finite_min = float("inf")
    finite_max = float("-inf")
    valid_count = 0
    value_count = 0
    same_source = target_dataset.name == validity_dataset.name
    for index in range(target_dataset.shape[0]):
        target_depth = np.asarray(target_dataset[index])
        target_finite = np.isfinite(target_depth)
        if target_finite.any():
            finite_values = target_depth[target_finite]
            finite_min = min(finite_min, float(finite_values.min()))
            finite_max = max(finite_max, float(finite_values.max()))
        validity_depth = target_depth if same_source else np.asarray(validity_dataset[index])
        valid_count += int((np.isfinite(validity_depth) & (validity_depth > 0)).sum())
        value_count += validity_depth.size
    if finite_min == float("inf"):
        raise ValueError("depth dataset contains no finite values")
    return finite_min, finite_max, 100.0 * valid_count / value_count


def main() -> int:
    args = parse_args()
    print(f"labeled file exists: {args.mat_path.is_file()} ({args.mat_path})")
    print(f"split file exists: {args.split_path.is_file()} ({args.split_path})")
    if not args.mat_path.is_file() or not args.split_path.is_file():
        return 1

    with h5py.File(args.mat_path, "r") as file:
        print(f"HDF5 keys: {sorted(file.keys())}")
        images = file["images"]
        depths = file[args.depth_source]
        validity_key = args.depth_source if args.validity_source == "target" else "rawDepths"
        validity_depths = file[validity_key]
        if not all(
            isinstance(dataset, h5py.Dataset) for dataset in (images, depths, validity_depths)
        ):
            raise TypeError("images, target depth, and validity depth must be HDF5 datasets")
        print(f"stored images: shape={images.shape}, dtype={images.dtype}")
        print(f"stored {args.depth_source}: shape={depths.shape}, dtype={depths.dtype}")
        print(f"validity source: {args.validity_source} ({validity_key})")
        finite_min, finite_max, valid_percentage = summarize_depth(depths, validity_depths)

    print(f"MATLAB split keys: {[name for name, _, _ in whosmat(args.split_path)]}")
    train = NYUDepthV2(
        args.mat_path,
        args.split_path,
        split="train",
        depth_source=args.depth_source,
        validity_source=args.validity_source,
    )
    test = NYUDepthV2(
        args.mat_path,
        args.split_path,
        split="test",
        depth_source=args.depth_source,
        validity_source=args.validity_source,
    )
    overlap = set(train.source_indices).intersection(test.source_indices)
    print(f"total samples: {train.total_samples}")
    print(f"train samples: {len(train)}")
    print(f"test samples: {len(test)}")
    print(f"train/test overlap: {len(overlap)}")
    print(f"finite depth range: [{finite_min:.6g}, {finite_max:.6g}] meters")
    print(f"valid finite positive depth: {valid_percentage:.3f}%")

    sample = train[0]
    print(f"converted image: shape={tuple(sample.image.shape)}, dtype={sample.image.dtype}")
    print(f"converted depth: shape={tuple(sample.depth.shape)}, dtype={sample.depth.dtype}")
    print(f"DepthSample validation: {isinstance(sample, DepthSample)}")
    train.close()
    test.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
