"""Depth-estimation evaluation metrics and spatial evaluation masks."""

from mde_transformers.metrics.crops import CropBounds
from mde_transformers.metrics.depth import (
    abs_rel,
    align_median,
    delta1,
    delta2,
    delta3,
    delta_accuracy,
    imagewise_mean,
    rmse,
    rmse_log,
    silog,
    sq_rel,
    valid_depth_mask,
)
from mde_transformers.metrics.nyu import (
    NYU_EIGEN_CROP,
    NYU_NATIVE_IMAGE_SIZE,
    NYUAlignment,
    NYUCrop,
    NYUEvaluationProtocol,
    nyu_eigen_protocol,
    nyu_native_crop_mask,
    raw_metric_protocol,
)

__all__ = [
    "CropBounds",
    "NYUAlignment",
    "NYUCrop",
    "NYUEvaluationProtocol",
    "NYU_EIGEN_CROP",
    "NYU_NATIVE_IMAGE_SIZE",
    "abs_rel",
    "align_median",
    "delta1",
    "delta2",
    "delta3",
    "delta_accuracy",
    "imagewise_mean",
    "nyu_eigen_protocol",
    "nyu_native_crop_mask",
    "rmse",
    "rmse_log",
    "raw_metric_protocol",
    "silog",
    "sq_rel",
    "valid_depth_mask",
]
