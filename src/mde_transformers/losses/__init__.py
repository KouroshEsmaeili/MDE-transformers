"""Architecture-independent losses for supervised monocular depth estimation."""

from mde_transformers.losses.depth import (
    berhu_loss,
    masked_huber_loss,
    masked_l1_loss,
    scale_invariant_log_loss,
)
from mde_transformers.losses.multiscale import (
    CompositeLossResult,
    CompositeLossWeights,
    MultiScaleLossResult,
    composite_depth_loss,
    multi_scale_depth_loss,
)
from mde_transformers.losses.regularization import (
    depth_gradient_consistency_loss,
    edge_aware_smoothness_loss,
    surface_normal_consistency_loss,
)

__all__ = [
    "CompositeLossResult",
    "CompositeLossWeights",
    "MultiScaleLossResult",
    "berhu_loss",
    "composite_depth_loss",
    "depth_gradient_consistency_loss",
    "edge_aware_smoothness_loss",
    "masked_huber_loss",
    "masked_l1_loss",
    "multi_scale_depth_loss",
    "scale_invariant_log_loss",
    "surface_normal_consistency_loss",
]
