"""Torchvision Swin adapters with an architecture-neutral feature contract.

The maintained torchvision implementation operates on NHWC tensors after patch embedding. This
module contains that backend-specific detail and exposes exactly four NCHW feature maps. Image
normalization is intentionally external to the encoder.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision.models import Swin_B_Weights, Swin_T_Weights, swin_b, swin_t
from torchvision.models.swin_transformer import (
    PatchMerging,
    SwinTransformer,
    SwinTransformerBlock,
)
from torchvision.ops.misc import Permute

SwinVariant = Literal["tiny", "base"]

_STAGE_INDICES = (1, 3, 5, 7)
_MERGE_INDICES = (2, 4, 6)
_EXPECTED_CHANNELS: dict[SwinVariant, tuple[int, int, int, int]] = {
    "tiny": (96, 192, 384, 768),
    "base": (128, 256, 512, 1024),
}
_EXPECTED_STRIDES = (4, 8, 16, 32)
_EXPECTED_PATCH_SIZE = (4, 4)
_EXPECTED_WINDOW_SIZE = (7, 7)


@dataclass(frozen=True, slots=True)
class EncoderFeatures:
    """Four NCHW feature maps ordered from stride 4 through stride 32."""

    f1: torch.Tensor
    f2: torch.Tensor
    f3: torch.Tensor
    f4: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(f1, f2, f3, f4)`` in increasing semantic depth."""
        return self.f1, self.f2, self.f3, self.f4

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.as_tuple())

    def __len__(self) -> int:
        return 4


class SwinEncoder(nn.Module):
    """Expose canonical torchvision Swin-Tiny or Swin-Base feature stages.

    Args:
        variant: Explicit canonical family, ``"tiny"`` or ``"base"``.
        pretrained: Load torchvision's official generic classification initialization when true.
            These weights are not the thesis's SimCLR weights and may require a network download.
            ``False`` is fully functional and never requests weights.

    Input tensors are floating point with shape ``[B, 3, H, W]``. No resizing or normalization is
    performed. Rectangular inputs are supported by torchvision's internal shifted-window padding.
    For sizes divisible by 32, the returned spatial resolutions are exactly strides 4, 8, 16, and
    32. Stage outputs are taken after each maintained torchvision Swin stage and converted from the
    backend's NHWC layout to the public NCHW contract.
    """

    variant: SwinVariant
    feature_channels: tuple[int, int, int, int]
    feature_strides: tuple[int, int, int, int]
    patch_size: tuple[int, int]
    window_size: tuple[int, int]

    def __init__(self, variant: SwinVariant, *, pretrained: bool = False) -> None:
        super().__init__()
        if variant not in ("tiny", "base"):
            raise ValueError("variant must be 'tiny' or 'base'")
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be a boolean")

        backbone = _create_torchvision_backbone(variant, pretrained)
        metadata = _inspect_backend(backbone, variant)
        self.variant = variant
        self.feature_channels = metadata.feature_channels
        self.feature_strides = metadata.feature_strides
        self.patch_size = metadata.patch_size
        self.window_size = metadata.window_size
        self._features = backbone.features

    def forward(self, image: torch.Tensor) -> EncoderFeatures:
        """Return four NCHW feature maps without hidden preprocessing or gradient policy."""
        _validate_input(image, self.patch_size)
        batch_size, _, input_height, input_width = image.shape
        expected_height = input_height // self.patch_size[0]
        expected_width = input_width // self.patch_size[1]
        internal = image
        outputs: list[torch.Tensor] = []

        for index, module in enumerate(self._features):
            internal = module(internal)
            if index not in _STAGE_INDICES:
                continue
            stage = len(outputs)
            expected_shape = (
                batch_size,
                expected_height,
                expected_width,
                self.feature_channels[stage],
            )
            if internal.shape != expected_shape:
                raise RuntimeError(
                    "torchvision Swin stage output violated the validated NHWC feature contract: "
                    f"stage {stage + 1} expected {expected_shape}, got {tuple(internal.shape)}"
                )
            outputs.append(internal.permute(0, 3, 1, 2).contiguous())
            expected_height = (expected_height + 1) // 2
            expected_width = (expected_width + 1) // 2

        if len(outputs) != 4:
            raise RuntimeError(f"torchvision Swin returned {len(outputs)} stages; expected four")
        return EncoderFeatures(*outputs)

    def set_trainable(self, trainable: bool) -> None:
        """Set only parameter ``requires_grad`` flags; do not alter train/eval mode."""
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean")
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)


@dataclass(frozen=True, slots=True)
class _BackendMetadata:
    feature_channels: tuple[int, int, int, int]
    feature_strides: tuple[int, int, int, int]
    patch_size: tuple[int, int]
    window_size: tuple[int, int]


def _create_torchvision_backbone(variant: SwinVariant, pretrained: bool) -> SwinTransformer:
    if variant == "tiny":
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        return swin_t(weights=weights)
    weights = Swin_B_Weights.DEFAULT if pretrained else None
    return swin_b(weights=weights)


def _inspect_backend(backbone: SwinTransformer, variant: SwinVariant) -> _BackendMetadata:
    features = backbone.features
    if len(features) != 8:
        raise RuntimeError("torchvision Swin features hierarchy must contain exactly eight modules")

    patch_embedding = features[0]
    if not isinstance(patch_embedding, nn.Sequential) or len(patch_embedding) != 3:
        raise RuntimeError("torchvision Swin patch embedding structure has changed")
    projection = patch_embedding[0]
    permutation = patch_embedding[1]
    normalization = patch_embedding[2]
    if not isinstance(projection, nn.Conv2d):
        raise RuntimeError("torchvision Swin patch projection must be Conv2d")
    if not isinstance(permutation, Permute) or tuple(permutation.dims) != (0, 2, 3, 1):
        raise RuntimeError("torchvision Swin patch embedding must convert NCHW to NHWC")
    if not isinstance(normalization, nn.LayerNorm):
        raise RuntimeError("torchvision Swin patch embedding must end with LayerNorm")

    patch_size = tuple(projection.kernel_size)
    patch_stride = tuple(projection.stride)
    if patch_size != _EXPECTED_PATCH_SIZE or patch_stride != _EXPECTED_PATCH_SIZE:
        raise RuntimeError(
            f"torchvision Swin patch size/stride must be {_EXPECTED_PATCH_SIZE}, "
            f"got kernel {patch_size} and stride {patch_stride}"
        )

    channels: list[int] = []
    observed_window_sizes: set[tuple[int, int]] = set()
    for stage_index in _STAGE_INDICES:
        stage = features[stage_index]
        if not isinstance(stage, nn.Sequential) or len(stage) == 0:
            raise RuntimeError("torchvision Swin feature stage structure has changed")
        stage_channel: int | None = None
        for block in stage:
            if not isinstance(block, SwinTransformerBlock):
                raise RuntimeError("torchvision Swin stage contains an unexpected module")
            normalized_shape = tuple(block.norm1.normalized_shape)
            if len(normalized_shape) != 1:
                raise RuntimeError("torchvision Swin block channel metadata is invalid")
            block_channel = normalized_shape[0]
            if stage_channel is None:
                stage_channel = block_channel
            elif block_channel != stage_channel:
                raise RuntimeError("torchvision Swin stage contains inconsistent channel counts")
            observed_window_sizes.add(tuple(block.attn.window_size))
        if stage_channel is None:
            raise RuntimeError("torchvision Swin feature stage must not be empty")
        channels.append(stage_channel)

    feature_channels = tuple(channels)
    expected_channels = _EXPECTED_CHANNELS[variant]
    if feature_channels != expected_channels:
        raise RuntimeError(
            f"torchvision Swin-{variant} channels must be {expected_channels}, "
            f"got {feature_channels}"
        )
    if projection.in_channels != 3 or projection.out_channels != feature_channels[0]:
        raise RuntimeError("torchvision Swin patch projection channel contract has changed")
    if tuple(normalization.normalized_shape) != (feature_channels[0],):
        raise RuntimeError("torchvision Swin patch normalization channel contract has changed")

    for merge_index, input_channels, output_channels in zip(
        _MERGE_INDICES,
        feature_channels[:-1],
        feature_channels[1:],
        strict=True,
    ):
        merge = features[merge_index]
        if not isinstance(merge, PatchMerging):
            raise RuntimeError("torchvision Swin hierarchy must use PatchMerging between stages")
        if merge.reduction.in_features != 4 * input_channels:
            raise RuntimeError("torchvision Swin PatchMerging input contract has changed")
        if merge.reduction.out_features != output_channels:
            raise RuntimeError("torchvision Swin PatchMerging output contract has changed")

    if observed_window_sizes != {_EXPECTED_WINDOW_SIZE}:
        raise RuntimeError(
            f"torchvision Swin window size must be {_EXPECTED_WINDOW_SIZE}, "
            f"got {sorted(observed_window_sizes)}"
        )
    feature_strides = tuple(patch_stride[0] * 2**stage for stage in range(4))
    if patch_stride[0] != patch_stride[1] or feature_strides != _EXPECTED_STRIDES:
        raise RuntimeError(
            f"torchvision Swin feature strides must be {_EXPECTED_STRIDES}, got {feature_strides}"
        )
    return _BackendMetadata(
        feature_channels=feature_channels,
        feature_strides=feature_strides,
        patch_size=patch_size,
        window_size=next(iter(observed_window_sizes)),
    )


def _validate_input(image: torch.Tensor, patch_size: tuple[int, int]) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 4:
        raise ValueError("image must have shape [B, 3, H, W]")
    if image.shape[1] != 3:
        raise ValueError("image must have exactly three RGB channels")
    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor")
    if image.shape[0] == 0:
        raise ValueError("image batch must not be empty")
    if image.shape[2] < patch_size[0] or image.shape[3] < patch_size[1]:
        raise ValueError("image height and width must be at least the patch size")
