"""Coarse-to-fine residual decoder for hierarchical encoder features.

The encoder's finest feature ``F1`` is stride 4, so the fourth residual stage naturally produces
``D4`` at one-quarter input resolution. The thesis also describes ``D4`` as full resolution; this
implementation resolves that inconsistency explicitly by applying a separate final bilinear
resize from physical ``D4`` to the caller-provided output size. That resize is not a fifth
refinement stage and consumes no encoder feature.

The default constant decoder width of 128 channels is a conservative implementation choice, not
a thesis-specified value. Every width remains configurable. Stage blocks follow the simple
two-3x3-convolution interpretation of the thesis table.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from mde_transformers.models.encoders import EncoderFeatures

FeatureChannels = tuple[int, int, int, int]
DepthStages = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

DEFAULT_DECODER_CHANNELS: FeatureChannels = (128, 128, 128, 128)


@dataclass(frozen=True, slots=True)
class DecoderOutput:
    """Positive depth stages and the explicit full-resolution output.

    ``d1`` through ``d4`` correspond respectively to encoder features ``F4``, ``F3``, ``F2``,
    and ``F1``. :meth:`stage_depths` deliberately excludes ``full_resolution`` so the latter
    cannot accidentally become a fifth multi-scale supervision stage.
    """

    d1: torch.Tensor
    d2: torch.Tensor
    d3: torch.Tensor
    d4: torch.Tensor
    full_resolution: torch.Tensor

    def stage_depths(self) -> DepthStages:
        """Return ``(d1, d2, d3, d4)`` from coarsest to finest refinement stage."""
        return self.d1, self.d2, self.d3, self.d4


class CoarseToFineDepthDecoder(nn.Module):
    """Decode four NCHW encoder features into coarse-to-fine positive depth.

    The feature-channel tuples are ordered ``(F1, F2, F3, F4)``. Each feature is independently
    adapted by a 1x1 convolution. Stage 1 produces a latent scalar from projected ``F4``. Stages
    2--4 resize the preceding physical depth to the exact next-feature size, apply a 3x3
    convolution, and add a learned residual in latent-depth space. Every exposed stage is mapped
    to physical positive depth as ``softplus(latent) + minimum_depth``. Thus refinement is truly
    additive before the documented positive mapping, with no hard clamping.

    No torchvision details are used here; the decoder depends only on :class:`EncoderFeatures`
    or an equivalent four-tensor sequence.
    """

    feature_channels: FeatureChannels
    decoder_channels: FeatureChannels
    minimum_depth: float

    def __init__(
        self,
        feature_channels: FeatureChannels,
        *,
        decoder_channels: FeatureChannels = DEFAULT_DECODER_CHANNELS,
        minimum_depth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_channels = _validate_channel_tuple(feature_channels, "feature_channels")
        self.decoder_channels = _validate_channel_tuple(decoder_channels, "decoder_channels")
        self.minimum_depth = _validate_minimum_depth(minimum_depth)

        self.feature_projections = nn.ModuleList(
            nn.Conv2d(input_channels, output_channels, kernel_size=1)
            for input_channels, output_channels in zip(
                self.feature_channels,
                self.decoder_channels,
                strict=True,
            )
        )
        coarse_channels = self.decoder_channels[3]
        self.coarse_head = nn.Sequential(
            nn.Conv2d(coarse_channels, coarse_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(coarse_channels, 1, kernel_size=3, padding=1),
        )
        self.upsamplers = nn.ModuleList(_DepthUpsampler() for _ in range(3))
        self.refinement_blocks = nn.ModuleList(
            _ResidualRefinementBlock(self.decoder_channels[index]) for index in (2, 1, 0)
        )

    def forward(
        self,
        features: EncoderFeatures | Sequence[torch.Tensor],
        *,
        output_size: tuple[int, int],
    ) -> DecoderOutput:
        """Return four physical depth stages and an explicit requested-size output."""
        feature_tuple = _validate_features(features, self.feature_channels)
        full_size = _validate_output_size(output_size)
        projected = tuple(
            projection(feature)
            for projection, feature in zip(
                self.feature_projections,
                feature_tuple,
                strict=True,
            )
        )

        latent = self.coarse_head(projected[3])
        stage_depths: list[torch.Tensor] = [self._to_physical_depth(latent)]
        for feature_index, upsampler, refinement in zip(
            (2, 1, 0),
            self.upsamplers,
            self.refinement_blocks,
            strict=True,
        ):
            projected_feature = projected[feature_index]
            upsampled_latent = upsampler(
                stage_depths[-1],
                size=projected_feature.shape[-2:],
            )
            upsampled_depth = self._to_physical_depth(upsampled_latent)
            residual = refinement(torch.cat((upsampled_depth, projected_feature), dim=1))
            refined_latent = upsampled_latent + residual
            stage_depths.append(self._to_physical_depth(refined_latent))

        d1, d2, d3, d4 = stage_depths
        full_resolution = functional.interpolate(
            d4,
            size=full_size,
            mode="bilinear",
            align_corners=False,
        )
        return DecoderOutput(d1, d2, d3, d4, full_resolution)

    def _to_physical_depth(self, latent: torch.Tensor) -> torch.Tensor:
        return functional.softplus(latent) + self.minimum_depth


class _DepthUpsampler(nn.Module):
    """Bilinearly resize physical depth to an exact size, then apply a 3x3 convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, depth: torch.Tensor, *, size: tuple[int, int]) -> torch.Tensor:
        resized = functional.interpolate(
            depth,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return self.convolution(resized)


class _ResidualRefinementBlock(nn.Module):
    """Predict only a latent residual using two 3x3 convolutions."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.hidden = nn.Conv2d(feature_channels + 1, feature_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.output = nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1)

    def forward(self, concatenated: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(concatenated)))


def _validate_channel_tuple(channels: FeatureChannels, name: str) -> FeatureChannels:
    if not isinstance(channels, tuple) or len(channels) != 4:
        raise ValueError(f"{name} must be a four-element tuple ordered F1 through F4")
    for channel in channels:
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError(f"{name} values must be integers")
        if channel <= 0:
            raise ValueError(f"{name} values must be positive")
    return channels


def _validate_minimum_depth(minimum_depth: float) -> float:
    if isinstance(minimum_depth, bool) or not isinstance(minimum_depth, int | float):
        raise TypeError("minimum_depth must be a real number")
    value = float(minimum_depth)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("minimum_depth must be finite and positive")
    return value


def _validate_features(
    features: EncoderFeatures | Sequence[torch.Tensor],
    expected_channels: FeatureChannels,
) -> DepthStages:
    feature_tuple = (
        features.as_tuple() if isinstance(features, EncoderFeatures) else tuple(features)
    )
    if len(feature_tuple) != 4:
        raise ValueError("decoder requires exactly four features ordered F1 through F4")

    first = feature_tuple[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("every feature must be a torch.Tensor")
    batch_size = first.shape[0] if first.ndim > 0 else -1
    device = first.device
    dtype = first.dtype
    spatial_sizes: list[tuple[int, int]] = []
    for index, (feature, channels) in enumerate(zip(feature_tuple, expected_channels, strict=True)):
        if not isinstance(feature, torch.Tensor):
            raise TypeError("every feature must be a torch.Tensor")
        if feature.ndim != 4:
            raise ValueError(f"F{index + 1} must have shape [B, C, H, W]")
        if not feature.is_floating_point():
            raise TypeError(f"F{index + 1} must be a floating-point tensor")
        if feature.shape[0] == 0:
            raise ValueError("feature batch must not be empty")
        if feature.shape[0] != batch_size:
            raise ValueError("all features must have the same batch size")
        if feature.shape[1] != channels:
            raise ValueError(f"F{index + 1} must have {channels} channels, got {feature.shape[1]}")
        if feature.shape[2] == 0 or feature.shape[3] == 0:
            raise ValueError("feature spatial dimensions must be positive")
        if feature.device != device:
            raise ValueError("all features must be on the same device")
        if feature.dtype != dtype:
            raise ValueError("all features must have the same dtype")
        spatial_sizes.append((feature.shape[2], feature.shape[3]))

    for finer, coarser in zip(spatial_sizes, spatial_sizes[1:], strict=False):
        if coarser[0] > finer[0] or coarser[1] > finer[1]:
            raise ValueError("features must be ordered spatially from finest F1 to coarsest F4")
    if spatial_sizes[0] == spatial_sizes[-1]:
        raise ValueError("F1 must be spatially finer than F4")
    return feature_tuple


def _validate_output_size(output_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(output_size, tuple) or len(output_size) != 2:
        raise ValueError("output_size must be a (height, width) tuple")
    height, width = output_size
    for value in output_size:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("output_size values must be integers")
        if value <= 0:
            raise ValueError("output_size values must be positive")
    return height, width
