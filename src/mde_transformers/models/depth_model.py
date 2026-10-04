"""Complete encoder-decoder monocular depth model assembly.

This module defines the stable model and ``state_dict`` boundary. It performs no image resizing,
value scaling, color conversion, or normalization; preprocessing remains the caller's explicit
responsibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from mde_transformers.models.decoders import (
    DEFAULT_DECODER_CHANNELS,
    CoarseToFineDepthDecoder,
    DecoderOutput,
)
from mde_transformers.models.encoders import SwinEncoder
from mde_transformers.models.encoders.swin import SwinVariant

ModelChannels = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class DepthModelConfig:
    """Immutable architecture-only configuration for :class:`MonocularDepthModel`.

    Torchvision pretrained weights are generic classification initialization, not the thesis's
    SimCLR initialization. Optimizer, dataset, augmentation, loss, and checkpoint choices are
    intentionally outside this configuration.
    """

    encoder_variant: SwinVariant = "tiny"
    encoder_pretrained: bool = False
    decoder_channels: ModelChannels = DEFAULT_DECODER_CHANNELS
    minimum_depth: float = 1e-6

    def __post_init__(self) -> None:
        if self.encoder_variant not in ("tiny", "base"):
            raise ValueError("encoder_variant must be 'tiny' or 'base'")
        if not isinstance(self.encoder_pretrained, bool):
            raise TypeError("encoder_pretrained must be a boolean")
        _validate_decoder_channels(self.decoder_channels)
        _validate_minimum_depth(self.minimum_depth)


class MonocularDepthModel(nn.Module):
    """Compose a Swin encoder and coarse-to-fine positive-depth decoder.

    ``forward`` passes the supplied image unchanged to the encoder and asks the decoder to return
    its separate final output at exactly the original image height and width. No torchvision
    preprocessing metadata or transforms are invoked.
    """

    def __init__(
        self,
        encoder_variant: SwinVariant = "tiny",
        *,
        encoder_pretrained: bool = False,
        decoder_channels: ModelChannels = DEFAULT_DECODER_CHANNELS,
        minimum_depth: float = 1e-6,
    ) -> None:
        super().__init__()
        config = DepthModelConfig(
            encoder_variant=encoder_variant,
            encoder_pretrained=encoder_pretrained,
            decoder_channels=decoder_channels,
            minimum_depth=minimum_depth,
        )
        self._config = config
        self.encoder = SwinEncoder(
            variant=config.encoder_variant,
            pretrained=config.encoder_pretrained,
        )
        self.decoder = CoarseToFineDepthDecoder(
            self.encoder.feature_channels,
            decoder_channels=config.decoder_channels,
            minimum_depth=config.minimum_depth,
        )

    @classmethod
    def from_config(cls, config: DepthModelConfig) -> MonocularDepthModel:
        """Construct a model from an already validated immutable configuration."""
        if not isinstance(config, DepthModelConfig):
            raise TypeError("config must be a DepthModelConfig")
        return cls(
            encoder_variant=config.encoder_variant,
            encoder_pretrained=config.encoder_pretrained,
            decoder_channels=config.decoder_channels,
            minimum_depth=config.minimum_depth,
        )

    @property
    def config(self) -> DepthModelConfig:
        """Return the immutable construction configuration."""
        return self._config

    @property
    def encoder_variant(self) -> SwinVariant:
        return self.encoder.variant

    @property
    def feature_channels(self) -> ModelChannels:
        return self.encoder.feature_channels

    @property
    def feature_strides(self) -> ModelChannels:
        return self.encoder.feature_strides

    @property
    def decoder_channels(self) -> ModelChannels:
        return self.decoder.decoder_channels

    @property
    def minimum_depth(self) -> float:
        return self.decoder.minimum_depth

    def forward(self, image: torch.Tensor) -> DecoderOutput:
        """Predict positive depth at four stages and the exact input spatial size."""
        features = self.encoder(image)
        return self.decoder(features, output_size=(image.shape[-2], image.shape[-1]))

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Change only encoder ``requires_grad`` flags; leave mode and decoder untouched."""
        self.encoder.set_trainable(trainable)


def _validate_decoder_channels(channels: ModelChannels) -> None:
    if not isinstance(channels, tuple) or len(channels) != 4:
        raise ValueError("decoder_channels must be a four-element tuple")
    for channel in channels:
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError("decoder_channels values must be integers")
        if channel <= 0:
            raise ValueError("decoder_channels values must be positive")


def _validate_minimum_depth(minimum_depth: float) -> None:
    if isinstance(minimum_depth, bool) or not isinstance(minimum_depth, int | float):
        raise TypeError("minimum_depth must be a real number")
    if not math.isfinite(float(minimum_depth)) or minimum_depth <= 0:
        raise ValueError("minimum_depth must be finite and positive")
