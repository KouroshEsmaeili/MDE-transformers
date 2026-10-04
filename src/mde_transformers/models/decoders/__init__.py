"""Depth decoder interfaces."""

from mde_transformers.models.decoders.coarse_to_fine import (
    DEFAULT_DECODER_CHANNELS,
    CoarseToFineDepthDecoder,
    DecoderOutput,
)

__all__ = ["DEFAULT_DECODER_CHANNELS", "CoarseToFineDepthDecoder", "DecoderOutput"]
