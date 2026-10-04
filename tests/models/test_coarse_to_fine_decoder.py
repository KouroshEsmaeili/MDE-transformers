"""Tests for the architecture-neutral coarse-to-fine depth decoder."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import nn

from mde_transformers.losses import multi_scale_depth_loss
from mde_transformers.models.decoders import CoarseToFineDepthDecoder, DecoderOutput
from mde_transformers.models.encoders import EncoderFeatures

TINY_CHANNELS = (96, 192, 384, 768)
BASE_CHANNELS = (128, 256, 512, 1024)
TEST_DECODER_CHANNELS = (8, 8, 8, 8)
STANDARD_SIZES = ((32, 40), (16, 20), (8, 10), (4, 5))


def _features(
    channels: tuple[int, int, int, int],
    sizes: Sequence[tuple[int, int]] = STANDARD_SIZES,
    *,
    requires_grad: bool = False,
) -> EncoderFeatures:
    tensors = [
        torch.randn((1, channel, height, width), requires_grad=requires_grad)
        for channel, (height, width) in zip(channels, sizes, strict=True)
    ]
    return EncoderFeatures(*tensors)


def _decoder(
    channels: tuple[int, int, int, int] = TINY_CHANNELS,
) -> CoarseToFineDepthDecoder:
    return CoarseToFineDepthDecoder(
        channels,
        decoder_channels=TEST_DECODER_CHANNELS,
    )


def test_tiny_features_produce_expected_stage_and_full_shapes() -> None:
    decoder = _decoder()

    output = decoder(_features(TINY_CHANNELS), output_size=(128, 160))

    assert isinstance(output, DecoderOutput)
    assert [tuple(depth.shape) for depth in output.stage_depths()] == [
        (1, 1, 4, 5),
        (1, 1, 8, 10),
        (1, 1, 16, 20),
        (1, 1, 32, 40),
    ]
    assert output.full_resolution.shape == (1, 1, 128, 160)
    assert len(output.stage_depths()) == 4


def test_base_features_use_the_same_decoder_class() -> None:
    decoder = _decoder(BASE_CHANNELS)

    output = decoder(_features(BASE_CHANNELS), output_size=(64, 80))

    assert decoder.feature_channels == BASE_CHANNELS
    assert [depth.shape[-2:] for depth in output.stage_depths()] == [
        (4, 5),
        (8, 10),
        (16, 20),
        (32, 40),
    ]
    assert output.full_resolution.shape[-2:] == (64, 80)


def test_feature_projections_are_spatially_identity_one_by_one_convolutions() -> None:
    decoder = _decoder()

    assert [projection.kernel_size for projection in decoder.feature_projections] == [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ]
    assert [projection.in_channels for projection in decoder.feature_projections] == list(
        TINY_CHANNELS
    )
    assert [projection.out_channels for projection in decoder.feature_projections] == list(
        TEST_DECODER_CHANNELS
    )


def test_zero_residual_heads_preserve_upsampled_depth_identity() -> None:
    decoder = _decoder().eval()
    features = _features(TINY_CHANNELS)
    for block in decoder.refinement_blocks:
        nn.init.zeros_(block.output.weight)
        nn.init.zeros_(block.output.bias)

    with torch.no_grad():
        output = decoder(features, output_size=(128, 160))
        projected = tuple(
            projection(feature)
            for projection, feature in zip(
                decoder.feature_projections,
                features,
                strict=True,
            )
        )
        latent = decoder.coarse_head(projected[3])
        expected_stages = [decoder._to_physical_depth(latent)]
        for feature_index, upsampler in zip((2, 1, 0), decoder.upsamplers, strict=True):
            latent = upsampler(expected_stages[-1], size=projected[feature_index].shape[-2:])
            expected_stages.append(decoder._to_physical_depth(latent))

    for actual, expected in zip(output.stage_depths(), expected_stages, strict=True):
        torch.testing.assert_close(actual, expected)


def test_odd_feature_sizes_are_used_as_exact_upsampling_targets() -> None:
    sizes = ((31, 39), (16, 20), (8, 10), (4, 5))
    decoder = _decoder()

    output = decoder(
        _features(TINY_CHANNELS, sizes),
        output_size=(125, 157),
    )

    assert [depth.shape[-2:] for depth in output.stage_depths()] == [
        (4, 5),
        (8, 10),
        (16, 20),
        (31, 39),
    ]
    assert output.full_resolution.shape[-2:] == (125, 157)


def test_decoder_does_not_mutate_input_features() -> None:
    decoder = _decoder()
    features = _features(TINY_CHANNELS)
    originals = tuple(feature.clone() for feature in features)

    decoder(features, output_size=(128, 160))

    for feature, original in zip(features, originals, strict=True):
        torch.testing.assert_close(feature, original)


def test_all_exposed_depth_outputs_are_finite_and_strictly_positive() -> None:
    decoder = _decoder()

    output = decoder(_features(TINY_CHANNELS), output_size=(128, 160))

    for depth in (*output.stage_depths(), output.full_resolution):
        assert torch.isfinite(depth).all()
        assert (depth > 0).all()


def test_all_outputs_backpropagate_to_decoder_and_features() -> None:
    decoder = _decoder()
    features = _features(TINY_CHANNELS, requires_grad=True)

    output = decoder(features, output_size=(64, 80))
    loss = sum(depth.mean() for depth in output.stage_depths()) + output.full_resolution.mean()
    loss.backward()

    parameter_gradients = [parameter.grad for parameter in decoder.parameters()]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in parameter_gradients if gradient is not None
    )
    for feature in features:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()


def test_multiscale_loss_accepts_exactly_the_four_decoder_stages() -> None:
    decoder = _decoder()
    features = _features(TINY_CHANNELS, requires_grad=True)
    target = torch.ones((1, 1, 64, 80))
    mask = torch.ones_like(target, dtype=torch.bool)

    output = decoder(features, output_size=(64, 80))
    result = multi_scale_depth_loss(
        output.stage_depths(),
        target,
        mask,
        (0.1, 0.2, 0.3, 1.0),
    )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert len(result.stage_losses) == 4
    assert all(feature.grad is not None for feature in features)


def test_wrong_feature_count_is_rejected() -> None:
    decoder = _decoder()
    features = _features(TINY_CHANNELS).as_tuple()
    with pytest.raises(ValueError, match="exactly four"):
        decoder(features[:3], output_size=(128, 160))


def test_wrong_feature_channel_count_is_rejected() -> None:
    decoder = _decoder()
    features = list(_features(TINY_CHANNELS).as_tuple())
    features[1] = torch.randn((1, 191, 16, 20))
    with pytest.raises(ValueError, match="F2 must have 192 channels"):
        decoder(features, output_size=(128, 160))


def test_mismatched_feature_batch_is_rejected() -> None:
    decoder = _decoder()
    features = list(_features(TINY_CHANNELS).as_tuple())
    features[3] = torch.randn((2, 768, 4, 5))
    with pytest.raises(ValueError, match="same batch"):
        decoder(features, output_size=(128, 160))


def test_non_floating_feature_is_rejected() -> None:
    decoder = _decoder()
    features = list(_features(TINY_CHANNELS).as_tuple())
    features[2] = torch.ones((1, 384, 8, 10), dtype=torch.int64)
    with pytest.raises(TypeError, match="F3.*floating-point"):
        decoder(features, output_size=(128, 160))


def test_non_nchw_feature_is_rejected() -> None:
    decoder = _decoder()
    features = list(_features(TINY_CHANNELS).as_tuple())
    features[0] = torch.randn((96, 32, 40))
    with pytest.raises(ValueError, match=r"F1.*\[B, C, H, W\]"):
        decoder(features, output_size=(128, 160))


def test_reversed_spatial_hierarchy_is_rejected() -> None:
    decoder = _decoder()
    features = _features(TINY_CHANNELS, ((4, 5), (8, 10), (16, 20), (32, 40)))
    with pytest.raises(ValueError, match="finest F1"):
        decoder(features, output_size=(128, 160))


@pytest.mark.parametrize(
    "output_size",
    [(0, 10), (10, -1), (True, 10), (10.0, 20)],
)
def test_invalid_output_size_is_rejected(output_size: object) -> None:
    decoder = _decoder()
    with pytest.raises((TypeError, ValueError), match="output_size"):
        decoder(
            _features(TINY_CHANNELS),
            output_size=output_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("feature_channels", "decoder_channels", "minimum_depth"),
    [
        ((96, 192, 384), TEST_DECODER_CHANNELS, 1e-6),
        (TINY_CHANNELS, (8, 8, 8, 0), 1e-6),
        (TINY_CHANNELS, TEST_DECODER_CHANNELS, 0.0),
    ],
)
def test_invalid_constructor_configuration_is_rejected(
    feature_channels: object,
    decoder_channels: object,
    minimum_depth: float,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CoarseToFineDepthDecoder(
            feature_channels,  # type: ignore[arg-type]
            decoder_channels=decoder_channels,  # type: ignore[arg-type]
            minimum_depth=minimum_depth,
        )
