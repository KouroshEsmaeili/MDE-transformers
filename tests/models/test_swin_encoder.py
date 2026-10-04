"""CPU-only tests for the torchvision Swin encoder adapter."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from mde_transformers.models.encoders import EncoderFeatures, SwinEncoder


@pytest.fixture(scope="module")
def tiny_encoder() -> Iterator[SwinEncoder]:
    encoder = SwinEncoder("tiny", pretrained=False)
    encoder.eval()
    yield encoder


@pytest.fixture(scope="module")
def base_encoder() -> Iterator[SwinEncoder]:
    encoder = SwinEncoder("base", pretrained=False)
    encoder.eval()
    yield encoder


def test_tiny_metadata_is_canonical_and_backend_validated(tiny_encoder: SwinEncoder) -> None:
    assert tiny_encoder.variant == "tiny"
    assert tiny_encoder.feature_channels == (96, 192, 384, 768)
    assert tiny_encoder.feature_strides == (4, 8, 16, 32)
    assert tiny_encoder.patch_size == (4, 4)
    assert tiny_encoder.window_size == (7, 7)
    assert sum(parameter.numel() for parameter in tiny_encoder.parameters()) == 27_517_818


def test_base_metadata_is_canonical_and_backend_validated(base_encoder: SwinEncoder) -> None:
    assert base_encoder.variant == "base"
    assert base_encoder.feature_channels == (128, 256, 512, 1024)
    assert base_encoder.feature_strides == (4, 8, 16, 32)
    assert base_encoder.patch_size == (4, 4)
    assert base_encoder.window_size == (7, 7)
    assert sum(parameter.numel() for parameter in base_encoder.parameters()) == 86_741_176


def test_tiny_rectangular_forward_returns_ordered_nchw_features(
    tiny_encoder: SwinEncoder,
) -> None:
    image = torch.randn((1, 3, 128, 160))

    with torch.no_grad():
        features = tiny_encoder(image)

    assert isinstance(features, EncoderFeatures)
    assert len(features) == 4
    assert [tuple(feature.shape) for feature in features] == [
        (1, 96, 32, 40),
        (1, 192, 16, 20),
        (1, 384, 8, 10),
        (1, 768, 4, 5),
    ]
    for feature, channels, stride in zip(
        features,
        tiny_encoder.feature_channels,
        tiny_encoder.feature_strides,
        strict=True,
    ):
        assert feature.shape[0] == image.shape[0]
        assert feature.shape[1] == channels
        assert feature.shape[-2:] == (image.shape[-2] // stride, image.shape[-1] // stride)
        assert feature.is_contiguous()


def test_tiny_backward_reaches_all_trainable_parameters(tiny_encoder: SwinEncoder) -> None:
    tiny_encoder.set_trainable(True)
    tiny_encoder.zero_grad(set_to_none=True)
    image = torch.randn((1, 3, 32, 64))

    features = tiny_encoder(image)
    loss = sum(feature.square().mean() for feature in features)
    loss.backward()

    gradients = [
        parameter.grad for parameter in tiny_encoder.parameters() if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_set_trainable_changes_only_requires_grad(tiny_encoder: SwinEncoder) -> None:
    original_training_mode = tiny_encoder.training

    tiny_encoder.set_trainable(False)
    assert not any(parameter.requires_grad for parameter in tiny_encoder.parameters())
    assert tiny_encoder.training is original_training_mode

    tiny_encoder.set_trainable(True)
    assert all(parameter.requires_grad for parameter in tiny_encoder.parameters())
    assert tiny_encoder.training is original_training_mode


def test_invalid_variant_is_rejected_before_backend_construction() -> None:
    with pytest.raises(ValueError, match="variant"):
        SwinEncoder("small", pretrained=False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("image", "error_type", "message"),
    [
        (torch.ones((3, 32, 32)), ValueError, "shape"),
        (torch.ones((1, 1, 32, 32)), ValueError, "three RGB"),
        (torch.ones((1, 3, 32, 32), dtype=torch.int64), TypeError, "floating-point"),
    ],
)
def test_invalid_inputs_are_rejected(
    tiny_encoder: SwinEncoder,
    image: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        tiny_encoder(image)


def test_non_boolean_pretrained_and_trainable_flags_are_rejected(
    tiny_encoder: SwinEncoder,
) -> None:
    with pytest.raises(TypeError, match="pretrained"):
        SwinEncoder("tiny", pretrained=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="trainable"):
        tiny_encoder.set_trainable(1)  # type: ignore[arg-type]
