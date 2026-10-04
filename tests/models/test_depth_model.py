"""End-to-end tests for the complete monocular depth model boundary."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError

import pytest
import torch

from mde_transformers.losses import multi_scale_depth_loss
from mde_transformers.models import DepthModelConfig, MonocularDepthModel


@pytest.fixture(scope="module")
def tiny_model() -> Iterator[MonocularDepthModel]:
    model = MonocularDepthModel(encoder_variant="tiny", encoder_pretrained=False)
    model.eval()
    yield model


def test_default_tiny_configuration_uses_encoder_as_channel_authority(
    tiny_model: MonocularDepthModel,
) -> None:
    assert tiny_model.config == DepthModelConfig()
    assert tiny_model.encoder_variant == "tiny"
    assert tiny_model.feature_channels == (96, 192, 384, 768)
    assert tiny_model.feature_strides == (4, 8, 16, 32)
    assert tiny_model.decoder.feature_channels == tiny_model.encoder.feature_channels
    assert tiny_model.decoder_channels == (128, 128, 128, 128)
    assert tiny_model.minimum_depth == pytest.approx(1e-6)
    assert not tiny_model.config.encoder_pretrained


def test_configuration_is_immutable_and_supports_explicit_construction() -> None:
    config = DepthModelConfig(decoder_channels=(8, 8, 8, 8), minimum_depth=1e-4)
    with pytest.raises(FrozenInstanceError):
        config.minimum_depth = 1.0  # type: ignore[misc]

    model = MonocularDepthModel.from_config(config)

    assert model.config == config
    assert model.decoder_channels == (8, 8, 8, 8)
    assert model.minimum_depth == pytest.approx(1e-4)


def test_rectangular_forward_has_exact_contract_and_no_hidden_preprocessing(
    tiny_model: MonocularDepthModel,
) -> None:
    image = torch.randn((1, 3, 128, 160))
    observed_inputs: list[torch.Tensor] = []

    def record_encoder_input(module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]) -> None:
        del module
        observed_inputs.append(arguments[0])

    hook = tiny_model.encoder.register_forward_pre_hook(record_encoder_input)
    try:
        with torch.no_grad():
            output = tiny_model(image)
    finally:
        hook.remove()

    assert len(observed_inputs) == 1
    assert observed_inputs[0] is image
    assert [tuple(depth.shape) for depth in output.stage_depths()] == [
        (1, 1, 4, 5),
        (1, 1, 8, 10),
        (1, 1, 16, 20),
        (1, 1, 32, 40),
    ]
    assert output.full_resolution.shape == (1, 1, 128, 160)
    assert len(output.stage_depths()) == 4
    for depth in (*output.stage_depths(), output.full_resolution):
        assert torch.isfinite(depth).all()
        assert (depth > 0).all()


def test_odd_rectangular_input_returns_exact_original_size(
    tiny_model: MonocularDepthModel,
) -> None:
    image = torch.randn((1, 3, 65, 97))

    with torch.no_grad():
        output = tiny_model(image)

    spatial_sizes = [depth.shape[-2:] for depth in output.stage_depths()]
    assert spatial_sizes == [(2, 3), (4, 6), (8, 12), (16, 24)]
    assert all(
        finer[0] >= coarser[0] and finer[1] >= coarser[1]
        for coarser, finer in zip(spatial_sizes, spatial_sizes[1:], strict=False)
    )
    assert output.full_resolution.shape[-2:] == image.shape[-2:]


def test_base_configuration_uses_same_complete_model_class() -> None:
    model = MonocularDepthModel(encoder_variant="base", encoder_pretrained=False)

    assert model.encoder_variant == "base"
    assert model.feature_channels == (128, 256, 512, 1024)
    assert model.feature_strides == (4, 8, 16, 32)
    assert model.decoder.feature_channels == model.encoder.feature_channels
    assert sum(parameter.numel() for parameter in model.encoder.parameters()) == 86_741_176
    assert sum(parameter.numel() for parameter in model.decoder.parameters()) == 844_706
    assert sum(parameter.numel() for parameter in model.parameters()) == 87_585_882


def test_encoder_freeze_and_restore_leave_decoder_and_mode_unchanged(
    tiny_model: MonocularDepthModel,
) -> None:
    original_encoder_mode = tiny_model.encoder.training
    original_decoder_mode = tiny_model.decoder.training

    tiny_model.set_encoder_trainable(False)
    assert not any(parameter.requires_grad for parameter in tiny_model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in tiny_model.decoder.parameters())
    assert tiny_model.encoder.training is original_encoder_mode
    assert tiny_model.decoder.training is original_decoder_mode

    tiny_model.set_encoder_trainable(True)
    assert all(parameter.requires_grad for parameter in tiny_model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in tiny_model.decoder.parameters())
    assert tiny_model.encoder.training is original_encoder_mode
    assert tiny_model.decoder.training is original_decoder_mode


def test_frozen_encoder_allows_finite_decoder_gradients(
    tiny_model: MonocularDepthModel,
) -> None:
    tiny_model.zero_grad(set_to_none=True)
    tiny_model.set_encoder_trainable(False)
    image = torch.randn((1, 3, 32, 64))

    output = tiny_model(image)
    loss = sum(depth.mean() for depth in output.stage_depths()) + output.full_resolution.mean()
    loss.backward()

    assert all(parameter.grad is None for parameter in tiny_model.encoder.parameters())
    decoder_gradients = [parameter.grad for parameter in tiny_model.decoder.parameters()]
    assert decoder_gradients
    assert all(gradient is not None for gradient in decoder_gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in decoder_gradients if gradient is not None
    )
    tiny_model.set_encoder_trainable(True)


def test_trainable_model_integrates_with_external_multiscale_loss(
    tiny_model: MonocularDepthModel,
) -> None:
    tiny_model.zero_grad(set_to_none=True)
    tiny_model.set_encoder_trainable(True)
    image = torch.randn((1, 3, 32, 64))
    target = torch.ones((1, 1, 32, 64))
    mask = torch.ones_like(target, dtype=torch.bool)

    output = tiny_model(image)
    depth_result = multi_scale_depth_loss(
        output.stage_depths(),
        target,
        mask,
        stage_weights=(0.1, 0.2, 0.3, 1.0),
    )
    loss = depth_result.total + output.full_resolution.mean()
    loss.backward()

    assert torch.isfinite(depth_result.total)
    assert len(depth_result.stage_losses) == 4
    for component in (tiny_model.encoder, tiny_model.decoder):
        gradients = [parameter.grad for parameter in component.parameters()]
        assert any(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_strict_state_dict_roundtrip_reproduces_outputs() -> None:
    config = DepthModelConfig(decoder_channels=(8, 8, 8, 8))
    model_a = MonocularDepthModel.from_config(config).eval()
    state = {name: value.detach().clone() for name, value in model_a.state_dict().items()}
    model_b = MonocularDepthModel.from_config(config).eval()

    incompatible = model_b.load_state_dict(state, strict=True)
    image = torch.randn((1, 3, 32, 64))
    with torch.no_grad():
        output_a = model_a(image)
        output_b = model_b(image)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for depth_a, depth_b in zip(
        (*output_a.stage_depths(), output_a.full_resolution),
        (*output_b.stage_depths(), output_b.full_resolution),
        strict=True,
    ):
        torch.testing.assert_close(depth_a, depth_b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder_variant": "small"},
        {"encoder_pretrained": 1},
        {"decoder_channels": (8, 8, 8)},
        {"decoder_channels": (8, 8, 8, 0)},
        {"minimum_depth": 0.0},
        {"minimum_depth": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        DepthModelConfig(**kwargs)  # type: ignore[arg-type]


def test_from_config_rejects_untyped_mapping() -> None:
    with pytest.raises(TypeError, match="DepthModelConfig"):
        MonocularDepthModel.from_config({})  # type: ignore[arg-type]
