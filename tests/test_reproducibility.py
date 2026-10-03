from __future__ import annotations

import random
from unittest.mock import Mock

import numpy as np
import torch

from mde_transformers.utils import ReproducibilityConfig, seed_everything


def _draw_random_values() -> tuple[float, np.ndarray, torch.Tensor]:
    return random.random(), np.random.random(4), torch.rand(4)


def test_seed_repeats_python_numpy_and_torch_sequences() -> None:
    seed_everything(2025)
    first_python, first_numpy, first_torch = _draw_random_values()

    seed_everything(2025)
    second_python, second_numpy, second_torch = _draw_random_values()

    assert first_python == second_python
    np.testing.assert_array_equal(first_numpy, second_numpy)
    torch.testing.assert_close(first_torch, second_torch, rtol=0, atol=0)


def test_seed_does_not_call_cuda_seeding_when_cuda_is_unavailable(monkeypatch) -> None:
    cpu_seed = Mock()
    cuda_seed = Mock()
    monkeypatch.setattr(torch, "manual_seed", cpu_seed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seed)

    seed_everything(17)

    cpu_seed.assert_called_once_with(17)
    cuda_seed.assert_not_called()


def test_reproducibility_config_enables_deterministic_algorithms() -> None:
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    try:
        ReproducibilityConfig(
            seed=31,
            deterministic_algorithms=True,
            warn_only=True,
        ).apply()
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
