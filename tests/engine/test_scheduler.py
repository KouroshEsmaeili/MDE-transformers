from __future__ import annotations

import math

import pytest
import torch

from mde_transformers.engine import WarmupCosineScheduler


def _scheduler() -> tuple[torch.optim.SGD, WarmupCosineScheduler]:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=6,
        warmup_steps=2,
        warmup_start_factor=0.1,
    )
    return optimizer, scheduler


def test_warmup_cosine_learning_rate_used_by_each_optimizer_step() -> None:
    optimizer, scheduler = _scheduler()
    used_learning_rates: list[float] = []

    for _ in range(6):
        used_learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()

    expected = [
        0.1,
        0.55,
        1.0,
        0.5 * (1 + math.cos(math.pi / 4)),
        0.5,
        0.5 * (1 + math.cos(3 * math.pi / 4)),
    ]
    assert used_learning_rates == pytest.approx(expected)
    assert scheduler.step_count == 6
    assert scheduler.get_last_lr() == pytest.approx((0.0,))


def test_scheduler_state_resume_matches_uninterrupted_sequence() -> None:
    uninterrupted_optimizer, uninterrupted = _scheduler()
    for _ in range(3):
        uninterrupted_optimizer.step()
        uninterrupted.step()
    saved_state = uninterrupted.state_dict()

    resumed_optimizer, resumed = _scheduler()
    resumed.load_state_dict(saved_state)
    assert resumed.step_count == 3
    assert resumed.get_last_lr() == pytest.approx(uninterrupted.get_last_lr())

    uninterrupted_lrs: list[float] = []
    resumed_lrs: list[float] = []
    for _ in range(3):
        uninterrupted_lrs.append(uninterrupted.get_last_lr()[0])
        resumed_lrs.append(resumed.get_last_lr()[0])
        uninterrupted_optimizer.step()
        resumed_optimizer.step()
        uninterrupted.step()
        resumed.step()

    assert resumed_lrs == pytest.approx(uninterrupted_lrs)
    assert resumed.state_dict() == uninterrupted.state_dict()


@pytest.mark.parametrize(
    ("total_steps", "warmup_steps"),
    [(0, 0), (2, -1), (2, 2), (2, 3)],
)
def test_scheduler_rejects_invalid_step_configuration(
    total_steps: int,
    warmup_steps: int,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    with pytest.raises((TypeError, ValueError)):
        WarmupCosineScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
