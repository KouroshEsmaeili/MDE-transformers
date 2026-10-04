"""Explicit step-based linear-warmup and cosine-decay scheduling."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer


class WarmupCosineScheduler:
    """Linearly warm an optimizer, then cosine-decay its learning rate to zero.

    Before optimization step 0, the learning rate is ``base_lr * warmup_start_factor`` when
    warmup is enabled, so AdamW cannot accidentally perform its first update at base LR. After
    ``warmup_steps`` completed updates it reaches ``base_lr`` for the next optimization step.
    The remaining schedule follows ``0.5 * base_lr * (1 + cos(pi * progress))``. The last
    optimization step uses the final positive cosine value; calling :meth:`step` after it sets
    the now-exhausted schedule to zero. Call :meth:`step` exactly once after every optimizer step.

    For ``total_steps=6``, ``warmup_steps=2``, base LR 1, and start factor 0.1, optimization
    steps 0--5 use ``[0.1, 0.55, 1.0, 0.853553..., 0.5, 0.146446...]``; state after step 5 is
    zero. Thus the warmup endpoint, first decayed cosine value, and final-update LR are explicit.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        warmup_start_factor: float = 0.1,
    ) -> None:
        _validate_step_count(total_steps, "total_steps", positive=True)
        _validate_step_count(warmup_steps, "warmup_steps", positive=False)
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        if (
            isinstance(warmup_start_factor, bool)
            or not isinstance(warmup_start_factor, int | float)
            or not math.isfinite(float(warmup_start_factor))
            or not 0 < warmup_start_factor < 1
        ):
            raise ValueError("warmup_start_factor must be finite and lie strictly between 0 and 1")
        if not optimizer.param_groups:
            raise ValueError("optimizer must contain at least one parameter group")

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = float(warmup_start_factor)
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if any(not math.isfinite(lr) or lr <= 0 for lr in self.base_lrs):
            raise ValueError("optimizer learning rates must be finite and positive")
        self.step_count = 0
        self._set_learning_rates(self._factor(0))

    def step(self) -> None:
        """Advance after one optimizer update and set the LR for the next update."""
        if self.step_count >= self.total_steps:
            raise RuntimeError("scheduler has already reached total_steps")
        self.step_count += 1
        self._set_learning_rates(self._factor(self.step_count))

    def get_last_lr(self) -> tuple[float, ...]:
        """Return current optimizer learning rates."""
        return tuple(float(group["lr"]) for group in self.optimizer.param_groups)

    def state_dict(self) -> dict[str, Any]:
        """Return complete scheduling state for exact continuation."""
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_start_factor": self.warmup_start_factor,
            "base_lrs": self.base_lrs,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, rejecting a schedule constructed with incompatible settings."""
        required = {
            "total_steps",
            "warmup_steps",
            "warmup_start_factor",
            "base_lrs",
            "step_count",
        }
        if set(state_dict) != required:
            raise ValueError("scheduler state has unexpected or missing fields")
        if int(state_dict["total_steps"]) != self.total_steps:
            raise ValueError("scheduler total_steps does not match checkpoint")
        if int(state_dict["warmup_steps"]) != self.warmup_steps:
            raise ValueError("scheduler warmup_steps does not match checkpoint")
        if float(state_dict["warmup_start_factor"]) != self.warmup_start_factor:
            raise ValueError("scheduler warmup_start_factor does not match checkpoint")
        restored_base_lrs = tuple(float(value) for value in state_dict["base_lrs"])
        if restored_base_lrs != self.base_lrs:
            raise ValueError("scheduler base learning rates do not match checkpoint")
        restored_step = int(state_dict["step_count"])
        if not 0 <= restored_step <= self.total_steps:
            raise ValueError("scheduler step_count is outside the configured schedule")
        self.step_count = restored_step
        self._set_learning_rates(self._factor(self.step_count))

    def _factor(self, completed_steps: int) -> float:
        if self.warmup_steps > 0 and completed_steps < self.warmup_steps:
            progress = completed_steps / self.warmup_steps
            return self.warmup_start_factor + (1 - self.warmup_start_factor) * progress
        cosine_steps = self.total_steps - self.warmup_steps
        cosine_progress = (completed_steps - self.warmup_steps) / cosine_steps
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * cosine_progress))

    def _set_learning_rates(self, factor: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor


def _validate_step_count(value: int, name: str, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
