"""Config-selectable learning-rate schedules.

Each factory takes the optimizer plus the run geometry -- ``warmup_steps`` and
``total_steps``, both counted in optimizer steps (batches) -- and returns a
torch ``LRScheduler`` that the :class:`~lofop.training.Trainer` steps once per
batch. Registering them in the ``scheduler`` group makes the schedule
selectable from a training config by name (``scheduler: warmup_cosine``).

The default ``warmup_cosine`` reproduces the trainer's historical schedule
(linear warmup, cosine decay to 5% of peak) exactly, so existing runs are
unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lofop.registries import SCHEDULERS


def _lambda(optimizer: Optimizer, factor: Callable[[int], float]) -> LRScheduler:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _warmup_factor(step: int, warmup_steps: int) -> float | None:
    """Linear warmup multiplier, or ``None`` once warmup is over."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    return None


@SCHEDULERS.register(name="warmup_cosine", aliases=("cosine",))
def warmup_cosine(
    optimizer: Optimizer, *, warmup_steps: int, total_steps: int, min_factor: float = 0.05,
) -> LRScheduler:
    """Linear warmup, then cosine decay from the peak to ``min_factor`` of it."""
    def factor(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        if warm is not None:
            return warm
        span = max(total_steps - warmup_steps, 1)
        progress = min((step - warmup_steps) / span, 1.0)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lambda(optimizer, factor)


@SCHEDULERS.register(name="warmup_linear", aliases=("linear",))
def warmup_linear(
    optimizer: Optimizer, *, warmup_steps: int, total_steps: int, min_factor: float = 0.01,
) -> LRScheduler:
    """Linear warmup, then linear decay from the peak to ``min_factor`` of it."""
    def factor(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        if warm is not None:
            return warm
        span = max(total_steps - warmup_steps, 1)
        progress = min((step - warmup_steps) / span, 1.0)
        return min_factor + (1.0 - min_factor) * (1.0 - progress)

    return _lambda(optimizer, factor)


@SCHEDULERS.register(name="constant")
def constant(optimizer: Optimizer, *, warmup_steps: int, total_steps: int) -> LRScheduler:
    """Linear warmup, then hold the peak learning rate."""
    def factor(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        return warm if warm is not None else 1.0

    return _lambda(optimizer, factor)


@SCHEDULERS.register(name="step")
def step_decay(
    optimizer: Optimizer, *, warmup_steps: int, total_steps: int,
    gamma: float = 0.1, milestones: Sequence[float] = (0.7, 0.9),
) -> LRScheduler:
    """Linear warmup, then multiply the LR by ``gamma`` at each milestone.

    Milestones are fractions of ``total_steps`` so the same config works at any
    epoch count or dataset size.
    """
    cutoffs = [int(m * total_steps) for m in milestones]

    def factor(step: int) -> float:
        warm = _warmup_factor(step, warmup_steps)
        if warm is not None:
            return warm
        return gamma ** sum(1 for cutoff in cutoffs if step >= cutoff)

    return _lambda(optimizer, factor)


def build_scheduler(
    name: str, optimizer: Optimizer, *, warmup_steps: int, total_steps: int, **kwargs,
) -> LRScheduler:
    """Instantiate a registered scheduler by name with the run geometry."""
    return SCHEDULERS.get(name)(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps, **kwargs
    )
