"""Exponential moving average of model weights.

EMA weights consistently evaluate 0.5-1 mAP above the raw weights for
detectors, a well-established training-strategy improvement. The decay ramps
up over early updates so the average is not dominated by random
initialization.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


class ModelEMA:
    """Maintains an exponential moving average copy of a model.

    Args:
        model: Model to track (must be the unwrapped module, not DDP).
        decay: Asymptotic decay rate.
        warmup_updates: Time constant of the decay ramp: effective decay is
            ``decay * (1 - exp(-updates / warmup_updates))``.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995, warmup_updates: int = 500) -> None:
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.warmup_updates = warmup_updates
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold the model's current state into the average (call per step)."""
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / self.warmup_updates))
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_state[key].copy_(value)

    def state_dict(self) -> dict:
        """Serializable state for checkpointing."""
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        """Restore from :meth:`state_dict` output."""
        self.module.load_state_dict(state["module"])
        self.updates = int(state["updates"])
