"""Checkpoint manager: atomic saves, resume, best-model tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from lofop.core.exceptions import LofopError
from lofop.core.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Writes and restores training state under one directory.

    Layout: ``last.pt`` (every save), ``best.pt`` (highest metric so far).
    Writes are atomic (temp file + rename) so a crash mid-save never
    corrupts a resumable checkpoint.

    Args:
        directory: Checkpoint directory; created if missing.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("-inf")

    def save(
        self,
        model: nn.Module,
        *,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        ema_state: dict | None = None,
        metric: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write ``last.pt``; also refresh ``best.pt`` when ``metric`` improves.

        Returns:
            Path of the written ``last.pt``.
        """
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model": model.state_dict(),
            "best_metric": self.best_metric,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if ema_state is not None:
            payload["ema"] = ema_state
        if metric is not None:
            payload["metric"] = metric
        if extra:
            payload["extra"] = extra

        last = self._atomic_write(payload, self.directory / "last.pt")
        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            payload["best_metric"] = metric
            self._atomic_write(payload, self.directory / "best.pt")
            logger.info("New best checkpoint at epoch %d (metric %.4f)", epoch, metric)
        return last

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Read a checkpoint payload (defaults to ``last.pt``).

        Raises:
            LofopError: If the checkpoint file does not exist.
        """
        target = Path(path) if path is not None else self.directory / "last.pt"
        if not target.is_file():
            raise LofopError("Checkpoint not found", context={"path": str(target)})
        payload = torch.load(target, map_location="cpu", weights_only=True)
        self.best_metric = float(payload.get("best_metric", float("-inf")))
        return payload

    @staticmethod
    def _atomic_write(payload: dict[str, Any], target: Path) -> Path:
        temp = target.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(target)
        return target
