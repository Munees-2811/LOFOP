"""Training engine for LOFOP detectors.

One `Trainer` owns the loop: AMP autocast, gradient clipping, warmup+cosine
learning-rate schedule, EMA weights, periodic evaluation with the EMA copy,
and atomic checkpointing with resume. Lifecycle events are emitted on the
LOFOP event bus (``train.start``, ``train.epoch_end``, ``eval.end``,
``checkpoint.saved``, ``train.end``) so trackers and plugins observe training
without the trainer knowing about them.

Multi-GPU: when ``torch.distributed`` is initialized (torchrun sets
WORLD_SIZE), the model is wrapped in DistributedDataParallel and the sampler
is sharded. This code path follows the standard recipe but is not exercised
by CI in this repository yet -- treat the first multi-GPU run as
verification.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, distributed

from lofop.core.events import EventBus
from lofop.core.logging import get_logger
from lofop.registries import EVENTS, OPTIMIZERS
from lofop.training.checkpoint import CheckpointManager
from lofop.training.ema import ModelEMA
from lofop.training.evaluator import DetectionMetrics, evaluate_detections
from lofop.training.torch_data import DetectionTorchDataset, detection_collate

logger = get_logger(__name__)

if "SGD" not in OPTIMIZERS:
    OPTIMIZERS.register(torch.optim.SGD, name="SGD", aliases=("sgd",))
    OPTIMIZERS.register(torch.optim.AdamW, name="AdamW", aliases=("adamw",))


def warmup_cosine_lr(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup then cosine decay to 5% of peak."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / span, 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """Trains a LOFOP detector on canonical datasets.

    Args:
        model: Detector exposing ``compute_losses(images, targets)`` and
            ``predict(images)`` (e.g. ``LofopDetect``).
        train_data: Torch-facing training dataset.
        val_data: Optional evaluation dataset (no augmentation).
        epochs: Total epochs.
        batch_size: Per-process batch size.
        lr: Peak learning rate.
        optimizer: Name in the optimizer registry (``"SGD"``, ``"AdamW"``).
        weight_decay: L2 regularization.
        warmup_epochs: Linear warmup duration.
        amp: Mixed precision (effective on CUDA; no-op on CPU).
        workers: DataLoader worker processes.
        checkpoint_dir: Where ``last.pt``/``best.pt`` are written.
        events: Event bus (defaults to the framework bus).
        device: Compute device; auto-selects CUDA when available.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: DetectionTorchDataset,
        val_data: DetectionTorchDataset | None = None,
        *,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.01,
        optimizer: str = "SGD",
        weight_decay: float = 5e-4,
        warmup_epochs: float = 1.0,
        amp: bool = True,
        workers: int = 2,
        checkpoint_dir: str | Path = "runs/train",
        events: EventBus | None = None,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.model = model.to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model)
        self.epochs = epochs
        self.events = events or EVENTS

        sampler = distributed.DistributedSampler(train_data) if self.distributed else None
        drop_last = len(train_data) > batch_size
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
            num_workers=workers, collate_fn=detection_collate, drop_last=drop_last,
        )
        self.val_data = val_data

        opt_kwargs = {"momentum": 0.9} if optimizer.lower() == "sgd" else {}
        self.optimizer = OPTIMIZERS.get(optimizer)(
            self._unwrapped.parameters(), lr=lr, weight_decay=weight_decay, **opt_kwargs
        )
        steps_per_epoch = max(len(self.train_loader), 1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: warmup_cosine_lr(
                step,
                warmup_steps=int(warmup_epochs * steps_per_epoch),
                total_steps=epochs * steps_per_epoch,
            ),
        )
        self.amp = amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.ema = ModelEMA(self._unwrapped)
        self.checkpoints = CheckpointManager(checkpoint_dir)
        self.start_epoch = 0

    @property
    def _unwrapped(self) -> nn.Module:
        return self.model.module if self.distributed else self.model

    def resume(self, path: str | Path | None = None) -> int:
        """Restore model/EMA/optimizer state; returns the epoch to resume at."""
        payload = self.checkpoints.load(path)
        self._unwrapped.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "ema" in payload:
            self.ema.load_state_dict(payload["ema"])
        self.start_epoch = int(payload["epoch"]) + 1
        logger.info("Resumed from epoch %d", self.start_epoch)
        return self.start_epoch

    def fit(self) -> DetectionMetrics | None:
        """Run the training loop; returns the final evaluation metrics."""
        self.events.emit("train.start", epochs=self.epochs, device=str(self.device))
        metrics: DetectionMetrics | None = None
        for epoch in range(self.start_epoch, self.epochs):
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            epoch_loss = self._train_epoch(epoch)
            metrics = self.evaluate() if self.val_data is not None else None
            metric_value = metrics.map50_95 if metrics else -epoch_loss
            if self._is_main_process:
                path = self.checkpoints.save(
                    self._unwrapped, epoch=epoch, optimizer=self.optimizer,
                    ema_state=self.ema.state_dict(), metric=metric_value,
                )
                self.events.emit("checkpoint.saved", epoch=epoch, path=str(path))
            self.events.emit(
                "train.epoch_end", epoch=epoch, loss=epoch_loss,
                metrics=metrics.__dict__ if metrics else None,
            )
        self.events.emit("train.end", metrics=metrics.__dict__ if metrics else None)
        return metrics

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, batches = 0.0, 0
        start = time.perf_counter()
        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = [
                {key: value.to(self.device) for key, value in target.items()}
                for target in targets
            ]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.amp):
                loss = self._unwrapped.compute_losses(images, targets)["total"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema.update(self._unwrapped)
            total += float(loss.detach())
            batches += 1
        mean_loss = total / max(batches, 1)
        logger.info(
            "epoch %d: loss %.4f (%d batches, %.1fs, lr %.5f)",
            epoch, mean_loss, batches, time.perf_counter() - start,
            self.optimizer.param_groups[0]["lr"],
        )
        return mean_loss

    @torch.no_grad()
    def evaluate(self, batch_size: int = 8) -> DetectionMetrics:
        """Evaluate the EMA weights on the validation dataset."""
        loader = DataLoader(
            self.val_data, batch_size=batch_size, collate_fn=detection_collate,
        )
        model = self.ema.module.to(self.device)
        predictions, targets = [], []
        for images, batch_targets in loader:
            for result in model.predict(images.to(self.device)):
                predictions.append({key: value.cpu() for key, value in result.items()})
            targets.extend(batch_targets)
        metrics = evaluate_detections(predictions, targets)
        logger.info(
            "eval: mAP50 %.4f, mAP50-95 %.4f, P %.4f, R %.4f",
            metrics.map50, metrics.map50_95, metrics.precision, metrics.recall,
        )
        self.events.emit("eval.end", metrics=metrics.__dict__)
        return metrics

    @property
    def _is_main_process(self) -> bool:
        return not self.distributed or int(os.environ.get("RANK", "0")) == 0
