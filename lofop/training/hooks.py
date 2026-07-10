"""Optional training hooks that subscribe to the event bus.

Hooks observe training through the event bus rather than being wired into the
trainer, so enabling one never changes the training code path. The TensorBoard
hook is the reference example: attach it and every subsequent run logs scalars,
detach it and the trainer is untouched.
"""

from __future__ import annotations

from pathlib import Path

from lofop.core.events import Event, EventBus
from lofop.core.logging import get_logger
from lofop.registries import EVENTS

logger = get_logger(__name__)

_SCALAR_KEYS = ("map50", "map50_95", "precision", "recall", "f1")


class TensorBoardHook:
    """Log training loss and evaluation metrics to TensorBoard.

    Subscribes to ``train.epoch_end`` (loss plus any evaluation metrics carried
    on the event) and writes them as scalars under ``log_dir``. Requires
    ``torch.utils.tensorboard`` (ships with torch); construction raises a clear
    error if TensorBoard's writer is unavailable.

    Args:
        log_dir: Directory for the TensorBoard event files.
        events: Event bus to subscribe to (defaults to the framework bus).
    """

    def __init__(self, log_dir: str | Path, *, events: EventBus | None = None) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(log_dir))
        self.events = events or EVENTS
        self._subscription = self.events.subscribe("train.epoch_end", self._on_epoch_end)
        logger.info("TensorBoard logging to %s", log_dir)

    def _on_epoch_end(self, event: Event) -> None:
        epoch = int(event.get("epoch", 0))
        loss = event.get("loss")
        if loss is not None:
            self.writer.add_scalar("train/loss", float(loss), epoch)
        metrics = event.get("metrics")
        if metrics:
            for key in _SCALAR_KEYS:
                if key in metrics and metrics[key] is not None:
                    self.writer.add_scalar(f"eval/{key}", float(metrics[key]), epoch)
        self.writer.flush()

    def close(self) -> None:
        """Unsubscribe and close the writer; safe to call more than once."""
        self.events.unsubscribe(self._subscription)
        self.writer.close()


def attach_tensorboard(log_dir: str | Path, *, events: EventBus | None = None) -> TensorBoardHook:
    """Create and attach a :class:`TensorBoardHook`; return it for later close."""
    return TensorBoardHook(log_dir, events=events)
