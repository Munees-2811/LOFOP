"""Run tracking: a local, file-based experiment registry.

Every tracked training run gets a directory under a registry root::

    <root>/<run_id>/run.json      # settings, environment, status, final metrics
    <root>/<run_id>/history.json  # per-epoch loss and evaluation metrics

Tracking attaches to the framework event bus (``train.start``,
``train.epoch_end``, ``eval.end``, ``checkpoint.saved``, ``train.end``), so no
trainer or SDK code changes are needed and any training path -- SDK, CLI, or a
custom loop emitting the standard events -- is tracked identically. Storage is
plain JSON with no deep-learning dependency, so listing and comparing runs
works anywhere the LOFOP core does.

Usage::

    from lofop.mlops import track

    with track("runs/registry", name="person-finetune", tags=["coco"]):
        detector.train(...)

    lofop runs list --root runs/registry
    lofop runs show <run_id> --root runs/registry
    lofop runs compare <id1> <id2> --root runs/registry
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from lofop.core.events import Event, EventBus
from lofop.core.exceptions import LofopError
from lofop.registries import EVENTS
from lofop.version import __version__

_FINAL_METRIC_KEYS = ("map50", "map50_95", "precision", "recall", "f1")


def _environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "lofop": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:  # torch is optional; record it when the training stack is present
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return env


@dataclass
class RunRecord:
    """One tracked run, as stored in ``run.json``."""

    run_id: str
    name: str
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str | None = None
    status: str = "running"
    epochs_planned: int | None = None
    epochs_completed: int = 0
    duration_seconds: float | None = None
    device: str | None = None
    checkpoints: list[str] = field(default_factory=list)
    final_metrics: dict[str, float] = field(default_factory=dict)
    best_map50: float | None = None
    best_epoch: int | None = None


class RunTracker:
    """Subscribes to training events and persists one run's record.

    Use through :func:`track`; the context manager guarantees the
    subscriptions are removed and the record finalized even on failure.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        events: EventBus | None = None,
    ) -> None:
        self.events = events or EVENTS
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = Path(root)
        run_id = stamp
        suffix = 0
        while (self.root / run_id).exists():
            suffix += 1
            run_id = f"{stamp}_{suffix:02d}"
        self.run_dir = self.root / run_id
        self.record = RunRecord(
            run_id=run_id,
            name=name or run_id,
            tags=list(tags or []),
            meta=dict(meta or {}),
            environment=_environment(),
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.history: list[dict[str, Any]] = []
        self._start = time.perf_counter()
        self._subscriptions: list = []

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> RunTracker:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        pairs = [
            ("train.start", self._on_start),
            ("train.epoch_end", self._on_epoch_end),
            ("checkpoint.saved", self._on_checkpoint),
            ("train.end", self._on_end),
        ]
        self._subscriptions = [self.events.subscribe(t, h) for t, h in pairs]
        self._write()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for subscription in self._subscriptions:
            self.events.unsubscribe(subscription)
        if self.record.status == "running":
            self.record.status = "failed" if exc_type is not None else "completed"
        self.record.finished_at = datetime.now().isoformat(timespec="seconds")
        self.record.duration_seconds = round(time.perf_counter() - self._start, 2)
        self._write()

    # -- event handlers --------------------------------------------------------

    def _on_start(self, event: Event) -> None:
        self.record.epochs_planned = event.get("epochs")
        self.record.device = event.get("device")
        self._write()

    def _on_epoch_end(self, event: Event) -> None:
        epoch = int(event.get("epoch", 0)) + 1
        metrics = event.get("metrics") or {}
        entry: dict[str, Any] = {"epoch": epoch, "loss": event.get("loss")}
        for key in _FINAL_METRIC_KEYS:
            if metrics.get(key) is not None:
                entry[key] = metrics[key]
        self.history.append(entry)
        self.record.epochs_completed = epoch
        map50 = metrics.get("map50")
        if map50 is not None and (self.record.best_map50 is None or map50 > self.record.best_map50):
            self.record.best_map50 = float(map50)
            self.record.best_epoch = epoch
        self._write()

    def _on_checkpoint(self, event: Event) -> None:
        path = event.get("path")
        if path and path not in self.record.checkpoints:
            self.record.checkpoints.append(path)

    def _on_end(self, event: Event) -> None:
        metrics = event.get("metrics") or {}
        self.record.final_metrics = {
            key: float(metrics[key])
            for key in _FINAL_METRIC_KEYS
            if metrics.get(key) is not None
        }
        self.record.status = "completed"
        self._write()

    # -- persistence -------------------------------------------------------------

    def _write(self) -> None:
        (self.run_dir / "run.json").write_text(
            json.dumps(self.record.__dict__, indent=2) + "\n", encoding="utf-8"
        )
        (self.run_dir / "history.json").write_text(
            json.dumps(self.history, indent=2) + "\n", encoding="utf-8"
        )


def track(
    root: str | Path,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    events: EventBus | None = None,
) -> RunTracker:
    """Create a :class:`RunTracker` context manager for one training run."""
    return RunTracker(root, name=name, tags=tags, meta=meta, events=events)


# -- registry queries (torch-free) ---------------------------------------------


def list_runs(root: str | Path) -> list[dict[str, Any]]:
    """All run records under ``root``, newest first."""
    root = Path(root)
    records = []
    if root.is_dir():
        for run_json in sorted(root.glob("*/run.json"), reverse=True):
            try:
                records.append(json.loads(run_json.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return records


def load_run(root: str | Path, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(record, history)`` for one run.

    Raises:
        LofopError: If the run does not exist under ``root``.
    """
    run_dir = Path(root) / run_id
    record_file = run_dir / "run.json"
    if not record_file.is_file():
        known = [r["run_id"] for r in list_runs(root)]
        raise LofopError(f"Unknown run {run_id!r}", context={"root": str(root), "known": known})
    record = json.loads(record_file.read_text(encoding="utf-8"))
    history_file = run_dir / "history.json"
    history = (
        json.loads(history_file.read_text(encoding="utf-8")) if history_file.is_file() else []
    )
    return record, history


def compare_runs(root: str | Path, run_ids: list[str]) -> str:
    """Render a side-by-side markdown table of run settings and final metrics."""
    records = [load_run(root, run_id)[0] for run_id in run_ids]
    rows = [
        ("name", lambda r: r.get("name", "-")),
        ("status", lambda r: r.get("status", "-")),
        ("epochs", lambda r: f"{r.get('epochs_completed', 0)}/{r.get('epochs_planned') or '?'}"),
        ("device", lambda r: r.get("device") or "-"),
        ("duration (s)", lambda r: r.get("duration_seconds") or "-"),
        ("best mAP@50", lambda r: _fmt(r.get("best_map50"))),
        ("best epoch", lambda r: r.get("best_epoch") or "-"),
    ]
    rows += [
        (f"final {key}", lambda r, k=key: _fmt(r.get("final_metrics", {}).get(k)))
        for key in _FINAL_METRIC_KEYS
    ]
    header = ["Run"] + [r["run_id"] for r in records]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, extract in rows:
        lines.append("| " + " | ".join([label] + [str(extract(r)) for r in records]) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else "-"
