"""Synchronous event bus.

The event bus decouples framework stages from cross-cutting concerns:
trainers emit lifecycle events (``train.epoch_end``, ``checkpoint.saved``,
...) and subscribers (experiment trackers, checkpoint managers, plugins)
react without the emitter knowing they exist.

Design decisions:

* **Synchronous, in-process delivery.** Training loops need deterministic
  ordering (e.g. "save checkpoint before uploading it"), which async
  delivery would forfeit. Handlers run in priority order on the emitting
  thread.
* **Error isolation by default.** One misbehaving subscriber must not kill a
  multi-hour training run, so handler exceptions are logged and collected.
  Emitters that need strictness pass ``raise_errors=True`` to get an
  :class:`~lofop.core.exceptions.EventError` carrying every failure.
* **Plain string topics.** Dotted names (``"train.epoch_end"``) keep the
  system open: plugins can define their own topics without central
  coordination. Constants for built-in topics will live with the modules
  that emit them.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from lofop.core.exceptions import EventError
from lofop.core.logging import get_logger

logger = get_logger(__name__)

Handler = Callable[["Event"], None]
HandlerDecorator = Callable[[Handler], Handler]


@dataclass(frozen=True)
class Event:
    """An immutable event delivered to subscribers.

    Attributes:
        topic: Dotted topic name, e.g. ``"train.epoch_end"``.
        payload: Arbitrary keyword data supplied by the emitter.
        timestamp: Unix time at emission.
    """

    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``payload[key]`` or ``default`` when absent."""
        return self.payload.get(key, default)


@dataclass(frozen=True)
class Subscription:
    """Handle returned by :meth:`EventBus.subscribe`; pass to ``unsubscribe``."""

    topic: str
    handler: Handler
    priority: int
    once: bool
    _order: int


class EventBus:
    """Priority-ordered synchronous publish/subscribe hub."""

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[Subscription]] = {}
        self._counter = itertools.count()

    def subscribe(
        self,
        topic: str,
        handler: Handler,
        *,
        priority: int = 0,
        once: bool = False,
    ) -> Subscription:
        """Register ``handler`` for ``topic``.

        Args:
            topic: Topic to listen on (exact match).
            handler: Callable invoked with the :class:`Event`.
            priority: Higher priorities run first; ties run in subscription
                order.
            once: Automatically unsubscribe after the first delivery.

        Returns:
            A :class:`Subscription` handle for later removal.
        """
        if not topic:
            raise EventError("Event topic must be a non-empty string")
        if not callable(handler):
            raise EventError("Event handler must be callable", context={"topic": topic})
        subscription = Subscription(topic, handler, priority, once, next(self._counter))
        listeners = self._subscriptions.setdefault(topic, [])
        listeners.append(subscription)
        listeners.sort(key=lambda s: (-s.priority, s._order))
        return subscription

    def on(self, topic: str, *, priority: int = 0, once: bool = False) -> HandlerDecorator:
        """Decorator form of :meth:`subscribe`::

            @bus.on("train.epoch_end")
            def log_metrics(event: Event) -> None: ...
        """
        def decorator(handler: Handler) -> Handler:
            self.subscribe(topic, handler, priority=priority, once=once)
            return handler

        return decorator

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Remove a subscription; returns ``False`` if it was already gone."""
        listeners = self._subscriptions.get(subscription.topic, [])
        try:
            listeners.remove(subscription)
            return True
        except ValueError:
            return False

    def emit(
        self,
        topic: str,
        *,
        raise_errors: bool = False,
        **payload: Any,
    ) -> list[tuple[str, Exception]]:
        """Deliver an event to all subscribers of ``topic``.

        Args:
            topic: Topic to publish on.
            raise_errors: Raise :class:`EventError` if any handler fails;
                otherwise failures are logged and returned.
            **payload: Event payload fields.

        Returns:
            A list of ``(handler_name, exception)`` pairs for handlers that
            raised (empty on full success).
        """
        event = Event(topic=topic, payload=payload)
        failures: list[tuple[str, Exception]] = []
        for subscription in list(self._subscriptions.get(topic, [])):
            if subscription.once:
                self.unsubscribe(subscription)
            try:
                subscription.handler(event)
            except Exception as exc:
                name = getattr(subscription.handler, "__qualname__", repr(subscription.handler))
                failures.append((name, exc))
                logger.exception("Event handler %r failed for topic %r", name, topic)
        if failures and raise_errors:
            raise EventError(
                f"{len(failures)} handler(s) failed for topic {topic!r}",
                failures=failures,
                context={"topic": topic},
            )
        return failures

    def listeners(self, topic: str) -> list[Subscription]:
        """Return current subscriptions for ``topic`` in delivery order."""
        return list(self._subscriptions.get(topic, []))

    def clear(self, topic: str | None = None) -> None:
        """Drop all subscriptions, or only those for ``topic``."""
        if topic is None:
            self._subscriptions.clear()
        else:
            self._subscriptions.pop(topic, None)
