"""Tests for lofop.core.events."""

import pytest

from lofop.core.events import EventBus
from lofop.core.exceptions import EventError


class TestSubscribeEmit:
    def test_basic_delivery_with_payload(self):
        bus = EventBus()
        seen = []
        bus.subscribe("train.epoch_end", lambda e: seen.append((e.topic, e["epoch"])))
        failures = bus.emit("train.epoch_end", epoch=3)
        assert failures == []
        assert seen == [("train.epoch_end", 3)]

    def test_no_subscribers_is_fine(self):
        assert EventBus().emit("nobody.listens") == []

    def test_topic_isolation(self):
        bus = EventBus()
        seen = []
        bus.subscribe("a", lambda e: seen.append("a"))
        bus.emit("b")
        assert seen == []

    def test_priority_order_then_fifo(self):
        bus = EventBus()
        order = []
        bus.subscribe("t", lambda e: order.append("low"), priority=-1)
        bus.subscribe("t", lambda e: order.append("first"), priority=10)
        bus.subscribe("t", lambda e: order.append("mid1"))
        bus.subscribe("t", lambda e: order.append("mid2"))
        bus.emit("t")
        assert order == ["first", "mid1", "mid2", "low"]

    def test_once_unsubscribes_after_delivery(self):
        bus = EventBus()
        seen = []
        bus.subscribe("t", lambda e: seen.append(1), once=True)
        bus.emit("t")
        bus.emit("t")
        assert seen == [1]
        assert bus.listeners("t") == []

    def test_on_decorator(self):
        bus = EventBus()
        seen = []

        @bus.on("t", priority=5)
        def handler(event):
            seen.append(event.get("x", "default"))

        bus.emit("t")
        assert seen == ["default"]

    def test_unsubscribe(self):
        bus = EventBus()
        seen = []
        sub = bus.subscribe("t", lambda e: seen.append(1))
        assert bus.unsubscribe(sub) is True
        assert bus.unsubscribe(sub) is False
        bus.emit("t")
        assert seen == []

    def test_clear(self):
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        bus.clear("a")
        assert bus.listeners("a") == [] and len(bus.listeners("b")) == 1
        bus.clear()
        assert bus.listeners("b") == []


class TestErrorHandling:
    def test_failures_isolated_and_reported(self):
        bus = EventBus()
        seen = []

        def bad(event):
            raise RuntimeError("boom")

        bus.subscribe("t", bad, priority=1)
        bus.subscribe("t", lambda e: seen.append(1))
        failures = bus.emit("t")
        assert seen == [1]
        assert len(failures) == 1
        assert isinstance(failures[0][1], RuntimeError)

    def test_raise_errors_collects_all_failures(self):
        bus = EventBus()

        def bad(event):
            raise RuntimeError("boom")

        bus.subscribe("t", bad)
        bus.subscribe("t", bad)
        with pytest.raises(EventError) as excinfo:
            bus.emit("t", raise_errors=True)
        assert len(excinfo.value.failures) == 2

    def test_invalid_subscriptions_rejected(self):
        bus = EventBus()
        with pytest.raises(EventError):
            bus.subscribe("", lambda e: None)
        with pytest.raises(EventError):
            bus.subscribe("t", 42)
