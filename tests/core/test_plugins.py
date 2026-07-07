"""Tests for lofop.core.plugins."""

import pytest

from lofop.core.events import EventBus
from lofop.core.exceptions import PluginError
from lofop.core.plugins import PluginContext, PluginManager
from lofop.core.registry import RegistryHub


@pytest.fixture
def manager():
    return PluginManager(RegistryHub(), EventBus())


class TestProgrammaticPlugins:
    def test_add_and_activate(self, manager):
        calls = []

        def setup(context):
            assert isinstance(context, PluginContext)
            context.hub.get_or_new("backbone").register(lambda: "net", name="tiny")
            calls.append(1)

        manager.add("tiny-nets", setup)
        assert "tiny-nets" in manager
        assert not manager.is_active("tiny-nets")

        manager.activate("tiny-nets")
        assert manager.is_active("tiny-nets")
        assert "tiny" in manager._context.hub["backbone"]

        manager.activate("tiny-nets")  # idempotent
        assert calls == [1]

    def test_duplicate_name_rejected_unless_override(self, manager):
        manager.add("p", lambda ctx: None)
        with pytest.raises(PluginError):
            manager.add("p", lambda ctx: None)
        manager.add("p", lambda ctx: None, override=True)

    def test_non_callable_setup_rejected(self, manager):
        with pytest.raises(PluginError):
            manager.add("p", 42)

    def test_unknown_plugin(self, manager):
        with pytest.raises(PluginError, match="Unknown plugin"):
            manager.activate("ghost")

    def test_setup_failure_wrapped(self, manager):
        def setup(context):
            raise RuntimeError("bad wiring")

        manager.add("broken", setup)
        with pytest.raises(PluginError, match="bad wiring"):
            manager.activate("broken")
        assert not manager.is_active("broken")

    def test_activate_all(self, manager):
        order = []
        manager.add("b", lambda ctx: order.append("b"))
        manager.add("a", lambda ctx: order.append("a"))
        assert manager.activate_all() == ["a", "b"]
        assert order == ["a", "b"]
        assert manager.activate_all() == []


class TestEntryPointDiscovery:
    def test_discover_reads_entry_points(self, manager, monkeypatch):
        class FakeEntryPoint:
            name = "ext"
            value = "ext_pkg:setup"

            @staticmethod
            def load():
                def setup(context):
                    context.events.subscribe("x", lambda e: None)

                return setup

        monkeypatch.setattr(
            "lofop.core.plugins.metadata.entry_points",
            lambda group: [FakeEntryPoint()] if group == "lofop.plugins" else [],
        )
        assert manager.discover() == ["ext"]
        assert manager.discover() == []  # idempotent
        manager.activate("ext")
        assert manager.is_active("ext")

    def test_entry_point_load_failure_wrapped(self, manager, monkeypatch):
        class BrokenEntryPoint:
            name = "broken"
            value = "broken_pkg:setup"

            @staticmethod
            def load():
                raise ImportError("no module")

        monkeypatch.setattr(
            "lofop.core.plugins.metadata.entry_points",
            lambda group: [BrokenEntryPoint()],
        )
        manager.discover()
        with pytest.raises(PluginError, match="Failed to load"):
            manager.activate("broken")

    def test_entry_point_resolving_to_non_callable(self, manager, monkeypatch):
        class WeirdEntryPoint:
            name = "weird"
            value = "weird_pkg:thing"

            @staticmethod
            def load():
                return object()

        monkeypatch.setattr(
            "lofop.core.plugins.metadata.entry_points",
            lambda group: [WeirdEntryPoint()],
        )
        manager.discover()
        with pytest.raises(PluginError, match="did not resolve to a callable"):
            manager.activate("weird")
