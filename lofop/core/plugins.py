"""Plugin manager.

Plugins are how third parties extend LOFOP without forking it: a plugin is a
callable that receives a :class:`PluginContext` (registry hub + event bus) and
registers components, subscribes to events, or wires in new capabilities.

Two distribution channels are supported:

* **Entry points** (``lofop.plugins`` group): installed packages advertise a
  setup callable in their packaging metadata and are discovered automatically.
  This is the enterprise path -- plugins version, install, and uninstall like
  any dependency.
* **Programmatic registration**: applications and tests hand a setup callable
  directly to :meth:`PluginManager.add`, with no packaging required.

Plugins load lazily -- discovery only reads metadata; a plugin's module is
imported when it is activated. A failing plugin raises :class:`PluginError`
with the plugin name attached, and never leaves itself half-activated.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib import metadata
from typing import Callable

from lofop.core.events import EventBus
from lofop.core.exceptions import PluginError
from lofop.core.logging import get_logger
from lofop.core.registry import RegistryHub

ENTRY_POINT_GROUP = "lofop.plugins"

logger = get_logger(__name__)


@dataclass(frozen=True)
class PluginContext:
    """Framework surface handed to plugin setup callables.

    Attributes:
        hub: Registry hub to attach components to.
        events: Event bus for subscribing to framework lifecycle topics.
    """

    hub: RegistryHub
    events: EventBus


SetupFn = Callable[[PluginContext], None]


@dataclass
class _PluginRecord:
    name: str
    loader: Callable[[], SetupFn]
    source: str
    active: bool = field(default=False)


class PluginManager:
    """Discovers, activates, and tracks plugins for one framework instance.

    Args:
        hub: Registry hub passed to plugins on activation.
        events: Event bus passed to plugins on activation.
    """

    def __init__(self, hub: RegistryHub, events: EventBus) -> None:
        self._context = PluginContext(hub=hub, events=events)
        self._plugins: dict[str, _PluginRecord] = {}

    def discover(self, group: str = ENTRY_POINT_GROUP) -> list[str]:
        """Scan installed packages for plugin entry points.

        Discovery is metadata-only: no plugin code is imported until
        :meth:`activate`. Re-discovering an already-known name is a no-op so
        repeated calls are safe.

        Returns:
            Names of newly discovered plugins.
        """
        discovered: list[str] = []
        for entry_point in metadata.entry_points(group=group):
            if entry_point.name in self._plugins:
                continue
            self._plugins[entry_point.name] = _PluginRecord(
                name=entry_point.name,
                loader=entry_point.load,
                source=f"entry-point:{entry_point.value}",
            )
            discovered.append(entry_point.name)
        if discovered:
            logger.debug("Discovered plugins: %s", ", ".join(sorted(discovered)))
        return discovered

    def add(self, name: str, setup: SetupFn, *, override: bool = False) -> None:
        """Register a plugin programmatically.

        Args:
            name: Unique plugin name.
            setup: Callable invoked with a :class:`PluginContext` on
                activation.
            override: Replace an existing plugin of the same name (it is
                deactivated in bookkeeping terms, but any registrations it
                already made are not undone).
        """
        if name in self._plugins and not override:
            raise PluginError("Plugin name already registered", context={"plugin": name})
        if not callable(setup):
            raise PluginError("Plugin setup must be callable", context={"plugin": name})
        self._plugins[name] = _PluginRecord(name=name, loader=lambda: setup, source="programmatic")

    def activate(self, name: str) -> None:
        """Load and run a plugin's setup callable.

        Activating an already-active plugin is a no-op. Any exception from
        loading or setup is wrapped in :class:`PluginError`.
        """
        record = self._get(name)
        if record.active:
            return
        try:
            setup = record.loader()
        except Exception as exc:
            raise PluginError(
                f"Failed to load plugin: {exc}",
                context={"plugin": name, "source": record.source},
            ) from exc
        if not callable(setup):
            raise PluginError(
                "Plugin entry point did not resolve to a callable",
                context={"plugin": name, "source": record.source},
            )
        try:
            setup(self._context)
        except Exception as exc:
            raise PluginError(
                f"Plugin setup failed: {exc}",
                context={"plugin": name, "source": record.source},
            ) from exc
        record.active = True
        logger.info("Activated plugin %r (%s)", name, record.source)

    def activate_all(self) -> list[str]:
        """Activate every known inactive plugin; returns activated names."""
        activated = []
        for name in sorted(self._plugins):
            if not self._plugins[name].active:
                self.activate(name)
                activated.append(name)
        return activated

    def is_active(self, name: str) -> bool:
        """Whether the named plugin has been activated."""
        return self._get(name).active

    def _get(self, name: str) -> _PluginRecord:
        try:
            return self._plugins[name]
        except KeyError:
            raise PluginError(
                f"Unknown plugin {name!r}",
                context={"available": sorted(self._plugins)},
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __iter__(self) -> Iterator[str]:
        return iter(self._plugins)

    def __len__(self) -> int:
        return len(self._plugins)

    def __repr__(self) -> str:
        active = sorted(n for n, r in self._plugins.items() if r.active)
        return f"PluginManager(plugins={sorted(self._plugins)}, active={active})"
