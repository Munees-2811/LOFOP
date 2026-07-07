"""LOFOP core engine: task-agnostic infrastructure the framework builds on.

Nothing in this package imports vision, data, or training code -- it provides
registries, configuration, events, plugins, logging, and the exception
hierarchy, and must stay importable in the leanest deployment environments.
"""

from lofop.core.config import Config
from lofop.core.events import Event, EventBus, Subscription
from lofop.core.exceptions import (
    BuildError,
    ConfigError,
    EventError,
    LofopError,
    PluginError,
    RegistryError,
)
from lofop.core.logging import configure_logging, get_logger
from lofop.core.plugins import PluginContext, PluginManager
from lofop.core.registry import Registry, RegistryHub

__all__ = [
    "Config",
    "Event",
    "EventBus",
    "Subscription",
    "LofopError",
    "ConfigError",
    "RegistryError",
    "BuildError",
    "PluginError",
    "EventError",
    "configure_logging",
    "get_logger",
    "PluginContext",
    "PluginManager",
    "Registry",
    "RegistryHub",
]
