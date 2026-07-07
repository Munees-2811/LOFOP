"""LOFOP's default framework instance and standard component groups.

The core engine (:mod:`lofop.core`) is deliberately generic -- it knows
nothing about vision. This module instantiates the default :class:`RegistryHub`
and declares the standard component groups the framework builds on, plus the
default event bus and plugin manager wired to them.

Applications that need full isolation (e.g. hosting two LOFOP configurations
in one process) can build their own hub/bus/manager from :mod:`lofop.core`
instead of using these globals.
"""

from __future__ import annotations

from lofop.core.events import EventBus
from lofop.core.plugins import PluginManager
from lofop.core.registry import Registry, RegistryHub

HUB = RegistryHub()
EVENTS = EventBus()
PLUGINS = PluginManager(HUB, EVENTS)

# Standard component groups. Qualified config type names use these group
# names, e.g. ``type: loss/FocalCost`` or ``type: backbone/RidgeNet``.
MODELS: Registry = HUB.new("model")
BACKBONES: Registry = HUB.new("backbone")
NECKS: Registry = HUB.new("neck")
HEADS: Registry = HUB.new("head")
LOSSES: Registry = HUB.new("loss")
METRICS: Registry = HUB.new("metric")
OPTIMIZERS: Registry = HUB.new("optimizer")
SCHEDULERS: Registry = HUB.new("scheduler")
DATASETS: Registry = HUB.new("dataset")
TRANSFORMS: Registry = HUB.new("transform")
HOOKS: Registry = HUB.new("hook")

__all__ = [
    "HUB",
    "EVENTS",
    "PLUGINS",
    "MODELS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "METRICS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "DATASETS",
    "TRANSFORMS",
    "HOOKS",
]
