"""Exception hierarchy for the LOFOP framework.

Every error raised by LOFOP derives from :class:`LofopError`, so callers can
catch framework failures with a single ``except LofopError`` clause while still
being able to discriminate by subsystem. Subsystem errors carry an optional
``context`` mapping with structured diagnostic data (component name, config
path, plugin name, ...) that is appended to the rendered message.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class LofopError(Exception):
    """Base class for all LOFOP framework errors.

    Args:
        message: Human-readable description of the failure.
        context: Optional structured data describing where the failure
            happened. Rendered as ``key=value`` pairs after the message.
    """

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        self.message = message
        self.context: dict[str, Any] = dict(context) if context else {}
        super().__init__(self._render())

    def _render(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({details})"


class ConfigError(LofopError):
    """Raised for invalid, unreadable, or inconsistent configuration."""


class RegistryError(LofopError):
    """Raised for registry misuse: duplicate names, unknown components, bad specs."""


class BuildError(LofopError):
    """Raised when instantiating a component from a config spec fails."""


class PluginError(LofopError):
    """Raised when discovering, loading, or activating a plugin fails."""


class EventError(LofopError):
    """Raised when one or more event handlers fail and errors are not suppressed."""

    def __init__(
        self,
        message: str,
        *,
        failures: list[tuple[str, Exception]] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.failures: list[tuple[str, Exception]] = list(failures or [])
        super().__init__(message, context=context)
