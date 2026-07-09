"""LOFOP: a modular, enterprise-grade computer vision framework.

Public entry point. Import surface is intentionally small; subsystems live in
their own packages (``lofop.core``, ``lofop.registries``, and -- as the
framework grows -- ``lofop.data``, ``lofop.models``, ``lofop.training``, ...).
"""

from lofop.core import Config, LofopError, configure_logging, get_logger
from lofop.registries import EVENTS, HUB, PLUGINS
from lofop.version import __version__

__all__ = [
    "__version__",
    "Config",
    "LofopError",
    "configure_logging",
    "get_logger",
    "HUB",
    "EVENTS",
    "PLUGINS",
    "Detector",
]


def __getattr__(name: str):
    # Lazy so `import lofop` stays torch-free; the SDK needs lofop[models].
    if name == "Detector":
        from lofop.sdk import Detector

        return Detector
    raise AttributeError(f"module 'lofop' has no attribute {name!r}")
