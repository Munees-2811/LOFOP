"""Logging utilities for LOFOP.

LOFOP uses the standard :mod:`logging` machinery under a single ``"lofop"``
root logger so that host applications keep full control: they can attach their
own handlers, silence the framework, or route records into structured logging
pipelines. :func:`configure_logging` installs an opinionated default setup
(console + optional file, Rich formatting when available) for users who want
logs to "just work"; libraries embedding LOFOP can skip it entirely.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROOT_LOGGER_NAME = "lofop"
_ENV_LEVEL_VAR = "LOFOP_LOG_LEVEL"
_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under the LOFOP root logger.

    Args:
        name: Dotted suffix, typically ``__name__`` of the calling module.
            ``None`` returns the framework root logger. Names already prefixed
            with ``"lofop"`` are used as-is so ``get_logger(__name__)`` works
            from inside the package.
    """
    if not name or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str | None = None,
    *,
    log_file: str | Path | None = None,
    use_rich: bool | None = None,
    force: bool = False,
) -> logging.Logger:
    """Install default handlers on the LOFOP root logger.

    Safe to call multiple times: after the first call it is a no-op unless
    ``force=True``, so applications and plugins cannot stack duplicate
    handlers by accident.

    Args:
        level: Logging level (name or numeric). Defaults to the
            ``LOFOP_LOG_LEVEL`` environment variable, falling back to INFO.
        log_file: Optional path that additionally receives all records at
            DEBUG level. Parent directories are created if needed.
        use_rich: Force Rich console formatting on/off. By default Rich is
            used when importable and stderr is a TTY.
        force: Re-apply configuration, replacing previously installed
            handlers.

    Returns:
        The configured ``"lofop"`` root logger.
    """
    global _configured
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if _configured and not force:
        return logger

    resolved_level = _resolve_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_handler = _make_console_handler(use_rich)
    console_handler.setLevel(resolved_level)
    logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)
        logger.setLevel(min(resolved_level, logging.DEBUG))

    _configured = True
    return logger


def _resolve_level(level: int | str | None) -> int:
    if level is None:
        level = os.environ.get(_ENV_LEVEL_VAR, "INFO")
    if isinstance(level, str):
        numeric = logging.getLevelName(level.upper())
        if not isinstance(numeric, int):
            raise ValueError(f"Unknown log level: {level!r}")
        return numeric
    return int(level)


def _make_console_handler(use_rich: bool | None) -> logging.Handler:
    if use_rich is not False:
        try:
            from rich.logging import RichHandler

            if use_rich or sys.stderr.isatty():
                return RichHandler(rich_tracebacks=True, show_path=False)
        except ImportError:
            if use_rich:
                raise
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    return handler
