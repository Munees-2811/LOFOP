"""Build and load the native C++ ops library.

The C++ core is optional by design: every op in :mod:`lofop.ops` has a pure
Python reference implementation, and the native library is a drop-in
accelerator loaded through ctypes. This keeps LOFOP importable on machines
with no compiler while letting deployments (and the Docker images) opt into
native speed with one call::

    python -c "from lofop.ops.native import build_native; build_native()"

The compiled library lands next to this module as ``_lofop_ops.so`` (or in
``~/.cache/lofop`` when the package directory is read-only) and is picked up
automatically on the next import.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sysconfig
from pathlib import Path

from lofop.core.exceptions import LofopError
from lofop.core.logging import get_logger

_LIB_STEM = "_lofop_ops"
_SOURCE = Path(__file__).resolve().parent.parent / "csrc" / "box_ops.cpp"

logger = get_logger(__name__)
_loaded: ctypes.CDLL | None = None
_load_attempted = False


def _lib_suffix() -> str:
    return sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"


def _candidate_paths() -> list[Path]:
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lofop"
    return [
        Path(__file__).resolve().parent / f"{_LIB_STEM}{_lib_suffix()}",
        cache_dir / f"{_LIB_STEM}{_lib_suffix()}",
    ]


def build_native(*, force: bool = False, compiler: str | None = None) -> Path:
    """Compile the C++ ops library and return its path.

    Args:
        force: Rebuild even if a library already exists.
        compiler: C++ compiler binary; defaults to ``$CXX`` or ``g++``.

    Raises:
        LofopError: If no compiler is available or compilation fails.
    """
    existing = find_library()
    if existing is not None and not force:
        return existing
    compiler = compiler or os.environ.get("CXX", "g++")
    last_error: Exception | None = None
    for target in _candidate_paths():
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
            str(_SOURCE), "-o", str(target),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise LofopError(
                "No C++ compiler found; install g++/clang++ or set CXX",
                context={"compiler": compiler},
            ) from exc
        except (subprocess.CalledProcessError, OSError) as exc:
            last_error = exc
            continue
        logger.info("Built native ops library at %s", target)
        _reset_cache()
        return target
    stderr = getattr(last_error, "stderr", "")
    raise LofopError(
        f"Failed to build native ops library: {last_error}",
        context={"source": str(_SOURCE), "stderr": stderr},
    )


def find_library() -> Path | None:
    """Return the path of a previously built library, if any."""
    for path in _candidate_paths():
        if path.is_file():
            return path
    return None


def load_native() -> ctypes.CDLL | None:
    """Load the native library, or return ``None`` when unavailable.

    The result is cached; call :func:`build_native` (which resets the cache)
    to pick up a fresh build in the same process.
    """
    global _loaded, _load_attempted
    if _load_attempted:
        return _loaded
    _load_attempted = True
    path = find_library()
    if path is None:
        logger.debug("Native ops library not built; using pure Python ops")
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        logger.warning("Failed to load native ops library %s: %s", path, exc)
        return None
    lib.lofop_iou_matrix.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.lofop_iou_matrix.restype = None
    lib.lofop_nms.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_float, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.lofop_nms.restype = ctypes.c_int32
    _loaded = lib
    logger.debug("Loaded native ops library from %s", path)
    return lib


def _reset_cache() -> None:
    global _loaded, _load_attempted
    _loaded = None
    _load_attempted = False
