#!/usr/bin/env python3
"""Fail if the package version is not consistent across its sources.

The version lives in two places that must agree: ``lofop/version.py`` (the
runtime source of truth, exposed as ``lofop.__version__``) and the
``[project].version`` field in ``pyproject.toml`` (what the built wheel
reports). This check keeps a release from shipping with a mismatch. Run it
locally or in CI; it exits non-zero and prints the mismatch on failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _version_py() -> str:
    text = (ROOT / "lofop" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise SystemExit("could not find __version__ in lofop/version.py")
    return match.group(1)


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        # Fallback for 3.9/3.10 without the stdlib TOML parser: read the
        # version line from the [project] table directly.
        match = re.search(r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']', text)
        if not match:
            raise SystemExit("could not find version in pyproject.toml") from None
        return match.group(1)
    return tomllib.loads(text)["project"]["version"]


def main() -> int:
    runtime = _version_py()
    packaged = _pyproject_version()
    if runtime != packaged:
        print(
            "version mismatch:\n"
            f"  lofop/version.py : {runtime}\n"
            f"  pyproject.toml   : {packaged}",
            file=sys.stderr,
        )
        return 1
    print(f"version OK: {runtime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
