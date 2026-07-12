"""The packaged version must match the runtime version, and both the module."""

import importlib.util
from pathlib import Path

from lofop.version import __version__

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_version_sync.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_version_sync", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_versions_agree():
    module = _load_script()
    assert module._version_py() == module._pyproject_version() == __version__


def test_check_passes():
    module = _load_script()
    assert module.main() == 0
