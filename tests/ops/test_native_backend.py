"""Tests for cross-platform native-ops building and backend reporting."""

import shutil

import pytest

from lofop.ops import backend, nms
from lofop.ops.native import _build_command, _compiler_candidates

HAVE_GNU = shutil.which("g++") or shutil.which("clang++")


class TestBackendReporting:
    def test_backend_is_one_of_two(self):
        assert backend() in ("native", "python")

    def test_python_path_always_available(self):
        # Regardless of backend, native=False must work everywhere.
        assert nms([[0, 0, 10, 10], [1, 1, 11, 11]], [0.9, 0.8],
                   iou_threshold=0.5, native=False) == [0]

    @pytest.mark.skipif(not HAVE_GNU, reason="no g++/clang++ available")
    def test_native_matches_python_when_built(self):
        from lofop.ops import build_native
        from lofop.ops.native import _reset_cache

        build_native()
        _reset_cache()
        assert backend() == "native"
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
        scores = [0.9, 0.8, 0.7]
        assert nms(boxes, scores, native=True) == nms(boxes, scores, native=False)


class TestCompilerSelection:
    def test_explicit_compiler_wins(self):
        assert _compiler_candidates("my-cxx") == ["my-cxx"]

    def test_env_cxx_used(self, monkeypatch):
        monkeypatch.setenv("CXX", "envcc")
        assert _compiler_candidates(None) == ["envcc"]

    def test_default_candidates_nonempty(self, monkeypatch):
        monkeypatch.delenv("CXX", raising=False)
        assert _compiler_candidates(None)


class TestBuildCommands:
    def test_gnu_command_shape(self, tmp_path):
        cmd, workdir = _build_command("g++", tmp_path / "src.cpp", tmp_path / "out.so")
        assert cmd[0] == "g++"
        assert "-std=c++17" in cmd and "-shared" in cmd
        assert workdir is None

    def test_msvc_command_uses_temp_workdir(self, tmp_path):
        cmd, workdir = _build_command("cl", tmp_path / "src.cpp", tmp_path / "out.dll")
        assert cmd[0] == "cl"
        assert "/LD" in cmd and any(a.startswith("/Fe:") for a in cmd)
        assert workdir is not None and workdir.exists()
        import shutil as _sh
        _sh.rmtree(workdir, ignore_errors=True)

    def test_msvc_detected_by_stem(self, tmp_path):
        # A full path to cl.exe must still route to the MSVC branch.
        cmd, workdir = _build_command(
            r"C:\VS\cl.exe", tmp_path / "s.cpp", tmp_path / "o.dll"
        )
        assert "/LD" in cmd
        if workdir:
            import shutil as _sh
            _sh.rmtree(workdir, ignore_errors=True)
