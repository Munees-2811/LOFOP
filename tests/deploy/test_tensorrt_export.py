"""Tests for TensorRT export.

TensorRT and a CUDA GPU are not available in CI, so these tests verify the
parts that must work without them: the module imports, the ONNX intermediate
is produced, argument validation fires, and the missing-TensorRT path raises
a clear, actionable error rather than an opaque ImportError.
"""

import importlib.util

import pytest

torch = pytest.importorskip("torch")

from lofop.core.exceptions import LofopError  # noqa: E402
from lofop.deploy import build_engine_from_onnx, export_tensorrt  # noqa: E402
from lofop.models import ApexHead, DeltaFusion, LofopDetect, RidgeNet  # noqa: E402

HAVE_TENSORRT = importlib.util.find_spec("tensorrt") is not None
IMAGE_SIZE = 64


def tiny_model():
    backbone = RidgeNet(widths=(16, 32, 64, 128), depths=(1, 1, 1, 1))
    neck = DeltaFusion(in_channels=backbone.out_channels, width=32)
    head = ApexHead(num_classes=3, width=32, num_convs=1)
    return LofopDetect(backbone, neck, head).eval()


class TestArgumentValidation:
    def test_int8_requires_calibration(self, tmp_path):
        with pytest.raises(LofopError, match="calibration"):
            export_tensorrt(
                tiny_model(), tmp_path / "m.engine", image_size=IMAGE_SIZE,
                int8=True, calibration_inputs=None,
            )

    def test_build_from_missing_onnx(self, tmp_path):
        if not HAVE_TENSORRT:
            pytest.skip("tensorrt not installed")
        with pytest.raises(LofopError, match="not found"):
            build_engine_from_onnx(tmp_path / "nope.onnx", tmp_path / "m.engine")


@pytest.mark.skipif(HAVE_TENSORRT, reason="exercises the no-TensorRT path")
class TestWithoutTensorRT:
    def test_export_produces_onnx_then_fails_clearly(self, tmp_path):
        engine = tmp_path / "model.engine"
        with pytest.raises(LofopError, match="TensorRT is not installed"):
            export_tensorrt(
                tiny_model(), engine, image_size=IMAGE_SIZE, keep_onnx=True,
                onnx_path=tmp_path / "model.onnx",
            )
        # The ONNX stage runs before the GPU stage, so the intermediate exists
        # and is reusable even though the engine build could not proceed.
        assert (tmp_path / "model.onnx").is_file()

    def test_build_engine_reports_missing_dependency(self, tmp_path):
        onnx = tmp_path / "m.onnx"
        onnx.write_bytes(b"not a real onnx")  # never parsed; import fails first
        with pytest.raises(LofopError, match="TensorRT is not installed"):
            build_engine_from_onnx(onnx, tmp_path / "m.engine")


@pytest.mark.skipif(not HAVE_TENSORRT, reason="requires TensorRT + CUDA GPU")
class TestWithTensorRT:
    def test_fp32_engine_builds(self, tmp_path):
        engine = export_tensorrt(tiny_model(), tmp_path / "m.engine", image_size=IMAGE_SIZE)
        assert engine.is_file() and engine.stat().st_size > 0
