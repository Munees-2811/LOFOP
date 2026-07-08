"""TensorRT export for LOFOP detectors.

TensorRT is NVIDIA's inference optimizer: it consumes an ONNX graph and emits
a serialized ``.engine`` tuned for a specific GPU, typically the fastest way
to run a detector on NVIDIA hardware. LOFOP exports in two stages -- model ->
ONNX (via :mod:`lofop.deploy.onnx_export`), then ONNX -> engine here -- so the
same verified ONNX graph feeds both the ONNX Runtime and TensorRT paths, and
post-processing (:func:`lofop.deploy.postprocess.postprocess_dense`) is shared
across runtimes.

TensorRT and a CUDA GPU are required only for the engine-build step; the
module imports without them, and the ONNX intermediate is produced regardless
so failures are localized to the GPU-specific stage. FP16 is a single builder
flag; INT8 additionally needs representative calibration data.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from lofop.core.exceptions import LofopError
from lofop.core.logging import get_logger
from lofop.deploy.onnx_export import export_onnx

if TYPE_CHECKING:
    from torch import nn

logger = get_logger(__name__)

# Default TensorRT builder workspace (bytes). 1 GiB suits the LOFOP-Detect
# family; raise for larger models or more aggressive tactic search.
_DEFAULT_WORKSPACE = 1 << 30


def export_tensorrt(
    model: nn.Module,
    engine_path: str | Path,
    *,
    image_size: int = 640,
    fp16: bool = False,
    int8: bool = False,
    calibration_inputs: Sequence | None = None,
    workspace: int = _DEFAULT_WORKSPACE,
    onnx_path: str | Path | None = None,
    keep_onnx: bool = False,
) -> Path:
    """Export a detector to a TensorRT engine (via an ONNX intermediate).

    Args:
        model: Detector to export (weights already loaded).
        engine_path: Output ``.engine`` path.
        image_size: Fixed input resolution baked into the graph.
        fp16: Enable FP16 kernels (large speedup, negligible accuracy cost on
            most detectors).
        int8: Enable INT8 kernels; requires ``calibration_inputs``.
        calibration_inputs: Sequence of representative ``(1, 3, S, S)`` input
            arrays used to calibrate INT8 dynamic ranges.
        workspace: Builder workspace memory budget in bytes.
        onnx_path: Where to write the intermediate ONNX; defaults to the
            engine path with a ``.onnx`` suffix.
        keep_onnx: Keep the intermediate ONNX file after building the engine.

    Returns:
        The written engine path.

    Raises:
        LofopError: If TensorRT is unavailable, INT8 is requested without
            calibration data, or the build fails.
    """
    if int8 and not calibration_inputs:
        raise LofopError("INT8 export requires calibration_inputs with representative data")

    engine_path = Path(engine_path)
    intermediate = Path(onnx_path) if onnx_path else engine_path.with_suffix(".onnx")
    export_onnx(model, intermediate, image_size=image_size, verify=False)
    try:
        build_engine_from_onnx(
            intermediate, engine_path, fp16=fp16, int8=int8,
            calibration_inputs=calibration_inputs, workspace=workspace,
        )
    finally:
        if not keep_onnx and intermediate.exists():
            intermediate.unlink()
    return engine_path


def build_engine_from_onnx(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp16: bool = False,
    int8: bool = False,
    calibration_inputs: Sequence | None = None,
    workspace: int = _DEFAULT_WORKSPACE,
) -> Path:
    """Build a serialized TensorRT engine from an existing ONNX file.

    Raises:
        LofopError: If TensorRT is not importable or the build fails.
    """
    trt = _import_tensorrt()
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    if not onnx_path.is_file():
        raise LofopError("ONNX file not found for engine build", context={"onnx": str(onnx_path)})

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise LofopError("TensorRT failed to parse ONNX", context={"errors": errors})

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if fp16:
        if not builder.platform_has_fast_fp16:
            logger.warning("Platform reports no fast FP16; enabling anyway")
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _make_calibrator(trt, network, calibration_inputs)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise LofopError("TensorRT engine build returned no result")
    engine_path.write_bytes(serialized)
    logger.info(
        "Built TensorRT engine at %s (%.1f MB, fp16=%s, int8=%s)",
        engine_path, engine_path.stat().st_size / 1e6, fp16, int8,
    )
    return engine_path


def _import_tensorrt():
    try:
        import tensorrt as trt

        return trt
    except ImportError as exc:
        raise LofopError(
            "TensorRT is not installed. Install the NVIDIA 'tensorrt' package "
            "on a CUDA-capable machine to build engines.",
        ) from exc


def _make_calibrator(trt, network, calibration_inputs: Sequence):
    """Build a minimal entropy calibrator over provided input arrays."""
    import numpy as np
    from cuda import cudart  # provided by the tensorrt/cuda stack

    input_tensor = network.get_input(0)
    shape = tuple(input_tensor.shape)

    class _EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._batches = iter(calibration_inputs)
            nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
            self._device_input = cudart.cudaMalloc(nbytes)[1]

        def get_batch_size(self) -> int:
            return shape[0]

        def get_batch(self, names):
            try:
                batch = np.ascontiguousarray(next(self._batches), dtype=np.float32)
            except StopIteration:
                return None
            cudart.cudaMemcpy(
                self._device_input, batch.ctypes.data, batch.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            return [int(self._device_input)]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            return None

    return _EntropyCalibrator()
