"""LOFOP deployment subsystem: model export and runtime post-processing.

``onnx_export`` needs PyTorch (and onnxruntime for verification);
``postprocess`` is torch-free so inference hosts only need the core
framework beside their ONNX runtime.
"""

from lofop.deploy.postprocess import Detections, postprocess_dense

__all__ = [
    "Detections", "postprocess_dense",
    "export_onnx", "DenseExportWrapper",
    "export_tensorrt", "build_engine_from_onnx",
]

_ONNX_EXPORTS = {"export_onnx", "DenseExportWrapper"}
_TRT_EXPORTS = {"export_tensorrt", "build_engine_from_onnx"}


def __getattr__(name: str):
    # Lazy so importing lofop.deploy never requires torch/tensorrt.
    if name in _ONNX_EXPORTS:
        from lofop.deploy import onnx_export

        return getattr(onnx_export, name)
    if name in _TRT_EXPORTS:
        from lofop.deploy import tensorrt_export

        return getattr(tensorrt_export, name)
    raise AttributeError(f"module 'lofop.deploy' has no attribute {name!r}")
