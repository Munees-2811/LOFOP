"""LOFOP deployment subsystem: model export and runtime post-processing.

``onnx_export`` needs PyTorch (and onnxruntime for verification);
``postprocess`` is torch-free so inference hosts only need the core
framework beside their ONNX runtime.
"""

from lofop.deploy.postprocess import Detections, postprocess_dense

__all__ = ["Detections", "postprocess_dense", "export_onnx", "DenseExportWrapper"]


def __getattr__(name: str):
    if name in ("export_onnx", "DenseExportWrapper"):
        from lofop.deploy import onnx_export

        return getattr(onnx_export, name)
    raise AttributeError(f"module 'lofop.deploy' has no attribute {name!r}")
