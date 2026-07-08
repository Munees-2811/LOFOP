"""LOFOP ops: performance-critical primitives with native C++ acceleration.

Every op has a pure Python reference implementation and, when the library in
``lofop/csrc`` has been compiled (``lofop.ops.native.build_native()``), a
drop-in C++ fast path producing identical results.
"""

from lofop.ops.boxes import batched_nms, iou_matrix, nms
from lofop.ops.native import backend, build_native, find_library, load_native

__all__ = [
    "iou_matrix", "nms", "batched_nms",
    "backend", "build_native", "find_library", "load_native",
]
