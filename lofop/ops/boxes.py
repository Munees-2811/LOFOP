"""Box operations: IoU and non-maximum suppression.

These are the CPU-side hot path of detection post-processing. A multi-scale
detector predicts candidates for tiny objects on fine pyramid levels and huge
objects on coarse levels; after score filtering, thousands of overlapping
candidates per image reach NMS. Each op therefore has two implementations:

* a pure Python reference (always available, used for verification), and
* the native C++ kernel from ``lofop/csrc`` (used automatically when built --
  see :mod:`lofop.ops.native`).

Both produce identical results; tests assert it. Inputs are any sequences of
floats -- ``list``s, ``array``s, numpy arrays -- with boxes in absolute xyxy.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

from lofop.core.exceptions import LofopError
from lofop.ops.native import load_native

BoxLike = Sequence[Sequence[float]]
IouMatrix = list[list[float]]

# Class-aware NMS separates classes by shifting each class onto its own
# coordinate island; the offset just needs to exceed any real coordinate.
# Public so callers doing the shift themselves (e.g. in tensor math, to hit
# the zero-copy nms path) stay consistent with batched_nms.
CLASS_OFFSET = 1e7


def _parse_boxes(boxes: BoxLike) -> list[tuple[float, float, float, float]]:
    parsed = []
    for box in boxes:
        if len(box) != 4:
            raise LofopError("Boxes must be xyxy quadruples", context={"got": list(box)})
        parsed.append(tuple(float(v) for v in box))
    return parsed


def _flatten_boxes(boxes: BoxLike) -> list[float]:
    return [v for box in _parse_boxes(boxes) for v in box]


def _c_float_view(obj: Any, count: int):
    """ctypes float pointer for ``obj``, zero-copy when possible.

    A contiguous float32 CPU tensor-like (duck-typed via ``data_ptr``) is
    viewed in place -- no per-element Python conversion -- which is what makes
    the native NMS path fast on detector outputs. Anything else is copied
    through the generic list path. Returns ``(pointer, keepalive)``; the
    keepalive must stay referenced while the pointer is in use.
    """
    if callable(getattr(obj, "data_ptr", None)):
        try:
            zero_copy = (
                str(obj.dtype) == "torch.float32"
                and obj.device.type == "cpu"
                and obj.is_contiguous()
                and obj.numel() == count
            )
        except AttributeError:
            zero_copy = False
        if zero_copy:
            return ctypes.cast(obj.data_ptr(), ctypes.POINTER(ctypes.c_float)), obj
        flat = obj.detach().reshape(-1).tolist()
        array = (ctypes.c_float * count)(*flat)
        return array, array
    return None, None


def iou_matrix(boxes_a: BoxLike, boxes_b: BoxLike, *, native: bool | None = None) -> IouMatrix:
    """Pairwise IoU between two box sets.

    Args:
        boxes_a: N boxes, xyxy.
        boxes_b: M boxes, xyxy.
        native: Force the native (``True``) or Python (``False``) path;
            ``None`` auto-selects native when built.

    Returns:
        N x M nested lists of IoU values in ``[0, 1]``.
    """
    lib = load_native() if native in (None, True) else None
    if native is True and lib is None:
        raise LofopError("Native ops library is not built; run lofop.ops.native.build_native()")
    n, m = len(boxes_a), len(boxes_b)
    if n == 0 or m == 0:
        return [[0.0] * m for _ in range(n)]
    if lib is not None:
        a = (ctypes.c_float * (n * 4))(*_flatten_boxes(boxes_a))
        b = (ctypes.c_float * (m * 4))(*_flatten_boxes(boxes_b))
        out = (ctypes.c_float * (n * m))()
        lib.lofop_iou_matrix(a, n, b, m, out)
        return [list(out[i * m:(i + 1) * m]) for i in range(n)]
    parsed_a, parsed_b = _parse_boxes(boxes_a), _parse_boxes(boxes_b)
    return [[_pair_iou(pa, pb) for pb in parsed_b] for pa in parsed_a]


def nms(
    boxes: BoxLike,
    scores: Sequence[float],
    *,
    iou_threshold: float = 0.5,
    max_keep: int = 0,
    native: bool | None = None,
) -> list[int]:
    """Greedy non-maximum suppression.

    Args:
        boxes: N boxes, xyxy.
        scores: N confidence scores.
        iou_threshold: Boxes overlapping a kept box above this are dropped.
        max_keep: Stop after keeping this many boxes (0 = no limit).
        native: Force the native (``True``) or Python (``False``) path;
            ``None`` auto-selects native when built.

    Returns:
        Indices of kept boxes, highest score first.
    """
    if len(boxes) != len(scores):
        raise LofopError(
            "boxes and scores must have equal length",
            context={"boxes": len(boxes), "scores": len(scores)},
        )
    n = len(boxes)
    if n == 0:
        return []
    lib = load_native() if native in (None, True) else None
    if native is True and lib is None:
        raise LofopError("Native ops library is not built; run lofop.ops.native.build_native()")
    if lib is not None:
        c_boxes, keep_boxes = _c_float_view(boxes, n * 4)
        if c_boxes is None:
            c_boxes = (ctypes.c_float * (n * 4))(*_flatten_boxes(boxes))
            keep_boxes = c_boxes
        c_scores, keep_scores = _c_float_view(scores, n)
        if c_scores is None:
            c_scores = (ctypes.c_float * n)(*[float(s) for s in scores])
            keep_scores = c_scores
        keep = (ctypes.c_int32 * n)()
        kept = lib.lofop_nms(c_boxes, c_scores, n, float(iou_threshold), int(max_keep), keep)
        del keep_boxes, keep_scores  # buffers alive through the native call
        return list(keep[:kept])
    if hasattr(boxes, "tolist"):
        boxes, scores = boxes.tolist(), [float(s) for s in scores]
    return _nms_python(boxes, scores, iou_threshold, max_keep)


def batched_nms(
    boxes: BoxLike,
    scores: Sequence[float],
    class_ids: Sequence[int],
    *,
    iou_threshold: float = 0.5,
    max_keep: int = 0,
    native: bool | None = None,
) -> list[int]:
    """Class-aware NMS: boxes only suppress boxes of the same class.

    Implemented with the standard coordinate-offset trick, so one native NMS
    call handles all classes at once.
    """
    if not (len(boxes) == len(scores) == len(class_ids)):
        raise LofopError(
            "boxes, scores, and class_ids must have equal length",
            context={"boxes": len(boxes), "scores": len(scores), "classes": len(class_ids)},
        )
    shifted = [
        [float(v) + CLASS_OFFSET * int(cls) for v in box]
        for box, cls in zip(boxes, class_ids)
    ]
    return nms(shifted, scores, iou_threshold=iou_threshold, max_keep=max_keep, native=native)


def _pair_iou(a: tuple, b: tuple) -> float:
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _nms_python(boxes: BoxLike, scores: Sequence[float], iou_threshold: float,
                max_keep: int) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: -float(scores[i]))
    parsed = _parse_boxes(boxes)
    suppressed = [False] * len(boxes)
    keep: list[int] = []
    for oi, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        if max_keep > 0 and len(keep) >= max_keep:
            break
        for j in order[oi + 1:]:
            if not suppressed[j] and _pair_iou(parsed[i], parsed[j]) > iou_threshold:
                suppressed[j] = True
    return keep
