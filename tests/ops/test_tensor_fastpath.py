"""Tests for the zero-copy tensor fast path into the native NMS."""

import shutil

import pytest

torch = pytest.importorskip("torch")

from lofop.ops import nms  # noqa: E402
from lofop.ops.boxes import CLASS_OFFSET, batched_nms  # noqa: E402

HAVE_COMPILER = shutil.which("g++") or shutil.which("clang++")


def random_case(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 400
    wh = torch.rand(n, 2, generator=g) * 100 + 1
    boxes = torch.cat([xy, xy + wh], dim=1)
    scores = torch.rand(n, generator=g)
    labels = torch.randint(0, 3, (n,), generator=g)
    return boxes, scores, labels


class TestTensorInputs:
    def test_tensor_matches_list_python_path(self):
        boxes, scores, _ = random_case()
        from_tensor = nms(boxes, scores, iou_threshold=0.5, native=False)
        from_lists = nms(boxes.tolist(), scores.tolist(), iou_threshold=0.5, native=False)
        assert from_tensor == from_lists

    @pytest.mark.skipif(not HAVE_COMPILER, reason="no C++ compiler available")
    def test_tensor_matches_list_native_path(self):
        from lofop.ops import build_native
        from lofop.ops.native import _reset_cache

        build_native()
        _reset_cache()
        boxes, scores, _ = random_case(seed=1)
        from_tensor = nms(boxes.contiguous(), scores.contiguous(),
                          iou_threshold=0.5, native=True)
        from_lists = nms(boxes.tolist(), scores.tolist(), iou_threshold=0.5, native=True)
        assert from_tensor == from_lists

    @pytest.mark.skipif(not HAVE_COMPILER, reason="no C++ compiler available")
    def test_noncontiguous_and_float64_fall_back_to_copy(self):
        from lofop.ops import build_native
        from lofop.ops.native import _reset_cache

        build_native()
        _reset_cache()
        boxes, scores, _ = random_case(seed=2)
        noncontig = boxes.t().t()[::1]  # same values; may lose contiguity via slicing
        double = boxes.double()
        expected = nms(boxes, scores, iou_threshold=0.5, native=True)
        assert nms(noncontig, scores, iou_threshold=0.5, native=True) == expected
        assert nms(double, scores, iou_threshold=0.5, native=True) == expected

    def test_offset_shift_matches_batched_nms(self):
        boxes, scores, labels = random_case(seed=3)
        shifted = boxes + (labels.to(boxes.dtype) * CLASS_OFFSET).unsqueeze(1)
        via_shift = nms(shifted.contiguous(), scores.contiguous(),
                        iou_threshold=0.5, native=False)
        via_batched = batched_nms(
            boxes.tolist(), scores.tolist(), labels.tolist(),
            iou_threshold=0.5, native=False,
        )
        assert via_shift == via_batched

    def test_empty_tensor_input(self):
        assert nms(torch.zeros((0, 4)), torch.zeros((0,)), iou_threshold=0.5) == []
