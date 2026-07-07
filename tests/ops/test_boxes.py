"""Tests for lofop.ops: pure Python reference and native C++ parity."""

import random
import shutil

import pytest

from lofop.core.exceptions import LofopError
from lofop.ops import batched_nms, iou_matrix, nms
from lofop.ops.native import build_native, load_native

HAVE_COMPILER = shutil.which("g++") or shutil.which("clang++")

needs_compiler = pytest.mark.skipif(not HAVE_COMPILER, reason="no C++ compiler available")


@pytest.fixture(scope="module")
def native_lib():
    if not HAVE_COMPILER:
        pytest.skip("no C++ compiler available")
    build_native()
    lib = load_native()
    assert lib is not None
    return lib


def random_boxes(n, seed, span=1000.0):
    rng = random.Random(seed)
    boxes, scores = [], []
    for _ in range(n):
        x1 = rng.uniform(0, span)
        y1 = rng.uniform(0, span)
        boxes.append([x1, y1, x1 + rng.uniform(1, 120), y1 + rng.uniform(1, 120)])
        scores.append(rng.random())
    return boxes, scores


class TestIouPython:
    def test_known_values(self):
        a = [[0, 0, 10, 10]]
        b = [[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]]
        row = iou_matrix(a, b, native=False)[0]
        assert row[0] == pytest.approx(1.0)
        assert row[1] == pytest.approx(25 / 175)
        assert row[2] == 0.0

    def test_empty_inputs(self):
        assert iou_matrix([], [[0, 0, 1, 1]], native=False) == []
        assert iou_matrix([[0, 0, 1, 1]], [], native=False) == [[]]

    def test_degenerate_boxes_have_zero_iou(self):
        assert iou_matrix([[5, 5, 5, 5]], [[0, 0, 10, 10]], native=False)[0][0] == 0.0

    def test_bad_box_shape(self):
        with pytest.raises(LofopError):
            iou_matrix([[0, 0, 1]], [[0, 0, 1, 1]], native=False)


class TestNmsPython:
    def test_suppresses_overlaps_keeps_score_order(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
        scores = [0.9, 0.8, 0.7]
        assert nms(boxes, scores, iou_threshold=0.5, native=False) == [0, 2]

    def test_low_threshold_keeps_disjoint(self):
        boxes = [[0, 0, 10, 10], [100, 100, 110, 110]]
        assert nms(boxes, [0.5, 0.9], iou_threshold=0.1, native=False) == [1, 0]

    def test_max_keep(self):
        boxes, scores = random_boxes(50, seed=1)
        assert len(nms(boxes, scores, iou_threshold=0.99, max_keep=5, native=False)) == 5

    def test_empty_and_mismatched(self):
        assert nms([], [], native=False) == []
        with pytest.raises(LofopError):
            nms([[0, 0, 1, 1]], [], native=False)

    def test_batched_nms_isolates_classes(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11]]
        scores = [0.9, 0.8]
        # Same class: second box suppressed. Different classes: both kept.
        assert batched_nms(boxes, scores, [0, 0], iou_threshold=0.5, native=False) == [0]
        assert batched_nms(boxes, scores, [0, 1], iou_threshold=0.5, native=False) == [0, 1]


class TestNativeParity:
    def test_build_is_idempotent(self, native_lib):
        first = build_native()
        assert build_native() == first

    def test_iou_matches_python(self, native_lib):
        a, _ = random_boxes(60, seed=2)
        b, _ = random_boxes(40, seed=3)
        py = iou_matrix(a, b, native=False)
        cc = iou_matrix(a, b, native=True)
        for row_py, row_cc in zip(py, cc):
            assert row_cc == pytest.approx(row_py, abs=1e-5)

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_nms_matches_python(self, native_lib, threshold):
        boxes, scores = random_boxes(300, seed=4, span=400.0)
        py = nms(boxes, scores, iou_threshold=threshold, native=False)
        cc = nms(boxes, scores, iou_threshold=threshold, native=True)
        assert cc == py

    def test_nms_max_keep_matches(self, native_lib):
        boxes, scores = random_boxes(200, seed=5, span=300.0)
        py = nms(boxes, scores, iou_threshold=0.5, max_keep=10, native=False)
        cc = nms(boxes, scores, iou_threshold=0.5, max_keep=10, native=True)
        assert cc == py

    def test_batched_nms_matches(self, native_lib):
        boxes, scores = random_boxes(200, seed=6, span=300.0)
        classes = [i % 3 for i in range(len(boxes))]
        py = batched_nms(boxes, scores, classes, iou_threshold=0.5, native=False)
        cc = batched_nms(boxes, scores, classes, iou_threshold=0.5, native=True)
        assert cc == py


def test_native_true_without_library_raises(monkeypatch):
    monkeypatch.setattr("lofop.ops.boxes.load_native", lambda: None)
    with pytest.raises(LofopError, match="not built"):
        nms([[0, 0, 1, 1]], [0.5], native=True)
