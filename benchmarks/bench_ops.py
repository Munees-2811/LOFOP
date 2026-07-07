"""Benchmark: pure Python vs native C++ box ops.

Compares the two implementations of IoU and NMS at candidate counts typical
of multi-scale detection post-processing (a pyramid detector emits hundreds
to thousands of score-filtered candidates per image, skewed toward small
objects on fine levels). Builds the native library on demand.

Usage::

    python benchmarks/bench_ops.py [-o report.md] [--repeats N]
"""

from __future__ import annotations

import argparse
import random
import sys
import timeit
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lofop.ops import build_native, iou_matrix, nms  # noqa: E402

_SIZES = [100, 500, 2000]
_IOU_SIZES = [(100, 100), (500, 500)]


def _random_boxes(n: int, seed: int) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    boxes, scores = [], []
    for _ in range(n):
        x1, y1 = rng.uniform(0, 1300), rng.uniform(0, 800)
        # Mix scales: mostly small objects, some large, like real pyramids.
        side = rng.uniform(4, 40) if rng.random() < 0.8 else rng.uniform(80, 400)
        boxes.append([x1, y1, x1 + side, y1 + side * rng.uniform(0.5, 2.0)])
        scores.append(rng.random())
    return boxes, scores


def _time_ms(fn, repeats: int) -> float:
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeats, number=number)) / number * 1e3


def run(repeats: int) -> list[str]:
    build_native()
    rows = [
        "| Operation | Size | Python (ms) | C++ (ms) | Speedup |",
        "|---|---|---:|---:|---:|",
    ]
    for n in _SIZES:
        boxes, scores = _random_boxes(n, seed=n)
        py = _time_ms(lambda bx=boxes, sc=scores: nms(bx, sc, iou_threshold=0.5, native=False),
                      repeats)
        cc = _time_ms(lambda bx=boxes, sc=scores: nms(bx, sc, iou_threshold=0.5, native=True),
                      repeats)
        rows.append(f"| NMS | {n} boxes | {py:.3f} | {cc:.3f} | {py / cc:.1f}x |")
        print(f"  nms n={n}: python {py:.3f} ms, native {cc:.3f} ms", file=sys.stderr)
    for n, m in _IOU_SIZES:
        a, _ = _random_boxes(n, seed=n + 1)
        b, _ = _random_boxes(m, seed=m + 2)
        py = _time_ms(lambda ba=a, bb=b: iou_matrix(ba, bb, native=False), repeats)
        cc = _time_ms(lambda ba=a, bb=b: iou_matrix(ba, bb, native=True), repeats)
        rows.append(f"| IoU matrix | {n}x{m} | {py:.3f} | {cc:.3f} | {py / cc:.1f}x |")
        print(f"  iou {n}x{m}: python {py:.3f} ms, native {cc:.3f} ms", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Python vs native C++ box ops.")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    rows = run(args.repeats)
    report = "\n".join([
        "# LOFOP ops benchmark: pure Python vs native C++",
        "",
        "Box counts reflect score-filtered candidates per image in multi-scale detection.",
        "Times are best-of-repeats per call; native results are verified equal to the",
        "Python reference by the test suite.",
        "",
        *rows,
        "",
    ])
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
