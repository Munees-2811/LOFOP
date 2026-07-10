"""Standardized LOFOP-Detect benchmark suite: one runner, three output formats.

Benchmarks every model variant in ``lofop/configs/lofop-detect/`` (excluding
``base``) with the shared :func:`lofop.utils.benchmark.benchmark_model`
instrument -- parameters, FLOPs, state size, CPU (and GPU when available)
forward FPS, and peak process memory -- then writes ``results.md``,
``results.csv``, and ``results.json`` so the same run is readable by humans and
by scripts/CI. Accuracy columns stay empty unless a checkpoint + evaluation is
supplied elsewhere; this suite never invents numbers.

For clean per-model memory, ``--isolated`` re-runs each variant in a fresh
subprocess (peak RSS is a monotonic process-wide figure otherwise).

Usage::

    python benchmarks/run_suite.py [--results-dir benchmarks/results] [--size 640] [--isolated]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lofop.models  # noqa: E402, F401  (registers model components)
from lofop.core.config import Config  # noqa: E402
from lofop.registries import HUB  # noqa: E402
from lofop.utils import ModelReport, benchmark_model, render_table, write_reports  # noqa: E402

_CONFIG_DIR = REPO_ROOT / "lofop" / "configs" / "lofop-detect"


def _variants() -> list[Path]:
    return [p for p in sorted(_CONFIG_DIR.glob("*.yaml")) if p.stem != "base"]


def _benchmark_one(config_path: Path, size: int) -> ModelReport:
    model = HUB.build(Config.load(config_path).model)
    return benchmark_model(model, f"lofop-detect-{config_path.stem}", image_size=size)


def _benchmark_isolated(config_path: Path, size: int) -> ModelReport:
    """Run one variant in a subprocess and parse its single-record JSON."""
    proc = subprocess.run(
        [sys.executable, __file__, "--one", str(config_path), "--size", str(size)],
        check=True, capture_output=True, text=True,
    )
    record = json.loads(proc.stdout)
    accuracy = None
    return ModelReport(
        name=record["name"], parameters=record["parameters"], flops=record["flops"],
        size_mb=record["size_mb"], cpu_fps=record["cpu_fps"], gpu_fps=record["gpu_fps"],
        peak_rss_mb=record["peak_rss_mb"], accuracy=accuracy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LOFOP-Detect benchmark suite.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results")
    parser.add_argument("--size", type=int, default=640, help="input resolution")
    parser.add_argument(
        "--isolated", action="store_true", help="benchmark each variant in a fresh process",
    )
    parser.add_argument("--one", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    # Worker mode: benchmark a single config and emit its record as JSON.
    if args.one is not None:
        report = _benchmark_one(args.one, args.size)
        print(json.dumps(report.to_record()))
        return 0

    reports: list[ModelReport] = []
    for config_path in _variants():
        report = (
            _benchmark_isolated(config_path, args.size)
            if args.isolated else _benchmark_one(config_path, args.size)
        )
        print(
            f"  {report.name}: {report.parameters:,} params, "
            f"{report.flops / 1e9:.2f} GFLOPs, {report.cpu_fps:.1f} CPU FPS",
            file=sys.stderr,
        )
        reports.append(report)

    print(render_table(reports))
    written = write_reports(reports, args.results_dir)
    for fmt, path in written.items():
        print(f"  wrote {fmt}: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
