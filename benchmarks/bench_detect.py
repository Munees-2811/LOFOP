"""Benchmark LOFOP-Detect variants: parameters and forward latency.

Measures what this machine can honestly measure -- model size and CPU forward
latency at 640x640 -- for each config in configs/lofop-detect/. RT-DETR's
published parameter counts are printed alongside as context; accuracy (mAP)
comparison requires the Phase 4 training run on GPU hardware and is
deliberately absent here (see docs/lofop-detect.md section 5).

Usage::

    python benchmarks/bench_detect.py [-o report.md] [--size 640] [--repeats 5]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import lofop.models  # noqa: E402, F401  (registers model components)
from lofop.core.config import Config  # noqa: E402
from lofop.registries import HUB  # noqa: E402

_CONFIG_DIR = REPO_ROOT / "configs" / "lofop-detect"
# Published reference points (RT-DETR paper); different hardware, shown as
# scale context only, never as a measured comparison.
_REFERENCES = [("RT-DETR-R18 (paper)", 20_000_000), ("RT-DETR-R50 (paper)", 42_000_000)]


def measure_variant(config_path: Path, size: int, repeats: int) -> tuple[str, int, float]:
    cfg = Config.load(config_path)
    model = HUB.build(cfg.model).eval()
    params = sum(p.numel() for p in model.parameters())
    images = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        for _ in range(2):
            model(images)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(images)
            times.append((time.perf_counter() - start) * 1e3)
    return config_path.stem, params, statistics.median(times)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark LOFOP-Detect variants.")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--size", type=int, default=640, help="input resolution")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    lines = [
        "# LOFOP-Detect benchmark",
        "",
        f"Forward pass at {args.size}x{args.size}, batch 1, CPU ({torch.get_num_threads()} "
        f"threads), torch {torch.__version__}. Latency is the median of "
        f"{args.repeats} runs after warmup.",
        "",
        "| Model | Params | CPU forward (ms) |",
        "|---|---:|---:|",
    ]
    for config_path in sorted(_CONFIG_DIR.glob("*.yaml")):
        if config_path.stem == "base":
            continue
        name, params, latency = measure_variant(config_path, args.size, args.repeats)
        print(f"  lofop-detect-{name}: {params:,} params, {latency:.1f} ms", file=sys.stderr)
        lines.append(f"| lofop-detect-{name} | {params:,} | {latency:.1f} |")
    for ref_name, ref_params in _REFERENCES:
        lines.append(f"| {ref_name} | {ref_params:,} | n/a (GPU model) |")
    lines += [
        "",
        "Reference rows are published parameter counts for scale context; latency and",
        "accuracy comparisons against RT-DETR require the Phase 4 GPU training run.",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
