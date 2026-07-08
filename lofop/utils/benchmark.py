"""Model benchmarking: the LOFOP metric table.

Produces the standard comparison table -- mAP@50, mAP@50:95, Precision,
Recall, GPU FPS, CPU FPS, Parameters, FLOPs, Model Size -- for one or more
models. Structural metrics (params/FLOPs/size/FPS) are always measurable;
accuracy rows fill in when an evaluation dataset (or precomputed metrics)
is supplied, and render as "-" otherwise, so the table never shows numbers
that were not actually measured.

FLOPs are counted by forward hooks on Conv2d and Linear layers as
2 * multiply-accumulates; normalization and activation costs are excluded
(sub-percent for detection models).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from lofop.training.evaluator import DetectionMetrics


@dataclass
class ModelReport:
    """All metrics for one model column of the table.

    Attributes:
        name: Column heading.
        parameters: Trainable + non-trainable parameter count.
        flops: Estimated forward FLOPs at the benchmark resolution.
        size_mb: State size (parameters + buffers) in megabytes.
        cpu_fps: Batch-1 throughput on CPU.
        gpu_fps: Batch-1 throughput on CUDA, when available.
        accuracy: Detection metrics, when an evaluation was run.
    """

    name: str
    parameters: int
    flops: int
    size_mb: float
    cpu_fps: float | None = None
    gpu_fps: float | None = None
    accuracy: DetectionMetrics | None = None
    notes: dict[str, str] = field(default_factory=dict)


def count_flops(model: nn.Module, image_size: int = 640) -> int:
    """Estimate forward FLOPs (2 x MACs) of conv and linear layers."""
    totals: list[int] = []
    hooks = []

    def conv_hook(module: nn.Conv2d, _inputs, output) -> None:
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        macs = output.numel() * kernel_ops * (module.in_channels // module.groups)
        totals.append(2 * macs)

    def linear_hook(module: nn.Linear, _inputs, output) -> None:
        totals.append(2 * output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    try:
        with torch.inference_mode():
            model(torch.randn(1, 3, image_size, image_size))
    finally:
        for hook in hooks:
            hook.remove()
    return sum(totals)


def measure_fps(
    model: nn.Module, *, device: str, image_size: int = 640, repeats: int = 10
) -> float:
    """Median batch-1 forward throughput (frames per second) on ``device``."""
    target = torch.device(device)
    model = model.to(target)
    images = torch.randn(1, 3, image_size, image_size, device=target)
    with torch.inference_mode():
        for _ in range(3):
            model(images)
        if target.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(images)
            if target.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return 1.0 / statistics.median(times)


def benchmark_model(
    model: nn.Module,
    name: str,
    *,
    image_size: int = 640,
    accuracy: DetectionMetrics | None = None,
) -> ModelReport:
    """Measure every structural metric for one model; attach accuracy if given."""
    model = model.eval().cpu()
    parameters = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    report = ModelReport(
        name=name,
        parameters=parameters,
        flops=count_flops(model, image_size),
        size_mb=size_bytes / 1e6,
        cpu_fps=measure_fps(model, device="cpu", image_size=image_size),
        accuracy=accuracy,
    )
    if torch.cuda.is_available():
        report.gpu_fps = measure_fps(model, device="cuda", image_size=image_size)
        model.cpu()
    return report


def render_table(
    reports: list[ModelReport], *, reference_columns: dict[str, dict] | None = None
) -> str:
    """Render the metric table as markdown.

    Args:
        reports: Measured model columns.
        reference_columns: Optional extra columns of published numbers, e.g.
            ``{"RT-DETR-R18 (paper)": {"Parameters": "20,000,000", ...}}`` --
            rendered verbatim so measured and quoted values are never mixed
            silently.
    """
    def acc(report: ModelReport, attr: str) -> str:
        if report.accuracy is None:
            return "-"
        return f"{getattr(report.accuracy, attr):.4f}"

    rows = [
        ("mAP@50", lambda r: acc(r, "map50")),
        ("mAP@50:95", lambda r: acc(r, "map50_95")),
        ("Precision", lambda r: acc(r, "precision")),
        ("Recall", lambda r: acc(r, "recall")),
        ("GPU FPS", lambda r: f"{r.gpu_fps:.1f}" if r.gpu_fps else "-"),
        ("CPU FPS", lambda r: f"{r.cpu_fps:.1f}" if r.cpu_fps else "-"),
        ("Parameters", lambda r: f"{r.parameters:,}"),
        ("FLOPs", lambda r: f"{r.flops / 1e9:.2f} G"),
        ("Model Size", lambda r: f"{r.size_mb:.1f} MB"),
    ]
    references = reference_columns or {}
    header = ["Metric"] + [r.name for r in reports] + list(references)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for label, extract in rows:
        cells = [label] + [extract(r) for r in reports]
        cells += [str(column.get(label, "-")) for column in references.values()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
