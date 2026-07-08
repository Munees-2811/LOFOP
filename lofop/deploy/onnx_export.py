"""ONNX export for LOFOP detectors.

The exported graph contains the full network plus box decoding -- everything
that is tensor-shaped and runtime-friendly. NMS deliberately stays OUTSIDE
the graph: runtimes disagree on NMS operator support and thresholds are a
deployment-time decision, so the graph outputs dense ``(boxes, scores)`` and
:func:`lofop.deploy.postprocess.postprocess_dense` finishes the job with
LOFOP's native C++ NMS (torch-free, works next to any ONNX runtime).

Export is fixed-resolution: pyramid decode points are baked in for one input
size, which is what edge runtimes (TensorRT, OpenVINO) want anyway. Export
multiple sizes if a deployment needs them.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from lofop.core.exceptions import LofopError
from lofop.core.logging import get_logger
from lofop.models.detector import LofopDetect

logger = get_logger(__name__)


class DenseExportWrapper(nn.Module):
    """Wraps a detector into an ONNX-friendly dense-output module.

    Outputs, for a (1, 3, S, S) input:
        boxes: (1, N, 4) decoded xyxy boxes in input pixels.
        scores: (1, N, C) quality-fused class scores in [0, 1].
    """

    def __init__(self, model: LofopDetect) -> None:
        super().__init__()
        self.model = model
        self.eval()

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        cls_out, box_out, quality_out = self.model(images)
        points, _ = self.model.head.level_points(cls_out)
        cls, box, quality = self.model._flatten(cls_out, box_out, quality_out)
        scores = (cls.sigmoid() * quality.sigmoid().unsqueeze(-1)).sqrt()
        batch = images.shape[0]
        decoded = torch.stack(
            [self.model.head.decode_boxes(points, box[i]) for i in range(batch)]
        )
        return decoded, scores


def export_onnx(
    model: LofopDetect,
    target: str | Path,
    *,
    image_size: int = 640,
    opset: int = 18,
    verify: bool = True,
    tolerance: float = 1e-4,
) -> Path:
    """Export a detector to ONNX and (optionally) verify it numerically.

    Args:
        model: Detector to export (weights already loaded).
        target: Output ``.onnx`` path; parents are created.
        image_size: Fixed input resolution baked into the graph.
        opset: ONNX opset version.
        verify: Run the exported graph under onnxruntime and compare against
            the torch outputs.
        tolerance: Maximum allowed absolute output difference during
            verification.

    Returns:
        The written path.

    Raises:
        LofopError: If export fails or verification exceeds tolerance.
    """
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DenseExportWrapper(model).cpu()
    example = torch.randn(1, 3, image_size, image_size)
    try:
        torch.onnx.export(
            wrapper, (example,), str(path),
            input_names=["images"], output_names=["boxes", "scores"],
            opset_version=opset, dynamo=False,
        )
    except Exception as exc:
        raise LofopError(
            f"ONNX export failed: {exc}", context={"target": str(path)}
        ) from exc
    logger.info("Exported ONNX model to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    if verify:
        max_diff = verify_onnx(wrapper, path, example, tolerance=tolerance)
        logger.info("ONNX verification passed (max diff %.2e)", max_diff)
    return path


def verify_onnx(
    wrapper: DenseExportWrapper,
    path: Path,
    example: Tensor,
    *,
    tolerance: float,
) -> float:
    """Compare onnxruntime outputs against torch outputs on ``example``.

    Returns:
        The maximum absolute difference observed.

    Raises:
        LofopError: If onnxruntime is unavailable or outputs diverge.
    """
    try:
        import onnxruntime
    except ImportError as exc:
        raise LofopError(
            "onnxruntime is required for verification; install it or pass verify=False"
        ) from exc
    # The exporter restores the module's pre-export training mode, so pin
    # eval here or BatchNorm would verify against batch statistics.
    wrapper.eval()
    with torch.inference_mode():
        torch_boxes, torch_scores = wrapper(example)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_boxes, ort_scores = session.run(None, {"images": example.numpy()})
    diffs = [
        float((torch_boxes - torch.from_numpy(ort_boxes)).abs().max()),
        float((torch_scores - torch.from_numpy(ort_scores)).abs().max()),
    ]
    max_diff = max(diffs)
    if max_diff > tolerance:
        raise LofopError(
            "ONNX outputs diverge from torch outputs",
            context={"max_diff": max_diff, "tolerance": tolerance},
        )
    return max_diff
