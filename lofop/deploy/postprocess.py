"""Post-processing for dense exported-model outputs.

Torch-free companion to the ONNX graph: takes the dense ``(boxes, scores)``
arrays an exported LOFOP model produces, applies score thresholding and
LOFOP's native class-aware NMS, and returns final detections. Depends only
on the core framework (the C++ ops fast path applies automatically), so it
runs next to onnxruntime, TensorRT, or OpenVINO without a PyTorch install.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lofop.ops import batched_nms


@dataclass(frozen=True)
class Detections:
    """Final detections for one image.

    Attributes:
        boxes: (K, 4) xyxy boxes in input pixels.
        scores: K confidence scores, descending.
        labels: K class indices.
    """

    boxes: list[list[float]]
    scores: list[float]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.scores)


def postprocess_dense(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[Sequence[float]],
    *,
    score_threshold: float = 0.25,
    nms_iou: float = 0.6,
    max_detections: int = 300,
) -> Detections:
    """Turn one image's dense model outputs into final detections.

    Args:
        boxes: (N, 4) decoded xyxy boxes (the ONNX ``boxes`` output, one
            image; numpy arrays and nested lists both work).
        scores: (N, C) per-class scores (the ONNX ``scores`` output).
        score_threshold: Minimum class score to consider a candidate.
        nms_iou: IoU threshold for class-aware NMS.
        max_detections: Cap on returned detections.
    """
    candidate_boxes: list[list[float]] = []
    candidate_scores: list[float] = []
    candidate_labels: list[int] = []
    for box, class_scores in zip(boxes, scores):
        best_label, best_score = -1, score_threshold
        for label, score in enumerate(class_scores):
            if score > best_score:
                best_label, best_score = label, float(score)
        if best_label >= 0:
            candidate_boxes.append([float(v) for v in box])
            candidate_scores.append(best_score)
            candidate_labels.append(best_label)
    keep = batched_nms(
        candidate_boxes, candidate_scores, candidate_labels,
        iou_threshold=nms_iou, max_keep=max_detections,
    )
    return Detections(
        boxes=[candidate_boxes[i] for i in keep],
        scores=[candidate_scores[i] for i in keep],
        labels=[candidate_labels[i] for i in keep],
    )
