"""Detection evaluation: COCO-protocol mAP, precision, and recall.

Original implementation of the standard protocol: per class, detections are
matched greedily (score-descending) to unmatched ground truths at a given IoU
threshold; AP is the 101-point interpolated area under the precision/recall
curve; mAP@50:95 averages AP over IoU thresholds 0.50 to 0.95 in 0.05 steps.
Precision/recall are micro-averaged at IoU 0.5 over detections above a
confidence threshold. These are the numbers the LOFOP benchmark table
reports (mAP@50, mAP@50:95, Precision, Recall).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from lofop.models.losses import pairwise_iou

_IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]


@dataclass(frozen=True)
class DetectionMetrics:
    """Evaluation summary over a dataset.

    Attributes:
        map50: mAP at IoU 0.50.
        map50_95: mAP averaged over IoU 0.50:0.95.
        precision: Micro precision at IoU 0.5 above the confidence threshold.
        recall: Micro recall at IoU 0.5 above the confidence threshold.
        f1: Micro F1 (harmonic mean of ``precision`` and ``recall``).
        per_class_ap50: AP@50 for each class index that has ground truth.
        per_class_precision: Precision per class at IoU 0.5.
        per_class_recall: Recall per class at IoU 0.5.
        confusion_classes: Class indices labelling the confusion matrix axes;
            a trailing ``-1`` denotes the background (missed / spurious) row
            and column.
        confusion_matrix: ``(N+1) x (N+1)`` integer matrix where entry
            ``[i][j]`` counts ground truths of ``confusion_classes[i]``
            predicted as ``confusion_classes[j]``. The last row is false
            positives (predicted, no matching GT); the last column is false
            negatives (GT missed).
    """

    map50: float
    map50_95: float
    precision: float
    recall: float
    per_class_ap50: dict[int, float]
    f1: float = 0.0
    per_class_precision: dict[int, float] = field(default_factory=dict)
    per_class_recall: dict[int, float] = field(default_factory=dict)
    confusion_classes: list[int] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)


def evaluate_detections(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    *,
    confidence_threshold: float = 0.25,
) -> DetectionMetrics:
    """Compute detection metrics for one dataset pass.

    Args:
        predictions: Per image: ``{"boxes": (K,4), "scores": (K,),
            "labels": (K,)}``.
        targets: Per image: ``{"boxes": (M,4), "labels": (M,)}``.
        confidence_threshold: Score cut used for the precision/recall
            numbers (not for AP, which sweeps all scores).
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    classes = sorted(
        {int(c) for t in targets for c in t["labels"].tolist()}
    )
    ap_per_threshold: dict[float, dict[int, float]] = {t: {} for t in _IOU_THRESHOLDS}
    for class_index in classes:
        for threshold in _IOU_THRESHOLDS:
            ap_per_threshold[threshold][class_index] = _average_precision(
                predictions, targets, class_index, threshold
            )

    tp, fp, num_gt = _counts_at(predictions, targets, confidence_threshold, iou_threshold=0.5)
    ap50 = ap_per_threshold[0.5]
    map50 = sum(ap50.values()) / len(ap50) if ap50 else 0.0
    all_aps = [ap for by_class in ap_per_threshold.values() for ap in by_class.values()]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_class_precision, per_class_recall, confusion_classes, confusion_matrix = (
        _confusion_and_per_class(predictions, targets, confidence_threshold, iou_threshold=0.5)
    )
    return DetectionMetrics(
        map50=map50,
        map50_95=sum(all_aps) / len(all_aps) if all_aps else 0.0,
        precision=precision,
        recall=recall,
        per_class_ap50=ap50,
        f1=f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        confusion_classes=confusion_classes,
        confusion_matrix=confusion_matrix,
    )


def _match_image(
    pred_boxes: Tensor, gt_boxes: Tensor, iou_threshold: float
) -> list[bool]:
    """Greedy per-image matching; predictions must arrive score-sorted."""
    matched_gt: set[int] = set()
    hits = []
    ious = pairwise_iou(pred_boxes, gt_boxes) if len(gt_boxes) else None
    for pred_index in range(len(pred_boxes)):
        best_iou, best_gt = 0.0, -1
        if ious is not None:
            for gt_index in range(len(gt_boxes)):
                if gt_index in matched_gt:
                    continue
                iou = float(ious[pred_index, gt_index])
                if iou > best_iou:
                    best_iou, best_gt = iou, gt_index
        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            hits.append(True)
        else:
            hits.append(False)
    return hits


def _class_detections(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    class_index: int,
    iou_threshold: float,
) -> tuple[list[tuple[float, bool]], int]:
    """All (score, is_tp) pairs for one class, plus its ground-truth count."""
    scored: list[tuple[float, bool]] = []
    total_gt = 0
    for prediction, target in zip(predictions, targets):
        gt_mask = target["labels"] == class_index
        gt_boxes = target["boxes"][gt_mask]
        total_gt += int(gt_mask.sum())
        pred_mask = prediction["labels"] == class_index
        boxes = prediction["boxes"][pred_mask]
        scores = prediction["scores"][pred_mask]
        order = torch.argsort(scores, descending=True)
        boxes, scores = boxes[order], scores[order]
        hits = _match_image(boxes, gt_boxes, iou_threshold)
        scored.extend((float(s), hit) for s, hit in zip(scores.tolist(), hits))
    return scored, total_gt


def _average_precision(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    class_index: int,
    iou_threshold: float,
) -> float:
    scored, total_gt = _class_detections(predictions, targets, class_index, iou_threshold)
    if total_gt == 0:
        return 0.0
    if not scored:
        return 0.0
    scored.sort(key=lambda pair: -pair[0])
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for _, is_tp in scored:
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)
    # 101-point interpolation with monotone precision envelope.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap, point_index = 0.0, 0
    for recall_point in (i / 100.0 for i in range(101)):
        while point_index < len(recalls) and recalls[point_index] < recall_point:
            point_index += 1
        ap += precisions[point_index] if point_index < len(precisions) else 0.0
    return ap / 101.0


def _counts_at(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Micro TP/FP over confident detections, plus total ground truths."""
    tp = fp = num_gt = 0
    for prediction, target in zip(predictions, targets):
        num_gt += int(target["labels"].numel())
        keep = prediction["scores"] >= confidence_threshold
        boxes = prediction["boxes"][keep]
        scores = prediction["scores"][keep]
        labels = prediction["labels"][keep]
        order = torch.argsort(scores, descending=True)
        for class_index in labels.unique().tolist():
            class_mask = labels[order] == class_index
            class_boxes = boxes[order][class_mask]
            gt_boxes = target["boxes"][target["labels"] == class_index]
            hits = _match_image(class_boxes, gt_boxes, iou_threshold)
            tp += sum(hits)
            fp += len(hits) - sum(hits)
    return tp, fp, num_gt


def _confusion_and_per_class(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[dict[int, float], dict[int, float], list[int], list[list[int]]]:
    """Cross-class confusion matrix and per-class precision/recall.

    Confident detections are matched greedily (score-descending) to ground
    truths by IoU, ignoring class, so a detection that overlaps a ground truth
    of a different class is recorded as a misclassification rather than as an
    independent false positive plus false negative. Unmatched detections land
    in the background row; unmatched ground truths in the background column.
    """
    classes = sorted(
        {int(c) for t in targets for c in t["labels"].tolist()}
        | {int(c) for p in predictions for c in p["labels"].tolist()}
    )
    index_of = {c: i for i, c in enumerate(classes)}
    background = len(classes)
    matrix = [[0] * (background + 1) for _ in range(background + 1)]
    for prediction, target in zip(predictions, targets):
        keep = prediction["scores"] >= confidence_threshold
        boxes = prediction["boxes"][keep]
        scores = prediction["scores"][keep]
        labels = prediction["labels"][keep]
        order = torch.argsort(scores, descending=True)
        boxes = boxes[order]
        labels = labels[order].tolist()
        gt_boxes = target["boxes"]
        gt_labels = target["labels"].tolist()
        ious = pairwise_iou(boxes, gt_boxes) if len(gt_boxes) and len(boxes) else None
        matched_gt: set[int] = set()
        for pred_index in range(len(boxes)):
            best_iou, best_gt = iou_threshold, -1
            if ious is not None:
                for gt_index in range(len(gt_labels)):
                    if gt_index in matched_gt:
                        continue
                    iou = float(ious[pred_index, gt_index])
                    if iou >= best_iou:
                        best_iou, best_gt = iou, gt_index
            pred_class = index_of[int(labels[pred_index])]
            if best_gt >= 0:
                matched_gt.add(best_gt)
                matrix[index_of[int(gt_labels[best_gt])]][pred_class] += 1
            else:
                matrix[background][pred_class] += 1
        for gt_index in range(len(gt_labels)):
            if gt_index not in matched_gt:
                matrix[index_of[int(gt_labels[gt_index])]][background] += 1
    per_class_precision: dict[int, float] = {}
    per_class_recall: dict[int, float] = {}
    for class_index, i in index_of.items():
        hits = matrix[i][i]
        predicted = sum(matrix[row][i] for row in range(background + 1))
        actual = sum(matrix[i][col] for col in range(background + 1))
        per_class_precision[class_index] = hits / predicted if predicted else 0.0
        per_class_recall[class_index] = hits / actual if actual else 0.0
    return per_class_precision, per_class_recall, classes + [-1], matrix
