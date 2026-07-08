"""LofopDetect: the LOFOP flagship detector.

Assembles a registered backbone, neck, and head; computes training losses via
dynamic label assignment; decodes predictions through the native C++
class-aware NMS. Everything is config-driven -- see configs/lofop-detect/.
Design: docs/lofop-detect.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.assigner import DynamicTopKAssigner
from lofop.models.head import ApexHead
from lofop.models.losses import giou_loss, sigmoid_focal_loss
from lofop.ops import batched_nms
from lofop.registries import MODELS


@MODELS.register()
class LofopDetect(nn.Module):
    """Anchor-free multi-scale detector.

    Args:
        backbone: Feature extractor returning ``[C3, C4, C5]`` (typically a
            built ``backbone/RidgeNet`` spec).
        neck: Fusion module mapping those to ``[P3, P4, P5]``.
        head: Dense prediction head (an :class:`ApexHead`).
        box_weight: Weight of the GIoU loss term.
        score_threshold: Minimum score for a detection at inference.
        nms_iou: IoU threshold for class-aware NMS.
        max_detections: Detections kept per image after NMS.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: ApexHead,
        box_weight: float = 2.0,
        score_threshold: float = 0.05,
        nms_iou: float = 0.6,
        max_detections: int = 300,
    ) -> None:
        super().__init__()
        if not isinstance(head, ApexHead):
            got = type(head).__name__
            raise ModelError("LofopDetect requires an ApexHead", context={"got": got})
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.assigner = DynamicTopKAssigner()
        self.box_weight = box_weight
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Raw per-level head outputs for a batch of images."""
        return self.head(self.neck(self.backbone(images)))

    def _flatten(
        self, cls_out: list[Tensor], box_out: list[Tensor], quality_out: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Concatenate per-level maps into per-image rows: (B, N, ...)."""
        batch = cls_out[0].shape[0]
        cls = torch.cat(
            [level.permute(0, 2, 3, 1).reshape(batch, -1, self.head.num_classes)
             for level in cls_out], dim=1)
        box = torch.cat(
            [level.permute(0, 2, 3, 1).reshape(batch, -1, 4) for level in box_out], dim=1)
        quality = torch.cat(
            [level.permute(0, 2, 3, 1).reshape(batch, -1) for level in quality_out], dim=1)
        return cls, box, quality

    def compute_losses(self, images: Tensor, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Training losses for a batch.

        Args:
            images: (B, 3, H, W) input batch.
            targets: Per image: ``{"boxes": (M, 4) xyxy pixels, "labels": (M,)}``.

        Returns:
            ``{"cls": ..., "box": ..., "quality": ..., "total": ...}`` scalar
            tensors (box/quality are zero when the batch has no objects).
        """
        cls_out, box_out, quality_out = self.forward(images)
        points, strides = self.head.level_points(cls_out)
        cls, box, quality = self._flatten(cls_out, box_out, quality_out)

        total_cls = images.new_zeros(())
        total_box = images.new_zeros(())
        total_quality = images.new_zeros(())
        total_pos = 0
        for image_index, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            decoded = self.head.decode_boxes(points, box[image_index])
            assigned, iou_targets = self.assigner.assign(
                points, strides, cls[image_index].detach().sigmoid(), decoded.detach(),
                gt_boxes, gt_labels,
            )
            positive = assigned >= 0
            cls_targets = torch.zeros_like(cls[image_index])
            if positive.any():
                cls_targets[positive, gt_labels[assigned[positive]]] = 1.0
                total_box = total_box + giou_loss(
                    decoded[positive], gt_boxes[assigned[positive]]
                ).sum()
                total_quality = total_quality + F.binary_cross_entropy_with_logits(
                    quality[image_index][positive], iou_targets[positive], reduction="sum"
                )
                total_pos += int(positive.sum())
            total_cls = total_cls + sigmoid_focal_loss(cls[image_index], cls_targets).sum()

        norm = max(total_pos, 1)
        losses = {
            "cls": total_cls / norm,
            "box": self.box_weight * total_box / norm,
            "quality": total_quality / norm,
        }
        losses["total"] = losses["cls"] + losses["box"] + losses["quality"]
        return losses

    @torch.no_grad()
    def predict(self, images: Tensor) -> list[dict[str, Any]]:
        """Detect objects in a batch.

        Returns, per image: ``{"boxes": (K, 4) xyxy tensor, "scores": (K,),
        "labels": (K,)}`` sorted by score, at most ``max_detections`` each.
        """
        cls_out, box_out, quality_out = self.forward(images)
        points, _ = self.head.level_points(cls_out)
        cls, box, quality = self._flatten(cls_out, box_out, quality_out)
        scores_all = (cls.sigmoid() * quality.sigmoid().unsqueeze(-1)).sqrt()

        results = []
        for image_index in range(images.shape[0]):
            scores, labels = scores_all[image_index].max(dim=1)
            keep_mask = scores > self.score_threshold
            if not keep_mask.any():
                results.append(self._empty_result(images))
                continue
            boxes = self.head.decode_boxes(points[keep_mask], box[image_index][keep_mask])
            scores, labels = scores[keep_mask], labels[keep_mask]
            keep = batched_nms(
                boxes.tolist(), scores.tolist(), labels.tolist(),
                iou_threshold=self.nms_iou, max_keep=self.max_detections,
            )
            index = torch.as_tensor(keep, dtype=torch.long, device=images.device)
            results.append({
                "boxes": boxes[index], "scores": scores[index], "labels": labels[index],
            })
        return results

    @staticmethod
    def _empty_result(images: Tensor) -> dict[str, Any]:
        device = images.device
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "scores": torch.zeros((0,), device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
        }
