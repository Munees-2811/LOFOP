"""DynamicTopKAssigner: prediction-aware label assignment for dense heads.

For each ground-truth box, candidates are locations inside the box that also
lie within a center-prior radius. Candidates are ranked by a cost combining
the current classification score and IoU; each GT takes its top-k, with k
derived from the sum of its best IoUs (few good candidates -> small k).
Conflicted locations go to the lowest-cost GT. Runs under no_grad -- the
assignment shapes the loss but is not differentiated through.
Design rationale: docs/lofop-detect.md section 2.4.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lofop.models.losses import pairwise_iou


class DynamicTopKAssigner:
    """Assigns pyramid locations to ground-truth boxes.

    Args:
        center_radius: Center-prior radius in units of stride.
        max_k: Upper bound on positives per ground-truth box.
        cost_iou_weight: Weight of the (1 - IoU) term against the
            classification term in the candidate cost.
    """

    def __init__(
        self,
        center_radius: float = 2.5,
        max_k: int = 10,
        cost_iou_weight: float = 3.0,
    ) -> None:
        self.center_radius = center_radius
        self.max_k = max_k
        self.cost_iou_weight = cost_iou_weight

    @torch.no_grad()
    def assign(
        self,
        points: Tensor,
        strides: Tensor,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the assignment for one image.

        Args:
            points: (N, 2) location centers in image pixels.
            strides: (N,) stride of each location's level.
            pred_scores: (N, C) sigmoid classification scores.
            pred_boxes: (N, 4) currently decoded boxes, xyxy.
            gt_boxes: (M, 4) ground-truth boxes, xyxy.
            gt_labels: (M,) ground-truth class indices.

        Returns:
            ``(assigned_gt, iou_targets)``: (N,) index into the GT arrays or
            -1 for background, and (N,) IoU of each positive's predicted box
            with its GT (0 for background) to supervise the quality branch.
        """
        num_points = points.shape[0]
        num_gt = gt_boxes.shape[0]
        assigned = points.new_full((num_points,), -1, dtype=torch.long)
        iou_targets = points.new_zeros(num_points)
        if num_gt == 0:
            return assigned, iou_targets

        x, y = points[:, 0], points[:, 1]
        inside = (
            (x[:, None] >= gt_boxes[None, :, 0]) & (x[:, None] <= gt_boxes[None, :, 2])
            & (y[:, None] >= gt_boxes[None, :, 1]) & (y[:, None] <= gt_boxes[None, :, 3])
        )  # (N, M)
        centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # (M, 2)
        radius = self.center_radius * strides[:, None]  # (N, 1)
        near = (
            ((x[:, None] - centers[None, :, 0]).abs() <= radius)
            & ((y[:, None] - centers[None, :, 1]).abs() <= radius)
        )
        candidates = inside & near  # (N, M)

        ious = pairwise_iou(pred_boxes, gt_boxes)  # (N, M)
        cls_cost = -torch.log(pred_scores[:, gt_labels].clamp(min=1e-7))  # (N, M)
        cost = cls_cost + self.cost_iou_weight * (1.0 - ious)
        cost = torch.where(candidates, cost, torch.full_like(cost, float("inf")))

        best_cost = points.new_full((num_points,), float("inf"))
        for gt_index in range(num_gt):
            gt_cost = cost[:, gt_index]
            available = int(torch.isfinite(gt_cost).sum())
            if available == 0:
                continue
            top_ious = ious[:, gt_index][candidates[:, gt_index]]
            k = int(top_ious.topk(min(self.max_k, available)).values.sum().clamp(min=1.0))
            k = min(k, available)
            selected = gt_cost.topk(k, largest=False).indices
            for point_index in selected.tolist():
                if gt_cost[point_index] < best_cost[point_index]:
                    best_cost[point_index] = gt_cost[point_index]
                    assigned[point_index] = gt_index
                    iou_targets[point_index] = ious[point_index, gt_index]
        return assigned, iou_targets
