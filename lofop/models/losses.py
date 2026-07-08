"""Loss functions for LOFOP-Detect.

Functional implementations (stateless, easily unit-tested); the detector owns
the weighting and normalization. Rationale for each choice:
docs/lofop-detect.md section 2.5.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> Tensor:
    """Per-element sigmoid focal loss (no reduction).

    Args:
        logits: Raw predictions, any shape.
        targets: Same shape as ``logits``; values in ``[0, 1]`` (soft targets
            allowed).
        gamma: Down-weighting exponent for easy examples.
        alpha: Positive-class balance factor.
    """
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t) ** gamma * ce


def pairwise_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """IoU between two box sets, xyxy. Shapes (N,4) x (M,4) -> (N,M)."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * \
             (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * \
             (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-7)


def giou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Per-pair generalized IoU loss (no reduction). Shapes (N,4) x (N,4)."""
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * \
             (target[:, 3] - target[:, 1]).clamp(min=0)
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = (area_p + area_t - inter).clamp(min=1e-7)
    iou = inter / union

    hull_lt = torch.min(pred[:, :2], target[:, :2])
    hull_rb = torch.max(pred[:, 2:], target[:, 2:])
    hull_wh = (hull_rb - hull_lt).clamp(min=0)
    hull = (hull_wh[:, 0] * hull_wh[:, 1]).clamp(min=1e-7)
    giou = iou - (hull - union) / hull
    return 1.0 - giou
