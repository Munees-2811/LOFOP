"""ApexHead: anchor-free, quality-aware dense detection head.

Shared classification and regression towers run on every pyramid level. Per
location the head predicts class logits, ltrb distances to the box edges
(softplus, scaled by a learnable per-level scale and the stride), and a
quality logit supervised with the IoU of the predicted box. Inference scores
are the geometric mean of classification and quality probabilities.
Design rationale: docs/lofop-detect.md section 2.3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.registries import HEADS


def _group_count(width: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if width % groups == 0:
            return groups
    return 1


def _tower(width: int, num_convs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(num_convs):
        layers += [
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        ]
    return nn.Sequential(*layers)


@HEADS.register()
class ApexHead(nn.Module):
    """Dense prediction head over a feature pyramid.

    Args:
        num_classes: Number of object categories.
        width: Channel width of the incoming pyramid maps.
        strides: Stride of each pyramid level, fine to coarse.
        num_convs: Depth of the shared classification/regression towers.

    Forward takes ``[P3, P4, P5]`` and returns three lists (one entry per
    level): class logits (B, C, H, W), box distances (B, 4, H, W) in pixels,
    and quality logits (B, 1, H, W).
    """

    def __init__(
        self,
        num_classes: int,
        width: int = 96,
        strides: tuple[int, ...] = (8, 16, 32),
        num_convs: int = 2,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ModelError("num_classes must be >= 1", context={"num_classes": num_classes})
        self.num_classes = num_classes
        self.strides = tuple(strides)
        self.cls_tower = _tower(width, num_convs)
        self.reg_tower = _tower(width, num_convs)
        self.cls_pred = nn.Conv2d(width, num_classes, 3, padding=1)
        self.box_pred = nn.Conv2d(width, 4, 3, padding=1)
        self.quality_pred = nn.Conv2d(width, 1, 3, padding=1)
        self.scales = nn.Parameter(torch.ones(len(self.strides)))
        # Focal-loss convention: bias classification logits so initial
        # foreground probability is ~1%, keeping early loss finite and stable.
        nn.init.constant_(self.cls_pred.bias, -4.595)

    def forward(self, features: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        if len(features) != len(self.strides):
            raise ModelError(
                "Feature count does not match configured strides",
                context={"features": len(features), "strides": list(self.strides)},
            )
        cls_out, box_out, quality_out = [], [], []
        for level, feature in enumerate(features):
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)
            cls_out.append(self.cls_pred(cls_feat))
            distances = F.softplus(self.box_pred(reg_feat)) * self.scales[level]
            box_out.append(distances * self.strides[level])
            quality_out.append(self.quality_pred(reg_feat))
        return cls_out, box_out, quality_out

    def level_points(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Location centers and per-location strides for these feature maps.

        Returns:
            ``(points, strides)``: (N, 2) pixel coordinates of every location
            across all levels (fine to coarse), and (N,) matching strides.
        """
        all_points, all_strides = [], []
        for stride, feature in zip(self.strides, features):
            height, width = feature.shape[-2:]
            device = feature.device
            ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
            xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
            all_points.append(points)
            all_strides.append(torch.full((points.shape[0],), float(stride), device=device))
        return torch.cat(all_points), torch.cat(all_strides)

    @staticmethod
    def decode_boxes(points: Tensor, distances: Tensor) -> Tensor:
        """Convert ltrb distances at ``points`` into xyxy boxes.

        Args:
            points: (N, 2) location centers.
            distances: (N, 4) left/top/right/bottom distances in pixels.
        """
        x, y = points[:, 0], points[:, 1]
        return torch.stack(
            [x - distances[:, 0], y - distances[:, 1],
             x + distances[:, 2], y + distances[:, 3]],
            dim=1,
        )
