"""Tests for dynamic label assignment and loss functions."""

import pytest

torch = pytest.importorskip("torch")

from lofop.models import (  # noqa: E402
    DynamicTopKAssigner,
    giou_loss,
    pairwise_iou,
    sigmoid_focal_loss,
)


def grid_points(size=16, stride=8.0):
    coords = (torch.arange(size, dtype=torch.float32) + 0.5) * stride
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    return points, torch.full((points.shape[0],), stride)


class TestAssigner:
    def setup_method(self):
        self.assigner = DynamicTopKAssigner()
        self.points, self.strides = grid_points()
        n = self.points.shape[0]
        self.scores = torch.full((n, 3), 0.3)
        # Predictions: a fixed box around each point, so IoU varies with GT.
        self.boxes = torch.cat([self.points - 12.0, self.points + 12.0], dim=1)

    def test_no_gt_all_background(self):
        assigned, iou_targets = self.assigner.assign(
            self.points, self.strides, self.scores, self.boxes,
            torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long),
        )
        assert (assigned == -1).all() and (iou_targets == 0).all()

    def test_positives_are_inside_gt_and_near_center(self):
        gt = torch.tensor([[30.0, 30.0, 90.0, 90.0]])
        assigned, iou_targets = self.assigner.assign(
            self.points, self.strides, self.scores, self.boxes,
            gt, torch.tensor([1]),
        )
        positive = assigned >= 0
        assert positive.any()
        px, py = self.points[positive, 0], self.points[positive, 1]
        assert (px >= 30).all() and (px <= 90).all()
        assert (py >= 30).all() and (py <= 90).all()
        assert (iou_targets[positive] > 0).all()

    def test_conflicts_resolve_to_single_gt(self):
        # Two overlapping GTs: no point may be assigned twice (by construction
        # of the output format), and both GTs should get at least one point.
        gts = torch.tensor([[20.0, 20.0, 70.0, 70.0], [50.0, 50.0, 110.0, 110.0]])
        assigned, _ = self.assigner.assign(
            self.points, self.strides, self.scores, self.boxes,
            gts, torch.tensor([0, 2]),
        )
        assert set(assigned[assigned >= 0].unique().tolist()) == {0, 1}

    def test_tiny_gt_still_gets_a_positive(self):
        gt = torch.tensor([[60.0, 60.0, 68.0, 68.0]])  # 8x8 object
        assigned, _ = self.assigner.assign(
            self.points, self.strides, self.scores, self.boxes,
            gt, torch.tensor([0]),
        )
        assert (assigned >= 0).sum() >= 1


class TestLosses:
    def test_focal_loss_orders_hard_above_easy(self):
        target = torch.tensor([1.0])
        easy = sigmoid_focal_loss(torch.tensor([4.0]), target)
        hard = sigmoid_focal_loss(torch.tensor([-4.0]), target)
        assert hard.item() > easy.item() > 0

    def test_focal_loss_alpha_balances_classes(self):
        logits = torch.tensor([0.0])
        pos = sigmoid_focal_loss(logits, torch.tensor([1.0]), alpha=0.25)
        neg = sigmoid_focal_loss(logits, torch.tensor([0.0]), alpha=0.25)
        assert pos.item() == pytest.approx(neg.item() / 3.0, rel=1e-4)

    def test_giou_perfect_box_zero_loss(self):
        box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        assert giou_loss(box, box).item() == pytest.approx(0.0, abs=1e-6)

    def test_giou_disjoint_boxes_have_gradient_signal(self):
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
        target = torch.tensor([[100.0, 100.0, 120.0, 120.0]])
        loss = giou_loss(pred, target)
        assert loss.item() > 1.0  # GIoU < 0 for distant boxes
        loss.backward()
        assert pred.grad is not None and pred.grad.abs().sum() > 0

    def test_pairwise_iou_known_values(self):
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        iou = pairwise_iou(a, b)
        assert iou[0, 0].item() == pytest.approx(1.0)
        assert iou[0, 1].item() == pytest.approx(25 / 175, rel=1e-4)
