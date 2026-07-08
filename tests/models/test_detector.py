"""End-to-end tests for LofopDetect: config build, losses, prediction."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lofop.core.config import Config  # noqa: E402
from lofop.models import ApexHead, DeltaFusion, LofopDetect, RidgeNet  # noqa: E402
from lofop.registries import HUB  # noqa: E402

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "lofop-detect" / "n.yaml"


def tiny_detector(num_classes=5):
    backbone = RidgeNet(widths=(16, 32, 64, 128), depths=(1, 1, 1, 1))
    neck = DeltaFusion(in_channels=backbone.out_channels, width=32)
    head = ApexHead(num_classes=num_classes, width=32, num_convs=1)
    return LofopDetect(backbone, neck, head, score_threshold=0.01)


def sample_targets(device="cpu"):
    return [
        {"boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]], device=device),
         "labels": torch.tensor([2], device=device)},
        {"boxes": torch.zeros((0, 4), device=device),
         "labels": torch.zeros((0,), dtype=torch.long, device=device)},
    ]


class TestLofopDetect:
    def test_builds_from_repo_config(self):
        cfg = Config.load(REPO_CONFIG)
        model = HUB.build(cfg.model)
        assert isinstance(model, LofopDetect)
        assert model.head.num_classes == cfg.num_classes
        cls_out, box_out, quality_out = model(torch.randn(1, 3, 64, 64))
        assert len(cls_out) == len(box_out) == len(quality_out) == 3

    def test_losses_finite_and_backward(self):
        model = tiny_detector()
        images = torch.randn(2, 3, 64, 64)
        losses = model.compute_losses(images, sample_targets())
        for key in ("cls", "box", "quality", "total"):
            assert torch.isfinite(losses[key]), key
        losses["total"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_empty_batch_supervises_background_quality(self):
        model = tiny_detector()
        images = torch.randn(1, 3, 64, 64)
        targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]
        losses = model.compute_losses(images, targets)
        assert losses["cls"].item() > 0
        assert losses["box"].item() == 0.0
        # Quality is calibrated toward 0 on background even with no objects.
        assert losses["quality"].item() > 0.0
        assert torch.isfinite(losses["total"])

    def test_quality_starts_low_on_background(self):
        # The quality bias init keeps untrained background quality small, so
        # the fused score sqrt(cls * quality) cannot inflate weak detections.
        model = tiny_detector()
        _, _, quality_out = model(torch.randn(1, 3, 64, 64))
        for level in quality_out:
            assert level.sigmoid().mean().item() < 0.2

    def test_quality_loss_shrinks_as_background_calibrates(self):
        # Push quality logits strongly negative (calibrated background): the
        # modulated quality loss must be far below the fresh-init loss.
        model = tiny_detector()
        images = torch.randn(1, 3, 64, 64)
        targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]
        fresh = model.compute_losses(images, targets)["quality"].item()
        with torch.no_grad():
            model.head.quality_pred.bias.fill_(-8.0)
        calibrated = model.compute_losses(images, targets)["quality"].item()
        assert calibrated < fresh * 0.05

    def test_predict_output_contract(self):
        model = tiny_detector().eval()
        results = model.predict(torch.randn(2, 3, 64, 64))
        assert len(results) == 2
        for result in results:
            boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
            assert boxes.shape[1] == 4 and boxes.shape[0] == scores.shape[0] == labels.shape[0]
            assert boxes.shape[0] <= model.max_detections
            if scores.numel() > 1:
                assert (scores[:-1] >= scores[1:]).all()  # sorted by score

    def test_optimize_for_inference_preserves_predictions(self):
        torch.manual_seed(1)
        model = tiny_detector().eval()
        images = torch.randn(2, 3, 64, 64)
        before = model.predict(images)
        returned = model.optimize_for_inference()
        assert returned is model and model._channels_last
        after = model.predict(images)
        assert len(before) == len(after)
        for b, a in zip(before, after):
            assert b["labels"].tolist() == a["labels"].tolist()
            assert torch.allclose(b["boxes"], a["boxes"], atol=1e-4)
            assert torch.allclose(b["scores"], a["scores"], atol=1e-5)

    def test_training_step_reduces_loss_on_fixed_batch(self):
        torch.manual_seed(0)
        model = tiny_detector(num_classes=3)
        images = torch.randn(1, 3, 64, 64)
        targets = [{"boxes": torch.tensor([[12.0, 12.0, 52.0, 52.0]]),
                    "labels": torch.tensor([1])}]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        first = model.compute_losses(images, targets)["total"].item()
        for _ in range(8):
            optimizer.zero_grad()
            loss = model.compute_losses(images, targets)["total"]
            loss.backward()
            optimizer.step()
        final = model.compute_losses(images, targets)["total"].item()
        assert final < first
