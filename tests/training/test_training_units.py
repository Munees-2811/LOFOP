"""Unit tests for the training subsystem pieces (no full training loop)."""

import pytest

torch = pytest.importorskip("torch")

from lofop.data.synthetic import generate_shapes_dataset  # noqa: E402
from lofop.training import (  # noqa: E402
    CheckpointManager,
    DetectionTorchDataset,
    ModelEMA,
    detection_collate,
    evaluate_detections,
    warmup_cosine_lr,
)


@pytest.fixture(scope="module")
def shapes(tmp_path_factory):
    return generate_shapes_dataset(
        tmp_path_factory.mktemp("shapes"), num_images=6, image_size=64, seed=1
    )


class TestTorchData:
    def test_item_shapes_and_scaling(self, shapes):
        ds = DetectionTorchDataset(shapes, image_size=32)
        image, target = ds[0]
        assert image.shape == (3, 32, 32)
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert target["boxes"].shape[1] == 4
        assert (target["boxes"] <= 32.0 + 1e-5).all()  # scaled from 64 to 32

    def test_flip_preserves_box_extents(self, shapes):
        ds = DetectionTorchDataset(shapes, image_size=64, augment=True, flip_probability=1.0)
        plain = DetectionTorchDataset(shapes, image_size=64)
        _, flipped = ds[0]
        _, original = plain[0]
        widths_flipped = flipped["boxes"][:, 2] - flipped["boxes"][:, 0]
        widths_original = original["boxes"][:, 2] - original["boxes"][:, 0]
        assert torch.allclose(widths_flipped, widths_original, atol=1e-4)
        assert torch.allclose(
            flipped["boxes"][:, 0], 64 - original["boxes"][:, 2], atol=1e-4
        )

    def test_collate(self, shapes):
        ds = DetectionTorchDataset(shapes, image_size=32)
        images, targets = detection_collate([ds[0], ds[1]])
        assert images.shape == (2, 3, 32, 32)
        assert len(targets) == 2 and "boxes" in targets[0]


class TestSchedule:
    def test_warmup_then_cosine(self):
        kwargs = {"warmup_steps": 10, "total_steps": 100}
        assert warmup_cosine_lr(0, **kwargs) == pytest.approx(0.1)
        assert warmup_cosine_lr(9, **kwargs) == pytest.approx(1.0)
        assert warmup_cosine_lr(10, **kwargs) == pytest.approx(1.0)
        assert warmup_cosine_lr(99, **kwargs) < 0.06  # decayed near the 5% floor
        mid = warmup_cosine_lr(55, **kwargs)
        assert 0.4 < mid < 0.6


class TestEMA:
    def test_tracks_toward_model(self):
        model = torch.nn.Linear(4, 4, bias=False)
        ema = ModelEMA(model, decay=0.5, warmup_updates=1)
        with torch.no_grad():
            model.weight.fill_(1.0)
        for _ in range(20):
            ema.update(model)
        distance = (ema.module.weight - 1.0).abs().max().item()
        assert distance < 0.05

    def test_state_roundtrip(self):
        model = torch.nn.Linear(2, 2)
        ema = ModelEMA(model)
        ema.update(model)
        state = ema.state_dict()
        restored = ModelEMA(torch.nn.Linear(2, 2))
        restored.load_state_dict(state)
        assert restored.updates == 1
        assert torch.equal(restored.module.weight, ema.module.weight)


class TestCheckpoints:
    def test_save_load_and_best_tracking(self, tmp_path):
        model = torch.nn.Linear(3, 3)
        manager = CheckpointManager(tmp_path / "ckpt")
        manager.save(model, epoch=0, metric=0.1)
        manager.save(model, epoch=1, metric=0.3)
        manager.save(model, epoch=2, metric=0.2)
        last = manager.load()
        assert last["epoch"] == 2
        best = manager.load(tmp_path / "ckpt" / "best.pt")
        assert best["epoch"] == 1 and best["metric"] == pytest.approx(0.3)

    def test_missing_checkpoint_raises(self, tmp_path):
        from lofop.core.exceptions import LofopError

        with pytest.raises(LofopError):
            CheckpointManager(tmp_path / "empty").load()


class TestEvaluator:
    def targets(self):
        return [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]]),
            "labels": torch.tensor([0, 1]),
        }]

    def test_perfect_predictions_score_one(self):
        predictions = [{
            "boxes": self.targets()[0]["boxes"].clone(),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 1]),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        assert metrics.map50 == pytest.approx(1.0, abs=0.01)
        assert metrics.map50_95 == pytest.approx(1.0, abs=0.01)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)

    def test_missing_object_halves_recall(self):
        predictions = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        assert metrics.recall == pytest.approx(0.5)
        assert metrics.precision == pytest.approx(1.0)

    def test_wrong_class_is_false_positive(self):
        predictions = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),   # actually class 0
        }]
        metrics = evaluate_detections(predictions, self.targets())
        assert metrics.recall == 0.0
        assert metrics.map50 == 0.0

    def test_slightly_shifted_box_passes_50_fails_95(self):
        predictions = [{
            "boxes": torch.tensor([[13.0, 13.0, 53.0, 53.0], [60.0, 60.0, 90.0, 90.0]]),
            "scores": torch.tensor([0.9, 0.9]),
            "labels": torch.tensor([0, 1]),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        assert metrics.map50 == pytest.approx(1.0, abs=0.01)
        assert metrics.map50_95 < metrics.map50

    def test_no_predictions(self):
        predictions = [{
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.long),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        assert metrics.map50 == 0.0 and metrics.recall == 0.0

    def test_f1_is_harmonic_mean(self):
        predictions = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        # precision 1.0, recall 0.5 -> F1 = 2/3.
        assert metrics.f1 == pytest.approx(2 / 3)

    def test_perfect_predictions_f1_one(self):
        predictions = [{
            "boxes": self.targets()[0]["boxes"].clone(),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 1]),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        assert metrics.f1 == pytest.approx(1.0)

    def test_per_class_precision_recall(self):
        predictions = [{
            "boxes": self.targets()[0]["boxes"].clone(),
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([0, 1]),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        assert metrics.per_class_precision[0] == pytest.approx(1.0)
        assert metrics.per_class_recall[1] == pytest.approx(1.0)

    def test_confusion_matrix_records_misclassification(self):
        # A box on the class-0 object is predicted as class 1: the confusion
        # matrix must show a class-0 GT predicted as class 1 (not FP + FN).
        predictions = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        classes = metrics.confusion_classes
        assert classes[-1] == -1  # background axis
        gt0, pred1 = classes.index(0), classes.index(1)
        assert metrics.confusion_matrix[gt0][pred1] == 1
        background = len(classes) - 1
        # The unmatched class-1 ground truth is a false negative.
        assert metrics.confusion_matrix[classes.index(1)][background] == 1

    def test_confusion_matrix_counts_false_positive(self):
        predictions = [{
            "boxes": torch.tensor([
                [10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0],
                [200.0, 200.0, 250.0, 250.0],
            ]),
            "scores": torch.tensor([0.9, 0.8, 0.7]),
            "labels": torch.tensor([0, 1, 0]),
        }]
        metrics = evaluate_detections(predictions, self.targets())
        classes = metrics.confusion_classes
        background = len(classes) - 1
        # The spurious class-0 box overlaps no ground truth: a false positive.
        assert metrics.confusion_matrix[background][classes.index(0)] == 1
