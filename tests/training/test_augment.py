"""Tests for the detection augmentation ops and their dataset wiring."""

import random

import pytest

torch = pytest.importorskip("torch")

from lofop.training.augment import color_jitter, horizontal_flip, mosaic  # noqa: E402


def _item(size=64, num_boxes=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(3, size, size, generator=generator)
    boxes = torch.tensor([[4.0 + i * 10, 6.0, 24.0 + i * 10, 30.0] for i in range(num_boxes)])
    labels = torch.arange(num_boxes, dtype=torch.long)
    return image, boxes, labels


class TestHorizontalFlip:
    def test_boxes_mirrored(self):
        image, boxes, _ = _item()
        flipped_image, flipped = horizontal_flip(image, boxes, width=64)
        assert torch.equal(flipped_image, torch.flip(image, dims=[2]))
        assert flipped[0, 0] == pytest.approx(64 - boxes[0, 2].item())
        assert flipped[0, 2] == pytest.approx(64 - boxes[0, 0].item())
        # widths preserved
        assert torch.allclose(flipped[:, 2] - flipped[:, 0], boxes[:, 2] - boxes[:, 0])

    def test_empty_boxes_ok(self):
        image, _, _ = _item()
        _, out = horizontal_flip(image, torch.zeros((0, 4)), width=64)
        assert out.shape == (0, 4)


class TestColorJitter:
    def test_range_and_shape(self):
        image, _, _ = _item()
        out = color_jitter(image)
        assert out.shape == image.shape
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_changes_pixels(self):
        random.seed(3)
        image, _, _ = _item()
        out = color_jitter(image)
        assert not torch.equal(out, image)

    def test_zero_strength_is_identity(self):
        image, _, _ = _item()
        out = color_jitter(image, brightness=0.0, contrast=0.0, saturation=0.0)
        assert torch.allclose(out, image, atol=1e-6)


class TestMosaic:
    def test_output_geometry(self):
        items = [_item(seed=s) for s in range(4)]
        image, boxes, labels = mosaic(items, image_size=64)
        assert image.shape == (3, 64, 64)
        assert len(boxes) == len(labels)
        if boxes.numel():
            assert float(boxes.min()) >= 0.0
            assert float(boxes.max()) <= 64.0

    def test_boxes_scaled_into_quadrants(self):
        items = [_item(seed=s, num_boxes=1) for s in range(4)]
        _, boxes, _ = mosaic(items, image_size=64)
        # one box per quadrant survives (source boxes are 20px, half-scale 10px)
        assert len(boxes) == 4
        quadrant_hits = {(bool(b[0] >= 32), bool(b[1] >= 32)) for b in boxes}
        assert len(quadrant_hits) == 4

    def test_requires_four(self):
        with pytest.raises(ValueError):
            mosaic([_item()], image_size=64)


class TestDatasetWiring:
    def _dataset(self, tmp_path, strong):
        from PIL import Image

        from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample
        from lofop.training import DetectionTorchDataset

        root = tmp_path / "imgs"
        root.mkdir()
        canonical = Dataset("aug", [Category(1, "thing")], image_root=root)
        for i in range(6):
            Image.new("RGB", (80, 60), (i * 30 % 255, 90, 40)).save(root / f"{i}.png")
            canonical.add_sample(Sample(
                image=f"{i}.png", width=80, height=60,
                annotations=[BoxAnnotation(bbox=(10.0, 10.0, 50.0, 40.0), category_id=1)],
            ))
        return DetectionTorchDataset(
            canonical, image_size=64, augment=True, strong_augment=strong,
        )

    def test_default_path_unchanged(self, tmp_path):
        dataset = self._dataset(tmp_path, strong=False)
        image, target = dataset[0]
        assert image.shape == (3, 64, 64)
        assert target["boxes"].shape == (1, 4)

    def test_strong_augment_yields_valid_items(self, tmp_path):
        random.seed(0)
        dataset = self._dataset(tmp_path, strong=True)
        for index in range(len(dataset)):
            image, target = dataset[index]
            assert image.shape == (3, 64, 64)
            assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
            boxes = target["boxes"]
            assert len(boxes) == len(target["labels"])
            if boxes.numel():
                assert float(boxes.min()) >= 0.0 and float(boxes.max()) <= 64.0
