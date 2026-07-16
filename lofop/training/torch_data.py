"""Bridge from the canonical LOFOP dataset model to PyTorch training data.

`DetectionTorchDataset` wraps a :class:`lofop.data.Dataset`: it loads images
with Pillow, resizes to a square training resolution (aspect distortion is
accepted in v1 -- letterboxing is a planned refinement), scales boxes to
match, and optionally applies horizontal-flip augmentation. Tensors only;
no numpy dependency.
"""

from __future__ import annotations

import random

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from lofop.core.exceptions import DataError
from lofop.data.dataset import Dataset, Sample
from lofop.training.augment import color_jitter, horizontal_flip, mosaic


def image_to_tensor(image: Image.Image) -> Tensor:
    """Convert an RGB PIL image to a float CHW tensor in [0, 1]."""
    rgb = image.convert("RGB")
    data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    return data.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1).float() / 255.0


class DetectionTorchDataset(TorchDataset):
    """Torch-facing view of a canonical detection dataset.

    Args:
        dataset: Canonical dataset (its ``image_root`` must resolve files).
        image_size: Square side length images are resized to.
        augment: Apply random horizontal flip (training) or nothing (eval).
        flip_probability: Chance of the horizontal flip when augmenting.
        strong_augment: Additionally apply mosaic (probability
            ``mosaic_probability``, combining 4 random samples) and color
            jitter -- the richer recipe for real-data training. Off by
            default so existing runs and benchmarks are unchanged.
        mosaic_probability: Chance of building a mosaic per fetched item
            when ``strong_augment`` is enabled.

    ``__getitem__`` returns ``(image, target)`` where ``image`` is (3, S, S)
    and ``target`` is ``{"boxes": (M, 4) xyxy in resized pixels, "labels":
    (M,) contiguous class indices}``.
    """

    def __init__(
        self,
        dataset: Dataset,
        image_size: int = 640,
        augment: bool = False,
        flip_probability: float = 0.5,
        strong_augment: bool = False,
        mosaic_probability: float = 0.5,
    ) -> None:
        self.dataset = dataset
        self.image_size = image_size
        self.augment = augment or strong_augment
        self.flip_probability = flip_probability
        self.strong_augment = strong_augment
        self.mosaic_probability = mosaic_probability
        self.class_index = dataset.category_index()

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image, boxes, labels = self._load_scaled(index)
        if self.strong_augment:
            if random.random() < self.mosaic_probability:
                others = [random.randrange(len(self)) for _ in range(3)]
                items = [(image, boxes, labels)]
                items += [self._load_scaled(i) for i in others]
                image, boxes, labels = mosaic(items, self.image_size)
            image = color_jitter(image)
        if self.augment and random.random() < self.flip_probability:
            image, boxes = horizontal_flip(image, boxes, self.image_size)
        return image, {"boxes": boxes, "labels": labels}

    def _load_scaled(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample = self.dataset.samples[index]
        path = self.dataset.image_path(sample)
        try:
            with Image.open(path) as img:
                image = image_to_tensor(
                    img.resize((self.image_size, self.image_size), Image.BILINEAR)
                )
        except OSError as exc:
            raise DataError(f"Cannot load image: {exc}", context={"path": str(path)}) from exc
        boxes, labels = self._scaled_targets(sample)
        return image, boxes, labels

    def _scaled_targets(self, sample: Sample) -> tuple[Tensor, Tensor]:
        if not sample.annotations:
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        scale_x = self.image_size / sample.width
        scale_y = self.image_size / sample.height
        boxes = torch.tensor(
            [[b.bbox[0] * scale_x, b.bbox[1] * scale_y, b.bbox[2] * scale_x, b.bbox[3] * scale_y]
             for b in sample.annotations],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [self.class_index[b.category_id] for b in sample.annotations], dtype=torch.long
        )
        return boxes, labels


def detection_collate(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Stack images; keep per-image target dicts (variable object counts)."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
