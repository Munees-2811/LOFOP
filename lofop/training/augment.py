"""Original detection augmentations on (image tensor, boxes, labels) triples.

All ops work on CHW float tensors in [0, 1] with xyxy pixel boxes -- the same
representation :class:`~lofop.training.torch_data.DetectionTorchDataset`
produces -- and are implemented with plain tensor math (no numpy, no external
augmentation library). They are sampling-free building blocks; the dataset
decides when and with what probability to apply them, so eval paths and
existing flip-only training remain byte-for-byte unchanged.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import Tensor

_MIN_BOX_SIDE = 2.0  # boxes thinner than this after an op carry no signal


def horizontal_flip(image: Tensor, boxes: Tensor, width: int) -> tuple[Tensor, Tensor]:
    """Mirror the image and boxes around the vertical axis."""
    image = torch.flip(image, dims=[2])
    if boxes.numel():
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        boxes = flipped
    return image, boxes


def color_jitter(
    image: Tensor,
    *,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.4,
) -> Tensor:
    """Randomly perturb brightness, contrast, and saturation.

    Each strength ``s`` draws a factor uniformly from ``[1 - s, 1 + s]``.
    Brightness scales the pixels, contrast interpolates toward the image mean,
    and saturation interpolates toward the per-pixel gray value. Output stays
    clamped to [0, 1]. Geometry is untouched, so boxes need no update.
    """
    def factor(strength: float) -> float:
        return 1.0 + random.uniform(-strength, strength)

    image = image * factor(brightness)
    mean = image.mean()
    image = mean + (image - mean) * factor(contrast)
    gray = image.mean(dim=0, keepdim=True).expand_as(image)
    image = gray + (image - gray) * factor(saturation)
    return image.clamp(0.0, 1.0)


def _filter_valid(boxes: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    if not boxes.numel():
        return boxes, labels
    keep = (boxes[:, 2] - boxes[:, 0] >= _MIN_BOX_SIDE) & (
        boxes[:, 3] - boxes[:, 1] >= _MIN_BOX_SIDE
    )
    return boxes[keep], labels[keep]


def mosaic(
    items: list[tuple[Tensor, Tensor, Tensor]], image_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Combine four (image, boxes, labels) items into one 2x2 mosaic.

    Each source is resized to a quadrant of the output, its boxes scaled and
    offset into place, and degenerate leftovers dropped. Mosaic multiplies the
    objects seen per optimization step and exposes them at half scale, which is
    what makes it effective for small-object robustness on small datasets.
    """
    if len(items) != 4:
        raise ValueError("mosaic requires exactly 4 items")
    half = image_size // 2
    canvas = torch.zeros(3, half * 2, half * 2)
    out_boxes: list[Tensor] = []
    out_labels: list[Tensor] = []
    offsets = [(0, 0), (half, 0), (0, half), (half, half)]
    for (image, boxes, labels), (ox, oy) in zip(items, offsets):
        size = image.shape[-1]
        tile = F.interpolate(
            image.unsqueeze(0), size=(half, half), mode="bilinear", align_corners=False
        )[0]
        canvas[:, oy:oy + half, ox:ox + half] = tile
        if boxes.numel():
            scale = half / size
            moved = boxes * scale
            moved[:, [0, 2]] += ox
            moved[:, [1, 3]] += oy
            out_boxes.append(moved)
            out_labels.append(labels)
    if out_boxes:
        boxes = torch.cat(out_boxes)
        labels = torch.cat(out_labels)
    else:
        boxes = torch.zeros((0, 4))
        labels = torch.zeros((0,), dtype=torch.long)
    if canvas.shape[-1] != image_size:  # odd sizes: 2 * (S // 2) != S
        canvas = F.interpolate(
            canvas.unsqueeze(0), size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        )[0]
        rescale = image_size / (half * 2)
        boxes = boxes * rescale
    boxes, labels = _filter_valid(boxes, labels)
    return canvas, boxes, labels
