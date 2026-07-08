"""Synthetic detection dataset generator.

Renders images containing brightly colored rectangles ("box" class) and
ellipses ("disc" class) on dark noisy backgrounds, with exact bounding boxes.
Useful for smoke-training detectors in CI, demos, and verifying that a
training pipeline can actually learn (a detector that cannot fit this data
is broken; one that can is at least wired correctly).
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample

_CLASSES = ("box", "disc")
_COLORS = [(232, 72, 61), (61, 133, 232), (72, 201, 91), (240, 180, 40)]


def generate_shapes_dataset(
    root: str | Path,
    *,
    num_images: int = 64,
    image_size: int = 128,
    max_objects: int = 3,
    seed: int = 0,
    name: str = "shapes",
) -> Dataset:
    """Write a shapes dataset to ``root`` and return its canonical form.

    Args:
        root: Output directory for the PNG images.
        num_images: Number of images to render.
        image_size: Square image side in pixels.
        max_objects: Maximum shapes per image (minimum is 1).
        seed: RNG seed; same seed reproduces the same dataset.
        name: Dataset name.
    """
    rng = random.Random(seed)
    image_root = Path(root)
    image_root.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(
        name=name,
        categories=[Category(i, cls) for i, cls in enumerate(_CLASSES)],
        image_root=image_root,
    )
    for index in range(num_images):
        image = Image.new("RGB", (image_size, image_size), (24, 24, 28))
        draw = ImageDraw.Draw(image)
        for _ in range(image_size // 8):  # background clutter that is not an object
            x, y = rng.randrange(image_size), rng.randrange(image_size)
            draw.point((x, y), fill=(rng.randrange(60), rng.randrange(60), rng.randrange(60)))
        sample = Sample(image=f"{index:05d}.png", width=image_size, height=image_size)
        for _ in range(rng.randint(1, max_objects)):
            side_w = rng.randint(image_size // 8, image_size // 3)
            side_h = rng.randint(image_size // 8, image_size // 3)
            x1 = rng.randint(0, image_size - side_w - 1)
            y1 = rng.randint(0, image_size - side_h - 1)
            x2, y2 = x1 + side_w, y1 + side_h
            class_id = rng.randrange(len(_CLASSES))
            color = rng.choice(_COLORS)
            if class_id == 0:
                draw.rectangle((x1, y1, x2, y2), fill=color)
            else:
                draw.ellipse((x1, y1, x2, y2), fill=color)
            sample.annotations.append(
                BoxAnnotation(bbox=(float(x1), float(y1), float(x2), float(y2)),
                              category_id=class_id)
            )
        image.save(image_root / sample.image)
        dataset.add_sample(sample)
    return dataset
