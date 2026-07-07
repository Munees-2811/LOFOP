"""Shared fixtures: a tiny but real dataset written in each supported format."""

import json

import pytest
from PIL import Image

from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample


def make_image(path, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(120, 30, 200)).save(path)


@pytest.fixture
def canonical_dataset(tmp_path):
    """Two images, two categories, three boxes, images on disk."""
    image_root = tmp_path / "imgs"
    make_image(image_root / "a.png", 100, 80)
    make_image(image_root / "b.png", 64, 64)
    dataset = Dataset(
        name="mini",
        categories=[Category(1, "cat"), Category(3, "dog")],
        image_root=image_root,
    )
    dataset.add_sample(Sample(
        image="a.png", width=100, height=80,
        annotations=[
            BoxAnnotation(bbox=(10.0, 10.0, 50.0, 40.0), category_id=1),
            BoxAnnotation(bbox=(0.0, 0.0, 100.0, 80.0), category_id=3,
                          attributes={"iscrowd": 1}),
        ],
    ))
    dataset.add_sample(Sample(
        image="b.png", width=64, height=64,
        annotations=[BoxAnnotation(bbox=(16.0, 16.0, 48.0, 48.0), category_id=1)],
    ))
    return dataset


@pytest.fixture
def coco_file(tmp_path, canonical_dataset):
    """The canonical dataset as a COCO JSON file (hand-written, not adapter output)."""
    payload = {
        "images": [
            {"id": 0, "file_name": "a.png", "width": 100, "height": 80},
            {"id": 1, "file_name": "b.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 40, 30], "iscrowd": 0},
            {"id": 2, "image_id": 0, "category_id": 3, "bbox": [0, 0, 100, 80], "iscrowd": 1},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [16, 16, 32, 32], "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 3, "name": "dog"}],
    }
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def yolo_root(tmp_path):
    """A hand-written YOLO dataset root with real images."""
    root = tmp_path / "yolo_src"
    make_image(root / "images" / "a.png", 100, 80)
    make_image(root / "images" / "b.png", 64, 64)
    (root / "classes.txt").write_text("cat\ndog\n")
    labels = root / "labels"
    labels.mkdir()
    # a.png: cat box (10,10)-(50,40); dog box covering the full image.
    (labels / "a.txt").write_text("0 0.3 0.3125 0.4 0.375\n1 0.5 0.5 1.0 1.0\n")
    (labels / "b.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    return root
