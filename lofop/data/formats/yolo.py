"""YOLO (darknet-style) format adapter.

Layout under a dataset root directory::

    root/
        classes.txt        one class name per line; line number = class index
        images/            image files
        labels/            one .txt per image: "cls cx cy w h" normalized rows

YOLO labels are normalized center-format with contiguous class indices and no
image sizes, so this adapter reads sizes from the image files on load (via
Pillow) and maps category ids to contiguous indices on save. Category ids
become 0..N-1 after a YOLO round-trip -- that is inherent to the format, not a
bug in the adapter.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from lofop.core.exceptions import DataError
from lofop.core.logging import get_logger
from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample
from lofop.data.formats.base import FORMATS, DatasetAdapter

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

logger = get_logger(__name__)


@FORMATS.register(name="yolo")
class YoloAdapter(DatasetAdapter):
    """Adapter for darknet/YOLO-style dataset roots.

    ``source``/``target`` are the dataset root directory described in the
    module docstring.
    """

    def load(self, source: str | Path, *, name: str | None = None) -> Dataset:
        root = Path(source)
        classes_file = root / "classes.txt"
        images_dir = root / "images"
        labels_dir = root / "labels"
        if not classes_file.is_file():
            raise DataError("YOLO root has no classes.txt", context={"source": str(root)})
        if not images_dir.is_dir():
            raise DataError("YOLO root has no images/ directory", context={"source": str(root)})
        class_names = [
            line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        categories = [Category(id=i, name=n) for i, n in enumerate(class_names)]
        dataset = Dataset(name=name or root.name, categories=categories, image_root=images_dir)

        image_files = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        for image_file in image_files:
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except (OSError, UnidentifiedImageError) as exc:
                raise DataError(
                    f"Cannot read image: {exc}", context={"image": str(image_file)}
                ) from exc
            sample = Sample(image=image_file.name, width=width, height=height)
            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.is_file():
                sample.annotations = self._parse_label_file(
                    label_file, width, height, len(class_names)
                )
            dataset.add_sample(sample)
        return dataset

    def _parse_label_file(
        self, path: Path, width: int, height: int, num_classes: int
    ) -> list[BoxAnnotation]:
        annotations = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise DataError(
                    "YOLO label rows must have 5 fields: cls cx cy w h",
                    context={"label_file": str(path), "line": lineno},
                )
            try:
                cls = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError as exc:
                raise DataError(
                    f"Malformed YOLO label row: {exc}",
                    context={"label_file": str(path), "line": lineno},
                ) from exc
            if not 0 <= cls < num_classes:
                raise DataError(
                    "YOLO class index out of range",
                    context={"label_file": str(path), "line": lineno, "class": cls},
                )
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height
            annotations.append(BoxAnnotation(bbox=(x1, y1, x2, y2), category_id=cls))
        return annotations

    def save(self, dataset: Dataset, target: str | Path) -> None:
        root = Path(target)
        labels_dir = root / "labels"
        images_dir = root / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        index = dataset.category_index()
        (root / "classes.txt").write_text(
            "".join(f"{c.name}\n" for c in dataset.categories), encoding="utf-8"
        )
        for sample in dataset.samples:
            if sample.width <= 0 or sample.height <= 0:
                raise DataError(
                    "YOLO export needs positive image sizes to normalize boxes",
                    context={"dataset": dataset.name, "sample": sample.image},
                )
            rows = []
            for box in sample.annotations:
                x1, y1, x2, y2 = box.bbox
                cx = (x1 + x2) / 2 / sample.width
                cy = (y1 + y2) / 2 / sample.height
                w = (x2 - x1) / sample.width
                h = (y2 - y1) / sample.height
                rows.append(f"{index[box.category_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            stem = Path(sample.image).stem
            (labels_dir / f"{stem}.txt").write_text("".join(rows), encoding="utf-8")
            source_image = dataset.image_path(sample)
            link_target = images_dir / Path(sample.image).name
            if source_image.is_file() and not link_target.exists():
                link_target.write_bytes(source_image.read_bytes())
        logger.info("Wrote YOLO dataset %r to %s", dataset.name, root)
