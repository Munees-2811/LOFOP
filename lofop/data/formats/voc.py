"""Pascal VOC format adapter.

Layout under a dataset root directory::

    root/
        Annotations/       one .xml per image (VOC schema)
        JPEGImages/        image files

VOC boxes are absolute ``xmin/ymin/xmax/ymax`` (1-based, converted to 0-based
on load and back on save) and categories are names only, so ids are assigned
in first-seen order on load and dropped on save. The ``difficult`` flag is
preserved through ``BoxAnnotation.attributes``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from lofop.core.exceptions import DataError
from lofop.core.logging import get_logger
from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample
from lofop.data.formats.base import FORMATS, DatasetAdapter

logger = get_logger(__name__)


@FORMATS.register(name="voc")
class VocAdapter(DatasetAdapter):
    """Adapter for Pascal VOC dataset roots (``Annotations/`` + ``JPEGImages/``)."""

    def load(self, source: str | Path, *, name: str | None = None) -> Dataset:
        root = Path(source)
        annotations_dir = root / "Annotations"
        if not annotations_dir.is_dir():
            raise DataError("VOC root has no Annotations/ directory", context={"source": str(root)})

        categories: dict[str, int] = {}
        parsed: list[Sample] = []
        for xml_file in sorted(annotations_dir.glob("*.xml")):
            parsed.append(self._parse_annotation(xml_file, categories))
        dataset = Dataset(
            name=name or root.name,
            categories=[Category(id=i, name=n) for n, i in categories.items()],
            samples=parsed,
            image_root=root / "JPEGImages",
        )
        return dataset

    def _parse_annotation(self, path: Path, categories: dict[str, int]) -> Sample:
        try:
            xml_root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise DataError(f"Cannot parse VOC XML: {exc}", context={"file": str(path)}) from exc

        def text(node: ET.Element | None, tag: str) -> str:
            child = node.find(tag) if node is not None else None
            if child is None or child.text is None:
                raise DataError(
                    f"VOC XML is missing <{tag}>", context={"file": str(path)}
                )
            return child.text.strip()

        size = xml_root.find("size")
        sample = Sample(
            image=text(xml_root, "filename"),
            width=int(text(size, "width")),
            height=int(text(size, "height")),
        )
        for obj in xml_root.findall("object"):
            class_name = text(obj, "name")
            if class_name not in categories:
                categories[class_name] = len(categories)
            box = obj.find("bndbox")
            x1 = float(text(box, "xmin")) - 1
            y1 = float(text(box, "ymin")) - 1
            x2 = float(text(box, "xmax")) - 1
            y2 = float(text(box, "ymax")) - 1
            attributes = {}
            difficult = obj.find("difficult")
            if difficult is not None and difficult.text and int(difficult.text):
                attributes["difficult"] = 1
            sample.annotations.append(
                BoxAnnotation(
                    bbox=(x1, y1, x2, y2),
                    category_id=categories[class_name],
                    attributes=attributes,
                )
            )
        return sample

    def save(self, dataset: Dataset, target: str | Path) -> None:
        root = Path(target)
        annotations_dir = root / "Annotations"
        images_dir = root / "JPEGImages"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        names = {c.id: c.name for c in dataset.categories}
        for sample in dataset.samples:
            xml_root = ET.Element("annotation")
            ET.SubElement(xml_root, "folder").text = "JPEGImages"
            ET.SubElement(xml_root, "filename").text = Path(sample.image).name
            size = ET.SubElement(xml_root, "size")
            ET.SubElement(size, "width").text = str(sample.width)
            ET.SubElement(size, "height").text = str(sample.height)
            ET.SubElement(size, "depth").text = "3"
            for box in sample.annotations:
                obj = ET.SubElement(xml_root, "object")
                ET.SubElement(obj, "name").text = names[box.category_id]
                ET.SubElement(obj, "difficult").text = str(box.attributes.get("difficult", 0))
                bndbox = ET.SubElement(obj, "bndbox")
                x1, y1, x2, y2 = box.bbox
                ET.SubElement(bndbox, "xmin").text = str(round(x1 + 1))
                ET.SubElement(bndbox, "ymin").text = str(round(y1 + 1))
                ET.SubElement(bndbox, "xmax").text = str(round(x2 + 1))
                ET.SubElement(bndbox, "ymax").text = str(round(y2 + 1))
            xml_path = annotations_dir / f"{Path(sample.image).stem}.xml"
            ET.ElementTree(xml_root).write(xml_path, encoding="utf-8")
            source_image = dataset.image_path(sample)
            link_target = images_dir / Path(sample.image).name
            if source_image.is_file() and not link_target.exists():
                link_target.write_bytes(source_image.read_bytes())
        logger.info("Wrote VOC dataset %r to %s", dataset.name, root)
