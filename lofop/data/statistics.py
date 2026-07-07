"""Dataset statistics.

Answers the questions a practitioner asks before training: how many images
and boxes, how are classes balanced, how big are objects (COCO's
small/medium/large convention), and what image resolutions are present.
Output is a plain dataclass that renders to markdown for reports and to a
dict for programmatic use.
"""

from __future__ import annotations

import statistics as stats
from collections import Counter
from dataclasses import dataclass
from typing import Any

from lofop.data.dataset import Dataset

# COCO object-size convention (areas in squared pixels).
_SMALL_MAX_AREA = 32.0**2
_MEDIUM_MAX_AREA = 96.0**2


@dataclass(frozen=True)
class DatasetStats:
    """Summary statistics for one dataset.

    Attributes:
        name: Dataset name.
        num_images: Total samples.
        num_annotations: Total boxes.
        num_categories: Total categories.
        per_category: Category name -> box count (all categories, incl. 0).
        boxes_per_image_mean: Mean boxes per image.
        images_without_annotations: Samples with zero boxes.
        box_area_mean: Mean box area in squared pixels (0 when no boxes).
        size_breakdown: COCO-style counts: small (<32^2), medium, large.
        image_sizes: (width, height) -> image count.
    """

    name: str
    num_images: int
    num_annotations: int
    num_categories: int
    per_category: dict[str, int]
    boxes_per_image_mean: float
    images_without_annotations: int
    box_area_mean: float
    size_breakdown: dict[str, int]
    image_sizes: dict[tuple[int, int], int]

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (image size keys stringified for JSON use)."""
        result = self.__dict__.copy()
        result["image_sizes"] = {f"{w}x{h}": n for (w, h), n in self.image_sizes.items()}
        return result

    def to_markdown(self) -> str:
        """Render a compact markdown report."""
        lines = [
            f"# Dataset statistics: {self.name}",
            "",
            f"- Images: {self.num_images} ({self.images_without_annotations} without annotations)",
            f"- Annotations: {self.num_annotations} "
            f"(mean {self.boxes_per_image_mean:.2f} per image)",
            f"- Categories: {self.num_categories}",
            f"- Mean box area: {self.box_area_mean:.1f} px^2",
            f"- Object sizes (COCO convention): "
            f"small {self.size_breakdown['small']}, medium {self.size_breakdown['medium']}, "
            f"large {self.size_breakdown['large']}",
            "",
            "| Category | Boxes |",
            "|---|---:|",
        ]
        for cat_name, count in sorted(self.per_category.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {cat_name} | {count} |")
        lines += ["", "| Image size | Count |", "|---|---:|"]
        for (w, h), count in sorted(self.image_sizes.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {w}x{h} | {count} |")
        return "\n".join(lines) + "\n"


def compute_stats(dataset: Dataset) -> DatasetStats:
    """Compute :class:`DatasetStats` for a canonical dataset."""
    names = {c.id: c.name for c in dataset.categories}
    per_category: Counter[str] = Counter({name: 0 for name in names.values()})
    areas: list[float] = []
    sizes: Counter[tuple[int, int]] = Counter()
    breakdown = {"small": 0, "medium": 0, "large": 0}
    empty_images = 0

    for sample in dataset.samples:
        sizes[(sample.width, sample.height)] += 1
        if not sample.annotations:
            empty_images += 1
        for box in sample.annotations:
            per_category[names.get(box.category_id, f"<unknown:{box.category_id}>")] += 1
            areas.append(box.area)
            if box.area < _SMALL_MAX_AREA:
                breakdown["small"] += 1
            elif box.area < _MEDIUM_MAX_AREA:
                breakdown["medium"] += 1
            else:
                breakdown["large"] += 1

    num_images = len(dataset.samples)
    return DatasetStats(
        name=dataset.name,
        num_images=num_images,
        num_annotations=len(areas),
        num_categories=len(dataset.categories),
        per_category=dict(per_category),
        boxes_per_image_mean=len(areas) / num_images if num_images else 0.0,
        images_without_annotations=empty_images,
        box_area_mean=stats.fmean(areas) if areas else 0.0,
        size_breakdown=breakdown,
        image_sizes=dict(sizes),
    )
