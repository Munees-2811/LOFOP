"""Dataset validator.

Catches the dataset defects that otherwise surface hours into training as NaN
losses or silent accuracy drops: degenerate and out-of-bounds boxes, dangling
category references, missing or unreadable image files, duplicate image
entries, and empty datasets. Produces a structured
:class:`ValidationReport` -- errors mean the dataset should not be trained on
as-is; warnings are suspicious but trainable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from lofop.core.logging import get_logger
from lofop.data.dataset import Dataset, Sample

logger = get_logger(__name__)

# Boxes may legitimately exceed image bounds by a pixel or two after
# augmentation or rounding in upstream tools; beyond this it is an error.
_BOUNDS_TOLERANCE = 2.0


class Severity(str, Enum):
    """How bad a finding is: ERROR blocks training, WARNING is advisory."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One validation finding.

    Attributes:
        severity: :class:`Severity` of the finding.
        message: Human-readable description.
        sample_id: Offending sample id, or ``None`` for dataset-level issues.
    """

    severity: Severity
    message: str
    sample_id: int | None = None

    def __str__(self) -> str:
        location = f"sample {self.sample_id}: " if self.sample_id is not None else ""
        return f"[{self.severity.value}] {location}{self.message}"


@dataclass
class ValidationReport:
    """All findings for one dataset."""

    dataset_name: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True when the dataset has no errors (warnings allowed)."""
        return not self.errors

    def summary(self) -> str:
        """One-line human-readable outcome."""
        status = "OK" if self.ok else "FAILED"
        return (
            f"{self.dataset_name}: {status} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        )


def validate_dataset(dataset: Dataset, *, check_images: bool = True) -> ValidationReport:
    """Validate a canonical dataset and return a structured report.

    Args:
        dataset: Dataset to check.
        check_images: Also verify that each sample's image file exists (only
            meaningful when the dataset knows its ``image_root`` or stores
            absolute paths).
    """
    report = ValidationReport(dataset_name=dataset.name)
    add = report.issues.append

    if not dataset.categories:
        add(Issue(Severity.ERROR, "dataset has no categories"))
    if not dataset.samples:
        add(Issue(Severity.ERROR, "dataset has no samples"))

    known_categories = {c.id for c in dataset.categories}
    seen_images: dict[str, int] = {}
    for sample in dataset.samples:
        if sample.image in seen_images:
            add(Issue(
                Severity.WARNING,
                f"duplicate image entry {sample.image!r} "
                f"(also sample {seen_images[sample.image]})",
                sample.id,
            ))
        else:
            seen_images[sample.image] = sample.id
        if sample.width <= 0 or sample.height <= 0:
            add(Issue(
                Severity.ERROR,
                f"non-positive image size {sample.width}x{sample.height}",
                sample.id,
            ))
        if check_images and not dataset.image_path(sample).is_file():
            add(Issue(
                Severity.WARNING,
                f"image file not found: {dataset.image_path(sample)}",
                sample.id,
            ))
        _check_boxes(sample, known_categories, add)

    if dataset.samples and dataset.num_annotations == 0:
        add(Issue(Severity.WARNING, "dataset has no annotations at all"))
    logger.info("%s", report.summary())
    return report


def _check_boxes(sample: Sample, known_categories: set[int], add) -> None:
    for idx, box in enumerate(sample.annotations):
        x1, y1, x2, y2 = box.bbox
        where = f"box {idx} {box.bbox}"
        if any(not math.isfinite(v) for v in box.bbox):
            add(Issue(Severity.ERROR, f"{where}: non-finite coordinates", sample.id))
            continue
        if x2 <= x1 or y2 <= y1:
            add(Issue(Severity.ERROR, f"{where}: degenerate (zero or negative extent)", sample.id))
        if (
            x1 < -_BOUNDS_TOLERANCE
            or y1 < -_BOUNDS_TOLERANCE
            or x2 > sample.width + _BOUNDS_TOLERANCE
            or y2 > sample.height + _BOUNDS_TOLERANCE
        ):
            add(Issue(
                Severity.ERROR,
                f"{where}: outside image bounds {sample.width}x{sample.height}",
                sample.id,
            ))
        if box.category_id not in known_categories:
            add(Issue(
                Severity.ERROR,
                f"{where}: unknown category id {box.category_id}",
                sample.id,
            ))
