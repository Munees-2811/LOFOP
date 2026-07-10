"""Draw bounding boxes on images -- torch-free dataset and prediction visuals.

Pillow only, so this runs anywhere the data subsystem does (edge devices, CI)
without a deep-learning runtime. :func:`draw_boxes` is the generic primitive
(used for both ground truth and predictions); :func:`render_sample` and
:func:`visualize_dataset` render a canonical :class:`~lofop.data.dataset.Dataset`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from lofop.data.dataset import Dataset, Sample

# Deterministic, high-contrast palette; colors are picked by class index so a
# class keeps the same color across images in a run.
_PALETTE = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9", "#ffe119",
)

Box = Sequence[float]


def _color_for(index: int) -> str:
    return _PALETTE[index % len(_PALETTE)]


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Box],
    *,
    labels: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    class_names: Sequence[str] | None = None,
    width: int = 3,
) -> Image.Image:
    """Return a copy of ``image`` with ``boxes`` drawn on it.

    Args:
        image: Source PIL image (not mutated).
        boxes: ``(x1, y1, x2, y2)`` absolute-pixel boxes.
        labels: Optional class index per box; drives box color and, with
            ``class_names``, the caption text.
        scores: Optional confidence per box, appended to the caption.
        class_names: Optional index -> name mapping for captions.
        width: Box outline thickness in pixels.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _default_font()
    for i, box in enumerate(boxes):
        label = int(labels[i]) if labels is not None else i
        color = _color_for(label)
        x1, y1, x2, y2 = (float(v) for v in box)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        caption = _caption(label, scores[i] if scores is not None else None, class_names)
        if caption:
            _draw_caption(draw, caption, x1, y1, color, font)
    return canvas


def render_sample(dataset: Dataset, sample: Sample, *, width: int = 3) -> Image.Image:
    """Render one dataset sample's ground-truth boxes onto its image."""
    image = Image.open(dataset.image_path(sample))
    boxes = [ann.bbox for ann in sample.annotations]
    labels = [ann.category_id for ann in sample.annotations]
    names = {cat.id: cat.name for cat in dataset.categories}
    captions = _CategoryNames(names)
    return draw_boxes(image, boxes, labels=labels, class_names=captions, width=width)


def visualize_dataset(
    dataset: Dataset,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    width: int = 3,
) -> list[Path]:
    """Render dataset samples to ``output_dir`` as PNGs; return the paths.

    Renders the first ``limit`` samples (all of them when ``limit`` is None).
    File names are ``<sample id>_<image stem>.png`` so outputs are stable and
    collision-free.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = dataset.samples if limit is None else dataset.samples[:limit]
    written: list[Path] = []
    for sample in samples:
        rendered = render_sample(dataset, sample, width=width)
        stem = Path(sample.image).stem
        path = output_dir / f"{sample.id}_{stem}.png"
        rendered.save(path)
        written.append(path)
    return written


class _CategoryNames:
    """Index-keyed name lookup that tolerates missing ids (falls back to id)."""

    def __init__(self, names: dict[int, str]) -> None:
        self._names = names

    def __getitem__(self, index: int) -> str:
        return self._names.get(index, str(index))


def _caption(
    label: int, score: float | None, class_names: Sequence[str] | None
) -> str:
    name = class_names[label] if class_names is not None else str(label)
    return f"{name} {score:.2f}" if score is not None else str(name)


def _draw_caption(
    draw: ImageDraw.ImageDraw, text: str, x: float, y: float, color: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top
    box_top = max(0.0, y - th - 4)
    draw.rectangle((x, box_top, x + tw + 4, box_top + th + 4), fill=color)
    draw.text((x + 2, box_top + 2), text, fill=_text_color(color), font=font)


def _text_color(background: str) -> str:
    """Black or white text, whichever contrasts with the box color."""
    r, g, b = ImageColor.getrgb(background)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 140 else "#ffffff"


def _default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()
