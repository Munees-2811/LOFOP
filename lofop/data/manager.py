"""Dataset manager: format-agnostic load, save, and convert.

The manager is a thin façade over the ``dataset_format`` registry group, so
``convert_dataset("coco", ..., "yolo", ...)`` works for any pair of registered
formats -- including formats added by plugins -- without either side knowing
about the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lofop.core.logging import get_logger
from lofop.data.dataset import Dataset
from lofop.data.formats.base import FORMATS, DatasetAdapter

logger = get_logger(__name__)


def get_adapter(format_name: str) -> DatasetAdapter:
    """Instantiate the registered adapter for ``format_name``.

    Raises:
        RegistryError: If the format is unknown (message lists known formats).
    """
    return FORMATS.get(format_name)()


def load_dataset(format_name: str, source: str | Path, **kwargs: Any) -> Dataset:
    """Load a dataset via the named format adapter.

    Args:
        format_name: Registered format (``"coco"``, ``"yolo"``, ``"voc"``, ...).
        source: Format-specific source (see the adapter's docs).
        **kwargs: Adapter-specific options (e.g. ``image_root`` for COCO).
    """
    dataset = get_adapter(format_name).load(source, **kwargs)
    logger.info(
        "Loaded %r via %s: %d samples, %d annotations, %d categories",
        dataset.name, format_name, len(dataset), dataset.num_annotations,
        len(dataset.categories),
    )
    return dataset


def save_dataset(dataset: Dataset, format_name: str, target: str | Path, **kwargs: Any) -> None:
    """Save a dataset via the named format adapter."""
    get_adapter(format_name).save(dataset, target, **kwargs)


def convert_dataset(
    source_format: str,
    source: str | Path,
    target_format: str,
    target: str | Path,
    **load_kwargs: Any,
) -> Dataset:
    """Convert between any two registered formats through the canonical model.

    Returns:
        The intermediate canonical dataset (useful for chained validation or
        statistics without re-reading from disk).
    """
    dataset = load_dataset(source_format, source, **load_kwargs)
    save_dataset(dataset, target_format, target)
    logger.info("Converted %s -> %s (%s)", source_format, target_format, target)
    return dataset
