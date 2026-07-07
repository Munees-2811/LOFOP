"""Dataset format adapter contract.

An adapter translates one on-disk annotation format to and from the canonical
:class:`~lofop.data.dataset.Dataset` model. Adapters register into the
``dataset_format`` registry group under the format's user-facing name
(``"coco"``, ``"yolo"``, ``"voc"``), which is what the dataset manager and the
CLI resolve. Third-party formats plug in the same way from plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lofop.data.dataset import Dataset
from lofop.registries import HUB

FORMATS = HUB.get_or_new("dataset_format")


class DatasetAdapter(ABC):
    """Loads and saves one annotation format.

    ``source``/``target`` semantics are format-specific (a JSON file for COCO,
    a dataset root directory for YOLO and VOC); each adapter documents its
    layout. Adapters must be stateless so one instance can serve many calls.
    """

    @abstractmethod
    def load(self, source: str | Path, *, name: str | None = None) -> Dataset:
        """Read annotations from ``source`` into a canonical dataset.

        Args:
            source: Format-specific location of the annotations.
            name: Dataset name override; defaults to something derived from
                ``source``.

        Raises:
            DataError: On missing/unreadable/malformed inputs.
        """

    @abstractmethod
    def save(self, dataset: Dataset, target: str | Path) -> None:
        """Write ``dataset`` to ``target`` in this adapter's format.

        Args:
            dataset: Canonical dataset to export.
            target: Format-specific destination; parents are created.

        Raises:
            DataError: If the dataset cannot be represented or written.
        """
