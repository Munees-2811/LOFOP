"""Dataset format adapters.

Importing this package registers the built-in adapters (COCO, YOLO, VOC) into
the ``dataset_format`` registry group. Third-party formats register the same
way from plugins.
"""

from lofop.data.formats.base import FORMATS, DatasetAdapter
from lofop.data.formats.coco import CocoAdapter
from lofop.data.formats.voc import VocAdapter
from lofop.data.formats.yolo import YoloAdapter

__all__ = ["FORMATS", "DatasetAdapter", "CocoAdapter", "YoloAdapter", "VocAdapter"]
