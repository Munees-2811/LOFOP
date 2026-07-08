"""LOFOP model subsystem: the LOFOP-Detect flagship detector and its parts.

Requires PyTorch (install with ``pip install lofop[models]``); the rest of
the framework stays importable without it. Importing this package registers
``backbone/RidgeNet``, ``neck/DeltaFusion``, ``head/ApexHead``, and
``model/LofopDetect``. Design document: docs/lofop-detect.md.
"""

from lofop.models.assigner import DynamicTopKAssigner
from lofop.models.backbone import RidgeNet
from lofop.models.detector import LofopDetect
from lofop.models.head import ApexHead
from lofop.models.losses import giou_loss, pairwise_iou, sigmoid_focal_loss
from lofop.models.neck import DeltaFusion

__all__ = [
    "RidgeNet",
    "DeltaFusion",
    "ApexHead",
    "LofopDetect",
    "DynamicTopKAssigner",
    "sigmoid_focal_loss",
    "giou_loss",
    "pairwise_iou",
]
