"""LOFOP training subsystem: torch data bridge, trainer, EMA, checkpoints,
and detection evaluation. Requires the ``lofop[models]`` extra (PyTorch)."""

from lofop.training.checkpoint import CheckpointManager
from lofop.training.ema import ModelEMA
from lofop.training.evaluator import DetectionMetrics, evaluate_detections
from lofop.training.torch_data import DetectionTorchDataset, detection_collate, image_to_tensor
from lofop.training.trainer import Trainer, warmup_cosine_lr

__all__ = [
    "Trainer",
    "warmup_cosine_lr",
    "DetectionTorchDataset",
    "detection_collate",
    "image_to_tensor",
    "ModelEMA",
    "CheckpointManager",
    "DetectionMetrics",
    "evaluate_detections",
]
