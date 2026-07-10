"""LOFOP training subsystem: torch data bridge, trainer, EMA, checkpoints,
and detection evaluation. Requires the ``lofop[models]`` extra (PyTorch)."""

from lofop.training.checkpoint import CheckpointManager
from lofop.training.ema import ModelEMA
from lofop.training.evaluator import DetectionMetrics, evaluate_detections
from lofop.training.hooks import TensorBoardHook, attach_tensorboard
from lofop.training.schedulers import build_scheduler
from lofop.training.torch_data import DetectionTorchDataset, detection_collate, image_to_tensor
from lofop.training.trainer import Trainer, warmup_cosine_lr

__all__ = [
    "Trainer",
    "warmup_cosine_lr",
    "build_scheduler",
    "TensorBoardHook",
    "attach_tensorboard",
    "DetectionTorchDataset",
    "detection_collate",
    "image_to_tensor",
    "ModelEMA",
    "CheckpointManager",
    "DetectionMetrics",
    "evaluate_detections",
]
