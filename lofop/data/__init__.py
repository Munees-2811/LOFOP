"""LOFOP data subsystem: canonical dataset model, format adapters, manager,
validator, and statistics.

Torch-free by design -- everything here runs on edge devices and in CI without
a deep-learning runtime. Loader/transform integration with PyTorch lands in
the training subsystem.
"""

from lofop.data.dataset import BoxAnnotation, Category, Dataset, Sample
from lofop.data.formats import FORMATS, DatasetAdapter
from lofop.data.manager import convert_dataset, get_adapter, load_dataset, save_dataset
from lofop.data.statistics import DatasetStats, compute_stats
from lofop.data.validator import Issue, Severity, ValidationReport, validate_dataset
from lofop.data.visualize import draw_boxes, render_sample, visualize_dataset

__all__ = [
    "BoxAnnotation",
    "Category",
    "Dataset",
    "Sample",
    "FORMATS",
    "DatasetAdapter",
    "load_dataset",
    "save_dataset",
    "convert_dataset",
    "get_adapter",
    "validate_dataset",
    "ValidationReport",
    "Issue",
    "Severity",
    "compute_stats",
    "DatasetStats",
    "draw_boxes",
    "render_sample",
    "visualize_dataset",
]
