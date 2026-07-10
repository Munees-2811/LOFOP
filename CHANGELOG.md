# Changelog

All notable changes to LOFOP are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CLI: `lofop predict` (image inference), `lofop evaluate` (dataset metrics),
  and `lofop doctor` (environment and backend diagnostics).
- Evaluation: F1 score, per-class precision/recall, and a class confusion
  matrix, exposed on `DetectionMetrics` (all backward-compatible defaults).
- Data: torch-free visualization (`draw_boxes`, `render_sample`,
  `visualize_dataset`) and the `lofop dataset show` command.
- Training: config-driven learning-rate schedulers (`warmup_cosine`,
  `warmup_linear`, `constant`, `step`), opt-in early stopping
  (`patience`/`min_delta`), and an optional `TensorBoardHook`.
- Benchmarking: CSV/JSON export (`render_csv`, `render_json`, `write_reports`)
  and a peak-memory column; `lofop benchmark --results-dir` and a consolidated
  `benchmarks/run_suite.py` runner.
- Packaging: `tensorboard` optional-dependency extra; a version-sync check
  (`scripts/check_version_sync.py`) wired into CI.
- Testing: `pytest-cov` with a CI-enforced coverage floor (~90% measured).
- Project: CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, issue/PR templates.

## [0.1.0]

### Added
- Core engine: registry/hub, YAML config (inheritance + interpolation), event
  bus, plugin manager, structured logging, typed exceptions.
- LOFOP-Detect anchor-free detector with `n`/`s`/`ex` variants.
- Native C++ box ops (IoU/NMS) with a pure-Python fallback, cross-platform.
- Data subsystem: COCO/YOLO/VOC adapters, conversion, validation, statistics.
- Training: AMP, EMA, checkpoints/resume, COCO-protocol evaluator, DDP path.
- Deployment: verified ONNX export, TensorRT (FP16/INT8), torch-free
  post-processing.
- Python SDK (`Detector`) and the `lofop` CLI.
- Packaging for PyPI (wheel/sdist) and AUR; CI and release workflows.

[Unreleased]: https://github.com/tedo001/LOFOP/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tedo001/LOFOP/releases/tag/v0.1.0
