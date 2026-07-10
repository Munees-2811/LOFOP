# LOFOP

[![ci](https://github.com/tedo001/LOFOP/actions/workflows/ci.yml/badge.svg)](https://github.com/tedo001/LOFOP/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**LOFOP** is a modular, enterprise-grade computer vision framework built on PyTorch, with its own
original detector: **LOFOP-Detect**. It is an independent design and implementation that follows
modern computer-vision engineering practices while remaining self-contained.

> **Status:** Phases 1-5 complete — core engine, data subsystem, cross-platform native ops,
> LOFOP-Detect models, training engine, and ONNX + TensorRT export. 196 tests passing. See
> [`docs/architecture.md`](docs/architecture.md) for the full roadmap and per-phase status.

📖 **New here? Read the [Operator's Manual](MANUAL.md)** — a complete, step-by-step guide to
installing, training, exporting, deploying, and troubleshooting LOFOP.

## What works today

- **Datasets** — COCO, YOLO, and VOC support through one canonical model: any-to-any conversion,
  validation (degenerate/out-of-bounds boxes, missing files, dangling categories), and statistics.
- **LOFOP-Detect** — an original anchor-free detector (RidgeNet backbone, DeltaFusion neck with
  attention only on the cheap stride-32 level, ApexHead with an IoU-quality branch, dynamic top-k
  label assignment). Variants are pure config: `n` = 1.3M params, `s` = 3.8M, `ex` = 20.1M.
  Design + trade-offs: [`docs/lofop-detect.md`](docs/lofop-detect.md).
- **Python SDK** — `from lofop import Detector`: build, train, predict (boxes in original image
  coordinates), evaluate, and export through one documented class. Full reference:
  [`docs/sdk.md`](docs/sdk.md).
- **Training** — AMP, EMA weights, warmup+cosine schedule, gradient clipping, atomic
  checkpointing with resume, event-bus lifecycle hooks, and a COCO-protocol evaluator
  (mAP@50, mAP@50:95, precision, recall).
- **Cross-platform native ops** — IoU and class-aware NMS kernels (20-200x over pure Python) with a
  verified-identical Python fallback, so a compiler is never required. The C++ path builds with
  g++/clang on Linux/macOS and MinGW/clang/MSVC on Windows; `lofop.ops.backend()` reports which is
  active.
- **Benchmarking** — `lofop benchmark` renders the standard metric table (mAP, FPS, params,
  FLOPs, model size) and never prints a number that was not actually measured.
- **ONNX + TensorRT export** — `lofop export` writes a numerically verified ONNX graph (network +
  box decoding), or a TensorRT engine (`--format tensorrt --fp16`) via that same ONNX;
  `postprocess_dense` finishes inference torch-free with the C++ NMS, so serving hosts need only a
  runtime + the LOFOP core. Details: [`docs/deploy.md`](docs/deploy.md).
- **Deployment scaffolding** — CPU / CUDA / ONNX Runtime Docker images ([`docker/`](docker/README.md)).

## Installation

```bash
pip install lofop            # from PyPI
yay -S lofop                 # Arch Linux (AUR)
```

Optional feature sets (extras):

```bash
pip install "lofop[models]"      # + PyTorch, for lofop.models / lofop.training
pip install "lofop[deploy]"      # + onnx, onnxruntime, for ONNX export
pip install "lofop[all]"         # models + deploy in one go
python -c "from lofop.ops import build_native; build_native()"   # optional C++ fast path
```

From a source checkout instead:

```bash
pip install -e ".[dev]"          # framework + dev tools (pytest, ruff, build, twine)
```

Requires Python 3.9+. The core and data layers run without PyTorch (edge/CI friendly); torch
attaches only to the model and training subsystems. Maintainer release steps live in
[`docs/packaging.md`](docs/packaging.md).

## Quickstart

**Dataset tools** (no torch needed):

```bash
lofop dataset convert  --from coco --source instances.json --to yolo --target out/
lofop dataset validate --format yolo --source out/            # exit 1 on errors
lofop dataset stats    --format coco --source instances.json -o stats.md
```

**Train and benchmark** (five-minute CPU demo that fills the metric table end to end):

```bash
python examples/train_shapes.py --epochs 30 --workdir runs/shapes
lofop benchmark --config lofop/configs/lofop-detect/n.yaml --config lofop/configs/lofop-detect/s.yaml -o table.md
lofop predict  --config n --checkpoint runs/shapes/checkpoints/best.pt --source image.png
lofop evaluate --config n --checkpoint runs/shapes/checkpoints/best.pt --format coco --source val.json
lofop export --config lofop/configs/lofop-detect/n.yaml --checkpoint runs/shapes/checkpoints/best.pt -o model.onnx
lofop export --config lofop/configs/lofop-detect/n.yaml --format tensorrt --fp16 -o model.engine   # NVIDIA GPU
lofop doctor   # environment + backend diagnostics
```

Measured on the fixed-protocol benchmark (`benchmarks/quality_benchmark.py`, 30 CPU epochs,
128px shapes): mAP@50 0.92, best F1 0.90, 0.8 false positives/image at conf 0.25, 115 FPS
end-to-end predict. Accuracy on a real dataset awaits a full GPU training run — the protocol
is documented in `docs/lofop-detect.md`, and the table renders `-` until numbers are measured.

**Python SDK** — the one-import path ([full reference](docs/sdk.md)):

```python
from lofop import Detector

det = Detector("lofop-detect-ex", num_classes=2, class_names=["cat", "dog"])
det.train(data_format="coco", train_source="train.json", image_root="images/", epochs=100)
for hit in det.predict("photo.jpg"):                  # boxes in original image coordinates
    print(hit.boxes, hit.scores, hit.labels)
det.export("model.onnx")
```

Lower-level control remains fully public — registries, `Config`, `Trainer`, and the deploy
functions are the same objects the SDK uses:

```python
from lofop import Config, HUB
import lofop.models                                   # registers model components

cfg = Config.load("lofop/configs/lofop-detect/s.yaml")
model = HUB.build(cfg.model)                          # ready LofopDetect
```

Custom components plug in without touching the framework:

```python
from lofop.registries import BACKBONES

@BACKBONES.register()
class MyNet: ...
# then in YAML:  backbone: {type: backbone/MyNet, ...}
```

## Repository layout

```
lofop/
  core/          # registry, config, events, plugins, logging, exceptions (torch-free)
  data/          # canonical dataset model, COCO/YOLO/VOC adapters, validator, statistics
  models/        # LOFOP-Detect: RidgeNet, DeltaFusion, ApexHead, losses, assigner
  training/      # trainer, EMA, checkpoints, torch data bridge, COCO-protocol evaluator
  deploy/        # ONNX + TensorRT export, torch-free post-processing
  ops/ + csrc/   # native C++ IoU/NMS with Python fallback
  utils/         # model benchmarking (metric table, FLOPs, FPS)
  sdk.py         # high-level Python SDK: the Detector class (docs/sdk.md)
  configs/       # packaged model family definitions (n, s, ex)
  cli.py         # `lofop` command: dataset / train / benchmark / export
configs/         # training config examples
docker/          # CPU, CUDA, and ONNX Runtime images
docs/            # architecture, per-module references, LOFOP-Detect design doc
benchmarks/      # reusable performance measurement scripts
examples/        # end-to-end runnable demos
tests/           # pytest suite mirroring the package layout (177 tests)
```

## Development

```bash
python -m pytest                                   # run the test suite
ruff check lofop tests benchmarks examples         # lint
python benchmarks/bench_core.py -o report.md       # core engine micro-benchmarks
python benchmarks/bench_ops.py                     # C++ vs Python ops speedups
python benchmarks/bench_detect.py                  # detector params + latency
```

Every push and pull request runs the full test suite (Python 3.9/3.11/3.12, native C++ ops
built) and lint via GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), and a
distribution build check, so `main` stays releasable.

## Roadmap

Next phases: inference sources (video/RTSP/webcam), OpenVINO engines, and REST serving.
The full subsystem map with per-phase status lives in [`docs/architecture.md`](docs/architecture.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
