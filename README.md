# LOFOP

**LOFOP** is a modular, enterprise-grade computer vision framework built on PyTorch, with its own
original detector: **LOFOP-Detect**. It is an independent design informed by the architecture of
mature open-source projects (Ultralytics, MMDetection/MMEngine, Detectron2, OpenCV, PyTorch,
Transformers) without deriving from their code; RT-DETR serves as the research baseline that
LOFOP-Detect is designed against, never copied.

> **Status:** Phases 1-5 complete — core engine, data subsystem, native C++ ops, LOFOP-Detect
> models, training engine, and ONNX export. 184 tests passing. See
> [`docs/architecture.md`](docs/architecture.md) for the full roadmap and per-phase status.

## What works today

- **Datasets** — COCO, YOLO, and VOC support through one canonical model: any-to-any conversion,
  validation (degenerate/out-of-bounds boxes, missing files, dangling categories), and statistics.
- **LOFOP-Detect** — an original anchor-free detector (RidgeNet backbone, DeltaFusion neck with
  attention only on the cheap stride-32 level, ApexHead with an IoU-quality branch, dynamic top-k
  label assignment). Variants are pure config: `n` = 1.3M params, `s` = 3.8M.
  Design + trade-offs: [`docs/lofop-detect.md`](docs/lofop-detect.md).
- **Training** — AMP, EMA weights, warmup+cosine schedule, gradient clipping, atomic
  checkpointing with resume, event-bus lifecycle hooks, and a COCO-protocol evaluator
  (mAP@50, mAP@50:95, precision, recall).
- **Native C++ ops** — IoU and class-aware NMS kernels (20-200x over pure Python) with a
  verified-identical Python fallback, so a compiler is never required.
- **Benchmarking** — `lofop benchmark` renders the standard metric table (mAP, FPS, params,
  FLOPs, model size) and never prints a number that was not actually measured.
- **ONNX export** — `lofop export` writes a numerically verified ONNX graph (network + box
  decoding); `postprocess_dense` finishes inference torch-free with the C++ NMS, so serving
  hosts need only onnxruntime + the LOFOP core. Details: [`docs/deploy.md`](docs/deploy.md).
- **Deployment scaffolding** — CPU / CUDA / ONNX Runtime Docker images ([`docker/`](docker/README.md)).

## Installation

```bash
pip install -e ".[dev]"          # framework + dev tools (PyYAML, Pillow, pytest, ruff)
pip install -e ".[models]"       # + PyTorch, for lofop.models / lofop.training
python -c "from lofop.ops import build_native; build_native()"   # optional C++ fast path
```

Requires Python 3.9+. The core and data layers run without PyTorch (edge/CI friendly); torch
attaches only to the model and training subsystems.

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
lofop benchmark --config configs/lofop-detect/n.yaml --config configs/lofop-detect/s.yaml -o table.md
lofop export --config configs/lofop-detect/n.yaml --checkpoint runs/shapes/checkpoints/best.pt -o model.onnx
```

Measured on this repo's CI-sized shapes demo (30 CPU epochs, 128px): mAP@50 0.73,
mAP@50:95 0.49, 143 CPU FPS, 5.1 MB. Real accuracy comparisons against RT-DETR await the
COCO GPU run — the protocol is committed in `docs/lofop-detect.md`, and the table renders `-`
until numbers are measured.

**SDK** — everything is a registry entry built from YAML:

```python
from lofop import Config, HUB
import lofop.models                                   # registers model components

cfg = Config.load("configs/lofop-detect/s.yaml")
model = HUB.build(cfg.model)                          # ready LofopDetect
detections = model.predict(images)                    # boxes/scores/labels per image
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
  deploy/        # ONNX export + torch-free post-processing
  ops/ + csrc/   # native C++ IoU/NMS with Python fallback
  utils/         # model benchmarking (metric table, FLOPs, FPS)
  cli.py         # `lofop` command: dataset / train / benchmark / export
configs/         # model family definitions (n, s) via config inheritance
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

## Roadmap

Next phases: inference sources (video/RTSP/webcam), TensorRT/OpenVINO engines, REST serving, and CI.
The full subsystem map with per-phase status lives in [`docs/architecture.md`](docs/architecture.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
