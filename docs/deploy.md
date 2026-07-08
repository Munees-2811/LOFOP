# Deployment Reference

`lofop.deploy` turns trained detectors into runtime artifacts.

## ONNX export

```bash
lofop export --config configs/lofop-detect/s.yaml --checkpoint runs/train/best.pt \
    -o model.onnx --size 640
```

```python
from lofop.deploy import export_onnx
export_onnx(model, "model.onnx", image_size=640)   # verifies via onnxruntime by default
```

Design decisions:

- **The graph contains the network plus box decoding, not NMS.** Runtimes disagree on NMS
  operator support, and thresholds are deployment-time decisions. The graph outputs dense
  ``boxes (1, N, 4)`` and ``scores (1, N, C)``; post-processing finishes the job outside.
- **Fixed input resolution.** Pyramid decode points are baked in for one size, which is what
  TensorRT/OpenVINO engines want. Export once per deployed resolution.
- **Verification is on by default.** Export runs the graph under onnxruntime and compares
  against torch outputs; a checkpoint that exports but diverges numerically fails loudly
  instead of shipping.
- EMA weights are used automatically when a training checkpoint (`best.pt`/`last.pt`) is passed
  to the CLI.

## Torch-free post-processing

Inference hosts need only the LOFOP core (no PyTorch) next to their ONNX runtime:

```python
import onnxruntime
from lofop.deploy import postprocess_dense

session = onnxruntime.InferenceSession("model.onnx")
boxes, scores = session.run(None, {"images": batch})       # (1, N, 4), (1, N, C)
detections = postprocess_dense(boxes[0], scores[0],
                               score_threshold=0.25, nms_iou=0.6)
detections.boxes, detections.scores, detections.labels     # final results
```

`postprocess_dense` uses LOFOP's native C++ class-aware NMS when built, so the full
onnxruntime + LOFOP-core inference stack has no deep-learning-framework dependency. This is
exactly the stack the `docker/Dockerfile-onnx` image ships.

## TensorRT export

```bash
lofop export --config configs/lofop-detect/s.yaml --checkpoint best.pt \
    --format tensorrt --fp16 -o model.engine
```

```python
from lofop.deploy import export_tensorrt
export_tensorrt(model, "model.engine", image_size=640, fp16=True)
```

Export is two-stage by design: **model -> ONNX -> engine**, reusing the same verified ONNX graph
so the ONNX Runtime and TensorRT paths share one source of truth and one post-processing routine.
Notes:

- **FP16** is a single builder flag -- large speedup on NVIDIA hardware, negligible accuracy cost
  for detectors.
- **INT8** additionally needs representative calibration data
  (`export_tensorrt(..., int8=True, calibration_inputs=[...])`); a minimal entropy calibrator
  consumes the provided `(1, 3, S, S)` arrays.
- TensorRT and a CUDA GPU are needed only for the engine-build step. The module imports without
  them, the ONNX intermediate is always produced first, and a missing `tensorrt` package raises a
  clear, actionable error instead of an opaque `ImportError`.
- Engines are GPU- and TensorRT-version-specific -- build on the target hardware.

## Native ops portability

Post-processing NMS/IoU run through the native C++ fast path when the library is built, and
through the pure-Python fallback otherwise -- on every OS. The builder auto-selects a toolchain:
`g++`/`clang++`/`c++` on Linux/macOS, and `g++`/`clang++` (MinGW/LLVM) or MSVC `cl.exe` on Windows.
Check which backend is live:

```python
from lofop.ops import backend, build_native
build_native()      # optional; compiles the C++ library if a compiler is present
print(backend())    # "native" or "python"
```

Nothing requires the C++ path -- it is a pure accelerator. `lofop.ops` and
`postprocess_dense` work identically either way.

## Roadmap

OpenVINO engine builders consume the same ONNX artifact (planned); TorchScript export is planned
for torch-native serving.
