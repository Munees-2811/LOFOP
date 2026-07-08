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

## Roadmap

TensorRT and OpenVINO engine builders consume the same ONNX artifact (planned); TorchScript
export is planned for torch-native serving.
