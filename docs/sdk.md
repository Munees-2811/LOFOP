# Python SDK Reference

`lofop.sdk.Detector` is the high-level API: one class that builds, trains, evaluates, runs, and
exports a detector. It wraps the lower layers without hiding them — `Detector.model` is the
underlying torch module, and every lower-level API (registries, `Config`, `Trainer`, deploy
functions) stays public for users who need more control.

Requires the models extra: `pip install "lofop[models]"`.

```python
from lofop import Detector
```

## Quick start

```python
det = Detector("lofop-detect-s", num_classes=2, class_names=["cat", "dog"])

det.train(data_format="coco", train_source="train.json",
          val_source="val.json", image_root="images/", epochs=100)

for hit in det.predict(["a.jpg", "b.jpg"]):
    for box, score, label in zip(hit.boxes, hit.scores, hit.labels):
        print(det.class_names[label], f"{score:.2f}", box)

det.export("model.onnx")
```

## Constructing a Detector

```python
Detector(
    model="lofop-detect-s",   # what to build (see below)
    num_classes=80,           # head classes (ignored for prebuilt modules)
    checkpoint=None,          # optional weights: best.pt / last.pt / state_dict
    class_names=None,         # optional label names, len == num_classes
    image_size=640,           # inference resolution
    device=None,              # "cpu" / "cuda"; auto-selects CUDA when available
)
```

`model` accepts, in order of convenience:

| Form | Example | Use when |
|---|---|---|
| Variant name | `"lofop-detect-ex"` or just `"ex"` | you want a standard size |
| Config YAML path | `"my_model.yaml"` | you keep custom model configs |
| Config / dict | `Detector({"model": {...}})` | programmatic model specs |
| Built module | `Detector(my_lofop_detect)` | you assembled the model yourself |

### Model family

| Variant | Parameters | Intended use |
|---|---|---|
| `lofop-detect-n` | 1.3M | fast CPU inference, embedded |
| `lofop-detect-s` | 3.8M | general-purpose default |
| `lofop-detect-ex` | 20.1M | higher-accuracy GPU deployments |

All variants share one architecture; only widths/depths differ. Checkpoints load with
`Detector("ex", checkpoint="best.pt")` — EMA weights are picked automatically when present.

## Inference: `predict()`

```python
results = det.predict("photo.jpg")                       # one image
results = det.predict([img1, img2])                      # batch
results = det.predict(frame_tensor, score_threshold=0.5) # per-call threshold
```

- **Inputs**: file paths (`str`/`Path`), PIL images, or float CHW / 1xCHW tensors in `[0, 1]` —
  mixed freely in a list.
- **Output**: one `Detections` per image with three parallel lists — `boxes` (xyxy floats),
  `scores` (descending), `labels` (class indices; map to names via `det.class_names`).
- **Coordinates come back in the ORIGINAL image space** (resized internally to `image_size`,
  mapped back, clamped to the image bounds) — no manual rescaling.
- `score_threshold=` overrides the model's confidence cut for that call only.

## Training: `train()`

Two ways to provide data:

```python
# 1. From annotation files on disk (COCO / YOLO / VOC):
det.train(data_format="coco", train_source="train.json",
          val_source="val.json", image_root="images/", epochs=100)

# 2. From canonical datasets (anything lofop.data produces):
from lofop.data import load_dataset
train = load_dataset("yolo", "dataset_root/")
det.train(train_data=train, val_data=my_val, epochs=100)
```

Common knobs are first-class (`epochs`, `batch_size`, `lr`, `checkpoint_dir`); everything else
passes through to `lofop.training.Trainer` (`optimizer="AdamW"`, `warmup_epochs`, `amp`,
`workers`, ...). Training runs with AMP (on CUDA), EMA weights, warmup+cosine schedule, and
atomic checkpoints; **the trained EMA weights replace the detector's weights** when `train()`
returns, so `predict()` immediately uses the best result. Returns `DetectionMetrics`
(`map50`, `map50_95`, `precision`, `recall`) when validation data was given.

## Evaluation: `evaluate()`

```python
metrics = det.evaluate(val_dataset)     # canonical dataset in, COCO protocol out
print(metrics.map50, metrics.map50_95, metrics.precision, metrics.recall)
```

## Export: `export()`

```python
det.export("model.onnx")                       # ONNX, verified via onnxruntime
det.export("model.engine", fp16=True)          # TensorRT (suffix picks the format)
det.export("m.onnx", opset=18, verify=False)   # kwargs pass to the deploy layer
```

Format is inferred from the suffix (`.onnx` / `.engine`) or forced with `format=`. The exported
graph contains network + box decoding; finish inference torch-free with
`lofop.deploy.postprocess_dense` (see [docs/deploy.md](deploy.md)).

## Weights: `save()` / `checkpoint=`

```python
det.save("weights.pt")                                  # write current weights
det2 = Detector("s", num_classes=2, checkpoint="weights.pt")   # load them back
```

`checkpoint=` also accepts Trainer checkpoints (`best.pt`/`last.pt`); EMA weights win when
present.

## Performance: `optimize()`

```python
det.optimize()    # channels_last CPU inference: measured 1.57x forward at 640px
```

Chainable and in-place; see `LofopDetect.optimize_for_inference` for the measurement details.

## Introspection

```python
det.num_parameters     # e.g. 20_117_398 for "ex" at 80 classes
det.model              # the underlying LofopDetect torch module
det.class_names        # label names used by consumers of predict()
repr(det)              # Detector(classes=2, parameters=..., device=cpu, image_size=640)
```

## Errors

Everything raises `lofop.LofopError` subclasses with structured context — an unknown variant
name lists the valid ones, a bad `class_names` length reports both sizes, and so on. One
`except LofopError` catches all framework failures.

## Complete example: train on your own data, ship an engine

```python
from lofop import Detector

det = Detector("lofop-detect-ex", num_classes=3,
               class_names=["helmet", "vest", "person"])

metrics = det.train(
    data_format="yolo", train_source="site_data/",
    epochs=150, batch_size=16, lr=0.01,
    checkpoint_dir="runs/safety",
)
print(f"mAP@50: {metrics.map50:.3f}" if metrics else "trained (no val set)")

det.save("safety_weights.pt")
det.export("safety.onnx")                    # portable
det.export("safety.engine", fp16=True)       # NVIDIA deployment

for hit in det.predict("cctv_frame.jpg", score_threshold=0.4):
    for box, score, label in zip(hit.boxes, hit.scores, hit.labels):
        print(det.class_names[label], round(score, 2), [round(v) for v in box])
```
