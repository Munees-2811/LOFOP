# Data Subsystem Reference

`lofop.data` normalizes annotation formats into one canonical model and provides conversion,
validation, and statistics on top of it. It is torch-free: the only dependency is Pillow (used to
read image sizes for formats that do not store them).

## Canonical model

- `Dataset(name, categories, samples, image_root=...)` -- categories with unique ids/names.
- `Sample(image, width, height, annotations, id)` -- one image; `image` is relative to
  `image_root` when set.
- `BoxAnnotation(bbox, category_id, attributes)` -- boxes are `(x1, y1, x2, y2)` in **absolute
  pixels**; format quirks (COCO xywh, YOLO normalized cxcywh, VOC 1-based) are handled at the
  adapter boundary.

## Formats

Adapters live in the `dataset_format` registry group; plugins can add more.

| Format | `source` / `target` | Layout |
|---|---|---|
| `coco` | annotation JSON file | standard `images`/`annotations`/`categories` arrays |
| `yolo` | dataset root dir | `classes.txt`, `images/`, `labels/*.txt` (normalized cxcywh) |
| `voc`  | dataset root dir | `Annotations/*.xml`, `JPEGImages/` |

Round-trip notes: COCO preserves category/image ids and `iscrowd`; YOLO reassigns ids to
contiguous 0..N-1 indices (inherent to the format) and reads image sizes from the files; VOC
preserves the `difficult` flag and assigns ids in first-seen order.

## Python API

```python
from lofop.data import load_dataset, save_dataset, convert_dataset
from lofop.data import validate_dataset, compute_stats

ds = load_dataset("coco", "instances_train.json", image_root="train2017/")
save_dataset(ds, "yolo", "out/yolo_root")
ds = convert_dataset("yolo", "root/", "coco", "out.json")   # any registered pair

report = validate_dataset(ds)          # errors block training; warnings advise
if not report.ok:
    for issue in report.errors:
        print(issue)

stats = compute_stats(ds)
print(stats.to_markdown())             # or stats.to_dict() for JSON
```

The validator catches degenerate/out-of-bounds/non-finite boxes, dangling category references,
duplicate image entries, missing image files, and empty datasets. Statistics include per-category
counts, boxes-per-image, COCO-convention small/medium/large breakdown, and image-size counts.

## CLI

```bash
lofop dataset convert  --from coco --source instances.json --to yolo --target out/
lofop dataset validate --format yolo --source dataset_root/          # exit 1 on errors
lofop dataset stats    --format coco --source instances.json -o stats.md [--json]
lofop dataset show     --format coco --source instances.json -o vis/ [--limit N]
```

`--image-root` points COCO sources at their image directory so validation can check files (and
`show` can load them). `show` writes one annotated PNG per sample; `lofop.data.draw_boxes` /
`render_sample` / `visualize_dataset` expose the same rendering in code, Pillow-only and torch-free.

## Docker

See [`../docker/README.md`](../docker/README.md): CPU, GPU (CUDA), and ONNX Runtime images, all
with the `lofop` CLI as entrypoint.
