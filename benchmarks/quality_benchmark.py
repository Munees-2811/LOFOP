"""Fixed-protocol quality benchmark for LOFOP-Detect.

Runs the exact same experiment every time -- same synthetic dataset seeds,
same model seed, same schedule, same evaluation settings -- so two runs on
different code versions are directly comparable. This is the harness behind
every "improved X" claim: no number is reported that this script did not
measure.

Protocol (fixed unless overridden on the command line):
  data     shapes dataset, 128 train images (seed 3), 32 val images (seed 4),
           128px, 2 classes
  model    lofop-detect-n config, num_classes=2, torch.manual_seed(0)
  train    30 epochs, batch 8, SGD lr 0.02, warmup 2 epochs, EMA
  eval     COCO-protocol evaluator on the EMA weights; precision/recall and
           false positives per image at several confidence thresholds plus a
           best-F1 threshold scan
  speed    end-to-end predict() FPS (includes decode + NMS) and forward-only
           FPS, batch 1 at 128px, median of repeats
  memory   peak RSS of the process after training

Usage::

    python benchmarks/quality_benchmark.py -o baseline.json
    # ... change code ...
    python benchmarks/quality_benchmark.py -o candidate.json
    python benchmarks/quality_benchmark.py --compare baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import lofop.models  # noqa: E402, F401
from lofop.core.config import Config  # noqa: E402
from lofop.data.synthetic import generate_shapes_dataset  # noqa: E402
from lofop.models.losses import pairwise_iou  # noqa: E402
from lofop.registries import HUB  # noqa: E402
from lofop.training import DetectionTorchDataset, Trainer, evaluate_detections  # noqa: E402

_THRESHOLDS = [0.10, 0.25, 0.50]


def _collect_predictions(model, val_data):
    from torch.utils.data import DataLoader

    from lofop.training.torch_data import detection_collate

    loader = DataLoader(val_data, batch_size=8, collate_fn=detection_collate)
    predictions, targets = [], []
    with torch.inference_mode():
        for images, batch_targets in loader:
            for result in model.predict(images):
                predictions.append({k: v.cpu() for k, v in result.items()})
            targets.extend(batch_targets)
    return predictions, targets


def _pr_at(predictions, targets, threshold: float) -> dict:
    """Micro precision/recall and FP-per-image at a confidence threshold."""
    tp = fp = num_gt = 0
    for prediction, target in zip(predictions, targets):
        num_gt += int(target["labels"].numel())
        keep = prediction["scores"] >= threshold
        boxes = prediction["boxes"][keep]
        labels = prediction["labels"][keep]
        matched = set()
        for box, label in zip(boxes, labels):
            gt_mask = target["labels"] == label
            gt_boxes = target["boxes"][gt_mask]
            gt_index_map = gt_mask.nonzero().flatten().tolist()
            hit = False
            if len(gt_boxes):
                ious = pairwise_iou(box[None], gt_boxes)[0]
                best = int(ious.argmax())
                if float(ious[best]) >= 0.5 and gt_index_map[best] not in matched:
                    matched.add(gt_index_map[best])
                    hit = True
            tp += int(hit)
            fp += int(not hit)
    images = max(len(targets), 1)
    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / num_gt if num_gt else 0.0,
        "fp_per_image": fp / images,
    }


def _best_f1(predictions, targets) -> dict:
    best = {"threshold": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for i in range(1, 20):
        t = i / 20.0
        pr = _pr_at(predictions, targets, t)
        p, r = pr["precision"], pr["recall"]
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best["f1"]:
            best = {"threshold": t, "f1": f1, "precision": p, "recall": r}
    return best


def _measure_fps(model, image_size: int, repeats: int = 20) -> dict:
    images = torch.randn(1, 3, image_size, image_size)
    with torch.inference_mode():
        for _ in range(3):
            model.predict(images)
        predict_times, forward_times = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            model.predict(images)
            predict_times.append(time.perf_counter() - start)
        for _ in range(repeats):
            start = time.perf_counter()
            model(images)
            forward_times.append(time.perf_counter() - start)
    return {
        "predict_fps": 1.0 / statistics.median(predict_times),
        "forward_fps": 1.0 / statistics.median(forward_times),
    }


def run(epochs: int, image_size: int) -> dict:
    # Seed BOTH RNGs: torch drives init/shuffling, but the augmentation flip
    # uses the stdlib random module -- leaving it unseeded was measured to
    # cause ~1 mAP@50 run-to-run variance under identical code.
    import random

    random.seed(0)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_raw = generate_shapes_dataset(
            tmp_path / "train", num_images=128, image_size=image_size, seed=3
        )
        val_raw = generate_shapes_dataset(
            tmp_path / "val", num_images=32, image_size=image_size, seed=4
        )
        cfg = Config.load(
            REPO_ROOT / "lofop" / "configs" / "lofop-detect" / "n.yaml", resolve=False
        )
        cfg.num_classes = len(train_raw.categories)
        cfg.resolve()
        model = HUB.build(cfg.model)

        start = time.perf_counter()
        trainer = Trainer(
            model,
            DetectionTorchDataset(train_raw, image_size=image_size, augment=True),
            DetectionTorchDataset(val_raw, image_size=image_size),
            epochs=epochs, batch_size=8, lr=0.02, warmup_epochs=2.0,
            workers=0, checkpoint_dir=tmp_path / "ckpt", device="cpu",
        )
        trainer.fit()
        train_seconds = time.perf_counter() - start

        ema = trainer.ema.module.eval()
        predictions, targets = _collect_predictions(ema, trainer.val_data)
        metrics = evaluate_detections(predictions, targets)
        result = {
            "protocol": {"epochs": epochs, "image_size": image_size,
                         "torch": torch.__version__},
            "map50": metrics.map50,
            "map50_95": metrics.map50_95,
            "pr_curve": [_pr_at(predictions, targets, t) for t in _THRESHOLDS],
            "best_f1": _best_f1(predictions, targets),
            "speed": _measure_fps(ema, image_size),
            "train_seconds": train_seconds,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
    return result


def render(result: dict) -> str:
    lines = [
        f"mAP@50 {result['map50']:.4f} | mAP@50:95 {result['map50_95']:.4f}",
        f"best F1 {result['best_f1']['f1']:.4f} @ conf {result['best_f1']['threshold']:.2f} "
        f"(P {result['best_f1']['precision']:.4f} / R {result['best_f1']['recall']:.4f})",
    ]
    for pr in result["pr_curve"]:
        lines.append(
            f"conf {pr['threshold']:.2f}: P {pr['precision']:.4f} R {pr['recall']:.4f} "
            f"FP/img {pr['fp_per_image']:.2f}"
        )
    lines.append(
        f"speed: predict {result['speed']['predict_fps']:.1f} FPS, "
        f"forward {result['speed']['forward_fps']:.1f} FPS"
    )
    lines.append(
        f"train {result['train_seconds']:.0f}s, peak RSS {result['peak_rss_mb']:.0f} MB"
    )
    return "\n".join(lines)


def compare(before_path: Path, after_path: Path) -> str:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    rows = [
        ("mAP@50", before["map50"], after["map50"], "higher"),
        ("mAP@50:95", before["map50_95"], after["map50_95"], "higher"),
        ("best F1", before["best_f1"]["f1"], after["best_f1"]["f1"], "higher"),
        ("P @0.25", before["pr_curve"][1]["precision"],
         after["pr_curve"][1]["precision"], "higher"),
        ("R @0.25", before["pr_curve"][1]["recall"], after["pr_curve"][1]["recall"], "higher"),
        ("FP/img @0.25", before["pr_curve"][1]["fp_per_image"],
         after["pr_curve"][1]["fp_per_image"], "lower"),
        ("predict FPS", before["speed"]["predict_fps"], after["speed"]["predict_fps"], "higher"),
        ("forward FPS", before["speed"]["forward_fps"], after["speed"]["forward_fps"], "higher"),
        ("train seconds", before["train_seconds"], after["train_seconds"], "lower"),
        ("peak RSS MB", before["peak_rss_mb"], after["peak_rss_mb"], "lower"),
    ]
    lines = ["| Metric | Before | After | Delta | Better |", "|---|---:|---:|---:|---|"]
    for name, b, a, direction in rows:
        delta = a - b
        improved = delta > 0 if direction == "higher" else delta < 0
        mark = "yes" if improved else ("same" if abs(delta) < 1e-9 else "no")
        lines.append(f"| {name} | {b:.4f} | {a:.4f} | {delta:+.4f} | {mark} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixed-protocol LOFOP quality benchmark.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="write JSON here")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"),
                        help="compare two result JSONs instead of running")
    args = parser.parse_args(argv)

    if args.compare:
        print(compare(args.compare[0], args.compare[1]))
        return 0

    result = run(args.epochs, args.image_size)
    print(render(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
