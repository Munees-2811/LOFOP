"""End-to-end LOFOP demo: train LOFOP-Detect on synthetic shapes and produce
the standard metric table.

Generates a shapes dataset, trains the `n` variant from scratch, evaluates
with the COCO-protocol evaluator, and renders the benchmark table (mAP@50,
mAP@50:95, precision, recall, FPS, params, FLOPs, model size) with measured
values. Runs on CPU in a few minutes; it exists to prove the whole pipeline
learns, not to set records.

Usage::

    python examples/train_shapes.py --epochs 30 --workdir runs/shapes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lofop.models  # noqa: E402, F401  (registers model components)
from lofop.core.config import Config  # noqa: E402
from lofop.core.logging import configure_logging  # noqa: E402
from lofop.data.synthetic import generate_shapes_dataset  # noqa: E402
from lofop.registries import HUB  # noqa: E402
from lofop.training import DetectionTorchDataset, Trainer  # noqa: E402
from lofop.utils import benchmark_model, render_table  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train LOFOP-Detect-n on synthetic shapes.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-images", type=int, default=96)
    parser.add_argument("--val-images", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--workdir", type=Path, default=Path("runs/shapes"))
    args = parser.parse_args(argv)
    configure_logging(use_rich=False)

    train_raw = generate_shapes_dataset(
        args.workdir / "data" / "train", num_images=args.train_images,
        image_size=args.image_size, seed=3,
    )
    val_raw = generate_shapes_dataset(
        args.workdir / "data" / "val", num_images=args.val_images,
        image_size=args.image_size, seed=4,
    )

    cfg = Config.load(REPO_ROOT / "lofop" / "configs" / "lofop-detect" / "n.yaml", resolve=False)
    cfg.num_classes = len(train_raw.categories)
    cfg.resolve()
    model = HUB.build(cfg.model)

    trainer = Trainer(
        model,
        DetectionTorchDataset(train_raw, image_size=args.image_size, augment=True),
        DetectionTorchDataset(val_raw, image_size=args.image_size),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        warmup_epochs=2.0, workers=0, checkpoint_dir=args.workdir / "checkpoints",
    )
    metrics = trainer.fit()

    report = benchmark_model(
        trainer.ema.module, "lofop-detect-n (shapes)",
        image_size=args.image_size, accuracy=metrics,
    )
    table = render_table([report])
    table += (
        "\nMetrics are measured on the synthetic shapes dataset -- a wiring/verification\n"
        "run, not a benchmark on a real dataset.\n"
    )
    print(table)
    (args.workdir / "metric-table.md").write_text(table, encoding="utf-8")
    print(f"table written to {args.workdir / 'metric-table.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
