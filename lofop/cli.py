"""LOFOP command-line interface.

Built on argparse rather than a third-party CLI framework so the ``lofop``
command works in every environment the core does (containers, edge devices,
CI) with zero extra dependencies. Commands map one-to-one onto public SDK
functions -- the CLI parses arguments and prints; logic stays in the library.

Current commands::

    lofop version
    lofop dataset convert  --from coco --source ann.json --to yolo --target out/
    lofop dataset validate --format yolo --source dataset_root/
    lofop dataset stats    --format coco --source ann.json [-o stats.md]
    lofop train            --config configs/train_shapes.yaml
    lofop benchmark        --config configs/lofop-detect/n.yaml [...] [-o table.md]
    lofop export           --config configs/lofop-detect/s.yaml --checkpoint best.pt -o model.onnx

``train`` and ``benchmark`` need the ``lofop[models]`` extra (PyTorch); torch
imports happen inside those handlers so every other command works without it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lofop.core.exceptions import LofopError
from lofop.core.logging import configure_logging
from lofop.data import compute_stats, convert_dataset, load_dataset, validate_dataset
from lofop.version import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="lofop", description="LOFOP computer vision framework")
    parser.add_argument("--log-level", default=None, help="log level (default: env or INFO)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version", help="print the LOFOP version")

    dataset = commands.add_parser("dataset", help="dataset tools")
    actions = dataset.add_subparsers(dest="action", required=True)

    convert = actions.add_parser("convert", help="convert between annotation formats")
    convert.add_argument("--from", dest="source_format", required=True, help="source format name")
    convert.add_argument("--source", required=True, help="source file or directory")
    convert.add_argument("--to", dest="target_format", required=True, help="target format name")
    convert.add_argument("--target", required=True, help="target file or directory")
    convert.add_argument("--image-root", default=None, help="image directory (COCO sources)")

    validate = actions.add_parser("validate", help="validate a dataset")
    validate.add_argument("--format", dest="format_name", required=True, help="format name")
    validate.add_argument("--source", required=True, help="source file or directory")
    validate.add_argument("--image-root", default=None, help="image directory (COCO sources)")
    validate.add_argument(
        "--no-check-images", action="store_true", help="skip image file existence checks"
    )

    stats = actions.add_parser("stats", help="compute dataset statistics")
    stats.add_argument("--format", dest="format_name", required=True, help="format name")
    stats.add_argument("--source", required=True, help="source file or directory")
    stats.add_argument("--image-root", default=None, help="image directory (COCO sources)")
    stats.add_argument("-o", "--output", type=Path, default=None, help="write markdown report here")
    stats.add_argument("--json", action="store_true", help="print JSON instead of markdown")

    train = commands.add_parser("train", help="train a detector from a training config")
    train.add_argument("--config", required=True, help="training config YAML")
    train.add_argument("--resume", action="store_true", help="resume from last.pt")

    bench = commands.add_parser("benchmark", help="measure the LOFOP metric table for models")
    bench.add_argument(
        "--config", action="append", required=True,
        help="model config YAML; repeat for multiple columns",
    )
    bench.add_argument("--size", type=int, default=640, help="benchmark image resolution")
    bench.add_argument("--checkpoint", default=None, help="weights (best.pt/last.pt) to load")
    bench.add_argument("-o", "--output", type=Path, default=None, help="write the table here")

    export = commands.add_parser("export", help="export a detector to ONNX")
    export.add_argument("--config", required=True, help="model config YAML")
    export.add_argument("--checkpoint", default=None, help="weights (best.pt/last.pt) to load")
    export.add_argument("--size", type=int, default=640, help="input resolution for the graph")
    export.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    export.add_argument("--no-verify", action="store_true", help="skip onnxruntime verification")
    export.add_argument("-o", "--output", type=Path, required=True, help="output .onnx path")
    return parser


def _load_kwargs(args: argparse.Namespace) -> dict:
    return {"image_root": args.image_root} if getattr(args, "image_root", None) else {}


def _cmd_convert(args: argparse.Namespace) -> int:
    dataset = convert_dataset(
        args.source_format, args.source, args.target_format, args.target, **_load_kwargs(args)
    )
    print(
        f"Converted {len(dataset)} samples / {dataset.num_annotations} annotations: "
        f"{args.source_format} -> {args.target_format} ({args.target})"
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.format_name, args.source, **_load_kwargs(args))
    report = validate_dataset(dataset, check_images=not args.no_check_images)
    for issue in report.issues:
        print(issue)
    print(report.summary())
    return 0 if report.ok else 1


def _cmd_stats(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.format_name, args.source, **_load_kwargs(args))
    result = compute_stats(dataset)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.to_markdown())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.to_markdown(), encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import lofop.models  # noqa: F401  (registers model components)
    from lofop.core.config import Config
    from lofop.registries import HUB
    from lofop.training import DetectionTorchDataset, Trainer

    cfg = Config.load(args.config)
    model = HUB.build(cfg.model)
    data = cfg.data
    image_size = data.get("image_size", 640)
    load_kwargs = {"image_root": data.image_root} if "image_root" in data else {}
    train_ds = DetectionTorchDataset(
        load_dataset(data.format, data.train_source, **load_kwargs),
        image_size=image_size, augment=True,
    )
    val_ds = None
    if "val_source" in data:
        val_ds = DetectionTorchDataset(
            load_dataset(data.format, data.val_source, **load_kwargs), image_size=image_size,
        )
    trainer = Trainer(model, train_ds, val_ds, **cfg.get("training", Config()).to_dict())
    if args.resume:
        trainer.resume()
    metrics = trainer.fit()
    if metrics is not None:
        print(f"final: mAP50 {metrics.map50:.4f}, mAP50-95 {metrics.map50_95:.4f}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import torch

    import lofop.models  # noqa: F401  (registers model components)
    from lofop.core.config import Config
    from lofop.registries import HUB
    from lofop.utils import benchmark_model, render_table

    reports = []
    for config_path in args.config:
        cfg = Config.load(config_path)
        model = HUB.build(cfg.model)
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("ema", {}).get("module", payload.get("model", payload))
            model.load_state_dict(state)
        reports.append(benchmark_model(model, Path(config_path).stem, image_size=args.size))
    table = render_table(reports)
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    import torch

    import lofop.models  # noqa: F401  (registers model components)
    from lofop.core.config import Config
    from lofop.deploy import export_onnx
    from lofop.registries import HUB

    cfg = Config.load(args.config)
    model = HUB.build(cfg.model)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("ema", {}).get("module", payload.get("model", payload))
        model.load_state_dict(state)
    path = export_onnx(
        model, args.output, image_size=args.size, opset=args.opset,
        verify=not args.no_verify,
    )
    print(f"Exported {path} ({path.stat().st_size / 1e6:.1f} MB, verified={not args.no_verify})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    if args.command == "version":
        print(__version__)
        return 0
    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "benchmark":
            return _cmd_benchmark(args)
        if args.command == "export":
            return _cmd_export(args)
        handlers = {"convert": _cmd_convert, "validate": _cmd_validate, "stats": _cmd_stats}
        return handlers[args.action](args)
    except LofopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
