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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    if args.command == "version":
        print(__version__)
        return 0
    handlers = {"convert": _cmd_convert, "validate": _cmd_validate, "stats": _cmd_stats}
    try:
        return handlers[args.action](args)
    except LofopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
