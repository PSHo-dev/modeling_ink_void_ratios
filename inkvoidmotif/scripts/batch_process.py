#!/usr/bin/env python3
"""Batch-process every image in a directory through the inkvoidmotif pipeline.

For each input image we:
  1. Build a minimal per-image config (no manual motifs -> auto-plan via vision).
  2. Write it to <output>/<image_stem>/config.json.
  3. Run the full inkvoidmotif pipeline (plan -> sheet -> slice -> compose).

Usage examples:

  python scripts/batch_process.py \
      --input /path/to/paintings \
      --output runs/batch

  python scripts/batch_process.py \
      --input ~/Downloads/paintings \
      --output runs/batch \
      --count 8 \
      --columns 2 \
      --spacing-ratio 0.225 \
      --theme "a new Chinese landscape reusing the same motifs and brushwork" \
      --limit 3

The script reuses the same providers/models that ``inkvoidmotif run`` uses by
default. Override them with ``--text-provider`` / ``--image-provider`` /
``--text-model`` / ``--image-model`` if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Make ``import inkvoidmotif.*`` work when running this file directly.
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inkvoidmotif.pipeline import run_config  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

DEFAULT_THEME = (
    "a new Chinese landscape painting in the same brushwork and palette as "
    "the source, recombining the same motifs into a fresh, balanced composition"
)


def discover_images(input_dir: Path, recursive: bool) -> list[Path]:
    walker = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = [
        p
        for p in walker
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]
    images.sort()
    return images


def build_config(image_path: Path, run_dir: Path, args: argparse.Namespace) -> dict:
    composition: dict = {
        "name": f"{image_path.stem}_recombined",
        "theme": args.theme,
        "size": args.composition_size,
    }
    if args.include_scholar:
        composition["include_scholar"] = True
    if args.layout:
        composition["layout"] = args.layout
    if args.source_style:
        composition["source_style"] = args.source_style

    if args.ink_ratio:
        parts = [p for p in str(args.ink_ratio).replace(" ", "").split(",") if p != ""]
        if len(parts) != 3:
            raise SystemExit("--ink-ratio must be three numbers: void,transition,ink")
        void, transition, ink = (float(x) for x in parts)
        composition["ink_ratio"] = {
            "void": void,
            "transition": transition,
            "ink": ink,
            "white_t": args.ink_white_t,
            "dark_t": args.ink_dark_t,
            "tol": args.ink_tol,
            "metric": args.ink_metric,
            "iters": args.ink_iters,
            "guide": not args.no_ink_guide,
            "revise": not args.no_ink_revise,
            "polish": args.ink_polish,
            "strength": args.ink_strength,
            "preserve_color": not args.ink_no_preserve_color,
            "guide_composition": args.ink_guide_composition,
            "seed": args.ink_seed,
        }

    config: dict = {
        "source": str(image_path.resolve()),
        "out_dir": str(run_dir.resolve()),
        "mode": "sheet",
        "count": args.count,
        "skip_existing": not args.overwrite,
        "quality": args.quality,
        "motif_sheet_size": args.motif_sheet_size,
        "motif_sheet_columns": args.columns,
        "motif_sheet_spacing_ratio": args.spacing_ratio,
        "composition": composition,
    }
    if args.text_provider:
        config["text_provider"] = args.text_provider
    if args.image_provider:
        config["image_provider"] = args.image_provider
    if args.text_model:
        config["text_model"] = args.text_model
    if args.image_model:
        config["image_model"] = args.image_model
    if args.crop_margin is not None:
        config["crop_margin"] = args.crop_margin
    return config


def process_one(image_path: Path, output_root: Path, args: argparse.Namespace) -> dict:
    run_dir = output_root / image_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(image_path, run_dir, args)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_config(config_path)


def find_completed_elsewhere(image_path: Path, skip_roots: list[Path]) -> Path | None:
    """Return the existing composition path if this image is already finished
    in any of the provided ``--skip-from`` roots, else None.

    We consider an image "done" when ``<root>/<stem>/compositions/`` exists and
    contains at least one ``.png``.
    """
    for root in skip_roots:
        comp_dir = root / image_path.stem / "compositions"
        if not comp_dir.is_dir():
            continue
        pngs = sorted(comp_dir.glob("*.png"))
        if pngs:
            return pngs[0]
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the inkvoidmotif pipeline on every image in a directory.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Directory containing source paintings.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output root. Each image gets its own subfolder named after its stem.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories of --input (default: top-level only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images (after sorting). Useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run everything even if outputs already exist (default: skip cached steps).",
    )
    parser.add_argument(
        "--skip-from",
        action="append",
        default=[],
        type=Path,
        help=(
            "Directory holding already-completed per-image runs (by image stem). "
            "Any input whose stem matches a subdir with a non-empty compositions/ "
            "folder there is skipped without invoking the pipeline. Repeatable."
        ),
    )

    # Composition knobs.
    parser.add_argument("--theme", default=DEFAULT_THEME)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--source-style", default=None)
    parser.add_argument("--include-scholar", action="store_true")
    parser.add_argument("--composition-size", default="1024x1536")

    # Motif / sheet knobs.
    parser.add_argument("--count", type=int, default=8, help="Motifs per painting.")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--spacing-ratio", type=float, default=0.225)
    parser.add_argument("--motif-sheet-size", default="1024x1536")
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=None,
        help="Override crop_margin written into each config (default: leave unset).",
    )
    parser.add_argument(
        "--quality",
        default="high",
        choices=["low", "medium", "high", "auto"],
    )

    # Ink-ratio control (tonal balance of the recomposed painting). Off unless
    # --ink-ratio is supplied. Uses an iterative measure->critique->regenerate
    # loop that genuinely changes composition; each iteration is one image call.
    parser.add_argument(
        "--ink-ratio",
        metavar="VOID,TRANSITION,INK",
        default=None,
        help="Target tonal balance, e.g. 0.6,0.25,0.15 (auto-normalized). Enables ink-ratio control on the composition.",
    )
    parser.add_argument(
        "--ink-iters",
        type=int,
        default=3,
        help="Max measure->critique->regenerate iterations per painting; stops early on convergence (default 3).",
    )
    parser.add_argument(
        "--no-ink-guide",
        action="store_true",
        help="Do not attach the pre-generated three-tone composition guide.",
    )
    parser.add_argument(
        "--no-ink-revise",
        action="store_true",
        help="Regenerate from scratch each iteration instead of revising the previous attempt.",
    )
    parser.add_argument(
        "--ink-polish",
        action="store_true",
        help="Apply a gentle cosmetic tone curve to the final pick (does NOT change composition).",
    )
    parser.add_argument(
        "--ink-strength",
        type=float,
        default=0.5,
        help="Polish tone-curve strength 0..1 (only with --ink-polish; default 0.5).",
    )
    parser.add_argument(
        "--ink-no-preserve-color",
        action="store_true",
        help="Polish in monochrome instead of preserving hue.",
    )
    parser.add_argument(
        "--ink-guide-composition",
        default="high_distance_高远",
        help="Composition prior for the guide (high_distance_高远, level_distance_平远, deep_distance_深远, river/valley).",
    )
    parser.add_argument(
        "--ink-seed", type=int, default=0, help="Seed for the composition guide layout (default 0)."
    )
    parser.add_argument(
        "--ink-metric",
        default="mean",
        choices=["mean", "max", "sum", "tvd"],
        help="Combined deviation aggregation across the 3 bands (default mean).",
    )
    parser.add_argument(
        "--ink-white-t",
        type=float,
        default=0.72,
        help="Luminance above which a pixel counts as void (default 0.72).",
    )
    parser.add_argument(
        "--ink-dark-t",
        type=float,
        default=0.28,
        help="Luminance below which a pixel counts as ink (default 0.28).",
    )
    parser.add_argument(
        "--ink-tol",
        type=float,
        default=0.05,
        help="Convergence/pass deviation tolerance on the chosen metric (default 0.05).",
    )

    # Provider / model overrides. ``None`` here means "let the pipeline auto-detect":
    # NVIDIA if NVIDIA_API_KEY is present, OpenAI otherwise. Pass ``--text-provider``
    # or ``--image-provider`` explicitly to force one side.
    parser.add_argument(
        "--text-provider",
        default=None,
        choices=["openai", "nvidia"],
        help="Defaults to auto-detect (NVIDIA if NVIDIA_API_KEY is set, else OpenAI).",
    )
    parser.add_argument(
        "--image-provider",
        default=None,
        choices=["openai", "nvidia"],
        help="Defaults to auto-detect (NVIDIA if NVIDIA_API_KEY is set, else OpenAI).",
    )
    parser.add_argument("--text-model", default=None)
    parser.add_argument(
        "--image-model",
        default=None,
        help=(
            "Image model slug. Default: ``openai/openai/gpt-5.5`` for the NVIDIA "
            "provider (Responses + image_generation tool), ``gpt-image-2`` for "
            "direct OpenAI. Leave unset to use whichever matches the resolved "
            "provider."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_dir: Path = args.input.expanduser().resolve()
    output_root: Path = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"error: --input is not a directory: {input_dir}", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    images = discover_images(input_dir, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print(f"no images found under {input_dir}", file=sys.stderr)
        return 1

    skip_roots: list[Path] = [
        root.expanduser().resolve()
        for root in args.skip_from
        if root.expanduser().resolve().is_dir()
    ]
    if args.skip_from and not skip_roots:
        print(
            "[batch] warning: --skip-from values did not resolve to existing directories: "
            + ", ".join(str(p) for p in args.skip_from),
            file=sys.stderr,
        )

    print(f"[batch] {len(images)} image(s) to process; output root = {output_root}")
    if skip_roots:
        print(f"[batch] checking skip-from roots: {[str(r) for r in skip_roots]}")
    summary: list[dict] = []
    overall_start = time.time()

    for i, image_path in enumerate(images, start=1):
        print(f"\n[batch {i}/{len(images)}] {image_path}")
        start = time.time()
        existing = find_completed_elsewhere(image_path, skip_roots)
        if existing is not None:
            print(f"[batch {i}/{len(images)}] SKIP (already done at {existing})")
            summary.append(
                {
                    "image": str(image_path),
                    "status": "skipped",
                    "elapsed_s": 0.0,
                    "existing": str(existing),
                }
            )
            continue
        try:
            result = process_one(image_path, output_root, args)
            elapsed = time.time() - start
            print(f"[batch {i}/{len(images)}] OK in {elapsed:.1f}s")
            for key, value in result.items():
                print(f"    {key}: {value}")
            summary.append(
                {
                    "image": str(image_path),
                    "status": "ok",
                    "elapsed_s": round(elapsed, 1),
                    "result": result,
                }
            )
        except Exception as exc:  # noqa: BLE001 - we want to keep going on errors
            elapsed = time.time() - start
            print(
                f"[batch {i}/{len(images)}] FAILED in {elapsed:.1f}s: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            summary.append(
                {
                    "image": str(image_path),
                    "status": "error",
                    "elapsed_s": round(elapsed, 1),
                    "error": str(exc),
                }
            )

    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_root": str(output_root),
                "total_elapsed_s": round(time.time() - overall_start, 1),
                "items": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ok = sum(1 for entry in summary if entry["status"] == "ok")
    skipped = sum(1 for entry in summary if entry["status"] == "skipped")
    fail = sum(1 for entry in summary if entry["status"] == "error")
    print(f"\n[batch] done: {ok} ok, {skipped} skipped, {fail} failed; summary -> {summary_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
