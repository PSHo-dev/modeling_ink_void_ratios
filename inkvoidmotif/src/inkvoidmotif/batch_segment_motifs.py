"""
batch_segment_motifs.py

Batch-segments motifs out of every painting in a directory, using the
inkvoidmotif pipeline (plan_motifs + either the sheet workflow or the per-motif
workflow). Each extracted motif is renamed to:

    <PaintingSeason>_<MotifName>

where <PaintingSeason> is parsed from the original painting's filename: the
first token before the first "_" or " " separator (e.g. "Autumn_RiverValley.png"
-> "Autumn", "Winter river temple.jpg" -> "Winter").

All motifs from all paintings are merged into one combined motif bank at
<motif_dir>/motif_bank.json, ready to be passed straight into
inkvoidmotif.pipeline.compose_from_bank() for the recomposition stage.

Filename collisions (e.g. two paintings both producing an "Autumn_Pine
Cluster" motif, or a re-run of this script over the same motif_dir) are never
overwritten -- a numeric suffix (_2, _3, ...) is appended instead, checked
both against files already on disk and against names already claimed earlier
in the same run.

Usage:
    python batch_segment_motifs.py /path/to/paintings /path/to/motif_dir
    python batch_segment_motifs.py /path/to/paintings /path/to/motif_dir --mode per_motif --count 6
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

from inkvoidmotif.image_ops import ensure_dir, slugify
from inkvoidmotif.pipeline import (
    default_image_provider,
    extract_motifs,
    generate_motif_panel,
    import_chatgpt_motif_bank,
    plan_motifs,
    read_json,
    write_json,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_season(stem: str) -> str:
    """First token before the first '_' or ' ' in the painting filename.

    "Autumn_RiverValley"   -> "Autumn"
    "Winter river temple"  -> "Winter"
    "Spring"               -> "Spring"   (no separator found; whole stem used)
    """
    m = re.match(r"^([^_ ]+)[_ ]", stem)
    return m.group(1) if m else stem


def unique_path(directory: Path, filename: str, taken: set[str]) -> Path:
    """Return a non-colliding path for `filename` inside `directory`.

    Appends _2, _3, ... as needed. Checks both files already present on disk
    (protects across separate runs of this script over the same motif_dir)
    and names already claimed earlier in this same run via `taken` (which
    this function mutates).
    """
    stem, suffix = Path(filename).stem, Path(filename).suffix
    candidate = filename
    n = 2
    while (directory / candidate).exists() or candidate in taken:
        candidate = f"{stem}_{n}{suffix}"
        n += 1
    taken.add(candidate)
    return directory / candidate


def batch_segment_paintings(
    paintings_dir: Path,
    motif_dir: Path,
    *,
    mode: str = "sheet",  # "sheet" (1 generation call/painting) or "per_motif" (slower, cleaner cutouts)
    count: int = 8,
    instructions: str | None = None,
    quality_exemplar_path: Path | None = None,
    size: str = "1024x1024",
    quality: str = "high",
    motif_sheet_columns: int = 2,
    skip_existing: bool = True,
) -> Path:
    """Segment motifs out of every painting in `paintings_dir`, naming each
    motif `<PaintingSeason>_<MotifName>`, and collect them into one shared
    bank at `motif_dir/motif_bank.json`.

    Re-running this over the same `motif_dir` (e.g. after adding more
    paintings to `paintings_dir`) will not overwrite previously extracted
    motif assets -- name collisions get a numeric suffix instead, and the
    combined motif_bank.json is rewritten to include everything found so far
    under `motif_dir/motifs/`.
    """
    paintings_dir = Path(paintings_dir).expanduser().resolve()
    if not paintings_dir.is_dir():
        raise FileNotFoundError(f"Painting directory not found: {paintings_dir}")
    if mode not in {"sheet", "per_motif"}:
        raise ValueError(f"Unknown mode {mode!r}; expected 'sheet' or 'per_motif'")
    if not 1 <= int(count) <= 24:
        raise ValueError("count must be between 1 and 24")
    if not 1 <= int(motif_sheet_columns) <= 8:
        raise ValueError("motif_sheet_columns must be between 1 and 8")
    if quality not in {"low", "medium", "high", "auto"}:
        raise ValueError("quality must be low, medium, high, or auto")
    motif_dir = ensure_dir(Path(motif_dir).expanduser().resolve())
    work_root = ensure_dir(motif_dir / "_per_painting")
    assets_dir = ensure_dir(motif_dir / "motifs")

    sources = sorted(p for p in paintings_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not sources:
        raise FileNotFoundError(f"No images found in {paintings_dir}")

    # Load any existing combined bank so re-runs append rather than clobber it.
    combined_bank_path = motif_dir / "motif_bank.json"
    combined_motifs: list[dict[str, Any]] = []
    if combined_bank_path.exists():
        existing = read_json(combined_bank_path)
        combined_motifs = list(existing.get("motifs", []))

    # Every motif filename already on disk (from this run or prior runs)
    # counts as "taken" so unique_path() never has to guess via existence
    # checks alone -- this also covers files moved/copied in manually.
    claimed_names: set[str] = {
        Path(m["asset_path"]).name for m in combined_motifs if m.get("asset_path")
    }

    seen_sources = set(existing.get("sources", []) if combined_bank_path.exists() else [])

    for source_path in sources:
        if skip_existing and source_path.name in seen_sources:
            print(f"[batch] {source_path.name} already present in the combined bank; skipped")
            continue
        season = parse_season(source_path.stem)
        work_dir = ensure_dir(work_root / source_path.stem)
        print(f"[batch] {source_path.name}  (season={season})")

        plan_path = plan_motifs(
            source_path=source_path,
            output_dir=work_dir,
            count=count,
            instructions=instructions,
            quality_exemplar_path=quality_exemplar_path,
        )
        plan = read_json(plan_path)
        print(
            f"[batch]   planner returned {len(plan.get('motifs', []))} motifs (asked for {count})"
        )

        if mode == "per_motif":
            bank_path = extract_motifs(
                source_path=source_path,
                plan_path=plan_path,
                output_dir=work_dir,
                size=size,
                quality=quality,
                quality_exemplar_path=quality_exemplar_path,
                skip_existing=skip_existing,
            )
        elif mode == "sheet":
            sheet_path = work_dir / "motif_sheet.png"
            if not (sheet_path.exists() and skip_existing):
                generate_motif_panel(
                    source_path=source_path,
                    plan_path=plan_path,
                    output_path=sheet_path,
                    size="1024x1536",
                    quality=quality,
                    columns=motif_sheet_columns,
                )
            sheet_result = import_chatgpt_motif_bank(
                sheet_path=sheet_path,
                plan_path=plan_path,
                output_dir=work_dir,
                columns=motif_sheet_columns,
            )
            bank_path = Path(sheet_result["motif_bank"])
        bank = read_json(bank_path)

        for item in bank["motifs"]:
            original_name = item["name"]
            motif_slug = slugify(original_name)
            season_slug = slugify(season)
            base_filename = f"{season_slug}_{motif_slug}.png"

            src_asset = Path(item["asset_path"])
            if not src_asset.is_absolute():
                src_asset = (bank_path.parent / src_asset).resolve()

            dest_asset = unique_path(assets_dir, base_filename, claimed_names)
            shutil.copy2(src_asset, dest_asset)

            # If a numeric suffix was added to avoid a collision, reflect it
            # in the human-readable name too so it's obvious in
            # motif_bank.json which physical asset each entry points to.
            disambiguator = dest_asset.stem[len(Path(base_filename).stem) :]
            new_item = dict(item)  # preserves description/role/must_preserve/bbox/prompt
            new_item["name"] = f"{season}_{original_name}{disambiguator}"
            new_item["asset_path"] = str(dest_asset.relative_to(motif_dir))
            new_item["crop_path"] = None
            new_item["source_painting"] = str(source_path.name)
            new_item["season"] = season
            combined_motifs.append(new_item)

        seen_sources.add(source_path.name)

    combined_bank_path = write_json(
        motif_dir / "motif_bank.json",
        {
            "sources": sorted(seen_sources),
            "image_provider": default_image_provider(),
            "motifs": combined_motifs,
        },
    )

    """
    make_contact_sheet(
        [assets_dir / Path(m["asset_path"]).name for m in combined_motifs],
        motif_dir / "motif_contact_sheet.png",
    )
    """

    print(
        f"[batch] {len(combined_motifs)} total motifs from {len(seen_sources)} painting(s) -> {motif_dir}"
    )
    return combined_bank_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paintings_dir", type=Path, help="Directory of source painting images.")
    parser.add_argument(
        "motif_dir", type=Path, help="Directory to write/merge the motif bank into."
    )
    parser.add_argument(
        "--mode",
        choices=["sheet", "per_motif"],
        default="sheet",
        help="'sheet' = 1 generation call per painting, sliced afterwards (default, cheaper). "
        "'per_motif' = 1 generation call per motif (slower, often cleaner isolated cutouts).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Target number of motifs per painting (a suggestion to the planner, not a hard cap).",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Extra free-text planning instructions.",
    )
    parser.add_argument(
        "--quality-exemplar",
        type=Path,
        default=None,
        help="Optional reference image for extraction/isolation quality.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        help="Per-motif image size (per_motif mode only).",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="high",
        help="Image quality setting passed through to the image API.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="Grid columns for the motif sheet (sheet mode only).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Force regeneration even if outputs already exist for a painting.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    batch_segment_paintings(
        paintings_dir=args.paintings_dir,
        motif_dir=args.motif_dir,
        mode=args.mode,
        count=args.count,
        instructions=args.instructions,
        quality_exemplar_path=args.quality_exemplar,
        size=args.size,
        quality=args.quality,
        motif_sheet_columns=args.columns,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
