"""Quarantine source paintings that already have extracted outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def quarantine_extracted_sources(
    extracted_dir: Path,
    source_dir: Path,
    quarantine_dir: Path,
    *,
    apply: bool = False,
) -> list[Path]:
    """Find duplicate names and move sources only with an explicit ``apply``."""
    extracted_dir = extracted_dir.expanduser().resolve()
    source_dir = source_dir.expanduser().resolve()
    quarantine_dir = quarantine_dir.expanduser().resolve()
    if not extracted_dir.is_dir() or not source_dir.is_dir():
        raise FileNotFoundError("Extracted and source directories must both exist.")
    completed = {path.name for path in extracted_dir.iterdir() if path.is_file()}
    matches = [path for path in source_dir.iterdir() if path.is_file() and path.name in completed]
    if apply:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for path in matches:
            destination = quarantine_dir / path.name
            if destination.exists():
                raise FileExistsError(f"Quarantine target already exists: {destination}")
            shutil.move(str(path), destination)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extracted_dir", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--quarantine", type=Path, default=Path("removed"))
    parser.add_argument("--apply", action="store_true", help="Move matches; default is dry-run.")
    args = parser.parse_args()
    matches = quarantine_extracted_sources(
        args.extracted_dir, args.source_dir, args.quarantine, apply=args.apply
    )
    action = "quarantined" if args.apply else "would quarantine"
    for path in matches:
        print(f"{action}: {path}")


if __name__ == "__main__":
    main()
