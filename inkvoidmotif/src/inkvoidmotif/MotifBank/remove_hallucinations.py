"""Quarantine motif files whose suffixes appear in a hallucination review set."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _review_suffix(path: Path) -> str | None:
    parts = path.name.split("_", 1)
    return parts[1] if len(parts) == 2 else None


def quarantine_hallucinations(
    review_dir: Path,
    motifs_dir: Path,
    quarantine_dir: Path,
    *,
    apply: bool = False,
) -> list[Path]:
    """Return matches and move them only when ``apply`` is explicitly true."""
    review_dir = review_dir.expanduser().resolve()
    motifs_dir = motifs_dir.expanduser().resolve()
    quarantine_dir = quarantine_dir.expanduser().resolve()
    if not review_dir.is_dir() or not motifs_dir.is_dir():
        raise FileNotFoundError("Review and motif directories must both exist.")
    reviewed = {
        suffix
        for path in review_dir.iterdir()
        if path.is_file() and (suffix := _review_suffix(path))
    }
    matches = [
        path for path in motifs_dir.iterdir() if path.is_file() and _review_suffix(path) in reviewed
    ]
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
    parser.add_argument("review_dir", type=Path)
    parser.add_argument("motifs_dir", type=Path)
    parser.add_argument("--quarantine", type=Path, default=Path("quarantined_motifs"))
    parser.add_argument("--apply", action="store_true", help="Move matches; default is dry-run.")
    args = parser.parse_args()
    matches = quarantine_hallucinations(
        args.review_dir, args.motifs_dir, args.quarantine, apply=args.apply
    )
    action = "quarantined" if args.apply else "would quarantine"
    for path in matches:
        print(f"{action}: {path}")


if __name__ == "__main__":
    main()
