from __future__ import annotations

import argparse
import fnmatch
from datetime import datetime, timezone
from pathlib import Path

from inkvoidmotif.ink_ratio import control as ink_control
from inkvoidmotif.pipeline import (
    compose_with_layout,
    default_image_provider,
    refine_ink_with_mask,
    write_json,
)

# ... (keep select_motif_files / build_motif_bank_from_folder unchanged)
"""
python recompose_with_void_control_new.py \
    --motif-dir runs/default/motifs \
    --motif-pattern "autumn_*" \
    --out-dir   runs/autumn_preview \
    --theme     "maples over a mountain stream" \
    --void-target 0.4 \
    --preview-only

python recompose_with_void_control_new.py \
    --motif-dir runs/default/motifs \
    --motif-pattern "summer_*" \
    --out-dir   runs/summer01 \
    --theme     "misty river gorge in high summer" \
    --void-target 0.5 --ink-target 0.3 \
    --convention top_sky --spread 0.7 \
    --skip-mask-refine
"""


def select_motif_files(
    motif_dir: Path,
    pattern: str = "*",
    explicit_names: list[str] | None = None,
) -> list[Path]:
    """
    Pick which motif PNGs in motif_dir to include, by either:
      - explicit_names: an exact list of filenames or stems (e.g.
        ["winter_pine_01", "winter_rock_03.png"]) -- only these are used, and
        an error is raised if any are missing, so typos don't silently drop
        a motif you wanted.
      - pattern: a glob applied to the filename stem (e.g. "winter_*" picks
        winter_pine_01.png, winter_rock_03.png, ... but not spring_lake.png).
    explicit_names takes priority over pattern when both are given.
    """
    motif_dir = motif_dir.expanduser().resolve()
    all_files = sorted(motif_dir.glob("*.png"))
    if not all_files:
        raise FileNotFoundError(f"No .png files found in {motif_dir}")

    if explicit_names:
        wanted = {n[:-4] if n.lower().endswith(".png") else n for n in explicit_names}
        by_stem = {f.stem: f for f in all_files}
        missing = wanted - by_stem.keys()
        if missing:
            raise FileNotFoundError(
                f"Requested motif(s) not found in {motif_dir}: {sorted(missing)}. "
                f"Available: {sorted(by_stem.keys())}"
            )
        selected = [by_stem[n] for n in wanted]
    else:
        selected = [f for f in all_files if fnmatch.fnmatch(f.stem, pattern)]
        if not selected:
            raise FileNotFoundError(
                f"No motifs in {motif_dir} matched pattern '{pattern}'. "
                f"Available: {[f.stem for f in all_files]}"
            )

    return sorted(selected)


def build_motif_bank_from_folder(
    motif_dir: Path,
    bank_path: Path,
    image_provider: str | None = None,
    pattern: str = "*",
    explicit_names: list[str] | None = None,
) -> Path:
    """
    Wrap a folder of already-segmented motif PNGs into the motif_bank.json
    shape that compose_from_bank() expects. Use this when your motifs came
    from a separate segmentation step rather than extract_motifs() /
    import_chatgpt_motif_bank().

    Only motifs matching `pattern` (or, if given, exactly listed in
    `explicit_names`) are included in the bank -- so, e.g., a folder mixing
    winter_*, spring_*, summer_* motifs can be narrowed to just one season.
    """
    motif_dir = motif_dir.expanduser().resolve()
    bank_path = bank_path.expanduser().resolve()
    files = select_motif_files(motif_dir, pattern=pattern, explicit_names=explicit_names)
    print(f"      selected {len(files)} motif(s): {[f.stem for f in files]}")

    motifs = []
    for f in files:
        name = f.stem.replace("_", " ").strip()
        motifs.append(
            {
                "name": name,
                "description": name,
                "role": None,
                "must_preserve": [],
                # VERIFY: if MotifSpec.bbox is a required (non-Optional) field in
                # schemas.py, replace None with a placeholder like [0, 0, 0, 0] --
                # it's unused once you're past the crop stage, but the dataclass
                # may still demand a value.
                "bbox": None,
                "asset_path": str(f),
            }
        )

    bank = {
        "source": None,
        "plan_path": None,
        "image_provider": image_provider or default_image_provider(),
        "image_model": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motifs": motifs,
    }
    return write_json(bank_path, bank)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motif-dir", required=True, type=Path)
    parser.add_argument("--motif-pattern", default="*")
    parser.add_argument("--motif-names", default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--theme", required=True)
    parser.add_argument("--name", default="composition")
    parser.add_argument("--size", default="1024x1536")
    parser.add_argument("--quality", default="high")
    parser.add_argument("--image-provider", default=None)
    parser.add_argument("--image-model", default=None)

    # Void/transition/ink target -- void_target drives compose_with_layout's
    # geometry; the full 3-band spec is still needed for the mask-refine stage.
    parser.add_argument("--void-target", type=float, default=0.45)
    parser.add_argument("--transition-target", type=float, default=None)
    parser.add_argument("--ink-target", type=float, default=0.35)
    parser.add_argument("--tol", type=float, default=0.06)
    parser.add_argument("--white-t", type=float, default=0.72)
    parser.add_argument("--dark-t", type=float, default=0.28)
    parser.add_argument("--guide-composition", default="high_distance_高远")
    parser.add_argument("--seed", type=int, default=0)

    # Layout-specific controls (replace --iters).
    parser.add_argument(
        "--convention", default="top_sky", help="Composition-map convention, e.g. 'top_sky'."
    )
    parser.add_argument(
        "--peak-scale", type=float, default=1.0, help="Relative size of the dominant far peak."
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=0.6,
        help="How far the landmass/void spreads across the frame.",
    )
    parser.add_argument(
        "--max-motifs", type=int, default=12, help="Cap on motif references passed to the render."
    )
    parser.add_argument(
        "--no-restraint",
        action="store_true",
        help="Use the looser render prompt instead of the strict 三远/留白-restraint one.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only render the composition map/guide; skip generation.",
    )

    # Stage 2: mask-based refinement.
    parser.add_argument(
        "--mask-backend", choices=["nvidia_composite", "openai_mask"], default="nvidia_composite"
    )
    parser.add_argument("--mask-refine-iters", type=int, default=2)
    parser.add_argument("--skip-mask-refine", action="store_true")

    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "motif_bank.json"

    print(f"[1/3] Building motif bank from {args.motif_dir} -> {bank_path}")
    explicit_names = (
        [n.strip() for n in args.motif_names.split(",") if n.strip()] if args.motif_names else None
    )
    build_motif_bank_from_folder(
        args.motif_dir,
        bank_path,
        image_provider=args.image_provider,
        pattern=args.motif_pattern,
        explicit_names=explicit_names,
    )

    transition_target = args.transition_target
    if transition_target is None:
        transition_target = 1.0 - args.void_target - args.ink_target
        print(f"      --transition-target not given; derived as {transition_target:.3f}")

    # Still needed for stage 2 (refine_ink_with_mask), even though
    # compose_with_layout itself only uses --void-target.
    ink_cfg = {
        "void": args.void_target,
        "transition": transition_target,
        "ink": args.ink_target,
        "tol": args.tol,
        "white_t": args.white_t,
        "dark_t": args.dark_t,
        "metric": "mean",
    }
    ink_spec = ink_control.spec_from_config(ink_cfg)
    if ink_spec is None:
        raise RuntimeError("spec_from_config() returned None; check config values.")

    print(
        f"[2/3] Composing with layout control (void target {args.void_target:.2f}, "
        f"convention={args.convention})"
    )
    comp_path = out_dir / "compositions" / f"{args.name}.png"
    composed_path = compose_with_layout(
        bank_path=bank_path,
        output_path=comp_path,
        theme=args.theme,
        target_void=args.void_target,
        convention=args.convention,
        peak_scale=args.peak_scale,
        spread=args.spread,
        seed=args.seed,
        restraint=not args.no_restraint,
        image_model=args.image_model,
        image_provider=args.image_provider,
        size=args.size,
        quality=args.quality,
        max_motifs=args.max_motifs,
        preview_only=args.preview_only,
    )
    print(f"      -> {composed_path}")

    if args.preview_only:
        print("\nPreview-only: composition map/guide rendered, no painting generated.")
        return

    if args.skip_mask_refine:
        print("[3/3] Skipping mask-based refinement (--skip-mask-refine).")
        print(f"\nDone. Final painting: {composed_path}")
        return

    print(f"[3/3] Mask-based void-ratio refinement (backend={args.mask_backend})")
    refined_path = out_dir / "compositions" / f"{args.name}.refined.png"
    final_path = refine_ink_with_mask(
        image_path=composed_path,
        output_path=refined_path,
        ink_spec=ink_spec,
        guide_composition=args.guide_composition,
        seed=args.seed,
        refine_iters=args.mask_refine_iters,
        backend=args.mask_backend,
        gen_provider=args.image_provider or default_image_provider(),
        quality=args.quality,
    )
    print(f"      -> {final_path}")
    print(f"\nDone. Final painting: {final_path}")
    print(
        f"Reports: {composed_path.with_suffix('.layout_report.json')}, "
        f"{final_path.with_suffix('.ink_report.json')}"
    )


if __name__ == "__main__":
    main()
