from __future__ import annotations

import argparse
from pathlib import Path

from .ink_ratio import (
    correct_file as ink_correct_file,
)
from .ink_ratio import (
    fmt_devs as ink_fmt_devs,
)
from .ink_ratio import (
    fmt_ratio as ink_fmt_ratio,
)
from .ink_ratio import (
    make_ink_spec,
)
from .ink_ratio import (
    measure_file as ink_measure_file,
)
from .ink_ratio import (
    write_report as ink_write_report,
)
from .pipeline import (
    compose_from_bank,
    compose_from_sheet,
    compose_with_layout,
    default_image_provider,
    default_text_provider,
    extract_motifs,
    generate_motif_panel,
    import_chatgpt_motif_bank,
    plan_motifs,
    prepare_web_workflow,
    refine_ink_with_mask,
    run_config,
)


def _add_ink_args(p: argparse.ArgumentParser) -> None:
    """Shared ink-ratio flags for the generation commands (compose/recompose).

    The mechanism is an iterative measure -> critique -> regenerate loop that
    genuinely changes the painted content; it does NOT recolor pixels.
    """
    p.add_argument(
        "--ink-ratio",
        metavar="VOID,TRANSITION,INK",
        help="Target tonal balance, e.g. 0.6,0.25,0.15 (auto-normalized). Enables ink-ratio control.",
    )
    p.add_argument(
        "--ink-iters",
        type=int,
        default=3,
        help="Max measure->critique->regenerate iterations; stops early on convergence (default 3). Each iteration is one image-generation call.",
    )
    p.add_argument(
        "--no-ink-guide",
        action="store_true",
        help="Do not attach the pre-generated three-tone composition guide.",
    )
    p.add_argument(
        "--no-ink-revise",
        action="store_true",
        help="Regenerate from scratch each iteration instead of revising the previous attempt.",
    )
    p.add_argument(
        "--ink-polish",
        action="store_true",
        help="Apply a gentle cosmetic tone curve to the final pick (does NOT change composition; off by default).",
    )
    p.add_argument(
        "--ink-strength",
        type=float,
        default=0.5,
        help="Polish tone-curve strength 0..1 (only used with --ink-polish; default 0.5).",
    )
    p.add_argument(
        "--ink-no-preserve-color",
        action="store_true",
        help="Polish in monochrome instead of preserving hue.",
    )
    p.add_argument(
        "--ink-guide-composition",
        default="high_distance_高远",
        help="Composition prior for the guide (e.g. high_distance_高远, level_distance_平远, deep_distance_深远, river/valley).",
    )
    p.add_argument(
        "--ink-seed",
        type=int,
        default=0,
        help="Seed for the composition guide layout (default 0).",
    )
    p.add_argument(
        "--ink-metric",
        default="mean",
        choices=["mean", "max", "sum", "tvd"],
        help="How the combined deviation is aggregated across the 3 bands (default mean = combined deviation).",
    )
    p.add_argument(
        "--ink-white-t",
        type=float,
        default=0.72,
        help="Luminance above which a pixel counts as void (default 0.72).",
    )
    p.add_argument(
        "--ink-dark-t",
        type=float,
        default=0.28,
        help="Luminance below which a pixel counts as ink (default 0.28).",
    )
    p.add_argument(
        "--ink-tol",
        type=float,
        default=0.05,
        help="Convergence/pass deviation tolerance on the chosen metric (default 0.05).",
    )


def _parse_ratio(text: str) -> tuple[float, float, float]:
    parts = [p for p in str(text).replace(" ", "").split(",") if p != ""]
    if len(parts) != 3:
        raise SystemExit(
            "ink ratio must be three numbers: void,transition,ink (e.g. 0.6,0.25,0.15)"
        )
    try:
        v, t, i = (float(x) for x in parts)
    except ValueError as exc:
        raise SystemExit(f"invalid ink ratio {text!r}: {exc}") from exc
    return v, t, i


def _ink_kwargs_from_args(args: argparse.Namespace) -> dict:
    spec = None
    if getattr(args, "ink_ratio", None):
        v, t, i = _parse_ratio(args.ink_ratio)
        spec = make_ink_spec(
            v,
            t,
            i,
            white_t=args.ink_white_t,
            dark_t=args.ink_dark_t,
            tol=args.ink_tol,
            metric=args.ink_metric,
        )
    return {
        "ink_spec": spec,
        "ink_iters": args.ink_iters,
        "ink_guide": not args.no_ink_guide,
        "ink_revise": not args.no_ink_revise,
        "ink_polish": args.ink_polish,
        "ink_strength": args.ink_strength,
        "ink_preserve_color": not args.ink_no_preserve_color,
        "ink_guide_composition": args.ink_guide_composition,
        "seed": args.ink_seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inkvoidmotif",
        description="Prepare Chinese painting motif workflows for ChatGPT web or API generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Analyze a source painting and write motif_plan.json.")
    plan.add_argument("--source", required=True, type=Path)
    plan.add_argument("--out", required=True, type=Path)
    plan.add_argument("--count", type=int, default=8)
    plan.add_argument("--text-model")
    plan.add_argument(
        "--text-provider", default=default_text_provider(), choices=["openai", "nvidia"]
    )
    plan.add_argument("--instructions")
    plan.add_argument("--quality-exemplar", type=Path)

    extract = sub.add_parser("extract", help="Create transparent motif assets from a motif plan.")
    extract.add_argument("--source", required=True, type=Path)
    extract.add_argument("--plan", required=True, type=Path)
    extract.add_argument("--out", required=True, type=Path)
    extract.add_argument("--image-model")
    extract.add_argument(
        "--image-provider",
        default=default_image_provider(),
        choices=["openai", "nvidia"],
    )
    extract.add_argument("--size", default="1024x1024")
    extract.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"])
    extract.add_argument("--crop-margin", type=float, default=0.08)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--include-source-reference", action="store_true")
    extract.add_argument("--quality-exemplar", type=Path)
    extract.add_argument("--overwrite", action="store_true")

    compose = sub.add_parser("compose", help="Recompose a new painting from motif_bank.json.")
    compose.add_argument("--bank", required=True, type=Path)
    compose.add_argument("--out", required=True, type=Path)
    compose.add_argument("--theme", required=True)
    compose.add_argument("--image-model")
    compose.add_argument(
        "--image-provider",
        default=default_image_provider(),
        choices=["openai", "nvidia"],
    )
    compose.add_argument("--size", default="1024x1536")
    compose.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"])
    compose.add_argument("--include-scholar", action="store_true")
    compose.add_argument("--layout")
    compose.add_argument("--source-style")
    compose.add_argument("--quality-exemplar", type=Path)
    compose.add_argument("--name", default="composition")
    _add_ink_args(compose)

    run = sub.add_parser("run", help="Run plan, extract, and optional compose from a JSON config.")
    run.add_argument("config", type=Path)

    web_pack = sub.add_parser(
        "web-pack",
        help="Prepare crops and copy-ready prompts for the ChatGPT web image workflow.",
    )
    web_pack.add_argument("config", type=Path)

    import_bank = sub.add_parser(
        "import-bank",
        help="Slice a ChatGPT Image motif-bank sheet into transparent PNG motif assets.",
    )
    import_bank.add_argument("--sheet", required=True, type=Path)
    import_bank.add_argument("--plan", required=True, type=Path)
    import_bank.add_argument("--out", required=True, type=Path)
    import_bank.add_argument("--columns", type=int, default=2)
    import_bank.add_argument("--background-tolerance", type=int, default=34)
    import_bank.add_argument("--min-background-brightness", type=int, default=198)
    import_bank.add_argument(
        "--remove-background",
        action="store_true",
        help="Flood-fill the paper background to alpha. Off by default; the slices look like the sheet cells.",
    )

    sheet = sub.add_parser(
        "sheet",
        help="Generate a motif-bank sheet from a source painting via gpt-image-2 chat completions.",
    )
    sheet.add_argument("--source", required=True, type=Path)
    sheet.add_argument("--plan", required=True, type=Path)
    sheet.add_argument(
        "--out", required=True, type=Path, help="Output PNG path for the motif sheet."
    )
    sheet.add_argument("--image-model")
    sheet.add_argument(
        "--image-provider",
        default=default_image_provider(),
        choices=["openai", "nvidia"],
    )
    sheet.add_argument("--size", default="1024x1536")
    sheet.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"])
    sheet.add_argument("--columns", type=int, default=2)
    sheet.add_argument(
        "--spacing-ratio",
        type=float,
        default=0.225,
        help="Margin of empty paper around each motif as a fraction of cell width (default 0.225).",
    )

    recompose = sub.add_parser(
        "recompose",
        help="Generate a new Chinese landscape painting from an existing motif-bank sheet.",
    )
    recompose.add_argument("--sheet", required=True, type=Path)
    recompose.add_argument(
        "--out", required=True, type=Path, help="Output PNG path for the new painting."
    )
    recompose.add_argument("--theme", required=True)
    recompose.add_argument(
        "--plan",
        type=Path,
        help="Optional motif_plan.json to embed motif names in the prompt.",
    )
    recompose.add_argument(
        "--source",
        type=Path,
        help="Optional source painting to also include as a reference.",
    )
    recompose.add_argument("--image-model")
    recompose.add_argument(
        "--image-provider",
        default=default_image_provider(),
        choices=["openai", "nvidia"],
    )
    recompose.add_argument("--size", default="1024x1536")
    recompose.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"])
    recompose.add_argument("--include-scholar", action="store_true")
    recompose.add_argument("--layout")
    recompose.add_argument("--source-style")
    recompose.add_argument("--quality-exemplar", type=Path)
    _add_ink_args(recompose)

    ink_measure = sub.add_parser(
        "ink-measure",
        help="Measure the realized void/transition/ink ratio of an existing painting.",
    )
    ink_measure.add_argument("--image", required=True, type=Path)
    ink_measure.add_argument(
        "--ink-ratio",
        metavar="VOID,TRANSITION,INK",
        help="Optional target to compare against, e.g. 0.6,0.25,0.15.",
    )
    ink_measure.add_argument("--metric", default="mean", choices=["mean", "max", "sum", "tvd"])
    ink_measure.add_argument("--white-t", type=float, default=0.72)
    ink_measure.add_argument("--dark-t", type=float, default=0.28)
    ink_measure.add_argument("--tol", type=float, default=0.05)

    ink_correct = sub.add_parser(
        "ink-correct",
        help="Snap an existing painting's tonal balance to a target ratio with a tone curve.",
    )
    ink_correct.add_argument("--image", required=True, type=Path)
    ink_correct.add_argument("--out", required=True, type=Path)
    ink_correct.add_argument(
        "--ink-ratio",
        required=True,
        metavar="VOID,TRANSITION,INK",
        help="Target tonal balance, e.g. 0.6,0.25,0.15.",
    )
    ink_correct.add_argument("--metric", default="mean", choices=["mean", "max", "sum", "tvd"])
    ink_correct.add_argument("--strength", type=float, default=1.0)
    ink_correct.add_argument("--no-preserve-color", action="store_true")
    ink_correct.add_argument(
        "--no-bands",
        action="store_true",
        help="Do not write the band-visualization PNG.",
    )
    ink_correct.add_argument("--white-t", type=float, default=0.72)
    ink_correct.add_argument("--dark-t", type=float, default=0.28)
    ink_correct.add_argument("--tol", type=float, default=0.05)

    ink_refine = sub.add_parser(
        "ink-refine",
        help="Mask-first ink-ratio control: repaint only the off-target regions of an existing painting (OpenAI only).",
    )
    ink_refine.add_argument("--image", required=True, type=Path, help="Base painting to refine.")
    ink_refine.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output PNG path for the refined painting.",
    )
    ink_refine.add_argument(
        "--ink-ratio",
        required=True,
        metavar="VOID,TRANSITION,INK",
        help="Target tonal balance, e.g. 0.6,0.25,0.15.",
    )
    ink_refine.add_argument(
        "--guide",
        type=Path,
        help="Optional pre-built seed mask; otherwise one is built from the ratio.",
    )
    ink_refine.add_argument(
        "--ink-guide-composition",
        default="high_distance_高远",
        help="Composition prior for the seed mask when none is supplied.",
    )
    ink_refine.add_argument(
        "--ink-seed",
        type=int,
        default=0,
        help="Seed for the composition guide layout (default 0).",
    )
    ink_refine.add_argument(
        "--refine-iters",
        type=int,
        default=2,
        help="Max measure->mask->repaint rounds; stops early on convergence (default 2).",
    )
    ink_refine.add_argument(
        "--backend",
        default="nvidia_composite",
        choices=["nvidia_composite", "openai_mask"],
        help="nvidia_composite (default): regenerate conditioned on the guide + composite deficit regions (works on NVIDIA). "
        "openai_mask: true masked inpaint via OpenAI images.edit (needs an OpenAI key).",
    )
    ink_refine.add_argument(
        "--gen-provider",
        default="nvidia",
        choices=["openai", "nvidia"],
        help="Provider for the guide-conditioned regeneration (nvidia_composite backend; default nvidia).",
    )
    ink_refine.add_argument(
        "--gen-model",
        help="Override the regeneration image model (nvidia_composite backend).",
    )
    ink_refine.add_argument(
        "--edit-model",
        default="gpt-image-1",
        help="OpenAI image-edit model that supports masks (openai_mask backend; default gpt-image-1).",
    )
    ink_refine.add_argument(
        "--feather",
        type=float,
        help="Seam feather radius in px for composite blending (default ~1%% of the short edge).",
    )
    ink_refine.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"])
    ink_refine.add_argument("--metric", default="mean", choices=["mean", "max", "sum", "tvd"])
    ink_refine.add_argument("--white-t", type=float, default=0.72)
    ink_refine.add_argument("--dark-t", type=float, default=0.28)
    ink_refine.add_argument("--tol", type=float, default=0.05)

    layout = sub.add_parser(
        "layout",
        help="Composition control: build a coherent landmass/open-sky map at a target void, then paint one unified landscape into it.",
        description="Build a coherent composition map (landmass vs. open-silk 留白) sized to a target "
        "void and distributed across the three distances, then paint ONE unified landscape "
        "into it. The void proportion and the spatial distribution are controllable while "
        "the output stays a real painting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_io = layout.add_argument_group("inputs / output")
    g_io.add_argument(
        "--bank",
        required=True,
        type=Path,
        help="motif_bank.json (supplies the motif vocabulary).",
    )
    g_io.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output PNG path for the rendered painting.",
    )
    g_io.add_argument(
        "--theme",
        required=True,
        help="Scene description only, e.g. 'autumn river valley with light mist' "
        "(composition rules are built in; no need to hand-write them).",
    )

    g_comp = layout.add_argument_group(
        "composition control",
        "Dial the void proportion and how the landmass is arranged.",
    )
    g_comp.add_argument(
        "--void",
        type=float,
        required=True,
        help="Target void ratio (open silk / 留白), e.g. 0.62. Landmass = 1 - void.",
    )
    g_comp.add_argument(
        "--convention",
        default="top_sky",
        choices=["top_sky", "diagonal", "river_band"],
        help="Where the coherent void sits (top_sky: open sky above, landmass below).",
    )
    g_comp.add_argument(
        "--peak-scale",
        type=float,
        default=0.62,
        help="Height of the dominant mountain mass (1.0 = tall; lower frees up sky).",
    )
    g_comp.add_argument(
        "--spread",
        type=float,
        default=0.6,
        help="0 = one compact mass low in the frame; higher distributes the landmass "
        "across near/middle/far distance planes (three distances 三远).",
    )
    g_comp.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the composition map (changes mass placement / which side the far peak leans).",
    )
    g_comp.add_argument(
        "--loose",
        action="store_true",
        help="Disable the built-in 留白 restraint + three-distances instructions for a freer "
        "composition. Void will track the target less tightly.",
    )

    g_gen = layout.add_argument_group("generation")
    g_gen.add_argument(
        "--image-model",
        help="Override the image model (else the bank/provider default).",
    )
    g_gen.add_argument(
        "--image-provider",
        default=default_image_provider(),
        choices=["openai", "nvidia"],
        help="Image backend.",
    )
    g_gen.add_argument("--size", default="1024x1536", help="Output size WxH.")
    g_gen.add_argument(
        "--quality",
        default="high",
        choices=["low", "medium", "high", "auto"],
        help="Generation quality.",
    )
    g_gen.add_argument(
        "--max-motifs",
        type=int,
        default=12,
        help="Max number of motif assets passed as the visual vocabulary.",
    )

    g_wf = layout.add_argument_group("workflow")
    g_wf.add_argument(
        "--preview-only",
        action="store_true",
        help="Only build + save the composition map for approval; do not generate the painting.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "plan":
        path = plan_motifs(
            source_path=args.source,
            output_dir=args.out,
            count=args.count,
            model=args.text_model,
            provider=args.text_provider,
            instructions=args.instructions,
            quality_exemplar_path=args.quality_exemplar,
        )
        print(path)
        return

    if args.command == "extract":
        path = extract_motifs(
            source_path=args.source,
            plan_path=args.plan,
            output_dir=args.out,
            image_model=args.image_model,
            image_provider=args.image_provider,
            size=args.size,
            quality=args.quality,
            crop_margin=args.crop_margin,
            limit=args.limit,
            include_source_reference=args.include_source_reference,
            quality_exemplar_path=args.quality_exemplar,
            skip_existing=not args.overwrite,
        )
        print(path)
        return

    if args.command == "compose":
        path = compose_from_bank(
            bank_path=args.bank,
            output_dir=args.out,
            theme=args.theme,
            image_model=args.image_model,
            image_provider=args.image_provider,
            size=args.size,
            quality=args.quality,
            include_scholar=args.include_scholar,
            layout=args.layout,
            source_style=args.source_style,
            quality_exemplar_path=args.quality_exemplar,
            name=args.name,
            **_ink_kwargs_from_args(args),
        )
        print(path)
        return

    if args.command == "run":
        result = run_config(args.config)
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    if args.command == "web-pack":
        result = prepare_web_workflow(args.config)
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    if args.command == "import-bank":
        result = import_chatgpt_motif_bank(
            sheet_path=args.sheet,
            plan_path=args.plan,
            output_dir=args.out,
            columns=args.columns,
            background_tolerance=args.background_tolerance,
            min_background_brightness=args.min_background_brightness,
            remove_background=args.remove_background,
        )
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    if args.command == "sheet":
        path = generate_motif_panel(
            source_path=args.source,
            plan_path=args.plan,
            output_path=args.out,
            model=args.image_model,
            provider=args.image_provider,
            size=args.size,
            quality=args.quality,
            columns=args.columns,
            spacing_ratio=args.spacing_ratio,
        )
        print(path)
        return

    if args.command == "recompose":
        motifs = None
        if args.plan:
            from .pipeline import load_motifs_from_plan

            motifs = load_motifs_from_plan(args.plan)
        path = compose_from_sheet(
            sheet_path=args.sheet,
            output_path=args.out,
            theme=args.theme,
            motifs=motifs,
            source_path=args.source,
            image_model=args.image_model,
            image_provider=args.image_provider,
            size=args.size,
            quality=args.quality,
            include_scholar=args.include_scholar,
            layout=args.layout,
            source_style=args.source_style,
            quality_exemplar_path=args.quality_exemplar,
            **_ink_kwargs_from_args(args),
        )
        print(path)
        return

    if args.command == "ink-measure":
        if args.ink_ratio:
            v, t, i = _parse_ratio(args.ink_ratio)
            spec = make_ink_spec(
                v,
                t,
                i,
                white_t=args.white_t,
                dark_t=args.dark_t,
                tol=args.tol,
                metric=args.metric,
            )
        else:
            spec = make_ink_spec(
                1 / 3,
                1 / 3,
                1 / 3,
                white_t=args.white_t,
                dark_t=args.dark_t,
                tol=args.tol,
                metric=args.metric,
            )
        ratio, dev = ink_measure_file(args.image, spec)
        print(f"measured : {ink_fmt_ratio(ratio)}")
        if args.ink_ratio:
            print(f"target   : {ink_fmt_ratio((spec['void'], spec['transition'], spec['ink']))}")
            print(f"deviation: {ink_fmt_devs(spec, ratio)}")
            print(f"{args.metric} dev = {dev * 100:.1f}%   pass={dev <= spec['tol']}")
        return

    if args.command == "ink-correct":
        v, t, i = _parse_ratio(args.ink_ratio)
        spec = make_ink_spec(
            v,
            t,
            i,
            white_t=args.white_t,
            dark_t=args.dark_t,
            tol=args.tol,
            metric=args.metric,
        )
        report = ink_correct_file(
            args.image,
            args.out,
            spec,
            strength=args.strength,
            preserve_color=not args.no_preserve_color,
            write_bands=not args.no_bands,
        )
        report_path = ink_write_report(report, args.out.with_suffix(".ink_report.json"))
        print(f"target : {ink_fmt_ratio(report['target'])}")
        print(f"raw    : {ink_fmt_ratio(report['raw_ratio'])}  dev={report['raw_dev'] * 100:.1f}%")
        print(
            f"final  : {ink_fmt_ratio(report['final_ratio'])}  dev={report['final_dev'] * 100:.1f}%  pass={report['passed']}"
        )
        print(f"output : {report['output']}")
        if report.get("bands"):
            print(f"bands  : {report['bands']}")
        print(f"report : {report_path}")
        return

    if args.command == "ink-refine":
        v, t, i = _parse_ratio(args.ink_ratio)
        spec = make_ink_spec(
            v,
            t,
            i,
            white_t=args.white_t,
            dark_t=args.dark_t,
            tol=args.tol,
            metric=args.metric,
        )
        path = refine_ink_with_mask(
            image_path=args.image,
            output_path=args.out,
            ink_spec=spec,
            guide_path=args.guide,
            guide_composition=args.ink_guide_composition,
            seed=args.ink_seed,
            refine_iters=args.refine_iters,
            backend=args.backend,
            edit_model=args.edit_model,
            gen_model=args.gen_model,
            gen_provider=args.gen_provider,
            quality=args.quality,
            feather=args.feather,
        )
        print(path)
        return

    if args.command == "layout":
        path = compose_with_layout(
            bank_path=args.bank,
            output_path=args.out,
            theme=args.theme,
            target_void=args.void,
            convention=args.convention,
            peak_scale=args.peak_scale,
            spread=args.spread,
            seed=args.seed,
            restraint=not args.loose,
            preview_only=args.preview_only,
            image_model=args.image_model,
            image_provider=args.image_provider,
            size=args.size,
            quality=args.quality,
            max_motifs=args.max_motifs,
        )
        print(path)
        return

    parser.error("Unknown command")
