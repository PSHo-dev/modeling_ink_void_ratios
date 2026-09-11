from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .image_ops import (
    crop_source_image,
    ensure_dir,
    make_contact_sheet,
    slice_motif_bank_sheet,
    slugify,
)
from .ink_ratio import control as ink_control
from .openai_api import edit_image, nvidia_api_key, plan_with_vision
from .prompts import (
    QUALITY_EXEMPLAR_NOTES,
    composition_prompt,
    extraction_prompt,
    planning_prompt,
)
from .schemas import MotifAsset, MotifSpec


def _parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = str(size).lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 1024, 1536


# Supported OpenAI images.edit canvas sizes; the mask must match the image, so
# we normalize the base/guide/mask to the nearest one of these by aspect.
_EDIT_SIZES = ((1024, 1024), (1536, 1024), (1024, 1536))

_ADD_INK_PROMPT = (
    "You are editing an existing Chinese ink-and-wash landscape painting (山水画) on silk. "
    "Repaint ONLY the transparent (masked) regions and leave every opaque area exactly as it is, "
    "blending the new edges seamlessly into the surrounding brushwork. "
    "Inside the masked regions add SOLID, DARK, near-black ink mass — rocky cliff faces, dense "
    "pine and foliage clusters, or strong structural accents — using the same brush style, ink "
    "tone, and palette already present in the painting. Make these areas genuinely dark and "
    "opaque (rich 实 ink), never a pale grey wash. Do not add text, seals, borders, signatures, "
    "or any new subject outside the masked regions."
)

_CLEAR_VOID_PROMPT = (
    "You are editing an existing Chinese ink-and-wash landscape painting (山水画) on silk. "
    "Repaint ONLY the transparent (masked) regions and leave every opaque area exactly as it is, "
    "blending the new edges seamlessly into the surrounding paper. "
    "Inside the masked regions REMOVE the existing brushwork and restore plain, empty, untouched "
    "silk/paper (留白) — clean negative space such as open sky, mist, or still water — matching the "
    "paper tone of the adjacent empty areas. Do not add any new objects, texture, wash, or text; "
    "keep these regions blank."
)

# Used by the NVIDIA composite backend: regenerate the whole painting so its
# tonal layout follows the seed guide, then we keep only the deficit regions.
_REPAINT_TO_GUIDE_PROMPT = (
    "Repaint this Chinese ink-and-wash landscape (山水画) on silk so its tonal distribution follows "
    "the attached three-tone composition guide. In the guide, pure BLACK marks where dense, dark "
    "ink masses (实) belong — rock faces, pine and foliage clusters, structural accents; pure WHITE "
    "marks blank, untouched silk (留白) — open sky, mist, water; mid GREY marks light graded wash "
    "(过渡). Keep the same subjects, brushwork, ink palette, and overall style as the painting; only "
    "redistribute ink density to match the guide. Make the dark zones genuinely dark and opaque and "
    "the white zones truly empty. No text, seals, or borders."
)


def _nearest_edit_size(width: int, height: int) -> tuple[int, int]:
    aspect = width / max(1, height)
    return min(_EDIT_SIZES, key=lambda wh: abs((wh[0] / wh[1]) - aspect))


def refine_ink_with_mask(
    *,
    image_path: Path,
    output_path: Path,
    ink_spec: dict,
    guide_path: Path | None = None,
    guide_composition: str = "high_distance_高远",
    seed: int = 0,
    refine_iters: int = 2,
    backend: str = "nvidia_composite",
    edit_model: str = "gpt-image-1",
    gen_model: str | None = None,
    gen_provider: str = "nvidia",
    quality: str = "high",
    eps: float = 0.02,
    feather: float | None = None,
) -> Path:
    """Mask-first ink-ratio control: change only the regions that are off-target.

    A genuine content-level method (no pixel remapping). The pre-generated
    three-tone guide is the *seed* of the composition and the single source of
    truth for the tonal layout:

    * **Phase 0** builds the ratio-correct guide (areas exactly on target).
    * **Phase 2** measures the base image's real band areas.
    * **Phase 3** compares the guide to the current image and derives an
      area-capped, spatially-coherent *deficit mask* for each under-filled band
      (where the guide wants ink/void but the painting isn't there yet), then
      changes only those regions and re-measures — up to ``refine_iters`` times,
      always working from the best result so far.

    Two Phase-3 backends:

    * ``openai_mask`` — true inpainting via OpenAI ``images.edit`` with the mask
      (transparent = repaint). Cleanest, but needs a working OpenAI key.
    * ``nvidia_composite`` — works with the NVIDIA gateway (no mask API):
      regenerate the whole painting conditioned on the guide, then keep only the
      deficit regions and alpha-composite them over the original (feathered to
      blend seams). The injected ink/silk is genuinely painted content.

    Every artifact (regenerations, masks, mask previews, composites, band maps,
    prompts) is kept under ``<name>.ink_refine/``; the best result becomes
    ``output_path``.
    """
    import numpy as np
    from PIL import Image  # local: only needed on this path

    from .image_ops import ensure_dir
    from .ink_ratio import mask as ink_mask

    backend = (backend or "nvidia_composite").strip().lower()
    image_path = Path(image_path)
    output_path = ensure_dir(output_path.parent) / output_path.name
    white_t = float(ink_spec["white_t"])
    dark_t = float(ink_spec["dark_t"])
    tol = float(ink_spec["tol"])
    metric = str(ink_spec.get("metric", ink_control.DEFAULT_METRIC))

    work = ensure_dir(output_path.parent / f"{output_path.stem}.ink_refine")

    base = Image.open(image_path).convert("RGB")
    if backend == "openai_mask":
        # The mask must match an allowed images.edit canvas, so snap to one.
        w, h = _nearest_edit_size(*base.size)
    else:
        # Composite backend: keep native size (cap the long edge for safety).
        w, h = base.size
        long_edge = max(w, h)
        if long_edge > 2048:
            scale = 2048 / long_edge
            w, h = round(w * scale), round(h * scale)
    if base.size != (w, h):
        print(f"[ink/mask] normalizing base {base.size[0]}x{base.size[1]} -> {w}x{h}")
        base = base.resize((w, h), Image.LANCZOS)
    size = f"{w}x{h}"
    radius = feather if feather is not None else max(2.0, min(w, h) * 0.01)

    # Phase 0: the seed mask — built first, drives everything.
    guide_out = output_path.with_suffix(".ink_guide.png")
    if guide_path is None:
        guide_path = ink_control.build_guide_file(
            ink_spec,
            guide_out,
            width=w,
            height=h,
            seed=seed,
            composition=guide_composition,
        )
    else:
        g = Image.open(Path(guide_path)).convert("RGB")
        if g.size != (w, h):
            g = g.resize((w, h), Image.LANCZOS)
        g.save(guide_out)
        guide_path = guide_out
    guide_lum = ink_mask.load_luminance(guide_path)

    history: list[dict] = []

    def _record(p: Path, edit: str, extra: dict | None = None) -> tuple:
        ratio, dev = ink_control.measure_file(p, ink_spec)
        ink_control.write_bands_file(p, p.with_suffix(".bands.png"), ink_spec)
        row = {
            "pass": len(history),
            "edit": edit,
            "image": p.name,
            "ratio": list(ratio),
            "dev": dev,
            "devs": ink_control.all_devs(ink_spec, ratio),
        }
        if extra:
            row.update(extra)
        history.append(row)
        return ratio, dev

    current = work / "pass0.png"
    base.save(current)
    ratio, dev = _record(current, "seed")
    print(
        f"[ink/mask] seed       {ink_control.fmt_ratio(ratio)}  {ink_control.fmt_devs(ink_spec, ratio)}"
    )
    best: tuple[Path, tuple, float] = (current, ratio, dev)

    n = 0
    for r in range(max(1, int(refine_iters))):
        if best[2] <= tol:
            break
        ev_void, _ev_trans, ev_ink = ink_control.band_errors(ink_spec, best[1])  # target - measured
        plan: list[tuple[str, float]] = []
        if ev_ink > eps:
            plan.append(("ink", ev_ink))
        if ev_void > eps:
            plan.append(("void", ev_void))
        if not plan:
            print("[ink/mask] no band deficit beyond eps; stopping")
            break

        cur_path = best[0]
        cur_lum = ink_mask.load_luminance(cur_path)

        if backend == "openai_mask":
            from .openai_api import edit_image_with_mask

            for band_name, deficit in plan:
                band = ink_mask.BAND_INDEX[band_name]
                sel, mask_img = ink_mask.build_edit_mask(
                    guide_lum, cur_lum, band, deficit, white_t, dark_t
                )
                if not sel.any():
                    print(f"[ink/mask] {band_name}: no deficit region in guide; skipping")
                    continue
                n += 1
                mpath = work / f"pass{n}.{band_name}_mask.png"
                ink_mask.save_mask(mask_img, mpath)
                ink_mask.mask_preview(mask_img, work / f"pass{n}.{band_name}_mask_preview.png")
                out_p = work / f"pass{n}.png"
                prompt = _ADD_INK_PROMPT if band_name == "ink" else _CLEAR_VOID_PROMPT
                write_text(out_p.with_suffix(".prompt.md"), prompt)
                try:
                    edit_image_with_mask(
                        cur_path,
                        mpath,
                        prompt,
                        out_p,
                        model=edit_model,
                        size=size,
                        quality=quality,
                    )
                except Exception as exc:  # noqa: BLE001 -- a failed candidate should not abort refinement.
                    print(f"[ink/mask] {band_name} edit failed: {exc}")
                    break
                ratio, dev = _record(
                    out_p,
                    band_name,
                    {
                        "mask": mpath.name,
                        "mask_area": float(sel.mean()),
                        "deficit": float(deficit),
                    },
                )
                tag = "  <- best" if dev < best[2] else ""
                if dev < best[2]:
                    best = (out_p, ratio, dev)
                print(
                    f"[ink/mask] {band_name:<9}{ink_control.fmt_ratio(ratio)}  "
                    f"{ink_control.fmt_devs(ink_spec, ratio)}  (mask {sel.mean() * 100:.1f}%){tag}"
                )
                cur_lum = ink_mask.load_luminance(out_p)
                if best[2] <= tol:
                    break
        else:  # nvidia_composite
            gen_path = work / f"gen{r}.png"
            gprompt = _REPAINT_TO_GUIDE_PROMPT
            if r > 0:
                gprompt += ink_control.feedback_text(ink_spec, best[1])
            write_text(gen_path.with_suffix(".prompt.md"), gprompt)
            gmodel = gen_model or default_image_model(gen_provider)
            try:
                edit_image(
                    image_paths=[cur_path, guide_path],
                    prompt=gprompt,
                    output_path=gen_path,
                    model=gmodel,
                    size=size,
                    quality=quality,
                    background="opaque",
                    output_format="png",
                    provider=gen_provider,
                )
            except Exception as exc:  # noqa: BLE001 -- keep the best completed refinement candidate.
                print(f"[ink/mask] guide-conditioned regeneration failed: {exc}")
                break
            gen_im = Image.open(gen_path).convert("RGB")
            if gen_im.size != (w, h):
                gen_im = gen_im.resize((w, h), Image.LANCZOS)
            gen_arr = np.asarray(gen_im, np.uint8)
            composed = np.asarray(Image.open(cur_path).convert("RGB"), np.uint8)

            applied: list[dict] = []
            for band_name, deficit in plan:
                band = ink_mask.BAND_INDEX[band_name]
                sel, mask_img = ink_mask.build_edit_mask(
                    guide_lum, cur_lum, band, deficit, white_t, dark_t
                )
                if not sel.any():
                    print(f"[ink/mask] {band_name}: no deficit region in guide; skipping")
                    continue
                n += 1
                mpath = work / f"pass{r}.{band_name}_mask.png"
                ink_mask.save_mask(mask_img, mpath)
                ink_mask.mask_preview(mask_img, work / f"pass{r}.{band_name}_mask_preview.png")
                alpha = ink_mask.feather_alpha(sel, radius)
                composed = ink_mask.composite(composed, gen_arr, alpha)
                applied.append(
                    {
                        "band": band_name,
                        "mask": mpath.name,
                        "mask_area": float(sel.mean()),
                        "deficit": float(deficit),
                    }
                )
            if not applied:
                break
            out_p = work / f"pass{r + 1}.png"
            Image.fromarray(composed, "RGB").save(out_p)
            ratio, dev = _record(
                out_p,
                "+".join(a["band"] for a in applied),
                {
                    "gen_image": gen_path.name,
                    "regions": applied,
                    "mask_area": sum(a["mask_area"] for a in applied),
                },
            )
            tag = "  <- best" if dev < best[2] else ""
            if dev < best[2]:
                best = (out_p, ratio, dev)
            print(
                f"[ink/mask] composite {ink_control.fmt_ratio(ratio)}  "
                f"{ink_control.fmt_devs(ink_spec, ratio)}  "
                f"(regions {sum(a['mask_area'] for a in applied) * 100:.1f}%){tag}"
            )

    best_path, best_ratio, best_dev = best
    shutil.copyfile(best_path, output_path)
    bands_path = ink_control.write_bands_file(
        output_path, output_path.with_suffix(".ink_bands.png"), ink_spec
    )

    report: dict[str, Any] = {
        "method": f"mask_first/{backend}",
        "backend": backend,
        "spec": ink_spec,
        "metric": metric,
        "target": list(ink_control.target_tuple(ink_spec)),
        "base_image": str(image_path),
        "seed_mask": str(guide_path),
        "edit_model": edit_model if backend == "openai_mask" else None,
        "gen_model": (gen_model or default_image_model(gen_provider))
        if backend == "nvidia_composite"
        else None,
        "gen_provider": gen_provider if backend == "nvidia_composite" else None,
        "passes": history,
        "work_dir": str(work),
        "generated_ratio": list(best_ratio),
        "generated_dev": best_dev,
        "generated_devs": ink_control.all_devs(ink_spec, best_ratio),
        "final_ratio": list(best_ratio),
        "final_dev": best_dev,
        "bands": str(bands_path),
        "passed": best_dev <= tol,
        "output": str(output_path),
    }
    ink_control.write_report(report, output_path.with_suffix(".ink_report.json"))

    print(f"[ink/mask] target     {ink_control.fmt_ratio(report['target'])}")
    print(
        f"[ink/mask] final      {ink_control.fmt_ratio(best_ratio)}  "
        f"{ink_control.fmt_devs(ink_spec, best_ratio)}"
    )
    print(
        f"[ink/mask] metric={metric}  dev={best_dev * 100:.1f}%  tol={tol * 100:.0f}%  "
        f"pass={report['passed']}  ({len(history) - 1} refine pass(es), backend={backend})"
    )
    print(f"[ink/mask] artifacts kept in {work}")
    return output_path


def _generate_with_ink_control(
    *,
    image_paths: list[Path],
    prompt: str,
    output_path: Path,
    model: str,
    size: str,
    quality: str,
    provider: str,
    ink_spec: dict | None,
    ink_iters: int = 3,
    ink_guide: bool = True,
    ink_revise: bool = True,
    ink_polish: bool = False,
    ink_strength: float = 0.5,
    ink_preserve_color: bool = True,
    ink_guide_composition: str = "high_distance_高远",
    seed: int = 0,
) -> Path:
    """Generate a composition, optionally steering it toward a target ink ratio.

    When ``ink_spec`` is None this is a thin pass-through to :func:`edit_image`.

    Otherwise it runs a real *measure -> critique -> regenerate* loop that
    changes the painted content (how much blank silk vs. dark ink mass), rather
    than recoloring pixels:

    1. A pre-generated three-tone guide (areas exact) is attached as a spatial
       reference (``ink_guide``).
    2. Each attempt is measured on its actual pixels. If it misses the target,
       concrete per-band corrective instructions are appended and the previous
       attempt is handed back as a revise reference (``ink_revise``).
    3. The loop stops on convergence (within ``tol``) or after ``ink_iters``
       attempts, keeping the closest attempt.

    Convergence and "best" use the spec's combined deviation metric (default:
    mean absolute error across the three bands).

    ``ink_polish`` optionally applies the deterministic tone curve to the chosen
    attempt as a cosmetic final nudge (off by default; this is the old cosmetic
    behavior and does not change composition).

    Writes ``<name>.png`` (chosen attempt), ``<name>.ink_bands.png``,
    ``<name>.ink_report.json``, ``<name>.ink_guide.png`` (when guided), every
    attempt + its prompt + band map under ``<name>.ink_iters/``, and, when
    polishing, ``<name>.prepolish.png``.
    """

    def _gen(path: Path, this_prompt: str, refs: list[Path]) -> Path:
        return edit_image(
            image_paths=refs,
            prompt=this_prompt,
            output_path=path,
            model=model,
            size=size,
            quality=quality,
            background="opaque",
            output_format="png",
            provider=provider,
        )

    if not ink_spec:
        return _gen(output_path, prompt, image_paths)

    iters = max(1, int(ink_iters))
    tol = float(ink_spec["tol"])
    metric = str(ink_spec.get("metric", ink_control.DEFAULT_METRIC))
    iters_dir = ensure_dir(output_path.parent / f"{output_path.stem}.ink_iters")

    guide_path: Path | None = None
    guide_note = ""
    if ink_guide:
        gw, gh = _parse_size(size)
        guide_path = ink_control.build_guide_file(
            ink_spec,
            output_path.with_suffix(".ink_guide.png"),
            width=gw,
            height=gh,
            seed=seed,
            composition=ink_guide_composition,
        )
        guide_note = ink_control.guide_reference_note()

    history: list[dict] = []
    best: tuple[Path, tuple, float] | None = None

    for it in range(iters):
        refs = list(image_paths)
        if guide_path is not None:
            refs.append(guide_path)
        this_prompt = prompt + guide_note
        # Anchor critique + revise to the BEST attempt so far (not just the last),
        # which prevents the model from drifting/oscillating away from a good result.
        if it > 0 and best is not None:
            this_prompt += ink_control.feedback_text(ink_spec, best[1])
            if ink_revise:
                refs.append(best[0])
                this_prompt += ink_control.revise_reference_note()

        attempt = iters_dir / f"iter{it}.png"
        write_text(attempt.with_suffix(".prompt.md"), this_prompt)
        try:
            _gen(attempt, this_prompt, refs)
        except Exception as exc:  # keep any earlier good attempt
            print(f"[ink] iteration {it + 1}/{iters} failed: {exc}")
            if best is None:
                raise
            break

        ratio, dev = ink_control.measure_file(attempt, ink_spec)
        devs = ink_control.all_devs(ink_spec, ratio)
        history.append(
            {
                "iter": it,
                "ratio": list(ratio),
                "dev": dev,
                "devs": devs,
                "image": attempt.name,
                "prompt": attempt.with_suffix(".prompt.md").name,
            }
        )
        ink_control.write_bands_file(attempt, attempt.with_suffix(".bands.png"), ink_spec)
        tag = ""
        if best is None or dev < best[2]:
            best = (attempt, ratio, dev)
            tag = "  <- best"
        print(
            f"[ink] iter {it + 1}/{iters}  {ink_control.fmt_ratio(ratio)}  "
            f"{ink_control.fmt_devs(ink_spec, ratio)}{tag}"
        )
        if dev <= tol:
            print(f"[ink] converged ({metric} dev {dev * 100:.1f}% <= tol {tol * 100:.0f}%)")
            break

    if best is None:
        raise RuntimeError("Ink-ratio generation produced no candidates.")
    best_path, best_ratio, best_dev = best

    # Promote the chosen attempt's revised-prompt sidecar (NVIDIA Responses path).
    best_revised = best_path.with_suffix(".revised.txt")
    if best_revised.exists():
        shutil.copyfile(best_revised, output_path.with_suffix(".revised.txt"))

    report: dict[str, Any] = {
        "spec": ink_spec,
        "target": list(ink_control.target_tuple(ink_spec)),
        "method": "iterative_guided_regeneration",
        "metric": metric,
        "iterations": len(history),
        "history": history,
        "iters_dir": str(iters_dir),
        "chosen_iter": int(best_path.stem.rsplit("iter", 1)[-1])
        if "iter" in best_path.stem
        else None,
        "generated_ratio": list(best_ratio),
        "generated_dev": best_dev,
        "generated_devs": ink_control.all_devs(ink_spec, best_ratio),
        "guide": str(guide_path) if guide_path else None,
        "polished": False,
        "output": str(output_path),
    }

    if ink_polish:
        # Optional cosmetic nudge (does NOT change composition). Keep the
        # pre-polish image for transparency.
        shutil.copyfile(best_path, output_path.with_suffix(".prepolish.png"))
        polish = ink_control.correct_file(
            best_path,
            output_path,
            ink_spec,
            strength=float(ink_strength),
            preserve_color=bool(ink_preserve_color),
            write_bands=True,
        )
        report["polished"] = True
        report["polish_strength"] = float(ink_strength)
        report["prepolish"] = str(output_path.with_suffix(".prepolish.png"))
        report["final_ratio"] = polish["final_ratio"]
        report["final_dev"] = polish["final_dev"]
        report["bands"] = polish["bands"]
    else:
        shutil.copyfile(best_path, output_path)
        bands_path = ink_control.write_bands_file(
            output_path, output_path.with_suffix(".ink_bands.png"), ink_spec
        )
        report["final_ratio"] = list(best_ratio)
        report["final_dev"] = best_dev
        report["bands"] = str(bands_path)

    report["passed"] = report["final_dev"] <= tol
    ink_control.write_report(report, output_path.with_suffix(".ink_report.json"))

    print(f"[ink] target    {ink_control.fmt_ratio(report['target'])}")
    print(
        f"[ink] generated {ink_control.fmt_ratio(report['generated_ratio'])}  "
        f"{ink_control.fmt_devs(ink_spec, best_ratio)}  "
        f"(best of {len(history)} iter)"
    )
    if ink_polish:
        print(f"[ink] polished  {ink_control.fmt_ratio(report['final_ratio'])}  (cosmetic)")
    print(
        f"[ink] metric={metric}  dev={report['final_dev'] * 100:.1f}%  "
        f"tol={tol * 100:.0f}%  pass={report['passed']}"
    )
    print(f"[ink] iterations kept in {iters_dir}")
    return output_path


def default_text_provider() -> str:
    load_local_env()
    return (
        os.environ.get("INKVOIDMOTIF_TEXT_PROVIDER")
        or os.environ.get("INKVOIDMOTIF_PROVIDER")
        or ("nvidia" if nvidia_api_key() else "openai")
    )


def default_image_provider() -> str:
    load_local_env()
    return (
        os.environ.get("INKVOIDMOTIF_IMAGE_PROVIDER")
        or os.environ.get("INKVOIDMOTIF_PROVIDER")
        or ("nvidia" if nvidia_api_key() else "openai")
    )


def default_text_model(provider: str | None = None) -> str:
    load_local_env()
    selected_provider = (provider or default_text_provider()).strip().lower()
    if selected_provider == "nvidia":
        return os.environ.get("NVIDIA_TEXT_MODEL", "azure/openai/gpt-5.5")
    return os.environ.get("OPENAI_TEXT_MODEL", "gpt-5.5")


def default_image_model(provider: str | None = None) -> str:
    load_local_env()
    selected_provider = (provider or default_image_provider()).strip().lower()
    if selected_provider == "nvidia":
        # On NVIDIA we route image generation through the Responses API + the
        # ``image_generation`` tool, so the "image model" we hand the SDK is
        # actually a mainline chat model slug.
        return os.environ.get(
            "NVIDIA_IMAGE_MODEL",
            os.environ.get("NVIDIA_RESPONSES_MODEL", "openai/openai/gpt-5.5"),
        )
    return os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")


def load_local_env() -> None:
    load_dotenv(override=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def write_text(path: Path, text: str) -> Path:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(text.strip() + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_motifs_from_plan(plan_path: Path) -> list[MotifSpec]:
    plan = read_json(plan_path)
    return [MotifSpec.from_dict(item) for item in plan["motifs"]]


def load_motifs_from_config(config: dict[str, Any]) -> list[MotifSpec]:
    if "motifs" not in config:
        raise ValueError(
            "Web prep needs a config with manual motifs. Add a motifs list or run planning first."
        )
    return [MotifSpec.from_dict(item) for item in config["motifs"]]


def save_manual_plan(
    source_path: Path,
    motifs: list[dict[str, Any]],
    output_dir: Path,
    summary: str = "Manual motif plan.",
    quality_exemplar_path: Path | None = None,
) -> Path:
    source_path = source_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if quality_exemplar_path:
        quality_exemplar_path = quality_exemplar_path.expanduser().resolve()
    plan_path = output_dir / "motif_plan.json"
    payload = {
        "source": str(source_path),
        "source_summary": summary,
        "motifs": motifs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if quality_exemplar_path:
        payload["quality_exemplar"] = str(quality_exemplar_path)
    return write_json(plan_path, payload)


def motif_bank_web_prompt(
    motifs: list[MotifSpec],
    source_summary: str,
    use_quality_exemplar: bool = False,
) -> str:
    motif_lines = []
    for index, motif in enumerate(motifs, start=1):
        preserve = ""
        if motif.must_preserve:
            preserve = " Preserve: " + "; ".join(motif.must_preserve) + "."
        motif_lines.append(
            f"{index}. {motif.name}: {motif.description}. Role: {motif.role or 'motif'}.{preserve}"
        )

    exemplar = (
        "\nUse the uploaded motif-bank quality exemplar only as the standard for isolation quality, layout, brushwork fidelity, and clean motif separation. Do not copy its specific objects.\n"
        f"{QUALITY_EXEMPLAR_NOTES}\n"
        if use_quality_exemplar
        else ""
    )

    return f"""
Create a motif bank image from the uploaded Chinese landscape painting.

Source summary: {source_summary}

Extract exactly these motifs:
{chr(10).join(motif_lines)}

Requirements:
- Make one clean motif-bank/contact-sheet image containing all motifs.
- Segment whole reusable objects, not rectangular crop patches.
- Preserve the original Chinese painting brushwork, pale mineral color, line density, wash edges, roof geometry, trees, rocks, water texture, and mist.
- Remove unrelated paper/silk ground, borders, seals, calligraphy, labels, and neighboring objects from each motif.
- Keep each motif's natural hand-painted contour. Avoid sticker-like hard silhouettes, white halos, and square paper blocks.
- Arrange motifs in a tidy grid with enough spacing for review.
- If labels are included, put labels outside the motif images only. Do not put text inside any motif.
- Do not invent objects that are not visible in the source painting.
{exemplar}
""".strip()


def motif_panel_generation_prompt(
    motifs: list[MotifSpec],
    source_summary: str,
    columns: int = 2,
    spacing_ratio: float = 0.225,
) -> str:
    n = len(motifs)
    rows = (n + columns - 1) // columns
    motif_lines = "\n".join(
        f"  Cell {index} (row {((index - 1) // columns) + 1}, column {((index - 1) % columns) + 1}): "
        f"{motif.name} — {motif.description}"
        for index, motif in enumerate(motifs, start=1)
    )
    margin_pct = max(0, round(spacing_ratio * 100))
    return f"""
Create a motif-bank panel from the attached Chinese landscape painting.

Source summary: {source_summary}

Extract exactly these {n} motifs from the painting and place each one in its assigned cell, preserving the original brushwork and visual details:
{motif_lines}

Panel layout requirements:
- Arrange the {n} motifs in a strict {columns}-column by {rows}-row grid, in the order listed above (row-major: left to right within each row, then top to bottom).
- Every motif occupies its own cell. No motif overlaps another, none cross cell boundaries.
- Center each motif inside its cell.
- Leave a generous margin of approximately {margin_pct}% of the cell's width as empty paper background around each motif on all four sides, so neighbouring motifs are clearly separated and never touch. The gap between any two adjacent motifs should be at least {margin_pct * 2}% of a cell's width / height in total.
- Fill the entire sheet with a plain warm off-white or soft cream paper background. No frame, no border, no decorative elements.

Strict no-text rule (very important):
- DO NOT include ANY text, numbers, captions, labels, headings, titles, watermarks, signatures, seals, or calligraphy ANYWHERE in the image.
- DO NOT number the motifs visually and DO NOT write motif names anywhere on the sheet.
- The output must be purely visual; no glyphs of any kind.

Segmentation / style:
- Preserve the original ink brushwork, dry-brush texture, soft ink washes, pale color accents, and the exact silhouette of each motif.
- Remove the original painting paper, calligraphy, seals, borders, and unrelated surroundings.
- Keep natural irregular contours and soft wash edges. Do not create hard sticker cutouts.
- Do not invent modern objects, and do not substitute generic approximations for the listed motifs.
""".strip()


def composition_web_prompt(
    motifs: list[MotifSpec],
    composition_config: dict[str, Any],
    use_quality_exemplar: bool = False,
) -> str:
    prompt = composition_prompt(
        motifs=motifs,
        theme=composition_config["theme"],
        include_scholar=bool(composition_config.get("include_scholar", False)),
        layout=composition_config.get("layout"),
        source_style=composition_config.get("source_style"),
        use_quality_exemplar=use_quality_exemplar,
    ).replace("as closely as the API allows", "as closely as ChatGPT Image allows")
    return f"""
Use the uploaded motif-bank image as the visual source for a new Chinese landscape painting.

{prompt}
""".strip()


def prepare_web_workflow(config_path: Path) -> dict[str, str]:
    config_path = config_path.expanduser().resolve()
    config = read_json(config_path)
    source_path = Path(config["source"]).expanduser().resolve()
    output_dir = Path(config.get("out_dir", "runs/default")).expanduser().resolve()
    web_dir = ensure_dir(output_dir / "web_workflow")
    crops_dir = ensure_dir(web_dir / "crops")
    prompts_dir = ensure_dir(web_dir / "prompts")

    quality_exemplar_path = None
    if config.get("quality_exemplar"):
        quality_exemplar_path = Path(config["quality_exemplar"]).expanduser().resolve()

    motifs = load_motifs_from_config(config)
    plan_path = save_manual_plan(
        source_path=source_path,
        motifs=config["motifs"],
        output_dir=output_dir,
        summary=config.get("source_summary", "Manual motif plan from config."),
        quality_exemplar_path=quality_exemplar_path,
    )

    source_copy = web_dir / source_path.name
    if source_path.exists() and source_copy != source_path:
        shutil.copy2(source_path, source_copy)

    exemplar_copy = None
    if quality_exemplar_path and quality_exemplar_path.exists():
        exemplar_copy = web_dir / quality_exemplar_path.name
        if exemplar_copy != quality_exemplar_path:
            shutil.copy2(quality_exemplar_path, exemplar_copy)

    crop_paths: list[Path] = []
    for index, spec in enumerate(motifs, start=1):
        slug = slugify(spec.name)
        crop_path = crops_dir / f"{index:02d}_{slug}.png"
        crop_source_image(
            source_path,
            spec.bbox,
            crop_path,
            margin_ratio=float(config.get("crop_margin", 0.08)),
        )
        crop_paths.append(crop_path)
        write_text(
            prompts_dir / f"{index:02d}_{slug}_single_motif_prompt.md",
            extraction_prompt(spec, use_quality_exemplar=quality_exemplar_path is not None),
        )

    crop_sheet_path = make_contact_sheet(crop_paths, web_dir / "crop_contact_sheet.png")
    motif_prompt_path = write_text(
        prompts_dir / "01_motif_bank_prompt.md",
        motif_bank_web_prompt(
            motifs=motifs,
            source_summary=config.get("source_summary", "Manual motif plan from config."),
            use_quality_exemplar=quality_exemplar_path is not None,
        ),
    )

    result = {
        "web_workflow": str(web_dir),
        "source": str(source_copy if source_copy.exists() else source_path),
        "crop_contact_sheet": str(crop_sheet_path),
        "motif_bank_prompt": str(motif_prompt_path),
        "plan": str(plan_path),
    }
    if exemplar_copy:
        result["quality_exemplar"] = str(exemplar_copy)

    composition_config = config.get("composition")
    if composition_config:
        composition_prompt_path = write_text(
            prompts_dir / "02_composition_prompt.md",
            composition_web_prompt(
                motifs=motifs,
                composition_config=composition_config,
                use_quality_exemplar=bool(composition_config.get("use_quality_exemplar", False)),
            ),
        )
        result["composition_prompt"] = str(composition_prompt_path)

    upload_steps = [f"Upload `{Path(result['source']).name}`."]
    if exemplar_copy:
        upload_steps.append(f"Upload `{exemplar_copy.name}` as the quality exemplar.")
    upload_steps.append(f"Upload `{Path(result['crop_contact_sheet']).name}` as the crop guide.")

    motif_steps = ["Open ChatGPT with image generation.", *upload_steps]
    motif_steps.extend(
        [
            "Paste `prompts/01_motif_bank_prompt.md`.",
            "Save the generated motif-bank image into this folder.",
        ]
    )
    instructions = [
        "# ChatGPT Web Workflow",
        "",
        "This folder is for the simple web-interface method. It does not call any API.",
        "",
        "## Motif Bank",
        "",
        *[f"{index}. {step}" for index, step in enumerate(motif_steps, start=1)],
    ]
    if composition_config:
        instructions.extend(
            [
                "",
                "## Recomposition",
                "",
                "1. Start a new ChatGPT image-generation message.",
                "2. Upload the motif-bank image you saved from the first step.",
                f"3. Upload `{Path(result['source']).name}` if you want the original style available too.",
                "4. Paste `prompts/02_composition_prompt.md`.",
            ]
        )
    instructions.extend(
        [
            "",
            "Optional: if one motif is weak, upload its crop from `crops/` and paste the matching single-motif prompt from `prompts/`.",
        ]
    )
    result["instructions"] = str(write_text(web_dir / "README.md", "\n".join(instructions)))
    return result


def import_chatgpt_motif_bank(
    sheet_path: Path,
    plan_path: Path,
    output_dir: Path,
    columns: int = 2,
    background_tolerance: int = 34,
    min_background_brightness: int = 198,
    remove_background: bool = False,
    image_provider: str = "chatgpt_web",
    image_model: str = "ChatGPT Image web",
    source_kind: str = "uploaded",
) -> dict[str, str]:
    sheet_path = sheet_path.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    plan = read_json(plan_path)
    motifs = load_motifs_from_plan(plan_path)
    assets_dir = ensure_dir(output_dir / "motifs")

    asset_paths = slice_motif_bank_sheet(
        sheet_path=sheet_path,
        motif_names=[motif.name for motif in motifs],
        output_dir=assets_dir,
        columns=columns,
        background_tolerance=background_tolerance,
        min_background_brightness=min_background_brightness,
        remove_background=remove_background,
    )

    import_note = (
        "Imported from a motif-bank sheet and converted to a transparent PNG."
        if remove_background
        else "Imported from a motif-bank sheet as an unmasked review cell."
    )
    assets = [
        MotifAsset(
            spec=motif,
            crop_path=None,
            asset_path=asset_path,
            prompt=import_note,
        )
        for motif, asset_path in zip(motifs, asset_paths, strict=False)
    ]

    motif_payloads = [asset.to_dict() for asset in assets]
    for payload in motif_payloads:
        asset = Path(payload["asset_path"])
        if asset.is_relative_to(output_dir):
            payload["asset_path"] = str(asset.relative_to(output_dir))

    bank = {
        "source": plan.get("source"),
        "plan_path": str(plan_path),
        "motif_sheet": str(sheet_path),
        "image_provider": image_provider,
        "image_model": image_model,
        "source_kind": source_kind,
        "background_removed": bool(remove_background),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motifs": motif_payloads,
    }
    bank_path = write_json(output_dir / "motif_bank.json", bank)
    contact_sheet_path = make_contact_sheet(asset_paths, output_dir / "motif_contact_sheet.png")
    return {
        "motif_bank": str(bank_path),
        "motifs": str(assets_dir),
        "motif_contact_sheet": str(contact_sheet_path),
    }


def generate_motif_panel(
    source_path: Path,
    plan_path: Path,
    output_path: Path,
    model: str | None = None,
    provider: str | None = None,
    size: str = "1024x1536",
    quality: str = "high",
    columns: int = 2,
    spacing_ratio: float = 0.225,
) -> Path:
    source_path = source_path.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    selected_provider = (provider or default_image_provider()).strip().lower()
    selected_model = model or default_image_model(selected_provider)
    plan = read_json(plan_path)
    motifs = load_motifs_from_plan(plan_path)
    prompt = motif_panel_generation_prompt(
        motifs=motifs,
        source_summary=str(plan.get("source_summary", "Chinese landscape painting.")),
        columns=columns,
        spacing_ratio=spacing_ratio,
    )
    write_text(output_path.with_suffix(".prompt.md"), prompt)
    return edit_image(
        image_paths=[source_path],
        prompt=prompt,
        output_path=output_path,
        model=selected_model,
        size=size,
        quality=quality,
        background="opaque",
        output_format="png",
        provider=selected_provider,
    )


def compose_from_sheet(
    sheet_path: Path,
    output_path: Path,
    theme: str,
    motifs: list[MotifSpec] | None = None,
    source_path: Path | None = None,
    image_model: str | None = None,
    image_provider: str | None = None,
    size: str = "1024x1536",
    quality: str = "high",
    include_scholar: bool = False,
    layout: str | None = None,
    source_style: str | None = None,
    quality_exemplar_path: Path | None = None,
    ink_spec: dict | None = None,
    ink_iters: int = 3,
    ink_guide: bool = True,
    ink_revise: bool = True,
    ink_polish: bool = False,
    ink_strength: float = 0.5,
    ink_preserve_color: bool = True,
    ink_guide_composition: str = "high_distance_高远",
    seed: int = 0,
) -> Path:
    """Recompose a new Chinese landscape painting from a motif-bank sheet.

    This matches the collaborator's web workflow: the motif sheet itself is the
    single visual reference, and the model is asked to recombine those motifs
    into a new painting. Works through gpt-image-2 chat-completions on NVIDIA
    or through the OpenAI Images edit API on OpenAI.

    When ``ink_spec`` is provided, the target tonal balance (void/transition/ink)
    is enforced by an iterative measure->critique->regenerate loop steered by a
    pre-generated composition guide; see :func:`_generate_with_ink_control`.
    """

    sheet_path = sheet_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if source_path:
        source_path = source_path.expanduser().resolve()
    if quality_exemplar_path:
        quality_exemplar_path = quality_exemplar_path.expanduser().resolve()
    provider = (image_provider or default_image_provider()).strip().lower()
    model = image_model or default_image_model(provider)

    references: list[Path] = [sheet_path]
    if quality_exemplar_path and quality_exemplar_path != sheet_path:
        references.append(quality_exemplar_path)
    if source_path:
        references.append(source_path)

    prompt = composition_prompt(
        motifs=motifs or [],
        theme=theme,
        include_scholar=include_scholar,
        layout=layout,
        source_style=source_style,
        use_quality_exemplar=quality_exemplar_path is not None,
        ink_spec=ink_spec,
        ink_guide=False,  # guide note is added per-iteration by the control loop
    )
    write_text(output_path.with_suffix(".prompt.md"), prompt)

    return _generate_with_ink_control(
        image_paths=references,
        prompt=prompt,
        output_path=output_path,
        model=model,
        size=size,
        quality=quality,
        provider=provider,
        ink_spec=ink_spec,
        ink_iters=ink_iters,
        ink_guide=ink_guide,
        ink_revise=ink_revise,
        ink_polish=ink_polish,
        ink_strength=ink_strength,
        ink_preserve_color=ink_preserve_color,
        ink_guide_composition=ink_guide_composition,
        seed=seed,
    )


def plan_motifs(
    source_path: Path,
    output_dir: Path,
    count: int,
    model: str | None = None,
    provider: str | None = None,
    instructions: str | None = None,
    quality_exemplar_path: Path | None = None,
) -> Path:
    source_path = source_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if quality_exemplar_path:
        quality_exemplar_path = quality_exemplar_path.expanduser().resolve()
    ensure_dir(output_dir)
    prompt = planning_prompt(
        count=count,
        instructions=instructions,
        use_quality_exemplar=quality_exemplar_path is not None,
    )
    plan = plan_with_vision(
        source_path=source_path,
        prompt=prompt,
        model=model or default_text_model(provider),
        quality_exemplar_path=quality_exemplar_path,
        provider=provider or default_text_provider(),
    )
    if isinstance(plan, list):
        plan = {"motifs": plan}
    elif not isinstance(plan, dict):
        raise RuntimeError(
            f"Planning model returned {type(plan).__name__}; expected a dict with a 'motifs' field."
        )
    if "motifs" not in plan and any(key in plan for key in ("name", "description", "role")):
        plan = {"motifs": [plan]}
    plan["source"] = str(source_path)
    if quality_exemplar_path:
        plan["quality_exemplar"] = str(quality_exemplar_path)
    plan["created_at"] = datetime.now(timezone.utc).isoformat()
    plan["planning_prompt"] = prompt
    return write_json(output_dir / "motif_plan.json", plan)


def extract_motifs(
    source_path: Path,
    plan_path: Path,
    output_dir: Path,
    image_model: str | None = None,
    image_provider: str | None = None,
    size: str = "1024x1024",
    quality: str = "high",
    crop_margin: float = 0.08,
    limit: int | None = None,
    include_source_reference: bool = False,
    quality_exemplar_path: Path | None = None,
    skip_existing: bool = True,
) -> Path:
    source_path = source_path.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if quality_exemplar_path:
        quality_exemplar_path = quality_exemplar_path.expanduser().resolve()
    provider = image_provider or default_image_provider()
    model = image_model or default_image_model(provider)
    motifs = load_motifs_from_plan(plan_path)
    if limit is not None:
        motifs = motifs[:limit]

    crops_dir = ensure_dir(output_dir / "crops")
    assets_dir = ensure_dir(output_dir / "motifs")
    assets: list[MotifAsset] = []

    for index, spec in enumerate(motifs, start=1):
        slug = slugify(spec.name)
        crop_path = crops_dir / f"{index:02d}_{slug}.png"
        asset_path = assets_dir / f"{index:02d}_{slug}.png"
        crop_source_image(source_path, spec.bbox, crop_path, margin_ratio=crop_margin)

        prompt = extraction_prompt(spec, use_quality_exemplar=quality_exemplar_path is not None)
        if not (skip_existing and asset_path.exists()):
            references = [crop_path]
            if include_source_reference:
                references.append(source_path)
            if quality_exemplar_path:
                references.append(quality_exemplar_path)
            edit_image(
                image_paths=references,
                prompt=prompt,
                output_path=asset_path,
                model=model,
                size=size,
                quality=quality,
                background="transparent",
                output_format="png",
                provider=provider,
            )

        assets.append(
            MotifAsset(spec=spec, crop_path=crop_path, asset_path=asset_path, prompt=prompt)
        )

    bank = {
        "source": str(source_path),
        "plan_path": str(plan_path),
        "image_provider": provider,
        "image_model": model,
        "size": size,
        "quality": quality,
        "quality_exemplar": str(quality_exemplar_path) if quality_exemplar_path else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motifs": [asset.to_dict() for asset in assets],
    }
    bank_path = write_json(output_dir / "motif_bank.json", bank)
    make_contact_sheet(
        [asset.asset_path for asset in assets], output_dir / "motif_contact_sheet.png"
    )
    return bank_path


def motif_specs_from_bank(bank: dict[str, Any]) -> list[MotifSpec]:
    return [MotifSpec.from_dict(item) for item in bank["motifs"]]


def compose_from_bank(
    bank_path: Path,
    output_dir: Path,
    theme: str,
    image_model: str | None = None,
    image_provider: str | None = None,
    size: str = "1024x1536",
    quality: str = "high",
    include_scholar: bool = False,
    layout: str | None = None,
    source_style: str | None = None,
    quality_exemplar_path: Path | None = None,
    name: str = "composition",
    max_references: int = 16,
    ink_spec: dict | None = None,
    ink_iters: int = 3,
    ink_guide: bool = True,
    ink_revise: bool = True,
    ink_polish: bool = False,
    ink_strength: float = 0.5,
    ink_preserve_color: bool = True,
    ink_guide_composition: str = "high_distance_高远",
    seed: int = 0,
) -> Path:
    bank_path = bank_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if quality_exemplar_path:
        quality_exemplar_path = quality_exemplar_path.expanduser().resolve()
    bank = read_json(bank_path)
    provider = image_provider or bank.get("image_provider") or default_image_provider()
    motifs = motif_specs_from_bank(bank)
    motif_paths = []
    for item in bank["motifs"]:
        motif_path = Path(item["asset_path"]).expanduser()
        if not motif_path.is_absolute():
            motif_path = (bank_path.parent / motif_path).resolve()
        motif_paths.append(motif_path)
    reference_budget = max_references - 1 if quality_exemplar_path else max_references
    if len(motif_paths) > reference_budget:
        motif_paths = motif_paths[:reference_budget]
        motifs = motifs[:reference_budget]

    out = ensure_dir(output_dir / "compositions") / f"{slugify(name)}.png"
    composition_references = list(motif_paths)
    if quality_exemplar_path:
        composition_references = [quality_exemplar_path, *motif_paths]

    prompt = composition_prompt(
        motifs=motifs,
        theme=theme,
        include_scholar=include_scholar,
        layout=layout,
        source_style=source_style,
        use_quality_exemplar=quality_exemplar_path is not None,
        ink_spec=ink_spec,
        ink_guide=False,  # guide note is added per-iteration by the control loop
    )
    _generate_with_ink_control(
        image_paths=composition_references,
        prompt=prompt,
        output_path=out,
        model=image_model or default_image_model(provider),
        size=size,
        quality=quality,
        provider=provider,
        ink_spec=ink_spec,
        ink_iters=ink_iters,
        ink_guide=ink_guide,
        ink_revise=ink_revise,
        ink_polish=ink_polish,
        ink_strength=ink_strength,
        ink_preserve_color=ink_preserve_color,
        ink_guide_composition=ink_guide_composition,
        seed=seed,
    )

    write_json(
        out.with_suffix(".json"),
        {
            "motif_bank": str(bank_path),
            "image_provider": provider,
            "image_model": image_model or default_image_model(provider),
            "size": size,
            "quality": quality,
            "theme": theme,
            "include_scholar": include_scholar,
            "layout": layout,
            "source_style": source_style,
            "quality_exemplar": str(quality_exemplar_path) if quality_exemplar_path else None,
            "prompt": prompt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output": str(out),
        },
    )
    return out


_LAYOUT_RENDER_PROMPT = (
    "Paint ONE unified, coherent Chinese ink-and-wash landscape (山水画) on silk, following the "
    "attached composition map, where the SHADED areas mark painted landmass and the LIGHT areas stay "
    "as open silk. "
    "Compose with the THREE DISTANCES (三远): a grounded foreground, a middle distance, and a far peak "
    "set higher in the frame. DISTRIBUTE these elements across the height of the painting — do NOT pack "
    "everything into one compact lump at the bottom. Separate the distance planes with bands of mist, "
    "water, and open silk so the scene recedes naturally, yet still reads as ONE believable place: link "
    "the planes with atmosphere and terrain, with NO isolated cut-outs hovering arbitrarily. "
    "Exercise 留白 restraint: keep large connected areas of PURE BLANK silk (open sky, mist, water); use "
    "wash sparingly unless required by the theme; do NOT flood the sky or far distance with rows of extra hills. "
    "Keep the overall balance of painted landmass to empty silk close to the map. "
    "Use the supplied motifs as the visual vocabulary, redrawn and integrated into the scene rather than "
    "pasted separately. Consistent brushwork and ink palette; no text, seals, or borders."
)

_LAYOUT_RENDER_PROMPT_LOOSE = (
    "Paint ONE unified, coherent Chinese ink-and-wash landscape (山水画) on silk, loosely following the "
    "attached composition map (shaded = painted landmass, light = open silk). Build a single connected, "
    "believable scene with a clear foreground, middle distance, and far distance — no isolated cut-outs "
    "floating in space. Use the supplied motifs as the visual vocabulary, redrawn and integrated rather "
    "than pasted. Consistent brushwork and ink palette; no text, seals, or borders."
)


def compose_with_layout(
    *,
    bank_path: Path,
    output_path: Path,
    theme: str,
    target_void: float,
    convention: str = "top_sky",
    peak_scale: float = 1.0,
    spread: float = 0.6,
    seed: int = 0,
    restraint: bool = True,
    image_model: str | None = None,
    image_provider: str | None = None,
    size: str = "1024x1536",
    quality: str = "high",
    max_motifs: int = 12,
    preview_only: bool = False,
    source_path: Path | None = None,
    source_style: str | None = None,
    include_scholar: bool = False,
) -> Path:
    """Compose a real painting with a controlled void / landmass proportion.

    Instead of collaging motifs (which yields floating fragments), this builds a
    *coherent* composition map — one connected landmass anchored at the bottom
    with a dominant peak, and one connected open-sky 留白 sized to
    ``target_void`` — then generates a single unified landscape into the mass
    while leaving the sky blank. The motif bank supplies the visual vocabulary.
    Realized void is measured as a region (so paper between brushstrokes counts
    as landmass, not void) and reported. See :mod:`inkvoidmotif.layout`.
    """
    from . import layout as comp_layout

    bank_path = Path(bank_path).expanduser().resolve()
    output_path = ensure_dir(output_path.parent) / output_path.name
    bank = read_json(bank_path)
    provider = image_provider or bank.get("image_provider") or default_image_provider()
    model = image_model or default_image_model(provider)
    w, h = _parse_size(size)

    work = ensure_dir(output_path.parent / f"{output_path.stem}.layout")
    lay = comp_layout.build_mass_void_layout(
        w,
        h,
        target_void,
        convention=convention,
        peak_scale=peak_scale,
        seed=seed,
        spread=spread,
    )
    map_path = comp_layout.render_layout_preview(
        lay, w, h, work / "composition_map.png", label=True
    )
    guide_path = comp_layout.render_layout_preview(
        lay, w, h, work / "composition_guide.png", label=False
    )

    print(
        f"[layout] composition map: void {lay['achieved_void'] * 100:.0f}% open silk / "
        f"landmass {(1 - lay['achieved_void']) * 100:.0f}%  ({convention}, peak x{peak_scale:.1f}, "
        f"spread {spread:.1f})"
    )
    print(f"[layout] map: {map_path}")

    if preview_only:
        write_json(
            output_path.with_suffix(".layout_report.json"),
            {
                "method": "composition_layout/preview",
                "convention": convention,
                "target_void": float(target_void),
                "achieved_void": lay["achieved_void"],
                "peak_scale": float(peak_scale),
                "spread": float(spread),
                "composition_map": str(map_path),
                "composition_guide": str(guide_path),
            },
        )
        return map_path

    motif_paths: list[Path] = []
    for item in bank["motifs"]:
        asset = Path(item["asset_path"]).expanduser()
        if not asset.is_absolute():
            asset = (bank_path.parent / asset).resolve()
        if asset.exists():
            motif_paths.append(asset)
    motif_paths = motif_paths[:max_motifs]

    refs = [guide_path, *motif_paths]
    if source_path is not None:
        source_path = Path(source_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Source style reference not found: {source_path}")
        refs.append(source_path)
    render_prompt = _LAYOUT_RENDER_PROMPT if restraint else _LAYOUT_RENDER_PROMPT_LOOSE
    prompt_parts = [f"Theme: {theme}.", render_prompt]
    if include_scholar:
        prompt_parts.append(
            "Include one small, naturally scaled scholar as a quiet human focal point."
        )
    if source_style:
        prompt_parts.append(f"Style direction: {source_style.strip()}")
    if source_path is not None:
        prompt_parts.append(
            "The final reference image is the source painting: borrow its brush language, "
            "paper tone, and palette without copying its composition."
        )
    prompt = "\n\n".join(prompt_parts)
    write_text(output_path.with_suffix(".prompt.md"), prompt)
    edit_image(
        image_paths=refs,
        prompt=prompt,
        output_path=output_path,
        model=model,
        size=size,
        quality=quality,
        background="opaque",
        output_format="png",
        provider=provider,
    )

    import numpy as np
    from PIL import Image

    land_full = np.asarray(Image.open(guide_path).convert("L").resize((w, h)))
    # intended sky = light region of the guide; intended landmass = shaded region
    motif_region = land_full < 210
    occ = comp_layout.measure_occupancy(output_path, motif_region=motif_region)
    comp_layout.occupancy_preview(
        occ["mask"], output_path.with_suffix(".occupancy.png"), motif_region
    )
    rv = comp_layout.region_void(output_path)

    report = {
        "method": "composition_layout",
        "control": "void/landmass proportion via coherent composition map",
        "theme": theme,
        "size": size,
        "convention": convention,
        "peak_scale": float(peak_scale),
        "spread": float(spread),
        "restraint": bool(restraint),
        "include_scholar": bool(include_scholar),
        "source_style": source_style.strip() if source_style else None,
        "source_reference": str(source_path) if source_path is not None else None,
        "image_provider": provider,
        "image_model": model,
        "target_void": float(target_void),
        "map_void": lay["achieved_void"],
        "realized_void_region": rv["void"],
        "verify": {
            "sky_kept_blank": occ.get("void_kept_blank"),
            "sky_intruded": occ.get("void_intruded"),
            "landmass_filled": occ.get("motif_region_filled"),
            "ink_footprint": occ["ink_coverage"],
        },
        "composition_map": str(map_path),
        "composition_guide": str(guide_path),
        "occupancy_preview": str(output_path.with_suffix(".occupancy.png")),
        "output": str(output_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_path.with_suffix(".layout_report.json"), report)

    print(
        f"[layout] target void {target_void * 100:.0f}%  ->  realized void {rv['void'] * 100:.0f}% "
        f"(region)  |  sky kept blank {occ['void_kept_blank'] * 100:.0f}%  |  "
        f"landmass filled {occ['motif_region_filled'] * 100:.0f}%"
    )
    print(f"[layout] output {output_path}")
    return output_path


WEB_WORKFLOW_ALIASES = {"web", "web-pack", "web_pack", "chatgpt-web"}


def run_config(config_path: Path) -> dict[str, str]:
    config_path = config_path.expanduser().resolve()
    config = read_json(config_path)
    workflow = str(config.get("workflow", "")).strip().lower()

    source_path = Path(config["source"]).expanduser().resolve()
    output_dir = Path(config.get("out_dir", "runs/default")).expanduser().resolve()
    quality_exemplar_path = None
    if config.get("quality_exemplar"):
        quality_exemplar_path = Path(config["quality_exemplar"]).expanduser().resolve()
    text_provider = config.get("text_provider") or config.get("provider") or default_text_provider()
    image_provider = (
        config.get("image_provider") or config.get("provider") or default_image_provider()
    )
    ensure_dir(output_dir)

    if "motifs" in config:
        plan_path = save_manual_plan(
            source_path=source_path,
            motifs=config["motifs"],
            output_dir=output_dir,
            summary=config.get("source_summary", "Manual motif plan from config."),
            quality_exemplar_path=quality_exemplar_path,
        )
    else:
        plan_path = plan_motifs(
            source_path=source_path,
            output_dir=output_dir,
            count=int(config.get("count", 8)),
            model=config.get("text_model") or default_text_model(text_provider),
            provider=text_provider,
            instructions=config.get("planning_instructions"),
            quality_exemplar_path=quality_exemplar_path,
        )

    if workflow in WEB_WORKFLOW_ALIASES and not config.get("motif_sheet"):
        return prepare_web_workflow(config_path)

    # Resolve / generate the motif sheet -----------------------------------
    sheet_path: Path | None = None
    sheet_was_generated = False
    if config.get("motif_sheet"):
        sheet_path = Path(config["motif_sheet"]).expanduser().resolve()
    elif config.get("mode", "sheet").lower() == "sheet":
        sheet_path = output_dir / "motif_sheet.png"
        if not (sheet_path.exists() and bool(config.get("skip_existing", True))):
            generate_motif_panel(
                source_path=source_path,
                plan_path=plan_path,
                output_path=sheet_path,
                model=config.get("image_model") or default_image_model(image_provider),
                provider=image_provider,
                size=config.get("motif_sheet_size", "1024x1536"),
                quality=config.get("quality", "high"),
                columns=int(config.get("motif_sheet_columns", 2)),
                spacing_ratio=float(config.get("motif_sheet_spacing_ratio", 0.225)),
            )
            sheet_was_generated = True

    result: dict[str, str] = {"plan": str(plan_path)}

    if sheet_path is not None:
        slice_info = import_chatgpt_motif_bank(
            sheet_path=sheet_path,
            plan_path=plan_path,
            output_dir=output_dir,
            columns=int(config.get("motif_sheet_columns", 2)),
            background_tolerance=int(config.get("motif_sheet_background_tolerance", 34)),
            min_background_brightness=int(config.get("motif_sheet_min_background_brightness", 198)),
            remove_background=bool(config.get("motif_sheet_remove_background", False)),
        )
        result.update(slice_info)
        result["motif_sheet"] = str(sheet_path)
        if sheet_was_generated:
            result["motif_sheet_prompt"] = str(sheet_path.with_suffix(".prompt.md"))

        if workflow in WEB_WORKFLOW_ALIASES:
            web_result = prepare_web_workflow(config_path)
            result["web_workflow"] = web_result["web_workflow"]
            result["motif_bank_prompt"] = web_result["motif_bank_prompt"]
    else:
        # Legacy per-motif extraction path: only used when the user explicitly
        # opts in with `mode: "per-motif"` (slow, one API call per motif).
        bank_path = extract_motifs(
            source_path=source_path,
            plan_path=plan_path,
            output_dir=output_dir,
            image_model=config.get("image_model") or default_image_model(image_provider),
            image_provider=image_provider,
            size=config.get("motif_size", "1024x1024"),
            quality=config.get("quality", "high"),
            crop_margin=float(config.get("crop_margin", 0.08)),
            include_source_reference=bool(config.get("include_source_reference", False)),
            quality_exemplar_path=quality_exemplar_path,
            skip_existing=bool(config.get("skip_existing", True)),
        )
        result["motif_bank"] = str(bank_path)

    # Composition ----------------------------------------------------------
    composition_config = config.get("composition")
    if composition_config:
        comp_name = slugify(composition_config.get("name", "composition"))
        comp_path = ensure_dir(output_dir / "compositions") / f"{comp_name}.png"
        composition_exists = comp_path.exists() and bool(config.get("skip_existing", True))

        # Optional ink-ratio control: a per-composition block overrides a
        # top-level one. spec_from_config returns None when disabled/missing.
        ink_cfg = composition_config.get("ink_ratio") or config.get("ink_ratio")
        ink_spec = ink_control.spec_from_config(ink_cfg)
        ink_cfg = ink_cfg or {}
        # ``iters`` is the new key; tolerate the legacy ``candidates`` key.
        iters = int(ink_cfg.get("iters", ink_cfg.get("candidates", 3)))
        ink_kwargs: dict[str, Any] = {
            "ink_spec": ink_spec,
            "ink_iters": iters,
            "ink_guide": bool(ink_cfg.get("guide", True)),
            "ink_revise": bool(ink_cfg.get("revise", True)),
            "ink_polish": bool(ink_cfg.get("polish", ink_cfg.get("correct", False))),
            "ink_strength": float(ink_cfg.get("strength", 0.5)),
            "ink_preserve_color": bool(ink_cfg.get("preserve_color", True)),
            "ink_guide_composition": str(ink_cfg.get("guide_composition", "high_distance_高远")),
            "seed": int(ink_cfg.get("seed", 0)),
        }

        if composition_exists:
            result["composition"] = str(comp_path)
        elif sheet_path is not None:
            motifs_for_prompt = load_motifs_from_plan(plan_path)
            compose_from_sheet(
                sheet_path=sheet_path,
                output_path=comp_path,
                theme=composition_config["theme"],
                motifs=motifs_for_prompt,
                source_path=source_path
                if composition_config.get("include_source_reference")
                else None,
                image_model=config.get("image_model") or default_image_model(image_provider),
                image_provider=image_provider,
                size=composition_config.get("size", "1024x1536"),
                quality=composition_config.get("quality", config.get("quality", "high")),
                include_scholar=bool(composition_config.get("include_scholar", False)),
                layout=composition_config.get("layout"),
                source_style=composition_config.get("source_style"),
                quality_exemplar_path=quality_exemplar_path
                if composition_config.get("use_quality_exemplar", False)
                else None,
                **ink_kwargs,
            )
            result["composition"] = str(comp_path)
            if ink_spec is not None:
                result["ink_report"] = str(comp_path.with_suffix(".ink_report.json"))
        elif "motif_bank" in result:
            comp_out = compose_from_bank(
                bank_path=Path(result["motif_bank"]),
                output_dir=output_dir,
                theme=composition_config["theme"],
                image_model=config.get("image_model") or default_image_model(image_provider),
                image_provider=image_provider,
                size=composition_config.get("size", "1024x1536"),
                quality=composition_config.get("quality", config.get("quality", "high")),
                include_scholar=bool(composition_config.get("include_scholar", False)),
                layout=composition_config.get("layout"),
                source_style=composition_config.get("source_style"),
                quality_exemplar_path=quality_exemplar_path
                if composition_config.get("use_quality_exemplar", False)
                else None,
                name=composition_config.get("name", "composition"),
                **ink_kwargs,
            )
            result["composition"] = str(comp_out)
            if ink_spec is not None:
                result["ink_report"] = str(comp_out.with_suffix(".ink_report.json"))

    return result
