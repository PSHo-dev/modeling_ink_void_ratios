"""High-level ink-ratio control glue for the inkvoidmotif pipeline.

Bridges the vendored pure-numpy tonal core (:mod:`tonal`, :mod:`guide`) to
inkvoidmotif's file-based composition step. Provides:

* :func:`make_ink_spec` / :func:`spec_from_config` — build/validate a target spec
* :func:`measure_file` — measure the realized (void, transition, ink) ratio of a PNG
* :func:`choose_best` — best-of-N selection by closeness to the target ratio
* :func:`correct_file` — deterministically snap a painting's ratio to target
* :func:`build_guide_file` — write a composition guide PNG for soft steering

The two enforcement mechanisms mirror the standalone project:

1. *Soft* — pick the best of several candidates (and optionally a guide reference).
2. *Hard* — a monotonic, structure-preserving tone curve that makes the realized
   band areas equal the target exactly under the fixed thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from . import guide as _guide
from . import tonal

# Defaults match the standalone project. void/transition/ink are area fractions.
DEFAULT_WHITE_T = 0.72
DEFAULT_DARK_T = 0.28
DEFAULT_TOL = 0.05
# How a single "deviation" number is computed from the three per-band errors.
# "mean" = average |error| across the 3 bands (the combined deviation, default),
# "max"  = worst single band (strict, the old behavior),
# "sum"  = total |error|, "tvd" = total-variation distance = 0.5 * sum.
DEFAULT_METRIC = "mean"


def make_ink_spec(
    void: float,
    transition: float,
    ink: float,
    white_t: float = DEFAULT_WHITE_T,
    dark_t: float = DEFAULT_DARK_T,
    tol: float = DEFAULT_TOL,
    metric: str = DEFAULT_METRIC,
    normalize: bool = True,
) -> dict[str, Any]:
    if normalize:
        void, transition, ink = tonal.normalize_ratios(void, transition, ink)
    return {
        "void": float(void),
        "transition": float(transition),
        "ink": float(ink),
        "white_t": float(white_t),
        "dark_t": float(dark_t),
        "tol": float(tol),
        "metric": str(metric),
    }


def band_errors(spec: dict[str, Any], measured) -> tuple[float, float, float]:
    """Signed per-band errors (target - measured) for void, transition, ink."""
    tv, tt, ti = target_tuple(spec)
    return (tv - float(measured[0]), tt - float(measured[1]), ti - float(measured[2]))


def deviation_metric(spec: dict[str, Any], measured, metric: str | None = None) -> float:
    """Single combined-deviation number under the chosen aggregation."""
    e = [abs(x) for x in band_errors(spec, measured)]
    m = (metric or spec.get("metric") or DEFAULT_METRIC).lower()
    if m == "max":
        return max(e)
    if m == "sum":
        return sum(e)
    if m == "tvd":
        return 0.5 * sum(e)
    return sum(e) / 3.0  # mean


def all_devs(spec: dict[str, Any], measured) -> dict[str, float]:
    e = [abs(x) for x in band_errors(spec, measured)]
    return {"max": max(e), "mean": sum(e) / 3.0, "sum": sum(e), "tvd": 0.5 * sum(e)}


def fmt_devs(spec: dict[str, Any], measured) -> str:
    d = all_devs(spec, measured)
    return f"mean={d['mean'] * 100:4.1f}%  max={d['max'] * 100:4.1f}%  sum={d['sum'] * 100:4.1f}%"


def spec_from_config(cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build an ink spec from an ``ink_ratio`` config block, or None if disabled.

    Recognized keys: void, transition, ink (required to enable), white_t, dark_t,
    tol, plus an optional ``enabled`` flag. Returns None when the block is missing,
    ``enabled`` is false, or the three target fractions are not all present.
    """
    if not cfg or not isinstance(cfg, dict):
        return None
    if cfg.get("enabled") is False:
        return None
    if not all(k in cfg for k in ("void", "transition", "ink")):
        return None
    return make_ink_spec(
        void=float(cfg["void"]),
        transition=float(cfg["transition"]),
        ink=float(cfg["ink"]),
        white_t=float(cfg.get("white_t", DEFAULT_WHITE_T)),
        dark_t=float(cfg.get("dark_t", DEFAULT_DARK_T)),
        tol=float(cfg.get("tol", DEFAULT_TOL)),
        metric=str(cfg.get("metric", DEFAULT_METRIC)),
    )


def target_tuple(spec: dict[str, Any]) -> tuple[float, float, float]:
    return (spec["void"], spec["transition"], spec["ink"])


def _load(path: Path):
    return tonal.from_pil(Image.open(Path(path)))


def measure_array(img, spec: dict[str, Any]):
    """Return ((void, transition, ink), deviation) for an HxWx3 float image.

    ``deviation`` uses the spec's combined metric (default: mean abs error).
    """
    ratio = tonal.measure_ratios(tonal.luminance(img), spec["white_t"], spec["dark_t"])
    dev = deviation_metric(spec, ratio)
    return ratio, dev


def measure_file(path: Path, spec: dict[str, Any]):
    """Return ((void, transition, ink), max_deviation) for an image on disk."""
    return measure_array(_load(path), spec)


def choose_best(paths: list[Path], spec: dict[str, Any]):
    """Pick the candidate whose measured ratio is closest to target.

    Returns (best_path, best_ratio, best_dev).
    """
    best = None
    for p in paths:
        ratio, dev = measure_file(p, spec)
        if best is None or dev < best[2]:
            best = (Path(p), ratio, dev)
    if best is None:
        raise ValueError("choose_best requires at least one candidate path")
    return best


def correct_array(img, spec: dict[str, Any], strength: float = 1.0, preserve_color: bool = True):
    """Apply the deterministic tone curve so band areas equal the target.

    Returns (corrected_image, realized_ratio, realized_deviation).
    """
    target = target_tuple(spec)
    curve = tonal.fit_band_curve(tonal.luminance(img), *target, spec["white_t"], spec["dark_t"])
    out = tonal.apply_curve(img, curve, float(strength), bool(preserve_color))
    ratio, dev = measure_array(out, spec)
    return out, ratio, dev


def correct_file(
    in_path: Path,
    out_path: Path,
    spec: dict[str, Any],
    strength: float = 1.0,
    preserve_color: bool = True,
    write_bands: bool = True,
) -> dict[str, Any]:
    """Correct a painting on disk; write the corrected PNG (and optional band map).

    Returns a report dict with target/raw/final ratios and deviations.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = _load(in_path)
    raw_ratio, raw_dev = measure_array(img, spec)
    corrected, final_ratio, final_dev = correct_array(
        img, spec, strength=strength, preserve_color=preserve_color
    )
    tonal.save_png(corrected, str(out_path))
    bands_path = None
    if write_bands:
        bands_path = out_path.with_suffix(".ink_bands.png")
        bands = tonal.band_visualization(
            tonal.luminance(corrected), spec["white_t"], spec["dark_t"]
        )
        tonal.save_png(bands, str(bands_path))
    return {
        "spec": spec,
        "target": list(target_tuple(spec)),
        "raw_ratio": list(raw_ratio),
        "raw_dev": raw_dev,
        "final_ratio": list(final_ratio),
        "final_dev": final_dev,
        "passed": final_dev <= spec["tol"],
        "strength": float(strength),
        "preserve_color": bool(preserve_color),
        "source": str(in_path),
        "output": str(out_path),
        "bands": str(bands_path) if bands_path else None,
    }


def write_bands_file(in_path: Path, out_path: Path, spec: dict[str, Any]) -> Path:
    """Write a void/transition/ink band-visualization PNG for an image on disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bands = tonal.band_visualization(
        tonal.luminance(_load(in_path)), spec["white_t"], spec["dark_t"]
    )
    tonal.save_png(bands, str(out_path))
    return out_path


def build_guide_file(
    spec: dict[str, Any],
    out_path: Path,
    *,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    composition: str = "high_distance_高远",
    bias_strength: float = 0.55,
    softness: float = 0.06,
) -> Path:
    """Write a composition guide PNG whose band areas hit the target exactly."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = _guide.build_guide(
        spec,
        width,
        height,
        seed,
        composition,
        bias_strength=bias_strength,
        softness=softness,
    )
    tonal.save_png(arr, str(out_path))
    return out_path


def write_report(report: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def fmt_ratio(ratio) -> str:
    return (
        f"void={ratio[0] * 100:5.1f}%  transition={ratio[1] * 100:5.1f}%  "
        f"ink={ratio[2] * 100:5.1f}%"
    )


def guide_reference_note() -> str:
    """Prompt fragment telling the model how to read the attached guide image."""
    return (
        "\nOne of the reference images is an abstract three-tone composition guide: "
        "WHITE = leave as blank paper/silk (留白), GREY = mid-tone wash and mist (过渡), "
        "BLACK = dense dark ink mass (实). Follow its large-scale distribution of empty "
        "versus dense regions — match where the blank areas and the dark masses sit and "
        "roughly how much space each occupies — but paint real motifs with brushwork; "
        "do not copy the guide's shapes literally or render it as flat tone.\n"
    )


def feedback_text(spec: dict[str, Any], measured, eps: float = 0.02) -> str:
    """Concrete corrective instructions from the measured per-band deficit.

    ``measured`` is the realized (void, transition, ink) of the previous attempt.
    Returns painterly directions to push the *content* toward the target — more or
    less blank silk, more or less dark ink mass — not a request to recolor.

    Designed to counter the observed failure mode where "add more ink" makes the
    model flood the page with flat mid-tone wash and lose the blank paper: it
    states all three targets explicitly, tells the model which on-target bands to
    PRESERVE, and where to take area from when adding ink.
    """
    tv, tt, ti = target_tuple(spec)
    mv, mt, mi = float(measured[0]), float(measured[1]), float(measured[2])
    dv, dt, di = tv - mv, tt - mt, ti - mi

    directives: list[str] = []
    keep: list[str] = []

    # Ink mass is the strongest compositional lever, so lead with it.
    if di > eps:
        # Where should the new ink come from? Prefer eating mid-tone wash, not void.
        source = (
            "the mid-tone wash areas" if mt > tt + eps else "the least important parts of the scene"
        )
        directives.append(
            f"- Add MORE solid dark ink mass: about {ti * 100:.0f}% of the surface should be "
            f"deep near-black brushwork (rock faces, dense pine/foliage, structures); it was "
            f"only {mi * 100:.0f}%. Paint it as a few COMPACT, high-contrast dark masses, NOT "
            f"as thin grey shading. Take this area from {source}."
        )
    elif di < -eps:
        directives.append(
            f"- Reduce the dark ink mass to about {ti * 100:.0f}% of the surface (it was "
            f"{mi * 100:.0f}%): fewer or smaller dark masses."
        )
    else:
        keep.append(f"dark ink mass (~{mi * 100:.0f}%, on target)")

    if dv > eps:
        directives.append(
            f"- Leave MORE blank untouched paper/silk: about {tv * 100:.0f}% of the surface "
            f"must stay bright and empty (sky, water, mist); it was only {mv * 100:.0f}%. "
            f"Concentrate motifs to one side / corner and let large areas breathe."
        )
    elif dv < -eps:
        directives.append(
            f"- Reduce blank paper to about {tv * 100:.0f}% (it was {mv * 100:.0f}%) — but do "
            f"this by adding DARK ink masses, not flat grey wash."
        )
    else:
        keep.append(f"blank paper (~{mv * 100:.0f}%, on target — DO NOT cover it)")

    if dt > eps:
        directives.append(
            f"- Slightly more mid-tone wash/mist (about {tt * 100:.0f}%; it was {mt * 100:.0f}%)."
        )
    elif dt < -eps:
        directives.append(
            f"- Use much LESS flat mid-tone grey wash: about {tt * 100:.0f}% (it was a heavy "
            f"{mt * 100:.0f}%). Replace excess wash with either crisp blank paper or solid dark ink."
        )
    else:
        keep.append(f"mid-tone wash (~{mt * 100:.0f}%, on target)")

    if not directives:
        return ""

    keep_line = ("\n- PRESERVE what is already correct: " + "; ".join(keep) + ".") if keep else ""
    return (
        "\nTonal revision — the previous attempt's tonal balance must be corrected:\n"
        f"- Previous attempt: void {mv * 100:.0f}%, transition {mt * 100:.0f}%, ink {mi * 100:.0f}%. "
        f"TARGET: void {tv * 100:.0f}%, transition {tt * 100:.0f}%, ink {ti * 100:.0f}%.\n"
        + "\n".join(directives)
        + keep_line
        + "\n- Think of the page as three zones: large bright EMPTY paper, a little soft mist, "
        "and a few DENSE black ink masses. Avoid an even, all-over grey tone.\n"
        "- Keep the same motifs, brushwork identity, and palette; change only how the area is "
        "divided between blank paper, wash, and dark ink.\n"
    )


def revise_reference_note() -> str:
    """Prompt fragment for the img2img-style revise pass (previous attempt attached)."""
    return (
        "\nThe most recent reference image is your previous attempt at this painting. "
        "Treat it as the base to revise: preserve its motifs, layout identity, and "
        "brushwork, and adjust only the tonal balance as instructed above.\n"
    )
