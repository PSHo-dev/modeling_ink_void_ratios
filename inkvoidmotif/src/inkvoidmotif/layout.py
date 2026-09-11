"""Composition-level control: motif sizes and total void space.

This is *not* a tonal/luminance ratio. The thing being controlled is the
**composition**: how much of the canvas each motif occupies (its size) and,
as a direct consequence, how much is left as empty silk (留白 = void).

Each motif in the bank carries a normalized ``bbox`` ``[x, y, w, h]``. We treat
a motif's *size* as its bounding-box area (the user's chosen definition) and the
*void* as the canvas area covered by no motif box::

    motif coverage = area( union of motif boxes ) / canvas
    void           = 1 - motif coverage

To hit a target void we uniformly rescale every motif box about its own centre
and binary-search the scale factor until the union coverage matches. We then
paste the actual motif cut-outs at those sizes onto a paper canvas (multiply
blend, so each crop's paper background drops out and only its ink remains) to
form a real arrangement skeleton — the thing generation is conditioned on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PAPER_RGB = (236, 229, 209)


@dataclass
class PlacedMotif:
    name: str
    asset_path: Path
    box: tuple[float, float, float, float]  # normalized x, y, w, h (after scaling)


def _clamp_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    x = min(max(x, 0.0), 1.0 - w)
    y = min(max(y, 0.0), 1.0 - h)
    return x, y, w, h


def scale_box(
    box: list[float] | tuple[float, ...], factor: float
) -> tuple[float, float, float, float]:
    """Scale a normalized box about its centre by ``factor`` and clamp to canvas."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * factor, h * factor
    return _clamp_box(cx - nw / 2.0, cy - nh / 2.0, nw, nh)


def union_coverage(boxes: list[tuple[float, float, float, float]], grid: int = 512) -> float:
    """Fraction of the canvas covered by the union of normalized boxes."""
    mask = np.zeros((grid, grid), dtype=bool)
    for x, y, w, h in boxes:
        left, top = round(x * grid), round(y * grid)
        right, bottom = round((x + w) * grid), round((y + h) * grid)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return float(mask.mean())


def solve_scale_for_void(
    boxes: list[list[float]],
    target_void: float,
    lo: float = 0.05,
    hi: float = 5.0,
    iters: int = 28,
) -> tuple[float, float]:
    """Binary-search a uniform box scale so union coverage ~= 1 - target_void.

    Returns ``(factor, achieved_coverage)``. Coverage is monotonic in the
    factor, so a simple bisection converges; if the target is unreachable
    (boxes too clustered) it returns the closest achievable coverage.
    """
    target_cov = max(0.0, min(1.0, 1.0 - target_void))
    best = (1.0, union_coverage([scale_box(b, 1.0) for b in boxes]))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cov = union_coverage([scale_box(b, mid) for b in boxes])
        if abs(cov - target_cov) < abs(best[1] - target_cov):
            best = (mid, cov)
        if cov < target_cov:
            lo = mid
        else:
            hi = mid
    return best


def build_arrangement(
    placed: list[PlacedMotif],
    width: int,
    height: int,
    out_path: Path,
    paper: tuple[int, int, int] = PAPER_RGB,
) -> tuple[Path, Path, float]:
    """Paste motif cut-outs at their (scaled) boxes onto a paper canvas.

    Larger boxes are drawn first (background) and smaller ones on top, using a
    multiply blend so each crop's bright paper background disappears and only
    its ink contributes. Also writes a binary coverage mask. Returns
    ``(arrangement_path, mask_path, placed_coverage)``.
    """
    canvas = np.full((height, width, 3), paper, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)

    order = sorted(placed, key=lambda p: p.box[2] * p.box[3], reverse=True)
    for p in order:
        x, y, w, h = p.box
        left, top = round(x * width), round(y * height)
        right, bottom = round((x + w) * width), round((y + h) * height)
        if right <= left or bottom <= top:
            continue
        crop = (
            Image.open(p.asset_path)
            .convert("RGB")
            .resize((right - left, bottom - top), Image.LANCZOS)
        )
        crop_arr = np.asarray(crop, dtype=np.float32) / 255.0
        # Whiten the crop's own paper background so the multiply leaves the
        # canvas untouched there (no faint rectangles around each motif).
        crop_lum = 0.299 * crop_arr[..., 0] + 0.587 * crop_arr[..., 1] + 0.114 * crop_arr[..., 2]
        bg = crop_lum > 0.74
        crop_arr[bg] = 1.0
        region = canvas[top:bottom, left:right].astype(np.float32) / 255.0
        blended = region * crop_arr  # multiply: white keeps base, ink darkens
        canvas[top:bottom, left:right] = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        mask[top:bottom, left:right] = True  # coverage is the bbox footprint

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, "RGB").save(out_path)
    mask_path = out_path.with_suffix(".mask.png")
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), "L").save(mask_path)
    return out_path, mask_path, float(mask.mean())


# ---------------------------------------------------------------------------
# Coherent mass / void composition layout (not a collage of floating motifs).
#
# We designate connected painted landmass region(s) and connected 留白 (open
# sky / mist / water) region(s), sized so the void area matches the target.
#
#   spread == 0  -> ONE bottom-anchored skyline with a dominant peak (compact).
#   spread  > 0  -> THREE DISTANCES (三远): a grounded foreground plus a middle
#                   distance and a far peak set progressively higher and to the
#                   side, separated by bands of mist/void so the masses are
#                   distributed across the height instead of packed at the
#                   bottom. Larger ``spread`` pushes the planes further apart.
#
# Either way this is a *composition map*: generation paints one unified scene
# into the mass and keeps the void as open silk, so the result reads as a real
# painting while the void proportion (and now the distribution) is controlled.
# ---------------------------------------------------------------------------


def build_mass_void_layout(
    width: int,
    height: int,
    target_void: float,
    convention: str = "top_sky",
    peak_scale: float = 1.0,
    seed: int = 0,
    spread: float = 0.0,
) -> dict:
    """Return a coherent land/void layout hitting ``target_void``.

    ``land`` is a boolean HxW mask (True = painted landmass), ``void`` its
    complement (open sky / 留白). ``soft`` is a feathered [0,1] land field used
    as the generation guide. With ``spread > 0`` the landmass is distributed
    into separated distance planes (see module note) instead of one compact
    bottom mass; the threshold is solved by bisection so the land area always
    equals ``1 - target_void`` regardless of distribution.
    """
    target_land = max(0.0, min(1.0, 1.0 - target_void))
    from .ink_ratio import tonal

    if spread <= 0.0:
        rng = np.random.default_rng(seed)
        xs = np.linspace(0.0, 1.0, width)
        peak_x = {"top_sky": 0.60, "diagonal": 0.80, "river_band": 0.5}.get(convention, 0.6)
        hills = 0.045 * np.sin(2 * np.pi * (xs * 1.4 + rng.random())) + 0.03 * np.sin(
            2 * np.pi * (xs * 3.1 + rng.random())
        )
        peak = (0.30 * float(peak_scale)) * np.exp(-((xs - peak_x) ** 2) / (2 * 0.09**2))
        profile = hills - peak
        yy = np.linspace(0.0, 1.0, height)[:, None]

        def land_for(offset: float) -> np.ndarray:
            boundary = np.clip(offset + profile, 0.0, 1.0)[None, :]
            return yy > boundary

        lo, hi = 0.0, 1.0
        for _ in range(34):
            mid = 0.5 * (lo + hi)
            if float(land_for(mid).mean()) < target_land:
                hi = mid
            else:
                lo = mid
        offset = 0.5 * (lo + hi)
        land = land_for(offset)
    else:
        land, peak_x = _distributed_land(
            width, height, target_land, peak_scale, seed, spread, convention
        )

    soft = tonal.gaussian_blur(land.astype(np.float32), max(2.0, height * 0.02))
    return {
        "land": land,
        "void": ~land,
        "soft": soft,
        "peak_x": peak_x,
        "achieved_void": float((~land).mean()),
        "target_void": float(target_void),
        "peak_scale": float(peak_scale),
        "spread": float(spread),
        "convention": convention,
    }


def _distributed_land(
    width: int,
    height: int,
    target_land: float,
    peak_scale: float,
    seed: int,
    spread: float,
    convention: str,
) -> tuple[np.ndarray, float]:
    """Three-distance mass field: foreground + middle + far peak, separated.

    Builds a smooth density from a few Gaussian blobs placed at increasing
    height (and alternating sides), then thresholds it so the land area equals
    ``target_land``. Larger ``spread`` raises the far peak and pulls the planes
    apart, leaving mist/void bands between them. The foreground blob is anchored
    to the bottom edge so the scene stays grounded (no fully-floating masses).
    """
    rng = np.random.default_rng(seed)
    s = float(np.clip(spread, 0.0, 1.0))
    xs = np.linspace(0.0, 1.0, width)[None, :]
    ys = np.linspace(0.0, 1.0, height)[:, None]

    side = 1.0 if (seed % 2 == 0) else -1.0  # which way the far peak leans

    def jx(a):
        return float(np.clip(a + side * 0.05 * (rng.random() - 0.5), 0.08, 0.92))

    # (cx, cy, sx, sy, amp) — cy: 0 top .. 1 bottom. Foreground low, far peak high.
    blobs = [
        (jx(0.50), 0.96, 0.40, 0.13 + 0.03 * s, 1.00),  # foreground, anchored bottom
        (jx(0.50 + side * 0.18), 0.70 - 0.10 * s, 0.20, 0.12, 0.82),  # middle distance
        (
            jx(0.50 - side * 0.20),
            0.46 - 0.20 * s,
            0.12,
            (0.12 + 0.06 * peak_scale),
            0.60 + 0.25 * peak_scale,
        ),  # far peak, higher with spread
    ]
    peak_x = blobs[-1][0]

    density = np.zeros((height, width), dtype=np.float32)
    for cx, cy, sx, sy, amp in blobs:
        density += amp * np.exp(-(((xs - cx) ** 2) / (2 * sx**2) + ((ys - cy) ** 2) / (2 * sy**2)))

    lo, hi = 0.0, float(density.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if float((density > mid).mean()) > target_land:
            lo = mid  # too much land -> raise threshold
        else:
            hi = mid
    land = density > (0.5 * (lo + hi))
    return land, peak_x


def render_layout_image(layout: dict, width: int, height: int, label: bool = True) -> Image.Image:
    """Render a composition layout in memory for API and GUI callers."""
    from PIL import ImageDraw

    soft = layout["soft"]
    yy = np.linspace(0.0, 1.0, height)[:, None]
    land_tone = np.clip(0.40 + 0.45 * yy, 0.18, 0.92) * np.ones((height, width), np.float32)
    paper = np.array(PAPER_RGB, np.float32) / 255.0
    land_rgb = np.stack([land_tone] * 3, axis=-1) * np.array([0.92, 0.93, 0.88])
    alpha = soft[..., None]
    rgb = paper[None, None, :] * (1 - alpha) + land_rgb * alpha
    image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), "RGB")

    if label:
        draw = ImageDraw.Draw(image)
        achieved_void = layout["achieved_void"]
        spread = layout.get("spread", 0.0)
        draw.text(
            (int(width * 0.04), int(height * 0.04)),
            f"open sky / 留白  (void {achieved_void * 100:.0f}%)",
            fill=(90, 90, 90),
        )
        draw.text(
            (int(width * 0.04), int(height * 0.92)),
            f"landmass (motifs {(1 - achieved_void) * 100:.0f}%)  "
            f"peak x{layout['peak_scale']:.1f}  spread {spread:.1f}",
            fill=(60, 60, 60),
        )
    return image


def render_layout_preview(
    layout: dict, width: int, height: int, out_path: Path, label: bool = True
) -> Path:
    """Composition map: open-sky 留白 vs. shaded landmass.

    With ``label=True`` it annotates the void/coverage (for human approval);
    with ``label=False`` it is a clean soft guide for conditioning generation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_layout_image(layout, width, height, label=label).save(out_path)
    return out_path


def _luminance(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _detect_ink_from_lum(
    lum: np.ndarray,
    dark_margin: float = 0.10,
    edge_thresh: float = 0.10,
    paper_pct: float = 80.0,
    blur: float = 1.5,
    close: int = 3,
) -> np.ndarray:
    from PIL import ImageFilter

    from .ink_ratio import tonal

    paper = float(np.percentile(lum, paper_pct))
    darker = lum < (paper - dark_margin)
    g = tonal.gaussian_blur(lum, blur)
    gx = np.abs(np.diff(g, axis=1, append=g[:, -1:]))
    gy = np.abs(np.diff(g, axis=0, append=g[-1:, :]))
    painted = darker | ((gx + gy) > edge_thresh)
    pim = Image.fromarray(np.where(painted, 255, 0).astype(np.uint8), "L")
    if close > 1:
        pim = pim.filter(ImageFilter.MaxFilter(close)).filter(ImageFilter.MinFilter(close))
    return np.asarray(pim, dtype=np.uint8) > 127


def detect_ink(image_path: Path, **kw) -> np.ndarray:
    """Boolean HxW mask of where brushwork/ink sits (vs blank silk).

    The dominant paper tone is estimated as a high luminance percentile; a pixel
    is ink if it is clearly darker than paper OR sits on a strong edge (catches
    thin outlines). Measures the painted *footprint*, which for airy ink motifs
    is naturally sparse.
    """
    lum = _luminance(np.asarray(Image.open(Path(image_path)).convert("RGB"), dtype=np.uint8))
    return _detect_ink_from_lum(lum, **kw)


def region_void(image_path: Path, downscale: int = 384, close: int = 9) -> dict:
    """Realized void as a *region* (not raw footprint).

    Downsamples, detects ink, then morphologically closes strokes into the
    landmass they belong to, so paper showing between branches counts as motif —
    not void. ``void`` is then the large connected empty silk. Returns
    ``{"coverage", "void", "ink_footprint"}``.
    """
    im = Image.open(Path(image_path)).convert("RGB")
    w, h = im.size
    nw = int(downscale)
    nh = max(1, round(h * nw / w))
    small = np.asarray(im.resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
    lum = _luminance(small)
    foot = _detect_ink_from_lum(lum, close=1)
    region = _detect_ink_from_lum(lum, close=close)
    return {
        "coverage": float(region.mean()),
        "void": float(1.0 - region.mean()),
        "ink_footprint": float(foot.mean()),
    }


def measure_occupancy(image_path: Path, motif_region: np.ndarray | None = None, **kw):
    """Verify a render against the intended composition.

    ``motif_region`` is the boolean placement mask (True inside placed motif
    boxes). The headline checks are compositional:

    * ``void_kept_blank`` — fraction of the *intended void* the render left as
      empty silk (high = the void was respected).
    * ``void_intruded`` — fraction of the intended void that got painted (the
      model filling negative space; low = good).
    * ``motif_region_filled`` — fraction of the intended motif boxes that
      actually received ink.

    ``ink_coverage`` is the raw painted footprint over the whole canvas.
    """
    ink = detect_ink(Path(image_path), **kw)
    res: dict[str, Any] = {
        "ink_coverage": float(ink.mean()),
        "ink_void": float(1.0 - ink.mean()),
    }
    if motif_region is not None:
        void_region = ~motif_region
        vr = max(int(void_region.sum()), 1)
        mr = max(int(motif_region.sum()), 1)
        res["void_region_frac"] = float(void_region.mean())
        res["void_intruded"] = float((ink & void_region).sum() / vr)
        res["void_kept_blank"] = float((~ink & void_region).sum() / vr)
        res["motif_region_filled"] = float((ink & motif_region).sum() / mr)
    res["mask"] = ink
    return res


def occupancy_preview(ink: np.ndarray, path: Path, motif_region: np.ndarray | None = None) -> Path:
    """Visualize the verification.

    Grey = intended motif region; black = ink that landed there; red = ink that
    intruded into the intended void; white = void kept blank.
    """
    out = np.full((*ink.shape, 3), 250, np.uint8)
    if motif_region is not None:
        out[motif_region] = (224, 224, 224)
        out[motif_region & ink] = (30, 30, 30)
        out[(~motif_region) & ink] = (220, 40, 40)
    else:
        out[ink] = (40, 150, 70)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGB").save(path)
    return path
