"""Region masks for content-level ink-ratio control.

Where :mod:`tonal` measures and the tone curve only recolors, this module
builds *region masks* that say which parts of a painting must change to reach
the target ratio — so we can repaint only those regions (true content change)
via an inpainting edit.

Bands are classified by luminance against the same two thresholds used
everywhere else::

    void (留白)       : luminance >  white_t
    transition (过渡) : dark_t <= lum <= white_t
    ink  (实)         : luminance <  dark_t

A "deficit mask" for band B selects pixels where the *target* layout (the
pre-generated guide) wants band B but the *current* painting is not band B yet,
capped to the area shortfall and made spatially coherent. The mask is written in
OpenAI ``images.edit`` convention: **fully transparent = repaint here**, opaque
= keep unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import tonal

INK, TRANSITION, VOID = 0, 1, 2
BAND_INDEX = {"ink": INK, "transition": TRANSITION, "void": VOID}


def label_map(lum: np.ndarray, white_t: float, dark_t: float) -> np.ndarray:
    """HxW luminance -> HxW band labels (0=ink, 1=transition, 2=void)."""
    lbl = np.full(lum.shape, TRANSITION, dtype=np.uint8)
    lbl[lum > white_t] = VOID
    lbl[lum < dark_t] = INK
    return lbl


def load_luminance(path: Path) -> np.ndarray:
    return tonal.luminance(tonal.from_pil(Image.open(Path(path))))


def _band_preference(guide_lum: np.ndarray, band: int, white_t: float, dark_t: float) -> np.ndarray:
    """Higher = the guide more strongly wants this band at that pixel."""
    if band == INK:
        return dark_t - guide_lum  # darker guide -> stronger ink
    if band == VOID:
        return guide_lum - white_t  # brighter guide -> stronger void
    mid = 0.5 * (white_t + dark_t)
    return -np.abs(guide_lum - mid)  # closer to mid -> stronger transition


def build_edit_mask(
    guide_lum: np.ndarray,
    current_lum: np.ndarray,
    band: int,
    deficit_frac: float,
    white_t: float,
    dark_t: float,
    smooth: float = 2.5,
    overfill: float = 1.15,
):
    """Return (selected_bool_HxW, RGBA mask Image) for a band's deficit.

    ``deficit_frac`` is the area shortfall (target - current) for ``band``.
    Selects the guide-intended ``band`` pixels not yet in that band, ranked by a
    smoothed guide preference (so the selection forms coherent blobs), capped to
    ``deficit_frac * overfill`` of the canvas. ``overfill`` mildly enlarges the
    region to offset imperfect repainting.
    """
    h, w = current_lum.shape
    glbl = label_map(guide_lum, white_t, dark_t)
    clbl = label_map(current_lum, white_t, dark_t)
    candidates = (glbl == band) & (clbl != band)

    sel = np.zeros((h, w), dtype=bool)
    n_target = int(max(0.0, deficit_frac) * overfill * h * w)
    n_cand = int(candidates.sum())
    if n_target > 0 and n_cand > 0:
        pref = _band_preference(guide_lum, band, white_t, dark_t).astype(np.float32)
        pref = (pref - pref.min()) / (np.ptp(pref) + 1e-8)
        pref = np.where(candidates, pref, 0.0)
        # Smooth so high-score pixels cluster into coherent regions, then keep
        # only candidate pixels.
        pref = tonal.gaussian_blur(pref, smooth) * candidates
        flat = np.where(candidates.ravel(), pref.ravel(), -1.0)
        k = min(n_target, n_cand)
        idx = np.argpartition(flat, -k)[-k:]
        keep = np.zeros(h * w, dtype=bool)
        keep[idx] = True
        sel = (keep.reshape(h, w)) & candidates

    alpha = np.where(sel, 0, 255).astype(np.uint8)  # transparent = edit
    rgba = np.dstack([np.zeros((h, w, 3), np.uint8), alpha])
    return sel, Image.fromarray(rgba, "RGBA")


def feather_alpha(selected: np.ndarray, radius: float) -> np.ndarray:
    """Boolean selection -> soft [0,1] alpha (1 inside, fading at edges).

    Used by the NVIDIA composite backend to blend regenerated content into the
    original only where a band is in deficit, softening the region seams.
    """
    a = selected.astype(np.float32)
    if radius > 0:
        a = tonal.gaussian_blur(a, radius)
    return np.clip(a, 0.0, 1.0)


def composite(base_rgb: np.ndarray, gen_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Alpha-blend ``gen_rgb`` over ``base_rgb`` with a per-pixel ``alpha`` map."""
    a = alpha[..., None]
    out = base_rgb.astype(np.float32) * (1.0 - a) + gen_rgb.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def save_mask(mask_img: Image.Image, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(path)
    return path


def mask_preview(mask_img: Image.Image, path: Path) -> Path:
    """Opaque visualization of an edit mask: red = repaint region."""
    arr = np.asarray(mask_img.convert("RGBA"), dtype=np.uint8)
    edit = arr[..., 3] == 0
    out = np.full((*edit.shape, 3), 245, np.uint8)
    out[edit] = (220, 40, 40)
    return save_mask(Image.fromarray(out, "RGB"), path)
