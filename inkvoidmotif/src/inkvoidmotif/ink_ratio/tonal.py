"""Pure-numpy tonal core for ink-ratio control (no torch / no ComfyUI).

Vendored, unchanged, from the standalone ``comfyui-ink-ratio`` project so the
inkvoidmotif pipeline can enforce a target tonal composition on generated
paintings. Three tonal bands are classified by luminance against two fixed
thresholds:

    void  (留白)      : luminance >  white_t       -> blank paper
    transition (过渡) : dark_t <= lum <= white_t   -> mist / mid-tone wash ("edge")
    ink   (实)        : luminance <  dark_t        -> dark brush masses

The "ratio" is the share of canvas AREA in each band.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """HxWx3 (or HxW) float in [0,1] -> HxW luminance in [0,1]."""
    if img.ndim == 2:
        return img.astype(np.float32)
    return (img[..., :3] * REC709).sum(axis=-1).astype(np.float32)


def normalize_ratios(v: float, t: float, i: float):
    s = max(1e-8, v + t + i)
    return v / s, t / s, i / s


def measure_ratios(lum: np.ndarray, white_t: float, dark_t: float):
    """Return (void, transition, ink) area fractions for a luminance map."""
    void = float(np.mean(lum > white_t))
    ink = float(np.mean(lum < dark_t))
    return void, max(0.0, 1.0 - void - ink), ink


def deviation(target, measured) -> float:
    """Max absolute per-band deviation between two (v,t,i) tuples."""
    return float(max(abs(a - b) for a, b in zip(target, measured, strict=False)))


def _resize_bilinear(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(arr.astype(np.float32)).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )


def fbm_field(
    h: int,
    w: int,
    seed: int,
    octaves: int = 5,
    persistence: float = 0.55,
    base_cells: int = 3,
) -> np.ndarray:
    """Fractal value-noise in [0,1]; smooth, organic, deterministic per seed."""
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    field = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        grid = rng.standard_normal((max(2, base_cells * 2**o), max(2, base_cells * 2**o))).astype(
            np.float32
        )
        field += amp * _resize_bilinear(grid, h, w)
        total += amp
        amp *= persistence
    field /= total
    return (field - field.min()) / (np.ptp(field) + 1e-8)


def composition_bias(h: int, w: int, kind: str) -> np.ndarray:
    """Smooth 0..1 field (high=bright/void) encoding a 山水 composition prior.
    Only shapes WHERE tones go; exact areas are enforced by quantize_to_bands."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y, x = yy / max(1, h - 1), xx / max(1, w - 1)
    k = (kind or "").lower()
    if "high" in k or "高远" in k or "tall" in k:
        ridge = np.exp(-((x - 0.5) ** 2) / 0.06)
        return np.clip(0.15 + 0.95 * y - 0.55 * ridge * (1 - y), 0, 1)
    if "level" in k or "平远" in k or "wide" in k:
        bands = 0.5 + 0.5 * np.sin(2.0 * np.pi * (y * 2.3))
        return np.clip(0.25 + 0.7 * y + 0.15 * bands, 0, 1)
    if "deep" in k or "深远" in k:
        return np.clip(0.2 + 0.8 * (0.5 * y + 0.5 * x), 0, 1)
    if "river" in k or "valley" in k:
        channel = np.exp(-((x - (0.5 + 0.18 * np.sin(3.0 * np.pi * y))) ** 2) / 0.01)
        return np.clip(0.35 + 0.55 * channel + 0.2 * y, 0, 1)
    if "peak" in k:
        peak = np.exp(-(((x - 0.5) ** 2) / 0.04 + ((y - 0.45) ** 2) / 0.08))
        return np.clip(0.85 - 0.7 * peak, 0, 1)
    return np.clip(0.3 + 0.6 * np.abs(y - 0.45) + 0.2 * np.abs(x - 0.5), 0, 1)


def _lin(x, x0, x1, y0, y1):
    if x1 - x0 < 1e-9:
        return np.full_like(x, (y0 + y1) * 0.5)
    return y0 + np.clip((x - x0) / (x1 - x0), 0.0, 1.0) * (y1 - y0)


def quantize_to_bands(
    field: np.ndarray,
    v: float,
    t: float,
    i: float,
    white_t: float,
    dark_t: float,
    eps: float = 1e-3,
) -> np.ndarray:
    """Map a smooth field to a grayscale guide whose band AREAS equal (v,t,i) EXACTLY
    under (white_t, dark_t). Monotonic in `field`, so composition is preserved."""
    v, t, i = normalize_ratios(v, t, i)
    flat = field.ravel().astype(np.float32)
    n = flat.size
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(n, np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    q = ranks / max(1.0, n - 1.0)
    out = np.empty(n, np.float32)
    ink_m, trans_m, void_m = q < i, (q >= i) & (q < i + t), q >= i + t
    out[ink_m] = _lin(q[ink_m], 0.0, i, 0.0, max(0.0, dark_t - eps))
    out[trans_m] = _lin(q[trans_m], i, i + t, dark_t, white_t)
    out[void_m] = _lin(q[void_m], i + t, 1.0, min(1.0, white_t + eps), 1.0)
    return out.reshape(field.shape)


def fit_band_curve(lum: np.ndarray, v: float, t: float, i: float, white_t: float, dark_t: float):
    """Monotonic piecewise-linear tone curve f(L) so remapped band areas equal (v,t,i)."""
    v, t, i = normalize_ratios(v, t, i)
    qd, qw = float(np.quantile(lum, i)), float(np.quantile(lum, i + t))
    xs = np.array([0.0, qd, qw, 1.0], dtype=np.float32)
    for j in range(1, 4):
        if xs[j] <= xs[j - 1]:
            xs[j] = xs[j - 1] + 1e-4
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.array([0.0, dark_t, white_t, 1.0])
    return lambda L: np.interp(L, xs, ys).astype(np.float32)


def apply_curve(img: np.ndarray, f, strength: float, preserve_color: bool) -> np.ndarray:
    """Apply tone curve to LUMINANCE (guarantees realized ratio == designed).
    preserve_color=False -> monochrome ink; True -> keep hue via chroma scaling."""
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    L = luminance(img)
    Lb = (1 - strength) * L + strength * f(L)
    if preserve_color:
        return np.clip(img * (Lb / np.clip(L, 1e-4, None))[..., None], 0.0, 1.0)
    return np.clip(np.repeat(Lb[..., None], 3, axis=-1), 0.0, 1.0)


def band_visualization(lum: np.ndarray, white_t: float, dark_t: float) -> np.ndarray:
    """void=white, transition=teal, ink=indigo."""
    h, w = lum.shape
    out = np.zeros((h, w, 3), np.float32)
    void, ink = lum > white_t, lum < dark_t
    out[void] = (1.0, 1.0, 1.0)
    out[~void & ~ink] = (0.10, 0.55, 0.55)
    out[ink] = (0.12, 0.10, 0.30)
    return out


def gaussian_blur(gray: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return gray
    im = Image.fromarray((np.clip(gray, 0, 1) * 255 + 0.5).astype(np.uint8)).convert("L")
    return np.asarray(im.filter(ImageFilter.GaussianBlur(float(radius))), np.float32) / 255.0


# ---- IO helpers ------------------------------------------------------------


def to_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr.astype(np.float32), 0, 1)
    return np.repeat(arr[..., None], 3, -1) if arr.ndim == 2 else arr[..., :3]


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))


def from_pil(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), np.float32) / 255.0


def png_bytes(arr: np.ndarray, name: str = "ref.png") -> io.BytesIO:
    bio = io.BytesIO()
    to_pil(to_rgb(arr)).convert("RGB").save(bio, format="PNG")
    bio.seek(0)
    bio.name = name
    return bio


def save_png(arr: np.ndarray, path: str):
    to_pil(to_rgb(arr)).save(path)
