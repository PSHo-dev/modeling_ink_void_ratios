"""Composition guide whose three tonal bands hit the target areas EXACTLY.

Vendored from the standalone ``comfyui-ink-ratio`` project. The guide is an
optional reference image that *steers* gpt-image-2 toward the target tonal
composition; exact enforcement is still done afterwards by the deterministic
tone curve in :mod:`tonal`.
"""

from __future__ import annotations

import numpy as np

from . import tonal


def build_guide(spec, width, height, seed, composition, bias_strength=0.55, softness=0.06):
    """Return an HxWx3 grayscale guide image (numpy float [0,1])."""
    v, t, i = spec["void"], spec["transition"], spec["ink"]
    wt, dt = spec["white_t"], spec["dark_t"]
    field = (1 - bias_strength) * tonal.fbm_field(
        height, width, seed
    ) + bias_strength * tonal.composition_bias(height, width, composition)
    field = (field - field.min()) / (np.ptp(field) + 1e-8)
    guide = tonal.quantize_to_bands(field, v, t, i, wt, dt)
    if softness > 0:
        guide = tonal.gaussian_blur(guide, softness * 0.04 * (width + height) / 2.0)
    return tonal.to_rgb(guide)
