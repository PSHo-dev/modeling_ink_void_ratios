"""Tonal ink-ratio control for inkvoidmotif compositions.

Enforces a target tonal composition — the share of canvas area that is blank
paper (void / 留白), mid-tone wash (transition / 过渡), and dark brush masses
(ink / 实) — on generated Chinese landscape paintings.

The pure-numpy core (:mod:`tonal`, :mod:`guide`) is vendored unchanged from the
standalone ``comfyui-ink-ratio`` project; :mod:`control` adds the file-based
glue used by the inkvoidmotif pipeline and CLI.
"""

from . import control, guide, tonal
from .control import (
    DEFAULT_DARK_T,
    DEFAULT_METRIC,
    DEFAULT_TOL,
    DEFAULT_WHITE_T,
    all_devs,
    band_errors,
    build_guide_file,
    choose_best,
    correct_array,
    correct_file,
    deviation_metric,
    feedback_text,
    fmt_devs,
    fmt_ratio,
    guide_reference_note,
    make_ink_spec,
    measure_array,
    measure_file,
    revise_reference_note,
    spec_from_config,
    target_tuple,
    write_bands_file,
    write_report,
)

__all__ = [
    "DEFAULT_DARK_T",
    "DEFAULT_METRIC",
    "DEFAULT_TOL",
    "DEFAULT_WHITE_T",
    "all_devs",
    "band_errors",
    "build_guide_file",
    "choose_best",
    "control",
    "correct_array",
    "correct_file",
    "deviation_metric",
    "feedback_text",
    "fmt_devs",
    "fmt_ratio",
    "guide",
    "guide_reference_note",
    "make_ink_spec",
    "measure_array",
    "measure_file",
    "revise_reference_note",
    "spec_from_config",
    "target_tuple",
    "tonal",
    "write_bands_file",
    "write_report",
]
