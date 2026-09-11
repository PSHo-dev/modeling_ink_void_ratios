from __future__ import annotations

from .schemas import MotifSpec

QUALITY_EXEMPLAR_NOTES = """
Quality target learned from the collaborator's successful motif bank:
- Segment whole reusable objects, not arbitrary rectangular details.
- Preserve enough neighboring brushwork for the motif to remain legible: village clusters need trees and walls; bridges need adjacent shrubs and masonry; mountains need their wash edge and ridge texture.
- Remove the original painting ground cleanly, but keep soft ink/wash boundaries that belong to the motif.
- Keep the original hand-painted irregular outline; never force a sticker-like hard silhouette.
- Maintain the source palette, dry-brush texture, line density, and scale relationships.
- For review sheets, labels belong outside the motif images. Motif assets themselves must contain no text.
""".strip()


def planning_prompt(
    count: int,
    instructions: str | None = None,
    use_quality_exemplar: bool = False,
) -> str:
    extra = (
        f"\nAdditional collaborator instructions:\n{instructions.strip()}\n" if instructions else ""
    )
    exemplar = (
        "\nThe first image is the source painting. A later image may be a motif-bank quality exemplar. Use the exemplar only to understand segmentation quality and motif-bank style; do not copy its motifs.\n"
        f"{QUALITY_EXEMPLAR_NOTES}\n"
        if use_quality_exemplar
        else ""
    )
    return f"""
Analyze this Chinese landscape painting as a source for a motif bank.

Create exactly {count} reusable motifs. Return JSON only.

For each motif:
- Use a short snake_case name.
- Describe what should be isolated.
- Provide a normalized bbox as [x, y, width, height], where 0,0 is the top-left of the full image.
- Include all visible brushwork for that motif, but do not include unrelated neighboring motifs.
- Prefer motifs that can be recombined into a new coherent Chinese painting: mountains, hillsides, village clusters, pavilion groups, water/lotus surfaces, bridges, scholar figures, boats, clouds, rocks, shrubs, pines, willows, bamboo.
- Capture details that must be preserved: linework, wash texture, roof geometry, tree needles, rock texture, mist edges, and color accents.

Do not invent objects that are not visible in the source image.
Do not include calligraphy, seals, labels, borders, or paper texture as motifs unless explicitly requested.
{exemplar}
{extra}
""".strip()


def extraction_prompt(spec: MotifSpec, use_quality_exemplar: bool = False) -> str:
    preserve = ""
    if spec.must_preserve:
        preserve = "\nSpecific details to preserve: " + "; ".join(spec.must_preserve)
    exemplar = (
        "\nThe first image is the source crop to extract from. A later image may be a quality exemplar contact sheet. Use that exemplar only as a standard for clean motif isolation and brushwork fidelity; do not copy any objects from it.\n"
        f"{QUALITY_EXEMPLAR_NOTES}\n"
        if use_quality_exemplar
        else ""
    )

    return f"""
Create one reusable transparent-background motif asset from the reference image.

Motif name: {spec.name}
Motif to isolate: {spec.description}
Compositional role: {spec.role or "motif"}

Requirements:
- Output only this motif, centered on a transparent background.
- Remove all paper, silk, sky, labels, grid lines, captions, seals, and unrelated surroundings.
- Preserve the original Chinese painting brushwork, line density, ink texture, pale mineral color, wash edges, and fine architectural or botanical details.
- Preserve whole-object completeness: do not amputate roofs, trees, bridges, peaks, banks, or mist/wash edges that are part of the named motif.
- Keep motif scale and interior detail close to the reference crop; avoid generic repainting.
- Keep the motif's natural irregular contour. Avoid a rectangular paper patch or white halo.
- Do not simplify, modernize, vectorize, photorealize, or add new objects.
- Do not add text.
{exemplar}
{preserve}
""".strip()


def ink_ratio_guidance(spec: dict, with_guide: bool = False) -> str:
    """Painterly description of a target tonal balance for the text prompt.

    ``spec`` carries area fractions (void/transition/ink). This is *soft*
    steering only; the deterministic tone curve is what guarantees the ratio.
    """
    void = float(spec["void"]) * 100
    transition = float(spec["transition"]) * 100
    ink = float(spec["ink"]) * 100
    guide_line = (
        "\n- One reference image is an abstract three-tone composition guide "
        "(white = blank paper, grey = mid-tone wash, black = dense ink mass). "
        "Follow its large-scale distribution of empty versus dense areas, but "
        "paint with the motif brushwork; do not copy the guide's shapes literally."
        if with_guide
        else ""
    )
    return (
        "\nTonal balance (ink-wash economy / 墨分三色):\n"
        f"- Leave roughly {void:.0f}% of the surface as bright untouched paper "
        "or pale sky/water (留白 / void) so the painting can breathe.\n"
        f"- Keep roughly {transition:.0f}% as mid-tone wash, mist, and graded "
        "transitions (过渡).\n"
        f"- Concentrate the darkest ink masses (brush-dense rock, foliage, and "
        f"structures, 实) in roughly {ink:.0f}% of the surface.\n"
        "- Use asymmetric, classical Chinese spatial division; do not fill the "
        "whole sheet evenly or crowd every corner."
        f"{guide_line}\n"
    )


def composition_prompt(
    motifs: list[MotifSpec],
    theme: str,
    include_scholar: bool = False,
    layout: str | None = None,
    source_style: str | None = None,
    use_quality_exemplar: bool = False,
    ink_spec: dict | None = None,
    ink_guide: bool = False,
) -> str:
    motif_lines = "\n".join(f"- {motif.name}: {motif.description}" for motif in motifs)
    scholar_line = (
        "- Include a scholar figure in the painting. If a scholar motif is provided, use that motif; otherwise make the scholar visually compatible with the motif bank.\n"
        if include_scholar
        else ""
    )
    layout_line = f"\nComposition/layout request:\n{layout.strip()}\n" if layout else ""
    style_line = f"\nStyle reference notes:\n{source_style.strip()}\n" if source_style else ""
    exemplar = (
        f"\nMotif-bank quality target:\n{QUALITY_EXEMPLAR_NOTES}\n" if use_quality_exemplar else ""
    )
    ink_line = ink_ratio_guidance(ink_spec, with_guide=ink_guide) if ink_spec else ""

    return f"""
Create a new Chinese landscape painting on one coherent background of aged silk or xuan paper.

Theme: {theme}

Use the supplied motif-bank reference images as the source of brushwork and details.
Recombine them into a different composition, while preserving the recognizable brushwork, linework, color wash, architectural details, trees, rocks, water texture, and mist treatment of the motifs.

Motifs to embed:
{motif_lines}

Requirements:
- The final image must read as one continuous Chinese landscape painting, not a collage, contact sheet, sticker layout, or labeled diagram.
- Preserve motif details and brushwork as closely as the API allows.
- Integrate motifs with consistent scale, perspective, mist, paper tone, and water/land transitions.
- Reuse the visual identity of the supplied motifs; do not replace them with generic mountains, trees, houses, bridges, or figures.
- Remove transparent cutout edges by blending the motifs into the silk or paper ground.
- Do not add captions, motif names, grids, UI elements, or modern photographic details.
- Do not add unrelated seals or calligraphy unless the source references include them and the composition naturally calls for subtle seals.
{scholar_line}{layout_line}{style_line}{ink_line}{exemplar}
""".strip()
