"""Declarative Gradio component tree for inkvoidmotif Studio.

This module owns presentation only. Provider calls and workflow state remain in
``studio.py``; callback wiring and server policy remain in ``gradio_app.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr

from . import studio

IMAGE_SIZES = ("1024x1536", "1024x1024", "1536x1024")
PROVIDER_CHOICES = (
    ("Automatic", "auto"),
    ("OpenAI", "openai"),
    ("NVIDIA", "nvidia"),
)


@dataclass(frozen=True)
class SavedRunComponents:
    selector: gr.Dropdown
    refresh: gr.Button
    resume: gr.Button


@dataclass(frozen=True)
class SourceComponents:
    image: gr.Image
    motif_count: gr.Slider
    provider: gr.Dropdown
    planning_notes: gr.Textbox
    quota_acknowledged: gr.Checkbox
    analyze: gr.Button
    readiness: gr.Markdown


@dataclass(frozen=True)
class BankComponents:
    source_summary: gr.Textbox
    motif_table: gr.Dataframe
    upload: gr.Image
    sheet_size: gr.Dropdown
    columns: gr.Slider
    build: gr.Button


@dataclass(frozen=True)
class ComposeComponents:
    scene: gr.Textbox
    title: gr.Textbox
    convention: gr.Dropdown
    target_void: gr.Slider
    peak: gr.Slider
    spread: gr.Slider
    seed: gr.Slider
    scholar: gr.Checkbox
    include_source: gr.Checkbox
    style: gr.Textbox
    generate: gr.Button
    cancel: gr.Button


@dataclass(frozen=True)
class AdvancedComponents:
    text_model: gr.Textbox
    image_model: gr.Textbox
    quality: gr.Dropdown
    output_size: gr.Dropdown


@dataclass(frozen=True)
class PreviewComponents:
    status: gr.Markdown
    layout_image: gr.Image
    layout_caption: gr.Markdown
    crop_gallery: gr.Gallery
    motif_sheet: gr.Image
    motif_gallery: gr.Gallery
    result: gr.Image
    history: gr.Gallery
    plan_file: gr.File
    bank_file: gr.File
    report_file: gr.File
    download_run: gr.DownloadButton


@dataclass(frozen=True)
class StudioComponents:
    session: gr.State
    saved_runs: SavedRunComponents
    source: SourceComponents
    bank: BankComponents
    compose: ComposeComponents
    advanced: AdvancedComponents
    preview: PreviewComponents


def _header() -> None:
    gr.HTML(
        """
        <section class="studio-hero">
          <div class="eyebrow">inkvoidmotif · Painting Studio</div>
          <h1>Compose with breathing space.</h1>
          <p>Turn one Chinese landscape into a reusable motif vocabulary, then guide a new painting with explicit control over void, structure, and atmosphere.</p>
        </section>
        <div class="step-strip" aria-label="Workflow">
          <span class="step-chip"><b>01</b> Read the painting</span>
          <span class="step-chip"><b>02</b> Curate motif bank</span>
          <span class="step-chip"><b>03</b> Compose &amp; compare</span>
        </div>
        """
    )


def _saved_runs(allow_saved_runs: bool) -> SavedRunComponents:
    with gr.Accordion("Resume a saved run", open=False, visible=allow_saved_runs), gr.Row():
        selector = gr.Dropdown(
            choices=studio.list_run_ids(),
            label="Saved run",
            allow_custom_value=False,
            scale=5,
        )
        refresh = gr.Button("Refresh list", scale=1)
        resume = gr.Button("Resume", variant="secondary", scale=1)
    return SavedRunComponents(selector, refresh, resume)


def _source_panel() -> SourceComponents:
    with (
        gr.Column(scale=1, min_width=300, elem_classes="workflow-panel"),
        gr.Group(elem_classes="stage"),
    ):
        gr.Markdown("## 01 · Source", elem_classes="stage-title")
        image = gr.Image(
            type="filepath",
            sources=["upload"],
            label="Source painting",
            buttons=["fullscreen"],
            height=230,
        )
        with gr.Row(elem_classes="mobile-stack"):
            motif_count = gr.Slider(
                2,
                12,
                value=8,
                step=1,
                label="Motifs",
                info="8 is a balanced start",
            )
            provider = gr.Dropdown(
                choices=PROVIDER_CHOICES,
                value="auto",
                label="Provider",
            )
        planning_notes = gr.Textbox(
            label="Planner focus",
            placeholder="Optional — preserve the pavilion cluster…",
            lines=1,
            max_length=2000,
        )
        quota_acknowledged = gr.Checkbox(
            label="I understand provider privacy and quota use.",
            value=False,
        )
        analyze = gr.Button(
            "Analyze painting",
            variant="primary",
            elem_classes="primary-action",
        )
        readiness = gr.Markdown(studio.readiness_markdown(), elem_classes="quiet-note compact-note")
    return SourceComponents(
        image,
        motif_count,
        provider,
        planning_notes,
        quota_acknowledged,
        analyze,
        readiness,
    )


def _bank_panel() -> BankComponents:
    with (
        gr.Column(scale=1, min_width=300, elem_classes="workflow-panel"),
        gr.Group(elem_classes="stage"),
    ):
        gr.Markdown("## 02 · Motif bank", elem_classes="stage-title")
        source_summary = gr.Textbox(label="Source reading", lines=2, interactive=False)
        motif_table = gr.Dataframe(
            headers=[
                "Name",
                "Description",
                "Role",
                "BBox x,y,w,h",
                "Preserve (semicolon separated)",
            ],
            datatype=["str", "str", "str", "str", "str"],
            type="array",
            value=[],
            column_count=5,
            interactive=True,
            wrap=True,
            show_row_numbers=True,
            label="Editable motif plan",
            max_height=250,
        )
        gr.Markdown(
            "BBox is normalized `x, y, width, height`. Rebuild after editing.",
            elem_classes="quiet-note compact-note",
        )
        upload = gr.Image(
            type="filepath",
            sources=["upload"],
            label="Optional existing motif sheet",
            height=150,
        )
        with gr.Row(elem_classes="mobile-stack"):
            sheet_size = gr.Dropdown(
                IMAGE_SIZES,
                value="1024x1536",
                label="Sheet size",
            )
            columns = gr.Slider(1, 4, value=2, step=1, label="Columns")
        build = gr.Button(
            "Build / use motif bank",
            variant="primary",
            elem_classes="primary-action",
            interactive=False,
        )
    return BankComponents(source_summary, motif_table, upload, sheet_size, columns, build)


def _compose_panel() -> ComposeComponents:
    with (
        gr.Column(scale=1, min_width=300, elem_classes="workflow-panel"),
        gr.Group(elem_classes="stage"),
    ):
        gr.Markdown("## 03 · Compose", elem_classes="stage-title")
        scene = gr.Textbox(
            label="New landscape",
            value="A quiet early-autumn river valley with a pavilion among pines, a narrow bridge, distant misty peaks, and open water.",
            lines=3,
            max_length=1800,
        )
        title = gr.Textbox(
            label="Output name",
            value="autumn_river_recomposition",
            max_length=90,
        )
        with gr.Row(elem_classes="mobile-stack"):
            convention = gr.Dropdown(
                choices=[
                    ("High distance · open sky", "top_sky"),
                    ("Diagonal ascent", "diagonal"),
                    ("Winding river corridor", "river_band"),
                ],
                value="top_sky",
                label="Structure",
            )
            target_void = gr.Slider(
                0.25,
                0.85,
                value=0.62,
                step=0.01,
                label="Void",
                info="Reserved open silk",
            )
        with gr.Row():
            peak = gr.Slider(
                0.25,
                1.25,
                value=0.68,
                step=0.01,
                label="Peak",
                min_width=100,
            )
            spread = gr.Slider(
                0.0,
                1.0,
                value=0.62,
                step=0.01,
                label="Spread",
                min_width=100,
            )
            seed = gr.Slider(
                0,
                999,
                value=7,
                step=1,
                label="Seed",
                min_width=100,
            )
        with gr.Row(elem_classes="mobile-stack"):
            scholar = gr.Checkbox(label="Include scholar", value=False)
            include_source = gr.Checkbox(
                label="Use source as style reference",
                value=True,
            )
        style = gr.Textbox(
            label="Style direction",
            placeholder="Optional — mineral washes, dry-brush pines…",
            lines=1,
            max_length=1200,
        )
        generate = gr.Button(
            "Generate new painting",
            variant="primary",
            elem_classes="generate-action",
            interactive=False,
        )
        cancel = gr.Button("Cancel queued request", variant="stop", size="sm")
        gr.Markdown(
            "Active provider calls may finish before their timeout.",
            elem_classes="quiet-note compact-note",
        )
    return ComposeComponents(
        scene,
        title,
        convention,
        target_void,
        peak,
        spread,
        seed,
        scholar,
        include_source,
        style,
        generate,
        cancel,
    )


def _advanced_settings() -> AdvancedComponents:
    with gr.Accordion("Advanced model settings", open=False):
        with gr.Row(elem_classes="mobile-stack"):
            text_model = gr.Textbox(label="Text model override", placeholder="Automatic")
            image_model = gr.Textbox(label="Image model override", placeholder="Automatic")
            quality = gr.Dropdown(
                ["low", "medium", "high", "auto"],
                value="high",
                label="Quality",
            )
            output_size = gr.Dropdown(
                IMAGE_SIZES,
                value="1024x1536",
                label="Painting size",
            )
        gr.Markdown(
            "Each model-generated stage may use provider quota. Planning, motif sheets, and finished paintings are cached per run.",
            elem_classes="quiet-note",
        )
    return AdvancedComponents(text_model, image_model, quality, output_size)


def _preview_panel(initial_map: Any, initial_caption: str) -> PreviewComponents:
    with gr.Group(elem_classes=["stage", "preview-stage"]):
        gr.Markdown("## Preview & outputs", elem_classes="preview-heading")
        status = gr.Markdown(
            "### Ready to begin\n\nUpload a source painting, then analyze its reusable motifs.",
            elem_classes="status-card",
            elem_id="studio-status",
        )
        with gr.Tabs():
            with gr.Tab("Composition map"):
                layout_image = gr.Image(
                    value=initial_map,
                    label="Light = open silk · shaded = painted mass",
                    interactive=False,
                    buttons=["fullscreen"],
                    elem_classes="preview-frame",
                    height=560,
                )
                layout_caption = gr.Markdown(initial_caption)
            with gr.Tab("Source crops"):
                crop_gallery = gr.Gallery(
                    label="Planned source regions",
                    columns=4,
                    object_fit="contain",
                    height=520,
                    buttons=["fullscreen", "download_all"],
                )
            with (
                gr.Tab("Motif bank"),
                gr.Row(equal_height=False, elem_classes="mobile-stack"),
            ):
                motif_sheet = gr.Image(
                    label="Motif sheet",
                    interactive=False,
                    buttons=["fullscreen", "download"],
                    height=520,
                    scale=1,
                )
                motif_gallery = gr.Gallery(
                    label="Review cells",
                    columns=3,
                    object_fit="contain",
                    height=520,
                    buttons=["fullscreen", "download_all"],
                    scale=1,
                )
            with gr.Tab("New painting"):
                result = gr.Image(
                    label="Generated composition",
                    interactive=False,
                    buttons=["fullscreen", "download"],
                    height=690,
                    elem_classes="result-frame",
                )
                history = gr.Gallery(
                    label="This run's variations",
                    columns=4,
                    rows=1,
                    object_fit="contain",
                    height=220,
                    buttons=["fullscreen", "download_all"],
                )
        with gr.Row(elem_classes="mobile-stack"):
            plan_file = gr.File(label="Motif plan", interactive=False)
            bank_file = gr.File(label="Motif-bank metadata", interactive=False)
            report_file = gr.File(label="Composition report", interactive=False)
        download_run = gr.DownloadButton("Download complete run", visible=False)
        gr.Markdown(
            "Source paintings, generated/uploaded motif sheets, guides, and optional style references are sent to the selected provider at their relevant stages. Outputs stay in `runs/studio/` until you remove them; ZIP metadata uses portable relative paths. API secrets are never written into artifacts.",
            elem_classes="api-note",
        )
    return PreviewComponents(
        status,
        layout_image,
        layout_caption,
        crop_gallery,
        motif_sheet,
        motif_gallery,
        result,
        history,
        plan_file,
        bank_file,
        report_file,
        download_run,
    )


def build_interface(
    *, allow_saved_runs: bool, initial_map: Any, initial_caption: str
) -> StudioComponents:
    """Build the component tree inside an active ``gr.Blocks`` context."""
    session = gr.State(value=None)
    _header()
    saved_runs = _saved_runs(allow_saved_runs)
    with gr.Row(equal_height=True, elem_classes="workflow-row"):
        source = _source_panel()
        bank = _bank_panel()
        compose = _compose_panel()
    advanced = _advanced_settings()
    preview = _preview_panel(initial_map, initial_caption)
    return StudioComponents(session, saved_runs, source, bank, compose, advanced, preview)
