"""Gradio entry point and event wiring for inkvoidmotif Studio."""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any

import gradio as gr
from PIL import Image

from . import studio
from .gui_layout import StudioComponents, build_interface
from .gui_theme import CSS, HEAD, theme

Image.MAX_IMAGE_PIXELS = studio.MAX_SOURCE_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)


def _progress_adapter(progress: gr.Progress):
    return lambda fraction, message: progress(fraction, desc=message)


def _ui_error(exc: Exception) -> None:
    raise gr.Error(studio.friendly_error(exc), print_exception=False) from exc


def _require_quota_acknowledgement(acknowledged: bool, *, billable: bool = True) -> None:
    if billable and not acknowledged:
        raise studio.StudioError(
            "Confirm the provider privacy and quota notice before starting a model call."
        )


def ui_analyze(
    source: str | None,
    count: int,
    provider: str,
    text_model: str,
    notes: str,
    quota_acknowledged: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects this dependency.
):
    try:
        if not source:
            raise studio.StudioError("Upload a source painting first.")
        _require_quota_acknowledgement(quota_acknowledged)
        state, rows, summary, crops, plan = studio.analyze_painting(
            source,
            count,
            provider,
            text_model,
            notes,
            progress=_progress_adapter(progress),
        )
        status = (
            "### Motif plan ready\n\n"
            f"{len(rows)} motifs found. Review the crops and edit the plan if needed."
        )
        return (
            state,
            rows,
            summary,
            crops,
            plan,
            status,
            None,
            [],
            None,
            None,
            [],
            None,
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary converts provider failures.
        _ui_error(exc)


def ui_create_bank(
    state: dict[str, Any] | None,
    table: Any,
    uploaded_sheet: str | None,
    provider: str,
    image_model: str,
    sheet_size: str,
    quality: str,
    columns: int,
    quota_acknowledged: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects this dependency.
):
    try:
        _require_quota_acknowledgement(quota_acknowledged, billable=not bool(uploaded_sheet))
        state, sheet, gallery, bank, status = studio.create_motif_bank(
            state,
            table,
            uploaded_sheet,
            provider,
            image_model,
            sheet_size,
            quality,
            columns,
            progress=_progress_adapter(progress),
        )
        return (
            state,
            sheet,
            gallery,
            bank,
            status,
            gr.update(interactive=True),
            None,
            [],
            None,
            gr.update(value=None, visible=False),
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary converts provider failures.
        _ui_error(exc)


def ui_preview(void: float, convention: str, peak: float, spread: float, seed: int, size: str):
    try:
        return studio.preview_layout(void, convention, peak, spread, seed, size)
    except Exception as exc:  # noqa: BLE001 - UI boundary converts validation failures.
        _ui_error(exc)


def ui_compose(
    state: dict[str, Any] | None,
    table: Any,
    scene: str,
    title: str,
    void: float,
    convention: str,
    peak: float,
    spread: float,
    seed: int,
    scholar: bool,
    include_source: bool,
    style: str,
    provider: str,
    image_model: str,
    size: str,
    quality: str,
    quota_acknowledged: bool,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 - Gradio injects this dependency.
):
    try:
        _require_quota_acknowledgement(quota_acknowledged)
        state, output, gallery, report, archive, status = studio.compose_painting(
            state,
            table,
            scene,
            title,
            void,
            convention,
            peak,
            spread,
            seed,
            scholar,
            include_source,
            style,
            provider,
            image_model,
            size,
            quality,
            progress=_progress_adapter(progress),
        )
        return (
            state,
            output,
            gallery,
            report,
            gr.update(value=archive, visible=True),
            status,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary converts provider failures.
        _ui_error(exc)


def ui_run_choices():
    choices = studio.list_run_ids()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def ui_resume(run_id: str | None):
    try:
        data = studio.resume_run(run_id or "")
        has_bank = bool(data["bank"])
        return (
            data["state"],
            data["rows"],
            data["summary"],
            data["crops"],
            data["plan"],
            data["sheet"],
            data["motifs"],
            data["bank"],
            data["result"],
            data["history"],
            data["report"],
            gr.update(value=data["archive"], visible=True),
            f"### Run resumed\n\n`{run_id}` is ready to continue.",
            gr.update(interactive=True),
            gr.update(interactive=has_bank),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary converts validation failures.
        _ui_error(exc)


def _bind_events(demo: gr.Blocks, ui: StudioComponents, *, allow_saved_runs: bool) -> None:
    source = ui.source
    bank = ui.bank
    compose = ui.compose
    advanced = ui.advanced
    preview = ui.preview
    saved = ui.saved_runs

    analyze_event = source.analyze.click(
        ui_analyze,
        inputs=[
            source.image,
            source.motif_count,
            source.provider,
            advanced.text_model,
            source.planning_notes,
            source.quota_acknowledged,
        ],
        outputs=[
            ui.session,
            bank.motif_table,
            bank.source_summary,
            preview.crop_gallery,
            preview.plan_file,
            preview.status,
            preview.motif_sheet,
            preview.motif_gallery,
            preview.bank_file,
            preview.result,
            preview.history,
            preview.report_file,
            preview.download_run,
            bank.build,
            compose.generate,
            bank.upload,
        ],
        api_name="analyze_painting",
        scroll_to_output=True,
        concurrency_id="model_pipeline",
        concurrency_limit=1,
    )

    bank_event = bank.build.click(
        ui_create_bank,
        inputs=[
            ui.session,
            bank.motif_table,
            bank.upload,
            source.provider,
            advanced.image_model,
            bank.sheet_size,
            advanced.quality,
            bank.columns,
            source.quota_acknowledged,
        ],
        outputs=[
            ui.session,
            preview.motif_sheet,
            preview.motif_gallery,
            preview.bank_file,
            preview.status,
            compose.generate,
            preview.result,
            preview.history,
            preview.report_file,
            preview.download_run,
        ],
        api_name="create_motif_bank",
        scroll_to_output=True,
        concurrency_id="model_pipeline",
        concurrency_limit=1,
    )

    preview_inputs: list[Any] = [
        compose.target_void,
        compose.convention,
        compose.peak,
        compose.spread,
        compose.seed,
        advanced.output_size,
    ]
    for component in preview_inputs:
        component.change(
            ui_preview,
            inputs=preview_inputs,
            outputs=[preview.layout_image, preview.layout_caption],
            queue=False,
            trigger_mode="always_last",
            api_visibility="private",
        )

    compose_event = compose.generate.click(
        ui_compose,
        inputs=[
            ui.session,
            bank.motif_table,
            compose.scene,
            compose.title,
            compose.target_void,
            compose.convention,
            compose.peak,
            compose.spread,
            compose.seed,
            compose.scholar,
            compose.include_source,
            compose.style,
            source.provider,
            advanced.image_model,
            advanced.output_size,
            advanced.quality,
            source.quota_acknowledged,
        ],
        outputs=[
            ui.session,
            preview.result,
            preview.history,
            preview.report_file,
            preview.download_run,
            preview.status,
        ],
        api_name="compose_painting",
        scroll_to_output=True,
        concurrency_id="model_pipeline",
        concurrency_limit=1,
    )

    compose.cancel.click(
        fn=None,
        cancels=[analyze_event, bank_event, compose_event],
        queue=False,
        api_visibility="private",
    )
    bank.motif_table.input(
        lambda: gr.update(interactive=False),
        outputs=compose.generate,
        queue=False,
        api_visibility="private",
    )
    bank.upload.input(
        lambda: gr.update(interactive=False),
        outputs=compose.generate,
        queue=False,
        api_visibility="private",
    )
    saved.refresh.click(
        ui_run_choices,
        outputs=saved.selector,
        queue=False,
        api_visibility="private",
    )
    if allow_saved_runs:
        saved.resume.click(
            ui_resume,
            inputs=saved.selector,
            outputs=[
                ui.session,
                bank.motif_table,
                bank.source_summary,
                preview.crop_gallery,
                preview.plan_file,
                preview.motif_sheet,
                preview.motif_gallery,
                preview.bank_file,
                preview.result,
                preview.history,
                preview.report_file,
                preview.download_run,
                preview.status,
                bank.build,
                compose.generate,
                bank.upload,
            ],
            api_name="resume_run",
            scroll_to_output=True,
        )
    demo.load(
        studio.readiness_markdown,
        outputs=source.readiness,
        queue=False,
        api_visibility="private",
    )


def build_demo(*, allow_saved_runs: bool = True) -> gr.Blocks:
    """Build a queued Studio app without launching a server."""
    initial_map, initial_caption = studio.preview_layout(
        0.62, "top_sky", 0.68, 0.62, 7, "1024x1536"
    )
    with gr.Blocks(
        title="inkvoidmotif Studio",
        fill_width=True,
        delete_cache=(3600, 86400),
    ) as demo:
        ui = build_interface(
            allow_saved_runs=allow_saved_runs,
            initial_map=initial_map,
            initial_caption=initial_caption,
        )
        _bind_events(demo, ui, allow_saved_runs=allow_saved_runs)
    return demo.queue(max_size=12, default_concurrency_limit=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the inkvoidmotif Gradio studio.")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a temporary Gradio share link."
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser automatically."
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    studio.load_local_env()
    auth_user = os.environ.get("INKVOIDMOTIF_STUDIO_USER")
    auth_password = os.environ.get("INKVOIDMOTIF_STUDIO_PASSWORD")
    if bool(auth_user) != bool(auth_password):
        parser.error(
            "Set both INKVOIDMOTIF_STUDIO_USER and INKVOIDMOTIF_STUDIO_PASSWORD, or neither."
        )
    exposes_network = args.share or args.host not in {"127.0.0.1", "localhost", "::1"}
    if exposes_network and not (auth_user and auth_password):
        parser.error(
            "Refusing to expose paid generation endpoints without authentication. "
            "Set INKVOIDMOTIF_STUDIO_USER and INKVOIDMOTIF_STUDIO_PASSWORD before using --share "
            "or a non-loopback host."
        )
    studio.RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build_demo(allow_saved_runs=not exposes_network)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        share=args.share,
        auth=(auth_user, auth_password) if auth_user and auth_password else None,
        show_error=False,
        max_file_size="64mb",
        blocked_paths=[str((studio.PROJECT_ROOT / ".env").resolve())],
        theme=theme(),
        css=CSS,
        head=HEAD,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
