"""Validated application service layer for the inkvoidmotif Gradio studio.

The browser never receives provider credentials. Every run is isolated on disk,
and billable provider calls remain behind this testable service boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from . import layout as composition_layout
from .image_ops import crop_source_image, ensure_dir, make_contact_sheet, slugify
from .openai_api import nvidia_api_key
from .pipeline import (
    compose_with_layout,
    default_image_model,
    default_text_model,
    generate_motif_panel,
    import_chatgpt_motif_bank,
    load_local_env,
    plan_motifs,
    read_json,
    write_json,
)

load_local_env()

PROJECT_ROOT = Path.cwd().resolve()
RUNS_ROOT = (
    Path(os.environ.get("INKVOIDMOTIF_STUDIO_RUNS", PROJECT_ROOT / "runs" / "studio"))
    .expanduser()
    .resolve()
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_SOURCE_PIXELS = 60_000_000
MAX_API_EDGE = 2048
PAPER_RGB = (240, 235, 219)
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{8}")
Progress = Callable[[float, str], None]


class StudioError(ValueError):
    """An input or workflow error safe to show to an end user."""


def _progress(callback: Progress | None, fraction: float, message: str) -> None:
    if callback:
        callback(float(fraction), message)


def _real_key(value: str | None, prefixes: Iterable[str]) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    if any(marker in normalized for marker in ("your-key", "replace-me", "example")):
        return False
    return any(value.startswith(prefix) for prefix in prefixes)


def provider_readiness() -> dict[str, bool]:
    """Return key presence without exposing credential values."""
    load_local_env()
    return {
        "openai": _real_key(os.environ.get("OPENAI_API_KEY"), ("sk-",)),
        "nvidia": _real_key(nvidia_api_key(), ("nvapi-",)),
    }


def resolve_provider(choice: str | None) -> str:
    ready = provider_readiness()
    normalized = (choice or "auto").strip().lower()
    if normalized == "auto":
        configured = (
            (
                os.environ.get("INKVOIDMOTIF_IMAGE_PROVIDER")
                or os.environ.get("INKVOIDMOTIF_PROVIDER")
                or ""
            )
            .strip()
            .lower()
        )
        normalized = (
            configured if configured in ready else ("nvidia" if ready["nvidia"] else "openai")
        )
    if normalized not in ready:
        raise StudioError(f"Unsupported provider: {choice!r}.")
    if not ready[normalized]:
        env_name = "NVIDIA_API_KEY" if normalized == "nvidia" else "OPENAI_API_KEY"
        raise StudioError(
            f"{normalized.title()} is not configured. Add {env_name} to the project's "
            ".env file, then restart the studio."
        )
    return normalized


def readiness_markdown() -> str:
    ready = provider_readiness()

    def badge(name: str) -> str:
        return f"`{'configured' if ready[name] else 'not configured'}`"

    return (
        f"OpenAI {badge('openai')}  ·  NVIDIA {badge('nvidia')}  ·  credentials stay on this server"
    )


def _new_run_dir() -> Path:
    now = datetime.now(timezone.utc)
    return ensure_dir(RUNS_ROOT / f"{now:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _state_run_dir(state: dict[str, Any]) -> Path:
    value = state.get("run_dir")
    if not value:
        raise StudioError("Analyze a source painting first.")
    run_dir = Path(str(value)).resolve()
    if (
        run_dir.parent != RUNS_ROOT.resolve()
        or not RUN_ID_PATTERN.fullmatch(run_dir.name)
        or not run_dir.is_dir()
    ):
        raise StudioError("This studio run is invalid or no longer available.")
    return run_dir


def _state_file(state: dict[str, Any], key: str, run_dir: Path) -> Path:
    value = state.get(key)
    if not value:
        raise StudioError(f"This studio run has no {key.replace('_', ' ')}.")
    path = Path(str(value)).resolve()
    if not _inside(path, run_dir) or not path.is_file():
        raise StudioError("A run artifact is invalid or no longer available.")
    return path


def _validate_image(path: Path) -> tuple[int, int]:
    if not path.exists() or not path.is_file():
        raise StudioError("Choose an image file before continuing.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise StudioError("Use a JPG, PNG, WebP, TIFF, or BMP image.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise StudioError("The image is larger than 64 MB. Export a smaller copy first.")
    try:
        with Image.open(path) as probe:
            width, height = probe.size
            if width < 128 or height < 128:
                raise StudioError("The image is too small. Use at least 128 × 128 pixels.")
            if width * height > MAX_SOURCE_PIXELS:
                raise StudioError("The image exceeds 60 megapixels. Export a smaller copy first.")
            probe.verify()
    except StudioError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise StudioError("The uploaded file could not be read safely as an image.") from exc
    return width, height


def normalize_upload(
    source: str | Path, destination: Path, *, max_edge: int = MAX_API_EDGE
) -> Path:
    """Decode, orient, resize, and strip metadata from an uploaded image."""
    source = Path(source)
    _validate_image(source)
    ensure_dir(destination.parent)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            base = Image.new("RGBA", rgba.size, (*PAPER_RGB, 255))
            image = Image.alpha_composite(base, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        temporary = destination.with_name(f".{destination.name}.partial")
        if destination.suffix.lower() == ".png":
            image.save(temporary, "PNG", optimize=True)
        else:
            image.save(temporary, "JPEG", quality=92, optimize=True)
        temporary.replace(destination)
    return destination


def _table_rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if hasattr(value, "values") and hasattr(value.values, "tolist"):
        return value.values.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return value
    raise StudioError("The motif table has an unsupported format.")


def _bbox_text(bbox: Any) -> str:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return ""
    return ", ".join(f"{float(value):.3f}" for value in bbox)


def plan_to_rows(plan: dict[str, Any]) -> list[list[str]]:
    return [
        [
            str(item.get("name", "motif")),
            str(item.get("description", "")),
            str(item.get("role", "motif")),
            _bbox_text(item.get("bbox")),
            "; ".join(str(value) for value in item.get("must_preserve", [])),
        ]
        for item in plan.get("motifs", [])
    ]


def rows_to_motifs(value: Any) -> list[dict[str, Any]]:
    rows = _table_rows(value)
    if not 2 <= len(rows) <= 12:
        raise StudioError("Keep between 2 and 12 motifs in the plan.")
    motifs: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, row in enumerate(rows, start=1):
        cells = list(row) + [""] * (5 - len(row))
        raw_name = str(cells[0] or f"motif_{index}").strip()
        if len(raw_name) > 120:
            raise StudioError(f"Motif {index}'s name is too long.")
        name = slugify(raw_name)
        if name in used_names:
            raise StudioError(f"Motif names must be unique; {name!r} appears more than once.")
        used_names.add(name)
        description = str(cells[1] or "").strip()
        if not description:
            raise StudioError(f"Motif {index} needs a description.")
        if len(description) > 800:
            raise StudioError(f"Motif {index}'s description is longer than 800 characters.")
        try:
            bbox = [
                float(part) for part in re.split(r"[\s,]+", str(cells[3] or "").strip()) if part
            ]
        except ValueError as exc:
            raise StudioError(f"Motif {index} has an invalid bounding box.") from exc
        if len(bbox) != 4 or not all(math.isfinite(number) for number in bbox):
            raise StudioError(f"Motif {index} needs four finite bbox values: x, y, width, height.")
        x, y, width, height = bbox
        if min(x, y, width, height) < 0 or x + width > 1.0001 or y + height > 1.0001:
            raise StudioError(f"Motif {index}'s bbox must stay inside the normalized 0–1 canvas.")
        if width <= 0 or height <= 0:
            raise StudioError(f"Motif {index}'s bbox width and height must be positive.")
        preserve = [part.strip()[:200] for part in str(cells[4] or "").split(";") if part.strip()][
            :12
        ]
        motifs.append(
            {
                "name": name,
                "description": description,
                "role": str(cells[2] or "motif").strip()[:120],
                "bbox": [round(number, 6) for number in bbox],
                "must_preserve": preserve,
            }
        )
    return motifs


def _plan_hash(motifs: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        motifs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_table_plan(state: dict[str, Any], table: Any) -> Path:
    run_dir = _state_run_dir(state)
    plan_path = _state_file(state, "plan_path", run_dir)
    payload = read_json(plan_path)
    payload["motifs"] = rows_to_motifs(table)
    payload["edited_at"] = datetime.now(timezone.utc).isoformat()
    return write_json(plan_path, payload)


def _crop_gallery(source: Path, plan_path: Path, run_dir: Path) -> list[tuple[str, str]]:
    plan = read_json(plan_path)
    crops_dir = ensure_dir(run_dir / "source_crops")
    paths: list[Path] = []
    captions: list[str] = []
    for index, motif in enumerate(plan.get("motifs", []), start=1):
        name = slugify(str(motif.get("name", f"motif_{index}")))
        crop_path = crops_dir / f"{index:02d}_{name}.png"
        crop_source_image(source, motif.get("bbox"), crop_path, margin_ratio=0.06)
        paths.append(crop_path)
        captions.append(str(motif.get("name", name)))
    if paths:
        make_contact_sheet(paths, run_dir / "source_crop_contact_sheet.png")
    return [(str(path), caption) for path, caption in zip(paths, captions, strict=False)]


def _validate_model_name(value: str | None) -> str:
    model = (value or "").strip()
    if len(model) > 200 or any(ord(char) < 32 for char in model):
        raise StudioError("The model override is invalid.")
    return model


def analyze_painting(
    source_upload: str | Path,
    motif_count: int,
    provider_choice: str,
    text_model: str | None = None,
    instructions: str | None = None,
    progress: Progress | None = None,
) -> tuple[dict[str, Any], list[list[str]], str, list[tuple[str, str]], str]:
    provider = resolve_provider(provider_choice)
    count = int(motif_count)
    if not 2 <= count <= 12:
        raise StudioError("Choose between 2 and 12 motifs.")
    if instructions and len(instructions) > 2000:
        raise StudioError("Planning notes must be 2,000 characters or fewer.")
    model = _validate_model_name(text_model) or default_text_model(provider)
    run_dir = _new_run_dir()
    try:
        _progress(progress, 0.05, "Preparing the source painting")
        source = normalize_upload(source_upload, run_dir / "source.jpg")
        _progress(progress, 0.20, "Reading the composition and identifying motifs")
        plan_path = plan_motifs(
            source_path=source,
            output_dir=run_dir,
            count=count,
            model=model,
            provider=provider,
            instructions=(instructions or "").strip() or None,
        )
        plan = read_json(plan_path)
        rows = plan_to_rows(plan)
        motifs = rows_to_motifs(rows)
        _progress(progress, 0.82, "Preparing crop previews")
        crops = _crop_gallery(source, plan_path, run_dir)
        state = {
            "version": 2,
            "run_dir": str(run_dir),
            "source_path": str(source),
            "plan_path": str(plan_path),
            "plan_hash": _plan_hash(motifs),
            "sheet_path": None,
            "bank_path": None,
            "bank_plan_hash": None,
            "outputs": [],
            "provider": provider,
        }
        write_json(run_dir / "studio_session.json", state)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    _progress(progress, 1.0, "Motif plan ready")
    return (
        state,
        rows,
        str(plan.get("source_summary", "Motif plan ready.")),
        crops,
        str(plan_path),
    )


def create_motif_bank(
    state: dict[str, Any] | None,
    motif_table: Any,
    uploaded_sheet: str | Path | None,
    provider_choice: str,
    image_model: str | None,
    sheet_size: str,
    quality: str,
    columns: int,
    progress: Progress | None = None,
) -> tuple[dict[str, Any], str, list[tuple[str, str]], str, str]:
    if not state:
        raise StudioError("Analyze a source painting first.")
    run_dir = _state_run_dir(state)
    source = _state_file(state, "source_path", run_dir)
    motifs = rows_to_motifs(motif_table)
    plan_hash = _plan_hash(motifs)
    plan_path = save_table_plan(state, motif_table)
    if sheet_size not in {"1024x1024", "1024x1536", "1536x1024"}:
        raise StudioError("Choose a supported motif-sheet size.")
    if quality not in {"low", "medium", "high", "auto"}:
        raise StudioError("Choose a supported image quality.")
    columns = int(columns)
    if not 1 <= columns <= min(4, len(motifs)):
        raise StudioError("Grid columns must be between 1 and the motif count (maximum 4).")

    bank_dir = ensure_dir(run_dir / "banks" / f"{plan_hash[:12]}-{uuid4().hex[:6]}")
    sheet_path = bank_dir / "motif_sheet.png"
    try:
        if uploaded_sheet:
            _progress(progress, 0.15, "Importing the motif sheet")
            normalize_upload(uploaded_sheet, sheet_path, max_edge=2048)
            provider = "uploaded"
            model = "user-supplied motif sheet"
            source_label = "uploaded"
        else:
            provider = resolve_provider(provider_choice)
            model = _validate_model_name(image_model) or default_image_model(provider)
            _progress(progress, 0.12, "Generating the motif sheet")
            generate_motif_panel(
                source_path=source,
                plan_path=plan_path,
                output_path=sheet_path,
                model=model,
                provider=provider,
                size=sheet_size,
                quality=quality,
                columns=columns,
            )
            _validate_image(sheet_path)
            source_label = "generated"
        _progress(progress, 0.82, "Slicing motif review cells")
        imported = import_chatgpt_motif_bank(
            sheet_path=sheet_path,
            plan_path=plan_path,
            output_dir=bank_dir,
            columns=columns,
            remove_background=False,
            image_provider=provider,
            image_model=model,
            source_kind=source_label,
        )
        bank_path = Path(imported["motif_bank"])
        bank = read_json(bank_path)
        gallery = []
        for item in bank.get("motifs", []):
            asset = Path(str(item["asset_path"]))
            if not asset.is_absolute():
                asset = bank_path.parent / asset
            gallery.append((str(asset), str(item.get("name", "Motif"))))
        state = {
            **state,
            "plan_hash": plan_hash,
            "sheet_path": str(sheet_path),
            "bank_path": str(bank_path),
            "bank_plan_hash": plan_hash,
            "outputs": [],
            "provider": provider_choice if provider == "uploaded" else provider,
        }
        write_json(run_dir / "studio_session.json", state)
    except Exception as exc:
        write_json(
            bank_dir / "failure.json",
            {
                "stage": "motif_bank",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
            },
        )
        raise
    _progress(progress, 1.0, "Motif bank ready")
    status = f"### Motif bank ready\n\n{len(motifs)} review cells · {source_label} sheet"
    return state, str(sheet_path), gallery, str(bank_path), status


def _parse_size(size: str) -> tuple[int, int]:
    try:
        width, height = (int(value) for value in size.lower().split("x", 1))
    except (AttributeError, TypeError, ValueError) as exc:
        raise StudioError("Image size must look like 1024x1536.") from exc
    if (width, height) not in {(1024, 1024), (1024, 1536), (1536, 1024)}:
        raise StudioError("Choose 1024×1024, 1024×1536, or 1536×1024.")
    return width, height


def _layout_values(
    target_void: float, convention: str, peak_scale: float, spread: float, seed: int
) -> tuple[float, str, float, float, int]:
    values = [float(target_void), float(peak_scale), float(spread), float(seed)]
    if not all(math.isfinite(value) for value in values):
        raise StudioError("Layout controls must be finite numbers.")
    if not 0.25 <= values[0] <= 0.85:
        raise StudioError("Void ratio must be between 0.25 and 0.85.")
    if convention not in {"top_sky", "diagonal", "river_band"}:
        raise StudioError("Choose a supported composition structure.")
    if not 0.25 <= values[1] <= 1.25 or not 0 <= values[2] <= 1:
        raise StudioError("Peak scale or distance spread is outside its supported range.")
    seed_value = int(values[3])
    if values[3] != seed_value or not 0 <= seed_value <= 999:
        raise StudioError("Layout seed must be a whole number from 0 to 999.")
    return values[0], convention, values[1], values[2], seed_value


def build_composition_guide(
    target_void: float,
    convention: str,
    peak_scale: float,
    spread: float,
    seed: int,
    width: int = 512,
    height: int = 768,
    *,
    label: bool = False,
) -> tuple[Image.Image, float]:
    """Render the exact layout representation used by the production composer."""
    target_void, convention, peak_scale, spread, seed = _layout_values(
        target_void, convention, peak_scale, spread, seed
    )
    layout = composition_layout.build_mass_void_layout(
        width,
        height,
        target_void,
        convention=convention,
        peak_scale=peak_scale,
        spread=spread,
        seed=seed,
    )
    image = composition_layout.render_layout_image(layout, width, height, label=label)
    return image, float(layout["achieved_void"])


CONVENTION_LABELS = {
    "top_sky": "High distance · open sky",
    "diagonal": "Diagonal ascent",
    "river_band": "Winding river corridor",
}


def preview_layout(
    target_void: float,
    convention: str,
    peak_scale: float,
    spread: float,
    seed: int,
    output_size: str,
) -> tuple[Image.Image, str]:
    width, height = _parse_size(output_size)
    preview_width = 460 if width <= height else 690
    preview_height = max(320, round(preview_width * height / width))
    image, achieved = build_composition_guide(
        target_void,
        convention,
        peak_scale,
        spread,
        seed,
        preview_width,
        preview_height,
        label=True,
    )
    caption = (
        f"**Production composition map** · {CONVENTION_LABELS[convention]} · "
        f"{achieved * 100:.0f}% open silk · {(1 - achieved) * 100:.0f}% painted mass"
    )
    return image, caption


def estimate_void_ratio(image_path: Path) -> float:
    """Return the pipeline's region-based void measurement."""
    return float(composition_layout.region_void(image_path)["void"])


def _next_output_path(run_dir: Path, title: str) -> Path:
    output_dir = ensure_dir(run_dir / "compositions")
    stem = slugify(title) if title.strip() else "composition"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return output_dir / f"{stem}_{stamp}_{uuid4().hex[:6]}.png"


def _portable_value(value: Any, run_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _portable_value(item, run_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item, run_dir) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute() and _inside(candidate, run_dir):
            return candidate.resolve().relative_to(run_dir).as_posix()
    return value


def archive_run(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    if (
        run_dir.parent != RUNS_ROOT.resolve()
        or not RUN_ID_PATTERN.fullmatch(run_dir.name)
        or not run_dir.is_dir()
    ):
        raise StudioError("Cannot archive an invalid studio run.")
    archive = run_dir / f"{run_dir.name}.zip"
    files = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and not path.is_symlink() and path != archive and path.name != ".env"
    ]
    manifest: dict[str, Any] = {"format": 1, "files": {}}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            relative = path.relative_to(run_dir).as_posix()
            payload = path.read_bytes()
            if path.suffix.lower() == ".json":
                try:
                    portable = _portable_value(json.loads(payload), run_dir)
                    payload = (
                        json.dumps(
                            portable,
                            indent=2,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    pass
            manifest["files"][relative] = hashlib.sha256(payload).hexdigest()
            bundle.writestr(relative, payload)
        bundle.writestr("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return archive


def compose_painting(
    state: dict[str, Any] | None,
    motif_table: Any,
    theme: str,
    title: str,
    target_void: float,
    convention: str,
    peak_scale: float,
    spread: float,
    seed: int,
    include_scholar: bool,
    include_source: bool,
    source_style: str,
    provider_choice: str,
    image_model: str | None,
    output_size: str,
    quality: str,
    progress: Progress | None = None,
) -> tuple[dict[str, Any], str, list[tuple[str, str]], str, str, str]:
    if not state:
        raise StudioError("Analyze a source painting first.")
    run_dir = _state_run_dir(state)
    bank_path = _state_file(state, "bank_path", run_dir)
    plan_path = _state_file(state, "plan_path", run_dir)
    current_hash = _plan_hash(rows_to_motifs(motif_table))
    if current_hash != state.get("bank_plan_hash"):
        raise StudioError(
            "The motif plan changed after this bank was built. Rebuild the motif bank before composing."
        )
    theme = (theme or "").strip()
    title = (title or "").strip()
    if len(theme) < 12 or len(theme) > 1800:
        raise StudioError("Describe the new scene in 12 to 1,800 characters.")
    if len(title) > 90 or len(source_style or "") > 1200:
        raise StudioError("Output name or style notes are too long.")
    target_void, convention, peak_scale, spread, seed = _layout_values(
        target_void, convention, peak_scale, spread, seed
    )
    provider = resolve_provider(provider_choice)
    _parse_size(output_size)
    if quality not in {"low", "medium", "high", "auto"}:
        raise StudioError("Choose a supported image quality.")
    model = _validate_model_name(image_model) or default_image_model(provider)
    output_path = _next_output_path(run_dir, title or theme[:50])
    source_path = _state_file(state, "source_path", run_dir) if include_source else None
    _progress(progress, 0.08, "Building the production void and landmass guide")
    try:
        _progress(progress, 0.20, "Painting a new composition")
        compose_with_layout(
            bank_path=bank_path,
            output_path=output_path,
            theme=theme,
            target_void=target_void,
            convention=convention,
            peak_scale=peak_scale,
            spread=spread,
            seed=seed,
            image_model=model,
            image_provider=provider,
            size=output_size,
            quality=quality,
            source_path=source_path,
            source_style=(source_style or "").strip() or None,
            include_scholar=bool(include_scholar),
        )
        _validate_image(output_path)
    except Exception as exc:
        write_json(
            output_path.with_suffix(".failure.json"),
            {
                "stage": "composition",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
            },
        )
        raise
    _progress(progress, 0.90, "Verifying open-silk balance")
    report_path = output_path.with_suffix(".layout_report.json")
    if report_path.exists():
        report = read_json(report_path)
    else:
        report = {
            "method": "composition_layout",
            "target_void": target_void,
            "realized_void_region": estimate_void_ratio(output_path),
        }
    report.update(
        {
            "studio_version": 2,
            "motif_plan": str(plan_path),
            "motif_bank": str(bank_path),
            "plan_hash": current_hash,
        }
    )
    report_path = write_json(report_path, report)
    outputs = [*state.get("outputs", []), str(output_path)]
    state = {**state, "outputs": outputs, "provider": provider}
    write_json(run_dir / "studio_session.json", state)
    archive = archive_run(run_dir)
    gallery = [(path, Path(path).stem.replace("_", " ").title()) for path in reversed(outputs)]
    realized_void = float(report.get("realized_void_region", estimate_void_ratio(output_path)))
    delta = realized_void - target_void
    status = (
        "### Painting ready\n\n"
        f"Target void **{target_void * 100:.0f}%** · measured region void "
        f"**{realized_void * 100:.0f}%** · delta **{delta * 100:+.0f} pp**\n\n"
        "Measurement uses the pipeline's region-based morphology metric; inspect the occupancy preview too."
    )
    _progress(progress, 1.0, "Painting ready")
    return state, str(output_path), gallery, str(report_path), str(archive), status


def list_run_ids() -> list[str]:
    if not RUNS_ROOT.is_dir():
        return []
    return sorted(
        [
            path.name
            for path in RUNS_ROOT.iterdir()
            if path.is_dir() and RUN_ID_PATTERN.fullmatch(path.name)
        ],
        reverse=True,
    )


def resume_run(run_id: str) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch((run_id or "").strip()):
        raise StudioError("Choose a valid saved run.")
    run_dir = (RUNS_ROOT / run_id.strip()).resolve()
    if run_dir.parent != RUNS_ROOT.resolve():
        raise StudioError("Choose a valid saved run.")
    session_path = run_dir / "studio_session.json"
    if not session_path.is_file():
        raise StudioError("That run cannot be resumed because its session file is missing.")
    state = read_json(session_path)
    if Path(str(state.get("run_dir", ""))).resolve() != run_dir:
        raise StudioError("That saved session is invalid.")
    source = _state_file(state, "source_path", run_dir)
    plan_path = _state_file(state, "plan_path", run_dir)
    plan = read_json(plan_path)
    rows = plan_to_rows(plan)
    motifs = rows_to_motifs(rows)
    if state.get("bank_path") and not state.get("bank_plan_hash"):
        state["bank_plan_hash"] = _plan_hash(motifs)
        state["plan_hash"] = state["bank_plan_hash"]
        state["version"] = 2
        write_json(session_path, state)
    result = None
    outputs: list[str] = []
    for value in state.get("outputs", []):
        path = Path(str(value)).resolve()
        if _inside(path, run_dir) and path.is_file():
            outputs.append(str(path))
    state["outputs"] = outputs
    if outputs:
        result = outputs[-1]
    bank_file = None
    motif_sheet = None
    motif_gallery: list[tuple[str, str]] = []
    if state.get("bank_path"):
        bank_path = _state_file(state, "bank_path", run_dir)
        bank_file = str(bank_path)
        bank = read_json(bank_path)
        for item in bank.get("motifs", []):
            asset = Path(str(item["asset_path"]))
            if not asset.is_absolute():
                asset = bank_path.parent / asset
            if _inside(asset, run_dir) and asset.is_file():
                motif_gallery.append((str(asset), str(item.get("name", "Motif"))))
    if state.get("sheet_path"):
        motif_sheet = str(_state_file(state, "sheet_path", run_dir))
    report = str(Path(result).with_suffix(".layout_report.json")) if result else None
    archive = str(archive_run(run_dir))
    return {
        "state": state,
        "rows": rows,
        "summary": str(plan.get("source_summary", "Saved run")),
        "crops": _crop_gallery(source, plan_path, run_dir),
        "plan": str(plan_path),
        "sheet": motif_sheet,
        "motifs": motif_gallery,
        "bank": bank_file,
        "result": result,
        "history": [
            (path, Path(path).stem.replace("_", " ").title()) for path in reversed(outputs)
        ],
        "report": report if report and Path(report).is_file() else None,
        "archive": archive,
    }


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, StudioError):
        return str(exc)
    lowered = str(exc).lower()
    if any(
        word in lowered for word in ("401", "unauthorized", "incorrect api key", "authentication")
    ):
        return "The provider rejected the API key. Update .env and restart the studio."
    if any(word in lowered for word in ("429", "rate limit", "quota")):
        return (
            "The provider is rate-limited or out of quota. Wait briefly or choose another provider."
        )
    if any(word in lowered for word in ("timeout", "timed out", "connection")):
        return "The model request did not complete. Check the connection and try again; diagnostics were preserved."
    if "content" in lowered and any(word in lowered for word in ("policy", "moderation", "safety")):
        return "The provider declined this request. Revise the scene description and try again."
    return "The pipeline could not complete this stage. Safe diagnostic artifacts were preserved when a run existed."
