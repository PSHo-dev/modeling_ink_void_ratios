from __future__ import annotations

import base64
import binascii
import mimetypes
import os
import re
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# Retain Pillow's decompression-bomb protection. Large trusted museum scans can
# opt into a higher ceiling, while web-facing callers keep a finite default.
Image.MAX_IMAGE_PIXELS = int(os.environ.get("INKVOIDMOTIF_MAX_IMAGE_PIXELS", "120000000"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    return (value.strip("_") or "motif")[:80]


MAX_INPUT_IMAGE_EDGE = 2048
# Keep the raw-byte cap well below the OpenAI/Azure 6 MB request-payload cap so
# that base64 encoding (which inflates by ~33%) leaves plenty of headroom.
MAX_INPUT_IMAGE_BYTES = 3 * 1024 * 1024
MAX_GENERATED_IMAGE_BYTES = 64 * 1024 * 1024
WEB_FRIENDLY_INPUT_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def image_to_data_url(path: Path) -> str:
    """Return a base64 data URL for ``path``, downscaling / re-encoding large or
    non-web-friendly images so the API receives a sane payload.

    Files that are already JPEG/PNG/WebP, under :data:`MAX_INPUT_IMAGE_BYTES`,
    *and* have a long edge ≤ :data:`MAX_INPUT_IMAGE_EDGE` are passed through
    unchanged. Everything else (TIFF, BMP, very large JPEGs, or tall thin
    scans that exceed the dimension cap even at small byte sizes) is opened
    with PIL, downscaled so the long edge is at most
    :data:`MAX_INPUT_IMAGE_EDGE` px, and re-encoded as JPEG (or PNG if it has
    an alpha channel).
    """

    import io

    suffix = path.suffix.lower()
    size = path.stat().st_size
    if suffix in WEB_FRIENDLY_INPUT_EXTS and size <= MAX_INPUT_IMAGE_BYTES:
        with Image.open(path) as probe:
            probe_w, probe_h = probe.size
        if max(probe_w, probe_h) <= MAX_INPUT_IMAGE_EDGE:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"

    with Image.open(path) as image:
        has_alpha = image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        )
        image = image.convert("RGBA" if has_alpha else "RGB")
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > MAX_INPUT_IMAGE_EDGE:
            ratio = MAX_INPUT_IMAGE_EDGE / long_edge
            image = image.resize(
                (max(1, int(width * ratio)), max(1, int(height * ratio))),
                Image.LANCZOS,
            )

        buffer = io.BytesIO()
        if has_alpha:
            image.save(buffer, format="PNG", optimize=True)
            mime = "image/png"
        else:
            image.save(buffer, format="JPEG", quality=90, optimize=True)
            mime = "image/jpeg"
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"


def decode_b64_image(b64_json: str, output_path: Path) -> Path:
    try:
        payload = base64.b64decode(b64_json, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("The provider returned invalid image data.") from exc
    if not payload or len(payload) > MAX_GENERATED_IMAGE_BYTES:
        raise RuntimeError("The provider returned an empty or oversized image.")
    ensure_dir(output_path.parent)
    temporary = output_path.with_name(f".{output_path.name}.partial")
    temporary.write_bytes(payload)
    try:
        with Image.open(temporary) as probe:
            probe.verify()
        temporary.replace(output_path)
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("The provider returned unreadable image data.") from None
    return output_path


def expanded_crop_box(
    image_size: tuple[int, int],
    bbox: list[float],
    margin_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    width, height = image_size
    if len(bbox) != 4:
        raise ValueError(f"bbox must have four values, got {bbox!r}")

    x, y, w, h = bbox
    normalized = max(abs(v) for v in bbox) <= 1.5
    if normalized:
        x *= width
        w *= width
        y *= height
        h *= height

    margin = max(w, h) * margin_ratio
    left = max(0, round(x - margin))
    top = max(0, round(y - margin))
    right = min(width, round(x + w + margin))
    bottom = min(height, round(y + h + margin))

    if right <= left or bottom <= top:
        raise ValueError(f"bbox produced an empty crop: {bbox!r}")
    return left, top, right, bottom


def crop_source_image(
    source_path: Path,
    bbox: list[float] | None,
    output_path: Path,
    margin_ratio: float = 0.08,
) -> Path:
    ensure_dir(output_path.parent)
    with Image.open(source_path) as image:
        if bbox:
            box = expanded_crop_box(image.size, bbox, margin_ratio)
            crop = image.crop(box)
        else:
            crop = image.copy()
        crop.save(output_path)
    return output_path


def make_contact_sheet(
    motif_paths: Iterable[Path],
    output_path: Path,
    thumb_size: tuple[int, int] = (420, 320),
    columns: int = 2,
) -> Path:
    motif_paths = list(motif_paths)
    if not motif_paths:
        raise ValueError("Cannot make a contact sheet without motif images.")

    label_height = 46
    padding = 18
    rows = (len(motif_paths) + columns - 1) // columns
    sheet_width = columns * thumb_size[0] + (columns + 1) * padding
    sheet_height = rows * (thumb_size[1] + label_height) + (rows + 1) * padding

    sheet = Image.new("RGB", (sheet_width, sheet_height), "#f7f3e8")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, path in enumerate(motif_paths):
        row, column = divmod(index, columns)
        x = padding + column * (thumb_size[0] + padding)
        y = padding + row * (thumb_size[1] + label_height + padding)

        with Image.open(path) as image:
            image = image.convert("RGBA")
            image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            tile = Image.new("RGBA", thumb_size, (255, 255, 255, 0))
            offset = (
                (thumb_size[0] - image.width) // 2,
                (thumb_size[1] - image.height) // 2,
            )
            tile.alpha_composite(image, offset)

        tile_bg = Image.new("RGBA", thumb_size, "#fffdf6")
        tile_bg.alpha_composite(tile)
        sheet.paste(tile_bg.convert("RGB"), (x, y))
        label = path.stem
        draw.text((x, y + thumb_size[1] + 12), label, fill="#222222", font=font)

    ensure_dir(output_path.parent)
    sheet.save(output_path)
    return output_path


def _background_mask_from_edges(
    image: Image.Image,
    tolerance: int = 34,
    min_brightness: int = 198,
) -> set[tuple[int, int]]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()

    border_samples: list[tuple[int, int, int]] = []
    step = max(1, min(width, height) // 80)
    for x in range(0, width, step):
        border_samples.append(pixels[x, 0])
        border_samples.append(pixels[x, height - 1])
    for y in range(0, height, step):
        border_samples.append(pixels[0, y])
        border_samples.append(pixels[width - 1, y])

    bg = tuple(
        sum(sample[channel] for sample in border_samples) // len(border_samples)
        for channel in range(3)
    )

    def is_background(x: int, y: int) -> bool:
        pixel = pixels[x, y]
        if sum(pixel) // 3 < min_brightness:
            return False
        return sum(abs(pixel[channel] - bg[channel]) for channel in range(3)) <= tolerance * 3

    queue: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))

    background: set[tuple[int, int]] = set()
    while queue:
        x, y = queue.pop()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if not is_background(x, y):
            continue
        background.add((x, y))
        queue.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    return background


def remove_connected_background(
    image: Image.Image,
    tolerance: int = 34,
    min_brightness: int = 198,
) -> Image.Image:
    rgba = image.convert("RGBA")
    background = _background_mask_from_edges(
        rgba, tolerance=tolerance, min_brightness=min_brightness
    )
    pixels = rgba.load()
    for x, y in background:
        r, g, b, _ = pixels[x, y]
        pixels[x, y] = (r, g, b, 0)
    return rgba


def trim_alpha(image: Image.Image, padding: int = 12) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return rgba
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgba.width, right + padding)
    bottom = min(rgba.height, bottom + padding)
    return rgba.crop((left, top, right, bottom))


def clear_alpha_edges(image: Image.Image, edge_width: int = 2) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    for x in range(rgba.width):
        for y in range(min(edge_width, rgba.height)):
            r, g, b, _ = pixels[x, y]
            pixels[x, y] = (r, g, b, 0)
            r, g, b, _ = pixels[x, rgba.height - 1 - y]
            pixels[x, rgba.height - 1 - y] = (r, g, b, 0)
    for y in range(rgba.height):
        for x in range(min(edge_width, rgba.width)):
            r, g, b, _ = pixels[x, y]
            pixels[x, y] = (r, g, b, 0)
            r, g, b, _ = pixels[rgba.width - 1 - x, y]
            pixels[rgba.width - 1 - x, y] = (r, g, b, 0)
    return rgba


def slice_motif_bank_sheet(
    sheet_path: Path,
    motif_names: list[str],
    output_dir: Path,
    columns: int = 2,
    cell_padding_ratio: float = 0.02,
    background_tolerance: int = 34,
    min_background_brightness: int = 198,
    remove_background: bool = False,
) -> list[Path]:
    """Slice a motif-bank contact sheet into per-motif PNGs using a simple grid.

    The motif sheet is generated without text captions, so the slicer just
    cuts an even ``columns × rows`` grid and writes each cell to disk. Motif
    names come from the planning step, not the image, so no caption detection
    is needed. A small ``cell_padding_ratio`` is shaved off each cell to avoid
    capturing the seam between neighbouring cells.

    Pass ``remove_background=True`` to opt back into the legacy flood-fill
    alpha extraction. It can eat pale brushwork (snow, mist) so it's off by
    default.
    """

    if columns < 1:
        raise ValueError("columns must be at least 1")
    ensure_dir(output_dir)
    rows = (len(motif_names) + columns - 1) // columns
    if rows < 1:
        raise ValueError("motif_names cannot be empty")

    output_paths: list[Path] = []
    with Image.open(sheet_path) as sheet:
        sheet = sheet.convert("RGB")
        cell_width = sheet.width / columns
        cell_height = sheet.height / rows
        x_pad = round(cell_width * cell_padding_ratio)
        y_pad = round(cell_height * cell_padding_ratio)

        for index, name in enumerate(motif_names, start=1):
            row, column = divmod(index - 1, columns)
            left = round(column * cell_width) + x_pad
            top = round(row * cell_height) + y_pad
            right = round((column + 1) * cell_width) - x_pad
            bottom = round((row + 1) * cell_height) - y_pad
            tile = sheet.crop((left, top, right, bottom))

            output_path = output_dir / f"{index:02d}_{slugify(name)}.png"
            if remove_background:
                transparent = remove_connected_background(
                    tile,
                    tolerance=background_tolerance,
                    min_brightness=min_background_brightness,
                )
                transparent = trim_alpha(transparent, padding=14)
                transparent = clear_alpha_edges(transparent, edge_width=2)
                transparent.save(output_path)
            else:
                tile.save(output_path)
            output_paths.append(output_path)

    return output_paths
