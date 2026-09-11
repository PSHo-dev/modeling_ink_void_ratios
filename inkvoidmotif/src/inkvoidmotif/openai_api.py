from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import os
import re
import socket
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .image_ops import decode_b64_image, image_to_data_url
from .schemas import MOTIF_PLAN_SCHEMA


def _load_dotenv() -> None:
    load_dotenv(override=False)


def _usable_secret(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized not in {"0", "false", "none", "null", "off"} and not any(
        marker in normalized for marker in ("your-key", "replace-me", "example")
    )


def openai_client() -> Any:
    _load_dotenv()
    if not _usable_secret(os.environ.get("OPENAI_API_KEY")):
        raise RuntimeError("Set OPENAI_API_KEY or put it in a local .env file.")
    from openai import OpenAI

    return OpenAI(timeout=_api_timeout(), max_retries=_api_retries())


def nvidia_api_key() -> str | None:
    _load_dotenv()
    key = (
        os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVAPI_KEY")
        or os.environ.get("NV_API_KEY")
    )
    if _usable_secret(key):
        return key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and openai_key.startswith("nvapi-"):
        return openai_key
    return None


def normalize_openai_base_url(base_url: str) -> str:
    """Trim NVIDIA gateway endpoint suffixes so the OpenAI SDK can append its own path.

    The NVIDIA LLM Gateway exposes routes at the host root (e.g. `/responses`,
    `/chat/completions`), not under `/v1`. Strip any of those tails so callers
    can paste a full URL and we still produce a base the SDK is happy with.
    """

    base_url = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/v1"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            base_url = base_url.rstrip("/")
    return base_url


def nvidia_client() -> Any:
    api_key = nvidia_api_key()
    if not api_key:
        raise RuntimeError(
            "Set NVIDIA_API_KEY, NVAPI_KEY, or NV_API_KEY in your environment or local .env file."
        )
    from openai import OpenAI

    base_url = normalize_openai_base_url(
        os.environ.get("NVIDIA_BASE_URL", "https://inference-api.nvidia.com")
    )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=_api_timeout(),
        max_retries=_api_retries(),
    )


def _api_timeout() -> float:
    try:
        value = float(os.environ.get("INKVOIDMOTIF_API_TIMEOUT", "300"))
    except ValueError:
        value = 300.0
    return min(max(value, 10.0), 900.0)


def _api_retries() -> int:
    try:
        value = int(os.environ.get("INKVOIDMOTIF_API_RETRIES", "1"))
    except ValueError:
        value = 1
    return min(max(value, 0), 3)


def provider_name(provider: str | None = None) -> str:
    _load_dotenv()
    selected = (
        provider
        or os.environ.get("INKVOIDMOTIF_PROVIDER")
        or os.environ.get("INKVOIDMOTIF_API_PROVIDER")
        or "openai"
    )
    return selected.strip().lower()


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    if chunks:
        return "\n".join(chunks)

    return str(response)


def log_usage(response: Any, label: str) -> None:
    """Print a one-line summary of token usage for an API response.

    Works for both Chat Completions and Responses-API objects, which use
    slightly different field names for usage stats.
    """

    usage = getattr(response, "usage", None)
    if usage is None:
        print(f"[inkvoidmotif] {label}: token usage unavailable")
        return

    parts: list[str] = []
    for attr, display in (
        ("prompt_tokens", "prompt"),
        ("input_tokens", "input"),
        ("completion_tokens", "completion"),
        ("output_tokens", "output"),
        ("total_tokens", "total"),
    ):
        value = getattr(usage, attr, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(attr)
        if value is not None:
            parts.append(f"{display}={value}")

    extras: list[str] = []
    for detail_attr in ("input_tokens_details", "output_tokens_details"):
        details = getattr(usage, detail_attr, None)
        if details is None and isinstance(usage, dict):
            details = usage.get(detail_attr)
        if not details:
            continue
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        if isinstance(details, dict):
            sub = ", ".join(f"{k}={v}" for k, v in details.items() if isinstance(v, (int, float)))
            if sub:
                extras.append(f"{detail_attr}={{{sub}}}")

    summary = ", ".join(parts) if parts else "n/a"
    if extras:
        summary = f"{summary}; {'; '.join(extras)}"
    print(f"[inkvoidmotif] {label} tokens: {summary}")


def extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return str(response)
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
            else:
                text = getattr(part, "text", None) or getattr(part, "content", None)
            if text:
                chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
    return str(response)


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def plan_with_vision(
    source_path: Path,
    prompt: str,
    model: str,
    quality_exemplar_path: Path | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    selected_provider = provider_name(provider)
    if selected_provider == "nvidia":
        client = nvidia_client()
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_to_data_url(source_path)}},
        ]
        if quality_exemplar_path:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(quality_exemplar_path)},
                }
            )
        plan_max_tokens = int(
            os.environ.get("NVIDIA_PLAN_MAX_TOKENS") or os.environ.get("NVIDIA_MAX_TOKENS", "16384")
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=float(os.environ.get("NVIDIA_TEMPERATURE", "0.2")),
            max_tokens=plan_max_tokens,
        )
        raw = extract_chat_text(response).strip()
        if not raw:
            usage = getattr(response, "usage", None)
            details = []
            for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, attr, None) if usage is not None else None
                if value is not None:
                    details.append(f"{attr}={value}")
            detail_str = ", ".join(details) if details else "no usage stats"
            raise RuntimeError(
                "Planning model returned empty text (reasoning tokens likely "
                f"exceeded max_tokens={plan_max_tokens}; {detail_str}). Set "
                "NVIDIA_PLAN_MAX_TOKENS higher and retry."
            )
        return parse_json_object(raw)

    if selected_provider != "openai":
        raise ValueError(f"Unsupported API provider: {selected_provider!r}")

    client = openai_client()
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": image_to_data_url(source_path)},
    ]
    if quality_exemplar_path:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(quality_exemplar_path),
            }
        )

    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "motif_plan",
                    "schema": MOTIF_PLAN_SCHEMA,
                    "strict": True,
                }
            },
        )
    except TypeError:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
        )

    return parse_json_object(extract_output_text(response))


@contextmanager
def opened_images(paths: list[Path]) -> Iterator[Any]:
    handles = [path.open("rb") for path in paths]
    try:
        yield handles[0] if len(handles) == 1 else handles
    finally:
        for handle in handles:
            handle.close()


DATA_URL_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=\s_-]+)")
HTTP_IMAGE_RE = re.compile(r"https?://[^\s)\"']+")


def to_plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


def decode_possible_image_b64(value: str) -> bytes | None:
    cleaned = re.sub(r"\s+", "", value)
    if len(cleaned) < 32:
        return None
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            data = decoder(cleaned)
        except (binascii.Error, ValueError):
            continue
        if data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"RIFF")):
            return data
    return None


def iter_image_strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_image_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_image_strings(item)
    elif isinstance(value, str):
        yield value


def extract_image_bytes(payload: Any) -> bytes | None:
    plain = to_plain_data(payload)
    for text in iter_image_strings(plain):
        for match in DATA_URL_RE.finditer(text):
            data = decode_possible_image_b64(match.group(1))
            if data:
                return data

    for text in iter_image_strings(plain):
        data = decode_possible_image_b64(text)
        if data:
            return data

    return None


def extract_image_url(payload: Any) -> str | None:
    plain = to_plain_data(payload)
    for text in iter_image_strings(plain):
        for match in HTTP_IMAGE_RE.finditer(text):
            url = match.group(0)
            lower = url.lower()
            if any(marker in lower for marker in (".png", ".jpg", ".jpeg", ".webp", "image")):
                return url.rstrip(".,")
    return None


def _validate_remote_image_url(url: str) -> None:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        raise RuntimeError("The provider returned an unsafe image URL.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except OSError as exc:
        raise RuntimeError("The provider image host could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise RuntimeError("The provider returned a non-public image URL.")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_remote_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_image(url: str, output_path: Path) -> Path:
    _validate_remote_image_url(url)
    ensure_parent = output_path.parent
    ensure_parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(  # noqa: S310 -- URL is HTTPS-only and DNS-validated above.
        url, headers={"User-Agent": "inkvoidmotif/0.1"}
    )
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    # URL scheme, DNS results, and every redirect are constrained above.
    with opener.open(request, timeout=120) as response:  # nosec B310
        content_length = response.headers.get("Content-Length")
        limit = 64 * 1024 * 1024
        if content_length and int(content_length) > limit:
            raise RuntimeError("The provider returned an oversized image.")
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise RuntimeError("The provider returned an oversized image.")
    return decode_b64_image(base64.b64encode(payload).decode("ascii"), output_path)


def write_response_debug(response: Any, output_path: Path) -> Path:
    debug_path = output_path.with_suffix(".response.json")
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(
        json.dumps(to_plain_data(response), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return debug_path


def image_edit_capability_error(error: Exception, model: str) -> RuntimeError:
    return RuntimeError(
        "The configured NVIDIA endpoint/key is reachable, but this image pipeline "
        "requires an OpenAI-compatible image editing endpoint that accepts input "
        "images. The current NVIDIA model/base URL rejected that operation. "
        f"Configured model: {model!r}. For NVIDIA, use a Visual GenAI NIM image-edit "
        "model such as qwen-image-edit-2509, qwen-image-edit-2511, or flux.2-klein-4b "
        "on an endpoint that exposes /v1/images/edits. The public "
        "azure/openai/gpt-image-2 chat-completions route can validate the key, but it "
        "does not satisfy this motif-extraction/editing workflow. "
        f"Provider error: {error}"
    )


def edit_image_with_images_api(
    client: Any,
    image_paths: list[Path],
    prompt: str,
    output_path: Path,
    model: str,
    size: str = "1024x1024",
    quality: str = "high",
    background: str | None = None,
    output_format: str = "png",
    input_fidelity: str | None = None,
) -> Path:
    kwargs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_format": output_format,
    }
    if background:
        kwargs["background"] = background
    if input_fidelity:
        kwargs["input_fidelity"] = input_fidelity

    with opened_images(image_paths) as image_arg:
        result = client.images.edit(image=image_arg, **kwargs)

    if not getattr(result, "data", None):
        raise RuntimeError("Image edit response did not include image data.")
    b64_json = result.data[0].b64_json
    return decode_b64_image(b64_json, output_path)


def edit_image_with_mask(
    image_path: Path,
    mask_path: Path,
    prompt: str,
    output_path: Path,
    model: str = "gpt-image-1",
    size: str = "1024x1536",
    quality: str = "high",
) -> Path:
    """Inpaint ``image_path`` only where ``mask_path`` is transparent (OpenAI).

    Uses the OpenAI Images *edit* endpoint, which is the only path here that
    supports a mask. ``mask`` and ``image`` must share dimensions; transparent
    mask pixels are repainted from ``prompt`` while opaque pixels are preserved.
    Requires a real OpenAI key (api.openai.com); the NVIDIA gateway has no mask.
    """
    client = openai_client()
    with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
        result = client.images.edit(
            model=model,
            image=image_file,
            mask=mask_file,
            prompt=prompt,
            size=size,
            quality=quality,
        )
    if not getattr(result, "data", None):
        raise RuntimeError("Masked image edit response did not include image data.")
    return decode_b64_image(result.data[0].b64_json, output_path)


def edit_image_with_nvidia_images_api(
    image_paths: list[Path],
    prompt: str,
    output_path: Path,
    model: str,
    size: str = "1024x1024",
    quality: str = "high",
    background: str | None = None,
    output_format: str = "png",
    input_fidelity: str | None = None,
) -> Path:
    try:
        return edit_image_with_images_api(
            client=nvidia_client(),
            image_paths=image_paths,
            prompt=prompt,
            output_path=output_path,
            model=model,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            input_fidelity=input_fidelity,
        )
    except Exception as error:
        raise image_edit_capability_error(error, model) from error


def edit_image_with_nvidia_responses(
    image_paths: list[Path],
    prompt: str,
    output_path: Path,
    model: str | None = None,
    size: str = "1024x1024",
    quality: str = "high",
    background: str | None = None,
    output_format: str = "png",
) -> Path:
    """Generate or edit an image via the NVIDIA LLM Gateway Responses API.

    Uses a mainline chat model (default ``openai/openai/gpt-5.5``) with the
    ``image_generation`` built-in tool. ``tool_choice`` is pinned to that tool
    so the model always returns an ``image_generation_call`` instead of plain
    text. This matches the working shape from the nvidia-image-gen skill.
    """

    client = nvidia_client()

    responses_model = model or os.environ.get("NVIDIA_RESPONSES_MODEL") or "openai/openai/gpt-5.5"

    tool: dict[str, Any] = {"type": "image_generation", "quality": quality}
    if size:
        tool["size"] = size
    if background:
        tool["background"] = background
    if output_format and output_format.lower() != "png":
        tool["output_format"] = output_format.lower()

    # Multimodal input when references are provided; text-only otherwise.
    if image_paths:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image_path in image_paths:
            content.append({"type": "input_image", "image_url": image_to_data_url(image_path)})
        api_input: Any = [{"role": "user", "content": content}]
    else:
        api_input = prompt

    response = client.responses.create(
        model=responses_model,
        input=api_input,
        tools=[tool],
        tool_choice={"type": "image_generation"},
    )

    output_items = list(getattr(response, "output", []) or [])
    for item in output_items:
        if getattr(item, "type", None) != "image_generation_call":
            continue
        result = getattr(item, "result", None)
        if not result:
            continue
        revised = getattr(item, "revised_prompt", None)
        if revised:
            try:
                output_path.with_suffix(".revised.txt").parent.mkdir(parents=True, exist_ok=True)
                output_path.with_suffix(".revised.txt").write_text(str(revised))
            except OSError:
                pass
        return decode_b64_image(result, output_path)

    debug_path = write_response_debug(response, output_path)
    raise RuntimeError(
        "NVIDIA Responses API did not return an image_generation_call output. "
        f"Raw response saved to {debug_path}."
    )


def edit_image_with_nvidia_chat(
    image_paths: list[Path],
    prompt: str,
    output_path: Path,
    model: str,
    size: str = "1024x1024",
    quality: str = "high",
    background: str | None = None,
    output_format: str = "png",
) -> Path:
    client = nvidia_client()
    background_line = (
        f"Use a {background} background." if background else "Use the natural background requested."
    )
    generation_prompt = f"""
{prompt}

Output one generated image.
Target size: {size}.
Target quality: {quality}.
Target format: {output_format.upper()}.
{background_line}
If image references are provided below, use them as visual references for the generated image.
""".strip()
    content: list[dict[str, Any]] = [{"type": "text", "text": generation_prompt}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": float(os.environ.get("NVIDIA_TEMPERATURE", "0.7")),
        "max_tokens": int(os.environ.get("NVIDIA_MAX_TOKENS", "4096")),
    }
    if os.environ.get("NVIDIA_CHAT_EXTRA_BODY"):
        kwargs["extra_body"] = json.loads(os.environ["NVIDIA_CHAT_EXTRA_BODY"])

    response = client.chat.completions.create(**kwargs)
    image_bytes = extract_image_bytes(response)
    if image_bytes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        return output_path

    image_url = extract_image_url(response)
    if image_url:
        return download_image(image_url, output_path)

    debug_path = write_response_debug(response, output_path)
    raise RuntimeError(
        f"NVIDIA chat completion did not include image data. Raw response saved to {debug_path}."
    )


def generate_image_with_nvidia_chat(
    prompt: str,
    output_path: Path,
    model: str,
    image_paths: list[Path] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Path:
    client = nvidia_client()
    content: Any
    if image_paths:
        content = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image_path)},
                }
            )
    else:
        content = prompt

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature
        if temperature is not None
        else float(os.environ.get("NVIDIA_TEMPERATURE", "0.7")),
        max_tokens=max_tokens
        if max_tokens is not None
        else int(os.environ.get("NVIDIA_MAX_TOKENS", "4096")),
    )

    image_bytes = extract_image_bytes(response)
    if image_bytes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        return output_path

    image_url = extract_image_url(response)
    if image_url:
        return download_image(image_url, output_path)

    debug_path = write_response_debug(response, output_path)
    text = extract_chat_text(response)
    raise RuntimeError(
        f"NVIDIA chat completion did not include image data. Raw response saved to {debug_path}. Text response: {text}"
    )


def edit_image(
    image_paths: list[Path],
    prompt: str,
    output_path: Path,
    model: str,
    size: str = "1024x1024",
    quality: str = "high",
    background: str | None = None,
    output_format: str = "png",
    input_fidelity: str | None = None,
    provider: str | None = None,
) -> Path:
    selected_provider = provider_name(provider)
    if selected_provider == "nvidia":
        transport = os.environ.get("NVIDIA_IMAGE_TRANSPORT", "responses").strip().lower()
        if transport == "responses":
            # The mainline-model + image_generation tool path is the only NVIDIA
            # transport that supports image inputs for gpt-image-2-class output.
            # We deliberately ignore the caller's `model` here because the
            # incoming model slug refers to an image model (e.g. gpt-image-2),
            # whereas the Responses tool requires a mainline chat model slug.
            responses_model = os.environ.get("NVIDIA_RESPONSES_MODEL")
            return edit_image_with_nvidia_responses(
                image_paths=image_paths,
                prompt=prompt,
                output_path=output_path,
                model=responses_model,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
            )
        if transport == "chat":
            return edit_image_with_nvidia_chat(
                image_paths=image_paths,
                prompt=prompt,
                output_path=output_path,
                model=model,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
            )
        return edit_image_with_nvidia_images_api(
            image_paths=image_paths,
            prompt=prompt,
            output_path=output_path,
            model=model,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            input_fidelity=input_fidelity,
        )
    if selected_provider != "openai":
        raise ValueError(f"Unsupported API provider: {selected_provider!r}")

    return edit_image_with_images_api(
        client=openai_client(),
        image_paths=image_paths,
        prompt=prompt,
        output_path=output_path,
        model=model,
        size=size,
        quality=quality,
        background=background,
        output_format=output_format,
        input_fidelity=input_fidelity,
    )
