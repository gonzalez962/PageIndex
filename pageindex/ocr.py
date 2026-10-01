"""OCR through the indexer's vision model.

Pages without a usable text layer (scans, photographed pages, converted
image files) are rendered with pdfium and transcribed by the configured
model through LiteLLM multimodal messages. Pages that keep their text but
are dominated by figures get a model-written description appended, so
charts and diagrams become searchable too. One vision call per such page.
"""
from __future__ import annotations

import asyncio
import base64
import contextvars
import os
import re
from dataclasses import dataclass, field
from io import BytesIO

OCR_MODES = ("off", "auto", "force")
DEFAULT_OCR_MODE = "auto"

# Image files are converted to a scanned PDF; Pillow sniffs the content, the
# extension alone never decides. MPO is how Pillow reports some camera JPEGs.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif",
                              ".tiff", ".bmp", ".gif"})
_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "MPO", "WEBP", "TIFF", "BMP", "GIF"})

# A page with fewer non-whitespace characters than this has no usable text
# layer (blank, a page number, stray scanner noise) and is transcribed.
MIN_TEXT_CHARS = 10
# A text page whose images cover at least this share of its area gets a
# figure description appended.
IMAGE_COVERAGE_THRESHOLD = 0.3
RENDER_DPI = 150
MAX_RENDER_SIDE = 2048
# Default resolution for converted images whose files carry no usable DPI.
_DEFAULT_IMAGE_DPI = 150
VISION_CONCURRENCY = 4

FIGURE_OPEN = "[Figure description]"
FIGURE_CLOSE = "[/Figure description]"

TRANSCRIBE_PROMPT = (
    "Transcribe all text on this document page faithfully, in natural "
    "reading order. Preserve headings, lists and tables as Markdown. Do not "
    "summarize, translate, correct or add commentary. If the page contains "
    "no text, reply with an empty message."
)
DESCRIBE_PROMPT = (
    "Describe the figures, charts and diagrams on this document page "
    "concisely, including visible numbers, labels, axes and legends. Do not "
    "repeat the body text and do not add commentary. If the page contains "
    "no figures, reply with an empty message."
)


class OCRModelError(RuntimeError):
    """The vision model refused a page image."""


@dataclass
class OCRResult:
    """Final per-page texts and the 1-based pages the model transcribed."""

    page_texts: list[str]
    transcribed_pages: list[int] = field(default_factory=list)

    @property
    def needed_ocr(self) -> bool:
        """Some page had no usable text layer, so its text came from the
        model instead of the PDF."""
        return bool(self.transcribed_pages)


def validate_ocr_mode(mode) -> str:
    if mode not in OCR_MODES:
        raise ValueError(
            f"ocr must be one of {', '.join(repr(m) for m in OCR_MODES)}; "
            f"got {mode!r}.")
    return mode


# ── image files ──

def has_image_extension(name) -> bool:
    return os.path.splitext(str(name))[1].lower() in IMAGE_EXTENSIONS


def sniff_image_format(source) -> str | None:
    """Pillow's format name for a supported image (path or bytes), else
    None. Only the header is parsed; pixels are not decoded."""
    from PIL import Image, UnidentifiedImageError
    try:
        handle = BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        with Image.open(handle) as image:
            fmt = image.format
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    return fmt if fmt in _IMAGE_FORMATS else None


def is_image_path(path) -> bool:
    """A supported image extension whose content Pillow recognizes."""
    return (has_image_extension(path) and os.path.isfile(path)
            and sniff_image_format(path) is not None)


def _to_rgb(frame):
    from PIL import Image
    if frame.mode in ("RGBA", "LA") or (frame.mode == "P" and "transparency" in frame.info):
        rgba = frame.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return frame.convert("RGB")


def _image_dpi(image) -> float:
    dpi = image.info.get("dpi")
    try:
        value = float(dpi[0]) if dpi else 0.0
    except (TypeError, ValueError, IndexError):
        value = 0.0
    return value if 50 <= value <= 1200 else _DEFAULT_IMAGE_DPI


def image_to_pdf(src_path, out_path) -> int:
    """Write the image at src_path as a PDF without a text layer, one page
    per frame (multi-page TIFF, animated GIF). Returns the page count."""
    from PIL import Image, ImageOps, ImageSequence
    if sniff_image_format(src_path) is None:
        raise ValueError(f"{os.path.basename(str(src_path))} is not a supported image.")
    with Image.open(src_path) as image:
        dpi = _image_dpi(image)
        frames = [_to_rgb(ImageOps.exif_transpose(frame.copy()))
                  for frame in ImageSequence.Iterator(image)]
    frames[0].save(out_path, "PDF", save_all=True, append_images=frames[1:],
                   resolution=dpi)
    return len(frames)


# ── page analysis ──

def page_needs_transcription(text: str) -> bool:
    return len(re.sub(r"\s+", "", text or "")) < MIN_TEXT_CHARS


def image_coverage(page) -> float:
    """Share of the page area covered by image objects (overlaps counted
    twice, capped at 1)."""
    import pypdfium2.raw as pdfium_c
    width, height = page.get_size()
    if width <= 0 or height <= 0:
        return 0.0
    covered = 0.0
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
        left, bottom, right, top = obj.get_bounds()
        w = min(right, width) - max(left, 0)
        h = min(top, height) - max(bottom, 0)
        if w > 0 and h > 0:
            covered += w * h
    return min(covered / (width * height), 1.0)


def render_page_data_url(page) -> str:
    """The page as a PNG data URL at RENDER_DPI, longest side capped."""
    width, height = page.get_size()
    scale = RENDER_DPI / 72
    longest = max(width, height) * scale
    if longest > MAX_RENDER_SIDE:
        scale *= MAX_RENDER_SIDE / longest
    image = page.render(scale=scale).to_pil()
    buf = BytesIO()
    image.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── vision calls ──

async def _ask_vision(model: str, prompt: str, data_url: str) -> str:
    from . import utils
    content = [{"type": "text", "text": prompt},
               {"type": "image_url", "image_url": {"url": data_url}}]
    try:
        answer = await utils.llm_acompletion(model, content)
    except Exception as e:
        if getattr(e, "status_code", None) == 400:
            raise OCRModelError(
                f"The OCR model {model!r} rejected a page image; it may not "
                f"support image input. Use a vision-capable model (ocr_model) "
                f"or disable OCR (ocr='off'). Provider error: {e}") from e
        raise
    return (answer or "").strip()


def _plan(doc, page_texts: list[str], mode: str) -> dict[int, str]:
    """0-based page index → 'transcribe' or 'describe'."""
    plan = {}
    for i in range(min(len(doc), len(page_texts))):
        if mode == "force" or page_needs_transcription(page_texts[i]):
            plan[i] = "transcribe"
        elif image_coverage(doc[i]) >= IMAGE_COVERAGE_THRESHOLD:
            plan[i] = "describe"
    return plan


async def _run_plan(doc, page_texts, plan, model) -> list[str]:
    texts = list(page_texts)
    # pdfium is not thread-safe; rendering stays on this loop's thread and
    # only the model calls overlap.
    limit = asyncio.Semaphore(VISION_CONCURRENCY)

    async def one(index, kind):
        async with limit:
            data_url = render_page_data_url(doc[index])
            if kind == "transcribe":
                answer = await _ask_vision(model, TRANSCRIBE_PROMPT, data_url)
                if answer:
                    texts[index] = answer
            else:
                answer = await _ask_vision(model, DESCRIBE_PROMPT, data_url)
                if answer:
                    texts[index] = (f"{texts[index].rstrip()}\n\n{FIGURE_OPEN}\n"
                                    f"{answer}\n{FIGURE_CLOSE}")

    await asyncio.gather(*(one(i, kind) for i, kind in sorted(plan.items())))
    return texts


def ocr_pages(pdf_path, page_texts: list[str], model: str,
              mode: str = DEFAULT_OCR_MODE) -> OCRResult:
    """Final per-page texts for a PDF whose text layer extracted as
    page_texts. Runs in the caller's LLM backend scope."""
    import pypdfium2 as pdfium
    from .utils import run_off_loop
    validate_ocr_mode(mode)
    if mode == "off":
        return OCRResult(list(page_texts))
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        plan = _plan(doc, page_texts, mode)
        if not plan:
            return OCRResult(list(page_texts))
        # run_off_loop may hop threads; carry the backend contextvar along.
        context = contextvars.copy_context()
        texts = run_off_loop(context.run, asyncio.run,
                             _run_plan(doc, page_texts, plan, model))
    finally:
        doc.close()
    transcribed = [i + 1 for i, kind in sorted(plan.items()) if kind == "transcribe"]
    return OCRResult(texts, transcribed)
