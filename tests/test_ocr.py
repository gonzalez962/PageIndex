"""OCR through the indexer's vision model: page classification, image
conversion and the multimodal call. Every model call is mocked."""
import io

import pytest
from PIL import Image

import pageindex.utils
from pageindex import ocr
from conftest import build_pdf


class FakeVision:
    """Records multimodal prompts and answers by prompt kind."""

    def __init__(self, transcription="Transcribed text", description="A bar chart"):
        self.calls = []
        self.transcription = transcription
        self.description = description

    async def __call__(self, model, prompt):
        self.calls.append((model, prompt))
        text = prompt[0]["text"]
        if text == ocr.TRANSCRIBE_PROMPT:
            return self.transcription
        if text == ocr.DESCRIBE_PROMPT:
            return self.description
        raise AssertionError(f"unexpected prompt: {text!r}")

    def kinds(self):
        return ["transcribe" if p[0]["text"] == ocr.TRANSCRIBE_PROMPT else "describe"
                for _, p in self.calls]


@pytest.fixture
def vision(monkeypatch):
    fake = FakeVision()
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", fake)
    return fake


def _scanned_pdf(tmp_path, pages=2):
    """A PDF without a text layer: Pillow embeds each image as a page."""
    frames = [Image.new("RGB", (300, 400), color) for color in
              ("white", "lightgray", "beige")[:pages]]
    path = tmp_path / "scanned.pdf"
    frames[0].save(path, "PDF", save_all=True, append_images=frames[1:],
                   resolution=150)
    return str(path)


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


# ── modes ──

def test_validate_ocr_mode_accepts_known_and_rejects_unknown():
    assert ocr.OCR_MODES == ("off", "auto", "force")
    assert ocr.DEFAULT_OCR_MODE == "auto"
    for mode in ocr.OCR_MODES:
        assert ocr.validate_ocr_mode(mode) == mode
    with pytest.raises(ValueError, match="ocr must be one of 'off', 'auto', 'force'"):
        ocr.validate_ocr_mode("always")
    with pytest.raises(ValueError):
        ocr.validate_ocr_mode(None)


def test_text_pdf_in_auto_makes_no_vision_call(tmp_path, vision):
    texts = ["Hello page one about apples", "Second page about bananas"]
    path = _write(tmp_path, "text.pdf", build_pdf(texts))
    result = ocr.ocr_pages(path, texts, model="gpt-4o", mode="auto")
    assert result.page_texts == texts
    assert result.transcribed_pages == []
    assert not result.needed_ocr
    assert vision.calls == []


def test_off_mode_never_calls_the_model(tmp_path, vision):
    path = _scanned_pdf(tmp_path)
    result = ocr.ocr_pages(path, ["", ""], model="gpt-4o", mode="off")
    assert result.page_texts == ["", ""]
    assert not result.needed_ocr
    assert vision.calls == []


def test_scanned_pdf_pages_are_transcribed(tmp_path, vision):
    path = _scanned_pdf(tmp_path)
    result = ocr.ocr_pages(path, ["", "  \n"], model="gpt-4o", mode="auto")
    assert result.page_texts == ["Transcribed text", "Transcribed text"]
    assert result.transcribed_pages == [1, 2]
    assert result.needed_ocr
    assert vision.kinds() == ["transcribe", "transcribe"]
    model, prompt = vision.calls[0]
    assert model == "gpt-4o"
    image_part = prompt[1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_force_mode_transcribes_pages_that_have_text(tmp_path, vision):
    texts = ["Hello page one about apples", "Second page about bananas"]
    path = _write(tmp_path, "text.pdf", build_pdf(texts))
    result = ocr.ocr_pages(path, texts, model="gpt-4o", mode="force")
    assert result.page_texts == ["Transcribed text", "Transcribed text"]
    assert result.transcribed_pages == [1, 2]
    assert vision.kinds() == ["transcribe", "transcribe"]


def test_empty_transcription_keeps_the_extracted_text(tmp_path, monkeypatch):
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", FakeVision(transcription="  "))
    path = _write(tmp_path, "short.pdf", build_pdf(["p. 3"]))
    result = ocr.ocr_pages(path, ["p. 3"], model="gpt-4o", mode="auto")
    assert result.page_texts == ["p. 3"]


def test_image_rich_page_keeps_text_and_appends_description(tmp_path, vision):
    texts = ["Quarterly revenue overview", "Plain closing remarks here"]
    pdf = build_pdf(texts, images={0: (72, 100, 468, 500)})
    path = _write(tmp_path, "figures.pdf", pdf)
    result = ocr.ocr_pages(path, texts, model="gpt-4o", mode="auto")
    first = result.page_texts[0]
    assert first.startswith("Quarterly revenue overview")
    assert "[Figure description]\nA bar chart\n[/Figure description]" in first
    assert result.page_texts[1] == texts[1]
    assert result.transcribed_pages == []
    assert not result.needed_ocr
    assert vision.kinds() == ["describe"]


def test_small_image_does_not_trigger_a_description(tmp_path, vision):
    texts = ["A page with a small logo on it"]
    path = _write(tmp_path, "logo.pdf", build_pdf(texts, images={0: (500, 740, 40, 40)}))
    result = ocr.ocr_pages(path, texts, model="gpt-4o", mode="auto")
    assert result.page_texts == texts
    assert vision.calls == []


def test_image_coverage_measures_the_share_of_the_page(tmp_path):
    import pypdfium2 as pdfium
    path = _write(tmp_path, "cov.pdf", build_pdf(
        ["covered", "bare"], images={0: (0, 0, 306, 792)}))
    doc = pdfium.PdfDocument(path)
    try:
        assert ocr.image_coverage(doc[0]) == pytest.approx(0.5)
        assert ocr.image_coverage(doc[1]) == 0.0
    finally:
        doc.close()


def test_rejected_image_input_raises_a_clear_error_after_one_attempt(tmp_path, monkeypatch):
    import litellm

    class BadRequest(Exception):
        status_code = 400

    calls = []

    async def reject(**kwargs):
        calls.append(kwargs)
        raise BadRequest("image_url is not supported")

    monkeypatch.setattr(litellm, "acompletion", reject)
    path = _scanned_pdf(tmp_path, pages=1)
    with pytest.raises(ocr.OCRModelError) as err:
        ocr.ocr_pages(path, [""], model="openai/text-only-model", mode="auto")
    message = str(err.value)
    assert "openai/text-only-model" in message
    assert "may not support image input" in message
    assert len(calls) == 1


def test_non_400_errors_are_not_reworded(tmp_path, monkeypatch):
    class Unauthorized(Exception):
        status_code = 401

    async def deny(model, prompt):
        raise Unauthorized("bad key")

    monkeypatch.setattr(pageindex.utils, "llm_acompletion", deny)
    path = _scanned_pdf(tmp_path, pages=1)
    with pytest.raises(Unauthorized):
        ocr.ocr_pages(path, [""], model="gpt-4o", mode="auto")


def test_vision_call_uses_the_active_backend(tmp_path, monkeypatch):
    import litellm
    seen = []

    class Response:
        class _Choice:
            class message:
                content = "From backend"
        choices = [_Choice]

    async def fake(**kwargs):
        seen.append(kwargs)
        return Response

    monkeypatch.setattr(litellm, "acompletion", fake)
    path = _scanned_pdf(tmp_path, pages=1)
    token = pageindex.utils._llm_backend.set({"api_base": "http://vision.local"})
    try:
        result = ocr.ocr_pages(path, [""], model="gpt-4o", mode="auto")
    finally:
        pageindex.utils._llm_backend.reset(token)
    assert result.page_texts == ["From backend"]
    assert seen[0]["api_base"] == "http://vision.local"
    content = seen[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": ocr.TRANSCRIBE_PROMPT}


# ── image files ──

def _image_bytes(fmt, size=(64, 48), mode="RGB", frames=1):
    images = [Image.new(mode, size) for _ in range(frames)]
    buf = io.BytesIO()
    if frames > 1:
        images[0].save(buf, fmt, save_all=True, append_images=images[1:])
    else:
        images[0].save(buf, fmt)
    return buf.getvalue()


@pytest.mark.parametrize("name,fmt", [
    ("a.png", "PNG"), ("a.jpg", "JPEG"), ("a.jpeg", "JPEG"), ("a.PNG", "PNG"),
    ("a.JPG", "JPEG"),
])
def test_supported_images_are_detected_by_content(tmp_path, name, fmt):
    path = _write(tmp_path, name, _image_bytes(fmt))
    assert ocr.has_image_extension(name)
    assert ocr.is_image_path(path)


@pytest.mark.parametrize("ext,fmt", [
    ("webp", "WEBP"), ("tif", "TIFF"), ("tiff", "TIFF"), ("bmp", "BMP"),
    ("gif", "GIF"),
])
def test_only_png_and_jpeg_are_supported(tmp_path, ext, fmt):
    data = _image_bytes(fmt)
    native = _write(tmp_path, f"a.{ext}", data)
    assert not ocr.has_image_extension(native)
    assert not ocr.is_image_path(native)
    # The content decides too: another format behind a .png name is refused.
    disguised = _write(tmp_path, "b.png", data)
    assert ocr.sniff_image_format(data) is None
    assert not ocr.is_image_path(disguised)
    with pytest.raises(ValueError, match="not a supported image"):
        ocr.image_to_pdf(disguised, str(tmp_path / "out.pdf"))


def test_disguised_or_unsupported_files_are_not_images(tmp_path):
    disguised = _write(tmp_path, "fake.png", b"%PDF-1.4 not an image")
    assert not ocr.is_image_path(disguised)
    pdf = _write(tmp_path, "doc.pdf", build_pdf(["x"]))
    assert not ocr.has_image_extension(pdf)
    assert not ocr.is_image_path(pdf)
    # Real image content behind an unsupported extension stays unsupported.
    other = _write(tmp_path, "pic.ico", _image_bytes("PNG"))
    assert not ocr.is_image_path(other)


def test_decompression_bombs_are_not_images(tmp_path, monkeypatch):
    import warnings
    data = _image_bytes("PNG", size=(64, 48))  # 3072 pixels
    path = _write(tmp_path, "bomb.png", data)
    # Over twice the limit Pillow raises DecompressionBombError.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    assert ocr.sniff_image_format(data) is None
    assert not ocr.is_image_path(path)
    with pytest.raises(ValueError, match="not a supported image"):
        ocr.image_to_pdf(path, str(tmp_path / "out.pdf"))
    # Just over the limit it warns; a warning promoted to an error is a
    # rejection too, never an unhandled exception.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        assert ocr.sniff_image_format(data) is None
        assert not ocr.is_image_path(path)


def test_sniff_image_format_reads_bytes():
    assert ocr.sniff_image_format(_image_bytes("PNG")) == "PNG"
    assert ocr.sniff_image_format(b"%PDF-1.4") is None


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGBA"), ("JPEG", "L"),
                                       ("PNG", "P"), ("PNG", "LA")])
def test_image_to_pdf_produces_one_scanned_page(tmp_path, fmt, mode):
    import pypdfium2 as pdfium
    src = _write(tmp_path, f"img.{fmt.lower()}", _image_bytes(fmt, mode=mode))
    out = str(tmp_path / "out.pdf")
    assert ocr.image_to_pdf(src, out) == 1
    doc = pdfium.PdfDocument(out)
    try:
        assert len(doc) == 1
        assert doc[0].get_textpage().get_text_range().strip() == ""
    finally:
        doc.close()


def _apng_bytes(colors=("red", "green", "blue"), size=(200, 260)):
    frames = [Image.new("RGB", size, color) for color in colors]
    buf = io.BytesIO()
    frames[0].save(buf, "PNG", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def test_animated_png_uses_only_its_first_frame(tmp_path, vision):
    import pypdfium2 as pdfium
    data = _apng_bytes()
    with Image.open(io.BytesIO(data)) as image:
        assert getattr(image, "n_frames", 1) == 3  # really animated
    src = _write(tmp_path, "anim.png", data)
    out = str(tmp_path / "out.pdf")
    assert ocr.image_to_pdf(src, out) == 1
    doc = pdfium.PdfDocument(out)
    try:
        assert len(doc) == 1
        pixel = doc[0].render(scale=0.5).to_pil().convert("RGB").getpixel((10, 10))
    finally:
        doc.close()
    assert pixel[0] > 200 and pixel[1] < 60 and pixel[2] < 60  # the red frame
    result = ocr.ocr_pages(out, [""], model="gpt-4o", mode="auto")
    assert result.page_texts == ["Transcribed text"]
    assert len(vision.calls) == 1


def test_image_to_pdf_rejects_non_images(tmp_path):
    src = _write(tmp_path, "fake.png", b"not an image")
    with pytest.raises(ValueError, match="not a supported image"):
        ocr.image_to_pdf(src, str(tmp_path / "out.pdf"))


def test_converted_image_is_transcribed_end_to_end(tmp_path, vision):
    src = _write(tmp_path, "page.png", _image_bytes("PNG", size=(800, 1000)))
    out = str(tmp_path / "page.pdf")
    ocr.image_to_pdf(src, out)
    result = ocr.ocr_pages(out, [""], model="gpt-4o", mode="auto")
    assert result.page_texts == ["Transcribed text"]
    assert result.needed_ocr


def test_render_caps_the_longest_side(tmp_path):
    import base64
    import pypdfium2 as pdfium
    src = _write(tmp_path, "big.png", _image_bytes("PNG", size=(6000, 300)))
    out = str(tmp_path / "big.pdf")
    ocr.image_to_pdf(src, out)
    doc = pdfium.PdfDocument(out)
    try:
        url = ocr.render_page_data_url(doc[0])
    finally:
        doc.close()
    png = base64.b64decode(url.split(",", 1)[1])
    width, height = Image.open(io.BytesIO(png)).size
    assert max(width, height) <= ocr.MAX_RENDER_SIDE


def test_llm_completion_passes_content_parts_through(monkeypatch):
    import litellm
    seen = []

    class Response:
        class _Choice:
            finish_reason = "stop"
            class message:
                content = "ok"
        choices = [_Choice]

    monkeypatch.setattr(litellm, "completion", lambda **kw: seen.append(kw) or Response)
    parts = [{"type": "text", "text": "hi"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    assert pageindex.utils.llm_completion("gpt-4o", parts) == "ok"
    assert seen[0]["messages"] == [{"role": "user", "content": parts}]
    assert pageindex.utils.llm_completion("gpt-4o", "plain") == "ok"
    assert seen[1]["messages"] == [{"role": "user", "content": "plain"}]
