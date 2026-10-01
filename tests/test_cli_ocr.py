"""run_pageindex.py OCR flags: --ocr, --ocr-model and image inputs.

The script runs in-process with the indexers and the vision call stubbed."""
import json
import runpy
import sys
from pathlib import Path

import pytest
from PIL import Image

import pageindex
import pageindex.flash
import pageindex.page_index_classic
import pageindex.utils
from conftest import build_pdf

SCRIPT = Path(__file__).resolve().parent.parent / "run_pageindex.py"


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Run the CLI; returns what each stub saw."""
    seen = {"vision": [], "standard": None, "flash": None}

    async def vision(model, prompt):
        seen["vision"].append(model)
        return "Text read by the vision model"

    def fake_main(doc, opt=None, logger=None, page_list=None):
        seen["standard"] = {"doc": doc, "page_list": page_list, "model": opt.model}
        return {"doc_name": Path(doc).name, "structure": []}

    def fake_flash(path, **kwargs):
        seen["flash"] = path
        return {"structure": [{"title": "T", "start_index": 1, "end_index": 1}]}

    monkeypatch.setattr(pageindex.utils, "llm_acompletion", vision)
    monkeypatch.setattr(pageindex.page_index_classic, "page_index_main", fake_main)
    # The CLI star-imports pageindex, whose lazy attributes are cached.
    monkeypatch.setattr(pageindex, "page_index_main", fake_main, raising=False)
    monkeypatch.setattr(pageindex.flash, "page_index_flash", fake_flash)
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_pageindex.py", *argv])
        runpy.run_path(str(SCRIPT), run_name="__main__")
        return seen
    return run


def _image(tmp_path, name, fmt, frames=1):
    images = [Image.new("RGB", (200, 260), "white") for _ in range(frames)]
    path = tmp_path / name
    images[0].save(path, fmt, save_all=frames > 1, append_images=images[1:])
    return str(path)


def _scanned_pdf(tmp_path):
    path = tmp_path / "scanned.pdf"
    Image.new("RGB", (300, 400), "white").save(path, "PDF")
    return str(path)


def _text_pdf(tmp_path):
    path = tmp_path / "text.pdf"
    path.write_bytes(build_pdf(["Hello page one about apples"]))
    return str(path)


def test_png_is_indexed_through_ocr(cli, tmp_path):
    seen = cli("--pdf_path", _image(tmp_path, "scan.png", "PNG"),
               "--index-model", "openai/gpt-4o")
    assert seen["vision"] == ["openai/gpt-4o"]
    assert seen["flash"] is None
    assert seen["standard"]["doc"].endswith("scan.png")
    assert [t for t, _ in seen["standard"]["page_list"]] == [
        "Text read by the vision model"]
    out = json.loads((tmp_path / "results" / "scan_structure.json").read_text())
    assert out["doc_name"] == "scan.png"


def test_animated_png_yields_only_its_first_frame(cli, tmp_path):
    seen = cli("--pdf_path", _image(tmp_path, "anim.png", "PNG", frames=3))
    assert len(seen["standard"]["page_list"]) == 1
    assert len(seen["vision"]) == 1


@pytest.mark.parametrize("ext,fmt", [("webp", "WEBP"), ("tif", "TIFF"),
                                     ("tiff", "TIFF"), ("bmp", "BMP"),
                                     ("gif", "GIF")])
def test_images_other_than_png_and_jpeg_are_rejected(cli, tmp_path, ext, fmt):
    with pytest.raises(ValueError, match=r"must be a PDF \(\.pdf\) or a PNG or JPEG"):
        cli("--pdf_path", _image(tmp_path, f"pic.{ext}", fmt))
    # Their content behind a .png name is refused as well.
    with pytest.raises(ValueError, match="not a supported image"):
        cli("--pdf_path", _image(tmp_path, f"pic-{ext}.png", fmt))


def test_ocr_model_flag_picks_the_vision_model(cli, tmp_path):
    seen = cli("--pdf_path", _image(tmp_path, "scan.jpg", "JPEG"),
               "--index-model", "openai/gpt-4o-mini", "--ocr-model", "openai/gpt-4o")
    assert seen["vision"] == ["openai/gpt-4o"]
    assert seen["standard"]["model"] == "openai/gpt-4o-mini"


def test_scanned_pdf_in_flash_mode_switches_to_standard(cli, tmp_path, capsys):
    seen = cli("--pdf_path", _scanned_pdf(tmp_path))
    assert seen["flash"] is None
    assert [t for t, _ in seen["standard"]["page_list"]] == [
        "Text read by the vision model"]
    assert "standard mode" in capsys.readouterr().out


def test_text_pdf_in_auto_keeps_flash_without_vision_calls(cli, tmp_path):
    seen = cli("--pdf_path", _text_pdf(tmp_path))
    assert seen["vision"] == []
    assert seen["flash"] is not None
    assert seen["standard"] is None


def test_text_pdf_in_standard_mode_keeps_the_text_layer(cli, tmp_path):
    seen = cli("--pdf_path", _text_pdf(tmp_path), "--mode", "standard")
    assert seen["vision"] == []
    assert seen["standard"]["page_list"] is None


def test_ocr_off_leaves_scanned_pdfs_to_the_pipeline(cli, tmp_path):
    seen = cli("--pdf_path", _scanned_pdf(tmp_path), "--ocr", "off")
    assert seen["vision"] == []
    assert seen["flash"] is not None


def test_ocr_force_transcribes_text_pdfs(cli, tmp_path):
    seen = cli("--pdf_path", _text_pdf(tmp_path), "--ocr", "force")
    assert seen["vision"]
    assert [t for t, _ in seen["standard"]["page_list"]] == [
        "Text read by the vision model"]


def test_image_with_ocr_off_is_rejected(cli, tmp_path):
    with pytest.raises(ValueError, match="need OCR"):
        cli("--pdf_path", _image(tmp_path, "scan.png", "PNG"), "--ocr", "off")


def test_unsupported_extension_is_rejected(cli, tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("hi")
    with pytest.raises(ValueError, match="must be a PDF"):
        cli("--pdf_path", str(other))


def test_ocr_flags_require_pdf_path(cli, tmp_path):
    md = tmp_path / "a.md"
    md.write_text("# A\n")
    with pytest.raises(ValueError, match="--ocr requires --pdf_path"):
        cli("--md_path", str(md), "--ocr", "auto")
    with pytest.raises(ValueError, match="--ocr-model requires --pdf_path"):
        cli("--md_path", str(md), "--ocr-model", "m")


def test_unknown_ocr_mode_is_a_usage_error(cli, tmp_path):
    with pytest.raises(SystemExit):
        cli("--pdf_path", _text_pdf(tmp_path), "--ocr", "always")
