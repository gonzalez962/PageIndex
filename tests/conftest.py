import pytest


@pytest.fixture(autouse=True)
def _llm_key(monkeypatch):
    """Deterministic key presence for every test; missing-key tests delenv."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def build_pdf(page_texts, images=None):
    """Build a minimal, uncompressed PDF (one Helvetica line per page) whose
    text PyPDF2 can extract. Returns the PDF file bytes.

    ``images`` maps a 0-based page index to an ``(x, y, width, height)``
    box in points; that page also draws a small RGB image scaled to it."""
    images = images or {}
    n = len(page_texts)
    objects = []
    kids = " ".join(f"{3 + i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_obj = 3 + 2 * n
    image_obj = font_obj + 1
    for i in range(n):
        xobject = f"/XObject << /Im1 {image_obj} 0 R >> " if i in images else ""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> {xobject}>> "
            f"/Contents {3 + n + i} 0 R >>".encode()
        )
    for i, text in enumerate(page_texts):
        safe = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
        if i in images:
            x, y, w, h = images[i]
            stream += f"\nq {w} 0 0 {h} {x} {y} cm /Im1 Do Q".encode()
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    if images:
        pixels = bytes([200, 30, 30, 30, 200, 30, 30, 30, 200, 240, 240, 240])
        objects.append(
            b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 "
            b"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length %d >>\n"
            b"stream\n%s\nendstream" % (len(pixels), pixels))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref_pos))
    return bytes(out)


@pytest.fixture
def sample_pdf(tmp_path):
    """A 2-page PDF with known, extractable text."""
    path = tmp_path / "sample.pdf"
    path.write_bytes(build_pdf(["Hello page one about apples",
                                "Second page about bananas"]))
    return str(path)
