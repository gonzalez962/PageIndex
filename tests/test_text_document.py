import pytest

from pageindex import text_document


@pytest.fixture(autouse=True)
def deterministic_tokens(monkeypatch):
    monkeypatch.setattr(text_document, "count_tokens", lambda text, model=None: len(text))


def test_read_text_document_normalizes_bom_and_newlines(tmp_path):
    path = tmp_path / "notes.md"
    path.write_bytes(b"\xef\xbb\xbf# Title\r\n\rBody\r")

    assert text_document.read_text_document(path) == "# Title\n\nBody\n"


@pytest.mark.parametrize("contents, message", [
    (b"\xff", "UTF-8"),
    (b"hello\x00world", "NUL"),
    (b" \r\n\t", "blank"),
])
def test_read_text_document_rejects_invalid_content(tmp_path, contents, message):
    path = tmp_path / "input.txt"
    path.write_bytes(contents)

    with pytest.raises(ValueError, match=message):
        text_document.read_text_document(path)


def test_markdown_sections_nest_and_cover_preamble_pages():
    content = "Preamble.\n\n# Alpha\nA body.\n## Child\nChild body.\n# Beta\nB body."

    pages, structure = text_document.markdown_section_pages(content, 100)

    assert pages == ["Preamble.", "# Alpha\nA body.", "## Child\nChild body.", "# Beta\nB body."]
    assert structure == [
        {"title": "Alpha", "node_id": "0000", "start_index": 2, "end_index": 3,
         "nodes": [{"title": "Child", "node_id": "0001", "start_index": 3,
                    "end_index": 3, "nodes": []}]},
        {"title": "Beta", "node_id": "0002", "start_index": 4, "end_index": 4, "nodes": []},
    ]
    def check_ranges(nodes, parent=None):
        for node in nodes:
            assert 1 <= node["start_index"] <= node["end_index"] <= len(pages)
            assert node["title"] in pages[node["start_index"] - 1]
            if parent is not None:
                assert parent["start_index"] <= node["start_index"]
                assert parent["end_index"] >= node["end_index"]
            check_ranges(node["nodes"], node)
    check_ranges(structure)


def test_oversized_markdown_section_spans_multiple_pages():
    pages, tree = text_document.markdown_section_pages("# Long\n\none\n\ntwo\n\nthree", 9)

    assert pages == ["# Long", "one\n\ntwo", "three"]
    assert tree[0]["start_index"] == 1
    assert tree[0]["end_index"] == len(pages)
    assert "# Long" in pages[tree[0]["start_index"] - 1]


def test_headings_inside_fences_are_not_sections():
    content = "# Real\n\n```md\n# Not a heading\n```\ntext"

    assert text_document.has_markdown_headings(content)
    _, tree = text_document.markdown_section_pages(content, 100)
    assert [node["title"] for node in tree] == ["Real"]
    assert not text_document.has_markdown_headings("```\n# hidden\n```")


def test_plain_text_packs_paragraphs_and_round_trips():
    content = "one\nline\n\ntwo\n\nthree"

    pages = text_document.split_text_pages(content, 12)

    assert pages == ["one\nline", "two\n\nthree"]
    assert "\n\n".join(pages) == content


def test_oversized_paragraph_splits_by_lines_and_oversized_line_hard_splits():
    assert text_document.split_text_pages("first line\nsecond", 10) == ["first line", "second"]
    pages = text_document.split_text_pages("abcdefghijkl", 5)
    assert pages == ["abcde", "fghij", "kl"]
    assert "" not in pages


def test_plain_text_pages_preserve_content_without_splitting_fitting_paragraphs():
    import re

    content = "alpha beta\n\ngamma delta"
    pages = text_document.split_text_pages(content, 11)
    assert pages == ["alpha beta", "gamma delta"]
    assert re.sub(r"\s+", "", "".join(pages)) == re.sub(r"\s+", "", content)


def test_oversized_paragraph_pieces_keep_single_newline_not_paragraph_break():
    pages = text_document.split_text_pages("123456\nabcdef\nxyz", 13)
    assert all("\n\n" not in page for page in pages)
    assert "\n" in pages[0] or "\n" in pages[1]
