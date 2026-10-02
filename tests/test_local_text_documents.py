import pytest

from pageindex.errors import PageIndexAPIError
from pageindex.local_api import LocalAPI


@pytest.fixture
def api(tmp_path, monkeypatch):
    import pageindex.utils
    import pageindex.page_index_classic as classic
    monkeypatch.setattr(pageindex.utils, "llm_completion", lambda *a, **k: "description")
    monkeypatch.setattr(classic, "page_index_main", lambda path, opt, logger=None, page_list=None: {
        "structure": [{"title": "Text", "start_index": 1,
                       "end_index": len(page_list), "nodes": []}],
        "doc_description": "description"})
    return LocalAPI(str(tmp_path / "store"), "model", "summary")


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return str(path)


def test_markdown_sections_are_stored_and_retrievable(api, tmp_path, monkeypatch):
    import pageindex.page_index_classic as classic
    monkeypatch.setattr(classic, "page_index_main", lambda *a, **k: pytest.fail("structure LLM called"))
    path = write(tmp_path, "guide.md", "# Alpha\nBody\n## Child\nChild body")
    result = api.submit_document(path)
    meta = api._store.get_meta(result["doc_id"])
    pages = api._store.get_pages(result["doc_id"])
    tree = api.raw_tree(result["doc_id"])
    assert meta["mode"] == "markdown"
    assert [p["markdown"] for p in pages] == ["# Alpha\nBody", "## Child\nChild body"]
    assert tree[0]["prefix_summary"]
    assert tree[0]["nodes"][0]["summary"]
    assert api.get_tree(result["doc_id"])["result"]
    assert api.get_ocr(result["doc_id"])["result"] == pages
    assert meta["description"] == "description"
    assert all(1 <= n["start_index"] <= n["end_index"] <= len(pages) for n in tree)


def test_markdown_preamble_has_covering_root(api, tmp_path):
    result = api.submit_document(write(tmp_path, "pre.md", "Intro\n\n# Heading\nBody"))
    tree = api.raw_tree(result["doc_id"])
    assert [n["title"] for n in tree] == ["Preamble", "Heading"]
    assert [(n["start_index"], n["end_index"]) for n in tree] == [(1, 1), (2, 2)]


@pytest.mark.parametrize("name", ["plain.md", "plain.txt"])
def test_plain_text_uses_standard_page_list(api, tmp_path, monkeypatch, name):
    import pageindex.page_index_classic as classic
    seen = {}
    def fake(path, opt, logger=None, page_list=None):
        seen["pages"] = page_list
        return {"structure": [{"title": "Text", "start_index": 1, "end_index": len(page_list), "nodes": []}], "doc_description": "d"}
    monkeypatch.setattr(classic, "page_index_main", fake)
    result = api.submit_document(write(tmp_path, name, "plain content"))
    assert api._store.get_meta(result["doc_id"])["mode"] == "standard"
    assert seen["pages"][0][0] == "plain content"


@pytest.mark.parametrize("data", [b"\xff", b"a\x00b", b" \n"])
def test_bad_text_raises_api_error(api, tmp_path, data):
    with pytest.raises(PageIndexAPIError, match="Failed to submit document"):
        api.submit_document(write(tmp_path, "bad.txt", data))


def test_uppercase_text_ocr_off_and_flash_fallback(api, tmp_path):
    api._ocr = "off"
    result = api.submit_document(write(tmp_path, "UPPER.TXT", "plain"))
    assert api._store.get_meta(result["doc_id"])["mode"] == "standard"


def test_flash_only_options_are_ignored_for_text(api, tmp_path):
    api._summary_max_words = 20
    result = api.submit_document(write(tmp_path, "plain.txt", "text"))
    assert api._store.get_meta(result["doc_id"])["mode"] == "standard"


def test_markdown_summary_failure_is_wrapped_and_not_stored(api, tmp_path, monkeypatch):
    import pageindex.page_index_md as page_index_md
    def fail(*args, **kwargs):
        raise RuntimeError("summary failed")
    monkeypatch.setattr(page_index_md, "generate_summaries_for_structure_md", fail)

    with pytest.raises(PageIndexAPIError, match="Failed to submit document: summary failed"):
        api.submit_document(write(tmp_path, "guide.md", "# Heading\nBody"))

    assert api._store.list_metas() == []
