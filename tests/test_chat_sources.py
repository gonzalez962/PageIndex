import json

from server.chat_sources import (
    build_citations,
    build_sources_read,
    collect_run,
    number_citations,
    section_for_page,
)


def test_collect_run_joins_answer_and_parses_tool_call_arguments():
    answer, calls = collect_run([
        {"type": "thinking", "delta": "hidden"},
        {"type": "answer", "delta": "Hello "},
        {"type": "tool_call", "name": "get_page_content", "arguments": '{"doc_name":"A.pdf","pages":"2"}'},
        {"type": "tool_call", "name": "other", "arguments": {"x": 1}},
        {"type": "tool_result", "output": "ignored"},
        {"type": "answer", "delta": "world"},
    ])
    assert answer == "Hello world"
    assert calls == [
        {"type": "tool_call", "name": "get_page_content", "arguments": {"doc_name": "A.pdf", "pages": "2"}},
        {"type": "tool_call", "name": "other", "arguments": {"x": 1}},
    ]


def test_number_citations_deduplicates_legacy_and_preserves_inner_text():
    answer = ('A <cite doc="A.pdf" page="2"/> then '
              '<cite doc="A.pdf" page="2"/>! '
              '<cite doc="B.pdf" page="3">inner</cite> '
              '<doc=A.pdf;page=2;block=x> '
              '<cite doc="" page="0"/>')
    # Match get_citations/_parse_citations ordering: legacy tags precede cite tags.
    assert number_citations(answer) == "A [2] then [2]! [3]inner [1] "


def test_section_for_page_selects_deepest_covering_node():
    tree = {"title": "Root", "start_index": 1, "end_index": 20, "nodes": [
        {"title": "Chapter", "start_index": 3, "end_index": 12, "nodes": [
            {"title": "Section", "start_index": 5, "end_index": 7}
        ]}
    ]}
    assert section_for_page(tree, 6) == "Section"
    assert section_for_page(tree, 4) == "Chapter"
    assert section_for_page(tree, 21) is None


def test_section_for_page_walks_stored_root_list():
    # LocalAPI.raw_tree returns the stored structure: a list of root nodes.
    tree = [
        {"title": "Preamble", "start_index": 1, "end_index": 1},
        {"title": "Guide", "start_index": 2, "end_index": 5, "nodes": [
            {"title": "Install", "start_index": 3, "end_index": 4}]},
    ]
    assert section_for_page(tree, 1) == "Preamble"
    assert section_for_page(tree, 4) == "Install"
    assert section_for_page(tree, 5) == "Guide"
    assert section_for_page(tree, 9) is None


def test_build_citations_keeps_unresolved_and_missing_tree_section_none():
    entries = [
        {"document": "A.pdf", "doc_id": "id-a", "page": 6},
        {"document": "unknown", "doc_id": None, "page": 2},
    ]
    tree = {"title": "Root", "start_index": 1, "end_index": 10}
    assert build_citations(entries, {"id-a": tree}) == [
        {"index": 1, "document": "A.pdf", "doc_id": "id-a", "page": 6, "section": "Root"},
        {"index": 2, "document": "unknown", "doc_id": None, "page": 2, "section": None},
    ]
    assert build_citations([entries[0]], {})[0]["section"] is None


def test_build_sources_read_aggregates_pages_and_ignores_invalid_calls():
    calls = [
        {"name": "get_page_content", "arguments": {"doc_name": "A.pdf", "pages": "5,3,5"}},
        {"name": "other", "arguments": {"doc_name": "A.pdf", "pages": "9"}},
        {"name": "get_page_content", "arguments": json.dumps({"doc_name": "A.pdf", "pages": "3,7,10"})},
        {"name": "get_page_content", "arguments": {"doc_name": "B.pdf", "pages": "1-3,7,9-10"}},
        {"name": "get_page_content", "arguments": {"doc_name": "bad", "pages": "0,wat"}},
        {"name": "get_page_content", "arguments": "not json"},
        {"name": "get_page_content", "arguments": {"doc_name": "", "pages": "1"}},
    ]
    assert build_sources_read(calls, {"A.pdf": "id-a"}) == [
        {"document": "A.pdf", "doc_id": "id-a", "pages": [3, 5, 7, 10]},
        {"document": "B.pdf", "doc_id": None, "pages": [1, 2, 3, 7, 9, 10]},
    ]
