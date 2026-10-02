"""Pure helpers for shaping chat answers, citations, and source reads."""
from __future__ import annotations

import json
from typing import Any

from pageindex.client import (
    _CITE_TAG_RE,
    _OLD_CITATION_RE,
    _citation_key,
    _parse_citations,
)
from pageindex.agent_tools import _expand_pages


def collect_run(events: list[dict]) -> tuple[str, list[dict]]:
    """Join answer deltas and retain tool calls with decoded arguments."""
    answer: list[str] = []
    calls: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "answer" and isinstance(event.get("delta"), str):
            answer.append(event["delta"])
        elif event.get("type") == "tool_call":
            call = dict(event)
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = None
            call["arguments"] = args if isinstance(args, dict) else {}
            calls.append(call)
    return "".join(answer), calls


def number_citations(answer: str) -> str:
    """Replace citation tags with first-appearance indices; omit unkeyed tags."""
    indices = {key: i for i, key in enumerate(
        ((entry["document"], entry["page"], entry.get("block_id"))
         for entry in _parse_citations(answer)), 1)}

    def replace(match: Any) -> str:
        key = _citation_key(match)
        index = indices.get(key) if key else None
        inner = match.groupdict().get("inner") or ""
        return f"[{index}]{inner}" if index else inner

    return _CITE_TAG_RE.sub(replace, _OLD_CITATION_RE.sub(replace, answer))


def section_for_page(tree: list | dict, page: int) -> str | None:
    """Return the deepest titled node whose inclusive range contains page.
    tree is a stored structure (a list of root nodes) or a single node."""
    found = None

    def visit(node: Any) -> None:
        nonlocal found
        if isinstance(node, list):
            for root in node:
                visit(root)
            return
        if not isinstance(node, dict):
            return
        start, end = node.get("start_index"), node.get("end_index")
        if isinstance(start, int) and isinstance(end, int) and start <= page <= end:
            if isinstance(node.get("title"), str):
                found = node["title"]
            children = node.get("nodes", [])
            if isinstance(children, list):
                for child in children:
                    visit(child)

    visit(tree)
    return found


def build_citations(entries: list[dict], trees_by_doc_id: dict) -> list[dict]:
    """Add stable indices and best-effort tree section titles to citations."""
    result = []
    for index, entry in enumerate(entries, 1):
        doc_id, page = entry.get("doc_id"), entry.get("page")
        tree = trees_by_doc_id.get(doc_id) if doc_id else None
        section = section_for_page(tree, page) if tree and isinstance(page, int) else None
        result.append({"index": index, "document": entry.get("document"),
                       "doc_id": doc_id, "page": page, "section": section})
    return result


def build_sources_read(tool_calls: list[dict], doc_ids_by_name: dict) -> list[dict]:
    """Aggregate valid get_page_content reads in first-read document order."""
    collected: dict[str, set[int]] = {}
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("name") != "get_page_content":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                continue
        if not isinstance(args, dict):
            continue
        name, spec = args.get("doc_name"), args.get("pages")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            pages = _expand_pages(spec)
        except (ValueError, TypeError):
            continue
        collected.setdefault(name, set()).update(pages)
    return [{"document": name, "doc_id": doc_ids_by_name.get(name),
             "pages": sorted(pages)} for name, pages in collected.items()]
