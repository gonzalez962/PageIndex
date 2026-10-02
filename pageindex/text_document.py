"""Pure helpers for turning text documents into PageIndex pages and trees."""

from pathlib import Path
import re

from .page_index_md import extract_nodes_from_markdown
from .utils import count_tokens

# 1000 tokens keeps pseudo-pages manageable while allowing useful text context.
DEFAULT_PAGE_TOKENS = 1000


def read_text_document(path) -> str:
    """Read a UTF-8 text document, normalizing its BOM and line endings."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Text document is not valid UTF-8") from exc
    if "\x00" in text:
        raise ValueError("Text document contains NUL bytes")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("Text document is blank")
    return text


def _split_oversized_line(line, max_page_tokens, model):
    pieces = []
    remaining = line
    while remaining:
        low, high = 1, len(remaining)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            if count_tokens(remaining[:middle], model=model) <= max_page_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if not best:
            best = 1
        pieces.append(remaining[:best])
        remaining = remaining[best:]
    return pieces


def _split_paragraph(paragraph, max_page_tokens, model):
    if count_tokens(paragraph, model=model) <= max_page_tokens:
        return [paragraph]
    pages = []
    current = ""
    for line in paragraph.split("\n"):
        line_parts = ([line] if count_tokens(line, model=model) <= max_page_tokens
                      else _split_oversized_line(line, max_page_tokens, model))
        for part in line_parts:
            candidate = part if not current else current + "\n" + part
            if current and count_tokens(candidate, model=model) > max_page_tokens:
                pages.append(current)
                current = part
            else:
                current = candidate
    if current:
        pages.append(current)
    return pages


def split_text_pages(text, max_page_tokens=DEFAULT_PAGE_TOKENS, model=None) -> list[str]:
    """Pack blank-line-separated paragraphs without emitting empty pages."""
    if max_page_tokens <= 0:
        raise ValueError("max_page_tokens must be greater than zero")
    parts = re.split(r"(\n\s*\n)", text)
    paragraphs = [(parts[i].strip("\n"), parts[i + 1] if i + 1 < len(parts) else "")
                  for i in range(0, len(parts), 2) if parts[i].strip()]
    pages = []
    current = ""
    for index, (paragraph, separator) in enumerate(paragraphs):
        if count_tokens(paragraph, model=model) > max_page_tokens:
            if current:
                pages.append(current)
                current = ""
            pages.extend(_split_paragraph(paragraph, max_page_tokens, model))
            continue
        joiner = paragraphs[index - 1][1] if current else ""
        candidate = paragraph if not current else current + joiner + paragraph
        if current and count_tokens(candidate, model=model) > max_page_tokens:
            pages.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        pages.append(current)
    return pages


def has_markdown_headings(text) -> bool:
    """Return whether the Markdown parser recognizes any headings."""
    headings, _ = extract_nodes_from_markdown(text)
    return bool(headings)


def markdown_section_pages(text, max_page_tokens=DEFAULT_PAGE_TOKENS, model=None):
    """Return pages and a heading tree with inclusive, 1-based page ranges."""
    if max_page_tokens <= 0:
        raise ValueError("max_page_tokens must be greater than zero")
    headings, lines = extract_nodes_from_markdown(text)
    if not headings:
        return split_text_pages(text, max_page_tokens, model), []

    pages = []
    preamble = "\n".join(lines[:headings[0]["line_num"] - 1]).strip()
    if preamble:
        pages.extend(split_text_pages(preamble, max_page_tokens, model))

    roots = []
    stack = []
    if preamble:
        preamble_start = 1
        preamble_end = len(pages)
        roots.append({"title": "Preamble", "start_index": preamble_start,
                      "end_index": preamble_end, "nodes": []})
    for index, heading in enumerate(headings):
        start_line = heading["line_num"] - 1
        end_line = (headings[index + 1]["line_num"] - 1
                    if index + 1 < len(headings) else len(lines))
        section = "\n".join(lines[start_line:end_line]).strip()
        section_pages = split_text_pages(section, max_page_tokens, model)
        if not section_pages:
            section_pages = [heading["node_title"]]
        start_page = len(pages) + 1
        pages.extend(section_pages)
        node = {"title": heading["node_title"],
                "start_index": start_page, "end_index": len(pages), "nodes": []}
        while stack and stack[-1][0] >= heading["level"]:
            stack.pop()
        if stack:
            stack[-1][1]["nodes"].append(node)
        else:
            roots.append(node)
        stack.append((heading["level"], node))

    def include_descendants(node):
        for child in node["nodes"]:
            include_descendants(child)
            node["end_index"] = max(node["end_index"], child["end_index"])

    for root in roots:
        include_descendants(root)
    from .utils import write_node_id
    write_node_id(roots)
    return pages, roots
