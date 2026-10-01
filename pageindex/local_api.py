"""Local implementation of the PageIndex SDK surface."""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

from .errors import PageIndexAPIError
from .local_store import DocStore
from .naming import sanitize_filename, truncate_filename
from .utils import count_tokens, run_off_loop

logger = logging.getLogger(__name__)

_SURROGATES = re.compile("[\ud800-\udfff]")


def _scrub_surrogates(text: str) -> str:
    """Lone surrogates (surrogateescape'd names, PyPDF2's surrogatepass
    decodes) cannot encode to UTF-8; replace with U+FFFD."""
    return _SURROGATES.sub("\ufffd", text)


def _now_iso() -> str:
    """Naive UTC, millisecond precision."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return now.replace(microsecond=now.microsecond // 1000 * 1000).isoformat()




class LocalAPI:
    """Backs PageIndexClient's local mode. One instance per client."""

    def __init__(self, storage_path: str, model: str, summary_model: str,
                 index_backend: dict | None = None,
                 summary_max_words: int | None = None,
                 summary_concurrency: int | None = None,
                 use_embedded_toc: bool = True,
                 optimize: str = "full",
                 ocr: str = "auto",
                 ocr_model: str | None = None):
        self._store = DocStore(storage_path)
        self._model = model
        self._summary_model = summary_model
        self._index_backend = index_backend
        self._summary_max_words = summary_max_words
        self._summary_concurrency = summary_concurrency
        self._use_embedded_toc = use_embedded_toc
        self._optimize = optimize
        from .ocr import validate_ocr_mode
        self._ocr = validate_ocr_mode(ocr)
        self._ocr_model = ocr_model
        from .utils import ConfigLoader
        self._config_loader = ConfigLoader()

    def _with_backend(self, func, *args):
        """Scope the indexing lane's connection overrides around one
        operation — runs inside whatever thread run_off_loop picked."""
        from .utils import _llm_backend
        token = _llm_backend.set(self._index_backend)
        try:
            return func(*args)
        finally:
            _llm_backend.reset(token)

    # ── indexing ──

    def submit_document(
        self,
        file_path: str,
        mode: str | None = None,
        beta_headers: list[str] | None = None,
        folder_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        if getattr(multiprocessing.current_process(), "_inheriting", False):
            raise PageIndexAPIError(
                "Failed to submit document: called again while a spawned worker "
                "process was importing your script. Put your top-level code under "
                "if __name__ == '__main__': so worker processes do not re-run it."
            )
        if beta_headers is not None:
            raise PageIndexAPIError(
                "Failed to submit document: beta_headers is not supported in local mode."
            )
        if folder_id is not None:
            raise PageIndexAPIError(
                "Failed to submit document: folders are not supported in local mode."
            )
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise PageIndexAPIError(
                    "Failed to submit document: metadata must be a dict."
                )
            try:
                json.dumps(metadata, allow_nan=False)
            except (TypeError, ValueError) as e:
                raise PageIndexAPIError(
                    f"Failed to submit document: metadata must be valid JSON. {e}"
                ) from e
        if mode not in (None, "standard", "flash"):
            raise PageIndexAPIError(
                f"Failed to submit document: unknown local processing mode {mode!r}. "
                "Supported: 'flash' (default) or 'standard'."
            )
        if mode is None:
            mode = "flash"
        if mode == "standard" and (self._summary_max_words is not None
                                   or self._summary_concurrency is not None):
            raise PageIndexAPIError(
                "Failed to submit document: summary_max_words and "
                "summary_concurrency are flash-only; mode='standard' does not "
                "support them.")
        file_path = os.path.abspath(os.path.expanduser(str(file_path)))
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: {file_path}")
        doc_name = sanitize_filename(os.path.basename(file_path))
        from .ocr import has_image_extension
        is_image = has_image_extension(doc_name)
        if not (doc_name.lower().endswith(".pdf") or is_image):
            raise PageIndexAPIError(
                "Failed to submit document: only PDF files and PNG or JPEG "
                "images (png, jpg, jpeg) are supported in local mode."
            )
        if is_image and self._ocr == "off":
            raise PageIndexAPIError(
                "Failed to submit document: image files need OCR, which this "
                "client disables (ocr='off'). Use ocr='auto' or 'force'."
            )

        # Fail a taken name before any vision call is paid for; the final
        # check-then-write below still runs under the store lock.
        self._unique_doc_name(doc_name)

        # A converted image lives only for this submit.
        with tempfile.TemporaryDirectory(prefix="pageindex-") as scratch:
            pdf_path = file_path
            if is_image:
                pdf_path = os.path.join(
                    scratch, os.path.splitext(doc_name)[0] + ".pdf")
                try:
                    from .ocr import image_to_pdf
                    image_to_pdf(file_path, pdf_path)
                except Exception as e:
                    raise PageIndexAPIError(
                        f"Failed to submit document: could not read image: {e}"
                    ) from e
            structure, description, page_texts, mode = self._index_file(
                pdf_path, doc_name, mode, needs_text_pipeline=is_image)
        self._check_page_bounds(structure, len(page_texts))

        doc_id = "pi-" + uuid.uuid4().hex
        pages = [{"page_index": i + 1, "markdown": text}
                 for i, text in enumerate(page_texts)]
        from .utils import remove_fields
        # Check-then-write under the store lock; the early pre-check above
        # is advisory only.
        with self._store.lock():
            meta = {
                "id": doc_id,
                "name": self._unique_doc_name(doc_name),
                "description": description,
                "status": "completed",
                "createdAt": _now_iso(),
                "pageNum": len(page_texts),
                "folderId": None,
                "metadata": metadata,
                "mode": mode,
            }
            self._store.save_document(
                doc_id, meta, remove_fields(structure, fields=["text"]), pages)
        return {"doc_id": doc_id, "name": meta["name"]}

    def _index_file(self, pdf_path: str, doc_name: str, mode: str,
                    needs_text_pipeline: bool) -> tuple[list, str | None, list[str], str]:
        """Read, OCR and index one PDF. Returns the tree, description, the
        final page texts and the pipeline that ran."""
        try:
            page_texts = self._extract_page_texts(pdf_path)
        except PageIndexAPIError:
            raise
        except Exception as e:
            raise PageIndexAPIError(
                f"Failed to submit document: could not read PDF: {e}"
            ) from e
        if self._ocr != "off":
            from .ocr import ocr_pages
            try:
                # Inside the indexing lane: its model and its credentials.
                result = run_off_loop(
                    self._with_backend, ocr_pages, pdf_path, page_texts,
                    self._ocr_model or self._model, self._ocr)
            except Exception as e:
                raise PageIndexAPIError(
                    f"Failed to submit document: OCR failed: {e}") from e
            page_texts = [_scrub_surrogates(text) for text in result.page_texts]
            # Flash reads the PDF text layer itself and cannot see what OCR
            # transcribed; those documents take the standard pipeline.
            needs_text_pipeline = needs_text_pipeline or result.needed_ocr
        if not any(text.strip() for text in page_texts):
            raise PageIndexAPIError(
                "Failed to submit document: PDF has no content. All pages are blank."
            )
        if mode == "flash" and needs_text_pipeline:
            logger.info("Document needed OCR; indexing it in standard mode.")
            mode = "standard"
        try:
            if mode == "flash":
                structure, description = run_off_loop(
                    self._with_backend, self._index_flash, pdf_path
                )
            else:
                structure, description = run_off_loop(
                    self._with_backend, self._index_standard, pdf_path,
                    page_texts
                )
        except PageIndexAPIError:
            raise
        except Exception as e:
            raise PageIndexAPIError(f"Failed to submit document: {e}") from e
        return structure, description, page_texts, mode

    def _unique_doc_name(self, name: str) -> str:
        """Mirror the cloud upload: a taken name gets _1.._99 appended,
        beyond that the submit is rejected."""
        taken = {meta.get("name") for meta in self._store.list_metas()}
        if name not in taken:
            return name
        for num in range(1, 100):
            candidate = truncate_filename(name, suffix=f"_{num}")
            if candidate not in taken:
                return candidate
        raise PageIndexAPIError(
            "Failed to submit document: Too many files with similar names. "
            "Please use a different file name."
        )

    @staticmethod
    def _check_page_bounds(structure: list, page_count: int) -> None:
        """The tree (pdfium) and stored pages (PyPDF2) come from different
        parsers; a span outside 1..page_count IndexErrors every later read."""
        stack = list(structure)
        while stack:
            node = stack.pop()
            start, end = node.get("start_index"), node.get("end_index")
            if (start is not None and end is not None
                    and not (1 <= start and end <= page_count)):
                raise PageIndexAPIError(
                    f"Failed to submit document: the extracted structure "
                    f"references pages {start}-{end} outside the PDF's "
                    f"{page_count} readable pages."
                )
            stack.extend(node.get("nodes") or [])

    @staticmethod
    def _extract_page_texts(file_path: str) -> list[str]:
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # PyPDF2 decodes broken ToUnicode maps with surrogatepass; lone
            # surrogates would crash every utf-8 JSON save downstream.
            return [_scrub_surrogates(page.extract_text() or "")
                    for page in reader.pages]

    def _index_standard(self, file_path: str, page_texts: list[str]) -> tuple[list, str | None]:
        from .page_index_classic import page_index_main
        page_list = [(text, count_tokens(text, model=self._model)) for text in page_texts]
        opt = self._config_loader.load({
            "model": self._model,
            "summary_model": self._summary_model,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_node_text": "yes",
            "if_add_doc_description": "yes",
        })
        result = page_index_main(file_path, opt, logger=logger, page_list=page_list)
        structure = result.get("structure") or []
        if not structure:
            raise PageIndexAPIError(
                "Failed to submit document: standard indexing produced no structure."
            )
        return structure, result.get("doc_description")

    def _index_flash(self, file_path: str) -> tuple[list, str | None]:
        from .flash import page_index_flash
        from .flash.api import flash_rejection_reason
        from .utils import (create_clean_structure_for_description,
                            generate_doc_description, write_node_id)
        result = page_index_flash(file_path, summary=True,
                                  summary_model=self._summary_model,
                                  optimize=False if self._optimize == "off" else self._optimize,
                                  optimize_model=self._summary_model,
                                  summary_concurrency=self._summary_concurrency,
                                  summary_max_words=self._summary_max_words,
                                  use_embedded_toc=self._use_embedded_toc)
        structure = result.get("structure", [])
        reason = flash_rejection_reason(result)
        if reason:
            raise PageIndexAPIError(f"Failed to submit document: {reason}")
        write_node_id(structure)
        description = generate_doc_description(
            create_clean_structure_for_description(structure),
            model=self._summary_model,
        )
        return structure, description

    # ── tree / ocr ──

    def _load_tree_with_text(self, doc_id: str, error_prefix: str) -> list:
        from .utils import add_node_text
        structure = self._require_data(
            self._store.get_tree(doc_id), error_prefix)
        pages = self._require_pages(doc_id, error_prefix)
        pdf_pages = [(p.get("markdown", ""), 0) for p in pages]
        add_node_text(structure, pdf_pages)
        return structure

    def raw_tree(self, doc_id: str) -> list | None:
        """Stored tree verbatim — keeps start_index/end_index, which
        get_tree's cloud wire shape renames and drops."""
        return self._store.get_tree(doc_id)

    def get_tree(self, doc_id: str, node_summary: bool = False,
                 include_text: bool = True) -> dict[str, Any]:
        meta = self._require_doc(doc_id, "Failed to get tree result")
        if include_text:
            structure = self._load_tree_with_text(doc_id, "Failed to get tree result")
        else:
            structure = self._require_data(
                self._store.get_tree(doc_id), "Failed to get tree result")
        result = [_format_tree_node(node, node_summary) for node in structure]
        return self._completed_envelope(doc_id, result, meta)

    def get_ocr(self, doc_id: str, format: str = "page") -> dict[str, Any]:
        if format not in ["page", "node", "raw"]:
            raise ValueError("Format parameter must be 'page', 'node', or 'raw'")
        meta = self._require_doc(doc_id, "Failed to get OCR result")
        if format == "node":
            result: Any = []
            def _walk(nodes, level):
                for node in nodes:
                    result.append({
                        "title": node.get("title", ""),
                        "level": level,
                        "page_index": node.get("start_index"),
                        "text": node.get("text", ""),
                    })
                    _walk(node.get("nodes") or [], level + 1)
            _walk(self._load_tree_with_text(doc_id,
                                           "Failed to get OCR result"), 1)
        else:
            pages = self._require_pages(doc_id, "Failed to get OCR result")
            if format == "page":
                result = pages
            else:  # raw
                result = "\n\n".join(p.get("markdown", "") for p in pages)
        return self._completed_envelope(doc_id, result, meta)

    @staticmethod
    def _require_data(data, error_prefix: str):
        if data is None:
            raise PageIndexAPIError(f"{error_prefix}: stored document data is unreadable.")
        return data

    def _require_pages(self, doc_id: str, error_prefix: str) -> list:
        pages = self._require_data(self._store.get_pages(doc_id), error_prefix)
        if not pages:
            raise PageIndexAPIError(
                f"{error_prefix}: stored document has no page content.")
        return pages

    def _completed_envelope(self, doc_id: str, result, meta: dict) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "status": "completed",
            "retrieval_ready": True,
            "result": result,
            "metadata": meta.get("metadata"),
            "features": {},
        }

    # ── document management ──

    def _require_doc(self, doc_id: str, error_prefix: str) -> dict:
        meta = self._store.get_meta(doc_id)
        if meta is None:
            raise PageIndexAPIError(f"{error_prefix}: Document not found.")
        return meta

    def get_document(self, doc_id: str) -> dict[str, Any]:
        meta = self._store.get_meta(doc_id)
        if meta is None:
            raise PageIndexAPIError("Failed to get document metadata: Document not found")
        return {key: meta.get(key) for key in
                ("id", "name", "description", "status", "createdAt", "pageNum",
                 "folderId", "metadata")}

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        if not self._store.delete_document(doc_id):
            raise PageIndexAPIError("Failed to delete document: Document not found.")
        return {"message": "Document deleted successfully."}

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        folder_id: str | None = None,
        name: str | None = None,
        recursive: bool = False,
    ) -> dict[str, Any]:
        if limit < 1 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if folder_id is not None and folder_id not in ("", "root"):
            raise PageIndexAPIError(
                "Failed to list documents: folders are not supported in local mode."
            )
        metas = sorted(self._store.list_metas(), key=lambda m: m.get("id") or "")
        metas.sort(key=lambda m: m.get("createdAt") or "", reverse=True)
        if name is not None:
            metas = [m for m in metas if m.get("name") == name]
        documents = [{
            "id": m.get("id"),
            "name": m.get("name"),
            "description": m.get("description"),
            "status": m.get("status"),
            "createdAt": m.get("createdAt"),
            "pageNum": m.get("pageNum", 0),
            "folderId": None,
            "path": None,
            "metadata": m.get("metadata"),
            "features": {},
        } for m in metas[offset:offset + limit]]
        return {
            "documents": documents,
            "total": len(metas),
            "limit": limit,
            "offset": offset,
        }


def _format_tree_node(node: dict, node_summary: bool) -> dict:
    children = node.get("nodes") or []
    out = {
        "title": node.get("title", ""),
        "node_id": node.get("node_id"),
        "page_index": node.get("start_index"),
    }
    if node.get("key_items"):
        out["key_items"] = node["key_items"]
    if node_summary:
        summary = node.get("summary")
        if summary is not None:
            if children:
                out["prefix_summary"] = summary
            else:
                out["summary"] = summary
    if "text" in node:
        out["text"] = node["text"]
    if children:
        out["nodes"] = [_format_tree_node(child, node_summary) for child in children]
    return out
