"""HTTP API over a local-mode PageIndexClient.

Configuration comes from the environment (see .env.example):

- PAGEINDEX_INDEX_MODEL / PAGEINDEX_CHAT_MODEL: models for indexing and chat,
  each optional (unset means the SDK default).
- PAGEINDEX_INDEX_BASE_URL / _API_KEY, PAGEINDEX_CHAT_BASE_URL / _API_KEY:
  optional per-role provider overrides; unset falls back to OPENAI_*.
- PAGEINDEX_STORAGE_PATH: document store (default /app/storage).
- PAGEINDEX_API_TOKEN: when set, every endpoint but /health requires
  ``Authorization: Bearer <token>``.
- PAGEINDEX_MAX_UPLOAD_MB: upload size limit (default 50).
- PAGEINDEX_INDEX_WORKERS: documents indexed in parallel (default 1).
- PAGEINDEX_OCR: off, auto (default) or force — OCR through the index
  model's vision input for scanned pages, figures and image uploads.
- PAGEINDEX_OCR_MODEL: vision model for OCR (default: the index model).

Uploads are PDFs or images (png, jpg, jpeg, webp, tif, tiff, bmp, gif),
checked by extension and by content.

Uploads are queued as jobs under ``<storage>/jobs`` and indexed in the
background (see server/jobs.py); poll ``GET /jobs/{job_id}`` for the result.

Run with ``uvicorn server.app:app``.
"""
from __future__ import annotations

import hmac
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal, Mapping, Optional, Union

from fastapi import (Depends, FastAPI, File, HTTPException, Query, Request,
                     Response, UploadFile)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from pageindex import PageIndexAPIError
from pageindex.ocr import (IMAGE_EXTENSIONS, OCR_MODES, has_image_extension,
                           sniff_image_format)
from pageindex.utils import LLMRetriesExhausted
from server.jobs import JobRunner, JobStore

logger = logging.getLogger("pageindex.server")

DEFAULT_STORAGE_PATH = "/app/storage"
DEFAULT_MAX_UPLOAD_MB = 50
DEFAULT_INDEX_WORKERS = 1
# Seconds a failed model client build is cached before it is tried again.
CLIENT_RETRY_AFTER_SECONDS = 30.0
_MAX_JOB_ERROR = 300
_CHUNK = 1024 * 1024
# Room for multipart boundaries and part headers around the file itself.
_MULTIPART_MARGIN = 64 * 1024
# Exceptions raised by these packages come from the model provider round trip.
_UPSTREAM_MODULES = {"litellm", "openai", "httpx", "httpcore"}


# ---------- configuration ----------

def _set(env: Mapping[str, str], name: str) -> Optional[str]:
    value = env.get(name)
    return value if value else None


def _ocr_mode(env: Mapping[str, str]) -> Optional[str]:
    raw = _set(env, "PAGEINDEX_OCR")
    if raw is None:
        return None
    mode = raw.strip().lower()
    if mode not in OCR_MODES:
        raise ValueError(
            f"PAGEINDEX_OCR must be one of {', '.join(OCR_MODES)}, got {raw!r}")
    return mode


def client_kwargs(env: Mapping[str, str]) -> dict[str, Any]:
    """PageIndexClient constructor kwargs; unset variables are omitted so the
    SDK defaults (and OPENAI_* connection settings) apply."""
    kwargs: dict[str, Any] = {
        "storage_path": _set(env, "PAGEINDEX_STORAGE_PATH") or DEFAULT_STORAGE_PATH,
    }
    for key, var in (("index_model", "PAGEINDEX_INDEX_MODEL"),
                     ("chat_model", "PAGEINDEX_CHAT_MODEL"),
                     ("ocr_model", "PAGEINDEX_OCR_MODEL")):
        if _set(env, var):
            kwargs[key] = env[var]
    ocr = _ocr_mode(env)
    if ocr is not None:
        kwargs["ocr"] = ocr
    # index_backend takes LiteLLM's vocabulary (api_base); chat_backend takes
    # the chat surfaces' own (base_url).
    for key, url_key, prefix in (("index_backend", "api_base", "PAGEINDEX_INDEX_"),
                                 ("chat_backend", "base_url", "PAGEINDEX_CHAT_")):
        backend = {}
        if _set(env, prefix + "BASE_URL"):
            backend[url_key] = env[prefix + "BASE_URL"]
        if _set(env, prefix + "API_KEY"):
            backend["api_key"] = env[prefix + "API_KEY"]
        if backend:
            kwargs[key] = backend
    return kwargs


def build_client(env: Optional[Mapping[str, str]] = None,
                 client_cls: Optional[Callable[..., Any]] = None):
    if env is None:
        env = os.environ
    if client_cls is None:
        from pageindex import PageIndexClient
        client_cls = PageIndexClient
    return client_cls(**client_kwargs(env))


def _max_upload_bytes(env: Mapping[str, str]) -> int:
    raw = _set(env, "PAGEINDEX_MAX_UPLOAD_MB")
    if raw is None:
        return DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    try:
        mb = float(raw)
    except ValueError:
        mb = math.nan
    # A tiny positive value (e.g. 1e-9) would round down to 0 bytes and
    # refuse every upload, so the converted limit must be at least 1 byte.
    size = int(mb * 1024 * 1024) if math.isfinite(mb) and mb > 0 else 0
    if size < 1:
        raise ValueError(
            "PAGEINDEX_MAX_UPLOAD_MB must be a finite number > 0 that is at "
            f"least 1 byte (>= 1/1048576 MB), got {raw!r}")
    return size


def _index_workers(env: Mapping[str, str]) -> int:
    raw = _set(env, "PAGEINDEX_INDEX_WORKERS")
    if raw is None:
        return DEFAULT_INDEX_WORKERS
    try:
        workers = int(raw)
    except ValueError:
        workers = 0
    if workers < 1:
        raise ValueError(
            f"PAGEINDEX_INDEX_WORKERS must be an integer >= 1, got {raw!r}")
    return workers


# ---------- error mapping ----------

class ClientUnavailable(Exception):
    """The model client could not be built. A server-side configuration
    problem, never the caller's input; the cause is logged, not returned."""

    def __init__(self, cause: BaseException):
        super().__init__(type(cause).__name__)
        self.cause_name = type(cause).__name__


def _is_upstream(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, LLMRetriesExhausted):
            return True
        if type(exc).__module__.split(".")[0] in _UPSTREAM_MODULES:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _http_error(exc: Exception) -> HTTPException:
    """Translate an SDK failure. Upstream provider errors never echo their
    message: provider errors can quote request details."""
    if isinstance(exc, ClientUnavailable):
        return HTTPException(
            503, f"Model client is not configured ({exc.cause_name}); "
                 "check the server logs.")
    if _is_upstream(exc):
        root = exc
        while root.__cause__ is not None:
            root = root.__cause__
        name = type(root).__name__
        logger.error("Model provider call failed: %s", name)
        detail = (f"Upstream model provider error ({name}). "
                  "Check the model name, base URL and API key.")
        if name == "NotFoundError":
            # A fixed hint, never the provider's text: model ids containing
            # "/" are read by LiteLLM as "provider/model" unless prefixed.
            detail += (" If the model id contains '/', prefix it with "
                       "'openai/' (e.g. openai/vendor/model) so it is sent "
                       "to OPENAI_BASE_URL.")
        return HTTPException(502, detail)
    if isinstance(exc, PageIndexAPIError):
        message = str(exc)
        if "not found" in message.lower():
            return HTTPException(404, message)
        return HTTPException(400, message)
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return HTTPException(400, str(exc))
    raise exc


def _job_error(exc: Exception) -> str:
    """Error text stored on a failed job, under the same no-leak policy as
    ``_http_error``; anything unclassified is reported by class name only."""
    try:
        message = str(_http_error(exc).detail)
    except Exception:
        logger.exception("Indexing failed")
        message = f"Indexing failed ({type(exc).__name__})."
    if len(message) > _MAX_JOB_ERROR:
        message = message[:_MAX_JOB_ERROR - 3] + "..."
    return message


# ---------- request models ----------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_id: Optional[Union[str, list[str]]] = None


class ChatResponse(BaseModel):
    answer: str


# ---------- app ----------

_UPLOAD_TYPES = ", ".join([".pdf", *sorted(IMAGE_EXTENSIONS)])


def _safe_upload_name(filename: Optional[str]) -> str:
    """The bare file name of a PDF or supported image upload."""
    name = re.split(r"[\\/]", filename or "")[-1].strip()
    stem, ext = os.path.splitext(name)
    if not stem or not (ext.lower() == ".pdf" or has_image_extension(name)):
        raise HTTPException(
            415, f"Only PDF or image uploads are supported ({_UPLOAD_TYPES}).")
    return name


def _check_upload_content(name: str, path: str) -> None:
    """The bytes must match the declared kind: a PDF header, or an image
    Pillow recognizes."""
    if name.lower().endswith(".pdf"):
        with open(path, "rb") as check:
            if not check.read(5).startswith(b"%PDF-"):
                raise HTTPException(400, "File is not a PDF.")
    elif sniff_image_format(path) is None:
        raise HTTPException(400, "File is not a supported image.")


class _DeclaredUploadLimit:
    """ASGI middleware: rejects ``POST <path>`` by its declared Content-Length
    before any of the body is read, so an oversized upload is never parsed
    or spooled to disk. Requests without Content-Length (chunked) fall
    through to the handler's streaming size check."""

    def __init__(self, app, path: str, max_bytes: int, detail: str):
        self.app = app
        self.path = path
        self.max_bytes = max_bytes
        self.detail = detail

    async def __call__(self, scope, receive, send):
        if (scope["type"] == "http" and scope["method"] == "POST"
                and scope["path"] == self.path):
            declared = next((value for key, value in scope["headers"]
                             if key == b"content-length"), None)
            if declared is not None:
                rejection = None
                if not re.fullmatch(rb"\d+", declared.strip()):
                    rejection = (400, "Invalid Content-Length header.")
                elif int(declared) > self.max_bytes:
                    rejection = (413, self.detail)
                if rejection is not None:
                    status, detail = rejection
                    response = JSONResponse({"detail": detail}, status_code=status)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def create_app(client: Any = None, env: Optional[Mapping[str, str]] = None,
               client_factory: Callable[[Mapping[str, str]], Any] = build_client,
               client_retry_after: float = CLIENT_RETRY_AFTER_SECONDS,
               clock: Callable[[], float] = time.monotonic,
               ) -> FastAPI:
    """Build the API. Pass ``client`` to inject one (tests); otherwise it is
    built from ``env`` on first use, so importing this module is cheap. A
    failed build is retried at most once per ``client_retry_after`` seconds
    (measured with ``clock``)."""
    env = dict(os.environ if env is None else env)
    token = _set(env, "PAGEINDEX_API_TOKEN")
    max_upload = _max_upload_bytes(env)
    _ocr_mode(env)  # fail at startup, like the other settings
    too_large = f"Upload exceeds {max_upload / (1024 * 1024):g} MB."
    workers = _index_workers(env)
    storage = _set(env, "PAGEINDEX_STORAGE_PATH") or DEFAULT_STORAGE_PATH

    # "failed_at"/"error" hold the last failed build. Within the cooldown every
    # caller (including unauthenticated /health probes) fails fast instead of
    # rebuilding the client and logging a traceback each time.
    state = {"client": client, "failed_at": None, "error": None}
    client_lock = threading.Lock()

    def get_client():
        if state["client"] is None:
            with client_lock:
                if state["client"] is None:
                    _build_client_locked()
        return state["client"]

    def _build_client_locked() -> None:
        failed_at, error = state["failed_at"], state["error"]
        if failed_at is not None and clock() - failed_at < client_retry_after:
            raise ClientUnavailable(error) from error
        try:
            built = client_factory(env)
        except Exception as exc:
            if failed_at is None:
                logger.exception("Could not build the model client")
            else:
                # Same failure streak: the traceback was already logged once.
                logger.warning(
                    "Could not build the model client (%s); retrying in %gs",
                    type(exc).__name__, client_retry_after)
            state["failed_at"], state["error"] = clock(), exc
            raise ClientUnavailable(exc) from exc
        if failed_at is not None:
            logger.info("Model client recovered")
        state["client"], state["failed_at"], state["error"] = built, None, None

    def require_token(request: Request) -> None:
        if token is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
                supplied.strip().encode(), token.encode()):
            raise HTTPException(401, "Invalid or missing bearer token.",
                                headers={"WWW-Authenticate": "Bearer"})

    # Indexing runs many concurrent model calls per document; the worker
    # count (default 1) bounds provider load. The SDK store locks its writes.
    jobs = JobStore(os.path.join(storage, "jobs"))
    runner = JobRunner(jobs, get_client, _job_error, workers=workers)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runner.start()
        try:
            yield
        finally:
            # stop() joins worker threads; keep that off the event loop.
            await run_in_threadpool(runner.stop)

    app = FastAPI(title="PageIndex API", version="1.0", lifespan=lifespan)
    app.add_middleware(_DeclaredUploadLimit, path="/documents",
                       max_bytes=max_upload + _MULTIPART_MARGIN, detail=too_large)
    protected = [Depends(require_token)]

    @app.get("/health")
    def health(response: Response) -> dict[str, Any]:
        # 503 when the client cannot be built, so container healthchecks fail.
        # The cause stays in the server logs.
        try:
            get_client()
            client_state = "ok"
        except ClientUnavailable:
            client_state = "error"
            response.status_code = 503
        return {
            "status": "ok" if client_state == "ok" else "error",
            "client": client_state,
            "index_model": _set(env, "PAGEINDEX_INDEX_MODEL"),
            "chat_model": _set(env, "PAGEINDEX_CHAT_MODEL"),
            "auth": token is not None,
            "queue": jobs.counts(),
        }

    @app.post("/documents", status_code=202, dependencies=protected)
    def upload_document(response: Response,
                        file: UploadFile = File(...)) -> dict[str, Any]:
        name = _safe_upload_name(file.filename)
        workdir = tempfile.mkdtemp(prefix="pageindex-upload-")
        try:
            path = os.path.join(workdir, name)
            size = 0
            with open(path, "wb") as out:
                while chunk := file.file.read(_CHUNK):
                    size += len(chunk)
                    if size > max_upload:
                        raise HTTPException(
                            413, too_large)
                    out.write(chunk)
            _check_upload_content(name, path)
            try:
                job = jobs.create(name, path)
            except OSError as exc:
                logger.exception("Could not store upload as a job")
                raise HTTPException(
                    503, "Could not store the upload; try again later.") from exc
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        runner.enqueue(job["id"])
        response.headers["Location"] = f"/jobs/{job['id']}"
        return {"job_id": job["id"], "status": job["status"], "name": name}

    @app.get("/jobs", dependencies=protected)
    def list_jobs(status: Optional[Literal["queued", "processing", "done", "failed"]] = None,
                  limit: int = Query(50, ge=1, le=10000),
                  offset: int = Query(0, ge=0)) -> dict[str, Any]:
        found = [job for job in jobs.list() if status is None or job["status"] == status]
        return {"jobs": found[offset:offset + limit], "total": len(found),
                "limit": limit, "offset": offset}

    def _job_or_404(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found.")
        return job

    @app.get("/jobs/{job_id}", dependencies=protected)
    def get_job(job_id: str) -> dict[str, Any]:
        return _job_or_404(job_id)

    @app.post("/jobs/{job_id}/retry", status_code=202, dependencies=protected)
    def retry_job(job_id: str, response: Response) -> dict[str, Any]:
        _job_or_404(job_id)
        # Failed jobs, and processing jobs no worker is running (their final
        # state could not be saved), go back to the queue.
        job = runner.retry(job_id)
        if job is None:
            raise HTTPException(409, "Job is queued, running, or already done.")
        response.headers["Location"] = f"/jobs/{job_id}"
        return job

    @app.get("/documents", dependencies=protected)
    def list_documents(limit: int = Query(50, ge=1, le=10000),
                       offset: int = Query(0, ge=0)) -> dict[str, Any]:
        try:
            return get_client().list_documents(limit=limit, offset=offset)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.get("/documents/{doc_id}", dependencies=protected)
    def get_document(doc_id: str) -> dict[str, Any]:
        try:
            return get_client().get_document(doc_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.delete("/documents/{doc_id}", dependencies=protected)
    def delete_document(doc_id: str) -> dict[str, Any]:
        try:
            return get_client().delete_document(doc_id)
        except Exception as exc:
            raise _http_error(exc) from exc

    @app.post("/chat", response_model=ChatResponse, dependencies=protected)
    def chat(body: ChatRequest) -> ChatResponse:
        try:
            answer = get_client().chat(body.question, doc_id=body.doc_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return ChatResponse(answer=answer)

    return app


app = create_app()
