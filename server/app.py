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

Uploads are queued as jobs under ``<storage>/jobs`` and indexed in the
background (see server/jobs.py); poll ``GET /jobs/{job_id}`` for the result.

Run with ``uvicorn server.app:app``.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal, Mapping, Optional, Union

from fastapi import (Depends, FastAPI, File, HTTPException, Query, Request,
                     Response, UploadFile)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from pageindex import PageIndexAPIError
from pageindex.utils import LLMRetriesExhausted
from server.jobs import JobRunner, JobStore

logger = logging.getLogger("pageindex.server")

DEFAULT_STORAGE_PATH = "/app/storage"
DEFAULT_MAX_UPLOAD_MB = 50
DEFAULT_INDEX_WORKERS = 1
_MAX_JOB_ERROR = 300
_CHUNK = 1024 * 1024
# Exceptions raised by these packages come from the model provider round trip.
_UPSTREAM_MODULES = {"litellm", "openai", "httpx", "httpcore"}


# ---------- configuration ----------

def _set(env: Mapping[str, str], name: str) -> Optional[str]:
    value = env.get(name)
    return value if value else None


def client_kwargs(env: Mapping[str, str]) -> dict[str, Any]:
    """PageIndexClient constructor kwargs; unset variables are omitted so the
    SDK defaults (and OPENAI_* connection settings) apply."""
    kwargs: dict[str, Any] = {
        "storage_path": _set(env, "PAGEINDEX_STORAGE_PATH") or DEFAULT_STORAGE_PATH,
    }
    for key, var in (("index_model", "PAGEINDEX_INDEX_MODEL"),
                     ("chat_model", "PAGEINDEX_CHAT_MODEL")):
        if _set(env, var):
            kwargs[key] = env[var]
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
    try:
        mb = float(raw) if raw else DEFAULT_MAX_UPLOAD_MB
    except ValueError:
        raise ValueError(f"PAGEINDEX_MAX_UPLOAD_MB must be a number, got {raw!r}")
    return int(mb * 1024 * 1024)


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
    if _is_upstream(exc):
        root = exc
        while root.__cause__ is not None:
            root = root.__cause__
        logger.error("Model provider call failed: %s", type(root).__name__)
        return HTTPException(
            502, f"Upstream model provider error ({type(root).__name__}). "
                 "Check the model name, base URL and API key.")
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

def _safe_pdf_name(filename: Optional[str]) -> str:
    name = re.split(r"[\\/]", filename or "")[-1].strip()
    if not name.lower().endswith(".pdf") or name.lower() == ".pdf":
        raise HTTPException(415, "Only PDF uploads are supported (.pdf).")
    return name


def create_app(client: Any = None, env: Optional[Mapping[str, str]] = None,
               client_factory: Callable[[Mapping[str, str]], Any] = build_client
               ) -> FastAPI:
    """Build the API. Pass ``client`` to inject one (tests); otherwise it is
    built from ``env`` on first use, so importing this module is cheap."""
    env = dict(os.environ if env is None else env)
    token = _set(env, "PAGEINDEX_API_TOKEN")
    max_upload = _max_upload_bytes(env)
    workers = _index_workers(env)
    storage = _set(env, "PAGEINDEX_STORAGE_PATH") or DEFAULT_STORAGE_PATH

    state = {"client": client}
    client_lock = threading.Lock()

    def get_client():
        if state["client"] is None:
            with client_lock:
                if state["client"] is None:
                    state["client"] = client_factory(env)
        return state["client"]

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
    protected = [Depends(require_token)]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "index_model": _set(env, "PAGEINDEX_INDEX_MODEL"),
            "chat_model": _set(env, "PAGEINDEX_CHAT_MODEL"),
            "auth": token is not None,
            "queue": jobs.counts(),
        }

    @app.post("/documents", status_code=202, dependencies=protected)
    def upload_document(response: Response,
                        file: UploadFile = File(...)) -> dict[str, Any]:
        name = _safe_pdf_name(file.filename)
        workdir = tempfile.mkdtemp(prefix="pageindex-upload-")
        try:
            path = os.path.join(workdir, name)
            size = 0
            with open(path, "wb") as out:
                while chunk := file.file.read(_CHUNK):
                    size += len(chunk)
                    if size > max_upload:
                        raise HTTPException(
                            413, f"Upload exceeds {max_upload // (1024 * 1024)} MB.")
                    out.write(chunk)
            with open(path, "rb") as check:
                if not check.read(5).startswith(b"%PDF-"):
                    raise HTTPException(400, "File is not a PDF.")
            job = jobs.create(name, path)
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
