"""Persistent indexing queue for the HTTP API.

Each upload becomes a job directory under ``<storage>/jobs/<job_id>/`` holding
``job.json`` and the uploaded PDF. Because the queue lives on disk, pending
work survives restarts: ``JobStore.recover()`` re-queues it and
``JobRunner`` indexes it on background threads outside the HTTP threadpool.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger("pageindex.server.jobs")

STATUSES = ("queued", "processing", "done", "failed")
_JOB_ID = re.compile(r"^job-[0-9a-f]{32}$")
_JOB_FILE = "job.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_json_atomic(path: str, data: dict) -> None:
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class JobStore:
    """Job records on disk. Every status change goes through ``update`` under
    one lock, so transitions (claim, retry) are race-free in this process.
    An in-memory status index keeps ``counts()`` cheap for ``/health``."""

    def __init__(self, root: str):
        self._root = root
        self._lock = threading.Lock()
        self._statuses: dict[str, str] = {}
        self._last_created: Optional[datetime] = None

    # ---------- paths ----------

    def _dir(self, job_id: str) -> Optional[str]:
        if not isinstance(job_id, str) or not _JOB_ID.match(job_id):
            return None
        return os.path.join(self._root, job_id)

    def _read(self, job_id: str) -> Optional[dict]:
        job_dir = self._dir(job_id)
        if job_dir is None:
            return None
        try:
            with open(os.path.join(job_dir, _JOB_FILE), encoding="utf-8") as handle:
                job = json.load(handle)
        except (OSError, ValueError):
            return None
        if not isinstance(job, dict) or job.get("id") != job_id \
                or job.get("status") not in STATUSES:
            return None
        return job

    def pdf_path(self, job_id: str) -> str:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return os.path.join(self._dir(job_id), job["name"])

    # ---------- records ----------

    def create(self, name: str, src_path: str) -> dict:
        """Move the validated upload at ``src_path`` into a new queued job.
        ``name`` must already be a bare, safe ``.pdf`` file name."""
        job_id = "job-" + uuid.uuid4().hex
        job_dir = os.path.join(self._root, job_id)
        os.makedirs(job_dir)
        shutil.move(src_path, os.path.join(job_dir, name))
        with self._lock:
            now = _now()
            # Keep creation order strict even when the clock is coarse.
            if self._last_created is not None and now <= self._last_created:
                now = self._last_created + timedelta(microseconds=1)
            self._last_created = now
            job = {"id": job_id, "name": name, "status": "queued",
                   "created_at": _iso(now), "updated_at": _iso(now),
                   "doc_id": None, "error": None, "attempts": 0}
            # job.json is written last: a directory without it is ignored.
            _write_json_atomic(os.path.join(job_dir, _JOB_FILE), job)
            self._statuses[job_id] = "queued"
        return job

    def get(self, job_id: str) -> Optional[dict]:
        return self._read(job_id)

    def update(self, job_id: str, expect: Optional[tuple] = None,
               **fields: Any) -> Optional[dict]:
        """Apply ``fields`` atomically. With ``expect``, only when the current
        status is one of them (returns None otherwise). Unknown id: KeyError."""
        with self._lock:
            job = self._read(job_id)
            if job is None:
                raise KeyError(job_id)
            if expect is not None and job["status"] not in expect:
                return None
            job.update(fields)
            job["updated_at"] = _iso(_now())
            _write_json_atomic(os.path.join(self._dir(job_id), _JOB_FILE), job)
            self._statuses[job_id] = job["status"]
            return job

    def delete_pdf(self, job_id: str) -> None:
        try:
            os.remove(self.pdf_path(job_id))
        except (OSError, KeyError):
            pass

    def list(self) -> list[dict]:
        """All readable jobs, newest first."""
        try:
            names = os.listdir(self._root)
        except FileNotFoundError:
            return []
        jobs = [job for job in map(self._read, names) if job is not None]
        jobs.sort(key=lambda job: (job["created_at"], job["id"]), reverse=True)
        return jobs

    def recover(self) -> list[str]:
        """Rebuild the status index after a (re)start: interrupted
        ``processing`` jobs go back to ``queued``. Returns the queued ids,
        oldest first, for the runner to enqueue."""
        pending = []
        with self._lock:
            self._statuses = {}
        for job in reversed(self.list()):
            if job["status"] == "processing":
                job = self.update(job["id"], status="queued")
            with self._lock:
                self._statuses[job["id"]] = job["status"]
            if job["status"] == "queued":
                pending.append(job["id"])
        return pending

    def counts(self) -> dict[str, int]:
        with self._lock:
            values = list(self._statuses.values())
        return {"queued": values.count("queued"),
                "processing": values.count("processing")}


def _find_doc_for_job(client: Any, job_id: str) -> Optional[str]:
    """Document already indexed for ``job_id``, if any (matched through the
    ``job_id`` tag the runner stores in the document metadata)."""
    offset = 0
    while True:
        page = client.list_documents(limit=1000, offset=offset)
        docs = page.get("documents") or []
        for doc in docs:
            if (doc.get("metadata") or {}).get("job_id") == job_id:
                return doc.get("id")
        offset += len(docs)
        if not docs or offset >= page.get("total", 0):
            return None


class JobRunner:
    """Background indexing threads fed by an in-memory queue of job ids."""

    def __init__(self, store: JobStore, get_client: Callable[[], Any],
                 describe_error: Callable[[Exception], str], workers: int = 1):
        self._store = store
        self._get_client = get_client
        self._describe_error = describe_error
        self._workers = workers
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for job_id in self._store.recover():
            self._queue.put(job_id)
        for num in range(self._workers):
            thread = threading.Thread(target=self._loop, daemon=True,
                                      name=f"pageindex-indexer-{num}")
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        """Ask workers to exit after their current job. A job still running
        at shutdown stays ``processing`` on disk and resumes on next start."""
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout)
        self._threads = []

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                self._run(job_id)
            except Exception:  # never let one job kill the worker
                logger.exception("Job %s: unexpected runner failure", job_id)

    def _run(self, job_id: str) -> None:
        job = self._store.get(job_id)
        if job is None:
            return
        # Claim: only a queued job starts, so a duplicate queue entry is a no-op.
        job = self._store.update(job_id, expect=("queued",), status="processing",
                                 attempts=job["attempts"] + 1, error=None)
        if job is None:
            return
        try:
            client = self._get_client()
            doc_id = None
            if job["attempts"] > 1:
                # A previous attempt may have finished indexing and died
                # before recording it. Re-submitting would store a second copy
                # (the SDK renames a taken name to name_1 instead of
                # rejecting it), so reuse the document tagged with this job.
                doc_id = _find_doc_for_job(client, job_id)
            if doc_id is None:
                result = client.submit_document(self._store.pdf_path(job_id),
                                                metadata={"job_id": job_id})
                doc_id = result["doc_id"]
        except Exception as exc:
            error = self._describe_error(exc)
            logger.warning("Job %s failed: %s", job_id, error)
            self._store.update(job_id, status="failed", error=error)
            return
        self._store.update(job_id, status="done", doc_id=doc_id)
        self._store.delete_pdf(job_id)
