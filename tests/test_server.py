"""HTTP service tests: server.app wraps PageIndexClient behind FastAPI.

A fake client stands in for the SDK so no model or PDF pipeline runs.
"""
import asyncio
import json
import os
import re
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from pageindex import PageIndexAPIError
from server.app import build_client, client_kwargs, create_app

PDF_BYTES = b"%PDF-1.4\n%fake minimal body\n"


class FakeUpstreamError(Exception):
    """Mimics a LiteLLM provider exception (module-based detection)."""


FakeUpstreamError.__module__ = "litellm.exceptions"


class FakeStream:
    def __init__(self, events, events_error=None):
        self._events = events
        self.events_error = events_error
        self.closed = False

    @property
    def events(self):
        if self.events_error:
            raise self.events_error
        return self._events

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self):
        self.submitted = []
        self.chats = []
        self.docs = {"pi-1": {"id": "pi-1", "name": "a.pdf", "status": "completed"}}
        self.chat_error = None
        self.submit_error = None
        self.metadata = []
        self.gate = None
        self.chat_stream = None
        self.chat_calls = []
        self.citation_entries = []
        self.citation_error = None
        self.tree = None
        self.tree_error = None
        self._api = self

    def submit_document(self, file_path, metadata=None):
        with open(file_path, "rb") as handle:
            self.submitted.append((file_path, handle.read()))
        self.metadata.append(metadata)
        if self.gate is not None:
            self.gate.wait(5)
        if self.submit_error:
            raise self.submit_error
        return {"doc_id": "pi-new", "name": "report.pdf"}

    def list_documents(self, limit=50, offset=0):
        docs = list(self.docs.values())
        return {"documents": docs, "total": len(docs), "limit": limit, "offset": offset}

    def get_document(self, doc_id):
        if doc_id not in self.docs:
            raise PageIndexAPIError("Failed to get document metadata: Document not found")
        return self.docs[doc_id]

    def delete_document(self, doc_id):
        if self.docs.pop(doc_id, None) is None:
            raise PageIndexAPIError("Failed to delete document: Document not found.")
        return {"message": "Document deleted successfully."}

    def chat(self, question, doc_id=None, **kwargs):
        self.chats.append((question, doc_id))
        self.chat_calls.append((question, doc_id, kwargs))
        if self.chat_error:
            raise self.chat_error
        if kwargs.get("stream"):
            return self.chat_stream
        return f"answer to {question}"

    def get_citations(self, answer, doc_id=None):
        if self.citation_error:
            raise self.citation_error
        return self.citation_entries

    def raw_tree(self, doc_id):
        if self.tree_error:
            raise self.tree_error
        return self.tree


@pytest.fixture
def make(tmp_path):
    def _make(env=None, client=None):
        fake = client or FakeClient()
        full_env = {"PAGEINDEX_STORAGE_PATH": str(tmp_path), **(env or {})}
        return TestClient(create_app(client=fake, env=full_env)), fake
    return _make


JOB_ID = re.compile(r"^job-[0-9a-f]{32}$")


def wait_for_status(api, job_id, status, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        job = api.get(f"/jobs/{job_id}").json()
        if job.get("status") == status:
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job_id} stuck at {job.get('status')!r}")
        time.sleep(0.02)


# ---------- configuration ----------

def test_client_kwargs_keep_index_and_chat_separate():
    kwargs = client_kwargs({
        "PAGEINDEX_INDEX_MODEL": "openai/index-m",
        "PAGEINDEX_CHAT_MODEL": "openai/chat-m",
        "PAGEINDEX_INDEX_BASE_URL": "http://index:1/v1",
        "PAGEINDEX_INDEX_API_KEY": "ik",
        "PAGEINDEX_CHAT_BASE_URL": "http://chat:2/v1",
        "PAGEINDEX_CHAT_API_KEY": "ck",
        "PAGEINDEX_STORAGE_PATH": "/data/store",
    })
    assert kwargs == {
        "index_model": "openai/index-m",
        "chat_model": "openai/chat-m",
        "index_backend": {"api_base": "http://index:1/v1", "api_key": "ik"},
        "chat_backend": {"base_url": "http://chat:2/v1", "api_key": "ck"},
        "storage_path": "/data/store",
    }


def test_client_kwargs_omit_unset_values():
    assert client_kwargs({}) == {"storage_path": "/app/storage"}
    partial = client_kwargs({"PAGEINDEX_CHAT_BASE_URL": "http://chat/v1",
                             "PAGEINDEX_INDEX_MODEL": ""})
    assert partial == {"storage_path": "/app/storage",
                       "chat_backend": {"base_url": "http://chat/v1"}}


def test_build_client_passes_kwargs_to_constructor():
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return "client"

    env = {"PAGEINDEX_INDEX_MODEL": "a", "PAGEINDEX_CHAT_MODEL": "b"}
    assert build_client(env, client_cls=factory) == "client"
    assert seen["index_model"] == "a" and seen["chat_model"] == "b"


# ---------- health and auth ----------

def test_health_reports_models_without_secrets(make):
    env = {"PAGEINDEX_INDEX_MODEL": "idx", "PAGEINDEX_CHAT_MODEL": "cht",
           "PAGEINDEX_CHAT_API_KEY": "secret-chat-key",
           "PAGEINDEX_API_TOKEN": "secret-token"}
    api, _ = make(env)
    res = api.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["index_model"] == "idx" and body["chat_model"] == "cht"
    assert body["auth"] is True
    assert "secret" not in res.text


def test_health_defaults_when_models_unset(make):
    api, _ = make()
    body = api.get("/health").json()
    assert body["index_model"] is None and body["chat_model"] is None
    assert body["auth"] is False


def test_auth_required_when_token_set(make):
    api, _ = make({"PAGEINDEX_API_TOKEN": "tok"})
    assert api.get("/documents").status_code == 401
    wrong = api.get("/documents", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    ok = api.get("/documents", headers={"Authorization": "Bearer tok"})
    assert ok.status_code == 200
    assert api.get("/health").status_code == 200


def test_no_auth_when_token_unset(make):
    api, _ = make()
    assert api.get("/documents").status_code == 200


class FlakyFactory:
    """client_factory that fails until ``fixed`` is set."""

    def __init__(self, error=None):
        self.error = error or ValueError("bad model config: secret-detail")
        self.fixed = False
        self.calls = 0
        self.client = FakeClient()

    def __call__(self, env):
        self.calls += 1
        if not self.fixed:
            raise self.error
        return self.client


class FakeClock:
    """Monotonic clock the tests advance by hand."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _lazy_app(tmp_path, factory, env=None, **kwargs):
    full_env = {"PAGEINDEX_STORAGE_PATH": str(tmp_path), **(env or {})}
    return TestClient(create_app(env=full_env, client_factory=factory, **kwargs))


def test_health_reports_client_ok(make):
    api, _ = make()
    res = api.get("/health")
    assert res.status_code == 200
    assert res.json()["client"] == "ok"


def test_unbuildable_client_is_503_not_client_error(tmp_path):
    factory = FlakyFactory()
    api = _lazy_app(tmp_path, factory)
    for res in (api.post("/chat", json={"question": "q"}),
                api.get("/documents"),
                api.get("/documents/pi-1"),
                api.delete("/documents/pi-1")):
        assert res.status_code == 503
        detail = res.json()["detail"]
        assert detail == ("Model client is not configured (ValueError); "
                          "check the server logs.")
        assert "secret-detail" not in res.text


def test_health_is_503_when_client_cannot_be_built(tmp_path):
    api = _lazy_app(tmp_path, FlakyFactory())
    res = api.get("/health")
    assert res.status_code == 503
    body = res.json()
    assert body["status"] == "error" and body["client"] == "error"
    assert "secret-detail" not in res.text and "ValueError" not in res.text


def test_failed_client_build_is_retried_after_the_cooldown(tmp_path):
    factory, clock = FlakyFactory(), FakeClock()
    api = _lazy_app(tmp_path, factory, client_retry_after=30.0, clock=clock)
    assert api.get("/documents").status_code == 503
    factory.fixed = True
    # Still inside the cooldown: fails fast without rebuilding.
    assert api.get("/documents").status_code == 503
    assert factory.calls == 1
    clock.advance(30.0)
    assert api.get("/documents").status_code == 200
    assert api.get("/health").json()["client"] == "ok"
    assert factory.calls == 2


def test_health_probes_do_not_rebuild_a_failed_client_within_cooldown(tmp_path):
    factory, clock = FlakyFactory(), FakeClock()
    api = _lazy_app(tmp_path, factory, client_retry_after=30.0, clock=clock)
    for _ in range(5):
        res = api.get("/health")
        assert res.status_code == 503 and res.json()["client"] == "error"
        clock.advance(5.0)
    assert factory.calls == 1
    clock.advance(10.0)  # 35 s after the failure
    assert api.get("/health").status_code == 503
    assert factory.calls == 2
    factory.fixed = True
    assert api.get("/health").status_code == 503  # new cooldown just started
    clock.advance(30.0)
    assert api.get("/health").status_code == 200
    clock.advance(1000.0)
    for _ in range(3):
        assert api.get("/health").status_code == 200
    assert factory.calls == 3


def test_client_build_traceback_is_logged_once_per_failure_streak(tmp_path, caplog):
    factory, clock = FlakyFactory(), FakeClock()
    api = _lazy_app(tmp_path, factory, client_retry_after=30.0, clock=clock)
    caplog.set_level("INFO", logger="pageindex.server")

    def build_records():
        return [r for r in caplog.records
                if r.name == "pageindex.server" and "client" in r.getMessage()]

    for _ in range(3):  # three failed builds in one streak
        api.get("/health")
        api.get("/health")  # inside the cooldown: no log at all
        clock.advance(30.0)
    records = build_records()
    assert len(records) == 3
    assert records[0].levelname == "ERROR" and records[0].exc_info
    for rec in records[1:]:
        assert rec.levelname == "WARNING" and not rec.exc_info
        assert "ValueError" in rec.getMessage()
        assert "secret-detail" not in rec.getMessage()

    factory.fixed = True
    assert api.get("/health").status_code == 200
    recovered = build_records()[-1]
    assert recovered.levelname == "INFO" and "recovered" in recovered.getMessage()

    # Once built, the client is cached: no further builds or log lines.
    caplog.clear()
    assert api.get("/health").status_code == 200
    assert build_records() == []


# ---------- documents (queued upload) ----------

def upload(api, name="report.pdf", data=PDF_BYTES, headers=None):
    return api.post("/documents", files={"file": (name, data, "application/pdf")},
                    headers=headers or {})


def test_upload_queues_and_indexes_in_background(make):
    import os

    api, fake = make()
    with api:
        res = upload(api)
        assert res.status_code == 202
        body = res.json()
        assert body["status"] == "queued" and body["name"] == "report.pdf"
        assert JOB_ID.match(body["job_id"])
        assert res.headers["location"] == f"/jobs/{body['job_id']}"
        job = wait_for_status(api, body["job_id"], "done")
    assert job["doc_id"] == "pi-new" and job["attempts"] == 1
    path, data = fake.submitted[0]
    assert os.path.basename(path) == "report.pdf"
    assert data == PDF_BYTES
    assert not os.path.exists(path)
    assert fake.metadata[0] == {"job_id": body["job_id"]}


def test_upload_returns_before_indexing_finishes(make):
    fake = FakeClient()
    fake.gate = threading.Event()
    api, _ = make(client=fake)
    with api:
        started = time.monotonic()
        first = upload(api, "a.pdf").json()
        second = upload(api, "b.pdf").json()
        assert time.monotonic() - started < 2
        wait_for_status(api, first["job_id"], "processing")
        assert api.get(f"/jobs/{second['job_id']}").json()["status"] == "queued"
        assert api.get("/health").json()["queue"] == {"queued": 1, "processing": 1}
        assert api.post("/chat", json={"question": "q"}).status_code == 200
        fake.gate.set()
        wait_for_status(api, first["job_id"], "done")
        wait_for_status(api, second["job_id"], "done")
        assert api.get("/health").json()["queue"] == {"queued": 0, "processing": 0}


@pytest.mark.parametrize("name", ["notes.md", "notes.markdown", "notes.txt", "NOTES.MD"])
def test_text_uploads_are_queued_and_indexed(make, name):
    api, fake = make(env={"PAGEINDEX_OCR": "off"})
    content = b"# Notes\n\nSome text."
    with api:
        res = api.post("/documents", files={"file": (name, content, "text/plain")})
        assert res.status_code == 202
        body = res.json()
        job = wait_for_status(api, body["job_id"], "done")
    assert body["name"] == job["name"] == name
    assert "original_name" not in body
    assert os.path.basename(fake.submitted[0][0]) == name
    assert fake.submitted[0][1] == content


@pytest.mark.parametrize("name,data", [
    ("bad.txt", b"\xff"), ("nul.md", b"hello\x00world"), ("blank.markdown", b" \n\t")])
def test_invalid_text_uploads_are_rejected(make, name, data):
    api, fake = make()
    res = api.post("/documents", files={"file": (name, data, "text/plain")})
    assert res.status_code == 400
    assert res.json()["detail"]
    assert fake.submitted == []
    assert api.get("/jobs").json()["total"] == 0


def test_upload_rejects_non_pdf(make):
    api, fake = make()
    res = api.post("/documents", files={"file": ("notes.docx", b"hi", "text/plain")})
    assert res.status_code == 415
    assert "Markdown or text" in res.json()["detail"]
    assert fake.submitted == []


def test_upload_rejects_pdf_extension_without_pdf_content(make):
    api, fake = make()
    res = upload(api, "fake.pdf", b"not a pdf")
    assert res.status_code == 400
    assert fake.submitted == []


def test_upload_rejects_oversize(make):
    api, fake = make({"PAGEINDEX_MAX_UPLOAD_MB": "1"})
    res = upload(api, "big.pdf", PDF_BYTES + b"0" * (1024 * 1024))
    assert res.status_code == 413
    assert fake.submitted == []


def _multipart(name, data, boundary="pageindex-test-boundary"):
    crlf = "\r\n"
    head = (f"--{boundary}{crlf}"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{name}\"{crlf}"
            f"Content-Type: application/pdf{crlf}{crlf}").encode()
    tail = f"{crlf}--{boundary}--{crlf}".encode()
    return head + data + tail, f"multipart/form-data; boundary={boundary}"


def _raw_post(app, body_chunks, headers):
    """Drive the ASGI app directly; reports whether the body was read."""
    state = {"reads": 0, "status": None, "body": b""}
    chunks = list(body_chunks)

    async def receive():
        state["reads"] += 1
        if chunks:
            chunk = chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(chunks)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            state["status"] = message["status"]
        elif message["type"] == "http.response.body":
            state["body"] += message.get("body", b"")

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/documents",
             "raw_path": b"/documents", "root_path": "", "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
             "client": ("test", 1), "server": ("test", 80)}
    asyncio.run(app(scope, receive, send))
    return state


def _upload_app(tmp_path, fake=None, mb="1"):
    env = {"PAGEINDEX_STORAGE_PATH": str(tmp_path), "PAGEINDEX_MAX_UPLOAD_MB": mb}
    return create_app(client=fake or FakeClient(), env=env)


def test_oversized_content_length_is_413_before_body_is_read(tmp_path):
    app = _upload_app(tmp_path)
    body, ctype = _multipart("big.pdf", PDF_BYTES)
    state = _raw_post(app, [body], {"content-type": ctype,
                                    "content-length": str(2 * 1024 * 1024)})
    assert state["status"] == 413
    assert json.loads(state["body"])["detail"] == "Upload exceeds 1 MB."
    assert state["reads"] == 0


def test_invalid_content_length_is_400(tmp_path):
    app = _upload_app(tmp_path)
    for bad in ("abc", "-5"):
        state = _raw_post(app, [b""], {"content-type": "multipart/form-data; boundary=x",
                                       "content-length": bad})
        assert state["status"] == 400
        assert state["reads"] == 0


def test_upload_within_declared_limit_reaches_the_handler(tmp_path):
    app = _upload_app(tmp_path)
    body, ctype = _multipart("ok.pdf", PDF_BYTES)
    state = _raw_post(app, [body], {"content-type": ctype,
                                    "content-length": str(len(body))})
    assert state["status"] == 202
    assert state["reads"] >= 1


def test_oversized_upload_without_content_length_is_still_413(tmp_path):
    app = _upload_app(tmp_path)
    body, ctype = _multipart("big.pdf", PDF_BYTES + b"0" * (1024 * 1024 + 10))
    size = 256 * 1024
    chunks = [body[i:i + size] for i in range(0, len(body), size)]
    state = _raw_post(app, chunks, {"content-type": ctype})
    assert state["status"] == 413


def test_rejected_uploads_create_no_job(make):
    api, _ = make()
    upload(api, "fake.pdf", b"not a pdf")
    upload(api, "notes.docx", b"hi")
    assert api.get("/jobs").json()["total"] == 0


def test_upload_storage_failure_is_503_and_leaves_no_job(make, tmp_path,
                                                        monkeypatch):
    api, fake = make()

    def fail(path, data):
        raise OSError("disk full at /secret/path")
    monkeypatch.setattr("server.jobs._write_json_atomic", fail)
    res = upload(api)
    assert res.status_code == 503
    assert res.json()["detail"] == "Could not store the upload; try again later."
    assert "/secret/path" not in res.text
    jobs_root = tmp_path / "jobs"
    assert not jobs_root.exists() or os.listdir(jobs_root) == []
    assert api.get("/health").json()["queue"] == {"queued": 0, "processing": 0}
    assert fake.submitted == []


def test_upload_strips_path_components_from_filename(make):
    import os

    api, fake = make()
    with api:
        res = upload(api, "../../etc/evil.pdf")
        assert res.status_code == 202
        assert res.json()["name"] == "evil.pdf"
        wait_for_status(api, res.json()["job_id"], "done")
    assert os.path.basename(fake.submitted[0][0]) == "evil.pdf"


def test_sdk_rejection_fails_job_with_message(make):
    fake = FakeClient()
    fake.submit_error = PageIndexAPIError("Failed to submit document: PDF has no content.")
    api, _ = make(client=fake)
    with api:
        job_id = upload(api, "a.pdf").json()["job_id"]
        job = wait_for_status(api, job_id, "failed")
    assert "no content" in job["error"]
    assert job["doc_id"] is None


def test_upstream_failure_fails_job_without_leaking(make):
    fake = FakeClient()
    cause = FakeUpstreamError("AuthenticationError: bad key sk-leaky")
    err = PageIndexAPIError(f"Failed to submit document: {cause}")
    err.__cause__ = cause
    fake.submit_error = err
    api, _ = make(client=fake)
    with api:
        job_id = upload(api, "a.pdf").json()["job_id"]
        job = wait_for_status(api, job_id, "failed")
        assert "sk-leaky" not in api.get(f"/jobs/{job_id}").text
    assert "FakeUpstreamError" in job["error"]


def test_unexpected_error_fails_job_and_worker_survives(make):
    fake = FakeClient()
    fake.submit_error = RuntimeError("boom /secret/path")
    api, _ = make(client=fake)
    with api:
        first = upload(api, "a.pdf").json()["job_id"]
        job = wait_for_status(api, first, "failed")
        assert "RuntimeError" in job["error"] and "secret" not in job["error"]
        fake.submit_error = None
        second = upload(api, "b.pdf").json()["job_id"]
        wait_for_status(api, second, "done")


# ---------- jobs ----------

def test_failed_job_can_be_retried(make):
    import os

    fake = FakeClient()
    fake.submit_error = PageIndexAPIError("Failed to submit document: temporary")
    api, _ = make(client=fake)
    with api:
        job_id = upload(api, "a.pdf").json()["job_id"]
        wait_for_status(api, job_id, "failed")
        fake.submit_error = None
        res = api.post(f"/jobs/{job_id}/retry")
        assert res.status_code == 202
        assert res.json()["status"] == "queued"
        job = wait_for_status(api, job_id, "done")
        assert job["attempts"] == 2 and job["error"] is None
        assert api.post(f"/jobs/{job_id}/retry").status_code == 409
    assert not os.path.exists(fake.submitted[-1][0])


def test_retry_rejects_queued_job(make):
    api, _ = make()
    job_id = upload(api).json()["job_id"]  # no lifespan: the job stays queued
    res = api.post(f"/jobs/{job_id}/retry")
    assert res.status_code == 409
    assert res.json()["detail"] == "Job is queued, running, or already done."


def test_retry_rejects_running_job(make):
    fake = FakeClient()
    fake.gate = threading.Event()
    api, _ = make(client=fake)
    with api:
        try:
            job_id = upload(api).json()["job_id"]
            wait_for_status(api, job_id, "processing")
            assert api.post(f"/jobs/{job_id}/retry").status_code == 409
        finally:
            fake.gate.set()
        job = wait_for_status(api, job_id, "done")
    assert job["attempts"] == 1 and len(fake.submitted) == 1


class IndexingClient(FakeClient):
    """Fake client that keeps submitted documents listable with metadata."""

    def submit_document(self, file_path, metadata=None):
        result = super().submit_document(file_path, metadata)
        self.docs[result["doc_id"]] = {"id": result["doc_id"],
                                       "name": "report.pdf", "metadata": metadata}
        return result


def test_job_whose_final_state_was_not_saved_can_be_retried(make, monkeypatch):
    import os

    from server.jobs import JobStore

    original = JobStore.update
    left = [2]  # the ``done`` write and the fallback ``failed`` write

    def update(self, job_id, expect=None, **fields):
        if fields.get("status") in ("done", "failed") and left[0] > 0:
            left[0] -= 1
            raise OSError("No space left on device")
        return original(self, job_id, expect=expect, **fields)

    monkeypatch.setattr(JobStore, "update", update)
    fake = IndexingClient()
    api, _ = make(client=fake)
    with api:
        job_id = upload(api).json()["job_id"]
        # The job stays ``processing`` with no worker; retry succeeds once
        # the worker has let go of it (409 while it still runs).
        deadline = time.monotonic() + 5
        while (res := api.post(f"/jobs/{job_id}/retry")).status_code != 202:
            assert res.status_code == 409
            assert time.monotonic() < deadline, api.get(f"/jobs/{job_id}").json()
            time.sleep(0.02)
        assert left == [0]
        assert res.json()["status"] == "queued"
        job = wait_for_status(api, job_id, "done")
    # The re-run found the document through its job_id metadata.
    assert job["doc_id"] == "pi-new" and job["attempts"] == 2
    assert len(fake.submitted) == 1
    assert not os.path.exists(fake.submitted[0][0])


def test_list_jobs_newest_first_with_filter_and_paging(make):
    api, _ = make()
    ids = [upload(api, f"{n}.pdf").json()["job_id"] for n in ("a", "b", "c")]
    listing = api.get("/jobs").json()
    assert listing["total"] == 3
    assert [job["id"] for job in listing["jobs"]] == ids[::-1]
    page = api.get("/jobs", params={"limit": 1, "offset": 1}).json()
    assert [job["id"] for job in page["jobs"]] == [ids[1]]
    assert api.get("/jobs", params={"status": "queued"}).json()["total"] == 3
    assert api.get("/jobs", params={"status": "done"}).json()["total"] == 0
    assert api.get("/jobs", params={"status": "bogus"}).status_code == 422


def test_unknown_or_invalid_job_id_is_404(make):
    api, _ = make()
    assert api.get("/jobs/job-" + "0" * 32).status_code == 404
    assert api.get("/jobs/..%2F..%2Fetc").status_code == 404
    assert api.get("/jobs/not-a-job").status_code == 404
    assert api.post("/jobs/job-" + "0" * 32 + "/retry").status_code == 404


def _seed_job(tmp_path, name="report.pdf", status=None):
    from server.jobs import JobStore

    store = JobStore(str(tmp_path / "jobs"))
    src = tmp_path / ("upload-" + name)
    src.write_bytes(PDF_BYTES)
    job = store.create(name, str(src))
    if status:
        job = store.update(job["id"], status=status, attempts=1)
    return job


def test_interrupted_job_resumes_on_new_app(make, tmp_path):
    job = _seed_job(tmp_path, status="processing")
    api, fake = make()
    with api:
        done = wait_for_status(api, job["id"], "done")
    assert done["doc_id"] == "pi-new" and done["attempts"] == 2
    assert len(fake.submitted) == 1


def test_interrupted_job_already_indexed_is_not_indexed_twice(make, tmp_path):
    job = _seed_job(tmp_path, status="processing")
    fake = FakeClient()
    fake.docs["pi-done"] = {"id": "pi-done", "name": "report.pdf",
                            "metadata": {"job_id": job["id"]}}
    api, _ = make(client=fake)
    with api:
        done = wait_for_status(api, job["id"], "done")
    assert done["doc_id"] == "pi-done"
    assert fake.submitted == []


def test_queued_jobs_resume_in_creation_order(make, tmp_path):
    import os

    created = [_seed_job(tmp_path, name)["id"] for name in ("first.pdf", "second.pdf")]
    api, fake = make()
    with api:
        for job_id in created:
            wait_for_status(api, job_id, "done")
    assert [os.path.basename(p) for p, _ in fake.submitted] == ["first.pdf", "second.pdf"]


def test_index_workers_setting_is_validated(tmp_path):
    env = {"PAGEINDEX_STORAGE_PATH": str(tmp_path)}
    for bad in ("0", "-1", "two"):
        with pytest.raises(ValueError):
            create_app(client=FakeClient(), env={**env, "PAGEINDEX_INDEX_WORKERS": bad})
    create_app(client=FakeClient(), env={**env, "PAGEINDEX_INDEX_WORKERS": "3"})


@pytest.mark.parametrize("bad", ["0", "-1", "inf", "-inf", "nan", "big", "1e400",
                                 # positive, but below one byte once converted
                                 "1e-9", "5e-7", "1e-300"])
def test_max_upload_setting_is_validated(tmp_path, bad):
    env = {"PAGEINDEX_STORAGE_PATH": str(tmp_path), "PAGEINDEX_MAX_UPLOAD_MB": bad}
    with pytest.raises(ValueError, match="PAGEINDEX_MAX_UPLOAD_MB.*> 0"):
        create_app(client=FakeClient(), env=env)


def test_max_upload_accepts_exactly_one_byte(tmp_path):
    env = {"PAGEINDEX_STORAGE_PATH": str(tmp_path),
           "PAGEINDEX_MAX_UPLOAD_MB": repr(1 / (1024 * 1024))}
    create_app(client=FakeClient(), env=env)


def test_max_upload_accepts_fractional_megabytes(make):
    api, fake = make({"PAGEINDEX_MAX_UPLOAD_MB": "0.5"})
    res = upload(api, "big.pdf", PDF_BYTES + b"0" * (600 * 1024))
    assert res.status_code == 413
    assert res.json()["detail"] == "Upload exceeds 0.5 MB."
    assert upload(api).status_code == 202


def test_auth_enforced_on_mutating_and_job_routes(make):
    api, _ = make({"PAGEINDEX_API_TOKEN": "tok"})
    job = "/jobs/job-" + "0" * 32
    assert upload(api).status_code == 401
    assert api.delete("/documents/pi-1").status_code == 401
    assert api.get("/jobs").status_code == 401
    assert api.get(job).status_code == 401
    assert api.post(job + "/retry").status_code == 401
    ok = {"Authorization": "Bearer tok"}
    assert upload(api, headers=ok).status_code == 202
    assert api.get("/jobs", headers=ok).status_code == 200


PROTECTED_ROUTES = [
    ("post", "/documents"),
    ("get", "/documents"),
    ("get", "/documents/pi-1"),
    ("delete", "/documents/pi-1"),
    ("get", "/jobs"),
    ("get", "/jobs/job-" + "0" * 32),
    ("post", "/jobs/job-" + "0" * 32 + "/retry"),
    ("post", "/chat"),
]


def _call(api, method, path, headers):
    if path == "/documents" and method == "post":
        return upload(api, headers=headers)
    if path == "/chat":
        return api.post(path, json={"question": "q"}, headers=headers)
    return api.request(method.upper(), path, headers=headers)


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_every_non_health_route_requires_the_token(make, method, path):
    api, _ = make({"PAGEINDEX_API_TOKEN": "tok"})
    assert _call(api, method, path, {}).status_code == 401
    assert _call(api, method, path, {"Authorization": "Bearer nope"}).status_code == 401
    assert _call(api, method, path, {"Authorization": "Bearer tok"}).status_code != 401


def test_lazy_client_is_built_once_across_concurrent_first_requests(tmp_path):
    calls = []

    def factory(env):
        calls.append(env["PAGEINDEX_STORAGE_PATH"])
        time.sleep(0.2)  # keep the build in flight while the others arrive
        return FakeClient()

    app = create_app(env={"PAGEINDEX_STORAGE_PATH": str(tmp_path)},
                     client_factory=factory)
    api = TestClient(app)
    results = []

    def hit():
        results.append(api.get("/documents").status_code)

    threads = [threading.Thread(target=hit) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert results == [200] * 5
    assert calls == [str(tmp_path)]
    assert api.get("/documents").status_code == 200 and len(calls) == 1


def test_list_and_get_documents(make):
    api, _ = make()
    listing = api.get("/documents", params={"limit": 5}).json()
    assert listing["total"] == 1 and listing["limit"] == 5
    assert api.get("/documents/pi-1").json()["name"] == "a.pdf"
    assert api.get("/documents/missing").status_code == 404


def test_delete_document(make):
    api, fake = make()
    assert api.delete("/documents/pi-1").status_code == 200
    assert "pi-1" not in fake.docs
    assert api.delete("/documents/pi-1").status_code == 404


# ---------- chat ----------

def test_chat_happy_path_single_and_multi_doc(make):
    api, fake = make()
    res = api.post("/chat", json={"question": "what?", "doc_id": "pi-1"})
    assert res.status_code == 200
    assert res.json() == {"answer": "answer to what?"}
    api.post("/chat", json={"question": "all?", "doc_id": ["pi-1", "pi-2"]})
    api.post("/chat", json={"question": "library?"})
    assert fake.chats == [("what?", "pi-1"), ("all?", ["pi-1", "pi-2"]),
                          ("library?", None)]


def test_chat_optional_sources_shape_and_single_stream(make):
    api, fake = make()
    fake.chat_stream = FakeStream([
        {"type": "answer", "delta": 'Term <cite doc="a.pdf" page="3"/>'},
        {"type": "tool_call", "name": "get_page_content", "arguments": {"doc_name": "a.pdf", "pages": "2-3"}},
    ])
    fake.citation_entries = [{"document": "a.pdf", "doc_id": "pi-1", "page": 3}]
    fake.tree = [{"title": "Payment terms", "start_index": 2, "end_index": 3}]
    res = api.post("/chat", json={"question": "q", "citations": True, "sources_read": True})
    assert res.status_code == 200
    assert res.json() == {"answer": "Term [1]", "citations": [
        {"index": 1, "document": "a.pdf", "doc_id": "pi-1", "page": 3, "section": "Payment terms"}],
        "sources_read": [{"document": "a.pdf", "doc_id": "pi-1", "pages": [2, 3]}]}
    assert fake.chat_calls == [("q", None, {"citations": True, "stream": True, "show_process": False})]
    assert fake.chat_stream.closed


def test_chat_flags_false_preserves_chat_signature(make):
    api, fake = make()
    assert api.post("/chat", json={"question": "q", "citations": False,
                                    "sources_read": False}).json() == {"answer": "answer to q"}
    assert fake.chat_calls == [("q", None, {})]


def test_chat_citations_only_omits_sources_read(make):
    api, fake = make()
    fake.chat_stream = FakeStream([{"type": "answer", "delta": 'A <cite doc="a.pdf" page="1"/>'}])
    fake.citation_entries = [{"document": "a.pdf", "doc_id": "pi-1", "page": 1}]
    response = api.post("/chat", json={"question": "q", "citations": True})
    assert response.json() == {"answer": "A [1]", "citations": [
        {"index": 1, "document": "a.pdf", "doc_id": "pi-1", "page": 1, "section": None}]}
    assert "sources_read" not in response.json()
    assert fake.chat_calls == [("q", None, {"citations": True, "stream": True, "show_process": False})]


def test_chat_sources_read_only_keeps_answer_and_omits_citations(make):
    api, fake = make()
    fake.chat_stream = FakeStream([{"type": "answer", "delta": 'Literal <cite doc="a.pdf" page="1"/>'},
                                   {"type": "tool_call", "name": "get_page_content",
                                    "arguments": {"doc_name": "a.pdf", "pages": "2"}}])
    response = api.post("/chat", json={"question": "q", "sources_read": True})
    assert response.json() == {"answer": 'Literal <cite doc="a.pdf" page="1"/>',
                               "sources_read": [{"document": "a.pdf", "doc_id": "pi-1", "pages": [2]}]}
    assert "citations" not in response.json()
    assert fake.chat_calls == [("q", None, {"citations": False, "stream": True, "show_process": False})]


def test_chat_unresolved_citation_preserves_nulls(make):
    api, fake = make()
    fake.chat_stream = FakeStream([{"type": "answer", "delta": "Answer"}])
    fake.citation_entries = [{"document": "missing", "doc_id": None, "page": 9}]
    response = api.post("/chat", json={"question": "q", "citations": True})
    assert response.json()["citations"] == [{"index": 1, "document": "missing",
        "doc_id": None, "page": 9, "section": None}]


def test_chat_stream_closes_after_success_and_event_error_matches_legacy(make):
    api, fake = make()
    fake.chat_stream = FakeStream([{"type": "answer", "delta": "ok"}])
    assert api.post("/chat", json={"question": "q", "citations": True}).status_code == 200
    assert fake.chat_stream.closed
    fake.chat_stream = FakeStream([], events_error=PageIndexAPIError("Documents not found or access denied: pi-x"))
    stream_error = api.post("/chat", json={"question": "q", "sources_read": True})
    fake.chat_error = PageIndexAPIError("Documents not found or access denied: pi-x")
    legacy_error = api.post("/chat", json={"question": "q"})
    assert fake.chat_stream.closed
    assert (stream_error.status_code, stream_error.json()["detail"]) == (legacy_error.status_code, legacy_error.json()["detail"])


def test_chat_get_citations_error_uses_legacy_mapping_and_closes(make):
    api, fake = make()
    fake.chat_stream = FakeStream([{"type": "answer", "delta": "ok"}])
    fake.citation_error = PageIndexAPIError("Failed to get citations: unavailable")
    response = api.post("/chat", json={"question": "q", "citations": True})
    assert (response.status_code, response.json()["detail"]) == (400, "Failed to get citations: unavailable")
    assert fake.chat_stream.closed


def test_chat_raw_tree_error_returns_null_section(make):
    api, fake = make()
    fake.chat_stream = FakeStream([{"type": "answer", "delta": 'A <cite doc="a.pdf" page="1"/>'}])
    fake.citation_entries = [{"document": "a.pdf", "doc_id": "pi-1", "page": 1}]
    fake.tree_error = RuntimeError("tree unavailable")
    response = api.post("/chat", json={"question": "q", "citations": True})
    assert response.status_code == 200
    assert response.json()["citations"][0]["section"] is None


def test_chat_duplicate_document_names_leave_source_doc_id_null(make):
    api, fake = make()
    fake.docs["pi-2"] = {"id": "pi-2", "name": "a.pdf", "status": "completed"}
    fake.chat_stream = FakeStream([{"type": "tool_call", "name": "get_page_content",
                                   "arguments": {"doc_name": "a.pdf", "pages": "1"}}])
    response = api.post("/chat", json={"question": "q", "sources_read": True})
    assert response.json()["sources_read"] == [{"document": "a.pdf", "doc_id": None, "pages": [1]}]


def test_chat_rejects_empty_question(make):
    api, fake = make()
    assert api.post("/chat", json={"question": ""}).status_code == 422
    assert fake.chats == []


def test_chat_unknown_doc_is_404(make):
    fake = FakeClient()
    fake.chat_error = PageIndexAPIError("Documents not found or access denied: pi-x")
    api, _ = make(client=fake)
    assert api.post("/chat", json={"question": "q", "doc_id": "pi-x"}).status_code == 404


def test_chat_upstream_failure_is_502_without_leaking(make):
    fake = FakeClient()
    fake.chat_error = FakeUpstreamError("401 invalid api key sk-leaky")
    api, _ = make(client=fake)
    res = api.post("/chat", json={"question": "q"})
    assert res.status_code == 502
    assert "sk-leaky" not in res.text
    assert "FakeUpstreamError" in res.json()["detail"]


def test_model_not_found_502_hints_the_openai_prefix(make):
    class NotFoundError(Exception):
        pass

    NotFoundError.__module__ = "litellm.exceptions"
    fake = FakeClient()
    fake.chat_error = NotFoundError(
        "'acme_p1/secret-model' routes through LiteLLM, but 'acme_p1' is "
        "not a LiteLLM provider. key sk-leaky")
    api, _ = make(client=fake)
    res = api.post("/chat", json={"question": "q"})
    assert res.status_code == 502
    detail = res.json()["detail"]
    assert "NotFoundError" in detail and "openai/" in detail
    # The provider's own text is never echoed, only the fixed hint.
    assert "sk-leaky" not in res.text and "acme_p1" not in res.text


def test_other_upstream_errors_carry_no_model_hint(make):
    fake = FakeClient()
    fake.chat_error = FakeUpstreamError("boom")
    api, _ = make(client=fake)
    assert "openai/" not in api.post("/chat", json={"question": "q"}).json()["detail"]


def test_chat_retries_exhausted_is_502(make):
    from pageindex.utils import LLMRetriesExhausted

    fake = FakeClient()
    fake.chat_error = LLMRetriesExhausted("gave up", status_code=503)
    api, _ = make(client=fake)
    assert api.post("/chat", json={"question": "q"}).status_code == 502


# ---------- images and OCR ----------

def _png_bytes(fmt="PNG"):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buf, fmt)
    return buf.getvalue()


@pytest.mark.parametrize("name,fmt", [("scan.png", "PNG"), ("photo.JPG", "JPEG"),
                                      ("photo.jpeg", "JPEG")])
def test_upload_accepts_images(make, name, fmt):
    api, fake = make()
    data = _png_bytes(fmt)
    ext = ".png" if name.lower().endswith(".png") else ".jpg"
    with api:
        res = upload(api, name, data)
        assert res.status_code == 202
        body = res.json()
        # Stored under a fresh UUID name; the upload name stays visible.
        assert re.fullmatch(r"[0-9a-f]{32}" + re.escape(ext), body["name"])
        assert body["original_name"] == name
        job = wait_for_status(api, body["job_id"], "done")
    assert job["name"] == body["name"] and job["original_name"] == name
    path, submitted = fake.submitted[0]
    assert os.path.basename(path) == body["name"]
    assert submitted == data
    assert fake.metadata[0] == {"job_id": body["job_id"], "original_name": name}


class UniqueNameClient(FakeClient):
    """Refuses a document whose file name is already indexed, like a store
    that keys documents by name."""

    def submit_document(self, file_path, metadata=None):
        name = os.path.basename(file_path)
        if any(os.path.basename(path) == name for path, _ in self.submitted):
            raise PageIndexAPIError(f"Failed to submit document: {name} exists.")
        return super().submit_document(file_path, metadata)


def test_repeated_image_uploads_never_share_a_stored_name(make):
    api, fake = make(client=UniqueNameClient())
    with api:
        bodies = [upload(api, "scan.png", _png_bytes()).json() for _ in range(2)]
        jobs = [wait_for_status(api, body["job_id"], "done") for body in bodies]
    names = [body["name"] for body in bodies]
    assert names[0] != names[1]
    assert [job["original_name"] for job in jobs] == ["scan.png", "scan.png"]
    stored = [os.path.basename(path) for path, _ in fake.submitted]
    assert sorted(stored) == sorted(names)
    assert [meta["original_name"] for meta in fake.metadata] == ["scan.png"] * 2


def test_pdf_uploads_keep_their_name(make):
    api, fake = make()
    with api:
        body = upload(api, "My Report.pdf").json()
        job = wait_for_status(api, body["job_id"], "done")
    assert body["name"] == job["name"] == "My Report.pdf"
    assert "original_name" not in body and "original_name" not in job
    assert os.path.basename(fake.submitted[0][0]) == "My Report.pdf"
    assert fake.metadata[0] == {"job_id": body["job_id"]}


def test_upload_rejects_disguised_image(make):
    api, fake = make()
    for name, data in (("scan.png", PDF_BYTES), ("scan.jpg", b"not an image")):
        res = upload(api, name, data)
        assert res.status_code == 400
        assert "not a supported image" in res.json()["detail"]
    assert fake.submitted == []
    assert api.get("/jobs").json()["total"] == 0


def test_upload_rejects_decompression_bombs_with_400(make, monkeypatch):
    from PIL import Image
    api, fake = make()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # _png_bytes: 1200 px
    res = upload(api, "huge.png", _png_bytes())
    assert res.status_code == 400
    assert "not a supported image" in res.json()["detail"]
    assert fake.submitted == []
    assert api.get("/jobs").json()["total"] == 0


def test_upload_rejects_unsupported_types(make):
    api, fake = make()
    for name in ("notes.docx", "icon.ico", "scan.svg", ".png"):
        res = upload(api, name, _png_bytes())
        assert res.status_code == 415
        assert "Markdown or text" in res.json()["detail"]
    assert fake.submitted == []


@pytest.mark.parametrize("ext,fmt", [("webp", "WEBP"), ("tif", "TIFF"),
                                     ("tiff", "TIFF"), ("bmp", "BMP"),
                                     ("gif", "GIF")])
def test_upload_accepts_only_png_and_jpeg_images(make, ext, fmt):
    api, fake = make()
    data = _png_bytes(fmt)
    res = upload(api, f"pic.{ext}", data)
    assert res.status_code == 415
    assert "(.pdf, .jpeg, .jpg, .png, .markdown, .md, .txt)" in res.json()["detail"]
    # Their content behind a .png name is not a supported image either.
    res = upload(api, "pic.png", data)
    assert res.status_code == 400
    assert "not a supported image" in res.json()["detail"]
    assert fake.submitted == []
    assert api.get("/jobs").json()["total"] == 0


def test_image_upload_keeps_the_size_limit(make):
    api, fake = make(env={"PAGEINDEX_MAX_UPLOAD_MB": repr(100 / (1024 * 1024))})
    res = api.post("/documents", files={"file": ("big.png", _png_bytes() + b"\0" * 200,
                                                 "image/png")})
    assert res.status_code == 413
    assert fake.submitted == []


def test_client_kwargs_carry_ocr_settings():
    kwargs = client_kwargs({"PAGEINDEX_OCR": " Force ",
                            "PAGEINDEX_OCR_MODEL": "openai/gpt-4o"})
    assert kwargs["ocr"] == "force"
    assert kwargs["ocr_model"] == "openai/gpt-4o"
    assert "ocr" not in client_kwargs({"PAGEINDEX_OCR": ""})


def test_ocr_setting_is_validated(tmp_path):
    env = {"PAGEINDEX_STORAGE_PATH": str(tmp_path), "PAGEINDEX_OCR": "always"}
    with pytest.raises(ValueError, match="PAGEINDEX_OCR must be one of off, auto, force"):
        create_app(client=FakeClient(), env=env)
    with pytest.raises(ValueError, match="PAGEINDEX_OCR"):
        client_kwargs({"PAGEINDEX_OCR": "always"})
    create_app(client=FakeClient(), env={**env, "PAGEINDEX_OCR": "off"})


def test_ocr_settings_reach_the_real_client(tmp_path):
    from server.app import build_client
    client = build_client({"PAGEINDEX_STORAGE_PATH": str(tmp_path),
                           "PAGEINDEX_OCR": "off",
                           "PAGEINDEX_OCR_MODEL": "openai/gpt-4o"})
    assert (client._api._ocr, client._api._ocr_model) == ("off", "openai/gpt-4o")


def test_ocr_image_rejection_fails_job_with_a_hint_and_no_provider_text(make):
    from pageindex.ocr import OCRModelError
    fake = FakeClient()
    cause = FakeUpstreamError("BadRequestError: image_url unsupported, key sk-leaky")
    ocr_error = OCRModelError(f"The OCR model 'openai/text-model' rejected a page "
                              f"image; it may not support image input. {cause}")
    ocr_error.__cause__ = cause
    err = PageIndexAPIError(f"Failed to submit document: OCR failed: {ocr_error}")
    err.__cause__ = ocr_error
    fake.submit_error = err
    api, _ = make(client=fake, env={"PAGEINDEX_OCR_MODEL": "openai/text-model"})
    with api:
        job_id = upload(api, "scan.png", _png_bytes()).json()["job_id"]
        job = wait_for_status(api, job_id, "failed")
        assert "sk-leaky" not in api.get(f"/jobs/{job_id}").text
    assert "may not support image input" in job["error"]
    assert "PAGEINDEX_OCR" in job["error"]
