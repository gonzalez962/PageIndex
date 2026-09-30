"""HTTP service tests: server.app wraps PageIndexClient behind FastAPI.

A fake client stands in for the SDK so no model or PDF pipeline runs.
"""
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


class FakeClient:
    def __init__(self):
        self.submitted = []
        self.chats = []
        self.docs = {"pi-1": {"id": "pi-1", "name": "a.pdf", "status": "completed"}}
        self.chat_error = None
        self.submit_error = None
        self.metadata = []
        self.gate = None

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

    def chat(self, question, doc_id=None):
        self.chats.append((question, doc_id))
        if self.chat_error:
            raise self.chat_error
        return f"answer to {question}"


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


def test_upload_rejects_non_pdf(make):
    api, fake = make()
    res = api.post("/documents", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert res.status_code == 415
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


def test_rejected_uploads_create_no_job(make):
    api, _ = make()
    upload(api, "fake.pdf", b"not a pdf")
    upload(api, "notes.txt", b"hi")
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
