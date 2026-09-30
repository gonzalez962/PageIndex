"""HTTP service tests: server.app wraps PageIndexClient behind FastAPI.

A fake client stands in for the SDK so no model or PDF pipeline runs.
"""
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

    def submit_document(self, file_path):
        with open(file_path, "rb") as handle:
            self.submitted.append((file_path, handle.read()))
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


def make(env=None, client=None):
    fake = client or FakeClient()
    return TestClient(create_app(client=fake, env=env or {})), fake


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

def test_health_reports_models_without_secrets():
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


def test_health_defaults_when_models_unset():
    api, _ = make()
    body = api.get("/health").json()
    assert body["index_model"] is None and body["chat_model"] is None
    assert body["auth"] is False


def test_auth_required_when_token_set():
    api, _ = make({"PAGEINDEX_API_TOKEN": "tok"})
    assert api.get("/documents").status_code == 401
    wrong = api.get("/documents", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    ok = api.get("/documents", headers={"Authorization": "Bearer tok"})
    assert ok.status_code == 200
    assert api.get("/health").status_code == 200


def test_no_auth_when_token_unset():
    api, _ = make()
    assert api.get("/documents").status_code == 200


# ---------- documents ----------

def test_upload_indexes_pdf_and_cleans_temp_file():
    import os

    api, fake = make()
    res = api.post("/documents",
                   files={"file": ("report.pdf", PDF_BYTES, "application/pdf")})
    assert res.status_code == 201
    assert res.json() == {"doc_id": "pi-new", "name": "report.pdf"}
    path, data = fake.submitted[0]
    assert os.path.basename(path) == "report.pdf"
    assert data == PDF_BYTES
    assert not os.path.exists(path)


def test_upload_rejects_non_pdf():
    api, fake = make()
    res = api.post("/documents", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert res.status_code == 415
    assert fake.submitted == []


def test_upload_rejects_pdf_extension_without_pdf_content():
    api, fake = make()
    res = api.post("/documents", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert res.status_code == 400
    assert fake.submitted == []


def test_upload_rejects_oversize():
    api, fake = make({"PAGEINDEX_MAX_UPLOAD_MB": "1"})
    big = PDF_BYTES + b"0" * (1024 * 1024)
    res = api.post("/documents", files={"file": ("big.pdf", big, "application/pdf")})
    assert res.status_code == 413
    assert fake.submitted == []


def test_upload_strips_path_components_from_filename():
    import os

    api, fake = make()
    res = api.post("/documents",
                   files={"file": ("../../etc/evil.pdf", PDF_BYTES, "application/pdf")})
    assert res.status_code == 201
    assert os.path.basename(fake.submitted[0][0]) == "evil.pdf"


def test_upload_sdk_rejection_is_400():
    fake = FakeClient()
    fake.submit_error = PageIndexAPIError("Failed to submit document: PDF has no content.")
    api, _ = make(client=fake)
    res = api.post("/documents", files={"file": ("a.pdf", PDF_BYTES, "application/pdf")})
    assert res.status_code == 400
    assert "no content" in res.json()["detail"]


def test_upload_upstream_failure_is_502():
    fake = FakeClient()
    cause = FakeUpstreamError("AuthenticationError: bad key sk-leaky")
    err = PageIndexAPIError(f"Failed to submit document: {cause}")
    err.__cause__ = cause
    fake.submit_error = err
    api, _ = make(client=fake)
    res = api.post("/documents", files={"file": ("a.pdf", PDF_BYTES, "application/pdf")})
    assert res.status_code == 502
    assert "sk-leaky" not in res.text


def test_list_and_get_documents():
    api, _ = make()
    listing = api.get("/documents", params={"limit": 5}).json()
    assert listing["total"] == 1 and listing["limit"] == 5
    assert api.get("/documents/pi-1").json()["name"] == "a.pdf"
    assert api.get("/documents/missing").status_code == 404


def test_delete_document():
    api, fake = make()
    assert api.delete("/documents/pi-1").status_code == 200
    assert "pi-1" not in fake.docs
    assert api.delete("/documents/pi-1").status_code == 404


# ---------- chat ----------

def test_chat_happy_path_single_and_multi_doc():
    api, fake = make()
    res = api.post("/chat", json={"question": "what?", "doc_id": "pi-1"})
    assert res.status_code == 200
    assert res.json() == {"answer": "answer to what?"}
    api.post("/chat", json={"question": "all?", "doc_id": ["pi-1", "pi-2"]})
    api.post("/chat", json={"question": "library?"})
    assert fake.chats == [("what?", "pi-1"), ("all?", ["pi-1", "pi-2"]),
                          ("library?", None)]


def test_chat_rejects_empty_question():
    api, fake = make()
    assert api.post("/chat", json={"question": ""}).status_code == 422
    assert fake.chats == []


def test_chat_unknown_doc_is_404():
    fake = FakeClient()
    fake.chat_error = PageIndexAPIError("Documents not found or access denied: pi-x")
    api, _ = make(client=fake)
    assert api.post("/chat", json={"question": "q", "doc_id": "pi-x"}).status_code == 404


def test_chat_upstream_failure_is_502_without_leaking():
    fake = FakeClient()
    fake.chat_error = FakeUpstreamError("401 invalid api key sk-leaky")
    api, _ = make(client=fake)
    res = api.post("/chat", json={"question": "q"})
    assert res.status_code == 502
    assert "sk-leaky" not in res.text
    assert "FakeUpstreamError" in res.json()["detail"]


def test_chat_retries_exhausted_is_502():
    from pageindex.utils import LLMRetriesExhausted

    fake = FakeClient()
    fake.chat_error = LLMRetriesExhausted("gave up", status_code=503)
    api, _ = make(client=fake)
    assert api.post("/chat", json={"question": "q"}).status_code == 502
