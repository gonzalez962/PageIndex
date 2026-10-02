"""Contract between the job runner and the real local PageIndexClient.

The runner tags each submitted document with ``metadata={"job_id": ...}``
and, when it resumes an interrupted job, finds the stored document by that
tag (``server.jobs._find_doc_for_job``) instead of indexing it twice. The
server tests use a fake client; this pins the same behavior on the real SDK
store. Only the LLM indexing pipeline (``LocalAPI._index_flash``) is stubbed,
so no network or API key is needed.
"""
import os
import shutil

import pytest

from pageindex import PageIndexClient
from pageindex.local_api import LocalAPI
from server.jobs import _find_doc_for_job

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "data", "flash", "ja_report.pdf")


@pytest.fixture
def client(tmp_path, monkeypatch):
    def fake_index(self, file_path):
        tree = [{"title": "Report", "node_id": "0000",
                 "start_index": 1, "end_index": 1, "summary": "stub"}]
        return tree, "stub description"

    monkeypatch.setattr(LocalAPI, "_index_flash", fake_index)
    return PageIndexClient(storage_path=str(tmp_path / "store"))


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "report.pdf"
    shutil.copyfile(SAMPLE_PDF, path)
    return str(path)


def test_submit_metadata_is_listed_back_and_found_by_job_id(client, pdf):
    submitted = client.submit_document(pdf, metadata={"job_id": "job-x"})
    doc_id = submitted["doc_id"]

    listing = client.list_documents()
    docs = {doc["id"]: doc for doc in listing["documents"]}
    assert docs[doc_id]["metadata"]["job_id"] == "job-x"

    assert _find_doc_for_job(client, "job-x") == doc_id
    assert _find_doc_for_job(client, "job-other") is None


def test_job_id_lookup_distinguishes_same_named_documents(client, pdf):
    first = client.submit_document(pdf, metadata={"job_id": "job-a"})["doc_id"]
    # The SDK stores a taken name as report_1.pdf and warns; the job id,
    # not the name, is what tells the two documents apart.
    with pytest.warns(UserWarning, match="report_1.pdf"):
        second = client.submit_document(pdf, metadata={"job_id": "job-b"})["doc_id"]
    assert first != second
    assert _find_doc_for_job(client, "job-a") == first
    assert _find_doc_for_job(client, "job-b") == second
