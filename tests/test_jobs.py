"""File-backed job store behind the HTTP upload queue (server.jobs)."""
import json
import os

import pytest

from server.jobs import JobStore

PDF_BYTES = b"%PDF-1.4\n%fake\n"


@pytest.fixture
def store(tmp_path):
    return JobStore(str(tmp_path / "jobs"))


def _upload(tmp_path, name="a.pdf"):
    src = tmp_path / ("incoming-" + name)
    src.write_bytes(PDF_BYTES)
    return str(src)


def test_create_moves_pdf_into_job_dir(store, tmp_path):
    src = _upload(tmp_path)
    job = store.create("report.pdf", src)
    assert job["status"] == "queued" and job["attempts"] == 0
    assert job["doc_id"] is None and job["error"] is None
    assert job["created_at"] == job["updated_at"]
    assert job["created_at"].endswith("Z")
    assert not os.path.exists(src)
    pdf = store.pdf_path(job["id"])
    assert os.path.basename(pdf) == "report.pdf"
    with open(pdf, "rb") as handle:
        assert handle.read() == PDF_BYTES
    with open(os.path.join(os.path.dirname(pdf), "job.json")) as handle:
        assert json.load(handle) == job


def test_update_is_atomic_and_leaves_no_temp_files(store, tmp_path):
    job = store.create("a.pdf", _upload(tmp_path))
    updated = store.update(job["id"], status="failed", error="nope")
    assert updated["status"] == "failed" and updated["error"] == "nope"
    assert store.get(job["id"]) == updated
    job_dir = os.path.dirname(store.pdf_path(job["id"]))
    assert sorted(os.listdir(job_dir)) == ["a.pdf", "job.json"]


@pytest.mark.parametrize("bad", ["..", "../x", "job-xyz", "job-" + "0" * 32 + "/..",
                                 "", "job-" + "A" * 32])
def test_invalid_ids_are_rejected(store, bad):
    assert store.get(bad) is None
    with pytest.raises(KeyError):
        store.update(bad, status="done")


def test_list_skips_foreign_and_broken_entries(store, tmp_path):
    job = store.create("a.pdf", _upload(tmp_path))
    root = os.path.dirname(os.path.dirname(store.pdf_path(job["id"])))
    os.makedirs(os.path.join(root, "not-a-job"))
    broken = os.path.join(root, "job-" + "f" * 32)
    os.makedirs(broken)
    with open(os.path.join(broken, "job.json"), "w") as handle:
        handle.write("{truncated")
    assert [j["id"] for j in store.list()] == [job["id"]]


def test_recover_requeues_interrupted_jobs_in_creation_order(store, tmp_path):
    first = store.create("a.pdf", _upload(tmp_path, "a.pdf"))
    second = store.create("b.pdf", _upload(tmp_path, "b.pdf"))
    third = store.create("c.pdf", _upload(tmp_path, "c.pdf"))
    store.update(first["id"], status="processing", attempts=1)
    store.update(third["id"], status="done", doc_id="pi-1")

    fresh = JobStore(os.path.dirname(os.path.dirname(store.pdf_path(first["id"]))))
    assert fresh.recover() == [first["id"], second["id"]]
    assert fresh.get(first["id"])["status"] == "queued"
    assert fresh.counts() == {"queued": 2, "processing": 0}


def test_missing_root_lists_nothing(tmp_path):
    store = JobStore(str(tmp_path / "absent"))
    assert store.list() == [] and store.recover() == []
    assert store.counts() == {"queued": 0, "processing": 0}
