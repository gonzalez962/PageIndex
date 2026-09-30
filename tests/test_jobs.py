"""File-backed job store behind the HTTP upload queue (server.jobs)."""
import json
import os
import threading
import time

import pytest

from server.jobs import JobRunner, JobStore

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


# ---------- JobRunner shutdown ----------

class _BlockingClient:
    """Fake SDK client whose ``submit_document`` waits for ``release``."""

    def __init__(self):
        self.release = threading.Event()
        self.submitted = []
        self._lock = threading.Lock()
        self._started = threading.Condition(self._lock)

    def submit_document(self, path, metadata=None):
        with self._lock:
            self.submitted.append(path)
            self._started.notify_all()
        assert self.release.wait(10), "test never released the client"
        return {"doc_id": "pi-" + os.path.basename(path)}

    def list_documents(self, limit=1000, offset=0):
        return {"documents": [], "total": 0}

    def wait_submitted(self, count, timeout=5.0):
        with self._lock:
            assert self._started.wait_for(lambda: len(self.submitted) >= count,
                                          timeout), self.submitted


def _runner(store, client, workers=1):
    return JobRunner(store, lambda: client, str, workers=workers)


def _wait_status(store, job_id, status, timeout=5.0):
    deadline = time.monotonic() + timeout
    while store.get(job_id)["status"] != status:
        assert time.monotonic() < deadline, store.get(job_id)
        time.sleep(0.01)


def test_stop_leaves_unclaimed_jobs_queued(store, tmp_path):
    jobs = [store.create(n, _upload(tmp_path, n)) for n in ("a.pdf", "b.pdf", "c.pdf")]
    client = _BlockingClient()
    runner = _runner(store, client)
    runner.start()
    client.wait_submitted(1)
    assert store.get(jobs[0]["id"])["status"] == "processing"

    stopper = threading.Thread(target=runner.stop)
    stopper.start()
    # stop() has run once its wake-up sentinel sits behind the two pending ids.
    deadline = time.monotonic() + 5
    while runner._queue.qsize() < 3:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    client.release.set()
    stopper.join(10)
    assert not stopper.is_alive()

    assert store.get(jobs[0]["id"])["status"] == "done"
    for job in jobs[1:]:
        assert store.get(job["id"])["status"] == "queued"
        assert store.get(job["id"])["attempts"] == 0
    assert client.submitted == [store.pdf_path(jobs[0]["id"])]


def test_stop_returns_promptly_when_workers_are_idle(store):
    runner = _runner(store, _BlockingClient(), workers=4)
    runner.start()
    started = time.monotonic()
    runner.stop(timeout=5.0)
    assert time.monotonic() - started < 1.0


def test_stop_shares_one_deadline_across_stuck_workers(store, tmp_path):
    jobs = [store.create(n, _upload(tmp_path, n)) for n in ("a.pdf", "b.pdf", "c.pdf")]
    client = _BlockingClient()
    runner = _runner(store, client, workers=3)
    runner.start()
    client.wait_submitted(3)
    timeout = 0.3
    try:
        started = time.monotonic()
        runner.stop(timeout=timeout)
        elapsed = time.monotonic() - started
    finally:
        client.release.set()
    assert elapsed < 2 * timeout
    # Jobs still inside submit_document at shutdown are left to finish.
    for job in jobs:
        _wait_status(store, job["id"], "done")


def test_start_after_stop_processes_new_jobs(store, tmp_path):
    client = _BlockingClient()
    client.release.set()
    runner = _runner(store, client)
    runner.start()
    runner.stop(timeout=1.0)
    runner.start()
    try:
        job = store.create("a.pdf", _upload(tmp_path))
        runner.enqueue(job["id"])
        _wait_status(store, job["id"], "done")
    finally:
        runner.stop(timeout=1.0)
