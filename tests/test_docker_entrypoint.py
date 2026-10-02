"""The Dockerfile ENTRYPOINT maps PAGEINDEX_INDEX_MODEL to --index-model and
passes every other argument through unchanged. Runs the real ENTRYPOINT
string under sh with a stub `python` that echoes its arguments."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("sh") is None,
    reason="needs a POSIX sh with executable stubs",
)


def _entrypoint():
    match = re.search(r"^ENTRYPOINT\s+(\[.*\])\s*$", DOCKERFILE.read_text(), re.M)
    assert match, "Dockerfile has no exec-form ENTRYPOINT"
    return json.loads(match.group(1))


def _run(tmp_path, args, model=None):
    stub = tmp_path / "python"
    stub.write_text('#!/bin/sh\nprintf "%s\n" "$@"\n')
    stub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    if model is not None:
        env["PAGEINDEX_INDEX_MODEL"] = model
    out = subprocess.run(_entrypoint() + args, env=env, capture_output=True,
                         text=True, check=True)
    return out.stdout.splitlines()


def test_model_env_becomes_index_model_flag(tmp_path):
    argv = _run(tmp_path, ["--pdf_path", "data/a b.pdf"], model="openai/org/m x")
    assert argv == ["run_pageindex.py", "--index-model", "openai/org/m x",
                    "--pdf_path", "data/a b.pdf"]


def test_unset_or_empty_model_adds_no_flag(tmp_path):
    assert _run(tmp_path, ["--help"]) == ["run_pageindex.py", "--help"]
    assert _run(tmp_path, ["--help"], model="") == ["run_pageindex.py", "--help"]
