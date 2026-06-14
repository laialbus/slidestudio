"""
Tests for POST /resume/{job_id} — continue a failed GUI job from its checkpoint.

Uses FastAPI's TestClient. _run_pipeline_job is patched with an AsyncMock so the
real pipeline never runs; create_task still receives a valid coroutine and the
call args are recorded synchronously.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import server as srv
from server import JobState


@pytest.fixture()
def client(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "archive").mkdir()
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    with (
        patch.object(srv, "_OUTPUTS_DIR", outputs),
        patch.object(srv, "_ARCHIVE_DIR", outputs / "archive"),
        patch.object(srv, "_PDFS_DIR", pdfs),
    ):
        srv._jobs.clear()
        app = srv.create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, pdfs
    srv._jobs.clear()


def _pdf(pdfs: Path, name: str = "p.pdf") -> Path:
    path = pdfs / name
    path.write_bytes(b"%PDF-1.7 stub")
    return path


def test_unknown_job_returns_404(client):
    c, _ = client
    assert c.post("/resume/nope").status_code == 404


def test_non_failed_job_returns_409(client):
    c, pdfs = client
    srv._jobs["j1"] = JobState(status="running", pdf_path=_pdf(pdfs))
    assert c.post("/resume/j1").status_code == 409


def test_missing_pdf_returns_409(client):
    c, pdfs = client
    srv._jobs["j1"] = JobState(status="error", pdf_path=pdfs / "gone.pdf")
    assert c.post("/resume/j1").status_code == 409


def test_failed_job_reschedules_with_resume_true(client):
    c, pdfs = client
    pdf = _pdf(pdfs)
    srv._jobs["j1"] = JobState(status="error", pdf_path=pdf, error="boom")

    mock = AsyncMock()
    with patch.object(srv, "_run_pipeline_job", mock):
        r = c.post("/resume/j1")

    assert r.status_code == 200
    assert r.json() == {"job_id": "j1"}
    mock.assert_called_once_with("j1", pdf, resume=True)
    # The endpoint reset the error before rescheduling.
    assert srv._jobs["j1"].error is None
