"""
Tests for POST /archive/{slug}, POST /unarchive/{slug}, DELETE /archive/{slug}.

Uses FastAPI's TestClient and patches the module-level directory constants so
the real outputs/ dir is never touched.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import server as srv


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _write_single_deck(path: Path, title: str = "Paper") -> None:
    data = {
        "title": title, "type": "single_deck",
        "generated_at": "2026-05-01T10:00:00+00:00",
        "provider": "anthropic", "model": "claude-sonnet-4-6",
        "slides": [{"index": 1, "heading": "H", "body": "B.", "tag": "Key Concept",
                    "latex": None, "image_refs": []}],
        "images": [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_multi_deck(dir_path: Path, title: str = "Textbook") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    chapter = {
        "title": "Ch1", "type": "single_deck",
        "generated_at": "2026-05-01T10:00:00+00:00",
        "provider": "anthropic", "model": "claude-sonnet-4-6",
        "slides": [{"index": 1, "heading": "H", "body": "B.", "tag": "Key Concept",
                    "latex": None, "image_refs": []}],
        "images": [],
    }
    (dir_path / "01_chapter.json").write_text(json.dumps(chapter), encoding="utf-8")
    index = {
        "title": title, "type": "multi_deck",
        "generated_at": "2026-05-01T10:00:00+00:00",
        "provider": "anthropic", "model": "claude-sonnet-4-6",
        "decks": [{"chapter_title": "Ch1", "file": "01_chapter.json"}],
    }
    (dir_path / "index.json").write_text(json.dumps(index), encoding="utf-8")


@pytest.fixture()
def dirs(tmp_path):
    outputs = tmp_path / "outputs"
    archive = outputs / "archive"
    outputs.mkdir()
    archive.mkdir()
    return outputs, archive


@pytest.fixture()
def client(dirs, tmp_path):
    outputs, archive = dirs
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    html_dir = Path(srv.__file__).parent / "exporters" / "html"
    with (
        patch.object(srv, "_OUTPUTS_DIR", outputs),
        patch.object(srv, "_ARCHIVE_DIR", archive),
        patch.object(srv, "_PDFS_DIR", pdfs),
    ):
        app = srv.create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, outputs, archive


# ──────────────────────────────────────────────────────────────
# POST /archive/{slug}
# ──────────────────────────────────────────────────────────────

class TestArchiveEndpoint:
    def test_archives_single_deck(self, client):
        c, outputs, archive = client
        _write_single_deck(outputs / "paper.json")
        r = c.post("/archive/paper")
        assert r.status_code == 200
        assert not (outputs / "paper.json").exists()
        assert (archive / "paper.json").exists()

    def test_archives_multi_deck_directory(self, client):
        c, outputs, archive = client
        _write_multi_deck(outputs / "textbook")
        r = c.post("/archive/textbook")
        assert r.status_code == 200
        assert not (outputs / "textbook").exists()
        assert (archive / "textbook").is_dir()
        assert (archive / "textbook" / "index.json").exists()

    def test_archive_rebuilds_manifest(self, client):
        c, outputs, archive = client
        _write_single_deck(outputs / "paper.json")
        c.post("/archive/paper")
        manifest = json.loads((outputs / "library.json").read_text())
        entry = next((e for e in manifest if "paper" in e["file"]), None)
        assert entry is not None
        assert entry["archived"] is True

    def test_archive_missing_deck_returns_404(self, client):
        c, outputs, archive = client
        r = c.post("/archive/nonexistent")
        assert r.status_code == 404

    def test_archive_response_body(self, client):
        c, outputs, archive = client
        _write_single_deck(outputs / "paper.json")
        r = c.post("/archive/paper")
        assert r.json() == {"status": "archived"}

    def test_archive_collision_returns_409(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json", "Archived Version")
        _write_single_deck(outputs / "paper.json", "Library Version")
        r = c.post("/archive/paper")
        assert r.status_code == 409

    def test_archive_collision_leaves_both_intact(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json", "Archived Version")
        _write_single_deck(outputs / "paper.json", "Library Version")
        c.post("/archive/paper")
        archived = json.loads((archive / "paper.json").read_text())
        active = json.loads((outputs / "paper.json").read_text())
        assert archived["title"] == "Archived Version"
        assert active["title"] == "Library Version"

    def test_archive_multi_deck_collision_returns_409(self, client):
        c, outputs, archive = client
        _write_multi_deck(archive / "textbook")
        _write_multi_deck(outputs / "textbook")
        r = c.post("/archive/textbook")
        assert r.status_code == 409
        # without the guard, shutil.move nests the dir inside the existing one
        assert not (archive / "textbook" / "textbook").exists()


# ──────────────────────────────────────────────────────────────
# Cache headers — slug paths are reused across runs, so the
# browser must revalidate instead of trusting heuristic freshness
# ──────────────────────────────────────────────────────────────

class TestOutputCacheHeaders:
    def test_outputs_responses_require_revalidation(self, client):
        c, outputs, archive = client
        _write_single_deck(outputs / "paper.json")
        r = c.get("/outputs/paper.json")
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-cache"

    def test_library_response_requires_revalidation(self, client):
        c, outputs, archive = client
        r = c.get("/library")
        assert r.headers["cache-control"] == "no-cache"


# ──────────────────────────────────────────────────────────────
# POST /unarchive/{slug}
# ──────────────────────────────────────────────────────────────

class TestUnarchiveEndpoint:
    def test_restores_single_deck(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json")
        r = c.post("/unarchive/paper")
        assert r.status_code == 200
        assert (outputs / "paper.json").exists()
        assert not (archive / "paper.json").exists()

    def test_restores_multi_deck_directory(self, client):
        c, outputs, archive = client
        _write_multi_deck(archive / "textbook")
        r = c.post("/unarchive/textbook")
        assert r.status_code == 200
        assert (outputs / "textbook").is_dir()
        assert not (archive / "textbook").exists()

    def test_unarchive_rebuilds_manifest(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json")
        c.post("/unarchive/paper")
        manifest = json.loads((outputs / "library.json").read_text())
        entry = next((e for e in manifest if "paper" in e["file"]), None)
        assert entry is not None
        assert entry["archived"] is False

    def test_unarchive_missing_returns_404(self, client):
        c, outputs, archive = client
        r = c.post("/unarchive/nonexistent")
        assert r.status_code == 404

    def test_unarchive_collision_returns_409(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json", "Archived Version")
        _write_single_deck(outputs / "paper.json", "Library Version")
        r = c.post("/unarchive/paper")
        assert r.status_code == 409

    def test_unarchive_collision_leaves_both_intact(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json", "Archived")
        _write_single_deck(outputs / "paper.json", "Library")
        c.post("/unarchive/paper")
        assert (archive / "paper.json").exists()
        assert (outputs / "paper.json").exists()

    def test_unarchive_response_body(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json")
        r = c.post("/unarchive/paper")
        assert r.json() == {"status": "restored"}


# ──────────────────────────────────────────────────────────────
# DELETE /archive/{slug}
# ──────────────────────────────────────────────────────────────

class TestDeleteArchiveEndpoint:
    def test_deletes_archived_single_deck(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json")
        r = c.delete("/archive/paper")
        assert r.status_code == 200
        assert not (archive / "paper.json").exists()

    def test_deletes_archived_multi_deck_directory(self, client):
        c, outputs, archive = client
        _write_multi_deck(archive / "textbook")
        r = c.delete("/archive/textbook")
        assert r.status_code == 200
        assert not (archive / "textbook").exists()

    def test_delete_rebuilds_manifest(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json")
        c.delete("/archive/paper")
        manifest = json.loads((outputs / "library.json").read_text())
        assert not any("paper" in e.get("file", "") for e in manifest)

    def test_delete_missing_returns_404(self, client):
        c, outputs, archive = client
        r = c.delete("/archive/nonexistent")
        assert r.status_code == 404

    def test_delete_response_body(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json")
        r = c.delete("/archive/paper")
        assert r.json() == {"status": "deleted"}

    def test_delete_does_not_touch_library(self, client):
        c, outputs, archive = client
        _write_single_deck(archive / "paper.json", "Archived")
        _write_single_deck(outputs / "paper2.json", "Active")
        c.delete("/archive/paper")
        assert (outputs / "paper2.json").exists()
