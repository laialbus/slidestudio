"""
Tests for utils/library.py — upsert_library_manifest and rebuild_library_manifest.

All file I/O uses the pytest tmp_path fixture; the real outputs/ dir is never touched.
"""

import json
from pathlib import Path

import pytest

from utils.library import rebuild_library_manifest, upsert_library_manifest


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _entry(
    title="Paper A",
    file_key="outputs/paper_a.json",
    generated_at="2026-05-01T10:00:00+00:00",
) -> dict:
    return {
        "title":        title,
        "file":         file_key,
        "type":         "single_deck",
        "generated_at": generated_at,
        "provider":     "anthropic",
        "model":        "claude-sonnet-4-6",
        "slide_count":  10,
        "deck_count":   1,
    }


def _write_single_deck(path: Path, title: str, generated_at: str = "2026-05-01T10:00:00+00:00") -> None:
    data = {
        "title":        title,
        "type":         "single_deck",
        "generated_at": generated_at,
        "provider":     "anthropic",
        "model":        "claude-sonnet-4-6",
        "slides": [
            {"index": 1, "heading": "H", "body": "B.", "tag": "Key Concept",
             "latex": None, "image_refs": []}
        ],
        "images": [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_multi_deck(
    dir_path: Path,
    title: str,
    chapter_count: int = 2,
    generated_at: str = "2026-04-01T10:00:00+00:00",
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    decks = []
    for i in range(1, chapter_count + 1):
        fname = f"{i:02d}_chapter_{i}.json"
        chapter = {
            "title":        f"Chapter {i}",
            "type":         "single_deck",
            "generated_at": generated_at,
            "provider":     "anthropic",
            "model":        "claude-sonnet-4-6",
            "slides": [
                {"index": 1, "heading": "H", "body": "B.", "tag": "Key Concept",
                 "latex": None, "image_refs": []}
            ],
            "images": [],
        }
        (dir_path / fname).write_text(json.dumps(chapter), encoding="utf-8")
        decks.append({"chapter_title": f"Chapter {i}", "file": fname})
    index = {
        "title":        title,
        "type":         "multi_deck",
        "generated_at": generated_at,
        "provider":     "anthropic",
        "model":        "claude-sonnet-4-6",
        "decks":        decks,
    }
    (dir_path / "index.json").write_text(json.dumps(index), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# upsert_library_manifest
# ──────────────────────────────────────────────────────────────

class TestUpsertLibraryManifest:
    def test_creates_manifest_on_first_call(self, tmp_path):
        upsert_library_manifest(tmp_path, _entry())
        assert (tmp_path / "library.json").exists()

    def test_first_entry_appears_in_manifest(self, tmp_path):
        upsert_library_manifest(tmp_path, _entry(title="My Paper"))
        data = json.loads((tmp_path / "library.json").read_text())
        assert data[0]["title"] == "My Paper"

    def test_upsert_deduplicates_by_file(self, tmp_path):
        upsert_library_manifest(tmp_path, _entry(title="Version 1"))
        upsert_library_manifest(tmp_path, _entry(title="Version 2"))  # same file key
        data = json.loads((tmp_path / "library.json").read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Version 2"

    def test_upsert_appends_different_files(self, tmp_path):
        upsert_library_manifest(tmp_path, _entry(file_key="outputs/a.json"))
        upsert_library_manifest(tmp_path, _entry(file_key="outputs/b.json"))
        data = json.loads((tmp_path / "library.json").read_text())
        assert len(data) == 2

    def test_entries_sorted_newest_first(self, tmp_path):
        upsert_library_manifest(tmp_path, _entry(file_key="outputs/a.json", generated_at="2026-01-01T00:00:00+00:00"))
        upsert_library_manifest(tmp_path, _entry(file_key="outputs/b.json", generated_at="2026-05-01T00:00:00+00:00"))
        upsert_library_manifest(tmp_path, _entry(file_key="outputs/c.json", generated_at="2026-03-01T00:00:00+00:00"))
        data = json.loads((tmp_path / "library.json").read_text())
        timestamps = [e["generated_at"] for e in data]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_atomic_no_tmp_leftover(self, tmp_path):
        upsert_library_manifest(tmp_path, _entry())
        assert not list(tmp_path.glob("*.tmp"))

    def test_tolerates_missing_manifest(self, tmp_path):
        # no library.json exists yet — must not raise
        upsert_library_manifest(tmp_path, _entry())
        assert (tmp_path / "library.json").exists()

    def test_tolerates_corrupt_manifest(self, tmp_path):
        (tmp_path / "library.json").write_text("not valid json", encoding="utf-8")
        upsert_library_manifest(tmp_path, _entry())
        data = json.loads((tmp_path / "library.json").read_text())
        assert len(data) == 1

    def test_rerun_updates_slide_count(self, tmp_path):
        e1 = {**_entry(), "slide_count": 5}
        upsert_library_manifest(tmp_path, e1)
        e2 = {**_entry(), "slide_count": 12}
        upsert_library_manifest(tmp_path, e2)
        data = json.loads((tmp_path / "library.json").read_text())
        assert data[0]["slide_count"] == 12


# ──────────────────────────────────────────────────────────────
# rebuild_library_manifest
# ──────────────────────────────────────────────────────────────

class TestRebuildLibraryManifest:
    def test_finds_single_deck(self, tmp_path):
        _write_single_deck(tmp_path / "paper.json", "My Paper")
        entries = rebuild_library_manifest(tmp_path)
        assert any(e["title"] == "My Paper" for e in entries)

    def test_finds_multi_deck(self, tmp_path):
        _write_multi_deck(tmp_path / "textbook", "My Textbook")
        entries = rebuild_library_manifest(tmp_path)
        assert any(e["title"] == "My Textbook" for e in entries)

    def test_skips_library_json(self, tmp_path):
        (tmp_path / "library.json").write_text('[{"title":"ghost"}]', encoding="utf-8")
        entries = rebuild_library_manifest(tmp_path)
        assert not any(e.get("title") == "ghost" for e in entries)

    def test_skips_chapter_files(self, tmp_path):
        _write_multi_deck(tmp_path / "textbook", "Textbook", chapter_count=2)
        entries = rebuild_library_manifest(tmp_path)
        # Only the index should appear, not the two chapter files
        assert len(entries) == 1
        assert entries[0]["title"] == "Textbook"

    def test_skips_unknown_json(self, tmp_path):
        (tmp_path / "unknown.json").write_text('{"foo": "bar"}', encoding="utf-8")
        entries = rebuild_library_manifest(tmp_path)
        assert len(entries) == 0

    def test_multi_deck_slide_count_sums_chapters(self, tmp_path):
        _write_multi_deck(tmp_path / "textbook", "Textbook", chapter_count=3)
        entries = rebuild_library_manifest(tmp_path)
        tb = next(e for e in entries if e["title"] == "Textbook")
        # Each chapter has exactly 1 slide → total = 3
        assert tb["slide_count"] == 3

    def test_multi_deck_deck_count(self, tmp_path):
        _write_multi_deck(tmp_path / "textbook", "Textbook", chapter_count=4)
        entries = rebuild_library_manifest(tmp_path)
        tb = next(e for e in entries if e["title"] == "Textbook")
        assert tb["deck_count"] == 4

    def test_single_deck_slide_count(self, tmp_path):
        _write_single_deck(tmp_path / "paper.json", "Paper")
        entries = rebuild_library_manifest(tmp_path)
        assert entries[0]["slide_count"] == 1

    def test_sorted_newest_first(self, tmp_path):
        _write_single_deck(tmp_path / "old.json", "Old",  "2026-01-01T00:00:00+00:00")
        _write_single_deck(tmp_path / "new.json", "New",  "2026-05-01T00:00:00+00:00")
        _write_single_deck(tmp_path / "mid.json", "Mid",  "2026-03-01T00:00:00+00:00")
        entries = rebuild_library_manifest(tmp_path)
        assert entries[0]["title"] == "New"
        assert entries[-1]["title"] == "Old"

    def test_empty_outputs_produces_empty_manifest(self, tmp_path):
        entries = rebuild_library_manifest(tmp_path)
        assert entries == []
        assert (tmp_path / "library.json").exists()

    def test_manifest_written_to_disk(self, tmp_path):
        _write_single_deck(tmp_path / "paper.json", "Paper")
        rebuild_library_manifest(tmp_path)
        data = json.loads((tmp_path / "library.json").read_text())
        assert len(data) == 1

    def test_manifest_includes_provider_and_model(self, tmp_path):
        _write_single_deck(tmp_path / "paper.json", "Paper")
        entries = rebuild_library_manifest(tmp_path)
        assert entries[0]["provider"] == "anthropic"
        assert entries[0]["model"] == "claude-sonnet-4-6"

    def test_debug_subdir_skipped(self, tmp_path):
        debug_dir = tmp_path / "debug" / "paper"
        debug_dir.mkdir(parents=True)
        # Write a file that looks like a deck inside debug/
        _write_single_deck(debug_dir / "paper.json", "Debug Ghost")
        entries = rebuild_library_manifest(tmp_path)
        assert not any(e["title"] == "Debug Ghost" for e in entries)

    def test_active_entry_has_archived_false(self, tmp_path):
        _write_single_deck(tmp_path / "paper.json", "Paper")
        entries = rebuild_library_manifest(tmp_path)
        assert entries[0]["archived"] is False

    def test_multi_deck_active_has_archived_false(self, tmp_path):
        _write_multi_deck(tmp_path / "textbook", "Textbook")
        entries = rebuild_library_manifest(tmp_path)
        assert entries[0]["archived"] is False


# ──────────────────────────────────────────────────────────────
# rebuild_library_manifest — archive subdirectory
# ──────────────────────────────────────────────────────────────

class TestRebuildWithArchive:
    def test_archived_single_deck_tagged(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        _write_single_deck(archive / "paper.json", "Old Paper")
        entries = rebuild_library_manifest(tmp_path)
        assert len(entries) == 1
        assert entries[0]["title"] == "Old Paper"
        assert entries[0]["archived"] is True

    def test_archived_multi_deck_tagged(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        _write_multi_deck(archive / "textbook", "Old Textbook")
        entries = rebuild_library_manifest(tmp_path)
        assert len(entries) == 1
        assert entries[0]["title"] == "Old Textbook"
        assert entries[0]["archived"] is True

    def test_active_and_archived_coexist(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        _write_single_deck(tmp_path / "new.json", "New Paper")
        _write_single_deck(archive / "old.json", "Old Paper")
        entries = rebuild_library_manifest(tmp_path)
        assert len(entries) == 2
        active   = next(e for e in entries if e["title"] == "New Paper")
        archived = next(e for e in entries if e["title"] == "Old Paper")
        assert active["archived"]   is False
        assert archived["archived"] is True

    def test_archived_chapter_files_skipped(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        _write_multi_deck(archive / "textbook", "Textbook", chapter_count=3)
        entries = rebuild_library_manifest(tmp_path)
        # Only the index should appear, not the chapter files
        assert len(entries) == 1

    def test_archived_slide_count_sums_correctly(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        _write_multi_deck(archive / "textbook", "Textbook", chapter_count=4)
        entries = rebuild_library_manifest(tmp_path)
        assert entries[0]["slide_count"] == 4

    def test_archived_file_key_contains_archive_path(self, tmp_path):
        archive = tmp_path / "archive"
        archive.mkdir()
        _write_single_deck(archive / "paper.json", "Paper")
        entries = rebuild_library_manifest(tmp_path)
        assert "archive" in entries[0]["file"]

    def test_upsert_defaults_archived_to_false(self, tmp_path):
        entry = {
            "title": "Paper", "file": "outputs/paper.json",
            "type": "single_deck", "generated_at": "2026-05-01T00:00:00+00:00",
            "provider": "anthropic", "model": "claude-sonnet-4-6",
            "slide_count": 5, "deck_count": 1,
        }
        upsert_library_manifest(tmp_path, entry)
        data = json.loads((tmp_path / "library.json").read_text())
        assert data[0]["archived"] is False

    def test_upsert_preserves_explicit_archived_true(self, tmp_path):
        entry = {
            "title": "Paper", "file": "outputs/archive/paper.json",
            "type": "single_deck", "generated_at": "2026-05-01T00:00:00+00:00",
            "provider": "anthropic", "model": "claude-sonnet-4-6",
            "slide_count": 5, "deck_count": 1,
            "archived": True,
        }
        upsert_library_manifest(tmp_path, entry)
        data = json.loads((tmp_path / "library.json").read_text())
        assert data[0]["archived"] is True
