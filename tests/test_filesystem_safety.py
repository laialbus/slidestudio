"""
Milestone 7 — filesystem safety tests.

No mocking library. Uses types.SimpleNamespace for a minimal agents stub.
All file I/O uses the pytest tmp_path fixture; the real outputs/ dir is never touched.
"""

import json
import types

import pytest

from pipeline import write_deck_index
from schemas.global_skeleton import SectionEntry
from schemas.slides_final import FinalSlide, SlidesFinal


FORBIDDEN_CHARS = set('<>:"/\\|?*')

HOSTILE_HEADINGS = [
    'Chapter 1: The Cell / Is it Alive?',
    'Chapter 2: <H> & B >iology\\Root',
    "Chapter 3: Héros / Zéros",
    'Chapter 4: "Quotes" | Pipes * Stars',
    "Chapter 5: Question? Mark!",
]


def _slides_final() -> SlidesFinal:
    return SlidesFinal(
        title="Test",
        slides=[FinalSlide(index=1, heading="Slide 1", body="Bullet.", tag="Key Concept")],
    )


def _make_agents() -> dict:
    provider = types.SimpleNamespace(name="stub", model="stub-model")
    planner  = types.SimpleNamespace(provider=provider)
    return {"planner": planner}


def _make_decks_data() -> list[tuple]:
    return [
        (SectionEntry(heading=h, level=1, position=i), _slides_final())
        for i, h in enumerate(HOSTILE_HEADINGS)
    ]


class TestFilesystemSafety:
    def test_no_forbidden_chars_in_filenames(self, tmp_path):
        write_deck_index("Test Title", _make_decks_data(), [], _make_agents(), tmp_path)
        out_dir = tmp_path / "test_title"
        for f in out_dir.iterdir():
            for ch in FORBIDDEN_CHARS:
                assert ch not in f.name, f"Forbidden char {ch!r} found in {f.name!r}"

    def test_no_filename_exceeds_200_chars(self, tmp_path):
        write_deck_index("Test Title", _make_decks_data(), [], _make_agents(), tmp_path)
        out_dir = tmp_path / "test_title"
        for f in out_dir.iterdir():
            assert len(f.name) <= 200, f"Filename too long ({len(f.name)} chars): {f.name!r}"

    def test_index_file_fields_are_relative_paths(self, tmp_path):
        write_deck_index("Test Title", _make_decks_data(), [], _make_agents(), tmp_path)
        data = json.loads((tmp_path / "test_title" / "index.json").read_text())
        for deck in data["decks"]:
            assert not deck["file"].startswith("/"), f"Absolute path in file field: {deck['file']!r}"
            assert "/" not in deck["file"], f"Path separator in file field: {deck['file']!r}"
            assert "\\" not in deck["file"], f"Backslash in file field: {deck['file']!r}"

    def test_chapter_files_prefixed_with_1based_02d_numbering(self, tmp_path):
        write_deck_index("Test Title", _make_decks_data(), [], _make_agents(), tmp_path)
        out_dir = tmp_path / "test_title"
        chapter_files = sorted(f.name for f in out_dir.iterdir() if f.name != "index.json")
        assert len(chapter_files) == len(HOSTILE_HEADINGS)
        for i, fname in enumerate(chapter_files, start=1):
            expected_prefix = f"{i:02d}_"
            assert fname.startswith(expected_prefix), (
                f"Expected prefix {expected_prefix!r} but got filename {fname!r}"
            )

    def test_unicode_headings_produce_ascii_filenames(self, tmp_path):
        write_deck_index("Test Title", _make_decks_data(), [], _make_agents(), tmp_path)
        out_dir = tmp_path / "test_title"
        for f in out_dir.iterdir():
            assert f.name.isascii(), f"Non-ASCII characters in filename: {f.name!r}"
