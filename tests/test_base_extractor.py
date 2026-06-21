"""
Tests for the BaseExtractor abstraction and the make_extractor factory.

These cover the structural contract only — no PDF parsing — so they run without
model weights.
"""

import pytest

from extractors.base import BaseExtractor, ExtractionResult, ImageRecord, TocItem
from extractors.factory import make_extractor
from extractors.pdf import PDFExtractor


class TestFactory:
    def test_pymupdf4llm_returns_pdf_extractor(self):
        extractor = make_extractor("pymupdf4llm", chunk_size=8000, overlap_size=1500)
        assert isinstance(extractor, PDFExtractor)
        assert isinstance(extractor, BaseExtractor)

    def test_name_property(self):
        extractor = make_extractor("pymupdf4llm", chunk_size=8000, overlap_size=1500)
        assert extractor.name == "pymupdf4llm"

    def test_tunables_are_threaded_through(self):
        extractor = make_extractor("pymupdf4llm", chunk_size=1234, overlap_size=56)
        assert extractor._chunk_size == 1234
        assert extractor._overlap_size == 56

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError) as exc:
            make_extractor("nope", chunk_size=8000, overlap_size=1500)
        # Error lists the available extractors so the user can self-correct.
        assert "pymupdf4llm" in str(exc.value)


class TestBaseContract:
    def test_extract_not_implemented(self):
        base = BaseExtractor(chunk_size=8000, overlap_size=1500)
        with pytest.raises(NotImplementedError):
            base.extract("x.pdf")

    def test_name_not_implemented(self):
        base = BaseExtractor(chunk_size=8000, overlap_size=1500)
        with pytest.raises(NotImplementedError):
            _ = base.name


class TestBackCompatReexports:
    def test_models_importable_from_pdf_module(self):
        # Downstream + tests historically import these from extractors.pdf;
        # the re-export must keep that path valid after the move to base.py.
        from extractors.pdf import ExtractionResult as PdfResult
        from extractors.pdf import ImageRecord as PdfImage
        from extractors.pdf import TocItem as PdfToc
        assert PdfResult is ExtractionResult
        assert PdfImage is ImageRecord
        assert PdfToc is TocItem

    def test_chunk_helper_importable_from_pdf_module(self):
        from extractors.pdf import _chunk_by_chars
        assert _chunk_by_chars("", 10, 2) == []
