"""Tests for utils.pdf_hash.pdf_content_hash — content-based document identity."""

from utils.pdf_hash import DOC_HASH_LENGTH, pdf_content_hash


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


class TestPdfContentHash:
    def test_length_defaults_to_doc_hash_length(self, tmp_path):
        p = _write(tmp_path, "a.pdf", b"%PDF-1.7 hello")
        assert len(pdf_content_hash(p)) == DOC_HASH_LENGTH

    def test_custom_length(self, tmp_path):
        p = _write(tmp_path, "a.pdf", b"%PDF-1.7 hello")
        assert len(pdf_content_hash(p, length=16)) == 16

    def test_is_hex(self, tmp_path):
        p = _write(tmp_path, "a.pdf", b"%PDF-1.7 hello")
        int(pdf_content_hash(p), 16)  # raises ValueError if not hex

    def test_identical_content_same_hash(self, tmp_path):
        a = _write(tmp_path, "a.pdf", b"same bytes")
        b = _write(tmp_path, "b.pdf", b"same bytes")
        # Filename differs, content identical → identity matches (the whole point).
        assert pdf_content_hash(a) == pdf_content_hash(b)

    def test_different_content_different_hash(self, tmp_path):
        a = _write(tmp_path, "a.pdf", b"one")
        b = _write(tmp_path, "b.pdf", b"two")
        assert pdf_content_hash(a) != pdf_content_hash(b)

    def test_deterministic_across_calls(self, tmp_path):
        p = _write(tmp_path, "a.pdf", b"%PDF-1.7 stable")
        assert pdf_content_hash(p) == pdf_content_hash(p)

    def test_streams_large_file(self, tmp_path):
        # Larger than the internal read chunk — exercises the streaming loop.
        p = _write(tmp_path, "big.pdf", b"x" * (65536 * 3 + 17))
        assert len(pdf_content_hash(p)) == DOC_HASH_LENGTH
