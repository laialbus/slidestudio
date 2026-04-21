import pytest

from utils.slugify import slugify


class TestSlugigy:
    def test_windows_forbidden_characters(self):
        result = slugify('Chapter 1: The Cell / Is it Alive?')
        assert result == "chapter_1_the_cell_is_it_alive"
        for ch in '<>:"/\\|?*':
            assert ch not in result

    def test_leading_trailing_spaces(self):
        assert slugify("  leading / trailing  ") == "leading_trailing"

    def test_unicode_accent_normalised(self):
        result = slugify("Café résumé")
        assert result.isascii()
        assert "cafe" in result

    def test_unicode_section_symbol(self):
        result = slugify("§3.2 ATP Synthesis & the Mitochondria")
        assert result.isascii()
        assert "atp_synthesis" in result

    def test_empty_string(self):
        assert slugify("") == ""

    def test_long_heading_truncated_at_80(self):
        result = slugify("a" * 200)
        assert len(result) == 80

    def test_long_heading_custom_max_length(self):
        result = slugify("b" * 200, max_length=40)
        assert len(result) == 40

    def test_colon_removed(self):
        assert ":" not in slugify("Chapter 1: Introduction")

    def test_slash_removed(self):
        assert "/" not in slugify("Input/Output")

    def test_multiple_spaces_collapse_to_single_underscore(self):
        result = slugify("word   another")
        assert "__" not in result
        assert result == "word_another"

    def test_hyphen_becomes_underscore(self):
        result = slugify("well-known result")
        assert result == "well_known_result"

    def test_already_clean_string(self):
        assert slugify("simple title") == "simple_title"

    def test_unicode_ligature(self):
        result = slugify("ﬁle")  # ﬁ is a ligature for fi
        assert result.isascii()
