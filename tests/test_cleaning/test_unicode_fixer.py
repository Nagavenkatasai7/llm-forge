"""Tests for the unicode_fixer module.

Covers fix_unicode, invisible character removal, and NFC normalization.
Skips gracefully if the module or its optional deps are not available.
"""

from __future__ import annotations

import unicodedata

import pytest

try:
    from llm_forge.data.cleaning.unicode_fixer import fix_unicode, process_dataset

    _MODULE_AVAILABLE = True
except ImportError:
    _MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODULE_AVAILABLE,
    reason="llm_forge.data.cleaning.unicode_fixer not importable",
)


# ===================================================================
# Basic unicode fixing
# ===================================================================


class TestFixUnicode:
    """Test the fix_unicode function."""

    def test_clean_text_unchanged(self) -> None:
        """Normal ASCII text passes through unchanged."""
        text = "Hello, world! This is clean text."
        assert fix_unicode(text) == text

    def test_empty_string(self) -> None:
        assert fix_unicode("") == ""

    def test_nfc_normalization(self) -> None:
        """Characters in NFD form are converted to NFC."""
        # 'e' + combining acute accent (NFD) -> e-acute (NFC)
        nfd_text = "caf\u0065\u0301"
        result = fix_unicode(nfd_text)
        assert result == unicodedata.normalize("NFC", nfd_text)

    def test_invisible_character_removal(self) -> None:
        """Zero-width spaces and other invisible chars are stripped."""
        text_with_invisible = "Hello\u200bWorld\u200c!\ufeff"
        result = fix_unicode(text_with_invisible)
        assert "\u200b" not in result
        assert "\u200c" not in result
        assert "\ufeff" not in result
        assert "HelloWorld!" in result

    def test_control_characters_removed(self) -> None:
        """C0/C1 control characters (except tab/LF/CR/space) are stripped."""
        text_with_control = "Hello\x00World\x01\x02\x7f"
        result = fix_unicode(text_with_control)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x7f" not in result

    def test_tabs_and_newlines_preserved(self) -> None:
        """Tab, LF, and CR characters are preserved."""
        text = "Hello\tWorld\nNew line\rCarriage return"
        result = fix_unicode(text)
        assert "\t" in result
        assert "\n" in result

    def test_soft_hyphen_removed(self) -> None:
        """Soft hyphens are stripped."""
        text = "hyphen\u00adated"
        result = fix_unicode(text)
        assert "\u00ad" not in result

    def test_unicode_whitespace(self) -> None:
        """Normal spaces are preserved."""
        text = "word1 word2"
        result = fix_unicode(text)
        assert result == "word1 word2"


# ===================================================================
# Dataset processing
# ===================================================================


class TestProcessDataset:
    """Test applying fix_unicode to a HuggingFace Dataset."""

    def test_process_dataset(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": "Hello\u200bWorld"},
                {"text": "Clean text here"},
            ]
        )
        result = process_dataset(ds, text_field="text")
        assert "\u200b" not in result[0]["text"]
        assert result[1]["text"] == "Clean text here"

    def test_missing_text_field_raises(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list([{"content": "hello"}])
        with pytest.raises(ValueError, match="text"):
            process_dataset(ds, text_field="text")
