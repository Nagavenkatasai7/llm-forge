"""Tests for the heuristic_filter module.

Covers word count checks, alpha ratio checks, boilerplate rejection,
and valid-content pass-through. Skips gracefully if not importable.
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.data.cleaning.heuristic_filter import (
        HeuristicFilter,
        HeuristicThresholds,
    )

    _MODULE_AVAILABLE = True
except ImportError:
    _MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODULE_AVAILABLE,
    reason="llm_forge.data.cleaning.heuristic_filter not importable",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Build a text that passes default thresholds: 50+ words, high alpha ratio,
# stop words present, lines ending in punctuation, no boilerplate.
GOOD_TEXT = (
    "The quick brown fox jumps over the lazy dog in the park. "
    "This is a wonderful sentence that contains many words and is quite long. "
    "We have several stop words like the, is, and, for, to, with, that appear. "
    "Machine learning is a subset of artificial intelligence that has grown. "
    "The researchers published their findings in a prestigious journal. "
    "Natural language processing enables computers to understand human language. "
    "Deep learning models have achieved remarkable results on many benchmarks. "
    "The training data was carefully curated to ensure high quality output."
)


# ===================================================================
# Word count filtering
# ===================================================================


class TestWordCountFilter:
    """Test word count threshold checks."""

    def test_too_few_words_rejected(self) -> None:
        """Text with fewer words than min_word_count is rejected."""
        short_text = "Too short."
        hf = HeuristicFilter(thresholds=HeuristicThresholds(min_word_count=50))
        passed, reason = hf.check(short_text)
        assert passed is False
        assert "word count" in reason.lower()

    def test_too_many_words_rejected(self) -> None:
        """Text exceeding max_word_count is rejected."""
        huge_text = " ".join(["word"] * 200_000)
        hf = HeuristicFilter(thresholds=HeuristicThresholds(max_word_count=100_000))
        passed, reason = hf.check(huge_text)
        assert passed is False
        assert "word count" in reason.lower()

    def test_acceptable_word_count_passes(self) -> None:
        """Text within [min, max] word count passes the word-count check."""
        hf = HeuristicFilter(thresholds=HeuristicThresholds(min_word_count=5, max_word_count=200))
        passed, _ = hf._check_word_count(GOOD_TEXT.split())
        assert passed is True


# ===================================================================
# Alpha ratio filtering
# ===================================================================


class TestAlphaRatioFilter:
    """Test alphabetic character ratio threshold."""

    def test_low_alpha_ratio_rejected(self) -> None:
        """Text with mostly numbers/symbols is rejected."""
        numeric_text = "12345 67890 " * 100
        hf = HeuristicFilter(thresholds=HeuristicThresholds(min_alpha_ratio=0.5))
        passed, reason = hf._check_alpha_ratio(numeric_text)
        assert passed is False
        assert "alpha ratio" in reason.lower()

    def test_high_alpha_ratio_passes(self) -> None:
        """Normal English text has a high alpha ratio."""
        hf = HeuristicFilter()
        passed, _ = hf._check_alpha_ratio(GOOD_TEXT)
        assert passed is True


# ===================================================================
# Boilerplate rejection
# ===================================================================


class TestBoilerplateRejection:
    """Test lorem-ipsum and web boilerplate detection."""

    def test_lorem_ipsum_rejected(self) -> None:
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        hf = HeuristicFilter()
        passed, reason = hf._check_boilerplate(text)
        assert passed is False
        assert "lorem ipsum" in reason.lower()

    def test_javascript_keyword_rejected(self) -> None:
        text = "Please enable JavaScript in your browser to continue."
        hf = HeuristicFilter()
        passed, reason = hf._check_boilerplate(text)
        assert passed is False
        assert "javascript" in reason.lower()

    def test_cookie_policy_rejected(self) -> None:
        text = "By continuing you accept our cookie policy and terms."
        hf = HeuristicFilter()
        passed, reason = hf._check_boilerplate(text)
        assert passed is False
        assert "cookie policy" in reason.lower()

    def test_clean_text_not_rejected(self) -> None:
        text = "This is a perfectly normal sentence about science."
        hf = HeuristicFilter()
        passed, _ = hf._check_boilerplate(text)
        assert passed is True


# ===================================================================
# Full check pass-through
# ===================================================================


class TestFullCheck:
    """Test that well-formed text passes all checks."""

    def test_good_text_passes(self) -> None:
        hf = HeuristicFilter()
        passed, reason = hf.check(GOOD_TEXT)
        assert passed is True
        assert reason == ""


# ===================================================================
# Dataset-level filtering
# ===================================================================


class TestDatasetFiltering:
    """Test filter_dataset method."""

    def test_filter_dataset_removes_bad_records(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": GOOD_TEXT},
                {"text": "short"},  # too few words
            ]
        )
        hf = HeuristicFilter()
        filtered = hf.filter_dataset(ds, text_field="text")
        assert len(filtered) <= len(ds)

    def test_missing_text_field_raises(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list([{"content": "hello"}])
        hf = HeuristicFilter()
        with pytest.raises(ValueError, match="text"):
            hf.filter_dataset(ds, text_field="text")
