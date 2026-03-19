"""Tests for the deduplication module.

Covers exact SHA-256 dedup, unique documents (no removal), and
all-duplicates scenarios. Skips gracefully if not importable.
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.data.cleaning.deduplication import (
        DeduplicationStats,
        Deduplicator,
        exact_dedup,
    )

    _MODULE_AVAILABLE = True
except ImportError:
    _MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODULE_AVAILABLE,
    reason="llm_forge.data.cleaning.deduplication not importable",
)


# ===================================================================
# Exact deduplication
# ===================================================================


class TestExactDedup:
    """Test exact SHA-256 based deduplication."""

    def test_removes_exact_duplicates(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": "Hello world. This is a test document."},
                {"text": "Hello world. This is a test document."},
                {"text": "A different document entirely."},
            ]
        )
        result = exact_dedup(ds, text_field="text")
        assert len(result) == 2

    def test_whitespace_normalization(self) -> None:
        """Differing whitespace should still be treated as duplicates."""
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": "hello  world"},
                {"text": "hello world"},
            ]
        )
        result = exact_dedup(ds, text_field="text")
        assert len(result) == 1

    def test_case_normalization(self) -> None:
        """Differing case should be treated as duplicates."""
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": "Hello World"},
                {"text": "hello world"},
            ]
        )
        result = exact_dedup(ds, text_field="text")
        assert len(result) == 1


class TestUniqueDocuments:
    """Test that unique documents are not removed."""

    def test_no_removal_for_unique(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": "Document one about cats."},
                {"text": "Document two about dogs."},
                {"text": "Document three about birds."},
            ]
        )
        result = exact_dedup(ds, text_field="text")
        assert len(result) == 3


class TestAllDuplicates:
    """Test behavior when all documents are identical."""

    def test_keeps_only_one(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": "Identical content."},
                {"text": "Identical content."},
                {"text": "Identical content."},
                {"text": "Identical content."},
            ]
        )
        result = exact_dedup(ds, text_field="text")
        assert len(result) == 1

    def test_first_occurrence_kept(self) -> None:
        """The first occurrence in the dataset is the one retained."""
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": "same text", "id": 1},
                {"text": "same text", "id": 2},
                {"text": "same text", "id": 3},
            ]
        )
        result = exact_dedup(ds, text_field="text")
        assert len(result) == 1
        assert result[0]["id"] == 1


# ===================================================================
# Deduplicator orchestrator
# ===================================================================


class TestDeduplicator:
    """Test the Deduplicator orchestrator class."""

    def test_exact_tier(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list(
            [
                {"text": "Doc A."},
                {"text": "Doc A."},
                {"text": "Doc B."},
            ]
        )
        dedup = Deduplicator()
        result, stats = dedup.deduplicate(ds, tiers=["exact"], text_field="text")
        assert len(result) == 2
        assert isinstance(stats, DeduplicationStats)
        assert stats.exact_removed == 1

    def test_invalid_tier_raises(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list([{"text": "hello"}])
        dedup = Deduplicator()
        with pytest.raises(ValueError, match="Unknown deduplication tier"):
            dedup.deduplicate(ds, tiers=["nonexistent"], text_field="text")

    def test_missing_text_field_raises(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_list([{"content": "hello"}])
        with pytest.raises(ValueError, match="text"):
            exact_dedup(ds, text_field="text")


# ===================================================================
# DeduplicationStats
# ===================================================================


class TestDeduplicationStats:
    """Test the stats dataclass properties."""

    def test_total_removed(self) -> None:
        stats = DeduplicationStats(initial_count=100, final_count=80)
        assert stats.total_removed == 20

    def test_removal_rate(self) -> None:
        stats = DeduplicationStats(initial_count=100, final_count=75)
        assert stats.removal_rate == pytest.approx(0.25)

    def test_zero_initial_count(self) -> None:
        stats = DeduplicationStats(initial_count=0, final_count=0)
        assert stats.removal_rate == 0.0
