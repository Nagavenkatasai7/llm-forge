"""Tests for the evaluation MetricsComputer.

Covers BLEU, ROUGE, exact_match, F1, and accuracy computation.
Skips if torch or other deps are not available.
"""

from __future__ import annotations

import pytest

_MODULE_AVAILABLE = False
_SKIP_REASON = ""

try:
    from llm_forge.evaluation.metrics import MetricsComputer

    _MODULE_AVAILABLE = True
except ImportError as e:
    _SKIP_REASON = f"evaluation.metrics not importable: {e}"

pytestmark = pytest.mark.skipif(
    not _MODULE_AVAILABLE,
    reason=_SKIP_REASON or "MetricsComputer not available",
)


@pytest.fixture()
def mc() -> MetricsComputer:
    return MetricsComputer()


# ===================================================================
# BLEU
# ===================================================================


class TestBLEU:
    """Test BLEU score computation."""

    def test_bleu_perfect_match(self, mc: MetricsComputer) -> None:
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = mc.compute_bleu(preds, refs)
        assert "bleu" in result
        assert result["bleu"] > 0.9

    def test_bleu_no_overlap(self, mc: MetricsComputer) -> None:
        preds = ["foo bar baz"]
        refs = ["the cat sat on the mat"]
        result = mc.compute_bleu(preds, refs)
        assert "bleu" in result
        # Very low score expected
        assert result["bleu"] < 0.5

    def test_bleu_partial_overlap(self, mc: MetricsComputer) -> None:
        preds = ["the cat sat"]
        refs = ["the cat sat on the mat"]
        result = mc.compute_bleu(preds, refs)
        assert "bleu" in result
        assert 0.0 < result["bleu"] < 1.0

    def test_bleu_returns_individual_ngrams(self, mc: MetricsComputer) -> None:
        preds = ["hello world"]
        refs = ["hello world"]
        result = mc.compute_bleu(preds, refs)
        assert "bleu_1" in result
        assert "bleu_2" in result

    def test_bleu_mismatched_lengths_raises(self, mc: MetricsComputer) -> None:
        with pytest.raises(ValueError, match="same length"):
            mc.compute_bleu(["a"], ["b", "c"])


# ===================================================================
# ROUGE
# ===================================================================


class TestROUGE:
    """Test ROUGE score computation."""

    def test_rouge_perfect_match(self, mc: MetricsComputer) -> None:
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = mc.compute_rouge(preds, refs)
        assert "rougeL" in result or "rouge1" in result
        # Perfect match should have high score
        for key in result:
            assert result[key] > 0.9

    def test_rouge_no_overlap(self, mc: MetricsComputer) -> None:
        preds = ["xyz abc def"]
        refs = ["the cat sat on the mat"]
        result = mc.compute_rouge(preds, refs)
        # Should be low
        for key in result:
            assert result[key] < 0.5

    def test_rouge_partial_overlap(self, mc: MetricsComputer) -> None:
        preds = ["the quick brown cat"]
        refs = ["the cat sat on the mat"]
        result = mc.compute_rouge(preds, refs)
        # ROUGE-1 should show overlap ("the", "cat" are shared)
        assert 0.0 < result["rouge1"] < 1.0
        # ROUGE-2 may be 0 if no bigrams overlap; just check it exists
        assert "rouge2" in result or "rougeL" in result

    def test_rouge_mismatched_lengths_raises(self, mc: MetricsComputer) -> None:
        with pytest.raises(ValueError, match="same length"):
            mc.compute_rouge(["a"], ["b", "c"])


# ===================================================================
# Exact Match
# ===================================================================


class TestExactMatch:
    """Test exact match computation."""

    def test_exact_match_perfect(self, mc: MetricsComputer) -> None:
        preds = ["hello", "world"]
        refs = ["hello", "world"]
        result = mc.compute_exact_match(preds, refs)
        assert result["exact_match"] == 1.0

    def test_exact_match_none(self, mc: MetricsComputer) -> None:
        preds = ["hello", "world"]
        refs = ["goodbye", "earth"]
        result = mc.compute_exact_match(preds, refs)
        assert result["exact_match"] == 0.0

    def test_exact_match_partial(self, mc: MetricsComputer) -> None:
        preds = ["hello", "goodbye"]
        refs = ["hello", "world"]
        result = mc.compute_exact_match(preds, refs)
        assert result["exact_match"] == pytest.approx(0.5)

    def test_exact_match_with_normalization(self, mc: MetricsComputer) -> None:
        """Normalization should make case/punctuation insensitive."""
        preds = ["Hello, World!"]
        refs = ["hello world"]
        result = mc.compute_exact_match(preds, refs, normalize=True)
        assert result["exact_match"] == 1.0

    def test_exact_match_without_normalization(self, mc: MetricsComputer) -> None:
        preds = ["Hello, World!"]
        refs = ["hello world"]
        result = mc.compute_exact_match(preds, refs, normalize=False)
        assert result["exact_match"] == 0.0


# ===================================================================
# F1 Score
# ===================================================================


class TestF1:
    """Test token-level F1 score computation."""

    def test_f1_perfect_match(self, mc: MetricsComputer) -> None:
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = mc.compute_f1(preds, refs)
        assert "f1" in result
        assert result["f1"] == pytest.approx(1.0)

    def test_f1_no_overlap(self, mc: MetricsComputer) -> None:
        preds = ["xyz"]
        refs = ["abc"]
        result = mc.compute_f1(preds, refs)
        assert result["f1"] == 0.0

    def test_f1_partial_overlap(self, mc: MetricsComputer) -> None:
        preds = ["the cat sat"]
        refs = ["a cat sat on the mat"]
        result = mc.compute_f1(preds, refs)
        assert 0.0 < result["f1"] < 1.0
        assert "precision" in result
        assert "recall" in result

    def test_f1_mismatched_lengths_raises(self, mc: MetricsComputer) -> None:
        with pytest.raises(ValueError, match="same length"):
            mc.compute_f1(["a"], ["b", "c"])


# ===================================================================
# Accuracy
# ===================================================================


class TestAccuracy:
    """Test accuracy computation."""

    def test_accuracy_perfect(self, mc: MetricsComputer) -> None:
        result = mc.compute_accuracy(["a", "b", "c"], ["a", "b", "c"])
        assert result["accuracy"] == 1.0
        assert result["correct"] == 3
        assert result["total"] == 3

    def test_accuracy_zero(self, mc: MetricsComputer) -> None:
        result = mc.compute_accuracy(["a", "b"], ["x", "y"])
        assert result["accuracy"] == 0.0

    def test_accuracy_with_integers(self, mc: MetricsComputer) -> None:
        result = mc.compute_accuracy([1, 2, 3], [1, 2, 4])
        assert result["accuracy"] == pytest.approx(2 / 3)

    def test_accuracy_mismatched_lengths_raises(self, mc: MetricsComputer) -> None:
        with pytest.raises(ValueError, match="same length"):
            mc.compute_accuracy(["a"], ["b", "c"])


# ===================================================================
# compute_all
# ===================================================================


class TestComputeAll:
    """Test the combined compute_all method."""

    def test_compute_all_default(self, mc: MetricsComputer) -> None:
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = mc.compute_all(preds, refs)
        # Should contain keys from multiple metrics
        assert "bleu" in result or "exact_match" in result or "f1" in result

    def test_compute_all_selected(self, mc: MetricsComputer) -> None:
        preds = ["hello"]
        refs = ["hello"]
        result = mc.compute_all(preds, refs, include=["exact_match", "f1"])
        assert "exact_match" in result
        assert "f1" in result
