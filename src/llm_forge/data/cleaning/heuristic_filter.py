"""Rule-based heuristic quality filtering inspired by Gopher, C4, and FineWeb.

Applies a battery of fast, deterministic text-quality checks to identify
and remove low-quality documents from web-scraped corpora.  Each rule
short-circuits on rejection so that expensive checks are never reached for
obviously bad documents.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_forge.utils.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("data.cleaning.heuristic_filter")

# ---------------------------------------------------------------------------
# Built-in English stop words (top ~45, sufficient for the >= 2 check)
# ---------------------------------------------------------------------------

ENGLISH_STOP_WORDS: set[str] = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "if",
    "about",
    "who",
    "which",
    "when",
}

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_LOREM_RE = re.compile(r"lorem\s+ipsum", re.IGNORECASE)
_CURLY_BRACKET_RE = re.compile(r"[{}]")
_JS_RE = re.compile(r"\bjavascript\b", re.IGNORECASE)
_COOKIE_RE = re.compile(r"cookie\s+policy", re.IGNORECASE)
_TOS_RE = re.compile(r"terms[\s-]+of[\s-]+use", re.IGNORECASE)
_ELLIPSIS_RE = re.compile(r"\.{3}|…")
_BULLET_RE = re.compile(r"^[\s]*[•\-\*▪►▸◆◇○●■□➤→»]", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"[.!?\"'\u201d\u2019]$")
_SYMBOL_CHARS = set("#…†‡§¶©®™°±×÷")


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------


@dataclass
class HeuristicThresholds:
    """Configurable thresholds for heuristic quality checks.

    Defaults match the Gopher / C4 / FineWeb recommended values.
    """

    min_word_count: int = 50
    max_word_count: int = 100_000
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0
    min_alpha_ratio: float = 0.70
    max_symbol_to_word_ratio: float = 0.10
    min_stop_word_count: int = 2
    max_ellipsis_proportion: float = 0.30
    max_bullet_proportion: float = 0.80
    max_duplicate_line_char_fraction: float = 0.20
    max_duplicate_paragraph_char_fraction: float = 0.20
    max_duplicate_5gram_char_fraction: float = 0.15
    min_lines_ending_punctuation: float = 0.12


# ---------------------------------------------------------------------------
# Heuristic filter class
# ---------------------------------------------------------------------------


class HeuristicFilter:
    """Battery of rule-based quality checks for text documents.

    Each check method returns ``(passed: bool, reason: str)``. The main
    :meth:`check` method runs them in sequence and short-circuits on the
    first failure.

    Parameters
    ----------
    thresholds:
        Custom threshold values.  Uses :class:`HeuristicThresholds` defaults
        when *None*.
    stop_words:
        Custom stop-word set.  Uses :data:`ENGLISH_STOP_WORDS` when *None*.
    """

    def __init__(
        self,
        thresholds: HeuristicThresholds | None = None,
        stop_words: set[str] | None = None,
    ) -> None:
        self.t = thresholds or HeuristicThresholds()
        self.stop_words = stop_words or ENGLISH_STOP_WORDS

    # -- individual checks (ordered cheapest to most expensive) ------------

    def _check_boilerplate(self, text: str) -> tuple[bool, str]:
        """Reject lorem-ipsum and web boilerplate."""
        if _LOREM_RE.search(text):
            return (False, "contains lorem ipsum")
        if _CURLY_BRACKET_RE.search(text):
            return (False, "contains curly brackets (likely code/template)")
        if _JS_RE.search(text):
            return (False, "contains 'javascript' keyword")
        if _COOKIE_RE.search(text):
            return (False, "contains 'cookie policy'")
        if _TOS_RE.search(text):
            return (False, "contains 'terms-of-use'")
        return (True, "")

    def _check_word_count(self, words: list[str]) -> tuple[bool, str]:
        """Check that word count is in [min, max]."""
        n = len(words)
        if n < self.t.min_word_count:
            return (False, f"word count {n} < {self.t.min_word_count}")
        if n > self.t.max_word_count:
            return (False, f"word count {n} > {self.t.max_word_count}")
        return (True, "")

    def _check_mean_word_length(self, words: list[str]) -> tuple[bool, str]:
        """Check that mean word length is in [min, max]."""
        if not words:
            return (False, "no words")
        mean_len = sum(len(w) for w in words) / len(words)
        if mean_len < self.t.min_mean_word_length:
            return (False, f"mean word length {mean_len:.1f} < {self.t.min_mean_word_length}")
        if mean_len > self.t.max_mean_word_length:
            return (False, f"mean word length {mean_len:.1f} > {self.t.max_mean_word_length}")
        return (True, "")

    def _check_alpha_ratio(self, text: str) -> tuple[bool, str]:
        """Check that alphabetic characters make up >= threshold of all chars."""
        if not text:
            return (False, "empty text")
        alpha_count = sum(1 for c in text if c.isalpha())
        total = len(text)
        ratio = alpha_count / total
        if ratio < self.t.min_alpha_ratio:
            return (False, f"alpha ratio {ratio:.2f} < {self.t.min_alpha_ratio}")
        return (True, "")

    def _check_symbol_ratio(self, words: list[str], text: str) -> tuple[bool, str]:
        """Check that symbol-to-word ratio is <= threshold."""
        if not words:
            return (False, "no words")
        symbol_count = sum(1 for c in text if c in _SYMBOL_CHARS)
        ratio = symbol_count / len(words)
        if ratio > self.t.max_symbol_to_word_ratio:
            return (False, f"symbol/word ratio {ratio:.2f} > {self.t.max_symbol_to_word_ratio}")
        return (True, "")

    def _check_stop_words(self, words: list[str]) -> tuple[bool, str]:
        """Check that the text contains >= min stop words."""
        lower_words = {w.lower() for w in words}
        stop_count = len(lower_words & self.stop_words)
        if stop_count < self.t.min_stop_word_count:
            return (False, f"stop word count {stop_count} < {self.t.min_stop_word_count}")
        return (True, "")

    def _check_ellipsis_proportion(self, lines: list[str]) -> tuple[bool, str]:
        """Check that ellipsis lines / total lines <= threshold."""
        if not lines:
            return (True, "")
        ellipsis_lines = sum(1 for line in lines if _ELLIPSIS_RE.search(line))
        proportion = ellipsis_lines / len(lines)
        if proportion > self.t.max_ellipsis_proportion:
            return (
                False,
                f"ellipsis proportion {proportion:.2f} > {self.t.max_ellipsis_proportion}",
            )
        return (True, "")

    def _check_bullet_proportion(self, lines: list[str]) -> tuple[bool, str]:
        """Check that bullet-point lines / total lines <= threshold."""
        if not lines:
            return (True, "")
        bullet_lines = sum(1 for line in lines if _BULLET_RE.match(line))
        proportion = bullet_lines / len(lines)
        if proportion > self.t.max_bullet_proportion:
            return (False, f"bullet proportion {proportion:.2f} > {self.t.max_bullet_proportion}")
        return (True, "")

    def _check_lines_ending_punctuation(self, lines: list[str]) -> tuple[bool, str]:
        """Check that >= threshold of non-empty lines end with sentence-ending punctuation."""
        non_empty = [line.strip() for line in lines if line.strip()]
        if not non_empty:
            return (True, "")
        ending_punct = sum(1 for line in non_empty if _SENTENCE_END_RE.search(line))
        proportion = ending_punct / len(non_empty)
        if proportion < self.t.min_lines_ending_punctuation:
            return (
                False,
                f"lines ending in punctuation {proportion:.2f} "
                f"< {self.t.min_lines_ending_punctuation}",
            )
        return (True, "")

    def _check_duplicate_line_char_fraction(self, lines: list[str]) -> tuple[bool, str]:
        """Check that duplicate lines don't account for too many characters."""
        if not lines:
            return (True, "")

        total_chars = sum(len(line) for line in lines)
        if total_chars == 0:
            return (True, "")

        line_counts: Counter[str] = Counter(lines)
        dup_chars = sum(len(line) * (count - 1) for line, count in line_counts.items() if count > 1)
        fraction = dup_chars / total_chars
        if fraction > self.t.max_duplicate_line_char_fraction:
            return (
                False,
                f"duplicate line char fraction {fraction:.2f} "
                f"> {self.t.max_duplicate_line_char_fraction}",
            )
        return (True, "")

    def _check_duplicate_paragraph_char_fraction(self, text: str) -> tuple[bool, str]:
        """Check that duplicate paragraphs don't account for too many chars."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return (True, "")

        total_chars = sum(len(p) for p in paragraphs)
        if total_chars == 0:
            return (True, "")

        para_counts: Counter[str] = Counter(paragraphs)
        dup_chars = sum(len(p) * (count - 1) for p, count in para_counts.items() if count > 1)
        fraction = dup_chars / total_chars
        if fraction > self.t.max_duplicate_paragraph_char_fraction:
            return (
                False,
                f"duplicate paragraph char fraction {fraction:.2f} "
                f"> {self.t.max_duplicate_paragraph_char_fraction}",
            )
        return (True, "")

    def _check_duplicate_5gram_char_fraction(self, words: list[str]) -> tuple[bool, str]:
        """Check that duplicate 5-grams don't account for too many chars."""
        if len(words) < 5:
            return (True, "")

        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            return (True, "")

        # Build 5-grams and count
        ngrams: list[str] = []
        for i in range(len(words) - 4):
            ngram = " ".join(words[i : i + 5])
            ngrams.append(ngram)

        ngram_counts: Counter[str] = Counter(ngrams)
        dup_chars = sum(len(ng) * (count - 1) for ng, count in ngram_counts.items() if count > 1)
        fraction = dup_chars / total_chars
        if fraction > self.t.max_duplicate_5gram_char_fraction:
            return (
                False,
                f"duplicate 5-gram char fraction {fraction:.2f} "
                f"> {self.t.max_duplicate_5gram_char_fraction}",
            )
        return (True, "")

    # -- main check method -------------------------------------------------

    def check(self, text: str) -> tuple[bool, str]:
        """Run all heuristic checks on a document. Short-circuits on failure.

        Parameters
        ----------
        text:
            The document text to evaluate.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` if the document passes all checks, or
            ``(False, reason)`` with a human-readable rejection reason.
        """
        # Precompute shared derived data
        words = text.split()
        lines = text.split("\n")

        # Ordered from cheapest to most expensive, short-circuiting on failure
        checks = [
            lambda: self._check_boilerplate(text),
            lambda: self._check_word_count(words),
            lambda: self._check_mean_word_length(words),
            lambda: self._check_alpha_ratio(text),
            lambda: self._check_symbol_ratio(words, text),
            lambda: self._check_stop_words(words),
            lambda: self._check_ellipsis_proportion(lines),
            lambda: self._check_bullet_proportion(lines),
            lambda: self._check_lines_ending_punctuation(lines),
            lambda: self._check_duplicate_line_char_fraction(lines),
            lambda: self._check_duplicate_paragraph_char_fraction(text),
            lambda: self._check_duplicate_5gram_char_fraction(words),
        ]

        for check_fn in checks:
            passed, reason = check_fn()
            if not passed:
                return (False, reason)

        return (True, "")

    # -- dataset-level processing ------------------------------------------

    def filter_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
    ) -> Dataset:
        """Filter a HuggingFace Dataset using heuristic quality rules.

        Parameters
        ----------
        dataset:
            A ``datasets.Dataset`` instance.
        text_field:
            Name of the column containing text to evaluate.

        Returns
        -------
        Dataset
            The filtered dataset with only documents that pass all checks.

        Raises
        ------
        ValueError
            If *text_field* is not found in the dataset.
        """
        if text_field not in dataset.column_names:
            raise ValueError(
                f"Text field '{text_field}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        initial_count = len(dataset)
        rejection_reasons: dict[str, int] = {}

        logger.info(
            "Running heuristic quality filter on %d records (field='%s')...",
            initial_count,
            text_field,
        )

        def _keep(example: dict) -> bool:
            passed, reason = self.check(example[text_field])
            if not passed:
                # Aggregate by the first word of the reason for a summary
                key = reason.split("(")[0].strip() if "(" in reason else reason
                rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
            return passed

        dataset = dataset.filter(
            _keep,
            desc="Heuristic filtering",
        )

        final_count = len(dataset)
        removed = initial_count - final_count
        logger.info(
            "Heuristic filtering complete: %d -> %d records (removed %d, %.1f%%).",
            initial_count,
            final_count,
            removed,
            (removed / initial_count * 100) if initial_count else 0,
        )

        if rejection_reasons:
            logger.info("Rejection breakdown:")
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
                logger.info("  %s: %d", reason, count)

        return dataset
