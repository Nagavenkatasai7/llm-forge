"""Two-tier quality classification: FastText classifier + KenLM perplexity.

Scores documents on a 0-1 quality scale and filters datasets according to
quality presets (permissive, balanced, strict).  When trained models are not
available, falls back to a heuristic scorer based on text statistics
(vocabulary diversity, sentence structure, length).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from llm_forge.utils.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("data.cleaning.quality_classifier")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import fasttext

    _FASTTEXT_AVAILABLE = True
except ImportError:
    _FASTTEXT_AVAILABLE = False

try:
    import kenlm  # type: ignore[import-untyped]

    _KENLM_AVAILABLE = True
except ImportError:
    _KENLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------

# Each preset defines the percentile threshold: documents scoring above
# this percentile (of the full dataset score distribution) are retained.
QUALITY_PRESETS: dict[str, float] = {
    "permissive": 0.50,  # retain top 50%
    "balanced": 0.30,  # retain top 30%
    "strict": 0.10,  # retain top 10%
}

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"[.!?]")
_WORD_RE = re.compile(r"\b\w+\b")


# ---------------------------------------------------------------------------
# Heuristic quality scorer (always-available fallback)
# ---------------------------------------------------------------------------


def _heuristic_quality_score(text: str) -> float:
    """Compute a heuristic quality score in [0, 1] based on text statistics.

    The score combines several signals:
    - Vocabulary diversity (type-token ratio, capped)
    - Sentence structure (average sentence length in words)
    - Text length (log-scaled, favoring medium-length documents)
    - Punctuation ratio (proper use of sentence-ending punctuation)
    - Paragraph structure (presence of paragraph breaks)

    Parameters
    ----------
    text:
        The document text.

    Returns
    -------
    float
        A quality score in [0.0, 1.0] where higher is better.
    """
    if not text or not text.strip():
        return 0.0

    words = _WORD_RE.findall(text.lower())
    num_words = len(words)

    if num_words < 5:
        return 0.05

    # -- Signal 1: Vocabulary diversity (type-token ratio) --
    # Cap at 500 words to avoid penalizing long documents
    sample_words = words[:500]
    unique_words = len(set(sample_words))
    ttr = unique_words / len(sample_words)
    # Map TTR from [0.1, 0.8] to [0, 1]
    vocab_score = min(1.0, max(0.0, (ttr - 0.1) / 0.7))

    # -- Signal 2: Sentence structure --
    sentences = _SENTENCE_END_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)
    avg_sentence_length = num_words / num_sentences
    # Ideal avg sentence length is 10-25 words
    if 10 <= avg_sentence_length <= 25:
        sentence_score = 1.0
    elif avg_sentence_length < 5 or avg_sentence_length > 60:
        sentence_score = 0.2
    else:
        # Linearly interpolate
        if avg_sentence_length < 10:
            sentence_score = 0.2 + 0.8 * (avg_sentence_length - 5) / 5
        else:
            sentence_score = 1.0 - 0.8 * (avg_sentence_length - 25) / 35

    # -- Signal 3: Text length (log-scaled) --
    # Sweet spot: 100-5000 words
    if 100 <= num_words <= 5000:
        length_score = 1.0
    elif num_words < 100:
        length_score = max(0.2, math.log1p(num_words) / math.log1p(100))
    else:
        # Slight penalty for very long docs, but don't punish too harshly
        length_score = max(0.5, 1.0 - (math.log(num_words / 5000) / 10))

    # -- Signal 4: Punctuation usage --
    punct_count = sum(1 for c in text if c in ".!?,;:\"'()-")
    punct_ratio = punct_count / len(text)
    # Good range: 0.02 - 0.08
    if 0.02 <= punct_ratio <= 0.08:
        punct_score = 1.0
    elif punct_ratio < 0.01 or punct_ratio > 0.15:
        punct_score = 0.2
    else:
        punct_score = 0.6

    # -- Signal 5: Paragraph structure --
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        para_score = 1.0
    elif len(text.split("\n")) >= 3:
        para_score = 0.7
    else:
        para_score = 0.4

    # -- Weighted combination --
    score = (
        0.30 * vocab_score
        + 0.25 * sentence_score
        + 0.15 * length_score
        + 0.15 * punct_score
        + 0.15 * para_score
    )

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# FastText quality classifier interface
# ---------------------------------------------------------------------------


class _FastTextQualityModel:
    """Wrapper for a trained FastText binary quality classifier.

    The model is expected to be a supervised FastText model trained on
    (high_quality, low_quality) labels.  If the model file is not found,
    this wrapper logs a warning and returns *None* for all predictions.
    """

    def __init__(self, model_path: str) -> None:
        self.model = None
        self.model_path = model_path

        if not _FASTTEXT_AVAILABLE:
            logger.warning(
                "fasttext not installed; FastText quality classifier disabled. "
                "Install with: pip install fasttext-wheel"
            )
            return

        path = Path(model_path)
        if path.exists():
            logger.info("Loading FastText quality model from %s", model_path)
            self.model = fasttext.load_model(str(path))
        else:
            logger.warning(
                "FastText quality model not found at '%s'. "
                "Falling back to heuristic scoring. "
                "To train a quality classifier, see the llm-forge documentation.",
                model_path,
            )

    def predict(self, text: str) -> float | None:
        """Predict quality score using the FastText model.

        Returns
        -------
        float or None
            Quality score in [0, 1], or *None* if model is unavailable.
        """
        if self.model is None:
            return None

        # Replace newlines for FastText (expects single-line input)
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return 0.0

        predictions = self.model.predict(clean_text, k=2)
        labels = predictions[0]
        probs = predictions[1]

        # Expected labels: __label__hq (high quality), __label__lq (low quality)
        for label, prob in zip(labels, probs, strict=False):
            if "hq" in label or "high" in label:
                return float(prob)

        # If only low-quality label found, score = 1 - prob(lq)
        return 1.0 - float(probs[0])


# ---------------------------------------------------------------------------
# KenLM perplexity scorer interface
# ---------------------------------------------------------------------------


class _KenLMScorer:
    """Wrapper for KenLM perplexity-based quality scoring.

    Lower perplexity (under a well-trained LM on high-quality text) indicates
    higher quality.  The perplexity is converted to a [0, 1] score via a
    sigmoid-like mapping.
    """

    def __init__(self, model_path: str) -> None:
        self.model = None
        self.model_path = model_path

        if not _KENLM_AVAILABLE:
            logger.warning(
                "kenlm not installed; KenLM perplexity scorer disabled. "
                "Install with: pip install https://github.com/kpu/kenlm/archive/master.zip"
            )
            return

        path = Path(model_path)
        if path.exists():
            logger.info("Loading KenLM model from %s", model_path)
            self.model = kenlm.Model(str(path))
        else:
            logger.warning(
                "KenLM model not found at '%s'. Falling back to heuristic scoring.",
                model_path,
            )

    def score(self, text: str) -> float | None:
        """Score text quality via perplexity.

        Returns
        -------
        float or None
            Quality score in [0, 1] (lower perplexity -> higher score),
            or *None* if the model is unavailable.
        """
        if self.model is None:
            return None

        text = text.replace("\n", " ").strip()
        if not text:
            return 0.0

        # KenLM returns log10 probability per word
        words = text.split()
        if not words:
            return 0.0

        log_score = self.model.score(text, bos=True, eos=True)
        # Convert to perplexity: 10^(-log_score / num_words)
        num_words = len(words)
        perplexity = 10 ** (-log_score / num_words)

        # Map perplexity to [0, 1] via sigmoid-like function
        # Good web text: perplexity ~100-500
        # Bad text: perplexity ~1000-10000+
        # We want: ppl=100 -> score~0.95, ppl=500 -> score~0.5, ppl=2000 -> score~0.1
        quality_score = 1.0 / (1.0 + (perplexity / 300) ** 1.5)

        return min(1.0, max(0.0, quality_score))


# ---------------------------------------------------------------------------
# Main quality classifier
# ---------------------------------------------------------------------------


class QualityClassifier:
    """Two-tier quality classifier with heuristic fallback.

    Combines FastText classification and KenLM perplexity scoring when
    models are available.  Falls back to a rule-based heuristic scorer
    when trained models are not present.

    Parameters
    ----------
    fasttext_model_path:
        Path to a trained FastText quality classification model (.bin).
    kenlm_model_path:
        Path to a trained KenLM language model (.arpa or .binary).
    """

    def __init__(
        self,
        fasttext_model_path: str | None = None,
        kenlm_model_path: str | None = None,
    ) -> None:
        self._fasttext = None
        self._kenlm = None

        if fasttext_model_path:
            self._fasttext = _FastTextQualityModel(fasttext_model_path)
        if kenlm_model_path:
            self._kenlm = _KenLMScorer(kenlm_model_path)

        # Log which scoring methods are active
        methods = []
        if self._fasttext and self._fasttext.model:
            methods.append("FastText classifier")
        if self._kenlm and self._kenlm.model:
            methods.append("KenLM perplexity")
        if not methods:
            methods.append("heuristic fallback")
        logger.info("Quality scoring methods: %s", ", ".join(methods))

    def score(self, text: str) -> float:
        """Score a document's quality on a [0, 1] scale.

        Combines available model scores with equal weight.  Falls back to
        the heuristic scorer if no models are available.

        Parameters
        ----------
        text:
            The document text.

        Returns
        -------
        float
            Quality score in [0.0, 1.0] where higher is better.
        """
        scores: list[float] = []

        # Try FastText
        if self._fasttext:
            ft_score = self._fasttext.predict(text)
            if ft_score is not None:
                scores.append(ft_score)

        # Try KenLM
        if self._kenlm:
            klm_score = self._kenlm.score(text)
            if klm_score is not None:
                scores.append(klm_score)

        # Fallback to heuristic if no model scores available
        if not scores:
            return _heuristic_quality_score(text)

        # Average available model scores
        return sum(scores) / len(scores)

    def filter_dataset(
        self,
        dataset: Dataset,
        preset: str = "balanced",
        text_field: str = "text",
    ) -> Dataset:
        """Filter a dataset by quality score according to a preset.

        Scores all documents, computes the threshold from the score
        distribution, and retains only documents above that threshold.

        Parameters
        ----------
        dataset:
            A HuggingFace ``Dataset`` instance.
        preset:
            Quality preset name: ``"permissive"`` (top 50%),
            ``"balanced"`` (top 30%), or ``"strict"`` (top 10%).
        text_field:
            Column containing the document text.

        Returns
        -------
        Dataset
            Filtered dataset containing only high-quality documents.

        Raises
        ------
        ValueError
            If *preset* is unknown or *text_field* is not in the dataset.
        """
        if preset not in QUALITY_PRESETS:
            raise ValueError(
                f"Unknown quality preset '{preset}'. "
                f"Valid presets: {sorted(QUALITY_PRESETS.keys())}"
            )

        if text_field not in dataset.column_names:
            raise ValueError(
                f"Text field '{text_field}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        retain_fraction = QUALITY_PRESETS[preset]
        initial_count = len(dataset)

        logger.info(
            "Scoring %d documents for quality (preset='%s', retain top %.0f%%)...",
            initial_count,
            preset,
            retain_fraction * 100,
        )

        # Score all documents
        scores: list[float] = []
        for i in range(initial_count):
            text = dataset[i][text_field]
            score = self.score(text)
            scores.append(score)

        if not scores:
            logger.warning("No documents to score.")
            return dataset

        # Compute the threshold: retain the top `retain_fraction` of documents
        # This means we discard the bottom (1 - retain_fraction)
        import numpy as np

        scores_arr = np.array(scores)
        percentile = (1.0 - retain_fraction) * 100
        threshold = float(np.percentile(scores_arr, percentile))

        logger.info(
            "Quality score distribution: min=%.3f, median=%.3f, max=%.3f, threshold (p%.0f)=%.3f",
            float(scores_arr.min()),
            float(np.median(scores_arr)),
            float(scores_arr.max()),
            percentile,
            threshold,
        )

        # Add scores as a column, filter, then remove the column
        dataset = dataset.add_column("__quality_score__", scores)
        dataset = dataset.filter(
            lambda example: example["__quality_score__"] >= threshold,
            desc=f"Quality filtering ({preset})",
        )
        dataset = dataset.remove_columns(["__quality_score__"])

        final_count = len(dataset)
        removed = initial_count - final_count
        logger.info(
            "Quality filtering complete: %d -> %d records (removed %d, %.1f%%).",
            initial_count,
            final_count,
            removed,
            (removed / initial_count * 100) if initial_count else 0,
        )

        return dataset
