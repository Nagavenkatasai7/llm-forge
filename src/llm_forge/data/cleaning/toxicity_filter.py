"""Toxicity detection and filtering using the Detoxify library.

Scores documents across multiple toxicity dimensions (toxic, severe_toxic,
obscene, threat, insult, identity_attack) and filters datasets by rejecting
documents that exceed configurable thresholds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from llm_forge.utils.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("data.cleaning.toxicity_filter")

# ---------------------------------------------------------------------------
# Optional dependency: Detoxify
# ---------------------------------------------------------------------------

try:
    from detoxify import Detoxify

    _DETOXIFY_AVAILABLE = True
except ImportError:
    _DETOXIFY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Supported model variants
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {"original", "unbiased", "multilingual"}

# Toxicity categories returned by Detoxify
TOXICITY_CATEGORIES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]

# Maximum text length to send to the model (longer texts are truncated)
_MAX_TEXT_LENGTH = 512


# ---------------------------------------------------------------------------
# ToxicityFilter class
# ---------------------------------------------------------------------------


class ToxicityFilter:
    """Toxicity scoring and filtering using Detoxify transformer models.

    Parameters
    ----------
    model_name:
        Which Detoxify model variant to use:

        - ``"original"``: Trained on the Jigsaw Toxic Comment dataset
        - ``"unbiased"``: Trained with Unintended Bias in Toxicity Classification
        - ``"multilingual"``: XLM-RoBERTa based, supports multiple languages

    device:
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).  Defaults to
        ``"cpu"``.

    Raises
    ------
    ImportError
        If the ``detoxify`` library is not installed.
    ValueError
        If *model_name* is not a supported variant.
    """

    def __init__(
        self,
        model_name: Literal["original", "unbiased", "multilingual"] = "unbiased",
        device: str = "cpu",
    ) -> None:
        if not _DETOXIFY_AVAILABLE:
            raise ImportError(
                "Detoxify is required for toxicity filtering. Install it with: pip install detoxify"
            )

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model variant '{model_name}'. "
                f"Supported models: {sorted(SUPPORTED_MODELS)}"
            )

        self.model_name = model_name
        self.device = device

        logger.info(
            "Loading Detoxify model '%s' on device '%s'...",
            model_name,
            device,
        )
        self._model = Detoxify(model_name, device=device)
        logger.info("Detoxify model loaded successfully.")

    def score_text(self, text: str) -> dict[str, float]:
        """Score a single text for toxicity across all categories.

        Parameters
        ----------
        text:
            The text to evaluate.

        Returns
        -------
        dict[str, float]
            A dictionary mapping each toxicity category to its score in
            [0.0, 1.0], where higher means more toxic.

            Keys: ``toxicity``, ``severe_toxicity``, ``obscene``,
            ``threat``, ``insult``, ``identity_attack``.
        """
        if not text or not text.strip():
            return {cat: 0.0 for cat in TOXICITY_CATEGORIES}

        # Truncate to avoid OOM on very long texts
        truncated = text[:_MAX_TEXT_LENGTH]

        raw_scores = self._model.predict(truncated)

        # Normalize keys and ensure float values
        scores: dict[str, float] = {}
        for cat in TOXICITY_CATEGORIES:
            # Detoxify may return slightly different key names depending on
            # model variant.  Handle both formats.
            value = raw_scores.get(cat, raw_scores.get(cat.replace("_", " "), 0.0))
            scores[cat] = float(value)

        return scores

    def score_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Score a batch of texts for toxicity.

        More efficient than calling :meth:`score_text` in a loop because
        the model processes the entire batch at once.

        Parameters
        ----------
        texts:
            List of text strings to evaluate.

        Returns
        -------
        list[dict[str, float]]
            List of score dictionaries, one per input text.
        """
        if not texts:
            return []

        # Truncate and handle empty strings
        processed = []
        for text in texts:
            if not text or not text.strip():
                processed.append(" ")  # Detoxify needs non-empty strings
            else:
                processed.append(text[:_MAX_TEXT_LENGTH])

        raw_scores = self._model.predict(processed)

        # raw_scores is {category: [score1, score2, ...]}
        num_texts = len(texts)
        results: list[dict[str, float]] = []

        for i in range(num_texts):
            scores: dict[str, float] = {}
            for cat in TOXICITY_CATEGORIES:
                value_list = raw_scores.get(
                    cat, raw_scores.get(cat.replace("_", " "), [0.0] * num_texts)
                )
                if isinstance(value_list, list):
                    scores[cat] = float(value_list[i])
                else:
                    scores[cat] = float(value_list)
            results.append(scores)

        return results

    def filter_dataset(
        self,
        dataset: Dataset,
        threshold: float = 0.6,
        text_field: str = "text",
        batch_size: int = 64,
        categories: list[str] | None = None,
    ) -> Dataset:
        """Filter a dataset by removing documents that exceed the toxicity threshold.

        A document is removed if *any* of the evaluated toxicity categories
        exceeds the threshold.

        Parameters
        ----------
        dataset:
            A HuggingFace ``Dataset`` instance.
        threshold:
            Maximum toxicity score allowed.  Documents with any category
            score above this are removed.
        text_field:
            Column containing the document text.
        batch_size:
            Number of documents to score in each batch.
        categories:
            Which toxicity categories to evaluate.  Defaults to all
            categories.  Useful if you only care about e.g. ``["toxicity",
            "severe_toxicity"]``.

        Returns
        -------
        Dataset
            Filtered dataset with toxic documents removed.

        Raises
        ------
        ValueError
            If *text_field* is not found in the dataset, or if any
            category name is invalid.
        """
        if text_field not in dataset.column_names:
            raise ValueError(
                f"Text field '{text_field}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        eval_categories = categories or TOXICITY_CATEGORIES
        for cat in eval_categories:
            if cat not in TOXICITY_CATEGORIES:
                raise ValueError(
                    f"Unknown toxicity category '{cat}'. Valid categories: {TOXICITY_CATEGORIES}"
                )

        initial_count = len(dataset)
        logger.info(
            "Running toxicity filter on %d records (threshold=%.2f, model='%s', categories=%s)...",
            initial_count,
            threshold,
            self.model_name,
            eval_categories,
        )

        # Score all documents in batches
        all_max_scores: list[float] = []
        category_rejection_counts: dict[str, int] = {cat: 0 for cat in eval_categories}

        num_batches = math.ceil(initial_count / batch_size)
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, initial_count)

            batch_texts = [dataset[i][text_field] for i in range(start, end)]
            batch_scores = self.score_batch(batch_texts)

            for scores in batch_scores:
                max_score = 0.0
                for cat in eval_categories:
                    score = scores.get(cat, 0.0)
                    if score > max_score:
                        max_score = score
                    if score > threshold:
                        category_rejection_counts[cat] += 1
                all_max_scores.append(max_score)

            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                logger.info(
                    "  Scored batch %d/%d (%d documents)...",
                    batch_idx + 1,
                    num_batches,
                    end,
                )

        # Filter: keep documents where max score across categories <= threshold
        dataset = dataset.add_column("__toxicity_max__", all_max_scores)
        dataset = dataset.filter(
            lambda example: example["__toxicity_max__"] <= threshold,
            desc="Toxicity filtering",
        )
        dataset = dataset.remove_columns(["__toxicity_max__"])

        final_count = len(dataset)
        removed = initial_count - final_count
        logger.info(
            "Toxicity filtering complete: %d -> %d records (removed %d, %.1f%%).",
            initial_count,
            final_count,
            removed,
            (removed / initial_count * 100) if initial_count else 0,
        )

        if removed > 0:
            logger.info("Rejection breakdown by category (documents may trigger multiple):")
            for cat, count in sorted(category_rejection_counts.items(), key=lambda x: -x[1]):
                if count > 0:
                    logger.info("  %s: %d", cat, count)

        return dataset
