"""Metrics computation module for llm-forge evaluation.

Provides perplexity, BLEU, ROUGE-L, exact match, F1, and accuracy metrics.
All optional dependencies (rouge-score, nltk) are handled gracefully via
try/except imports.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch

from llm_forge.utils.logging import get_logger

logger = get_logger("evaluation.metrics")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer as _rouge_scorer_module

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_simple(text: str) -> list[str]:
    """Whitespace tokenization after normalization."""
    return _normalize_text(text).split()


# ---------------------------------------------------------------------------
# MetricsComputer
# ---------------------------------------------------------------------------


class MetricsComputer:
    """Unified metrics computation engine for llm-forge evaluation.

    Supports perplexity (model-based), BLEU, ROUGE-L, exact match, F1, and
    accuracy.  All methods degrade gracefully when optional dependencies
    (rouge-score, nltk) are not installed.

    Examples
    --------
    >>> mc = MetricsComputer()
    >>> mc.compute_f1(["the cat sat"], ["a cat sat on the mat"])
    {'f1': ...}
    """

    def __init__(self) -> None:
        if _NLTK_AVAILABLE:
            # Ensure punkt tokenizer data is available for BLEU
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                try:
                    nltk.download("punkt", quiet=True)
                except Exception:
                    logger.debug(
                        "Could not download NLTK punkt data; BLEU may use simple tokenizer."
                    )
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                try:
                    nltk.download("punkt_tab", quiet=True)
                except Exception:
                    logger.debug("Could not download NLTK punkt_tab data.")

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_perplexity(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
        max_length: int = 2048,
        batch_size: int = 4,
        device: str | None = None,
    ) -> dict[str, float]:
        """Compute perplexity over a list of text samples.

        Parameters
        ----------
        model:
            A HuggingFace causal-LM model.
        tokenizer:
            The corresponding tokenizer.
        texts:
            List of text strings to evaluate.
        max_length:
            Maximum sequence length for tokenization.
        batch_size:
            Number of texts per forward pass.
        device:
            Device to run inference on. Auto-detected if *None*.

        Returns
        -------
        dict
            ``{"perplexity": float, "avg_loss": float, "num_tokens": int}``
        """
        if device is None:
            device = str(next(model.parameters()).device)

        model.eval()
        total_loss = 0.0
        total_tokens = 0

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # Labels are the same as input_ids; padding tokens are ignored via -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Count non-padding tokens in labels
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100.0))  # cap to avoid overflow

        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "num_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    # BLEU
    # ------------------------------------------------------------------

    def compute_bleu(
        self,
        predictions: list[str],
        references: list[str],
        max_n: int = 4,
    ) -> dict[str, float]:
        """Compute corpus-level and average sentence-level BLEU scores.

        Falls back to a simple n-gram overlap implementation when NLTK is
        not installed.

        Parameters
        ----------
        predictions:
            List of predicted / generated strings.
        references:
            List of reference / ground-truth strings.
        max_n:
            Maximum n-gram order (default 4 for BLEU-4).

        Returns
        -------
        dict
            ``{"bleu": float, "bleu_1": float, ..., "bleu_<max_n>": float}``
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length."
            )

        if _NLTK_AVAILABLE:
            return self._bleu_nltk(predictions, references, max_n)
        return self._bleu_simple(predictions, references, max_n)

    def _bleu_nltk(
        self, predictions: list[str], references: list[str], max_n: int
    ) -> dict[str, float]:
        """BLEU via NLTK."""
        smoothing = SmoothingFunction().method1
        ref_tokens_list = [[_tokenize_simple(r)] for r in references]
        pred_tokens_list = [_tokenize_simple(p) for p in predictions]

        results: dict[str, float] = {}

        # Corpus BLEU (cumulative)
        weights_full = tuple(1.0 / max_n for _ in range(max_n))
        corpus_score = corpus_bleu(
            ref_tokens_list,
            pred_tokens_list,
            weights=weights_full,
            smoothing_function=smoothing,
        )
        results["bleu"] = corpus_score

        # Individual n-gram BLEU
        for n in range(1, max_n + 1):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(max_n))
            score = corpus_bleu(
                ref_tokens_list,
                pred_tokens_list,
                weights=weights,
                smoothing_function=smoothing,
            )
            results[f"bleu_{n}"] = score

        return results

    def _bleu_simple(
        self, predictions: list[str], references: list[str], max_n: int
    ) -> dict[str, float]:
        """Simple n-gram precision BLEU without NLTK."""
        logger.warning("NLTK not installed; using simplified BLEU computation.")

        def _get_ngrams(tokens: list[str], n: int) -> Counter:
            return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

        def _brevity_penalty(ref_len: int, pred_len: int) -> float:
            if pred_len >= ref_len:
                return 1.0
            return math.exp(1.0 - ref_len / max(pred_len, 1))

        total_precisions: dict[int, list[float]] = {n: [] for n in range(1, max_n + 1)}
        total_bp = []

        for pred_str, ref_str in zip(predictions, references, strict=False):
            pred_tok = _tokenize_simple(pred_str)
            ref_tok = _tokenize_simple(ref_str)
            bp = _brevity_penalty(len(ref_tok), len(pred_tok))
            total_bp.append(bp)

            for n in range(1, max_n + 1):
                pred_ngrams = _get_ngrams(pred_tok, n)
                ref_ngrams = _get_ngrams(ref_tok, n)
                clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
                total_count = max(sum(pred_ngrams.values()), 1)
                total_precisions[n].append(clipped / total_count)

        results: dict[str, float] = {}
        avg_bp = sum(total_bp) / max(len(total_bp), 1)

        for n in range(1, max_n + 1):
            avg_p = sum(total_precisions[n]) / max(len(total_precisions[n]), 1)
            results[f"bleu_{n}"] = avg_bp * avg_p

        # Combined BLEU (geometric mean of n-gram precisions)
        combined_precisions = []
        for n in range(1, max_n + 1):
            avg_p = sum(total_precisions[n]) / max(len(total_precisions[n]), 1)
            combined_precisions.append(max(avg_p, 1e-10))

        log_avg = sum(math.log(p) for p in combined_precisions) / max_n
        results["bleu"] = avg_bp * math.exp(log_avg)

        return results

    # ------------------------------------------------------------------
    # ROUGE
    # ------------------------------------------------------------------

    def compute_rouge(
        self,
        predictions: list[str],
        references: list[str],
        rouge_types: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L by default).

        Falls back to a simple longest-common-subsequence ROUGE-L
        implementation when the ``rouge-score`` package is not installed.

        Parameters
        ----------
        predictions:
            List of predicted strings.
        references:
            List of reference strings.
        rouge_types:
            ROUGE variants to compute. Defaults to
            ``["rouge1", "rouge2", "rougeL"]``.

        Returns
        -------
        dict
            Mapping from ROUGE type to F1 score,
            e.g. ``{"rouge1": 0.45, "rouge2": 0.30, "rougeL": 0.42}``.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length."
            )

        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]

        if _ROUGE_AVAILABLE:
            return self._rouge_lib(predictions, references, rouge_types)
        return self._rouge_simple(predictions, references)

    def _rouge_lib(
        self,
        predictions: list[str],
        references: list[str],
        rouge_types: list[str],
    ) -> dict[str, float]:
        """ROUGE via the rouge-score library."""
        scorer = _rouge_scorer_module.RougeScorer(rouge_types, use_stemmer=True)
        aggregated: dict[str, list[float]] = {rt: [] for rt in rouge_types}

        for pred, ref in zip(predictions, references, strict=False):
            scores = scorer.score(ref, pred)
            for rt in rouge_types:
                aggregated[rt].append(scores[rt].fmeasure)

        return {rt: sum(vals) / max(len(vals), 1) for rt, vals in aggregated.items()}

    def _rouge_simple(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Simple LCS-based ROUGE-L without the rouge-score package."""
        logger.warning("rouge-score not installed; using simplified ROUGE-L computation.")

        def _lcs_length(x: list[str], y: list[str]) -> int:
            m, n = len(x), len(y)
            if m == 0 or n == 0:
                return 0
            prev = [0] * (n + 1)
            for i in range(1, m + 1):
                curr = [0] * (n + 1)
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        curr[j] = prev[j - 1] + 1
                    else:
                        curr[j] = max(curr[j - 1], prev[j])
                prev = curr
            return prev[n]

        rouge_l_scores: list[float] = []
        rouge_1_scores: list[float] = []

        for pred, ref in zip(predictions, references, strict=False):
            pred_tok = _tokenize_simple(pred)
            ref_tok = _tokenize_simple(ref)

            # ROUGE-L
            lcs = _lcs_length(pred_tok, ref_tok)
            precision = lcs / max(len(pred_tok), 1)
            recall = lcs / max(len(ref_tok), 1)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            rouge_l_scores.append(f1)

            # ROUGE-1 (unigram overlap)
            pred_counter = Counter(pred_tok)
            ref_counter = Counter(ref_tok)
            overlap = sum((pred_counter & ref_counter).values())
            p1 = overlap / max(sum(pred_counter.values()), 1)
            r1 = overlap / max(sum(ref_counter.values()), 1)
            f1_1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 > 0 else 0.0
            rouge_1_scores.append(f1_1)

        return {
            "rouge1": sum(rouge_1_scores) / max(len(rouge_1_scores), 1),
            "rougeL": sum(rouge_l_scores) / max(len(rouge_l_scores), 1),
        }

    # ------------------------------------------------------------------
    # Exact Match
    # ------------------------------------------------------------------

    def compute_exact_match(
        self,
        predictions: list[str],
        references: list[str],
        normalize: bool = True,
    ) -> dict[str, float]:
        """Compute exact match accuracy.

        Parameters
        ----------
        predictions:
            List of predicted strings.
        references:
            List of reference strings.
        normalize:
            If *True*, apply text normalisation before comparison.

        Returns
        -------
        dict
            ``{"exact_match": float}`` in [0, 1].
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length."
            )

        matches = 0
        for pred, ref in zip(predictions, references, strict=False):
            p = _normalize_text(pred) if normalize else pred
            r = _normalize_text(ref) if normalize else ref
            if p == r:
                matches += 1

        return {"exact_match": matches / max(len(predictions), 1)}

    # ------------------------------------------------------------------
    # Token-level F1
    # ------------------------------------------------------------------

    def compute_f1(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Compute token-level F1 score (macro-averaged over samples).

        This is the standard QA F1 metric used in SQuAD-style evaluation:
        precision and recall are computed over whitespace-tokenized word
        overlap between prediction and reference.

        Parameters
        ----------
        predictions:
            List of predicted strings.
        references:
            List of reference strings.

        Returns
        -------
        dict
            ``{"f1": float, "precision": float, "recall": float}``
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length."
            )

        f1_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []

        for pred, ref in zip(predictions, references, strict=False):
            pred_tokens = _tokenize_simple(pred)
            ref_tokens = _tokenize_simple(ref)

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                continue

            precision = num_same / max(len(pred_tokens), 1)
            recall = num_same / max(len(ref_tokens), 1)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)

        n = max(len(predictions), 1)
        return {
            "f1": sum(f1_scores) / n,
            "precision": sum(precision_scores) / n,
            "recall": sum(recall_scores) / n,
        }

    # ------------------------------------------------------------------
    # Accuracy (classification)
    # ------------------------------------------------------------------

    def compute_accuracy(
        self,
        predictions: Sequence[str | int],
        references: Sequence[str | int],
        normalize_strings: bool = True,
    ) -> dict[str, float]:
        """Compute simple accuracy (fraction of exact matches).

        Works for both string labels and integer class indices.

        Parameters
        ----------
        predictions:
            Predicted labels.
        references:
            Ground-truth labels.
        normalize_strings:
            Normalise string inputs before comparison.

        Returns
        -------
        dict
            ``{"accuracy": float, "correct": int, "total": int}``
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length."
            )

        correct = 0
        for pred, ref in zip(predictions, references, strict=False):
            if isinstance(pred, str) and isinstance(ref, str) and normalize_strings:
                if _normalize_text(pred) == _normalize_text(ref):
                    correct += 1
            elif pred == ref:
                correct += 1

        total = max(len(predictions), 1)
        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": len(predictions),
        }

    # ------------------------------------------------------------------
    # Convenience: compute all applicable metrics
    # ------------------------------------------------------------------

    def compute_all(
        self,
        predictions: list[str],
        references: list[str],
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute multiple metrics at once.

        Parameters
        ----------
        predictions:
            Predicted strings.
        references:
            Reference strings.
        include:
            List of metric names to compute. If *None*, computes all
            text-based metrics (bleu, rouge, exact_match, f1, accuracy).

        Returns
        -------
        dict
            Merged dictionary of all computed metrics.
        """
        available = {
            "bleu": self.compute_bleu,
            "rouge": self.compute_rouge,
            "exact_match": self.compute_exact_match,
            "f1": self.compute_f1,
            "accuracy": self.compute_accuracy,
        }

        if include is None:
            include = list(available.keys())

        results: dict[str, Any] = {}
        for metric_name in include:
            if metric_name not in available:
                logger.warning("Unknown metric '%s'; skipping.", metric_name)
                continue
            try:
                metric_result = available[metric_name](predictions, references)
                results.update(metric_result)
            except Exception as exc:
                logger.error("Error computing '%s': %s", metric_name, exc)
                results[metric_name] = None

        return results
