"""Cross-encoder reranking module for the llm-forge RAG pipeline.

Uses cross-encoder models via sentence-transformers to rescore
query-document pairs and re-sort retrieval results for improved
precision.
"""

from __future__ import annotations

from typing import Any

from llm_forge.rag.retriever import RetrievedChunk
from llm_forge.utils.logging import get_logger

logger = get_logger("rag.reranker")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import CrossEncoder

    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Cross-encoder reranker for improving retrieval precision.

    Scores each (query, document) pair independently using a cross-encoder
    model and re-sorts the results by the new scores.  Falls back to
    passthrough (no reranking) when sentence-transformers is not installed
    or the model cannot be loaded.

    Parameters
    ----------
    model_name : str
        Cross-encoder model name or path.
    device : str, optional
        Device for inference. Auto-detected if *None*.
    max_length : int
        Maximum input sequence length for the cross-encoder.
    batch_size : int
        Batch size for cross-encoder inference.

    Examples
    --------
    >>> reranker = Reranker()
    >>> reranked = reranker.rerank("What is LoRA?", chunks, top_k=3)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Any = None
        self._available = _CROSS_ENCODER_AVAILABLE

        # Device detection
        if device is not None:
            self.device = device
        else:
            self.device = self._detect_device()

        if not self._available:
            logger.warning(
                "sentence-transformers not installed; reranking is disabled. "
                "Install with: pip install 'llm-forge[rag]'"
            )

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Load the cross-encoder model. Returns True if successful."""
        if self._model is not None:
            return True

        if not self._available:
            return False

        try:
            logger.info(
                "Loading cross-encoder model '%s' on device '%s'.",
                self.model_name,
                self.device,
            )
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            logger.info("Cross-encoder model loaded successfully.")
            return True
        except Exception as exc:
            logger.error(
                "Failed to load cross-encoder model '%s': %s. Reranking will be disabled.",
                self.model_name,
                exc,
            )
            self._available = False
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks using cross-encoder scoring.

        Parameters
        ----------
        query:
            The user query.
        chunks:
            Retrieved chunks to rerank.
        top_k:
            Number of top results to return after reranking.
            If *None*, returns all chunks reranked.

        Returns
        -------
        list[RetrievedChunk]
            Reranked chunks with updated scores, sorted by
            cross-encoder score (highest first).
        """
        if not chunks:
            return []

        if top_k is not None and top_k <= 0:
            return []

        # Try to load model; fall back to passthrough
        if not self._load_model():
            logger.debug("Reranking unavailable; returning chunks in original order.")
            if top_k is not None:
                return chunks[:top_k]
            return chunks

        # Prepare query-document pairs
        pairs = [(query, chunk.text) for chunk in chunks]

        # Score in batches
        scores = self._score_pairs(pairs)

        # Create reranked chunks with updated scores
        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores, strict=False):
            reranked_chunk = RetrievedChunk(
                text=chunk.text,
                score=float(score),
                metadata={**chunk.metadata, "original_score": chunk.score},
                source=chunk.source,
                dense_score=chunk.dense_score,
                bm25_score=chunk.bm25_score,
            )
            reranked.append(reranked_chunk)

        # Sort by cross-encoder score (highest first)
        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        logger.debug(
            "Reranked %d chunks (returning top %d).",
            len(chunks),
            len(reranked),
        )

        return reranked

    def _score_pairs(self, pairs: list[tuple]) -> list[float]:
        """Score query-document pairs using the cross-encoder."""
        all_scores: list[float] = []

        for start in range(0, len(pairs), self.batch_size):
            end = min(start + self.batch_size, len(pairs))
            batch = pairs[start:end]

            batch_scores = self._model.predict(
                batch,
                show_progress_bar=False,
            )

            if hasattr(batch_scores, "tolist"):
                all_scores.extend(batch_scores.tolist())
            else:
                all_scores.extend(list(batch_scores))

        return all_scores

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Whether reranking is available (model can be loaded)."""
        return self._available

    def score(self, query: str, text: str) -> float:
        """Score a single query-document pair.

        Parameters
        ----------
        query:
            The query string.
        text:
            The document text.

        Returns
        -------
        float
            Cross-encoder relevance score.
        """
        if not self._load_model():
            return 0.0

        scores = self._model.predict(
            [(query, text)],
            show_progress_bar=False,
        )

        score = scores[0] if hasattr(scores, "__getitem__") else float(scores)
        return float(score)
