"""Hybrid retrieval module for the llm-forge RAG pipeline.

Combines dense vector similarity search with BM25 keyword retrieval.
Scores are fused with a configurable alpha weight to balance semantic
and lexical matching.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from llm_forge.rag.embeddings import EmbeddingEngine
from llm_forge.rag.vectorstore import VectorStore
from llm_forge.utils.logging import get_logger

logger = get_logger("rag.retriever")

# ---------------------------------------------------------------------------
# Optional BM25 import
# ---------------------------------------------------------------------------

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A retrieved chunk with combined scoring information.

    Attributes
    ----------
    text : str
        The chunk text content.
    score : float
        Combined retrieval score.
    metadata : dict
        Chunk metadata (doc_id, chunk_index, etc.).
    source : str
        Retrieval source (``"dense"``, ``"bm25"``, or ``"hybrid"``).
    dense_score : float
        Score from dense retrieval (0.0 if not used).
    bm25_score : float
        Score from BM25 retrieval (0.0 if not used).
    """

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "dense"
    dense_score: float = 0.0
    bm25_score: float = 0.0


# ---------------------------------------------------------------------------
# Simple BM25 implementation (fallback)
# ---------------------------------------------------------------------------


class _SimpleBM25:
    """Minimal BM25 implementation for when rank_bm25 is unavailable.

    Implements the Okapi BM25 scoring function with standard parameter
    defaults (k1=1.5, b=0.75).
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)

        # Compute IDF for all terms
        self.idf: dict[str, float] = {}
        df: Counter = Counter()
        for doc in corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1

        for term, freq in df.items():
            # Standard BM25 IDF with smoothing
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        # Pre-compute term frequencies per document
        self.doc_tfs: list[Counter] = [Counter(doc) for doc in corpus]

    def get_scores(self, query: list[str]) -> list[float]:
        """Compute BM25 scores for all documents given a query."""
        scores = [0.0] * self.corpus_size

        for term in query:
            if term not in self.idf:
                continue

            idf = self.idf[term]

            for i in range(self.corpus_size):
                tf = self.doc_tfs[i].get(term, 0)
                dl = self.doc_lens[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                if denom > 0:
                    scores[i] += idf * (tf * (self.k1 + 1)) / denom

        return scores


# ---------------------------------------------------------------------------
# Tokenizer for BM25
# ---------------------------------------------------------------------------

_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization for BM25."""
    return _WORD_PATTERN.findall(text.lower())


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Hybrid retrieval engine combining dense and BM25 search.

    Dense retrieval uses vector similarity via a :class:`VectorStore`.
    BM25 provides keyword-based retrieval for complementary recall.
    Scores are combined using a weighted sum controlled by ``alpha``:

        ``combined = alpha * dense_score + (1 - alpha) * bm25_score``

    Parameters
    ----------
    vector_store : VectorStore
        Dense retrieval backend.
    embedding_engine : EmbeddingEngine
        Embedding model for query vectorisation.
    alpha : float
        Weight for dense retrieval scores. 1.0 = pure dense,
        0.0 = pure BM25, 0.5 = equal blend.
    enable_bm25 : bool
        Whether to enable BM25 retrieval alongside dense search.

    Examples
    --------
    >>> retriever = HybridRetriever(store, embedder, alpha=0.7)
    >>> results = retriever.retrieve("What is LoRA?", top_k=5)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_engine: EmbeddingEngine,
        alpha: float = 0.7,
        enable_bm25: bool = True,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.alpha = max(0.0, min(1.0, alpha))
        self.enable_bm25 = enable_bm25

        # BM25 index (built lazily or via build_bm25_index)
        self._bm25_index: Any = None
        self._bm25_documents: list[str] = []
        self._bm25_metadata: list[dict[str, Any]] = []

        if enable_bm25 and not _BM25_AVAILABLE:
            logger.info(
                "rank_bm25 not installed; using built-in BM25 implementation. "
                "For better performance, install: pip install rank-bm25"
            )

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def build_bm25_index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build or rebuild the BM25 index from documents.

        Parameters
        ----------
        documents:
            List of document texts to index.
        metadata:
            Optional metadata for each document.
        """
        if not documents:
            logger.warning("No documents provided for BM25 indexing.")
            return

        self._bm25_documents = documents
        self._bm25_metadata = metadata or [{} for _ in documents]

        tokenized = [_tokenize_for_bm25(doc) for doc in documents]

        if _BM25_AVAILABLE:
            self._bm25_index = BM25Okapi(tokenized)
        else:
            self._bm25_index = _SimpleBM25(tokenized)

        logger.info("BM25 index built with %d documents.", len(documents))

    def add_to_bm25_index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the BM25 index incrementally.

        Rebuilds the entire index (BM25 does not support incremental updates
        natively).

        Parameters
        ----------
        documents:
            New documents to add.
        metadata:
            Optional metadata for the new documents.
        """
        new_meta = metadata or [{} for _ in documents]
        self._bm25_documents.extend(documents)
        self._bm25_metadata.extend(new_meta)

        tokenized = [_tokenize_for_bm25(doc) for doc in self._bm25_documents]

        if _BM25_AVAILABLE:
            self._bm25_index = BM25Okapi(tokenized)
        else:
            self._bm25_index = _SimpleBM25(tokenized)

        logger.debug("BM25 index rebuilt with %d total documents.", len(self._bm25_documents))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using hybrid retrieval.

        Parameters
        ----------
        query:
            The user query.
        top_k:
            Number of final results to return.
        alpha:
            Override the default alpha for this query.
        dense_top_k:
            Number of candidates from dense retrieval.
            Defaults to ``top_k * 2``.
        bm25_top_k:
            Number of candidates from BM25.
            Defaults to ``top_k * 2``.

        Returns
        -------
        list[RetrievedChunk]
            Ranked retrieval results with combined scores.
        """
        effective_alpha = alpha if alpha is not None else self.alpha
        effective_alpha = max(0.0, min(1.0, effective_alpha))

        candidate_k = top_k * 2
        dk = dense_top_k or candidate_k
        bk = bm25_top_k or candidate_k

        # Dense retrieval
        dense_results = self._dense_retrieve(query, dk)

        # BM25 retrieval
        bm25_results: list[RetrievedChunk] = []
        if self.enable_bm25 and self._bm25_index is not None and effective_alpha < 1.0:
            bm25_results = self._bm25_retrieve(query, bk)

        # If only dense, return directly
        if not bm25_results or effective_alpha >= 1.0:
            return dense_results[:top_k]

        # If only BM25
        if effective_alpha <= 0.0:
            return bm25_results[:top_k]

        # Hybrid fusion
        fused = self._fuse_results(dense_results, bm25_results, effective_alpha)
        return fused[:top_k]

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def _dense_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve via dense vector similarity."""
        query_embedding = self.embedding_engine.embed_query(query)
        search_results = self.vector_store.search(query_embedding, top_k=top_k)

        chunks: list[RetrievedChunk] = []
        for sr in search_results:
            chunks.append(
                RetrievedChunk(
                    text=sr.text,
                    score=sr.score,
                    metadata=sr.metadata,
                    source="dense",
                    dense_score=sr.score,
                    bm25_score=0.0,
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    def _bm25_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve via BM25 keyword matching."""
        if self._bm25_index is None:
            return []

        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            return []

        if _BM25_AVAILABLE:
            scores = self._bm25_index.get_scores(query_tokens)
        else:
            scores = self._bm25_index.get_scores(query_tokens)

        # Get top-k indices
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]

        # Normalize BM25 scores to [0, 1]
        max_score = max(scores) if scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        chunks: list[RetrievedChunk] = []
        for idx, score in top_indices:
            if score <= 0 or idx >= len(self._bm25_documents):
                continue

            normalized_score = score / max_score

            chunks.append(
                RetrievedChunk(
                    text=self._bm25_documents[idx],
                    metadata=self._bm25_metadata[idx] if idx < len(self._bm25_metadata) else {},
                    score=normalized_score,
                    source="bm25",
                    dense_score=0.0,
                    bm25_score=normalized_score,
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    def _fuse_results(
        self,
        dense_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        alpha: float,
    ) -> list[RetrievedChunk]:
        """Fuse dense and BM25 results using weighted combination.

        Deduplicates by text content and combines scores:
            ``score = alpha * dense_score + (1 - alpha) * bm25_score``
        """
        # Build a map keyed by text content (or first 200 chars for efficiency)
        fused: dict[str, RetrievedChunk] = {}

        # Normalize dense scores to [0, 1]
        max_dense = max((r.dense_score for r in dense_results), default=1.0)
        if max_dense <= 0:
            max_dense = 1.0

        for chunk in dense_results:
            key = chunk.text[:200]
            norm_dense = chunk.dense_score / max_dense
            fused[key] = RetrievedChunk(
                text=chunk.text,
                score=alpha * norm_dense,
                metadata=chunk.metadata,
                source="hybrid",
                dense_score=norm_dense,
                bm25_score=0.0,
            )

        for chunk in bm25_results:
            key = chunk.text[:200]
            if key in fused:
                # Merge scores
                existing = fused[key]
                existing.bm25_score = chunk.bm25_score
                existing.score = alpha * existing.dense_score + (1 - alpha) * chunk.bm25_score
                existing.source = "hybrid"
            else:
                fused[key] = RetrievedChunk(
                    text=chunk.text,
                    score=(1 - alpha) * chunk.bm25_score,
                    metadata=chunk.metadata,
                    source="hybrid",
                    dense_score=0.0,
                    bm25_score=chunk.bm25_score,
                )

        # Sort by combined score
        results = sorted(fused.values(), key=lambda x: x.score, reverse=True)
        return results
