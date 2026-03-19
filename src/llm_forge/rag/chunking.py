"""Text chunking strategies for the llm-forge RAG pipeline.

Provides fixed-size, semantic, hierarchical, and adaptive chunking with
configurable overlap.  All strategies return chunks annotated with metadata
(doc_id, chunk_index, byte_offset).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("rag.chunking")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ChunkStrategy(str, Enum):
    """Supported chunking strategies."""

    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class Chunk:
    """A text chunk with metadata.

    Attributes
    ----------
    text : str
        The chunk content.
    doc_id : str
        Identifier for the source document.
    chunk_index : int
        Zero-based index of this chunk within its document.
    byte_offset : int
        Byte offset of this chunk's start within the original document.
    metadata : dict
        Additional metadata (strategy used, overlap info, etc.).
    """

    text: str
    doc_id: str
    chunk_index: int
    byte_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A source document for chunking.

    Attributes
    ----------
    text : str
        Full document text.
    doc_id : str
        Unique identifier. Auto-generated if not provided.
    metadata : dict
        Optional document-level metadata.
    """

    text: str
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Regex for sentence boundary detection. Handles common abbreviations and
# avoids splitting on Mr., Mrs., Dr., U.S., etc.
_ABBREVIATIONS = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|inc|ltd|co|dept|univ|govt)"
_SENTENCE_BOUNDARY = re.compile(
    rf"(?<!\b{_ABBREVIATIONS})"  # Negative lookbehind for abbreviations
    r"(?<=[.!?])"  # After sentence-ending punctuation
    r"(?:\s+)"  # Followed by whitespace
    r"(?=[A-Z\"])",  # Next char is uppercase or opening quote
    re.UNICODE,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based boundary detection."""
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (by double newline or more)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


# ---------------------------------------------------------------------------
# Chunker class
# ---------------------------------------------------------------------------


class Chunker:
    """Text chunking engine with multiple strategy support.

    Implements the strategy pattern for chunking, allowing fixed-size,
    semantic (sentence-aware), hierarchical (paragraph then sentence),
    and adaptive strategies.

    Parameters
    ----------
    strategy : str or ChunkStrategy
        Default chunking strategy.
    chunk_size : int
        Target chunk size in characters.
    overlap : int
        Number of characters to overlap between consecutive chunks.

    Examples
    --------
    >>> chunker = Chunker(strategy="semantic", chunk_size=500, overlap=50)
    >>> chunks = chunker.chunk_text("Long document text here...", doc_id="doc1")
    >>> len(chunks)
    3
    """

    def __init__(
        self,
        strategy: str | ChunkStrategy = ChunkStrategy.SEMANTIC,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> None:
        if isinstance(strategy, str):
            strategy = ChunkStrategy(strategy.lower())
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size}).")

        self._strategy_map = {
            ChunkStrategy.FIXED: self._chunk_fixed,
            ChunkStrategy.SEMANTIC: self._chunk_semantic,
            ChunkStrategy.HIERARCHICAL: self._chunk_hierarchical,
            ChunkStrategy.ADAPTIVE: self._chunk_adaptive,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        doc_id: str = "",
        strategy: str | ChunkStrategy | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[Chunk]:
        """Chunk a single text string.

        Parameters
        ----------
        text:
            Input text to chunk.
        doc_id:
            Document identifier for metadata.
        strategy:
            Override the default strategy for this call.
        chunk_size:
            Override the default chunk size for this call.
        overlap:
            Override the default overlap for this call.

        Returns
        -------
        list[Chunk]
            List of chunks with metadata.
        """
        if not text or not text.strip():
            return []

        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]

        strat = self.strategy
        if strategy is not None:
            strat = ChunkStrategy(strategy) if isinstance(strategy, str) else strategy

        cs = chunk_size or self.chunk_size
        ov = overlap if overlap is not None else self.overlap

        chunk_fn = self._strategy_map.get(strat)
        if chunk_fn is None:
            raise ValueError(f"Unknown chunking strategy: {strat}")

        chunks = chunk_fn(text, doc_id, cs, ov)

        logger.debug(
            "Chunked doc '%s' into %d chunks (strategy=%s, size=%d, overlap=%d).",
            doc_id,
            len(chunks),
            strat.value,
            cs,
            ov,
        )

        return chunks

    def chunk_documents(
        self,
        documents: Sequence[Document | dict[str, Any] | str],
        strategy: str | ChunkStrategy | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[Chunk]:
        """Chunk multiple documents in batch.

        Parameters
        ----------
        documents:
            Sequence of :class:`Document` objects, dicts with ``text`` and
            optional ``doc_id``/``metadata`` keys, or plain strings.
        strategy:
            Override the default strategy.
        chunk_size:
            Override the default chunk size.
        overlap:
            Override the default overlap.

        Returns
        -------
        list[Chunk]
            All chunks from all documents.
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            if isinstance(doc, str):
                doc = Document(text=doc)
            elif isinstance(doc, dict):
                doc = Document(
                    text=doc.get("text", ""),
                    doc_id=doc.get("doc_id", ""),
                    metadata=doc.get("metadata", {}),
                )

            chunks = self.chunk_text(
                text=doc.text,
                doc_id=doc.doc_id,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            # Propagate document metadata
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)

            all_chunks.extend(chunks)

        logger.info("Chunked %d documents into %d total chunks.", len(documents), len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------
    # Strategy: Fixed-size character splits
    # ------------------------------------------------------------------

    def _chunk_fixed(self, text: str, doc_id: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Split text into fixed-size character chunks with overlap."""
        chunks: list[Chunk] = []
        step = chunk_size - overlap
        text.encode("utf-8")
        idx = 0
        pos = 0

        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            chunk_text = text[pos:end]

            byte_offset = len(text[:pos].encode("utf-8"))

            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=idx,
                    byte_offset=byte_offset,
                    metadata={"strategy": "fixed"},
                )
            )

            pos += step
            idx += 1

            if end == len(text):
                break

        return chunks

    # ------------------------------------------------------------------
    # Strategy: Semantic (sentence-boundary-aware)
    # ------------------------------------------------------------------

    def _chunk_semantic(self, text: str, doc_id: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Split text at sentence boundaries, respecting chunk_size."""
        sentences = _split_sentences(text)

        if not sentences:
            return self._chunk_fixed(text, doc_id, chunk_size, overlap)

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0
        idx = 0

        for sent in sentences:
            sent_len = len(sent)

            # If a single sentence exceeds chunk_size, fall back to fixed split
            if sent_len > chunk_size and not current_sentences:
                sub_chunks = self._chunk_fixed(sent, doc_id, chunk_size, overlap)
                for sc in sub_chunks:
                    sc.chunk_index = idx
                    sc.metadata["strategy"] = "semantic"
                    sc.metadata["note"] = "long_sentence_fallback"
                    chunks.append(sc)
                    idx += 1
                continue

            if current_len + sent_len + (1 if current_sentences else 0) > chunk_size:
                # Flush current chunk
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    byte_offset = text.find(current_sentences[0])
                    if byte_offset == -1:
                        byte_offset = 0
                    byte_offset = len(text[:byte_offset].encode("utf-8"))

                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            doc_id=doc_id,
                            chunk_index=idx,
                            byte_offset=byte_offset,
                            metadata={"strategy": "semantic"},
                        )
                    )
                    idx += 1

                    # Overlap: keep trailing sentences that fit within overlap chars
                    overlap_sentences: list[str] = []
                    overlap_len = 0
                    for s in reversed(current_sentences):
                        if overlap_len + len(s) + 1 <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break

                    current_sentences = overlap_sentences
                    current_len = sum(len(s) for s in current_sentences) + max(
                        0, len(current_sentences) - 1
                    )

            current_sentences.append(sent)
            current_len += sent_len + (1 if len(current_sentences) > 1 else 0)

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            byte_offset = text.find(current_sentences[0])
            if byte_offset == -1:
                byte_offset = 0
            byte_offset = len(text[:byte_offset].encode("utf-8"))

            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=idx,
                    byte_offset=byte_offset,
                    metadata={"strategy": "semantic"},
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Strategy: Hierarchical (paragraph -> sentence)
    # ------------------------------------------------------------------

    def _chunk_hierarchical(
        self, text: str, doc_id: str, chunk_size: int, overlap: int
    ) -> list[Chunk]:
        """Two-level chunking: first by paragraph, then by sentence within.

        Paragraphs that fit within ``chunk_size`` are kept whole.
        Larger paragraphs are split at sentence boundaries.
        """
        paragraphs = _split_paragraphs(text)

        if not paragraphs:
            return self._chunk_semantic(text, doc_id, chunk_size, overlap)

        chunks: list[Chunk] = []
        idx = 0

        for para in paragraphs:
            if len(para) <= chunk_size:
                # Paragraph fits in one chunk
                byte_offset = text.find(para)
                if byte_offset == -1:
                    byte_offset = 0
                byte_offset = len(text[:byte_offset].encode("utf-8"))

                chunks.append(
                    Chunk(
                        text=para,
                        doc_id=doc_id,
                        chunk_index=idx,
                        byte_offset=byte_offset,
                        metadata={"strategy": "hierarchical", "level": "paragraph"},
                    )
                )
                idx += 1
            else:
                # Split paragraph into sentence-level chunks
                para_chunks = self._chunk_semantic(para, doc_id, chunk_size, overlap)
                for pc in para_chunks:
                    pc.chunk_index = idx
                    pc.metadata["strategy"] = "hierarchical"
                    pc.metadata["level"] = "sentence"
                    chunks.append(pc)
                    idx += 1

        return chunks

    # ------------------------------------------------------------------
    # Strategy: Adaptive (adjusts size based on content density)
    # ------------------------------------------------------------------

    def _chunk_adaptive(self, text: str, doc_id: str, chunk_size: int, overlap: int) -> list[Chunk]:
        """Adaptively adjust chunk size based on content density.

        Dense text (high ratio of information-bearing words) gets smaller
        chunks; sparse/repetitive text gets larger chunks.
        """
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            paragraphs = [text]

        chunks: list[Chunk] = []
        idx = 0

        for para in paragraphs:
            density = self._compute_density(para)
            # Scale chunk size inversely with density
            # density ~ 1.0 -> use chunk_size * 0.6
            # density ~ 0.0 -> use chunk_size * 1.4
            scale = 1.4 - 0.8 * density
            adaptive_size = max(100, int(chunk_size * scale))
            adaptive_overlap = max(0, int(overlap * scale))
            adaptive_overlap = min(adaptive_overlap, adaptive_size - 1)

            if len(para) <= adaptive_size:
                byte_offset = text.find(para)
                if byte_offset == -1:
                    byte_offset = 0
                byte_offset = len(text[:byte_offset].encode("utf-8"))

                chunks.append(
                    Chunk(
                        text=para,
                        doc_id=doc_id,
                        chunk_index=idx,
                        byte_offset=byte_offset,
                        metadata={
                            "strategy": "adaptive",
                            "density": round(density, 3),
                            "adaptive_size": adaptive_size,
                        },
                    )
                )
                idx += 1
            else:
                sub_chunks = self._chunk_semantic(para, doc_id, adaptive_size, adaptive_overlap)
                for sc in sub_chunks:
                    sc.chunk_index = idx
                    sc.metadata["strategy"] = "adaptive"
                    sc.metadata["density"] = round(density, 3)
                    sc.metadata["adaptive_size"] = adaptive_size
                    chunks.append(sc)
                    idx += 1

        return chunks

    @staticmethod
    def _compute_density(text: str) -> float:
        """Estimate content density of text.

        Returns a score between 0 and 1 where higher values indicate
        denser, more information-rich content.

        Heuristics used:
        - Unique word ratio
        - Average word length
        - Presence of numbers and technical terms
        """
        words = text.lower().split()
        if not words:
            return 0.5

        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)

        # Average word length (longer words tend to be more technical)
        avg_word_len = sum(len(w) for w in words) / len(words)
        length_score = min(avg_word_len / 10.0, 1.0)

        # Numeric content ratio
        numeric_count = sum(1 for w in words if any(c.isdigit() for c in w))
        numeric_ratio = numeric_count / len(words)

        # Combine signals
        density = 0.4 * unique_ratio + 0.35 * length_score + 0.25 * numeric_ratio
        return min(max(density, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    strategy: str = "semantic",
    chunk_size: int = 512,
    overlap: int = 64,
    doc_id: str = "",
) -> list[Chunk]:
    """Convenience function to chunk a single text.

    Parameters
    ----------
    text:
        Input text to chunk.
    strategy:
        Chunking strategy name (``"fixed"``, ``"semantic"``,
        ``"hierarchical"``, ``"adaptive"``).
    chunk_size:
        Target chunk size in characters.
    overlap:
        Overlap between consecutive chunks in characters.
    doc_id:
        Document identifier.

    Returns
    -------
    list[Chunk]
        Chunks with metadata.
    """
    chunker = Chunker(strategy=strategy, chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_text(text, doc_id=doc_id)


def chunk_documents(
    documents: Sequence[Document | dict[str, Any] | str],
    strategy: str = "semantic",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Convenience function to chunk multiple documents.

    Parameters
    ----------
    documents:
        Sequence of documents (Document objects, dicts, or strings).
    strategy:
        Chunking strategy name.
    chunk_size:
        Target chunk size in characters.
    overlap:
        Overlap between consecutive chunks.

    Returns
    -------
    list[Chunk]
        All chunks from all documents.
    """
    chunker = Chunker(strategy=strategy, chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_documents(documents)
