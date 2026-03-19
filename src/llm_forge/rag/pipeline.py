"""RAG pipeline orchestrator for llm-forge.

Orchestrates the full retrieval-augmented generation workflow:
document ingestion (chunking, embedding, indexing), retrieval, reranking,
and answer generation.  Supports incremental knowledge base updates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_forge.rag.chunking import Chunker, Document
from llm_forge.rag.embeddings import EmbeddingEngine
from llm_forge.rag.reranker import Reranker
from llm_forge.rag.retriever import HybridRetriever, RetrievedChunk
from llm_forge.rag.vectorstore import VectorStore, create_vectorstore
from llm_forge.utils.logging import get_logger

logger = get_logger("rag.pipeline")


# ---------------------------------------------------------------------------
# Configuration data class
# ---------------------------------------------------------------------------


@dataclass
class RAGPipelineConfig:
    """Configuration for the RAG pipeline.

    Attributes
    ----------
    chunk_strategy : str
        Chunking strategy (``"fixed"``, ``"semantic"``, ``"hierarchical"``,
        ``"adaptive"``).
    chunk_size : int
        Target chunk size in characters.
    chunk_overlap : int
        Overlap between chunks in characters.
    embedding_model : str
        Sentence-transformers model name for embeddings.
    vectorstore_backend : str
        Vector store backend (``"chromadb"`` or ``"faiss"``).
    persist_path : str or None
        Directory for persisting the vector store and metadata.
    top_k : int
        Number of chunks to retrieve per query.
    alpha : float
        Dense vs BM25 weight (1.0 = pure dense, 0.0 = pure BM25).
    enable_bm25 : bool
        Enable BM25 hybrid retrieval.
    reranker_model : str or None
        Cross-encoder model for reranking. *None* disables reranking.
    rerank_top_k : int or None
        Number of results after reranking. Defaults to ``top_k``.
    embedding_cache_dir : str or None
        Directory for caching embeddings.
    embedding_batch_size : int
        Batch size for embedding computation.
    """

    chunk_strategy: str = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "all-MiniLM-L6-v2"
    vectorstore_backend: str = "chromadb"
    persist_path: str | None = None
    top_k: int = 5
    alpha: float = 0.7
    enable_bm25: bool = True
    reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int | None = None
    embedding_cache_dir: str | None = None
    embedding_batch_size: int = 64


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".log"}
_JSON_EXTENSIONS = {".json", ".jsonl"}


def _load_documents_from_path(path: Path) -> list[Document]:
    """Recursively load documents from a file or directory.

    Supports plain text files (.txt, .md, .rst), JSON, and JSONL.
    """
    documents: list[Document] = []

    if path.is_file():
        documents.extend(_load_single_file(path))
    elif path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in (_TEXT_EXTENSIONS | _JSON_EXTENSIONS):
                documents.extend(_load_single_file(child))
    else:
        logger.warning("Path '%s' does not exist.", path)

    return documents


def _load_single_file(file_path: Path) -> list[Document]:
    """Load documents from a single file."""
    documents: list[Document] = []
    suffix = file_path.suffix.lower()

    try:
        if suffix in _TEXT_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append(
                    Document(
                        text=text,
                        doc_id=str(file_path),
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "file_type": suffix,
                        },
                    )
                )

        elif suffix == ".jsonl":
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", record.get("content", ""))
                        if isinstance(text, str) and text.strip():
                            doc_id = record.get("id", f"{file_path}:{line_num}")
                            metadata = {
                                k: v
                                for k, v in record.items()
                                if k not in ("text", "content")
                                and isinstance(v, (str, int, float, bool))
                            }
                            metadata["source"] = str(file_path)
                            documents.append(
                                Document(text=text, doc_id=str(doc_id), metadata=metadata)
                            )
                    except json.JSONDecodeError:
                        logger.debug(
                            "Skipping malformed JSON on line %d of %s.", line_num, file_path
                        )

        elif suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        text = item.get("text", item.get("content", ""))
                    elif isinstance(item, str):
                        text = item
                    else:
                        continue
                    if isinstance(text, str) and text.strip():
                        documents.append(
                            Document(
                                text=text,
                                doc_id=f"{file_path}:{i}",
                                metadata={"source": str(file_path)},
                            )
                        )
            elif isinstance(data, dict):
                text = data.get("text", data.get("content", json.dumps(data)))
                if isinstance(text, str) and text.strip():
                    documents.append(
                        Document(
                            text=text,
                            doc_id=str(file_path),
                            metadata={"source": str(file_path)},
                        )
                    )

    except Exception as exc:
        logger.error("Failed to load '%s': %s", file_path, exc)

    return documents


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Orchestrates document ingestion (chunking, embedding, indexing),
    retrieval (dense + BM25 hybrid), reranking, and context assembly.

    Parameters
    ----------
    config : RAGPipelineConfig or dict, optional
        Pipeline configuration. Defaults to sensible values.

    Examples
    --------
    >>> pipeline = RAGPipeline()
    >>> pipeline.build("./knowledge_base")
    >>> results = pipeline.query("What is gradient checkpointing?", top_k=5)
    >>> for chunk in results:
    ...     print(chunk.score, chunk.text[:80])
    """

    def __init__(
        self,
        config: RAGPipelineConfig | dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            self.config = RAGPipelineConfig()
        elif isinstance(config, dict):
            self.config = RAGPipelineConfig(**config)
        else:
            self.config = config

        # Components (initialized lazily or during build)
        self._chunker: Chunker | None = None
        self._embedding_engine: EmbeddingEngine | None = None
        self._vector_store: VectorStore | None = None
        self._retriever: HybridRetriever | None = None
        self._reranker: Reranker | None = None

        # State tracking
        self._is_built = False
        self._chunk_count = 0
        self._doc_count = 0

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Initialize pipeline components based on config."""
        # Chunker
        self._chunker = Chunker(
            strategy=self.config.chunk_strategy,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )

        # Embedding engine
        self._embedding_engine = EmbeddingEngine(
            model_name=self.config.embedding_model,
            cache_dir=self.config.embedding_cache_dir,
            batch_size=self.config.embedding_batch_size,
        )

        # Vector store
        self._vector_store = create_vectorstore(
            backend=self.config.vectorstore_backend,
            persist_path=self.config.persist_path,
            dimension=self._embedding_engine.dimension,
        )

        # Hybrid retriever
        self._retriever = HybridRetriever(
            vector_store=self._vector_store,
            embedding_engine=self._embedding_engine,
            alpha=self.config.alpha,
            enable_bm25=self.config.enable_bm25,
        )

        # Reranker
        if self.config.reranker_model:
            self._reranker = Reranker(model_name=self.config.reranker_model)
        else:
            self._reranker = None

    # ------------------------------------------------------------------
    # Build: process documents, chunk, embed, index
    # ------------------------------------------------------------------

    def build(
        self,
        knowledge_base_path: str | Path,
        config: RAGPipelineConfig | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a knowledge base: load, chunk, embed, and index documents.

        Parameters
        ----------
        knowledge_base_path:
            Path to a file or directory containing knowledge base documents.
        config:
            Optional config override for this build call.

        Returns
        -------
        dict
            Build statistics including document count, chunk count,
            and elapsed time.
        """
        if config is not None:
            if isinstance(config, dict):
                self.config = RAGPipelineConfig(**config)
            else:
                self.config = config

        start_time = time.time()
        logger.info("Building RAG pipeline from '%s'.", knowledge_base_path)

        # Initialize components
        self._init_components()

        # Load documents
        kb_path = Path(knowledge_base_path)
        documents = _load_documents_from_path(kb_path)
        if not documents:
            logger.warning("No documents found at '%s'.", kb_path)
            self._is_built = True
            return {"doc_count": 0, "chunk_count": 0, "elapsed_seconds": 0.0}

        logger.info("Loaded %d documents from '%s'.", len(documents), kb_path)
        self._doc_count = len(documents)

        # Chunk documents
        assert self._chunker is not None
        chunks = self._chunker.chunk_documents(documents)
        self._chunk_count = len(chunks)
        logger.info(
            "Chunked into %d chunks (strategy=%s, size=%d, overlap=%d).",
            len(chunks),
            self.config.chunk_strategy,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )

        if not chunks:
            self._is_built = True
            return {"doc_count": len(documents), "chunk_count": 0, "elapsed_seconds": 0.0}

        # Extract texts and metadata for indexing
        texts = [c.text for c in chunks]
        chunk_metadata = [
            {
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "byte_offset": c.byte_offset,
                **c.metadata,
            }
            for c in chunks
        ]

        # Compute embeddings
        assert self._embedding_engine is not None
        logger.info("Computing embeddings for %d chunks.", len(texts))
        embeddings = self._embedding_engine.embed_texts(texts)

        # Index in vector store
        assert self._vector_store is not None
        ids = self._vector_store.add(
            texts=texts,
            embeddings=embeddings,
            metadata=chunk_metadata,
        )
        logger.info("Indexed %d chunks in vector store.", len(ids))

        # Build BM25 index if enabled
        assert self._retriever is not None
        if self.config.enable_bm25:
            self._retriever.build_bm25_index(texts, chunk_metadata)

        # Persist
        if self.config.persist_path:
            self._vector_store.persist()

        elapsed = time.time() - start_time
        self._is_built = True

        stats = {
            "doc_count": len(documents),
            "chunk_count": len(chunks),
            "embedding_dimension": self._embedding_engine.dimension,
            "vectorstore_backend": self.config.vectorstore_backend,
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(
            "RAG pipeline built: %d docs, %d chunks in %.1f seconds.",
            len(documents),
            len(chunks),
            elapsed,
        )

        return stats

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: str | Path | list[str] | list[Document] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add documents to an existing knowledge base incrementally.

        Parameters
        ----------
        documents:
            New documents to add. Can be a file/directory path, a list
            of text strings, a list of :class:`Document` objects, or
            a list of dicts with ``text`` keys.

        Returns
        -------
        dict
            Update statistics.
        """
        if not self._is_built:
            raise RuntimeError("Pipeline has not been built yet. Call build() first.")

        assert self._chunker is not None
        assert self._embedding_engine is not None
        assert self._vector_store is not None
        assert self._retriever is not None

        start_time = time.time()

        # Resolve input format
        if isinstance(documents, (str, Path)):
            docs = _load_documents_from_path(Path(documents))
        elif isinstance(documents, list):
            docs = []
            for item in documents:
                if isinstance(item, str):
                    docs.append(Document(text=item))
                elif isinstance(item, Document):
                    docs.append(item)
                elif isinstance(item, dict):
                    docs.append(
                        Document(
                            text=item.get("text", ""),
                            doc_id=item.get("doc_id", ""),
                            metadata=item.get("metadata", {}),
                        )
                    )
        else:
            raise TypeError(f"Unsupported documents type: {type(documents)}")

        if not docs:
            logger.warning("No documents to add.")
            return {"added_docs": 0, "added_chunks": 0}

        # Chunk
        chunks = self._chunker.chunk_documents(docs)
        if not chunks:
            return {"added_docs": len(docs), "added_chunks": 0}

        texts = [c.text for c in chunks]
        chunk_metadata = [
            {
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "byte_offset": c.byte_offset,
                **c.metadata,
            }
            for c in chunks
        ]

        # Embed
        embeddings = self._embedding_engine.embed_texts(texts)

        # Index
        self._vector_store.add(
            texts=texts,
            embeddings=embeddings,
            metadata=chunk_metadata,
        )

        # Update BM25
        if self.config.enable_bm25:
            self._retriever.add_to_bm25_index(texts, chunk_metadata)

        # Persist
        if self.config.persist_path:
            self._vector_store.persist()

        self._doc_count += len(docs)
        self._chunk_count += len(chunks)
        elapsed = time.time() - start_time

        logger.info(
            "Incrementally added %d docs (%d chunks) in %.1f seconds.",
            len(docs),
            len(chunks),
            elapsed,
        )

        return {
            "added_docs": len(docs),
            "added_chunks": len(chunks),
            "total_docs": self._doc_count,
            "total_chunks": self._chunk_count,
            "elapsed_seconds": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Query: retrieve + rerank
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a question.

        Performs hybrid retrieval (dense + BM25) followed by optional
        cross-encoder reranking.

        Parameters
        ----------
        question:
            The user's question or query.
        top_k:
            Number of results to return. Defaults to config value.
        config:
            Optional query-time overrides (e.g. ``{"alpha": 0.5}``).

        Returns
        -------
        list[RetrievedChunk]
            Ranked chunks with scores and metadata.
        """
        if not self._is_built:
            raise RuntimeError("Pipeline has not been built yet. Call build() first.")

        assert self._retriever is not None

        effective_top_k = top_k or self.config.top_k
        query_alpha = self.config.alpha
        if config and "alpha" in config:
            query_alpha = config["alpha"]

        # Retrieve more candidates if reranking is enabled
        retrieve_k = effective_top_k
        if self._reranker is not None and self._reranker.is_available:
            retrieve_k = max(effective_top_k * 3, 15)

        # Hybrid retrieval
        candidates = self._retriever.retrieve(
            query=question,
            top_k=retrieve_k,
            alpha=query_alpha,
        )

        if not candidates:
            logger.debug("No retrieval results for query: '%s'.", question[:100])
            return []

        # Reranking
        if self._reranker is not None and self._reranker.is_available:
            rerank_k = self.config.rerank_top_k or effective_top_k
            results = self._reranker.rerank(
                query=question,
                chunks=candidates,
                top_k=rerank_k,
            )
        else:
            results = candidates[:effective_top_k]

        logger.debug(
            "Query '%s': retrieved %d candidates, returning %d results.",
            question[:60],
            len(candidates),
            len(results),
        )

        return results

    # ------------------------------------------------------------------
    # Context assembly (for feeding to a generative model)
    # ------------------------------------------------------------------

    def get_context(
        self,
        question: str,
        top_k: int | None = None,
        separator: str = "\n\n---\n\n",
        max_context_length: int | None = None,
    ) -> str:
        """Retrieve and assemble context for a generative model.

        Convenience wrapper around :meth:`query` that concatenates
        retrieved chunk texts into a single context string.

        Parameters
        ----------
        question:
            The user's question.
        top_k:
            Number of chunks to retrieve.
        separator:
            String to join chunks with.
        max_context_length:
            Maximum character length for the assembled context.

        Returns
        -------
        str
            Concatenated context string.
        """
        chunks = self.query(question, top_k=top_k)

        if not chunks:
            return ""

        texts = [chunk.text for chunk in chunks]
        context = separator.join(texts)

        if max_context_length is not None and len(context) > max_context_length:
            context = context[:max_context_length]
            # Try to end at a sentence boundary
            last_period = context.rfind(".")
            if last_period > max_context_length * 0.5:
                context = context[: last_period + 1]

        return context

    # ------------------------------------------------------------------
    # Pipeline state
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """Whether the pipeline has been built."""
        return self._is_built

    def stats(self) -> dict[str, Any]:
        """Return pipeline statistics.

        Returns
        -------
        dict
            Current state information.
        """
        store_count = self._vector_store.count() if self._vector_store else 0

        return {
            "is_built": self._is_built,
            "doc_count": self._doc_count,
            "chunk_count": self._chunk_count,
            "store_count": store_count,
            "config": {
                "chunk_strategy": self.config.chunk_strategy,
                "chunk_size": self.config.chunk_size,
                "embedding_model": self.config.embedding_model,
                "vectorstore_backend": self.config.vectorstore_backend,
                "enable_bm25": self.config.enable_bm25,
                "reranker_model": self.config.reranker_model,
            },
        }

    def clear(self) -> None:
        """Clear all indexed data and reset the pipeline."""
        if self._vector_store is not None:
            self._vector_store.delete()

        self._is_built = False
        self._chunk_count = 0
        self._doc_count = 0

        logger.info("RAG pipeline cleared.")
