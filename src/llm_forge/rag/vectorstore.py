"""Vector store backends for the llm-forge RAG pipeline.

Provides an abstract ``VectorStore`` interface with concrete implementations
for ChromaDB and FAISS.  Both backends are optional and handled via try/except
imports.  A factory function selects the appropriate backend at runtime.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from llm_forge.utils.logging import get_logger

logger = get_logger("rag.vectorstore")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _CHROMADB_AVAILABLE = True
except ImportError:
    _CHROMADB_AVAILABLE = False

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single search result from the vector store.

    Attributes
    ----------
    text : str
        The stored text content.
    score : float
        Similarity score (higher is more similar).
    metadata : dict
        Associated metadata.
    doc_id : str
        Document / chunk identifier.
    """

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract base class for vector store backends.

    Defines the common interface for adding, searching, deleting, and
    persisting vector data.
    """

    @abstractmethod
    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add texts and their embeddings to the store.

        Parameters
        ----------
        texts:
            Text content to store.
        embeddings:
            Corresponding embedding vectors.
        metadata:
            Optional metadata dicts, one per text.
        ids:
            Optional unique identifiers. Auto-generated if *None*.

        Returns
        -------
        list[str]
            The IDs assigned to the stored items.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for the most similar items.

        Parameters
        ----------
        query_embedding:
            The query vector.
        top_k:
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked search results (highest score first).
        """

    @abstractmethod
    def delete(self, ids: list[str] | None = None) -> int:
        """Delete items from the store.

        Parameters
        ----------
        ids:
            Specific IDs to delete. If *None*, deletes all items.

        Returns
        -------
        int
            Number of items deleted.
        """

    @abstractmethod
    def persist(self, path: str | Path | None = None) -> None:
        """Persist the store to disk.

        Parameters
        ----------
        path:
            Directory to save to. Uses the store's configured path if *None*.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of items in the store."""


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------


class ChromaDBStore(VectorStore):
    """Vector store backed by ChromaDB.

    Parameters
    ----------
    persist_path : str or Path, optional
        Directory for persistent storage. Uses in-memory mode if *None*.
    collection_name : str
        Name of the ChromaDB collection.

    Raises
    ------
    ImportError
        If ChromaDB is not installed.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        collection_name: str = "llm_forge_rag",
    ) -> None:
        if not _CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install 'llm-forge[rag]'"
            )

        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path:
            self._persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._persist_path),
            )
            logger.info("ChromaDB persistent store at '%s'.", self._persist_path)
        else:
            self._client = chromadb.Client()
            logger.info("ChromaDB in-memory store initialized.")

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add items to the ChromaDB collection."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length.")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadata is None:
            metadata = [{} for _ in texts]

        # ChromaDB requires metadata values to be str, int, float, or bool
        sanitized_metadata = []
        for m in metadata:
            sanitized = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                else:
                    sanitized[k] = str(v)
            sanitized_metadata.append(sanitized)

        # ChromaDB has a batch limit; process in chunks
        batch_size = 5000
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            self._collection.add(
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=sanitized_metadata[start:end],
                ids=ids[start:end],
            )

        logger.debug(
            "Added %d items to ChromaDB collection '%s'.", len(texts), self._collection_name
        )
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search the ChromaDB collection."""
        current_count = self._collection.count()
        if current_count == 0:
            return []

        effective_k = min(top_k, current_count)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids, strict=False):
            # ChromaDB returns distances; convert to similarity for cosine
            # cosine distance = 1 - cosine_similarity
            score = 1.0 - dist
            search_results.append(
                SearchResult(
                    text=doc or "",
                    score=score,
                    metadata=meta or {},
                    doc_id=doc_id,
                )
            )

        return search_results

    def delete(self, ids: list[str] | None = None) -> int:
        """Delete items from the ChromaDB collection."""
        if ids is None:
            # Delete entire collection and recreate
            count = self._collection.count()
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Deleted all %d items from ChromaDB collection.", count)
            return count

        if not ids:
            return 0

        self._collection.delete(ids=ids)
        logger.debug("Deleted %d items from ChromaDB collection.", len(ids))
        return len(ids)

    def persist(self, path: str | Path | None = None) -> None:
        """Persist the ChromaDB collection.

        For ``PersistentClient``, data is already persisted automatically.
        This method is provided for interface consistency.
        """
        if path and path != self._persist_path:
            logger.warning(
                "ChromaDB PersistentClient does not support saving to a different path. "
                "Data is persisted at the original path."
            )
        logger.debug("ChromaDB collection persisted.")

    def count(self) -> int:
        """Return number of items in the collection."""
        return self._collection.count()


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------


class FAISSStore(VectorStore):
    """Vector store backed by FAISS (faiss-cpu).

    Stores documents and metadata in parallel lists alongside the FAISS
    index.  Supports persistence via FAISS index serialization and a
    sidecar JSON metadata file.

    Parameters
    ----------
    dimension : int
        Embedding vector dimensionality.
    persist_path : str or Path, optional
        Directory for persistent storage.
    index_type : str
        FAISS index type. ``"flat"`` for brute-force, ``"ivfflat"``
        for approximate nearest neighbours.

    Raises
    ------
    ImportError
        If faiss-cpu is not installed.
    """

    def __init__(
        self,
        dimension: int = 384,
        persist_path: str | Path | None = None,
        index_type: str = "flat",
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install 'llm-forge[rag]'")

        self._dimension = dimension
        self._persist_path = Path(persist_path) if persist_path else None
        self._index_type = index_type

        # Document storage
        self._documents: list[str] = []
        self._metadata_store: list[dict[str, Any]] = []
        self._ids: list[str] = []

        # Try to load existing index
        loaded = False
        if self._persist_path:
            self._persist_path.mkdir(parents=True, exist_ok=True)
            loaded = self._try_load()

        if not loaded:
            self._index = self._create_index(dimension, index_type)
            logger.info("FAISS %s index created (dimension=%d).", index_type, dimension)

    def _create_index(self, dimension: int, index_type: str) -> Any:
        """Create a FAISS index of the specified type."""
        if index_type == "flat":
            return faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vecs)
        elif index_type == "ivfflat":
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
            return index
        else:
            logger.warning("Unknown index type '%s'; using flat.", index_type)
            return faiss.IndexFlatIP(dimension)

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add items to the FAISS index."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length.")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadata is None:
            metadata = [{} for _ in texts]

        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        # Train IVF index if needed
        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            if len(vectors) >= 100:
                self._index.train(vectors)
            else:
                # Not enough data for IVF; rebuild as flat
                logger.warning(
                    "Not enough data (%d) for IVF training; using flat index.", len(vectors)
                )
                self._index = faiss.IndexFlatIP(self._dimension)

        self._index.add(vectors)
        self._documents.extend(texts)
        self._metadata_store.extend(metadata)
        self._ids.extend(ids)

        logger.debug("Added %d items to FAISS index (total: %d).", len(texts), self._index.ntotal)
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search the FAISS index."""
        if self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        effective_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, effective_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._documents):
                continue
            results.append(
                SearchResult(
                    text=self._documents[idx],
                    score=float(score),
                    metadata=self._metadata_store[idx] if idx < len(self._metadata_store) else {},
                    doc_id=self._ids[idx] if idx < len(self._ids) else "",
                )
            )

        return results

    def delete(self, ids: list[str] | None = None) -> int:
        """Delete items from the FAISS index.

        FAISS does not natively support single-item deletion, so this
        rebuilds the index without the specified items.
        """
        if ids is None:
            # Delete everything
            count = self._index.ntotal
            self._index = self._create_index(self._dimension, self._index_type)
            self._documents.clear()
            self._metadata_store.clear()
            self._ids.clear()
            logger.info("Deleted all %d items from FAISS index.", count)
            return count

        if not ids:
            return 0

        ids_set = set(ids)
        keep_indices = [i for i, doc_id in enumerate(self._ids) if doc_id not in ids_set]
        deleted = len(self._ids) - len(keep_indices)

        if deleted == 0:
            return 0

        # Rebuild index with remaining items
        if keep_indices:
            vectors = np.array([self._index.reconstruct(i) for i in keep_indices], dtype=np.float32)
            new_docs = [self._documents[i] for i in keep_indices]
            new_meta = [self._metadata_store[i] for i in keep_indices]
            new_ids = [self._ids[i] for i in keep_indices]
        else:
            vectors = np.empty((0, self._dimension), dtype=np.float32)
            new_docs = []
            new_meta = []
            new_ids = []

        self._index = self._create_index(self._dimension, self._index_type)
        self._documents = new_docs
        self._metadata_store = new_meta
        self._ids = new_ids

        if len(vectors) > 0:
            if hasattr(self._index, "is_trained") and not self._index.is_trained:
                if len(vectors) >= 100:
                    self._index.train(vectors)
                else:
                    self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(vectors)

        logger.debug(
            "Deleted %d items from FAISS index (remaining: %d).", deleted, self._index.ntotal
        )
        return deleted

    def persist(self, path: str | Path | None = None) -> None:
        """Save the FAISS index and metadata to disk."""
        save_dir = Path(path) if path else self._persist_path
        if save_dir is None:
            logger.warning("No persist path configured; skipping save.")
            return

        save_dir.mkdir(parents=True, exist_ok=True)
        index_path = save_dir / "faiss_index.bin"
        meta_path = save_dir / "faiss_metadata.json"

        faiss.write_index(self._index, str(index_path))

        meta = {
            "documents": self._documents,
            "metadata": self._metadata_store,
            "ids": self._ids,
            "dimension": self._dimension,
            "index_type": self._index_type,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, default=str)

        logger.info("FAISS index persisted to '%s' (%d items).", save_dir, self._index.ntotal)

    def _try_load(self) -> bool:
        """Try to load an existing FAISS index from disk."""
        if self._persist_path is None:
            return False

        index_path = self._persist_path / "faiss_index.bin"
        meta_path = self._persist_path / "faiss_metadata.json"

        if not index_path.exists() or not meta_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(index_path))

            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            self._documents = meta.get("documents", [])
            self._metadata_store = meta.get("metadata", [])
            self._ids = meta.get("ids", [])
            self._dimension = meta.get("dimension", self._dimension)

            logger.info(
                "Loaded FAISS index from '%s' (%d items).",
                self._persist_path,
                self._index.ntotal,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)
            return False

    def count(self) -> int:
        """Return number of items in the index."""
        return self._index.ntotal


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_vectorstore(
    backend: str = "chromadb",
    persist_path: str | Path | None = None,
    dimension: int = 384,
    **kwargs: Any,
) -> VectorStore:
    """Create a vector store instance.

    Parameters
    ----------
    backend:
        Backend name: ``"chromadb"`` or ``"faiss"``.
    persist_path:
        Directory for persistent storage.
    dimension:
        Embedding dimension (used by FAISS).
    **kwargs:
        Additional keyword arguments passed to the backend constructor.

    Returns
    -------
    VectorStore
        An instance of the requested backend.

    Raises
    ------
    ValueError
        If the backend is unknown or its dependency is not installed.
    """
    backend_lower = backend.lower().strip()

    if backend_lower == "chromadb":
        if not _CHROMADB_AVAILABLE:
            raise ValueError(
                "ChromaDB is not installed. Install with: pip install 'llm-forge[rag]'. "
                "Alternatively, use backend='faiss'."
            )
        return ChromaDBStore(persist_path=persist_path, **kwargs)

    elif backend_lower == "faiss":
        if not _FAISS_AVAILABLE:
            raise ValueError(
                "FAISS is not installed. Install with: pip install 'llm-forge[rag]'. "
                "Alternatively, use backend='chromadb'."
            )
        return FAISSStore(dimension=dimension, persist_path=persist_path, **kwargs)

    else:
        available = []
        if _CHROMADB_AVAILABLE:
            available.append("chromadb")
        if _FAISS_AVAILABLE:
            available.append("faiss")

        raise ValueError(
            f"Unknown vector store backend '{backend}'. "
            f"Available backends: {available or ['none (install llm-forge[rag])']}"
        )
