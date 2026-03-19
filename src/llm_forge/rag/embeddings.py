"""Embedding engine for the llm-forge RAG pipeline.

Uses sentence-transformers for dense embeddings with support for
batch processing, progress reporting, and disk-based caching.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from llm_forge.utils.logging import get_logger

logger = get_logger("rag.embeddings")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------


class EmbeddingEngine:
    """Dense embedding engine for text vectorisation.

    Uses ``sentence-transformers`` models for generating embeddings.
    Supports batch processing with configurable batch size, progress
    logging, and disk-based embedding caching.

    Parameters
    ----------
    model_name : str
        Sentence-transformers model name or path.
    device : str, optional
        Device for inference. Auto-detected if *None*.
    cache_dir : str or Path, optional
        Directory for caching computed embeddings.  When *None*,
        caching is disabled.
    batch_size : int
        Batch size for embedding computation.
    normalize : bool
        Whether to L2-normalise the output embeddings.

    Examples
    --------
    >>> engine = EmbeddingEngine()
    >>> vectors = engine.embed_texts(["Hello world", "How are you?"])
    >>> len(vectors)
    2
    >>> len(vectors[0])  # embedding dimension
    384
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Any = None
        self._dimension: int | None = None

        # Device detection
        if device is not None:
            self.device = device
        else:
            self.device = self._detect_device()

        # Cache setup
        self._cache_dir: Path | None = None
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Embedding cache directory: %s", self._cache_dir)

        if not _ST_AVAILABLE:
            logger.warning(
                "sentence-transformers not installed. Install with: "
                "pip install 'llm-forge[rag]'.  Falling back to random embeddings."
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

    def _load_model(self) -> None:
        """Load the sentence-transformers model on first use."""
        if self._model is not None:
            return

        if not _ST_AVAILABLE:
            logger.warning("Using random fallback embeddings (dimension=384).")
            self._dimension = 384
            return

        logger.info(
            "Loading embedding model '%s' on device '%s'.",
            self.model_name,
            self.device,
        )
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded. Dimension: %d.", self._dimension)

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        self._load_model()
        assert self._dimension is not None
        return self._dimension

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for a text string."""
        content = f"{self.model_name}::{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _get_cached(self, text: str) -> list[float] | None:
        """Retrieve a cached embedding if available."""
        if self._cache_dir is None:
            return None

        key = self._cache_key(text)
        cache_file = self._cache_dir / f"{key}.npy"

        if cache_file.exists():
            try:
                arr = np.load(cache_file)
                return arr.tolist()
            except Exception:
                # Corrupt cache file; silently skip
                return None
        return None

    def _set_cached(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in the cache."""
        if self._cache_dir is None:
            return

        key = self._cache_key(text)
        cache_file = self._cache_dir / f"{key}.npy"

        try:
            np.save(cache_file, np.array(embedding, dtype=np.float32))
        except Exception as exc:
            logger.debug("Failed to cache embedding: %s", exc)

    # ------------------------------------------------------------------
    # Embedding methods
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Compute embeddings for a list of texts.

        Parameters
        ----------
        texts:
            List of text strings to embed.
        batch_size:
            Override the default batch size for this call.
        show_progress:
            Log progress during batch processing.

        Returns
        -------
        list[list[float]]
            List of embedding vectors, one per input text.
        """
        self._load_model()

        if not texts:
            return []

        bs = batch_size or self.batch_size

        # Check cache for all texts
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        cached_count = len(texts) - len(uncached_indices)
        if cached_count > 0:
            logger.debug("Retrieved %d/%d embeddings from cache.", cached_count, len(texts))

        # Compute uncached embeddings
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            uncached_embeddings = self._compute_embeddings(uncached_texts, bs, show_progress)

            for idx, emb in zip(uncached_indices, uncached_embeddings, strict=False):
                results[idx] = emb
                self._set_cached(texts[idx], emb)

        # All results should be populated now
        return [r for r in results if r is not None]

    def embed_query(self, query: str) -> list[float]:
        """Compute embedding for a single query string.

        Parameters
        ----------
        query:
            The query text.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        results = self.embed_texts([query], show_progress=False)
        return results[0]

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_embeddings(
        self,
        texts: list[str],
        batch_size: int,
        show_progress: bool,
    ) -> list[list[float]]:
        """Compute embeddings using the model (or fallback)."""
        if self._model is None:
            # Fallback to random embeddings
            return self._random_embeddings(texts)

        all_embeddings: list[list[float]] = []
        total = len(texts)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]

            embeddings = self._model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )

            if isinstance(embeddings, np.ndarray):
                all_embeddings.extend(embeddings.tolist())
            else:
                all_embeddings.extend([e.tolist() for e in embeddings])

            if show_progress and total > batch_size:
                progress = min(end, total)
                logger.info(
                    "Embedded %d / %d texts (%.0f%%).",
                    progress,
                    total,
                    progress / total * 100,
                )

        return all_embeddings

    def _random_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic pseudo-random embeddings as fallback.

        Uses a hash of the text to seed the random generator, ensuring
        the same text always maps to the same embedding.
        """
        dim = self._dimension or 384
        embeddings = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(dim).astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_cache(self) -> int:
        """Remove all cached embeddings.

        Returns
        -------
        int
            Number of cache files removed.
        """
        if self._cache_dir is None:
            return 0

        count = 0
        for f in self._cache_dir.glob("*.npy"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass

        logger.info("Cleared %d cached embeddings.", count)
        return count

    def cache_stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns
        -------
        dict
            ``{"num_files": int, "total_size_mb": float, "cache_dir": str}``
        """
        if self._cache_dir is None:
            return {"num_files": 0, "total_size_mb": 0.0, "cache_dir": None}

        files = list(self._cache_dir.glob("*.npy"))
        total_bytes = sum(f.stat().st_size for f in files)

        return {
            "num_files": len(files),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "cache_dir": str(self._cache_dir),
        }
