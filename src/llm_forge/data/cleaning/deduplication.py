"""Three-tier deduplication: exact (SHA-256), fuzzy (MinHash LSH), semantic.

Provides exact hash-based deduplication, fuzzy near-duplicate detection via
MinHash Locality-Sensitive Hashing, and semantic deduplication using
sentence embeddings with K-means clustering.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from llm_forge.utils.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("data.cleaning.deduplication")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from datasketch import MinHash, MinHashLSH

    _DATASKETCH_AVAILABLE = True
except ImportError:
    _DATASKETCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Text normalization for hashing
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace for consistent hashing."""
    return _WHITESPACE_RE.sub(" ", text.lower().strip())


# ---------------------------------------------------------------------------
# Tier 1: Exact deduplication (SHA-256)
# ---------------------------------------------------------------------------


def exact_dedup(
    dataset: Dataset,
    text_field: str = "text",
) -> Dataset:
    """Remove exact-duplicate documents using SHA-256 hashes.

    Normalizes text (lowercase, collapse whitespace) before hashing so that
    trivially different whitespace does not prevent deduplication.

    Parameters
    ----------
    dataset:
        A HuggingFace ``Dataset`` instance.
    text_field:
        Column containing the text to deduplicate on.

    Returns
    -------
    Dataset
        Dataset with exact duplicates removed (first occurrence kept).
    """
    if text_field not in dataset.column_names:
        raise ValueError(
            f"Text field '{text_field}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    initial_count = len(dataset)
    logger.info("Running exact deduplication on %d records...", initial_count)

    seen_hashes: set[str] = set()

    def _is_unique(example: dict) -> bool:
        normalized = _normalize_text(example[text_field])
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    dataset = dataset.filter(
        _is_unique,
        desc="Exact deduplication (SHA-256)",
    )

    final_count = len(dataset)
    removed = initial_count - final_count
    logger.info(
        "Exact dedup complete: %d -> %d records (removed %d duplicates, %.1f%%).",
        initial_count,
        final_count,
        removed,
        (removed / initial_count * 100) if initial_count else 0,
    )
    return dataset


# ---------------------------------------------------------------------------
# Tier 2: Fuzzy deduplication (MinHash LSH)
# ---------------------------------------------------------------------------


def _build_minhash(
    text: str,
    num_perm: int = 128,
    shingle_size: int = 5,
) -> MinHash:
    """Build a MinHash signature from word-level shingles.

    Parameters
    ----------
    text:
        Document text.
    num_perm:
        Number of permutations for MinHash.
    shingle_size:
        Number of consecutive words per shingle.

    Returns
    -------
    MinHash
        The computed MinHash object.
    """
    words = text.lower().split()
    mh = MinHash(num_perm=num_perm)

    if len(words) < shingle_size:
        # If text is shorter than shingle size, use the whole text as one shingle
        shingle = " ".join(words)
        mh.update(shingle.encode("utf-8"))
    else:
        for i in range(len(words) - shingle_size + 1):
            shingle = " ".join(words[i : i + shingle_size])
            mh.update(shingle.encode("utf-8"))

    return mh


def fuzzy_dedup(
    dataset: Dataset,
    text_field: str = "text",
    threshold: float = 0.75,
    num_perm: int = 128,
    shingle_size: int = 5,
) -> Dataset:
    """Remove near-duplicate documents using MinHash Locality-Sensitive Hashing.

    Builds MinHash signatures from word-level shingles and uses LSH to find
    candidate near-duplicates above the Jaccard similarity threshold.  For
    each cluster of near-duplicates, only the first document (by dataset
    index) is retained.

    Parameters
    ----------
    dataset:
        A HuggingFace ``Dataset`` instance.
    text_field:
        Column containing the text.
    threshold:
        Jaccard similarity threshold.  Document pairs above this are
        considered near-duplicates.
    num_perm:
        Number of permutations for the MinHash signature.
    shingle_size:
        Word n-gram size for shingling.

    Returns
    -------
    Dataset
        Dataset with near-duplicates removed.

    Raises
    ------
    ImportError
        If ``datasketch`` is not installed.
    """
    if not _DATASKETCH_AVAILABLE:
        raise ImportError(
            "datasketch is required for fuzzy deduplication. "
            "Install it with: pip install datasketch"
        )

    if text_field not in dataset.column_names:
        raise ValueError(
            f"Text field '{text_field}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    initial_count = len(dataset)
    logger.info(
        "Running fuzzy deduplication on %d records "
        "(threshold=%.2f, num_perm=%d, shingle_size=%d)...",
        initial_count,
        threshold,
        num_perm,
        shingle_size,
    )

    # Phase 1: Build MinHash signatures for all documents
    logger.info("Building MinHash signatures...")
    minhashes: list[MinHash] = []
    for i in range(initial_count):
        text = dataset[i][text_field]
        mh = _build_minhash(text, num_perm=num_perm, shingle_size=shingle_size)
        minhashes.append(mh)

    # Phase 2: Insert into LSH index and find duplicates
    logger.info("Building LSH index and finding near-duplicates...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    duplicate_indices: set[int] = set()

    for idx, mh in enumerate(minhashes):
        if idx in duplicate_indices:
            continue

        # Query LSH for candidates
        key = f"doc_{idx}"
        try:
            # Check if any already-inserted document is a near-duplicate
            candidates = lsh.query(mh)
            if candidates:
                # This document is a near-duplicate of an existing one; mark it
                duplicate_indices.add(idx)
                continue
            # Insert into LSH
            lsh.insert(key, mh)
        except ValueError:
            # Key already exists (shouldn't happen, but handle gracefully)
            pass

    # Phase 3: Filter out duplicates
    keep_indices = [i for i in range(initial_count) if i not in duplicate_indices]
    dataset = dataset.select(keep_indices)

    final_count = len(dataset)
    removed = initial_count - final_count
    logger.info(
        "Fuzzy dedup complete: %d -> %d records (removed %d near-duplicates, %.1f%%).",
        initial_count,
        final_count,
        removed,
        (removed / initial_count * 100) if initial_count else 0,
    )
    return dataset


# ---------------------------------------------------------------------------
# Tier 3: Semantic deduplication (SemDeDup)
# ---------------------------------------------------------------------------


def semantic_dedup(
    dataset: Dataset,
    text_field: str = "text",
    threshold: float = 0.75,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    num_clusters: int | None = None,
) -> Dataset:
    """Remove semantically similar documents using sentence embeddings.

    Embeds all documents, clusters them with K-means, then within each
    cluster removes documents whose pairwise cosine similarity exceeds the
    threshold (keeping the first occurrence).

    Parameters
    ----------
    dataset:
        A HuggingFace ``Dataset`` instance.
    text_field:
        Column containing the text.
    threshold:
        Cosine similarity threshold.  Pairs above this within a cluster are
        considered semantic duplicates.
    model_name:
        Sentence-transformers model name for computing embeddings.
    batch_size:
        Batch size for encoding.
    num_clusters:
        Number of K-means clusters.  If *None*, defaults to
        ``max(10, len(dataset) // 100)``.

    Returns
    -------
    Dataset
        Dataset with semantic near-duplicates removed.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for semantic deduplication. "
            "Install it with: pip install sentence-transformers"
        )

    if text_field not in dataset.column_names:
        raise ValueError(
            f"Text field '{text_field}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    initial_count = len(dataset)
    if initial_count < 2:
        logger.info("Dataset has fewer than 2 records; skipping semantic dedup.")
        return dataset

    logger.info(
        "Running semantic deduplication on %d records (threshold=%.2f, model='%s')...",
        initial_count,
        threshold,
        model_name,
    )

    # Phase 1: Embed all documents
    logger.info("Computing sentence embeddings...")
    model = SentenceTransformer(model_name)
    texts = [dataset[i][text_field][:512] for i in range(initial_count)]  # truncate for speed
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )

    # Phase 2: K-means clustering
    if num_clusters is None:
        num_clusters = max(10, initial_count // 100)
    num_clusters = min(num_clusters, initial_count)

    logger.info("Clustering into %d groups...", num_clusters)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=3)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Phase 3: Within each cluster, find and remove semantic duplicates
    logger.info("Finding semantic duplicates within clusters...")
    duplicate_indices: set[int] = set()

    # Group indices by cluster
    cluster_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(cluster_labels):
        cluster_to_indices.setdefault(int(label), []).append(idx)

    for _cluster_id, indices in cluster_to_indices.items():
        if len(indices) < 2:
            continue

        # Get embeddings for this cluster
        cluster_embeds = embeddings[indices]

        # Pairwise cosine similarity (embeddings are already L2-normalized)
        sim_matrix = np.dot(cluster_embeds, cluster_embeds.T)

        # For each pair above threshold, mark the later one as duplicate
        for i in range(len(indices)):
            if indices[i] in duplicate_indices:
                continue
            for j in range(i + 1, len(indices)):
                if indices[j] in duplicate_indices:
                    continue
                if sim_matrix[i, j] > threshold:
                    duplicate_indices.add(indices[j])

    # Phase 4: Filter out duplicates
    keep_indices = [i for i in range(initial_count) if i not in duplicate_indices]
    dataset = dataset.select(keep_indices)

    final_count = len(dataset)
    removed = initial_count - final_count
    logger.info(
        "Semantic dedup complete: %d -> %d records (removed %d semantic duplicates, %.1f%%).",
        initial_count,
        final_count,
        removed,
        (removed / initial_count * 100) if initial_count else 0,
    )
    return dataset


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------


@dataclass
class DeduplicationStats:
    """Statistics from a deduplication run."""

    initial_count: int = 0
    final_count: int = 0
    exact_removed: int = 0
    fuzzy_removed: int = 0
    semantic_removed: int = 0

    @property
    def total_removed(self) -> int:
        return self.initial_count - self.final_count

    @property
    def removal_rate(self) -> float:
        if self.initial_count == 0:
            return 0.0
        return self.total_removed / self.initial_count


class Deduplicator:
    """Multi-tier deduplication orchestrator.

    Runs configured deduplication tiers in sequence: exact -> fuzzy -> semantic.
    Each tier operates on the output of the previous tier.

    Parameters
    ----------
    threshold:
        Jaccard/cosine similarity threshold for fuzzy/semantic tiers.
    num_perm:
        Number of MinHash permutations (fuzzy tier).
    shingle_size:
        Word n-gram size for shingling (fuzzy tier).
    semantic_model:
        Sentence-transformers model name (semantic tier).
    semantic_threshold:
        Cosine similarity threshold for semantic tier.  If *None*, uses
        the same value as *threshold*.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        num_perm: int = 128,
        shingle_size: int = 5,
        semantic_model: str = "all-MiniLM-L6-v2",
        semantic_threshold: float | None = None,
    ) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.semantic_model = semantic_model
        self.semantic_threshold = semantic_threshold or threshold

    def deduplicate(
        self,
        dataset: Dataset,
        tiers: list[str] | None = None,
        text_field: str = "text",
    ) -> tuple[Dataset, DeduplicationStats]:
        """Run configured deduplication tiers in sequence.

        Parameters
        ----------
        dataset:
            A HuggingFace ``Dataset`` instance.
        tiers:
            List of tier names to run.  Valid values: ``"exact"``,
            ``"fuzzy"``, ``"semantic"``.  Defaults to ``["exact", "fuzzy"]``.
        text_field:
            Column containing the text to deduplicate on.

        Returns
        -------
        tuple[Dataset, DeduplicationStats]
            The deduplicated dataset and statistics about what was removed.
        """
        if tiers is None:
            tiers = ["exact", "fuzzy"]

        valid_tiers = {"exact", "fuzzy", "semantic"}
        for tier in tiers:
            if tier not in valid_tiers:
                raise ValueError(
                    f"Unknown deduplication tier '{tier}'. Valid tiers: {sorted(valid_tiers)}"
                )

        stats = DeduplicationStats(initial_count=len(dataset))

        logger.info(
            "Starting deduplication pipeline: tiers=%s, %d records.",
            tiers,
            stats.initial_count,
        )

        for tier in tiers:
            count_before = len(dataset)

            if tier == "exact":
                dataset = exact_dedup(dataset, text_field=text_field)
                stats.exact_removed = count_before - len(dataset)

            elif tier == "fuzzy":
                dataset = fuzzy_dedup(
                    dataset,
                    text_field=text_field,
                    threshold=self.threshold,
                    num_perm=self.num_perm,
                    shingle_size=self.shingle_size,
                )
                stats.fuzzy_removed = count_before - len(dataset)

            elif tier == "semantic":
                dataset = semantic_dedup(
                    dataset,
                    text_field=text_field,
                    threshold=self.semantic_threshold,
                    model_name=self.semantic_model,
                )
                stats.semantic_removed = count_before - len(dataset)

        stats.final_count = len(dataset)

        logger.info(
            "Deduplication pipeline complete: %d -> %d records "
            "(total removed: %d, %.1f%%). "
            "Breakdown: exact=%d, fuzzy=%d, semantic=%d.",
            stats.initial_count,
            stats.final_count,
            stats.total_removed,
            stats.removal_rate * 100,
            stats.exact_removed,
            stats.fuzzy_removed,
            stats.semantic_removed,
        )

        return dataset, stats
