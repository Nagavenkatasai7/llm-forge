"""Tests for the RAG module.

Covers RAGConfig validation (chunk strategies, metadata) and
skips embedding/vectorstore tests if deps are not available.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_forge.config.schema import RAGConfig

# ===================================================================
# RAGConfig validation
# ===================================================================


class TestRAGConfig:
    """Test RAGConfig pydantic model."""

    def test_defaults(self) -> None:
        rag = RAGConfig()
        assert rag.enabled is False
        assert rag.chunk_strategy == "recursive"
        assert rag.chunk_size == 512
        assert rag.chunk_overlap == 64
        assert rag.vectorstore == "chromadb"
        assert rag.top_k == 5

    def test_fixed_chunk_strategy(self) -> None:
        rag = RAGConfig(chunk_strategy="fixed")
        assert rag.chunk_strategy == "fixed"

    def test_semantic_chunk_strategy(self) -> None:
        rag = RAGConfig(chunk_strategy="semantic")
        assert rag.chunk_strategy == "semantic"

    def test_sentence_chunk_strategy(self) -> None:
        rag = RAGConfig(chunk_strategy="sentence")
        assert rag.chunk_strategy == "sentence"

    def test_invalid_chunk_strategy(self) -> None:
        with pytest.raises(ValidationError):
            RAGConfig(chunk_strategy="invalid_strategy")

    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic", "sentence"])
    def test_all_chunk_strategies_valid(self, strategy: str) -> None:
        rag = RAGConfig(chunk_strategy=strategy)
        assert rag.chunk_strategy == strategy


# ===================================================================
# Chunk size / overlap validation
# ===================================================================


class TestChunkValidation:
    """Test chunk_size and chunk_overlap constraints."""

    def test_chunk_overlap_less_than_size(self) -> None:
        rag = RAGConfig(chunk_size=512, chunk_overlap=64)
        assert rag.chunk_overlap < rag.chunk_size

    def test_chunk_overlap_equals_size_raises(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            RAGConfig(chunk_size=256, chunk_overlap=256)

    def test_chunk_overlap_exceeds_size_raises(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            RAGConfig(chunk_size=256, chunk_overlap=300)

    def test_chunk_size_min_bound(self) -> None:
        with pytest.raises(ValidationError):
            RAGConfig(chunk_size=32)  # min is 64

    def test_chunk_size_max_bound(self) -> None:
        with pytest.raises(ValidationError):
            RAGConfig(chunk_size=10000)  # max is 8192

    def test_chunk_overlap_zero_is_valid(self) -> None:
        rag = RAGConfig(chunk_size=512, chunk_overlap=0)
        assert rag.chunk_overlap == 0


# ===================================================================
# Chunk metadata
# ===================================================================


class TestChunkMetadata:
    """Test that RAGConfig carries metadata fields correctly."""

    def test_embedding_model(self) -> None:
        rag = RAGConfig(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        assert "MiniLM" in rag.embedding_model

    def test_custom_embedding_model(self) -> None:
        rag = RAGConfig(embedding_model="custom/my-embedder")
        assert rag.embedding_model == "custom/my-embedder"

    def test_knowledge_base_path(self) -> None:
        rag = RAGConfig(knowledge_base_path="/data/kb")
        assert rag.knowledge_base_path == "/data/kb"

    def test_similarity_threshold(self) -> None:
        rag = RAGConfig(similarity_threshold=0.8)
        assert rag.similarity_threshold == 0.8

    def test_similarity_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            RAGConfig(similarity_threshold=1.5)
        with pytest.raises(ValidationError):
            RAGConfig(similarity_threshold=-0.1)


# ===================================================================
# Vectorstore options
# ===================================================================


class TestVectorstoreOptions:
    """Test vectorstore backend choices."""

    @pytest.mark.parametrize("backend", ["chromadb", "faiss", "qdrant", "weaviate"])
    def test_valid_vectorstores(self, backend: str) -> None:
        rag = RAGConfig(vectorstore=backend)
        assert rag.vectorstore == backend

    def test_invalid_vectorstore(self) -> None:
        with pytest.raises(ValidationError):
            RAGConfig(vectorstore="not_a_vectorstore")


# ===================================================================
# Hybrid search and reranker
# ===================================================================


class TestAdvancedRAGOptions:
    """Test hybrid search and reranker config."""

    def test_hybrid_search_default_false(self) -> None:
        rag = RAGConfig()
        assert rag.hybrid_search is False

    def test_hybrid_search_enabled(self) -> None:
        rag = RAGConfig(hybrid_search=True)
        assert rag.hybrid_search is True

    def test_reranker_model_default_none(self) -> None:
        rag = RAGConfig()
        assert rag.reranker_model is None

    def test_reranker_model_custom(self) -> None:
        rag = RAGConfig(reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert "cross-encoder" in rag.reranker_model

    def test_top_k(self) -> None:
        rag = RAGConfig(top_k=10)
        assert rag.top_k == 10

    def test_top_k_min_bound(self) -> None:
        with pytest.raises(ValidationError):
            RAGConfig(top_k=0)
