"""Tests for the DataPreprocessor class.

Covers alpaca, sharegpt, completion, and custom formatting,
dataset splitting, and system prompt injection.
"""

from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset

from llm_forge.data.preprocessor import (
    DataPreprocessor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alpaca_dataset(data: list[dict[str, str]]) -> Dataset:
    return Dataset.from_list(data)


def _make_sharegpt_dataset(data: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(data)


# ===================================================================
# Alpaca formatting
# ===================================================================


class TestAlpacaFormatting:
    """Test alpaca instruction-response formatting."""

    def test_basic_alpaca_no_input(self, sample_alpaca_data: list[dict[str, str]]) -> None:
        """Records without input use the no-input template."""
        ds = _make_alpaca_dataset(sample_alpaca_data[:1])  # first item has empty input
        pp = DataPreprocessor(format_type="alpaca")
        formatted = pp.format_dataset(ds)

        assert "text" in formatted.column_names
        text = formatted[0]["text"]
        assert "### Instruction:" in text
        assert "### Response:" in text
        assert "capital of France" in text
        # No input section should be present
        assert "### Input:" not in text

    def test_alpaca_with_context_input(self, sample_alpaca_data: list[dict[str, str]]) -> None:
        """Records with non-empty input include the input section."""
        ds = _make_alpaca_dataset(sample_alpaca_data[1:2])  # second item has input
        pp = DataPreprocessor(format_type="alpaca")
        formatted = pp.format_dataset(ds)

        text = formatted[0]["text"]
        assert "### Instruction:" in text
        assert "### Input:" in text
        assert "### Response:" in text
        assert "Hello, how are you?" in text

    def test_alpaca_output_preserved(self, sample_alpaca_data: list[dict[str, str]]) -> None:
        ds = _make_alpaca_dataset(sample_alpaca_data)
        pp = DataPreprocessor(format_type="alpaca")
        formatted = pp.format_dataset(ds)

        assert len(formatted) == 5
        for i in range(5):
            assert sample_alpaca_data[i]["output"] in formatted[i]["text"]


# ===================================================================
# ShareGPT formatting
# ===================================================================


class TestShareGPTFormatting:
    """Test sharegpt multi-turn conversation formatting."""

    def test_basic_sharegpt(self, sample_sharegpt_data: list[dict[str, Any]]) -> None:
        ds = _make_sharegpt_dataset(sample_sharegpt_data[:1])
        pp = DataPreprocessor(format_type="sharegpt")
        formatted = pp.format_dataset(ds)

        text = formatted[0]["text"]
        assert "Human:" in text
        assert "Assistant:" in text
        assert "Python" in text

    def test_multiturn_sharegpt(self, sample_sharegpt_data: list[dict[str, Any]]) -> None:
        ds = _make_sharegpt_dataset(sample_sharegpt_data[1:2])
        pp = DataPreprocessor(format_type="sharegpt")
        formatted = pp.format_dataset(ds)

        text = formatted[0]["text"]
        # Should have multiple turns
        assert text.count("Human:") == 2
        assert text.count("Assistant:") == 2

    def test_sharegpt_with_system(self, sample_sharegpt_data: list[dict[str, Any]]) -> None:
        """ShareGPT data with a system turn includes it."""
        ds = _make_sharegpt_dataset(sample_sharegpt_data[2:3])
        pp = DataPreprocessor(format_type="sharegpt")
        formatted = pp.format_dataset(ds)

        text = formatted[0]["text"]
        assert "System:" in text
        assert "helpful assistant" in text


# ===================================================================
# Completion formatting
# ===================================================================


class TestCompletionFormatting:
    """Test plain text completion formatting."""

    def test_completion_with_text_field(self) -> None:
        ds = Dataset.from_list(
            [
                {"text": "Hello world"},
                {"text": "Goodbye world"},
            ]
        )
        pp = DataPreprocessor(format_type="completion")
        formatted = pp.format_dataset(ds)

        assert formatted[0]["text"] == "Hello world"
        assert formatted[1]["text"] == "Goodbye world"

    def test_completion_falls_back_to_input_field(self) -> None:
        ds = Dataset.from_list([{"instruction": "Do something"}])
        pp = DataPreprocessor(format_type="completion", input_field="instruction")
        formatted = pp.format_dataset(ds)

        assert "Do something" in formatted[0]["text"]


# ===================================================================
# Custom formatting
# ===================================================================


class TestCustomFormatting:
    """Test custom field mapping."""

    def test_custom_with_all_fields(self) -> None:
        ds = Dataset.from_list(
            [
                {
                    "question": "What is AI?",
                    "context": "AI is...",
                    "answer": "Artificial Intelligence",
                },
            ]
        )
        pp = DataPreprocessor(
            format_type="custom",
            input_field="question",
            output_field="answer",
            context_field="context",
        )
        formatted = pp.format_dataset(ds)

        text = formatted[0]["text"]
        assert "What is AI?" in text
        assert "AI is..." in text
        assert "Artificial Intelligence" in text

    def test_custom_without_context(self) -> None:
        ds = Dataset.from_list(
            [
                {"question": "What is AI?", "answer": "Artificial Intelligence"},
            ]
        )
        pp = DataPreprocessor(
            format_type="custom",
            input_field="question",
            output_field="answer",
            context_field=None,
        )
        formatted = pp.format_dataset(ds)
        assert "What is AI?" in formatted[0]["text"]


# ===================================================================
# System prompt
# ===================================================================


class TestSystemPrompt:
    """Test system prompt injection."""

    def test_alpaca_with_system_prompt(self, sample_alpaca_data: list[dict[str, str]]) -> None:
        ds = _make_alpaca_dataset(sample_alpaca_data[:1])
        pp = DataPreprocessor(
            format_type="alpaca",
            system_prompt="You are a helpful assistant.",
        )
        formatted = pp.format_dataset(ds)
        text = formatted[0]["text"]
        assert "### System:" in text
        assert "helpful assistant" in text

    def test_sharegpt_with_system_prompt(self, sample_sharegpt_data: list[dict[str, Any]]) -> None:
        ds = _make_sharegpt_dataset(sample_sharegpt_data[:1])
        pp = DataPreprocessor(
            format_type="sharegpt",
            system_prompt="Be concise.",
        )
        formatted = pp.format_dataset(ds)
        text = formatted[0]["text"]
        assert "System: Be concise." in text

    def test_completion_with_system_prompt(self) -> None:
        ds = Dataset.from_list([{"text": "Some content"}])
        pp = DataPreprocessor(
            format_type="completion",
            system_prompt="System instruction here.",
        )
        formatted = pp.format_dataset(ds)
        text = formatted[0]["text"]
        assert "System instruction here." in text
        assert "Some content" in text


# ===================================================================
# Dataset splitting
# ===================================================================


class TestDatasetSplitting:
    """Test train/eval split."""

    def test_split_basic(self, sample_alpaca_data: list[dict[str, str]]) -> None:
        ds = _make_alpaca_dataset(sample_alpaca_data)
        pp = DataPreprocessor()
        train, test = pp.split_dataset(ds, test_size=0.4, seed=42)

        assert len(train) + len(test) == len(ds)
        assert len(test) >= 1
        assert len(train) >= 1

    def test_split_zero_test_size(self, sample_alpaca_data: list[dict[str, str]]) -> None:
        ds = _make_alpaca_dataset(sample_alpaca_data)
        pp = DataPreprocessor()
        train, test = pp.split_dataset(ds, test_size=0, seed=42)

        assert len(train) == len(ds)
        assert len(test) == 0

    def test_split_reproducible(self, sample_alpaca_data: list[dict[str, str]]) -> None:
        ds = _make_alpaca_dataset(sample_alpaca_data)
        pp = DataPreprocessor()
        train1, test1 = pp.split_dataset(ds, test_size=0.2, seed=42)
        train2, test2 = pp.split_dataset(ds, test_size=0.2, seed=42)

        assert list(train1["instruction"]) == list(train2["instruction"])


# ===================================================================
# Unknown format
# ===================================================================


class TestUnknownFormat:
    """Test that an unknown format raises ValueError."""

    def test_unknown_format_raises(self) -> None:
        ds = Dataset.from_list([{"text": "hello"}])
        pp = DataPreprocessor(format_type="nonexistent_format")
        with pytest.raises(ValueError, match="Unknown format"):
            pp.format_dataset(ds)
