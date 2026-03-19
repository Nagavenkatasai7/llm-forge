"""Tests for the universal DataLoader class.

Covers loading from JSONL, JSON (array), CSV, text files, directories,
max_samples limiting, and error handling for unsupported/missing files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from llm_forge.data.loader import SUPPORTED_EXTENSIONS, DataLoader

# ===================================================================
# JSONL loading
# ===================================================================


class TestLoadJSONL:
    """Test loading JSONL files."""

    def test_load_jsonl_file(self, sample_jsonl_file: Path) -> None:
        loader = DataLoader(path=str(sample_jsonl_file))
        ds = loader.load()
        assert isinstance(ds, Dataset)
        assert len(ds) == 5
        assert "instruction" in ds.column_names
        assert "output" in ds.column_names

    def test_load_jsonl_content(
        self, sample_jsonl_file: Path, sample_alpaca_data: list[dict[str, str]]
    ) -> None:
        loader = DataLoader(path=str(sample_jsonl_file))
        ds = loader.load()
        assert ds[0]["instruction"] == sample_alpaca_data[0]["instruction"]


# ===================================================================
# JSON (array) loading
# ===================================================================


class TestLoadJSON:
    """Test loading JSON files with array of objects."""

    def test_load_json_array(self, tmp_path: Path) -> None:
        data = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
        ]
        filepath = tmp_path / "data.json"
        filepath.write_text(json.dumps(data), encoding="utf-8")

        loader = DataLoader(path=str(filepath))
        ds = loader.load()
        assert isinstance(ds, Dataset)
        assert len(ds) == 2
        assert ds[0]["instruction"] == "Q1"

    def test_load_json_single_object(self, tmp_path: Path) -> None:
        data = {"instruction": "Only one", "output": "Answer"}
        filepath = tmp_path / "single.json"
        filepath.write_text(json.dumps(data), encoding="utf-8")

        loader = DataLoader(path=str(filepath))
        ds = loader.load()
        assert isinstance(ds, Dataset)
        assert len(ds) == 1


# ===================================================================
# CSV loading
# ===================================================================


class TestLoadCSV:
    """Test loading CSV files."""

    def test_load_csv_file(self, sample_csv_file: Path) -> None:
        loader = DataLoader(path=str(sample_csv_file))
        ds = loader.load()
        assert isinstance(ds, Dataset)
        assert len(ds) == 3
        assert "instruction" in ds.column_names

    def test_csv_has_correct_columns(self, sample_csv_file: Path) -> None:
        loader = DataLoader(path=str(sample_csv_file))
        ds = loader.load()
        assert set(ds.column_names) == {"instruction", "input", "output"}


# ===================================================================
# Text file loading
# ===================================================================


class TestLoadText:
    """Test loading plain text files."""

    def test_load_text_file(self, sample_text_file: Path) -> None:
        loader = DataLoader(path=str(sample_text_file))
        ds = loader.load()
        assert isinstance(ds, Dataset)
        # Text files are split on double newlines into paragraphs
        assert len(ds) == 3
        assert "text" in ds.column_names

    def test_text_content(self, sample_text_file: Path) -> None:
        loader = DataLoader(path=str(sample_text_file))
        ds = loader.load()
        assert "machine learning" in ds[0]["text"]


# ===================================================================
# Directory loading
# ===================================================================


class TestLoadDirectory:
    """Test loading from a directory of files."""

    def test_load_from_directory(self, tmp_path: Path) -> None:
        # Create multiple files in a directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # A JSONL file
        f1 = data_dir / "a.jsonl"
        f1.write_text(
            json.dumps({"instruction": "from jsonl", "output": "response"}) + "\n",
            encoding="utf-8",
        )

        # A text file
        f2 = data_dir / "b.txt"
        f2.write_text(
            "First paragraph from text.\n\nSecond paragraph from text.",
            encoding="utf-8",
        )

        loader = DataLoader(path=str(data_dir))
        ds = loader.load()
        assert isinstance(ds, Dataset)
        assert len(ds) >= 2  # at least 1 from JSONL + 2 from text

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = DataLoader(path=str(empty_dir))
        with pytest.raises(FileNotFoundError, match="No supported files"):
            loader.load()


# ===================================================================
# max_samples limit
# ===================================================================


class TestMaxSamples:
    """Test the max_samples parameter."""

    def test_max_samples_limits_output(self, sample_jsonl_file: Path) -> None:
        loader = DataLoader(path=str(sample_jsonl_file), max_samples=2)
        ds = loader.load()
        assert len(ds) == 2

    def test_max_samples_larger_than_data(self, sample_jsonl_file: Path) -> None:
        loader = DataLoader(path=str(sample_jsonl_file), max_samples=100)
        ds = loader.load()
        assert len(ds) == 5  # original data has 5 records


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    """Test error paths for unsupported formats and missing files."""

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("some content", encoding="utf-8")

        loader = DataLoader(path=str(bad_file))
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load()

    def test_nonexistent_file_falls_to_hf(self) -> None:
        """A nonexistent local path that is not an HF ID raises an error."""
        loader = DataLoader(path="this_file_definitely_does_not_exist.jsonl")
        # This will attempt HuggingFace loading and fail
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            loader.load()

    def test_supported_extensions_constant(self) -> None:
        """SUPPORTED_EXTENSIONS contains the expected core formats."""
        assert ".jsonl" in SUPPORTED_EXTENSIONS
        assert ".json" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".parquet" in SUPPORTED_EXTENSIONS
