"""Shared pytest fixtures for the llm-forge test suite.

All fixtures use small/mock data suitable for CPU-only execution.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Temporary directory
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test outputs."""
    return tmp_path


# ---------------------------------------------------------------------------
# Sample data: Alpaca format
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_alpaca_data() -> list[dict[str, str]]:
    """Return 5 alpaca-format dicts."""
    return [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris.",
        },
        {
            "instruction": "Translate the following to Spanish.",
            "input": "Hello, how are you?",
            "output": "Hola, como estas?",
        },
        {
            "instruction": "Summarize the following text.",
            "input": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "output": "ML is an AI subset allowing systems to learn from data.",
        },
        {
            "instruction": "Write a haiku about programming.",
            "input": "",
            "output": "Lines of code converge\nSilicon thoughts awaken\nBugs hide in the night",
        },
        {
            "instruction": "Explain what a neural network is.",
            "input": "",
            "output": "A neural network is a computational model inspired by biological neurons that processes information in layers.",
        },
    ]


# ---------------------------------------------------------------------------
# Sample data: ShareGPT format
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_sharegpt_data() -> list[dict[str, Any]]:
    """Return 3 sharegpt-format dicts."""
    return [
        {
            "conversations": [
                {"from": "human", "value": "What is Python?"},
                {"from": "gpt", "value": "Python is a high-level programming language."},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "How does recursion work?"},
                {
                    "from": "gpt",
                    "value": "Recursion is a technique where a function calls itself.",
                },
                {"from": "human", "value": "Can you give an example?"},
                {
                    "from": "gpt",
                    "value": "Sure! A factorial function: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                },
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": "Explain gradient descent."},
                {
                    "from": "gpt",
                    "value": "Gradient descent is an optimization algorithm that minimizes a loss function by iteratively moving in the direction of steepest descent.",
                },
            ]
        },
    ]


# ---------------------------------------------------------------------------
# File fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_jsonl_file(tmp_path: Path, sample_alpaca_data: list[dict[str, str]]) -> Path:
    """Create a temp JSONL file with sample alpaca data."""
    filepath = tmp_path / "train.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for record in sample_alpaca_data:
            f.write(json.dumps(record) + "\n")
    return filepath


@pytest.fixture()
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a temp CSV file with sample data."""
    filepath = tmp_path / "train.csv"
    rows = [
        {
            "instruction": "Explain gravity.",
            "input": "",
            "output": "Gravity is a force of attraction.",
        },
        {"instruction": "What is 2+2?", "input": "", "output": "4"},
        {"instruction": "Name a fruit.", "input": "", "output": "Apple"},
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "input", "output"])
        writer.writeheader()
        writer.writerows(rows)
    return filepath


@pytest.fixture()
def sample_text_file(tmp_path: Path) -> Path:
    """Create a temp .txt file with paragraphs."""
    filepath = tmp_path / "sample.txt"
    content = (
        "This is the first paragraph about machine learning.\n\n"
        "This is the second paragraph about deep learning.\n\n"
        "This is the third paragraph about natural language processing."
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_config_dict() -> dict[str, Any]:
    """Return a minimal valid config dict for LLMForgeConfig."""
    return {
        "model": {
            "name": "meta-llama/Llama-3.2-1B",
        },
        "data": {
            "train_path": "tatsu-lab/alpaca",
        },
    }


@pytest.fixture()
def sample_config_yaml(tmp_path: Path, minimal_config_dict: dict[str, Any]) -> Path:
    """Create a temp YAML file with a valid config."""
    filepath = tmp_path / "config.yaml"
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(minimal_config_dict, f, default_flow_style=False)
    return filepath
