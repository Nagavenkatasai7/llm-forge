"""Data-related tools for the Data Agent.
Extracted from the monolithic chat/tools.py. Each function delegates to the original.
"""
from __future__ import annotations


def scan_data(path: str) -> str:
    """Analyze a dataset file, directory, or HuggingFace dataset ID."""
    from llm_forge.chat.tools import _scan_data
    return _scan_data(path)


def detect_hardware() -> str:
    """Detect OS, CPU, RAM, GPU."""
    from llm_forge.chat.tools import _detect_hardware
    return _detect_hardware()


def search_huggingface(query: str, search_type: str = "model") -> str:
    """Search HuggingFace Hub for models or datasets."""
    from llm_forge.chat.tools import _search_huggingface
    return _search_huggingface(query, search_type)


def download_model(model_name: str, cache_dir: str | None = None) -> str:
    """Download a model from HuggingFace Hub."""
    from llm_forge.chat.tools import _download_model
    return _download_model(model_name, cache_dir)


def show_model_info(model_path: str) -> str:
    """Show model size, architecture, training config."""
    from llm_forge.chat.tools import _show_model_info
    return _show_model_info(model_path)


DATA_TOOL_DEFINITIONS = [
    {"name": "scan_data", "description": "Analyze a dataset file (JSONL, CSV, Parquet), directory, or HuggingFace dataset ID.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path, directory, or HuggingFace dataset ID"}}, "required": ["path"]}},
    {"name": "detect_hardware", "description": "Detect hardware: OS, CPU, RAM, GPU.", "parameters": {"type": "object", "properties": {}}},
    {"name": "search_huggingface", "description": "Search HuggingFace Hub for models or datasets.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "search_type": {"type": "string", "enum": ["model", "dataset"]}}, "required": ["query"]}},
    {"name": "download_model", "description": "Download a model from HuggingFace Hub.", "parameters": {"type": "object", "properties": {"model_name": {"type": "string", "description": "HuggingFace model ID"}, "cache_dir": {"type": "string"}}, "required": ["model_name"]}},
    {"name": "show_model_info", "description": "Show model size, architecture, and available checkpoints.", "parameters": {"type": "object", "properties": {"model_path": {"type": "string", "description": "Path to model or HuggingFace model ID"}}, "required": ["model_path"]}},
]
