"""Research-related tools for the Research Agent."""
from __future__ import annotations
import json


def search_huggingface(query: str, search_type: str = "model") -> str:
    """Search HuggingFace Hub for models or datasets."""
    from llm_forge.chat.tools import _search_huggingface
    return _search_huggingface(query, search_type)


def download_model(model_name: str, cache_dir: str | None = None) -> str:
    """Download a model from HuggingFace Hub."""
    from llm_forge.chat.tools import _download_model
    return _download_model(model_name, cache_dir)


def install_dependencies(feature: str) -> str:
    """Check and install missing dependencies."""
    from llm_forge.chat.tools import _install_dependencies
    return _install_dependencies(feature)


def web_search(query: str) -> str:
    """Search the web for information. (Placeholder — real implementation in Phase 4.)"""
    return json.dumps({
        "status": "not_implemented",
        "message": "Web search is not yet implemented. Use search_huggingface for HF Hub searches.",
        "query": query,
    })


def read_url(url: str) -> str:
    """Fetch and parse a web page. (Placeholder — real implementation in Phase 4.)"""
    return json.dumps({
        "status": "not_implemented",
        "message": "URL reading is not yet implemented.",
        "url": url,
    })


RESEARCH_TOOL_DEFINITIONS = [
    {
        "name": "search_huggingface",
        "description": "Search HuggingFace Hub for models or datasets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "search_type": {"type": "string", "enum": ["model", "dataset"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "download_model",
        "description": "Download a model from HuggingFace Hub.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "cache_dir": {"type": "string"},
            },
            "required": ["model_name"],
        },
    },
    {
        "name": "install_dependencies",
        "description": "Check and install missing dependencies for a feature.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "enum": ["training", "evaluation", "serving", "cleaning", "rag", "chat", "all"],
                },
            },
            "required": ["feature"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information (papers, docs, tutorials).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_url",
        "description": "Fetch and parse a web page to extract text content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
]
