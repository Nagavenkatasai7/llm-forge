"""Export-related tools for the Export Agent."""
from __future__ import annotations


def export_model(model_path: str, fmt: str, output_dir: str | None = None) -> str:
    """Export model to GGUF, safetensors, or ONNX."""
    from llm_forge.chat.tools import _export_model
    return _export_model(model_path, fmt, output_dir)


def deploy_to_ollama(model_path: str, model_name: str, system_prompt: str | None = None, quantization: str = "Q4_K_M") -> str:
    """Export to GGUF and deploy to Ollama."""
    from llm_forge.chat.tools import _deploy_to_ollama
    return _deploy_to_ollama(model_path, model_name, system_prompt, quantization)


def deploy_to_huggingface(model_path: str, repo_name: str, description: str = "", private: bool = False) -> str:
    """Upload model to HuggingFace Hub."""
    from llm_forge.chat.tools import _deploy_to_huggingface
    return _deploy_to_huggingface(model_path, repo_name, description, private)


def show_model_info(model_path: str) -> str:
    """Show model metadata."""
    from llm_forge.chat.tools import _show_model_info
    return _show_model_info(model_path)


EXPORT_TOOL_DEFINITIONS = [
    {
        "name": "export_model",
        "description": "Export model to GGUF, safetensors, or ONNX.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "fmt": {"type": "string", "enum": ["gguf", "safetensors", "onnx"]},
                "output_dir": {"type": "string"},
            },
            "required": ["model_path", "fmt"],
        },
    },
    {
        "name": "deploy_to_ollama",
        "description": "Export to GGUF and deploy to Ollama for local chat.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "model_name": {"type": "string"},
                "system_prompt": {"type": "string"},
                "quantization": {"type": "string"},
            },
            "required": ["model_path", "model_name"],
        },
    },
    {
        "name": "deploy_to_huggingface",
        "description": "Upload model to HuggingFace Hub.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "repo_name": {"type": "string"},
                "description": {"type": "string"},
                "private": {"type": "boolean"},
            },
            "required": ["model_path", "repo_name"],
        },
    },
    {
        "name": "show_model_info",
        "description": "Show model size, architecture, and checkpoints.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
            },
            "required": ["model_path"],
        },
    },
]
