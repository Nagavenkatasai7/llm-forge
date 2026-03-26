"""Config-related tools for the Config Agent.
Extracted from the monolithic chat/tools.py. Each function delegates to the original.
"""
from __future__ import annotations


def write_config(output_path: str, config: dict) -> str:
    """Write a YAML training configuration file."""
    from llm_forge.chat.tools import _write_config
    return _write_config(output_path, config)


def validate_config(config_path: str) -> str:
    """Validate a YAML config against the llm-forge Pydantic schema."""
    from llm_forge.chat.tools import _validate_config
    return _validate_config(config_path)


def list_configs() -> str:
    """List available example/template configs."""
    from llm_forge.chat.tools import _list_configs
    return _list_configs()


def estimate_training(
    model_name: str,
    mode: str,
    num_samples: int = 1000,
    num_epochs: int = 1,
    batch_size: int = 4,
    seq_length: int = 2048,
) -> str:
    """Estimate training time, memory usage, and hardware requirements."""
    from llm_forge.chat.tools import _estimate_training
    return _estimate_training(model_name, mode, num_samples, num_epochs, batch_size, seq_length)


CONFIG_TOOL_DEFINITIONS = [
    {
        "name": "write_config",
        "description": "Write a YAML training config file.",
        "parameters": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["output_path", "config"],
        },
    },
    {
        "name": "validate_config",
        "description": "Validate a YAML config against the llm-forge schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string"},
            },
            "required": ["config_path"],
        },
    },
    {
        "name": "list_configs",
        "description": "List available example/template configs.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "estimate_training",
        "description": "Estimate training time, VRAM usage, and hardware fit.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "mode": {"type": "string", "enum": ["lora", "qlora", "full"]},
                "num_samples": {"type": "integer"},
                "num_epochs": {"type": "integer"},
                "batch_size": {"type": "integer"},
                "seq_length": {"type": "integer"},
            },
            "required": ["model_name", "mode"],
        },
    },
]
