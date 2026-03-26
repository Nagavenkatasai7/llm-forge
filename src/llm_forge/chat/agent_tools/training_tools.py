"""Training-related tools for the Training Agent.
Extracted from the monolithic chat/tools.py.
"""
from __future__ import annotations


def start_training(config_path: str, verbose: bool = True) -> str:
    """Start training in a subprocess and attach background monitor."""
    from llm_forge.chat.tools import _start_training
    return _start_training(config_path, verbose)


def check_training_status() -> str:
    """Check if training is running."""
    from llm_forge.chat.tools import _check_training_status
    return _check_training_status()


def read_training_logs(output_dir: str, last_n_lines: int = 30) -> str:
    """Read recent training logs from output directory."""
    from llm_forge.chat.tools import _read_training_logs
    return _read_training_logs(output_dir, last_n_lines)


TRAINING_TOOL_DEFINITIONS = [
    {
        "name": "start_training",
        "description": "Start model training from a YAML config file.",
        "parameters": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string", "description": "Path to YAML config file"},
                "verbose": {"type": "boolean", "description": "Enable verbose output"},
            },
            "required": ["config_path"],
        },
    },
    {
        "name": "check_training_status",
        "description": "Check if training is currently running and get progress.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_training_logs",
        "description": "Read training logs and metrics from output directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string", "description": "Training output directory"},
                "last_n_lines": {"type": "integer", "description": "Number of recent log lines"},
            },
            "required": ["output_dir"],
        },
    },
]
