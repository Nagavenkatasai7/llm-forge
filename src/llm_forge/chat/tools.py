"""Tool definitions and implementations for the LLM Forge chat assistant."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
from pathlib import Path

from llm_forge.chat.execution import (
    EXECUTION_TOOL_NAMES,
    EXECUTION_TOOLS,
    execute_execution_tool,
)
from llm_forge.chat.training_monitor import TrainingMonitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Active training monitor (module-level singleton)
# TODO: Move to ChatEngine instance for multi-session safety
# ---------------------------------------------------------------------------
_active_monitor: TrainingMonitor | None = None

# ---------------------------------------------------------------------------
# Tool Definitions (JSON schema for Claude API)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "detect_hardware",
        "description": "Detect the user's hardware: GPU type, VRAM, RAM, CPU, OS. Call this before making any model or training recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "scan_data",
        "description": "Scan a file or directory to understand the user's training data. Returns format, sample count, size, and a preview of the first few records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path, directory path, or HuggingFace dataset ID (e.g., 'tatsu-lab/alpaca')",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_config",
        "description": "Write a YAML training configuration file. The config controls the entire training pipeline.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path": {
                    "type": "string",
                    "description": "Where to save the config file (e.g., './config.yaml')",
                },
                "config": {
                    "type": "object",
                    "description": "The configuration dictionary with model, data, training, and other sections",
                },
            },
            "required": ["output_path", "config"],
        },
    },
    {
        "name": "validate_config",
        "description": "Validate a YAML config file against the llm-forge schema. Returns validation errors or confirms the config is valid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to the YAML config file to validate",
                }
            },
            "required": ["config_path"],
        },
    },
    {
        "name": "start_training",
        "description": "Start the training pipeline with a given config file. This launches training in the background.",
        "input_schema": {
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to the YAML config file",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose logging",
                    "default": True,
                },
            },
            "required": ["config_path"],
        },
    },
    {
        "name": "check_training_status",
        "description": "Check if training is currently running and get the latest metrics (loss, step, ETA).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "export_model",
        "description": "Export a trained model to a specific format (GGUF for Ollama, safetensors for HuggingFace, ONNX for production).",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the trained model directory",
                },
                "format": {
                    "type": "string",
                    "enum": ["gguf", "safetensors", "onnx"],
                    "description": "Export format",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Where to save the exported model",
                },
            },
            "required": ["model_path", "format"],
        },
    },
    {
        "name": "list_configs",
        "description": "List available example configs that come with llm-forge. Useful for showing users what's possible.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_huggingface",
        "description": (
            "Search HuggingFace Hub for models or datasets. Model results carry "
            "the real parameter count and a per-method memory-fit verdict for "
            "THIS machine, so use it before recommending any base model. Dataset "
            "results carry a ground_truth score (benchmark registration, held-out "
            "splits, linked paper, annotation source) plus license -- use it when "
            "the user needs a dataset whose answers can be verified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'finance instruction dataset', 'llama 1b')",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["models", "datasets"],
                    "description": "Whether to search for models or datasets",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many results to return (default 5, max 20)",
                },
            },
            "required": ["query", "search_type"],
        },
    },
    {
        "name": "read_document",
        "description": (
            "Read a PDF, DOCX, image, or text file and return its content. "
            "USE THIS instead of read_file for any document -- read_file on a "
            "PDF returns raw binary. Pages with no extractable text (scans, "
            "certificates, photographed pages) are automatically transcribed "
            "with a vision model, so image-only documents still come back as "
            "text. Set analyze_figures to also describe charts and diagrams."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the document"},
                "analyze_figures": {
                    "type": "boolean",
                    "description": "Also describe images/charts embedded in the document",
                },
                "max_vision_pages": {
                    "type": "integer",
                    "description": "Cap on pages sent to the vision model (default 10)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_folder",
        "description": (
            "Read EVERY supported document in a folder in one call (PDF, DOCX, "
            "txt, csv, json). Use this when the user points at a folder -- it "
            "avoids listing filenames and then reading them one at a time. "
            "Scanned pages are transcribed with vision automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Path to the folder"},
                "max_files": {
                    "type": "integer",
                    "description": "Cap on files read (default 25)",
                },
            },
            "required": ["folder"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the live web and get back an answer with source URLs. Use "
            "for anything beyond the HuggingFace Hub: which base model is "
            "current, what a benchmark actually measures, how a dataset was "
            "built, whether a license permits a use. Prefer this over answering "
            "from memory when the answer could have changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, phrased as a question",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "deploy_to_ollama",
        "description": "Deploy a trained model to Ollama for local chat. Exports to GGUF, creates a Modelfile, and runs 'ollama create'. After this, the user can chat with their model using 'ollama run <name>'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the trained/merged model directory",
                },
                "model_name": {
                    "type": "string",
                    "description": "Name for the Ollama model (e.g., 'my-finance-bot')",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt for the model's personality",
                },
                "quantization": {
                    "type": "string",
                    "description": "GGUF quantization type (default: Q4_K_M)",
                    "default": "Q4_K_M",
                },
            },
            "required": ["model_path", "model_name"],
        },
    },
    {
        "name": "deploy_to_huggingface",
        "description": "Upload a trained model to HuggingFace Hub so others can use it. Creates a model card with benchmarks and usage instructions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the trained/merged model directory",
                },
                "repo_name": {
                    "type": "string",
                    "description": "HuggingFace repo name (e.g., 'my-finance-model')",
                },
                "description": {
                    "type": "string",
                    "description": "Short description of the model",
                },
                "private": {
                    "type": "boolean",
                    "description": "Make the repo private (default: false)",
                    "default": False,
                },
            },
            "required": ["model_path", "repo_name"],
        },
    },
    {
        "name": "run_evaluation",
        "description": "Run benchmarks on a trained model to measure its quality. Returns scores on standard benchmarks like MMLU, HellaSwag, ARC, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the model to evaluate",
                },
                "benchmarks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Benchmarks to run (e.g., ['mmlu', 'hellaswag', 'arc_easy']). Defaults to a standard set.",
                },
            },
            "required": ["model_path"],
        },
    },
    {
        "name": "download_model",
        "description": "Download a base model from HuggingFace Hub to local storage. Use this before training when the user has chosen a model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "HuggingFace model ID (e.g., 'meta-llama/Llama-3.2-1B-Instruct')",
                },
                "cache_dir": {
                    "type": "string",
                    "description": "Where to cache the model (optional, uses HF default)",
                },
            },
            "required": ["model_name"],
        },
    },
    {
        "name": "install_dependencies",
        "description": "Check and install missing Python dependencies needed for a specific feature (training, evaluation, serving, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "enum": ["training", "evaluation", "serving", "cleaning", "rag", "chat", "all"],
                    "description": "Which feature group to install dependencies for",
                },
            },
            "required": ["feature"],
        },
    },
    {
        "name": "read_training_logs",
        "description": "Read the latest training logs to show progress, loss values, and any errors. Use this to give the user real-time updates on training.",
        "input_schema": {
            "type": "object",
            "properties": {
                "output_dir": {
                    "type": "string",
                    "description": "Training output directory to read logs from",
                },
                "last_n_lines": {
                    "type": "integer",
                    "description": "Number of recent log lines to return (default: 30)",
                    "default": 30,
                },
            },
            "required": ["output_dir"],
        },
    },
    {
        "name": "show_model_info",
        "description": "Show detailed information about a trained model: size, architecture, training config, and available checkpoints.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the model directory",
                },
            },
            "required": ["model_path"],
        },
    },
    {
        "name": "estimate_training",
        "description": "Estimate training time, memory usage, and whether the model fits the hardware. ALWAYS call this before start_training to warn the user about potential issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Base model name (e.g., 'meta-llama/Llama-3.2-1B')",
                },
                "mode": {
                    "type": "string",
                    "enum": ["lora", "qlora", "full"],
                    "description": "Training mode",
                },
                "num_samples": {
                    "type": "integer",
                    "description": "Number of training samples",
                },
                "num_epochs": {
                    "type": "integer",
                    "description": "Number of epochs",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Per-device batch size",
                },
                "seq_length": {
                    "type": "integer",
                    "description": "Max sequence length",
                },
            },
            "required": ["model_name", "mode", "num_samples"],
        },
    },
    {
        "name": "detect_project",
        "description": "Scan a directory and determine what kind of project it is (Node.js, Python, Rust, etc.) and whether LLM Forge is already set up. Use this before setup_project to understand the user's directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory to scan (defaults to current working directory)",
                    "default": ".",
                },
            },
            "required": [],
        },
    },
    {
        "name": "setup_project",
        "description": "Create the LLM Forge project structure in a directory. Creates configs/, data/, outputs/, examples/data/, .llmforge/, a starter config, and a .gitignore. Never overwrites existing files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory where the project should be set up",
                },
                "mode": {
                    "type": "string",
                    "enum": ["root", "subdirectory", "auto"],
                    "description": "Where to create the structure: 'root' (directly in directory), 'subdirectory' (in directory/llm-forge/), or 'auto' (detect based on existing content)",
                    "default": "auto",
                },
                "include_examples": {
                    "type": "boolean",
                    "description": "Include example training data files (default: true)",
                    "default": True,
                },
            },
            "required": ["directory"],
        },
    },
    # ----- Memory tools (handled by ChatEngine, not execute_tool) -----
    {
        "name": "save_memory",
        "description": "Save an important insight about the user, their project, or a training lesson to long-term memory. Call this proactively when you learn something worth remembering across sessions. Categories: user_preference, project_decision, training_lesson, user_behavior.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "user_preference",
                        "project_decision",
                        "training_lesson",
                        "user_behavior",
                    ],
                    "description": "Memory category",
                },
                "content": {
                    "type": "string",
                    "description": "What to remember (be specific and concise)",
                },
                "relevance": {
                    "type": "number",
                    "description": "How important this is (0.0 to 1.0, default 1.0)",
                    "default": 1.0,
                },
            },
            "required": ["category", "content"],
        },
    },
    {
        "name": "recall_memory",
        "description": "Search your long-term memory for past insights, decisions, or lessons. Use this when the user references past work or when you need context from previous sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keyword or topic)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_project_state",
        "description": "Get the current state of the project directory: configs, trained models, data files, active training. Use at session start to understand what the user has.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_session_history",
        "description": "Get summaries of past conversation sessions. Use when the user wants to resume previous work or references something from a past session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent sessions to retrieve (default: 5)",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "log_training_run",
        "description": "Record a training run's details and outcome in persistent history. Call after training starts or completes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "config_path": {"type": "string", "description": "Config file used"},
                "model_name": {"type": "string", "description": "Name of the trained model"},
                "base_model": {"type": "string", "description": "Base model used"},
                "mode": {"type": "string", "description": "Training mode (lora, qlora, full)"},
                "output_dir": {"type": "string", "description": "Output directory"},
                "final_loss": {"type": "number", "description": "Final training loss"},
                "eval_loss": {"type": "number", "description": "Evaluation loss"},
                "status": {"type": "string", "description": "started, completed, or failed"},
                "notes": {"type": "string", "description": "Any notes about this run"},
            },
            "required": ["config_path", "model_name", "base_model", "mode", "output_dir"],
        },
    },
    # ----- NVIDIA-powered tools (synthetic data & LLM-as-Judge) -----
    {
        "name": "generate_training_data",
        "description": "Generate synthetic training data using a large AI model. Give it a topic, a few examples, and it creates hundreds of high-quality Q&A pairs for fine-tuning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic for data generation (e.g., 'New Zealand visa regulations')",
                },
                "num_samples": {
                    "type": "integer",
                    "description": "Number of Q&A pairs to generate (default: 50)",
                    "default": 50,
                },
                "examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: 3-5 example Q&A pairs to guide the style",
                },
                "output_path": {
                    "type": "string",
                    "description": "Where to save the generated data (default: data/synthetic_train.jsonl)",
                },
                "format": {
                    "type": "string",
                    "enum": ["alpaca", "sharegpt"],
                    "description": "Output format",
                    "default": "alpaca",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "evaluate_with_llm",
        "description": "Evaluate your trained model's outputs using a large AI judge model. Scores responses on relevance, accuracy, helpfulness, and coherence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model outputs to evaluate",
                },
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The questions that produced these outputs",
                },
                "criteria": {
                    "type": "string",
                    "description": "What to evaluate (default: relevance, accuracy, helpfulness)",
                    "default": "relevance, accuracy, helpfulness",
                },
            },
            "required": ["model_outputs", "questions"],
        },
    },
    # ----- Test model (NVIDIA API) -----
    {
        "name": "test_model",
        "description": "Chat with any base model through NVIDIA API to test what it knows before fine-tuning. Send a question and get the model's response. Use this to evaluate if a model needs training on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model to test (e.g., 'meta/llama-3.2-3b-instruct', 'meta/llama-3.1-8b-instruct'). Use NVIDIA NIM model IDs.",
                },
                "question": {
                    "type": "string",
                    "description": "Question to ask the model",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt to set the model's role",
                    "default": "You are a helpful AI assistant.",
                },
                "num_questions": {
                    "type": "integer",
                    "description": "If provided, test with multiple questions (pass questions as newline-separated in 'question' field)",
                    "default": 1,
                },
            },
            "required": ["model", "question"],
        },
    },
    # ----- NVIDIA-powered tools: embeddings -----
    {
        "name": "generate_embeddings",
        "description": "Generate text embeddings using NVIDIA's embedding models for RAG, semantic search, or data deduplication. Supports batch processing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of texts to embed",
                },
                "model": {
                    "type": "string",
                    "description": "Embedding model (default: nvidia/nv-embedqa-e5-v5)",
                    "default": "nvidia/nv-embedqa-e5-v5",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional: save embeddings to a JSON file",
                },
            },
            "required": ["texts"],
        },
    },
    # ----- Code generation & A/B testing (NVIDIA API) -----
    {
        "name": "generate_script",
        "description": (
            "Generate a Python script for data preprocessing, format conversion, "
            "web scraping, or other custom tasks. Uses a code-specialized AI model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": (
                        "What the script should do (e.g., 'convert CSV to JSONL "
                        "Alpaca format', 'scrape FAQ from website')"
                    ),
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Where to save the script (default: scripts/generated_script.py)"
                    ),
                    "default": "scripts/generated_script.py",
                },
                "input_file": {
                    "type": "string",
                    "description": "Optional: input file the script should process",
                },
                "output_file": {
                    "type": "string",
                    "description": "Optional: output file the script should produce",
                },
            },
            "required": ["task_description"],
        },
    },
    {
        "name": "compare_models",
        "description": (
            "A/B test two models by asking the same questions to both and comparing "
            "their responses. Uses an AI judge to score each response."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_a": {
                    "type": "string",
                    "description": "First model (NVIDIA model ID)",
                },
                "model_b": {
                    "type": "string",
                    "description": "Second model (NVIDIA model ID)",
                },
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Questions to test both models on",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt for both models",
                    "default": "You are a helpful AI assistant.",
                },
            },
            "required": ["model_a", "model_b", "questions"],
        },
    },
    # ----- Execution tools (system-level, permission-gated) -----
    *EXECUTION_TOOLS,
]


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------


def _tool_schema(name: str) -> dict | None:
    """Return a tool's declared input schema, if it has one."""
    for tool in TOOLS:
        if tool.get("name") == name:
            return tool.get("input_schema")
    return None


def validate_tool_input(name: str, input_data: dict) -> str | None:
    """Check ``input_data`` against the tool's declared schema.

    Returns an error message, or None when the input is usable.

    The dispatcher reads required fields directly (``input_data["path"]``), so
    an omitted field raises KeyError and the model gets back ``{"error":
    "'path'"}`` -- a bare quoted key that says nothing about what was wrong or
    what the tool wanted. Models cannot self-correct from that and retry the
    same malformed call.

    Validating here means the model gets the field name, the reason, and the
    schema, which is enough to fix the call on the next turn. Borrowed from
    grok-build's approach of making the typed input the source of truth and
    failing structurally at the boundary rather than inside the tool body.
    """
    schema = _tool_schema(name)
    if not schema:
        return None

    if not isinstance(input_data, dict):
        return f"Tool '{name}' expects a JSON object of arguments, got {type(input_data).__name__}."

    properties: dict = schema.get("properties", {}) or {}
    required: list = schema.get("required", []) or []

    missing = [field for field in required if field not in input_data]
    if missing:
        return json.dumps(
            {
                "status": "error",
                "error": f"Missing required argument(s) for '{name}': {', '.join(missing)}",
                "required": required,
                "you_sent": sorted(input_data.keys()),
                "schema": properties,
            },
            indent=2,
        )

    # Enum mismatches are the other common model error, and produce equally
    # opaque failures deeper in the tool.
    for field, value in input_data.items():
        spec = properties.get(field)
        if not isinstance(spec, dict):
            continue
        allowed = spec.get("enum")
        if allowed and value not in allowed:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"Invalid value for '{field}' in '{name}': {value!r}"
                    ),
                    "allowed": allowed,
                },
                indent=2,
            )

    return None


def execute_tool(name: str, input_data: dict) -> str:
    """Execute a tool and return the result as a string."""
    problem = validate_tool_input(name, input_data)
    if problem is not None:
        return problem

    try:
        if name == "detect_hardware":
            return _detect_hardware()
        elif name == "scan_data":
            return _scan_data(input_data["path"])
        elif name == "write_config":
            return _write_config(input_data["output_path"], input_data["config"])
        elif name == "validate_config":
            return _validate_config(input_data["config_path"])
        elif name == "start_training":
            return _start_training(input_data["config_path"], input_data.get("verbose", True))
        elif name == "check_training_status":
            return _check_training_status()
        elif name == "export_model":
            return _export_model(
                input_data["model_path"],
                input_data["format"],
                input_data.get("output_dir"),
            )
        elif name == "list_configs":
            return _list_configs()
        elif name == "search_huggingface":
            return _search_huggingface(
                input_data["query"],
                input_data["search_type"],
                limit=min(int(input_data.get("limit", 5)), 20),
            )
        elif name == "web_search":
            from llm_forge.chat.discovery import web_search

            return web_search(input_data["query"])
        elif name == "read_document":
            from llm_forge.chat.documents import read_document_json

            return read_document_json(
                path=input_data["path"],
                analyze_figures=input_data.get("analyze_figures", False),
                max_vision_pages=int(input_data.get("max_vision_pages", 10)),
            )
        elif name == "read_folder":
            from llm_forge.chat.documents import read_folder_json

            return read_folder_json(
                folder=input_data["folder"],
                max_files=int(input_data.get("max_files", 25)),
            )
        elif name == "deploy_to_ollama":
            return _deploy_to_ollama(
                input_data["model_path"],
                input_data["model_name"],
                input_data.get("system_prompt"),
                input_data.get("quantization", "Q4_K_M"),
            )
        elif name == "deploy_to_huggingface":
            return _deploy_to_huggingface(
                input_data["model_path"],
                input_data["repo_name"],
                input_data.get("description", ""),
                input_data.get("private", False),
            )
        elif name == "run_evaluation":
            return _run_evaluation(
                input_data["model_path"],
                input_data.get("benchmarks"),
            )
        elif name == "download_model":
            return _download_model(
                input_data["model_name"],
                input_data.get("cache_dir"),
            )
        elif name == "install_dependencies":
            return _install_dependencies(input_data["feature"])
        elif name == "read_training_logs":
            return _read_training_logs(
                input_data["output_dir"],
                input_data.get("last_n_lines", 30),
            )
        elif name == "show_model_info":
            return _show_model_info(input_data["model_path"])
        elif name == "estimate_training":
            return _estimate_training(
                model_name=input_data["model_name"],
                mode=input_data["mode"],
                num_samples=input_data["num_samples"],
                num_epochs=input_data.get("num_epochs", 1),
                batch_size=input_data.get("batch_size", 4),
                seq_length=input_data.get("seq_length", 2048),
            )
        elif name == "detect_project":
            return _detect_project(input_data.get("directory", "."))
        elif name == "setup_project":
            return _setup_project(
                directory=input_data["directory"],
                mode=input_data.get("mode", "auto"),
                include_examples=input_data.get("include_examples", True),
            )
        elif name == "generate_training_data":
            return _generate_training_data(
                topic=input_data["topic"],
                num_samples=input_data.get("num_samples", 50),
                examples=input_data.get("examples"),
                output_path=input_data.get("output_path"),
                fmt=input_data.get("format", "alpaca"),
            )
        elif name == "evaluate_with_llm":
            return _evaluate_with_llm(
                model_outputs=input_data["model_outputs"],
                questions=input_data["questions"],
                criteria=input_data.get("criteria", "relevance, accuracy, helpfulness"),
            )
        elif name == "test_model":
            return _test_model(
                model=input_data["model"],
                question=input_data["question"],
                system_prompt=input_data.get("system_prompt", "You are a helpful AI assistant."),
                num_questions=input_data.get("num_questions", 1),
            )
        elif name == "generate_embeddings":
            return _generate_embeddings(
                texts=input_data["texts"],
                model=input_data.get("model", "nvidia/nv-embedqa-e5-v5"),
                output_path=input_data.get("output_path"),
            )
        elif name == "generate_script":
            return _generate_script(
                task_description=input_data["task_description"],
                output_path=input_data.get("output_path", "scripts/generated_script.py"),
                input_file=input_data.get("input_file"),
                output_file=input_data.get("output_file"),
            )
        elif name == "compare_models":
            return _compare_models(
                model_a=input_data["model_a"],
                model_b=input_data["model_b"],
                questions=input_data["questions"],
                system_prompt=input_data.get("system_prompt", "You are a helpful AI assistant."),
            )
        elif name in EXECUTION_TOOL_NAMES:
            return execute_execution_tool(name, input_data)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _detect_hardware() -> str:
    """Detect system hardware."""
    info: dict = {
        "os": platform.system(),
        "os_version": platform.release(),
        "cpu": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
    }

    # RAM
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    except ImportError:
        info["ram_total_gb"] = "unknown (install psutil)"

    # GPU detection
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_type"] = "nvidia_cuda"
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append(
                    {
                        "name": props.name,
                        # total_memory, not total_mem -- the latter does not
                        # exist and raised AttributeError on every CUDA machine.
                        "vram_gb": round(props.total_memory / (1024**3), 1),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            info["cuda_version"] = torch.version.cuda
            info["backend"] = "cuda"
            info["usable_memory_gb"] = info["gpus"][0]["vram_gb"]
            info["recommendation"] = _gpu_recommendation(
                info["gpus"][0]["vram_gb"], backend="cuda"
            )
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            from llm_forge.training.mac_utils import usable_unified_memory_gb

            total_ram = info["ram_total_gb"] if isinstance(info["ram_total_gb"], (int, float)) else 16.0
            usable = round(usable_unified_memory_gb(total_ram), 1)

            info["gpu_type"] = "apple_mps"
            info["backend"] = "mps"
            info["gpu_name"] = _get_apple_chip_name()
            info["unified_memory_gb"] = total_ram
            info["usable_memory_gb"] = usable
            info["memory_note"] = (
                f"Apple Silicon has no dedicated VRAM -- the GPU shares the "
                f"{total_ram} GB of system RAM. Budget {usable} GB for training; "
                f"the rest is the OS and other apps. Exceeding it swaps rather "
                f"than OOMs, which makes training pathologically slow instead of "
                f"failing fast."
            )
            info["unavailable_here"] = [
                "bitsandbytes (QLoRA 4-bit, 8-bit optimizers) -- CUDA only",
                "FlashAttention-2 -- CUDA only",
                "DeepSpeed / Megatron -- CUDA only",
            ]
            info["recommendation"] = _gpu_recommendation(usable, backend="mps")
        else:
            info["gpu_type"] = "none"
            info["backend"] = "cpu"
            info["usable_memory_gb"] = 0
            info["recommendation"] = {
                "mode": "lora",
                "max_model": "SmolLM2-135M (CPU testing only)",
                "note": (
                    "No GPU detected. Training will be very slow. Consider Google "
                    "Colab for free GPU access. QLoRA is not an option -- "
                    "bitsandbytes has no CPU kernel."
                ),
            }
    except ImportError:
        info["gpu_type"] = "unknown (torch not installed)"
        info["recommendation"] = {"note": "PyTorch not installed. Run: pip install torch"}

    return json.dumps(info, indent=2)


def _gpu_recommendation(vram_gb: float, backend: str = "cuda") -> dict:
    """Return training recommendations for a memory budget and backend.

    ``backend`` matters: recommending QLoRA on Apple Silicon sends the user
    down a path that cannot work, because bitsandbytes is CUDA-only. On MPS the
    4-bit option is MLX instead.
    """
    apple = backend == "mps"
    small_model_4bit = "mlx (4-bit)" if apple else "qlora"

    if vram_gb >= 80:
        rec = {
            "mode": "lora or full",
            "max_model": "Llama-3.2-3B (full) or 7B+ (LoRA)",
            "batch_size": "8-16",
        }
    elif vram_gb >= 40:
        rec = {
            "mode": "lora",
            "max_model": "Llama-3.2-3B or Phi-3-mini (3.8B)",
            "batch_size": "4-8",
        }
    elif vram_gb >= 24:
        rec = {
            "mode": f"lora or {small_model_4bit}",
            "max_model": f"Llama-3.2-3B (LoRA) or 7-8B ({small_model_4bit})",
            "batch_size": "2-4",
        }
    elif vram_gb >= 16:
        rec = {
            "mode": f"full (1B), lora (3B), or {small_model_4bit} (8B)",
            "max_model": "Llama-3.2-1B full fine-tune, or 8B with 4-bit adapters",
            "batch_size": "1-2",
        }
    elif vram_gb >= 12:
        rec = {
            "mode": f"lora or {small_model_4bit}",
            "max_model": f"Llama-3.2-1B (LoRA) or 3B ({small_model_4bit})",
            "batch_size": "1-2",
        }
    elif vram_gb >= 8:
        rec = {
            "mode": f"lora or {small_model_4bit}",
            "max_model": "Llama-3.2-1B",
            "batch_size": "1",
        }
    else:
        rec = {
            "mode": "lora",
            "max_model": "SmolLM2-135M",
            "batch_size": "1",
            "note": "Limited memory. Use a small model.",
        }

    if apple:
        rec["backend_note"] = (
            "Apple Silicon: set mlx.enabled: true for 4-bit work. "
            "training.mode: qlora will fail -- bitsandbytes is CUDA-only."
        )
    return rec


def _get_apple_chip_name() -> str:
    """Get Apple Silicon chip name."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "Apple Silicon"
    except Exception:
        return "Apple Silicon"


def _scan_data(path: str) -> str:
    """Scan a data source and return info about it."""
    result: dict = {"path": path}

    p = Path(path).expanduser()

    # Check if it's a HuggingFace dataset ID
    if not p.exists() and "/" in path and not path.startswith((".", "/")):
        result["source"] = "huggingface_hub"
        result["dataset_id"] = path
        try:
            from datasets import load_dataset

            ds = load_dataset(path, split="train", streaming=True)
            # Get first 3 samples
            samples = []
            for i, item in enumerate(ds):
                if i >= 3:
                    break
                samples.append(item)
            result["preview"] = samples
            result["columns"] = list(samples[0].keys()) if samples else []
            # Detect format
            cols = set(result["columns"])
            if {"instruction", "output"} <= cols:
                result["detected_format"] = "alpaca"
            elif "conversations" in cols:
                result["detected_format"] = "sharegpt"
            elif "text" in cols:
                result["detected_format"] = "completion"
            else:
                result["detected_format"] = "custom"
            result["status"] = "ok"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        return json.dumps(result, indent=2, default=str)

    # Local file/directory
    if not p.exists():
        result["status"] = "not_found"
        result["error"] = f"Path does not exist: {path}"
        return json.dumps(result, indent=2)

    if p.is_file():
        result["source"] = "local_file"
        result["size_mb"] = round(p.stat().st_size / (1024 * 1024), 2)
        result["extension"] = p.suffix.lower()

        # Read and preview
        try:
            if p.suffix.lower() in (".jsonl", ".json"):
                lines = p.read_text().strip().split("\n")
                result["sample_count"] = len(lines)
                samples = [json.loads(line) for line in lines[:3]]
                result["preview"] = samples
                result["columns"] = list(samples[0].keys()) if samples else []
                cols = set(result["columns"])
                if {"instruction", "output"} <= cols:
                    result["detected_format"] = "alpaca"
                elif "conversations" in cols:
                    result["detected_format"] = "sharegpt"
                elif "text" in cols:
                    result["detected_format"] = "completion"
                else:
                    result["detected_format"] = "custom"
            elif p.suffix.lower() in (".csv", ".tsv"):
                import csv

                with open(p) as f:
                    reader = csv.DictReader(f)
                    samples = [row for _, row in zip(range(3), reader, strict=False)]
                result["preview"] = samples
                result["columns"] = list(samples[0].keys()) if samples else []
                result["detected_format"] = "custom"
            elif p.suffix.lower() == ".txt":
                text = p.read_text()
                result["char_count"] = len(text)
                result["word_count"] = len(text.split())
                result["preview"] = text[:500]
                result["detected_format"] = "completion"
            else:
                result["detected_format"] = "unknown"
            result["status"] = "ok"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

    elif p.is_dir():
        result["source"] = "local_directory"
        files = [f for f in p.rglob("*") if f.is_file()]
        result["file_count"] = len(files)
        result["total_size_mb"] = round(sum(f.stat().st_size for f in files) / (1024 * 1024), 2)
        result["extensions"] = sorted(set(f.suffix.lower() for f in files if f.suffix))
        result["status"] = "ok"

    return json.dumps(result, indent=2, default=str)


def _write_config(output_path: str, config: dict) -> str:
    """Write a YAML config file."""
    try:
        import yaml

        p = Path(output_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return json.dumps(
            {
                "status": "ok",
                "path": str(p.resolve()),
                "message": f"Config saved to {p}",
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _validate_config(config_path: str) -> str:
    """Validate a config file."""
    try:
        import yaml

        from llm_forge.config.schema import LLMForgeConfig

        p = Path(config_path).expanduser()
        with open(p) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            return json.dumps({"status": "error", "error": "Config file is empty"})

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            LLMForgeConfig(**raw)

        return json.dumps({"status": "valid", "message": "Config is valid and ready for training!"})
    except Exception as e:
        return json.dumps({"status": "invalid", "errors": str(e)})


def _start_training(config_path: str, verbose: bool = True) -> str:
    """Start training in a subprocess and attach a background monitor."""
    global _active_monitor  # noqa: PLW0603

    # --- Gap 2: Resolve preset names from configs/ directory ---------------
    p = Path(config_path).expanduser()
    if not p.exists():
        configs_dir = Path.cwd() / "configs"
        candidate = configs_dir / config_path
        if candidate.exists():
            p = candidate
        elif (configs_dir / f"{config_path}.yaml").exists():
            p = configs_dir / f"{config_path}.yaml"
    if not p.exists():
        return json.dumps({"status": "error", "error": f"Config not found: {p}"})

    # --- Gap 1: GPU validation before training launch ----------------------
    try:
        hw_info = json.loads(_detect_hardware())
        gpu_type = hw_info.get("gpu_type", "none")

        # Parse config to get training mode and model name
        import yaml

        with open(p) as _cfg_f:
            _raw_cfg = yaml.safe_load(_cfg_f) or {}
        _training_sec = _raw_cfg.get("training", {}) if isinstance(_raw_cfg, dict) else {}
        _model_sec = _raw_cfg.get("model", {}) if isinstance(_raw_cfg, dict) else {}
        _mode = _training_sec.get("mode", "lora") if isinstance(_training_sec, dict) else "lora"
        _model_name = (
            _model_sec.get("name", "unknown") if isinstance(_model_sec, dict) else "unknown"
        )

        # Determine available VRAM
        _vram_gb: float = 0.0
        if gpu_type == "nvidia_cuda":
            gpus = hw_info.get("gpus", [])
            if gpus:
                _vram_gb = gpus[0].get("vram_gb", 0.0)
        elif gpu_type == "apple_mps":
            ram_total = hw_info.get("ram_total_gb", 0)
            _vram_gb = ram_total * 0.75 if isinstance(ram_total, (int, float)) else 8.0

        # VRAM checks per training mode
        hw_warnings: list[str] = []
        if _mode == "qlora" and _vram_gb > 0 and _vram_gb < 6:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"Insufficient VRAM for QLoRA: {_vram_gb:.1f} GB available, "
                        f"minimum 6 GB required. Model: {_model_name}"
                    ),
                }
            )
        elif _mode == "lora" and _vram_gb > 0 and _vram_gb < 8:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"Insufficient VRAM for LoRA: {_vram_gb:.1f} GB available, "
                        f"minimum 8 GB required. Model: {_model_name}"
                    ),
                }
            )
        elif _mode == "full" and _vram_gb > 0 and _vram_gb < 16:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"Insufficient VRAM for full fine-tuning: {_vram_gb:.1f} GB available, "
                        f"minimum 16 GB required. Model: {_model_name}"
                    ),
                }
            )

        if gpu_type == "none" and _mode != "qlora":
            hw_warnings.append(
                f"No GPU detected. Training in '{_mode}' mode will be very slow on CPU."
            )

        _hw_check_msg = (
            f"Hardware check passed: gpu={gpu_type}, vram={_vram_gb:.1f}GB, "
            f"mode={_mode}, model={_model_name}"
        )
        if hw_warnings:
            _hw_check_msg += f" | warnings: {'; '.join(hw_warnings)}"

    except Exception as hw_exc:
        # Hardware check is best-effort — don't block training on detection failure
        _hw_check_msg = f"Hardware check skipped: {hw_exc}"
        hw_warnings = []

    # --- Gap 4: Add --no-auto-optimize to prevent double optimization ------
    cmd = [
        sys.executable,
        "-m",
        "llm_forge.cli",
        "train",
        "--config",
        str(p),
        "--no-auto-optimize",
    ]
    if verbose:
        cmd.append("--verbose")

    try:
        # --- Gap 3: Capture stderr separately for error reporting ----------
        # Ensure PYTHONPATH includes src/ for iCloud paths where .pth files fail
        env = os.environ.copy()
        src_dir = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path.cwd()),
            env=env,
        )

        # Read first few lines to confirm it started
        output_lines = []
        for _ in range(20):
            line = process.stdout.readline()
            if not line:
                break
            output_lines.append(line.rstrip())

        # Check if the process already died (Gap 3)
        if process.poll() is not None and process.returncode != 0:
            stderr = process.stderr.read() if process.stderr else ""
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        f"Training failed immediately "
                        f"(exit code {process.returncode}): {stderr[-500:]}"
                    ),
                }
            )

        # Drain remaining stdout in a background thread to prevent
        # pipe buffer from filling up and deadlocking the subprocess.
        def _drain_pipe(pipe):
            try:
                while pipe.readline():
                    pass
            except (OSError, ValueError):
                pass

        drain_thread = threading.Thread(target=_drain_pipe, args=(process.stdout,), daemon=True)
        drain_thread.start()
        # Also drain stderr to prevent buffer deadlock
        drain_err_thread = threading.Thread(target=_drain_pipe, args=(process.stderr,), daemon=True)
        drain_err_thread.start()

        # Start a background monitor for real-time progress
        output_dir = _resolve_output_dir(p)
        if output_dir is not None:
            # Stop any previous monitor
            if _active_monitor is not None:
                _active_monitor.stop()
            _active_monitor = TrainingMonitor(str(output_dir))
            _active_monitor.start()

        return json.dumps(
            {
                "status": "started",
                "pid": process.pid,
                "config": str(p),
                "output_dir": str(output_dir) if output_dir else None,
                "initial_output": output_lines,
                "hardware_check": _hw_check_msg,
                "hardware_warnings": hw_warnings,
                "message": "Training started! You can ask me to check the status anytime.",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _resolve_output_dir(config_path: Path) -> Path | None:
    """Try to read the output_dir from a YAML config file."""
    try:
        import yaml

        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if raw and isinstance(raw, dict):
            training = raw.get("training", {})
            if isinstance(training, dict):
                out = training.get("output_dir")
                if out:
                    return Path(out).expanduser()
    except Exception:
        pass
    return None


def _check_training_status() -> str:
    """Check if training is running, using the background monitor when available."""
    global _active_monitor  # noqa: PLW0603

    # If an active monitor has real-time data, prefer it
    if _active_monitor is not None and _active_monitor.is_training_active():
        monitor_status = _active_monitor.get_status()
        if monitor_status.get("status") == "training":
            return json.dumps(monitor_status, indent=2)

    # Check for running llm-forge processes
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
        )
        # Match an actual training invocation, not merely any process whose
        # command line mentions both words. A pytest run, an editor, or a shell
        # doing something unrelated would otherwise be reported as "training in
        # progress" -- and the check is against the whole host process table,
        # so a false positive here is easy to hit and confusing to diagnose.
        training_markers = (
            "llm_forge.train",
            "llm-forge train",
            "llm_forge/train",
            "llm_forge.training",
        )
        forge_procs = [
            line
            for line in result.stdout.split("\n")
            if any(marker in line for marker in training_markers) and "grep" not in line
        ]
        if forge_procs:
            # Process is running but monitor may not have data yet
            info: dict = {
                "status": "running",
                "processes": len(forge_procs),
                "message": "Training is in progress.",
            }
            # Attach monitor data if available (even if status != "training")
            if _active_monitor is not None:
                info["monitor"] = _active_monitor.get_status()
            return json.dumps(info)
    except Exception:
        pass

    # No process running — stop the monitor if it's still active
    if _active_monitor is not None:
        _active_monitor.stop()
        _active_monitor = None

    # Check for recent output directories
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        subdirs = sorted(outputs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
        if subdirs:
            latest = subdirs[0]
            checkpoints = list(latest.glob("checkpoint-*"))
            return json.dumps(
                {
                    "status": "completed_or_idle",
                    "latest_output": str(latest),
                    "checkpoints": len(checkpoints),
                    "message": f"Latest training output: {latest.name}",
                }
            )

    return json.dumps({"status": "idle", "message": "No training detected."})


def _export_model(model_path: str, fmt: str, output_dir: str | None = None) -> str:
    """Export a model."""
    p = Path(model_path).expanduser()
    if not p.exists():
        return json.dumps({"status": "error", "error": f"Model not found: {p}"})

    out = Path(output_dir).expanduser() if output_dir else p.parent / fmt
    out.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            sys.executable,
            "-m",
            "llm_forge.cli",
            "export",
            "--config",
            "config.yaml",
            "--format",
            fmt,
            "--model-path",
            str(p),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return json.dumps(
            {
                "status": "ok" if result.returncode == 0 else "error",
                "output_dir": str(out),
                "stdout": result.stdout[-500:] if result.stdout else "",
                "stderr": result.stderr[-500:] if result.stderr else "",
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _list_configs() -> str:
    """List available example configs."""
    # Find configs directory relative to the package
    configs_dir = None
    for candidate in [
        Path(__file__).parent.parent.parent.parent / "configs",
        Path.cwd() / "configs",
    ]:
        if candidate.exists():
            configs_dir = candidate
            break

    if configs_dir is None:
        return json.dumps({"status": "error", "error": "Configs directory not found"})

    configs = []
    for f in sorted(configs_dir.glob("*.yaml")):
        configs.append({"name": f.name, "path": str(f)})

    return json.dumps({"configs": configs, "count": len(configs)}, indent=2)


def _local_memory_budget() -> tuple[float | None, str]:
    """Return ``(usable_gb, backend)`` for this machine.

    On Apple Silicon the GPU shares system RAM, so the budget is unified memory
    minus what the OS and other apps need -- not the full installed total.
    """
    try:
        from llm_forge.config.hardware_detector import detect_hardware

        hw = detect_hardware()
    except Exception:
        return None, "cuda"

    if getattr(hw, "is_mps", False):
        from llm_forge.training.mac_utils import usable_unified_memory_gb

        return usable_unified_memory_gb(hw.system_ram_gb), "mps"

    if getattr(hw, "has_gpu", False):
        return hw.max_gpu_vram_mb / 1024.0, "cuda"

    return None, "cpu"


def _search_huggingface(
    query: str,
    search_type: str,
    limit: int = 5,
    assess_local_fit: bool = True,
) -> str:
    """Search the HuggingFace Hub for models or datasets.

    Model results carry a real parameter count and a memory-fit verdict for
    this machine; dataset results carry a ground-truth assessment. See
    ``llm_forge.chat.discovery`` for how both are derived.
    """
    from llm_forge.chat.discovery import search_huggingface as _search

    budget_gb: float | None = None
    backend = "cuda"
    if assess_local_fit:
        budget_gb, backend = _local_memory_budget()

    return _search(
        query,
        search_type,
        limit=limit,
        budget_gb=budget_gb,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Phase 2 tool implementations
# ---------------------------------------------------------------------------


def _deploy_to_ollama(
    model_path: str,
    model_name: str,
    system_prompt: str | None = None,
    quantization: str = "Q4_K_M",
) -> str:
    """Deploy model to Ollama: GGUF export + Modelfile + ollama create."""
    import shutil

    p = Path(model_path).expanduser()
    if not p.exists():
        return json.dumps({"status": "error", "error": f"Model not found: {p}"})

    if not shutil.which("ollama"):
        return json.dumps(
            {
                "status": "error",
                "error": "Ollama is not installed. Install from: https://ollama.com/download",
            }
        )

    gguf_dir = p.parent / "gguf"
    gguf_dir.mkdir(exist_ok=True)
    gguf_file = gguf_dir / f"{model_name}-{quantization}.gguf"
    steps_done = []

    # Find existing GGUF or note we need one
    if not gguf_file.exists():
        existing = list(gguf_dir.glob("*.gguf")) + list(p.parent.glob("*.gguf"))
        if existing:
            gguf_file = existing[0]
            steps_done.append(f"Using existing GGUF: {gguf_file.name}")
        else:
            return json.dumps(
                {
                    "status": "needs_export",
                    "message": "No GGUF file found. Use export_model tool first to create one.",
                }
            )
    else:
        steps_done.append(f"GGUF exists: {gguf_file.name}")

    # Create Modelfile
    sys_prompt = system_prompt or "You are a helpful AI assistant."
    modelfile_lines = [
        f"FROM {gguf_file}",
        "",
        f'SYSTEM "{sys_prompt}"',
        "",
        "PARAMETER temperature 0.1",
        "PARAMETER top_k 40",
        "PARAMETER repeat_penalty 1.1",
        "PARAMETER num_predict 256",
        "PARAMETER num_ctx 2048",
        'PARAMETER stop "<|start_header_id|>"',
        'PARAMETER stop "<|eot_id|>"',
    ]
    modelfile_path = gguf_dir / "Modelfile"
    modelfile_path.write_text("\n".join(modelfile_lines) + "\n")
    steps_done.append("Created Modelfile")

    # Run ollama create
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            steps_done.append(f"Created Ollama model: {model_name}")
            return json.dumps(
                {
                    "status": "ok",
                    "model_name": model_name,
                    "steps": steps_done,
                    "run_command": f"ollama run {model_name}",
                    "message": f"Deployed! Run: ollama run {model_name}",
                }
            )
        else:
            return json.dumps(
                {"status": "error", "steps": steps_done, "error": result.stderr[-300:]}
            )
    except Exception as e:
        return json.dumps({"status": "error", "steps": steps_done, "error": str(e)})


def _deploy_to_huggingface(
    model_path: str, repo_name: str, description: str = "", private: bool = False
) -> str:
    """Upload a model to HuggingFace Hub."""
    p = Path(model_path).expanduser()
    if not p.exists():
        return json.dumps({"status": "error", "error": f"Model not found: {p}"})

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()["name"]
        repo_id = f"{user}/{repo_name}"

        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
        api.upload_folder(
            folder_path=str(p),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {repo_name} via LLM Forge",
        )

        url = f"https://huggingface.co/{repo_id}"
        return json.dumps(
            {
                "status": "ok",
                "repo_id": repo_id,
                "url": url,
                "message": f"Model uploaded to {url}",
            }
        )
    except ImportError:
        return json.dumps(
            {"error": "huggingface_hub not installed. Run: pip install huggingface_hub"}
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _run_evaluation(model_path: str, benchmarks: list[str] | None = None) -> str:
    """Run lm-eval benchmarks on a model."""
    p = Path(model_path).expanduser()
    if not p.exists():
        return json.dumps({"status": "error", "error": f"Model not found: {p}"})

    if benchmarks is None:
        benchmarks = ["hellaswag", "arc_easy", "mmlu", "truthfulqa_mc2"]

    tasks = ",".join(benchmarks)
    try:
        cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={p},dtype=bfloat16",
            "--tasks",
            tasks,
            "--batch_size",
            "4",
            "--output_path",
            str(p.parent / "eval_results"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            output_lines = result.stdout.split("\n")
            score_lines = [line for line in output_lines if "|" in line and "acc" in line.lower()]
            return json.dumps(
                {
                    "status": "ok",
                    "benchmarks": benchmarks,
                    "output_summary": score_lines[-10:] if score_lines else output_lines[-20:],
                    "message": "Evaluation complete!",
                }
            )
        else:
            return json.dumps({"status": "error", "error": result.stderr[-500:]})
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "error": "Evaluation timed out (>60 min)."})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _download_model(model_name: str, cache_dir: str | None = None) -> str:
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download

        kwargs: dict = {"repo_id": model_name}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        path = snapshot_download(**kwargs)
        return json.dumps(
            {
                "status": "ok",
                "model": model_name,
                "local_path": path,
                "message": f"Downloaded {model_name}",
            }
        )
    except ImportError:
        return json.dumps({"error": "huggingface_hub not installed."})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _install_dependencies(feature: str) -> str:
    """Install missing dependencies for a feature."""
    extras_map = {
        "training": "",
        "evaluation": "eval",
        "serving": "serve",
        "cleaning": "cleaning",
        "rag": "rag",
        "chat": "chat",
        "all": "all",
    }

    extra = extras_map.get(feature, feature)
    pkg = "llm-forge-new"
    install_cmd = f"{pkg}[{extra}]" if extra else pkg

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", install_cmd],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return json.dumps(
                {"status": "ok", "message": f"Dependencies for '{feature}' installed."}
            )
        else:
            return json.dumps({"status": "error", "error": result.stderr[-300:]})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _read_training_logs(output_dir: str, last_n_lines: int = 30) -> str:
    """Read recent training logs from an output directory."""
    p = Path(output_dir).expanduser()
    if not p.exists():
        return json.dumps({"status": "error", "error": f"Directory not found: {p}"})

    result_data: dict = {"output_dir": str(p)}

    # Check trainer_state.json for metrics
    trainer_state = p / "trainer_state.json"
    if trainer_state.exists():
        state = json.loads(trainer_state.read_text())
        log_history = state.get("log_history", [])
        recent = log_history[-5:] if len(log_history) > 5 else log_history
        result_data["recent_metrics"] = recent
        if log_history:
            latest = log_history[-1]
            result_data["current_step"] = latest.get("step", "?")
            result_data["current_loss"] = latest.get("loss", latest.get("train_loss", "?"))
            result_data["current_epoch"] = latest.get("epoch", "?")
        result_data["total_logged_steps"] = len(log_history)

    # Check for checkpoints
    checkpoints = sorted(p.glob("checkpoint-*"))
    if checkpoints:
        result_data["checkpoints"] = [c.name for c in checkpoints[-3:]]
        result_data["latest_checkpoint"] = str(checkpoints[-1])

    # Check for final model
    if (p / "model.safetensors").exists() or (p / "adapter_model.safetensors").exists():
        result_data["training_complete"] = True
        result_data["message"] = "Training is complete! Model files are saved."
    elif checkpoints:
        result_data["training_complete"] = False
        result_data["message"] = f"Training in progress. {len(checkpoints)} checkpoint(s)."
    else:
        result_data["training_complete"] = False
        result_data["message"] = "No training output found yet."

    result_data["status"] = "ok"
    return json.dumps(result_data, indent=2, default=str)


def _show_model_info(model_path: str) -> str:
    """Show information about a model."""
    p = Path(model_path).expanduser()
    if not p.exists():
        return json.dumps({"status": "error", "error": f"Path not found: {p}"})

    info: dict = {"path": str(p)}

    config_file = p / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text())
        info["architecture"] = config.get("architectures", ["unknown"])[0]
        info["model_type"] = config.get("model_type", "unknown")
        info["hidden_size"] = config.get("hidden_size", "?")
        info["num_layers"] = config.get("num_hidden_layers", "?")
        info["vocab_size"] = config.get("vocab_size", "?")

    safetensors = list(p.glob("*.safetensors"))
    if safetensors:
        total_size = sum(f.stat().st_size for f in safetensors)
        info["format"] = "safetensors"
        info["size_gb"] = round(total_size / (1024**3), 2)

    gguf_dirs = [p, p.parent / "gguf"]
    gguf_files = []
    for d in gguf_dirs:
        if d.exists():
            gguf_files.extend(d.glob("*.gguf"))
    if gguf_files:
        info["gguf_files"] = [f.name for f in gguf_files]

    info["has_tokenizer"] = (p / "tokenizer.json").exists()

    if (p / "adapter_config.json").exists():
        info["is_lora_adapter"] = True
        ac = json.loads((p / "adapter_config.json").read_text())
        info["lora_rank"] = ac.get("r", "?")
        info["base_model"] = ac.get("base_model_name_or_path", "?")

    checkpoints = sorted(p.glob("checkpoint-*"))
    if checkpoints:
        info["checkpoints"] = len(checkpoints)

    info["status"] = "ok"
    return json.dumps(info, indent=2)


# ---------------------------------------------------------------------------
# Training estimation
# ---------------------------------------------------------------------------


def _parse_model_params(model_name: str) -> float:
    """Extract approximate parameter count (in billions) from a model name.

    Handles patterns like "1B", "3B", "7B", "13B", "135M", "360M", "1.5B", etc.
    Returns the value in billions (e.g. 135M -> 0.135).
    """
    name_lower = model_name.lower()

    # Try billions first: "1b", "3b", "7b", "1.5b", "70b"
    m = re.search(r"(\d+(?:\.\d+)?)\s*b(?:illion)?", name_lower)
    if m:
        return float(m.group(1))

    # Try millions: "135m", "360m"
    m = re.search(r"(\d+(?:\.\d+)?)\s*m(?:illion)?", name_lower)
    if m:
        return float(m.group(1)) / 1000.0

    # Fallback: guess 1B
    return 1.0


def _detect_available_vram() -> tuple[float, str]:
    """Detect usable training memory in GB and the device class.

    Returns ``(usable_gb, device_type)`` where device_type is one of
    ``"a100"``, ``"consumer_gpu"``, ``"mps"``, or ``"cpu"``.

    "Usable" is deliberately not "installed". On Apple Silicon the GPU has no
    dedicated VRAM -- it shares system RAM with the OS -- so sizing a run
    against the full total makes it swap, which on Metal costs far more than it
    saves.
    """
    try:
        import torch
    except ImportError:
        return 0.0, "cpu"

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            # The attribute is total_memory. This previously read `total_mem`,
            # which does not exist -- so this raised AttributeError on every
            # CUDA machine and took estimate_training down with it.
            vram_gb = props.total_memory / (1024**3)
            name = props.name.lower()
            datacentre = ("a100", "h100", "h200", "a6000", "l40", "b200")
            device_type = "a100" if any(k in name for k in datacentre) else "consumer_gpu"
            return round(vram_gb, 1), device_type

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            from llm_forge.training.mac_utils import usable_unified_memory_gb

            try:
                import psutil

                total_ram = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                total_ram = 16.0
            return round(usable_unified_memory_gb(total_ram), 1), "mps"
    except Exception as exc:  # driver quirks, unusual torch builds
        logger.debug("VRAM detection failed, assuming CPU: %s", exc)

    return 0.0, "cpu"


def _estimate_training(
    model_name: str,
    mode: str,
    num_samples: int,
    num_epochs: int = 1,
    batch_size: int = 4,
    seq_length: int = 2048,
) -> str:
    """Estimate training time, VRAM, and feasibility.

    Returns a JSON object with fits_in_memory, estimated_vram_gb,
    available_vram_gb, estimated_time_minutes, steps_total, and
    recommendation.
    """
    from llm_forge.chat.discovery import METHOD_LABELS, assess_fit, recommended_method

    params_b = _parse_model_params(model_name)

    # --- Hardware detection ---
    available_vram_gb, device_type = _detect_available_vram()
    backend = "mps" if device_type == "mps" else ("cuda" if available_vram_gb > 0 else "cpu")

    # The memory model lives in discovery.assess_fit so the number the agent
    # quotes here is the same one search_huggingface used when it recommended
    # the model. Keeping a second copy of this arithmetic is how the two drift.
    mode_to_method = {
        "qlora": "mlx_lora_4bit" if backend in {"mps", "mlx"} else "qlora",
        "full": "full",
        "lora": "lora",
    }
    method = mode_to_method.get(mode, "lora")

    verdicts = assess_fit(
        params_b * 1e9,
        available_vram_gb,
        seq_length=seq_length,
        batch_size=batch_size,
        backend=backend,
    )
    by_method = {v.method: v for v in verdicts}
    chosen = by_method[method]

    estimated_vram_gb = round(chosen.required_gb, 1)
    fits = chosen.fits if available_vram_gb > 0 else False

    # --- Time estimation ---
    steps_total = math.ceil((num_samples * num_epochs) / batch_size)

    # Seconds per step by device type
    sps_lookup = {
        "cpu": 2.0,
        "mps": 0.5,
        "consumer_gpu": 0.3,
        "a100": 0.1,
    }
    seconds_per_step = sps_lookup.get(device_type, 1.0)
    # Scale by model size (baseline is 1B)
    seconds_per_step *= max(params_b, 0.1)

    estimated_time_seconds = steps_total * seconds_per_step
    estimated_time_minutes = round(estimated_time_seconds / 60, 1)

    # --- Recommendation ---
    recommendations: list[str] = []
    best_method = recommended_method(verdicts)

    if chosen.note:
        recommendations.append(chosen.note)

    if not fits and available_vram_gb > 0:
        # Suggest the most capable method that actually fits *this* backend --
        # blanket "switch to QLoRA" advice is wrong on Apple Silicon, where
        # bitsandbytes cannot run at all.
        if best_method and best_method != method:
            recommendations.append(
                f"Switch to {METHOD_LABELS[best_method]}: "
                f"{by_method[best_method].required_gb:.1f} GB vs "
                f"{estimated_vram_gb:.1f} GB for {METHOD_LABELS[method]}."
            )
        if batch_size > 1:
            recommendations.append(
                f"Reduce batch_size from {batch_size} to 1 and use gradient accumulation."
            )
        if seq_length > 1024:
            recommendations.append(f"Reduce seq_length from {seq_length} to 1024.")
        if best_method is None:
            recommendations.append(
                f"No method fits {params_b:.1f}B in {available_vram_gb:.0f} GB at "
                f"batch_size={batch_size}, seq_length={seq_length}. Try a smaller "
                "model -- run search_huggingface to find one that fits."
            )
    elif not fits and available_vram_gb == 0:
        recommendations.append(
            "No GPU detected. Training will be very slow on CPU. "
            "Consider Google Colab or a cloud GPU."
        )
    else:
        recommendations.append("Looks good -- this fits in memory.")

    result = {
        "status": "ok",
        "model_name": model_name,
        "estimated_params_billion": round(params_b, 3),
        "mode": mode,
        "fits_in_memory": fits,
        "estimated_vram_gb": estimated_vram_gb,
        "available_vram_gb": available_vram_gb,
        "device_type": device_type,
        "backend": backend,
        "steps_total": steps_total,
        "estimated_time_minutes": estimated_time_minutes,
        "recommended_mode": best_method,
        "breakdown": {k: round(v, 1) for k, v in chosen.breakdown.items()},
        "all_methods": [v.as_dict() for v in verdicts],
        "recommendation": " ".join(recommendations),
    }
    if device_type == "mps":
        result["memory_note"] = (
            f"available_vram_gb is usable unified memory, not installed RAM -- "
            f"the OS keeps the difference. Training sized against the full "
            f"total will swap."
        )
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Project setup tools
# ---------------------------------------------------------------------------


def _detect_project(directory: str) -> str:
    """Detect the project type of a directory."""
    from llm_forge.chat.project_setup import detect_project_type

    result = detect_project_type(directory)
    result["status"] = "ok"
    return json.dumps(result, indent=2)


def _setup_project(
    directory: str,
    mode: str = "auto",
    include_examples: bool = True,
) -> str:
    """Scaffold the LLM Forge project structure."""
    from llm_forge.chat.project_setup import scaffold_project

    result = scaffold_project(
        directory=directory,
        mode=mode,
        include_examples=include_examples,
    )
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# NVIDIA-powered tools (synthetic data generation & LLM-as-Judge evaluation)
# ---------------------------------------------------------------------------


def _generate_training_data(
    topic: str,
    num_samples: int = 50,
    examples: list[str] | None = None,
    output_path: str | None = None,
    fmt: str = "alpaca",
) -> str:
    """Generate synthetic training data using whichever LLM provider is set up."""
    from llm_forge.chat.utility_llm import NoUtilityProviderError, resolve_provider

    try:
        provider = resolve_provider()
    except NoUtilityProviderError as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    client = provider.client
    output_path = output_path or "data/synthetic_train.jsonl"

    # Build the generation prompt
    example_text = ""
    if examples:
        example_text = "\n\nHere are example Q&A pairs to match the style:\n"
        for i, ex in enumerate(examples, 1):
            example_text += f"{i}. {ex}\n"

    generated: list[dict] = []
    batch_size = 10  # Generate 10 at a time

    for batch_num in range(0, num_samples, batch_size):
        remaining = min(batch_size, num_samples - batch_num)

        prompt = (
            f"Generate exactly {remaining} diverse question-answer pairs about: {topic}"
            f"{example_text}\n"
            "Format each pair as a JSON object on its own line, like:\n"
            '{"instruction": "question here", "input": "", "output": "detailed answer here"}\n'
            "\nRules:\n"
            "- Each answer should be 2-5 sentences\n"
            "- Questions should be diverse (what, how, why, when, compare, explain)\n"
            "- Answers must be factually accurate\n"
            "- No duplicate questions\n"
            f"\nGenerate {remaining} pairs now, one JSON object per line:"
        )

        try:
            response = client.chat.completions.create(
                model=provider.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.8,
            )

            text = response.choices[0].message.content or ""

            # Parse JSON lines from response
            for line in text.strip().split("\n"):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        obj = json.loads(line)
                        if "instruction" in obj and "output" in obj:
                            if "input" not in obj:
                                obj["input"] = ""
                            generated.append(obj)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                    "generated_so_far": len(generated),
                }
            )

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in generated:
            f.write(json.dumps(item) + "\n")

    return json.dumps(
        {
            "status": "ok",
            "samples_generated": len(generated),
            "samples_requested": num_samples,
            "output_path": output_path,
            "format": fmt,
            "topic": topic,
            "message": f"Generated {len(generated)} training samples. Saved to {output_path}",
        }
    )


def _evaluate_with_llm(
    model_outputs: list[str],
    questions: list[str],
    criteria: str = "relevance, accuracy, helpfulness",
) -> str:
    """Evaluate model outputs using whichever LLM provider is set up."""
    from llm_forge.chat.utility_llm import NoUtilityProviderError, resolve_provider

    try:
        provider = resolve_provider()
    except NoUtilityProviderError as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    client = provider.client

    evaluations: list[dict] = []
    for q, output in zip(questions, model_outputs, strict=False):
        prompt = (
            f"Evaluate this AI model's response on: {criteria}\n\n"
            f"Question: {q}\n"
            f"Model Response: {output}\n\n"
            "Score each criterion 1-5 and explain briefly. Format as JSON:\n"
            '{"scores": {"relevance": N, "accuracy": N, "helpfulness": N}, '
            '"overall": N, "feedback": "brief feedback"}'
        )

        try:
            response = client.chat.completions.create(
                model=provider.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
            )
            text = response.choices[0].message.content or ""
            # Try to parse JSON from response
            parsed = False
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        eval_result = json.loads(line)
                        eval_result["question"] = q[:100]
                        evaluations.append(eval_result)
                        parsed = True
                        break
                    except json.JSONDecodeError:
                        continue
            if not parsed:
                evaluations.append({"question": q[:100], "error": "Failed to parse judge response"})
        except Exception as e:
            evaluations.append({"question": q[:100], "error": str(e)})

    # Calculate averages
    scores = [e.get("overall", 0) for e in evaluations if "overall" in e]
    avg = sum(scores) / len(scores) if scores else 0

    return json.dumps(
        {
            "status": "ok",
            "evaluations": evaluations,
            "average_score": round(avg, 2),
            "samples_evaluated": len(evaluations),
            "criteria": criteria,
        },
        indent=2,
    )


def _test_model(
    model: str,
    question: str,
    system_prompt: str = "You are a helpful AI assistant.",
    num_questions: int = 1,
) -> str:
    """Test a base model via the configured LLM provider to see its capabilities."""
    from llm_forge.chat.utility_llm import NoUtilityProviderError, resolve_provider

    try:
        provider = resolve_provider()
    except NoUtilityProviderError as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    client = provider.client

    # Handle multiple questions
    if num_questions > 1:
        questions = [q.strip() for q in question.split("\n") if q.strip()]
    else:
        questions = [question]

    results: list[dict] = []
    for q in questions:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            answer = response.choices[0].message.content or ""
            results.append(
                {
                    "question": q,
                    "answer": answer,
                    "model": model,
                    "tokens_used": (response.usage.total_tokens if response.usage else 0),
                }
            )
        except Exception as e:
            results.append({"question": q, "error": str(e), "model": model})

    return json.dumps(
        {
            "status": "ok",
            "model": model,
            "results": results,
            "total_questions": len(results),
            "message": f"Tested {model} with {len(results)} question(s)",
        },
        indent=2,
    )


def _generate_embeddings(
    texts: list[str],
    model: str = "nvidia/nv-embedqa-e5-v5",
    output_path: str | None = None,
) -> str:
    """Generate embeddings using NVIDIA NIM embedding models.

    Unlike the other LLM-backed tools, this one genuinely needs NVIDIA (or a
    local Ollama): Ollama Cloud returns 404 for /v1/embeddings, so there is no
    cloud fallback to route to.
    """
    from llm_forge.chat.nvidia_provider import has_nvidia_api_key, nvidia_client

    if not has_nvidia_api_key():
        return json.dumps(
            {
                "status": "error",
                "error": (
                    "Embeddings need an NVIDIA API key. Unlike the other tools, "
                    "this one has no Ollama Cloud fallback -- that endpoint does "
                    "not serve embeddings."
                ),
                "fix": (
                    "Get a free key at https://build.nvidia.com/ and set "
                    "NVIDIA_API_KEY, or run a local Ollama server which does "
                    "support embeddings."
                ),
            }
        )

    client = nvidia_client()

    try:
        # NVIDIA embedding API is OpenAI-compatible
        response = client.embeddings.create(
            model=model,
            input=texts,
            encoding_format="float",
        )

        embeddings = []
        for item in response.data:
            embeddings.append(
                {
                    "text": texts[item.index][:100],  # Preview only
                    "embedding_dim": len(item.embedding),
                    "embedding": item.embedding[:5],  # First 5 values as preview
                }
            )

        result: dict = {
            "status": "ok",
            "model": model,
            "num_texts": len(texts),
            "embedding_dimension": (len(response.data[0].embedding) if response.data else 0),
            "previews": embeddings[:3],  # Show first 3
            "message": (
                f"Generated {len(texts)} embeddings ({len(response.data[0].embedding)}d)"
                if response.data
                else f"Generated {len(texts)} embeddings"
            ),
        }

        # Save full embeddings if path provided
        if output_path:
            full_data = []
            for item in response.data:
                full_data.append(
                    {
                        "text": texts[item.index],
                        "embedding": item.embedding,
                    }
                )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(full_data, f)
            result["saved_to"] = output_path

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _generate_script(
    task_description: str,
    output_path: str = "scripts/generated_script.py",
    input_file: str | None = None,
    output_file: str | None = None,
) -> str:
    """Generate a Python script using the configured LLM provider."""
    from llm_forge.chat.utility_llm import NoUtilityProviderError, resolve_provider

    try:
        provider = resolve_provider()
    except NoUtilityProviderError as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    client = provider.client

    context = f"Task: {task_description}"
    if input_file:
        context += f"\nInput file: {input_file}"
    if output_file:
        context += f"\nOutput file: {output_file}"

    prompt = f"""Write a complete, runnable Python script for this task:

{context}

Requirements:
- Include all necessary imports
- Add error handling
- Add a main() function
- Add if __name__ == "__main__" guard
- Print progress messages
- Handle file not found errors gracefully
- Use pathlib for file paths

Write ONLY the Python code, no explanations:"""

    try:
        response = client.chat.completions.create(
            model=provider.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python programmer. Write clean, production-quality code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        code = response.choices[0].message.content or ""

        # Extract code from markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        # Save script
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(code)

        # Count lines
        lines = len(code.strip().split("\n"))

        return json.dumps(
            {
                "status": "ok",
                "output_path": output_path,
                "lines": lines,
                "preview": code[:300],
                "message": f"Script generated ({lines} lines). Saved to {output_path}",
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _compare_models(
    model_a: str,
    model_b: str,
    questions: list[str],
    system_prompt: str = "You are a helpful AI assistant.",
) -> str:
    """A/B test two models on the same questions with AI judging."""
    from llm_forge.chat.utility_llm import NoUtilityProviderError, resolve_provider

    try:
        provider = resolve_provider()
    except NoUtilityProviderError as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    client = provider.client

    comparisons: list[dict] = []
    a_wins = 0
    b_wins = 0
    ties = 0

    for q in questions:
        comp: dict = {"question": q}

        # Get response from both models
        for label, model in [("model_a", model_a), ("model_b", model_b)]:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q},
                    ],
                    max_tokens=300,
                    temperature=0.1,
                )
                comp[label] = resp.choices[0].message.content or ""
            except Exception as e:
                comp[label] = f"[Error: {e}]"

        # Judge with a large model
        try:
            judge_prompt = (
                "Compare these two AI responses to the same question.\n\n"
                f"Question: {q}\n\n"
                f"Response A ({model_a}):\n"
                f"{comp.get('model_a', '')}\n\n"
                f"Response B ({model_b}):\n"
                f"{comp.get('model_b', '')}\n\n"
                "Which response is better? Reply with ONLY a JSON object:\n"
                '{"winner": "A" or "B" or "tie", "reason": "brief explanation"}'
            )

            judge_resp = client.chat.completions.create(
                model=provider.model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=200,
                temperature=0.1,
            )

            judge_text = judge_resp.choices[0].message.content or ""
            # Parse judge response
            parsed_verdict = False
            for line in judge_text.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        verdict = json.loads(line)
                        comp["winner"] = verdict.get("winner", "tie")
                        comp["reason"] = verdict.get("reason", "")
                        if comp["winner"] == "A":
                            a_wins += 1
                        elif comp["winner"] == "B":
                            b_wins += 1
                        else:
                            ties += 1
                        parsed_verdict = True
                        break
                    except json.JSONDecodeError:
                        continue
            if not parsed_verdict:
                comp["winner"] = "tie"
                comp["reason"] = "Could not parse judge response"
                ties += 1
        except Exception:
            comp["winner"] = "tie"
            comp["reason"] = "Judge unavailable"
            ties += 1

        comparisons.append(comp)

    if a_wins > b_wins:
        verdict_str = f"{model_a} wins"
    elif b_wins > a_wins:
        verdict_str = f"{model_b} wins"
    else:
        verdict_str = "Tie"

    return json.dumps(
        {
            "status": "ok",
            "model_a": model_a,
            "model_b": model_b,
            "results": comparisons,
            "summary": {
                "model_a_wins": a_wins,
                "model_b_wins": b_wins,
                "ties": ties,
                "total": len(questions),
            },
            "verdict": verdict_str,
            "message": (
                f"A/B test: {model_a} ({a_wins} wins) vs {model_b} ({b_wins} wins), {ties} ties"
            ),
        },
        indent=2,
    )
