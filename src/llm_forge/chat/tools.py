"""Tool definitions and implementations for the LLM Forge chat assistant."""

from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

from llm_forge.chat.training_monitor import TrainingMonitor

# ---------------------------------------------------------------------------
# Active training monitor (module-level singleton)
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
        "description": "Search HuggingFace Hub for models or datasets matching a query.",
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
            },
            "required": ["query", "search_type"],
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
]


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------


def execute_tool(name: str, input_data: dict) -> str:
    """Execute a tool and return the result as a string."""
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
            return _search_huggingface(input_data["query"], input_data["search_type"])
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
                        "vram_gb": round(props.total_mem / (1024**3), 1),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            info["cuda_version"] = torch.version.cuda
            info["recommendation"] = _gpu_recommendation(info["gpus"][0]["vram_gb"])
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_type"] = "apple_mps"
            info["gpu_name"] = _get_apple_chip_name()
            info["recommendation"] = _gpu_recommendation(
                info["ram_total_gb"] * 0.75 if isinstance(info["ram_total_gb"], (int, float)) else 8
            )
        else:
            info["gpu_type"] = "none"
            info["recommendation"] = {
                "mode": "qlora",
                "max_model": "SmolLM2-135M (CPU testing only)",
                "note": "No GPU detected. Training will be very slow. Consider using Google Colab for free GPU access.",
            }
    except ImportError:
        info["gpu_type"] = "unknown (torch not installed)"
        info["recommendation"] = {"note": "PyTorch not installed. Run: pip install torch"}

    return json.dumps(info, indent=2)


def _gpu_recommendation(vram_gb: float) -> dict:
    """Return training recommendations based on available VRAM."""
    if vram_gb >= 80:
        return {
            "mode": "lora or full",
            "max_model": "Llama-3.2-3B (full) or 7B+ (LoRA)",
            "batch_size": "8-16",
        }
    elif vram_gb >= 40:
        return {
            "mode": "lora",
            "max_model": "Llama-3.2-3B or Phi-3-mini (3.8B)",
            "batch_size": "4-8",
        }
    elif vram_gb >= 24:
        return {
            "mode": "lora or qlora",
            "max_model": "Llama-3.2-3B (LoRA) or 7B (QLoRA)",
            "batch_size": "2-4",
        }
    elif vram_gb >= 12:
        return {
            "mode": "qlora",
            "max_model": "Llama-3.2-1B (LoRA) or 3B (QLoRA)",
            "batch_size": "1-2",
        }
    elif vram_gb >= 8:
        return {
            "mode": "qlora",
            "max_model": "Llama-3.2-1B",
            "batch_size": "1",
        }
    else:
        return {
            "mode": "qlora",
            "max_model": "SmolLM2-135M",
            "batch_size": "1",
            "note": "Limited VRAM. Consider QLoRA with a small model.",
        }


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

    p = Path(config_path).expanduser()
    if not p.exists():
        return json.dumps({"status": "error", "error": f"Config not found: {p}"})

    cmd = [sys.executable, "-m", "llm_forge.cli", "train", "--config", str(p)]
    if verbose:
        cmd.append("--verbose")

    try:
        # Start training as a subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(p.parent),
        )

        # Read first few lines to confirm it started
        output_lines = []
        for _ in range(20):
            line = process.stdout.readline()
            if not line:
                break
            output_lines.append(line.rstrip())

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
        forge_procs = [
            line
            for line in result.stdout.split("\n")
            if "llm_forge" in line and "train" in line and "grep" not in line
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


def _search_huggingface(query: str, search_type: str) -> str:
    """Search HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        results = []

        if search_type == "models":
            models = api.list_models(search=query, sort="downloads", limit=5)
            for m in models:
                results.append(
                    {
                        "id": m.id,
                        "downloads": m.downloads,
                        "likes": m.likes,
                    }
                )
        elif search_type == "datasets":
            datasets = api.list_datasets(search=query, sort="downloads", limit=5)
            for d in datasets:
                results.append(
                    {
                        "id": d.id,
                        "downloads": d.downloads,
                        "likes": d.likes,
                    }
                )

        return json.dumps({"query": query, "type": search_type, "results": results}, indent=2)
    except ImportError:
        return json.dumps(
            {"error": "huggingface_hub not installed. Run: pip install huggingface_hub"}
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


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
    """Detect available VRAM in GB and device type.

    Returns (vram_gb, device_type) where device_type is one of:
    "cuda", "mps", "cpu".
    """
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_mem / (1024**3)
            name = props.name.lower()
            if "a100" in name or "h100" in name or "a6000" in name:
                device_type = "a100"
            else:
                device_type = "consumer_gpu"
            return round(vram_gb, 1), device_type
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon — unified memory, estimate ~75% available for GPU
            try:
                import psutil

                total_ram = psutil.virtual_memory().total / (1024**3)
                return round(total_ram * 0.75, 1), "mps"
            except ImportError:
                return 8.0, "mps"
    except ImportError:
        pass

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
    params_b = _parse_model_params(model_name)

    # --- VRAM estimation ---
    # Bytes per parameter for the model weights in GPU memory
    if mode == "qlora":
        bytes_per_param = 0.5  # 4-bit quantised
    elif mode == "full":
        bytes_per_param = 4.0  # fp32 master weights
    else:  # lora — bf16 frozen weights
        bytes_per_param = 2.0

    model_vram_gb = (params_b * 1e9 * bytes_per_param) / (1024**3)

    # Gradient & optimizer overhead (Adam keeps 2 extra copies of trainable params)
    if mode == "lora":
        # Only ~2-5% of params are trainable
        trainable_fraction = 0.03
    elif mode == "qlora":
        trainable_fraction = 0.03
    else:
        trainable_fraction = 1.0

    trainable_params_gb = (params_b * 1e9 * trainable_fraction * 4) / (1024**3)
    # Adam states (momentum + variance): 2x the trainable params in fp32
    optimizer_vram_gb = trainable_params_gb * 2

    # Activation memory estimate (rough: proportional to batch_size * seq_length * hidden_dim)
    # Using a simplified heuristic: ~0.5 GB per billion params per batch element
    activation_vram_gb = params_b * batch_size * (seq_length / 2048) * 0.5

    estimated_vram_gb = round(
        model_vram_gb + trainable_params_gb + optimizer_vram_gb + activation_vram_gb,
        1,
    )

    # --- Hardware detection ---
    available_vram_gb, device_type = _detect_available_vram()

    fits = estimated_vram_gb <= available_vram_gb if available_vram_gb > 0 else False

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
    if not fits and available_vram_gb > 0:
        if mode != "qlora":
            recommendations.append("Switch to QLoRA mode to reduce memory by ~75%.")
        if batch_size > 1:
            recommendations.append(
                f"Reduce batch_size from {batch_size} to 1 and use gradient accumulation."
            )
        if seq_length > 1024:
            recommendations.append(f"Reduce seq_length from {seq_length} to 1024.")
        if params_b > 1:
            recommendations.append(
                f"Consider a smaller model (e.g., 1B instead of {params_b:.1f}B)."
            )
    elif not fits and available_vram_gb == 0:
        recommendations.append(
            "No GPU detected. Training will be very slow on CPU. "
            "Consider Google Colab or a cloud GPU."
        )
    else:
        recommendations.append("Looks good! The model should fit in memory.")

    result = {
        "status": "ok",
        "model_name": model_name,
        "estimated_params_billion": round(params_b, 3),
        "mode": mode,
        "fits_in_memory": fits,
        "estimated_vram_gb": estimated_vram_gb,
        "available_vram_gb": available_vram_gb,
        "device_type": device_type,
        "steps_total": steps_total,
        "estimated_time_minutes": estimated_time_minutes,
        "breakdown": {
            "model_weights_gb": round(model_vram_gb, 1),
            "gradients_gb": round(trainable_params_gb, 1),
            "optimizer_gb": round(optimizer_vram_gb, 1),
            "activations_gb": round(activation_vram_gb, 1),
        },
        "recommendation": " ".join(recommendations),
    }
    return json.dumps(result, indent=2)
