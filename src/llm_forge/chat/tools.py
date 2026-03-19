"""Tool definitions and implementations for the LLM Forge chat assistant."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

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
    """Start training in a subprocess."""
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

        return json.dumps(
            {
                "status": "started",
                "pid": process.pid,
                "config": str(p),
                "initial_output": output_lines,
                "message": "Training started! You can ask me to check the status anytime.",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _check_training_status() -> str:
    """Check if training is running."""

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
            return json.dumps(
                {
                    "status": "running",
                    "processes": len(forge_procs),
                    "message": "Training is in progress.",
                }
            )
    except Exception:
        pass

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
