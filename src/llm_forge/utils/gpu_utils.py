"""GPU utility functions for llm-forge.

Provides VRAM introspection, model memory estimation, and pretty-printed
hardware summaries used by the CLI ``info`` command and hardware auto-optimizer.
"""

from __future__ import annotations

import contextlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    """Information about a single GPU device."""

    index: int
    name: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    temperature_c: float | None = None
    utilization_pct: float | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_mb / 1024.0

    @property
    def used_memory_gb(self) -> float:
        return self.used_memory_mb / 1024.0

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory_mb / 1024.0

    @property
    def utilization_str(self) -> str:
        if self.utilization_pct is not None:
            return f"{self.utilization_pct:.0f}%"
        return "N/A"


@dataclass
class SystemGPUInfo:
    """Aggregate GPU information for the system."""

    cuda_available: bool = False
    cuda_version: str | None = None
    cudnn_version: str | None = None
    torch_version: str | None = None
    driver_version: str | None = None
    gpu_count: int = 0
    gpus: list[GPUInfo] = field(default_factory=list)
    mps_available: bool = False
    error: str | None = None

    @property
    def total_vram_gb(self) -> float:
        return sum(g.total_memory_gb for g in self.gpus)

    @property
    def total_free_vram_gb(self) -> float:
        return sum(g.free_memory_gb for g in self.gpus)


# ---------------------------------------------------------------------------
# Known model parameter counts (billions) for memory estimation
# ---------------------------------------------------------------------------

_KNOWN_MODEL_PARAMS: dict[str, float] = {
    # Llama family
    "meta-llama/Llama-2-7b": 7.0,
    "meta-llama/Llama-2-13b": 13.0,
    "meta-llama/Llama-2-70b": 70.0,
    "meta-llama/Meta-Llama-3-8B": 8.0,
    "meta-llama/Meta-Llama-3-70B": 70.0,
    "meta-llama/Meta-Llama-3.1-8B": 8.0,
    "meta-llama/Meta-Llama-3.1-70B": 70.0,
    "meta-llama/Meta-Llama-3.1-405B": 405.0,
    # Mistral
    "mistralai/Mistral-7B-v0.1": 7.0,
    "mistralai/Mistral-7B-v0.3": 7.0,
    "mistralai/Mixtral-8x7B-v0.1": 46.7,
    "mistralai/Mixtral-8x22B-v0.1": 141.0,
    # Phi
    "microsoft/phi-2": 2.7,
    "microsoft/Phi-3-mini-4k-instruct": 3.8,
    "microsoft/Phi-3-medium-4k-instruct": 14.0,
    # Qwen
    "Qwen/Qwen2-7B": 7.0,
    "Qwen/Qwen2-72B": 72.0,
    "Qwen/Qwen2.5-7B": 7.0,
    "Qwen/Qwen2.5-72B": 72.0,
    # Gemma
    "google/gemma-2b": 2.0,
    "google/gemma-7b": 7.0,
    "google/gemma-2-9b": 9.0,
    "google/gemma-2-27b": 27.0,
    # GPT-2
    "openai-community/gpt2": 0.124,
    "openai-community/gpt2-medium": 0.355,
    "openai-community/gpt2-large": 0.774,
    "openai-community/gpt2-xl": 1.5,
    # Falcon
    "tiiuae/falcon-7b": 7.0,
    "tiiuae/falcon-40b": 40.0,
}

# Bytes per parameter for each precision
_BYTES_PER_PARAM: dict[str, float] = {
    "float32": 4.0,
    "fp32": 4.0,
    "float16": 2.0,
    "fp16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "8bit": 1.0,
    "int4": 0.5,
    "4bit": 0.5,
    "nf4": 0.5,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def check_cuda_available() -> bool:
    """Check whether CUDA is available via PyTorch.

    Returns
    -------
    bool
        *True* if ``torch.cuda.is_available()`` returns *True*.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_mps_available() -> bool:
    """Check whether Apple Metal Performance Shaders backend is available."""
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def get_gpu_memory_usage() -> list[GPUInfo]:
    """Return current VRAM usage for every detected CUDA GPU.

    Returns
    -------
    list[GPUInfo]
        One entry per GPU.  Empty list if no GPUs are detected.
    """
    gpus: list[GPUInfo] = []

    try:
        import torch

        if not torch.cuda.is_available():
            return gpus

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024**2)  # bytes -> MB
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**2)
            torch.cuda.memory_allocated(i) / (1024**2)
            # "used" from PyTorch's perspective is the reserved memory
            mem_free = mem_total - mem_reserved

            gpu = GPUInfo(
                index=i,
                name=props.name,
                total_memory_mb=mem_total,
                used_memory_mb=mem_reserved,
                free_memory_mb=mem_free,
            )
            gpus.append(gpu)
    except ImportError:
        pass

    # Enrich with nvidia-smi data if available
    _enrich_with_nvidia_smi(gpus)

    return gpus


def _enrich_with_nvidia_smi(gpus: list[GPUInfo]) -> None:
    """Try to fill in temperature, utilization, driver version from nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "temperature.gpu,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            idx = int(parts[0])

            # If we already have a GPUInfo for this index, enrich it
            matching = [g for g in gpus if g.index == idx]
            if matching:
                gpu = matching[0]
            else:
                # PyTorch not available; create from nvidia-smi
                gpu = GPUInfo(
                    index=idx,
                    name=parts[1],
                    total_memory_mb=float(parts[2]),
                    used_memory_mb=float(parts[3]),
                    free_memory_mb=float(parts[4]),
                )
                gpus.append(gpu)

            with contextlib.suppress(ValueError, IndexError):
                gpu.temperature_c = float(parts[5])
            with contextlib.suppress(ValueError, IndexError):
                gpu.utilization_pct = float(parts[6])
            gpu.driver_version = parts[7] if parts[7] else None

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass


def get_system_gpu_info() -> SystemGPUInfo:
    """Gather comprehensive GPU/system information.

    Returns
    -------
    SystemGPUInfo
        Aggregated system GPU information.
    """
    info = SystemGPUInfo()

    try:
        import torch

        info.torch_version = torch.__version__
        info.cuda_available = torch.cuda.is_available()
        info.mps_available = _check_mps_available()

        if info.cuda_available:
            info.cuda_version = torch.version.cuda
            info.gpu_count = torch.cuda.device_count()
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                info.cudnn_version = str(torch.backends.cudnn.version())
            info.gpus = get_gpu_memory_usage()
            if info.gpus and info.gpus[0].driver_version:
                info.driver_version = info.gpus[0].driver_version
    except ImportError as exc:
        info.error = f"PyTorch not installed: {exc}"
    except Exception as exc:
        info.error = str(exc)

    return info


def get_available_vram() -> dict[int, float]:
    """Return available VRAM in GB for each GPU.

    Returns
    -------
    dict[int, float]
        Mapping from GPU index to free VRAM in gigabytes.
    """
    gpus = get_gpu_memory_usage()
    return {g.index: g.free_memory_gb for g in gpus}


def estimate_model_memory(
    model_name: str,
    precision: str = "float16",
    include_optimizer: bool = False,
    include_gradients: bool = False,
    lora_rank: int | None = None,
) -> dict[str, float]:
    """Estimate VRAM requirements for loading / training a model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``"meta-llama/Llama-2-7b"``).
    precision:
        Weight precision.  One of ``float32``, ``float16`` / ``bf16``,
        ``int8`` / ``8bit``, ``int4`` / ``4bit`` / ``nf4``.
    include_optimizer:
        Add estimate for AdamW optimizer states (2x model weights for
        fp32 master weights + momentum + variance).
    include_gradients:
        Add estimate for gradient storage.
    lora_rank:
        If set, estimate LoRA adapter memory instead of full model
        gradients/optimizer.

    Returns
    -------
    dict[str, float]
        Breakdown of estimated memory usage in GB:
        ``model_weights``, ``gradients``, ``optimizer``, ``activations``,
        ``total``.
    """
    # Resolve parameter count
    params_b = _resolve_params_b(model_name)
    params = params_b * 1e9  # convert to raw count

    bytes_per = _BYTES_PER_PARAM.get(precision.lower(), 2.0)

    # Model weights
    model_gb = (params * bytes_per) / (1024**3)

    # Gradients (same size as weights in training precision)
    grad_gb = 0.0
    if include_gradients:
        if lora_rank is not None:
            # LoRA trainable params ~ 2 * rank * hidden_dim * num_layers
            # Rough heuristic: trainable fraction ~ rank * 0.001
            trainable_fraction = min(lora_rank * 0.001, 0.05)
            grad_gb = model_gb * trainable_fraction
        else:
            grad_gb = model_gb  # full fine-tuning

    # Optimizer states (AdamW: 2 extra copies in fp32)
    opt_gb = 0.0
    if include_optimizer:
        if lora_rank is not None:
            trainable_fraction = min(lora_rank * 0.001, 0.05)
            # Optimizer holds fp32 copies + momentum + variance
            opt_gb = (params * trainable_fraction * 4.0 * 3) / (1024**3)
        else:
            # Full fine-tuning: fp32 master weights + momentum + variance
            opt_gb = (params * 4.0 * 3) / (1024**3)

    # Activation memory (rough heuristic: 10-20% of model weights)
    activation_gb = model_gb * 0.15

    total_gb = model_gb + grad_gb + opt_gb + activation_gb

    return {
        "model_weights_gb": round(model_gb, 2),
        "gradients_gb": round(grad_gb, 2),
        "optimizer_gb": round(opt_gb, 2),
        "activations_gb": round(activation_gb, 2),
        "total_gb": round(total_gb, 2),
        "params_billion": round(params_b, 2),
        "precision": precision,
    }


def _resolve_params_b(model_name: str) -> float:
    """Resolve the parameter count (in billions) for a model.

    First checks the known-model table, then attempts to parse the name
    (e.g. ``"Llama-2-7b"`` -> 7.0), and falls back to fetching the
    config from HuggingFace Hub.
    """
    # Direct lookup
    if model_name in _KNOWN_MODEL_PARAMS:
        return _KNOWN_MODEL_PARAMS[model_name]

    # Try case-insensitive / partial match
    lower = model_name.lower()
    for key, val in _KNOWN_MODEL_PARAMS.items():
        if key.lower() == lower:
            return val

    # Try to parse from name  (e.g., "70b", "7B", "2.7b")
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if match:
        return float(match.group(1))

    # Try fetching from HuggingFace Hub config
    try:
        import json

        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Common config keys for parameter estimation
        hidden = config.get("hidden_size", config.get("d_model", 4096))
        layers = config.get("num_hidden_layers", config.get("n_layer", 32))
        vocab = config.get("vocab_size", 32000)
        intermediate = config.get("intermediate_size", config.get("d_ff", hidden * 4))

        # Rough transformer parameter count
        # Attention: 4 * hidden^2 per layer
        # FFN: 2 * hidden * intermediate per layer (for gated: 3 * hidden * intermediate)
        # Embeddings: vocab * hidden
        attn_params = 4 * hidden * hidden * layers
        ffn_params = 3 * hidden * intermediate * layers  # assume gated
        embed_params = vocab * hidden * 2  # input + output embeddings
        total = attn_params + ffn_params + embed_params
        return total / 1e9
    except Exception:
        pass

    # Default fallback - guess from name or return a conservative 7B
    return 7.0


def format_gpu_info(console: Console | None = None) -> None:
    """Pretty-print GPU and system information to the console.

    Parameters
    ----------
    console:
        Rich Console instance.  If *None*, creates a new one.
    """
    if console is None:
        console = Console()

    info = get_system_gpu_info()

    # System table
    sys_table = Table(title="System Information", show_header=False, expand=True)
    sys_table.add_column("Property", style="bold cyan", width=24)
    sys_table.add_column("Value")

    sys_table.add_row("Platform", platform.platform())
    sys_table.add_row("Python", sys.version.split()[0])
    sys_table.add_row("PyTorch", info.torch_version or "[dim]not installed[/dim]")
    sys_table.add_row(
        "CUDA Available",
        "[green]Yes[/green]" if info.cuda_available else "[red]No[/red]",
    )
    sys_table.add_row("CUDA Version", info.cuda_version or "N/A")
    sys_table.add_row("cuDNN Version", info.cudnn_version or "N/A")
    sys_table.add_row("GPU Driver", info.driver_version or "N/A")
    sys_table.add_row("GPU Count", str(info.gpu_count))
    sys_table.add_row(
        "MPS (Apple Metal)",
        "[green]Yes[/green]" if info.mps_available else "[dim]No[/dim]",
    )

    console.print(sys_table)

    if info.gpus:
        gpu_table = Table(title="GPU Details", expand=True)
        gpu_table.add_column("GPU", style="bold")
        gpu_table.add_column("Name")
        gpu_table.add_column("Total VRAM", justify="right")
        gpu_table.add_column("Used", justify="right")
        gpu_table.add_column("Free", justify="right")
        gpu_table.add_column("Temp", justify="right")
        gpu_table.add_column("Util", justify="right")

        for g in info.gpus:
            temp_str = f"{g.temperature_c:.0f} C" if g.temperature_c else "N/A"
            gpu_table.add_row(
                str(g.index),
                g.name,
                f"{g.total_memory_gb:.1f} GB",
                f"{g.used_memory_gb:.1f} GB",
                f"{g.free_memory_gb:.1f} GB",
                temp_str,
                g.utilization_str,
            )

        console.print(gpu_table)

    if info.error:
        console.print(f"[yellow]Warning:[/yellow] {info.error}")

    # Installed packages summary
    _print_package_versions(console)


def _print_package_versions(console: Console) -> None:
    """Print versions of key ML packages."""
    packages = [
        "transformers",
        "peft",
        "trl",
        "accelerate",
        "datasets",
        "bitsandbytes",
        "safetensors",
        "wandb",
        "deepspeed",
        "vllm",
        "gradio",
        "fastapi",
        "chromadb",
        "sentence_transformers",
        "lm_eval",
    ]

    pkg_table = Table(title="Installed Packages", expand=True)
    pkg_table.add_column("Package", style="bold")
    pkg_table.add_column("Version")

    import importlib.metadata

    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg.replace("_", "-"))
            pkg_table.add_row(pkg, f"[green]{version}[/green]")
        except importlib.metadata.PackageNotFoundError:
            pkg_table.add_row(pkg, "[dim]not installed[/dim]")

    console.print(pkg_table)
