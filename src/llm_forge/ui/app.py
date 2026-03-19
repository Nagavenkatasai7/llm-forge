"""LLM Forge Dashboard — Gradio Blocks UI for config, training, and chat.

Three-tab interface:
  1. Configure — Visual config builder with live YAML preview
  2. Training  — Launch and monitor training runs
  3. Chat      — Interactive model inference (reuses serving.gradio_app)

Every UI action reads/writes standard YAML configs. The UI is a visual
layer on top of the same config system the CLI uses.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from collections.abc import Generator
from pathlib import Path
from typing import Any

import yaml

from llm_forge.utils.logging import get_logger

logger = get_logger("ui.app")

try:
    import gradio as gr

    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS_DIR = _PROJECT_ROOT / "configs"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"


def _list_config_files() -> list[str]:
    """Return sorted list of YAML config filenames from the configs/ dir."""
    if not _CONFIGS_DIR.exists():
        return []
    return sorted(p.name for p in _CONFIGS_DIR.glob("*.yaml") if p.is_file())


def _load_config_yaml(filename: str) -> str:
    """Load raw YAML text from a config file."""
    path = _CONFIGS_DIR / filename
    if not path.exists():
        return f"# File not found: {filename}"
    return path.read_text(encoding="utf-8")


def _validate_yaml(yaml_text: str) -> str:
    """Validate YAML text against LLMForgeConfig, return status message."""
    if not yaml_text or not yaml_text.strip():
        return "Nothing to validate — YAML preview is empty. Build a config first."

    import warnings

    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return f"YAML PARSE ERROR:\n{exc}"

    if not isinstance(raw, dict):
        return "ERROR: YAML must be a mapping at the top level."

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from llm_forge.config.validator import validate_config_dict

            validate_config_dict(raw)

        # Cross-field validation: assistant_only_loss vs data format
        extra_warnings: list[str] = []
        data_format = raw.get("data", {}).get("format", "alpaca")
        assistant_loss = raw.get("training", {}).get("assistant_only_loss", False)
        non_conversational = {"alpaca", "completion", "custom"}

        if assistant_loss and data_format in non_conversational:
            extra_warnings.append(
                f"assistant_only_loss=True but data format is '{data_format}' "
                "(non-conversational). This WILL cause a training error. "
                "Either uncheck 'Assistant-Only Loss' or change format to 'sharegpt'."
            )

        msg = "VALID — config passes all checks."
        all_warnings = [f"  - {w.message}" for w in caught] + [f"  - {w}" for w in extra_warnings]
        if all_warnings:
            msg = "VALID (with warnings):" if not extra_warnings else "WARNING — config may fail:"
            msg += "\n" + "\n".join(all_warnings)
        return msg
    except Exception as exc:
        return f"VALIDATION FAILED:\n{exc}"


def _build_yaml_from_ui(
    # Model
    model_name: str,
    max_seq_length: int,
    attn_implementation: str,
    torch_dtype: str,
    # LoRA
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
    # Data
    train_path: str,
    data_format: str,
    system_prompt: str,
    test_size: float,
    max_samples: int | None,
    cleaning_enabled: bool,
    # Training
    training_mode: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    lr_scheduler: str,
    warmup_ratio: float,
    bf16: bool,
    gradient_checkpointing: bool,
    completion_only_loss: bool,
    assistant_only_loss: bool,
    neftune_alpha: float | None,
    # Evaluation
    eval_enabled: bool,
    # Serving
    export_format: str | None,
    gguf_quant: str,
    merge_adapter: bool,
    generate_modelfile: bool,
) -> str:
    """Build a YAML config string from UI form values."""
    config: dict[str, Any] = {}

    # Model section
    config["model"] = {
        "name": model_name,
        "max_seq_length": int(max_seq_length),
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
    }

    # LoRA section (only for lora/qlora modes)
    if training_mode in ("lora", "qlora"):
        modules = [m.strip() for m in target_modules.split(",") if m.strip()]
        config["lora"] = {
            "r": int(lora_r),
            "alpha": int(lora_alpha),
            "dropout": float(lora_dropout),
            "target_modules": modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    # Data section
    data: dict[str, Any] = {
        "train_path": train_path,
        "format": data_format,
        "test_size": float(test_size),
        "seed": 42,
    }
    if system_prompt and system_prompt.strip():
        data["system_prompt"] = system_prompt.strip()
    if max_samples and int(max_samples) > 0:
        data["max_samples"] = int(max_samples)
    data["cleaning"] = {"enabled": cleaning_enabled}
    config["data"] = data

    # Training section
    training: dict[str, Any] = {
        "mode": training_mode,
        "output_dir": output_dir,
        "num_epochs": int(num_epochs),
        "per_device_train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_accum),
        "learning_rate": float(learning_rate),
        "lr_scheduler_type": lr_scheduler,
        "warmup_ratio": float(warmup_ratio),
        "bf16": bf16,
        "fp16": False,
        "gradient_checkpointing": gradient_checkpointing,
        "completion_only_loss": completion_only_loss,
        "assistant_only_loss": assistant_only_loss,
        "logging_steps": 10,
        "save_steps": 100,
        "report_to": ["none"],
    }
    if neftune_alpha and float(neftune_alpha) > 0:
        training["neftune_noise_alpha"] = float(neftune_alpha)
    config["training"] = training

    # Evaluation
    config["evaluation"] = {"enabled": eval_enabled}

    # Serving / export
    serving: dict[str, Any] = {"merge_adapter": merge_adapter}
    if export_format and export_format != "none":
        serving["export_format"] = export_format
        if export_format == "gguf":
            serving["gguf_quantization"] = gguf_quant
        serving["generate_modelfile"] = generate_modelfile
    config["serving"] = serving

    return yaml.dump(config, default_flow_style=False, sort_keys=False, width=120)


# ---------------------------------------------------------------------------
# Model path discovery (for Chat tab)
# ---------------------------------------------------------------------------


def _discover_model_paths() -> list[str]:
    """Scan outputs/ for directories that look like trained models.

    Looks for:
      - outputs/<name>/merged/          (merged LoRA weights)
      - outputs/<name>/                 (direct fine-tune output)
      - Any dir containing config.json or adapter_config.json
    """
    paths: list[str] = []
    if not _OUTPUTS_DIR.exists():
        return paths

    def _rel(p: Path) -> str:
        """Return path relative to project root for shorter display."""
        try:
            return str(p.relative_to(_PROJECT_ROOT))
        except ValueError:
            return str(p)

    for run_dir in sorted(_OUTPUTS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        # Check for merged/ subdirectory (best choice — fully merged model)
        merged = run_dir / "merged"
        if merged.is_dir() and (merged / "config.json").exists():
            paths.append(_rel(merged))
            continue

        # Check for adapter_config.json (LoRA adapter, needs base model)
        if (run_dir / "adapter_config.json").exists():
            paths.append(_rel(run_dir))
            continue

        # Check for config.json (full model)
        if (run_dir / "config.json").exists():
            paths.append(_rel(run_dir))
            continue

        # Check subdirectories (e.g. checkpoint-500/)
        for sub in sorted(run_dir.iterdir()):
            if sub.is_dir() and (sub / "config.json").exists():
                paths.append(_rel(sub))
                break

    return paths


# ---------------------------------------------------------------------------
# Training runner (background thread)
# ---------------------------------------------------------------------------

_training_lock = threading.Lock()
_training_thread: threading.Thread | None = None
_training_log_queue: queue.Queue = queue.Queue()
_training_stop_event = threading.Event()
_training_status: dict[str, Any] = {
    "running": False,
    "stage": "",
    "stages": [],
    "error": None,
    "start_time": None,
}


class _QueueLogHandler(logging.Handler):
    """Logging handler that pushes records into a thread-safe queue."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.q.put(msg)
        except Exception:
            pass


def _run_training_thread(config_yaml: str) -> None:
    """Run training in a background thread, streaming logs to the queue."""
    global _training_status

    # Install queue handler on root llm_forge logger
    handler = _QueueLogHandler(_training_log_queue)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    )
    root_logger = logging.getLogger("llm_forge")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    try:
        _training_status["running"] = True
        _training_status["error"] = None
        _training_status["start_time"] = time.time()
        _training_stop_event.clear()
        _training_log_queue.put("[UI] Parsing configuration...")

        raw = yaml.safe_load(config_yaml)

        _training_log_queue.put("[UI] Loading pipeline runner...")
        from llm_forge.pipeline.pipeline_runner import PipelineRunner

        runner = PipelineRunner()
        _training_log_queue.put("[UI] Starting training pipeline...")

        runner.run(raw, stop_event=_training_stop_event)

        if _training_stop_event.is_set():
            _training_log_queue.put("[UI] Training stopped by user.")
            _training_status["stage"] = "stopped"
        else:
            _training_log_queue.put("[UI] Training completed successfully!")
            _training_status["stage"] = "completed"

    except Exception as exc:
        tb = traceback.format_exc()
        _training_log_queue.put(f"[UI] TRAINING FAILED: {exc}\n{tb}")
        _training_status["error"] = str(exc)
        _training_status["stage"] = "failed"
    finally:
        _training_status["running"] = False
        root_logger.removeHandler(handler)


def _start_training(config_yaml: str) -> str:
    """Start training in a background thread. Returns status message."""
    global _training_thread

    if _training_status["running"]:
        return "Training is already running. Wait for it to finish."

    # Guard against empty/None config
    if not config_yaml or not config_yaml.strip():
        return (
            "Cannot start — no config loaded.\n\n"
            "Go to the Configure tab first:\n"
            "  1. Fill out the form fields, OR\n"
            "  2. Select a preset and click 'Load'\n"
            "Then come back here and click 'Start Training'."
        )

    # Validate first
    try:
        raw = yaml.safe_load(config_yaml)
    except yaml.YAMLError as exc:
        return f"Cannot start — YAML parse error: {exc}"

    if not isinstance(raw, dict):
        return "Cannot start — YAML is not a valid config (must be a mapping)."

    # Clear log queue
    while not _training_log_queue.empty():
        try:
            _training_log_queue.get_nowait()
        except queue.Empty:
            break

    _training_status["stage"] = "starting"
    _training_status["stages"] = []

    _training_thread = threading.Thread(
        target=_run_training_thread,
        args=(config_yaml,),
        daemon=True,
    )
    _training_thread.start()

    return "RUNNING — Training started! Logs will appear below automatically."


def _get_training_logs() -> str:
    """Drain the log queue and return accumulated log text."""
    lines = []
    while not _training_log_queue.empty():
        try:
            lines.append(_training_log_queue.get_nowait())
        except queue.Empty:
            break
    return "\n".join(lines)


def _get_training_status() -> str:
    """Return current training status as a formatted string."""
    if _training_status["running"]:
        elapsed = time.time() - (_training_status["start_time"] or time.time())
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"RUNNING — {mins}m {secs}s elapsed"
    elif _training_status["error"]:
        return f"FAILED — {_training_status['error'][:200]}"
    elif _training_status["stage"] == "completed":
        return "COMPLETED"
    elif _training_status["stage"] == "stopped":
        return "STOPPED by user"
    else:
        return "IDLE — No training in progress"


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------


def _get_system_info() -> str:
    """Return formatted system info string."""
    try:
        from llm_forge.config.hardware_detector import detect_hardware

        hw = detect_hardware()
        return hw.summary()
    except Exception as exc:
        return f"Hardware detection failed: {exc}"


# ---------------------------------------------------------------------------
# Build the Gradio Blocks app
# ---------------------------------------------------------------------------


def build_app() -> tuple[gr.Blocks, Any, str]:
    """Construct the 3-tab Gradio Blocks dashboard.

    Returns (app, theme, css) — theme and css are passed to launch() in Gradio 6.
    """
    if not _GRADIO_AVAILABLE:
        raise ImportError("gradio is required for the UI. Install with: pip install gradio")

    config_files = _list_config_files()
    default_config = config_files[0] if config_files else ""

    _theme = gr.themes.Soft()
    _css = """
    .yaml-preview { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; }
    .status-box { font-weight: bold; }
    .help-text { color: #666; font-size: 0.85em; line-height: 1.4; }
    """

    with gr.Blocks(title="LLM Forge") as app:
        # ---- Header ----
        gr.Markdown(
            "# LLM Forge Dashboard\n"
            "**Config-driven LLM fine-tuning platform.**  \n"
            "Build training configs visually, launch and monitor training runs, "
            "and chat with your fine-tuned models — all from this dashboard.  \n"
            "Every setting here maps to a YAML config field. "
            "The same configs work with the `llm-forge` CLI."
        )

        with gr.Tabs():
            # ==============================================================
            # TAB 1: Configure
            # ==============================================================
            with gr.Tab("Configure", id="configure"):
                gr.Markdown(
                    "### Build and Edit Training Configurations\n"
                    "Use the form on the left to set your training parameters. "
                    "The YAML preview on the right updates live as you make changes. "
                    "You can also edit the YAML directly.  \n"
                    "**Quick start:** Select a preset from the dropdown and click **Load** "
                    "to populate the form with a working config."
                )

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=config_files,
                        value=default_config,
                        label="Load Preset",
                        scale=3,
                        info=(
                            "Pre-built configs from the configs/ directory. "
                            "Pick one and click 'Load' to populate all fields."
                        ),
                    )
                    load_btn = gr.Button("Load", scale=1)
                    validate_btn = gr.Button("Validate", variant="primary", scale=1)
                    save_btn = gr.Button("Save Config", scale=1)

                with gr.Row(equal_height=True):
                    # ---- Left: Form editor ----
                    with gr.Column(scale=3):
                        # -- Model Settings --
                        with gr.Accordion("Model Settings", open=True):
                            gr.Markdown(
                                "<span class='help-text'>"
                                "Choose which pre-trained model to fine-tune. "
                                "Pick from the catalog, search HuggingFace, or type a model ID directly."
                                "</span>"
                            )
                            with gr.Row():
                                model_name = gr.Textbox(
                                    label="Model Name",
                                    value="unsloth/Llama-3.2-1B-Instruct",
                                    info=(
                                        "HuggingFace model ID or local path. "
                                        "Use the Browse / Search tabs below to find models."
                                    ),
                                    scale=3,
                                )
                                torch_dtype = gr.Dropdown(
                                    choices=["bf16", "fp16", "fp32"],
                                    value="bf16",
                                    label="Precision",
                                    scale=1,
                                    info=(
                                        "bf16 = best for modern GPUs (A100, H100, Apple M-series). "
                                        "fp16 = use if bf16 not supported. "
                                        "fp32 = full precision, uses 2x memory."
                                    ),
                                )

                            # -- Model Browser --
                            with gr.Accordion("Browse & Search Models", open=False), gr.Tabs():
                                with gr.Tab("Browse Catalog"):
                                    from llm_forge.ui.model_catalog import (
                                        get_catalog_choices,
                                        get_catalog_table,
                                        get_model_info,
                                    )

                                    gr.Markdown(
                                        "Pick a model from our curated list of models "
                                        "known to work well with llm-forge."
                                    )
                                    catalog_dropdown = gr.Dropdown(
                                        choices=get_catalog_choices(),
                                        label="Select Model",
                                        info="Curated models grouped by size and hardware requirement.",
                                    )
                                    catalog_info = gr.Markdown("")
                                    catalog_use_btn = gr.Button(
                                        "Use This Model",
                                        variant="primary",
                                        size="sm",
                                    )
                                    with gr.Accordion("Full Catalog", open=False):
                                        gr.Markdown(get_catalog_table())

                                with gr.Tab("Search HuggingFace"):
                                    gr.Markdown(
                                        "Search the HuggingFace Hub for any text-generation model."
                                    )
                                    with gr.Row():
                                        hub_search_input = gr.Textbox(
                                            label="Search",
                                            placeholder="e.g. llama, mistral, phi, qwen...",
                                            scale=3,
                                        )
                                        hub_search_btn = gr.Button(
                                            "Search",
                                            variant="primary",
                                            scale=1,
                                        )
                                    hub_results_display = gr.Markdown(
                                        "Enter a search term and click Search."
                                    )
                                    hub_results_state = gr.State(value=[])
                                    hub_pick_dropdown = gr.Dropdown(
                                        choices=[],
                                        label="Select from results",
                                        visible=False,
                                    )
                                    hub_use_btn = gr.Button(
                                        "Use This Model",
                                        variant="primary",
                                        size="sm",
                                        visible=False,
                                    )
                            with gr.Row():
                                max_seq_length = gr.Slider(
                                    minimum=128,
                                    maximum=8192,
                                    value=2048,
                                    step=128,
                                    label="Max Sequence Length",
                                    info=(
                                        "Maximum number of tokens per training sample. "
                                        "512 = short Q&A. 1024 = typical instruction tuning. "
                                        "2048 = multi-turn conversations. 4096+ = long documents. "
                                        "Higher values use more memory."
                                    ),
                                )
                                attn_impl = gr.Dropdown(
                                    choices=["sdpa", "flash_attention_2", "eager"],
                                    value="sdpa",
                                    label="Attention Implementation",
                                    info=(
                                        "sdpa = PyTorch's built-in scaled dot-product attention (recommended, works everywhere). "
                                        "flash_attention_2 = fastest, but requires NVIDIA GPU + flash-attn package installed. "
                                        "eager = basic implementation, slowest but most compatible."
                                    ),
                                )

                        # -- LoRA Settings --
                        with gr.Accordion("LoRA (Low-Rank Adaptation)", open=True):
                            gr.Markdown(
                                "<span class='help-text'>"
                                "LoRA trains a small set of adapter weights instead of modifying the entire model. "
                                "This dramatically reduces memory usage and training time while preserving the base model's knowledge. "
                                "These settings only apply when Training Mode is 'lora' or 'qlora'."
                                "</span>"
                            )
                            with gr.Row():
                                lora_r = gr.Slider(
                                    minimum=1,
                                    maximum=128,
                                    value=8,
                                    step=1,
                                    label="Rank (r)",
                                    info=(
                                        "Controls adapter capacity. "
                                        "r=8: minimal forgetting, good for small datasets (<10K). "
                                        "r=16: balanced choice for most tasks. "
                                        "r=32-64: more capacity, risk of forgetting base knowledge. "
                                        "r=128: near full fine-tune."
                                    ),
                                )
                                lora_alpha = gr.Slider(
                                    minimum=1,
                                    maximum=256,
                                    value=16,
                                    step=1,
                                    label="Alpha",
                                    info=(
                                        "Scaling factor. The effective learning scale is alpha/r. "
                                        "Common practice: set alpha = 2 * r (e.g. r=8, alpha=16). "
                                        "Higher alpha = stronger adapter effect."
                                    ),
                                )
                                lora_dropout = gr.Slider(
                                    minimum=0.0,
                                    maximum=0.5,
                                    value=0.05,
                                    step=0.01,
                                    label="Dropout",
                                    info=(
                                        "Regularization to prevent overfitting. "
                                        "0.0 = no dropout. 0.05 = light (recommended for most). "
                                        "0.1-0.2 = stronger, use for small datasets."
                                    ),
                                )
                            target_modules = gr.Textbox(
                                label="Target Modules",
                                value="q_proj, v_proj, k_proj, o_proj",
                                info=(
                                    "Which model layers to apply LoRA to (comma-separated). "
                                    "Attention-only: 'q_proj, v_proj, k_proj, o_proj' — safest, least forgetting. "
                                    "All linear: 'all-linear' — more capacity but higher forgetting risk. "
                                    "For small datasets (<20K samples), stick with attention-only."
                                ),
                            )

                        # -- Data Settings --
                        with gr.Accordion("Data", open=True):
                            gr.Markdown(
                                "<span class='help-text'>"
                                "Configure your training dataset. You can use a HuggingFace dataset ID "
                                "(e.g. 'tatsu-lab/alpaca') or a path to a local JSONL/CSV/Parquet file."
                                "</span>"
                            )
                            train_path = gr.Textbox(
                                label="Training Data",
                                value="tatsu-lab/alpaca",
                                info=(
                                    "HuggingFace dataset ID: 'tatsu-lab/alpaca', 'yahma/alpaca-cleaned', etc. "
                                    "Local file: '/path/to/data.jsonl' or 'data/train.csv'. "
                                    "URL: 'https://example.com/data.jsonl' (will be downloaded)."
                                ),
                            )
                            with gr.Row():
                                data_format = gr.Dropdown(
                                    choices=["alpaca", "sharegpt", "completion", "custom"],
                                    value="alpaca",
                                    label="Format",
                                    info=(
                                        "alpaca: instruction/input/output fields (most common). "
                                        "sharegpt: multi-turn conversations with roles. "
                                        "completion: raw text for continued pre-training. "
                                        "custom: bring your own column mapping."
                                    ),
                                )
                                test_size = gr.Slider(
                                    minimum=0.01,
                                    maximum=0.3,
                                    value=0.05,
                                    step=0.01,
                                    label="Eval Split",
                                    info=(
                                        "Fraction of data held out for evaluation. "
                                        "0.05 (5%) is typical. Higher = more eval data, less training data."
                                    ),
                                )
                                max_samples = gr.Number(
                                    label="Max Samples",
                                    value=0,
                                    info=(
                                        "Limit dataset size for testing. "
                                        "0 = use all data. "
                                        "Set to 100-500 for quick test runs to verify your config works."
                                    ),
                                    precision=0,
                                )
                            system_prompt = gr.Textbox(
                                label="System Prompt",
                                value="",
                                lines=2,
                                info=(
                                    "A system message prepended to every training sample. "
                                    "Example: 'You are a helpful finance assistant.' "
                                    "Leave blank if your dataset already includes system prompts."
                                ),
                            )
                            cleaning_enabled = gr.Checkbox(
                                label="Enable Data Cleaning",
                                value=True,
                                info=(
                                    "Runs a multi-step cleaning pipeline: Unicode normalization, "
                                    "heuristic quality filtering (removes junk/low-quality), "
                                    "exact + fuzzy deduplication, and quality scoring. "
                                    "Recommended for raw/web-scraped data. "
                                    "Disable for pre-cleaned datasets to save time."
                                ),
                            )

                        # -- Training Settings --
                        with gr.Accordion("Training", open=True):
                            gr.Markdown(
                                "<span class='help-text'>"
                                "Core training hyperparameters. These control how the model learns from your data."
                                "</span>"
                            )
                            with gr.Row():
                                training_mode = gr.Dropdown(
                                    choices=["lora", "qlora", "full"],
                                    value="lora",
                                    label="Mode",
                                    info=(
                                        "lora: Train small adapter weights (recommended, ~1-4 GB VRAM). "
                                        "qlora: Same as LoRA but loads base model in 4-bit (saves ~50% VRAM). "
                                        "full: Train all model weights (requires much more VRAM, risk of forgetting)."
                                    ),
                                )
                                num_epochs = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=1,
                                    step=1,
                                    label="Epochs",
                                    info=(
                                        "How many times to iterate over the full dataset. "
                                        "1 epoch is usually sufficient for LoRA on <20K samples. "
                                        "2-3 for larger datasets. More epochs = higher overfitting risk."
                                    ),
                                )
                            with gr.Row():
                                batch_size = gr.Slider(
                                    minimum=1,
                                    maximum=32,
                                    value=4,
                                    step=1,
                                    label="Batch Size (per device)",
                                    info=(
                                        "Samples processed per GPU per step. "
                                        "Higher = faster training but more VRAM. "
                                        "If you get OOM (out of memory), reduce this first."
                                    ),
                                )
                                grad_accum = gr.Slider(
                                    minimum=1,
                                    maximum=32,
                                    value=4,
                                    step=1,
                                    label="Gradient Accumulation Steps",
                                    info=(
                                        "Accumulate gradients over this many steps before updating weights. "
                                        "Effective batch = batch_size x grad_accum. "
                                        "Use this to simulate larger batches when VRAM is limited."
                                    ),
                                )
                            eff_batch_display = gr.Textbox(
                                label="Effective Batch Size",
                                value="16",
                                interactive=False,
                                info="= Batch Size x Gradient Accumulation. This is the true batch size the optimizer sees.",
                            )
                            with gr.Row():
                                learning_rate = gr.Number(
                                    label="Learning Rate",
                                    value=2e-5,
                                    info=(
                                        "How fast the model learns. "
                                        "LoRA typical: 1e-5 to 5e-5 (0.00001 to 0.00005). "
                                        "Full fine-tune: 1e-6 to 1e-5. "
                                        "WARNING: Values above 1e-3 almost always cause catastrophic forgetting."
                                    ),
                                )
                                lr_scheduler = gr.Dropdown(
                                    choices=[
                                        "cosine",
                                        "linear",
                                        "constant",
                                        "constant_with_warmup",
                                    ],
                                    value="cosine",
                                    label="LR Scheduler",
                                    info=(
                                        "How the learning rate changes during training. "
                                        "cosine: smoothly decays to 0 (recommended, most stable). "
                                        "linear: linearly decays to 0. "
                                        "constant: stays fixed (can cause instability late in training)."
                                    ),
                                )
                                warmup_ratio = gr.Slider(
                                    minimum=0.0,
                                    maximum=0.3,
                                    value=0.03,
                                    step=0.01,
                                    label="Warmup Ratio",
                                    info=(
                                        "Fraction of training spent ramping up the learning rate from 0. "
                                        "0.03 (3%) is typical. Helps stabilize early training. "
                                        "0.0 = no warmup (jump straight to full LR)."
                                    ),
                                )
                            output_dir = gr.Textbox(
                                label="Output Directory",
                                value="outputs/my-model",
                                info=(
                                    "Where to save the trained model, checkpoints, and logs. "
                                    "Relative paths are relative to the project root. "
                                    "Example: 'outputs/my-finance-model'."
                                ),
                            )
                            with gr.Row():
                                bf16 = gr.Checkbox(
                                    label="BF16 Training",
                                    value=True,
                                    info=(
                                        "Use bfloat16 mixed precision. Faster and uses less memory. "
                                        "Requires modern GPU (A100+) or Apple M-series. "
                                        "Disable only if you see NaN errors."
                                    ),
                                )
                                gradient_checkpointing = gr.Checkbox(
                                    label="Grad. Checkpoint",
                                    value=True,
                                    info=(
                                        "Trades compute speed for VRAM savings (~30-40% less memory). "
                                        "Recommended for most setups. "
                                        "Disable only if you have plenty of VRAM and want faster training."
                                    ),
                                )
                                completion_only = gr.Checkbox(
                                    label="Completion-Only Loss",
                                    value=True,
                                    info=(
                                        "Only compute loss on the completion/output tokens, "
                                        "not on the instruction/input tokens. "
                                        "This teaches the model WHAT to generate, not to memorize prompts. "
                                        "Should almost always be enabled for instruction tuning."
                                    ),
                                )
                                assistant_only = gr.Checkbox(
                                    label="Assistant-Only Loss",
                                    value=True,
                                    info=(
                                        "Only compute loss on assistant turns in multi-turn conversations. "
                                        "Requires chat template with generation markers. "
                                        "Critical for instruction tuning — prevents the model from "
                                        "learning to predict system prompts and user messages."
                                    ),
                                )
                            neftune_alpha = gr.Number(
                                label="NEFTune Alpha",
                                value=0,
                                info=(
                                    "Adds noise to embedding vectors during training (NEFTune paper: arxiv:2310.05914). "
                                    "0 = disabled. 5.0 = recommended for large datasets (>50K samples). "
                                    "Can improve generalization but amplifies forgetting on small datasets. "
                                    "Start with 0 unless you have >50K samples."
                                ),
                                precision=1,
                            )

                        # -- Evaluation & Export --
                        with gr.Accordion("Evaluation & Export", open=False):
                            gr.Markdown(
                                "<span class='help-text'>"
                                "Post-training options: run benchmarks and export the model for deployment."
                                "</span>"
                            )
                            eval_enabled = gr.Checkbox(
                                label="Run Benchmarks After Training",
                                value=False,
                                info=(
                                    "Automatically evaluate the model on standard benchmarks (MMLU, ARC, etc.) "
                                    "after training completes. This takes extra time but gives you a quality score."
                                ),
                            )
                            with gr.Row():
                                export_format = gr.Dropdown(
                                    choices=["none", "safetensors", "gguf", "onnx"],
                                    value="none",
                                    label="Export Format",
                                    info=(
                                        "none: just save the trained weights (HuggingFace format). "
                                        "safetensors: optimized HuggingFace format for fast loading. "
                                        "gguf: for running in llama.cpp / Ollama (local inference). "
                                        "onnx: for deployment with ONNX Runtime."
                                    ),
                                )
                                gguf_quant = gr.Dropdown(
                                    choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "F16"],
                                    value="Q4_K_M",
                                    label="GGUF Quantization",
                                    info=(
                                        "Only applies when export format is 'gguf'. "
                                        "Q4_K_M: best balance of size and quality (recommended). "
                                        "Q8_0: highest quality quantization. "
                                        "F16: no quantization, largest file size."
                                    ),
                                )
                            with gr.Row():
                                merge_adapter = gr.Checkbox(
                                    label="Merge Adapter",
                                    value=True,
                                    info=(
                                        "Merge LoRA adapter weights back into the base model. "
                                        "Creates a standalone model that doesn't need the adapter files. "
                                        "Required for GGUF export. Recommended for deployment."
                                    ),
                                )
                                generate_modelfile = gr.Checkbox(
                                    label="Generate Modelfile",
                                    value=True,
                                    info=(
                                        "Create an Ollama Modelfile alongside GGUF export. "
                                        "This lets you import the model into Ollama with: "
                                        "'ollama create my-model -f Modelfile'."
                                    ),
                                )

                    # ---- Right: YAML preview & validation ----
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "<span class='help-text'>"
                            "This YAML updates live as you change settings on the left. "
                            "You can also edit it directly — it's the exact same format "
                            "used by the `llm-forge train --config` CLI command."
                            "</span>"
                        )
                        yaml_preview = gr.Code(
                            label="YAML Preview (editable)",
                            language="yaml",
                            lines=40,
                            interactive=True,
                            elem_classes=["yaml-preview"],
                        )
                        validation_output = gr.Textbox(
                            label="Validation Result",
                            lines=4,
                            interactive=False,
                        )
                        save_filename = gr.Textbox(
                            label="Save As",
                            value="my_config.yaml",
                            info=(
                                "Filename to save in the configs/ directory. "
                                "Saved configs appear in the 'Load Preset' dropdown."
                            ),
                        )

                # ---- Configure tab event handlers ----

                # All form inputs that affect the YAML
                form_inputs = [
                    model_name,
                    max_seq_length,
                    attn_impl,
                    torch_dtype,
                    lora_r,
                    lora_alpha,
                    lora_dropout,
                    target_modules,
                    train_path,
                    data_format,
                    system_prompt,
                    test_size,
                    max_samples,
                    cleaning_enabled,
                    training_mode,
                    output_dir,
                    num_epochs,
                    batch_size,
                    grad_accum,
                    learning_rate,
                    lr_scheduler,
                    warmup_ratio,
                    bf16,
                    gradient_checkpointing,
                    completion_only,
                    assistant_only,
                    neftune_alpha,
                    eval_enabled,
                    export_format,
                    gguf_quant,
                    merge_adapter,
                    generate_modelfile,
                ]

                def update_yaml(*args: Any) -> str:
                    return _build_yaml_from_ui(*args)

                def update_eff_batch(bs: int, ga: int) -> str:
                    return str(int(bs) * int(ga))

                # Live YAML update on any form change
                for inp in form_inputs:
                    inp.change(
                        fn=update_yaml,
                        inputs=form_inputs,
                        outputs=yaml_preview,
                    )

                # Effective batch size display
                batch_size.change(
                    fn=update_eff_batch,
                    inputs=[batch_size, grad_accum],
                    outputs=eff_batch_display,
                )
                grad_accum.change(
                    fn=update_eff_batch,
                    inputs=[batch_size, grad_accum],
                    outputs=eff_batch_display,
                )

                # Auto-toggle assistant_only_loss based on data format
                def _auto_toggle_assistant_loss(fmt: str) -> bool:
                    return fmt == "sharegpt"

                data_format.change(
                    fn=_auto_toggle_assistant_loss,
                    inputs=data_format,
                    outputs=assistant_only,
                )

                # Auto-link LoRA alpha = 2 * r
                alpha_linked = gr.State(value=True)

                def _update_alpha_from_r(
                    r: int,
                    alpha: int,
                    linked: bool,
                ) -> tuple[int, bool]:
                    if linked:
                        return int(r) * 2, True
                    return alpha, linked

                lora_r.change(
                    fn=_update_alpha_from_r,
                    inputs=[lora_r, lora_alpha, alpha_linked],
                    outputs=[lora_alpha, alpha_linked],
                )

                def _check_alpha_link(r: int, alpha: int) -> bool:
                    return int(alpha) == int(r) * 2

                lora_alpha.change(
                    fn=_check_alpha_link,
                    inputs=[lora_r, lora_alpha],
                    outputs=alpha_linked,
                )

                # -- Model browser event handlers --
                def _on_catalog_select(selected: str) -> str:
                    return get_model_info(selected) if selected else ""

                catalog_dropdown.change(
                    fn=_on_catalog_select,
                    inputs=catalog_dropdown,
                    outputs=catalog_info,
                )

                catalog_use_btn.click(
                    fn=lambda m: m if m else gr.update(),
                    inputs=catalog_dropdown,
                    outputs=model_name,
                )

                def _on_hub_search(query: str) -> tuple:
                    from llm_forge.ui.model_catalog import (
                        format_hub_results,
                        search_hub_models,
                    )

                    if not query or not query.strip():
                        return (
                            "Enter a search term.",
                            [],
                            gr.update(choices=[], visible=False),
                            gr.update(visible=False),
                        )
                    results = search_hub_models(query.strip())
                    md = format_hub_results(results)
                    ids = [r["id"] for r in results]
                    return (
                        md,
                        results,
                        gr.update(choices=ids, value=ids[0] if ids else None, visible=bool(ids)),
                        gr.update(visible=bool(ids)),
                    )

                hub_search_btn.click(
                    fn=_on_hub_search,
                    inputs=hub_search_input,
                    outputs=[
                        hub_results_display,
                        hub_results_state,
                        hub_pick_dropdown,
                        hub_use_btn,
                    ],
                )
                hub_search_input.submit(
                    fn=_on_hub_search,
                    inputs=hub_search_input,
                    outputs=[
                        hub_results_display,
                        hub_results_state,
                        hub_pick_dropdown,
                        hub_use_btn,
                    ],
                )

                hub_use_btn.click(
                    fn=lambda m: m if m else gr.update(),
                    inputs=hub_pick_dropdown,
                    outputs=model_name,
                )

                def load_preset_fn(filename: str) -> tuple:
                    """Load a preset and populate form fields + save filename."""
                    if not filename:
                        return (gr.update(),) * (len(form_inputs) + 2)

                    raw_yaml = _load_config_yaml(filename)
                    try:
                        cfg = yaml.safe_load(raw_yaml)
                    except yaml.YAMLError:
                        return (gr.update(),) * len(form_inputs) + (raw_yaml, filename)

                    if not isinstance(cfg, dict):
                        return (gr.update(),) * len(form_inputs) + (raw_yaml, filename)

                    m = cfg.get("model", {})
                    lo = cfg.get("lora", {})
                    d = cfg.get("data", {})
                    t = cfg.get("training", {})
                    e = cfg.get("evaluation", {})
                    s = cfg.get("serving", {})
                    cl = d.get("cleaning", {})

                    targets = lo.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
                    targets_str = ", ".join(targets) if isinstance(targets, list) else str(targets)

                    ef = s.get("export_format")
                    if ef is None:
                        ef = "none"

                    neft = t.get("neftune_noise_alpha")
                    if neft is None:
                        neft = 0

                    ms = d.get("max_samples")
                    if ms is None:
                        ms = 0

                    return (
                        # Model
                        m.get("name", "unsloth/Llama-3.2-1B-Instruct"),
                        m.get("max_seq_length", 2048),
                        m.get("attn_implementation", "sdpa"),
                        m.get("torch_dtype", "bf16"),
                        # LoRA
                        lo.get("r", 8),
                        lo.get("alpha", 16),
                        lo.get("dropout", 0.05),
                        targets_str,
                        # Data
                        d.get("train_path", "tatsu-lab/alpaca"),
                        d.get("format", "alpaca"),
                        d.get("system_prompt", ""),
                        d.get("test_size", 0.05),
                        ms,
                        cl.get("enabled", True),
                        # Training
                        t.get("mode", "lora"),
                        t.get("output_dir", "outputs/my-model"),
                        t.get("num_epochs", 1),
                        t.get("per_device_train_batch_size", 4),
                        t.get("gradient_accumulation_steps", 4),
                        t.get("learning_rate", 2e-5),
                        t.get("lr_scheduler_type", "cosine"),
                        t.get("warmup_ratio", 0.03),
                        t.get("bf16", True),
                        t.get("gradient_checkpointing", True),
                        t.get("completion_only_loss", True),
                        t.get("assistant_only_loss", True),
                        neft,
                        # Eval
                        e.get("enabled", False),
                        # Serving
                        ef,
                        s.get("gguf_quantization", "Q4_K_M"),
                        s.get("merge_adapter", True),
                        s.get("generate_modelfile", True),
                        # YAML preview
                        raw_yaml,
                        # Save As filename
                        filename,
                    )

                load_btn.click(
                    fn=load_preset_fn,
                    inputs=preset_dropdown,
                    outputs=form_inputs + [yaml_preview, save_filename],
                )

                validate_btn.click(
                    fn=_validate_yaml,
                    inputs=yaml_preview,
                    outputs=validation_output,
                )

                def save_config_fn(yaml_text: str, filename: str) -> str:
                    if not filename or not filename.endswith(".yaml"):
                        return "ERROR: Filename must end with .yaml"
                    path = _CONFIGS_DIR / filename
                    path.write_text(yaml_text, encoding="utf-8")
                    return f"Saved to {path}"

                save_btn.click(
                    fn=save_config_fn,
                    inputs=[yaml_preview, save_filename],
                    outputs=validation_output,
                )

            # ==============================================================
            # TAB 2: Training
            # ==============================================================
            with gr.Tab("Training", id="training"):
                gr.Markdown(
                    "### Launch and Monitor Training Runs\n"
                    "This tab uses the YAML config from the **Configure** tab. "
                    "Make sure you have a valid config before starting.  \n"
                    "**Steps:** Go to Configure tab -> build or load a config -> "
                    "click Validate -> come here -> click Start Training."
                )

                config_summary = gr.Textbox(
                    label="Active Config",
                    value="No config loaded — go to Configure tab first",
                    interactive=False,
                    max_lines=2,
                    info="Shows the config that will be used when you click Start Training.",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        training_status = gr.Textbox(
                            label="Status",
                            value="IDLE — No training in progress",
                            interactive=False,
                            elem_classes=["status-box"],
                        )
                        with gr.Row():
                            start_btn = gr.Button(
                                "Start Training",
                                variant="primary",
                                scale=2,
                            )
                            stop_btn = gr.Button(
                                "Stop Training",
                                variant="stop",
                                scale=1,
                            )
                            refresh_btn = gr.Button("Refresh Logs", scale=1)

                        gr.Markdown(
                            "**How it works:**\n"
                            "- Click **Start Training** to launch the full pipeline "
                            "(data loading -> cleaning -> preprocessing -> training -> export)\n"
                            "- Logs stream automatically every 3 seconds\n"
                            "- Training runs in the background — you can switch tabs\n"
                            "- If training fails, the error will appear in the logs below"
                        )

                        with gr.Accordion("System Info", open=False):
                            gr.Markdown(
                                "<span class='help-text'>"
                                "Click 'Detect Hardware' to see your CPU, RAM, GPU, and VRAM. "
                                "This helps you choose appropriate batch sizes and model sizes."
                                "</span>"
                            )
                            sys_info = gr.Textbox(
                                label="Hardware",
                                value="Click 'Detect Hardware' to see your system info",
                                lines=10,
                                interactive=False,
                            )
                            detect_hw_btn = gr.Button("Detect Hardware")

                    with gr.Column(scale=3):
                        training_logs = gr.Code(
                            label="Training Logs (auto-refreshes every 3s)",
                            language=None,
                            lines=35,
                            interactive=False,
                        )

                # ---- Training tab: config summary from Configure tab ----
                def _summarize_config(yaml_text: str) -> str:
                    if not yaml_text or not yaml_text.strip():
                        return "No config loaded — go to Configure tab first"
                    try:
                        cfg = yaml.safe_load(yaml_text)
                        model = cfg.get("model", {}).get("name", "?")
                        data = cfg.get("data", {}).get("train_path", "?")
                        mode = cfg.get("training", {}).get("mode", "?")
                        out = cfg.get("training", {}).get("output_dir", "?")
                        return f"Model: {model}  |  Data: {data}  |  Mode: {mode}  |  Output: {out}"
                    except Exception:
                        return "Config loaded (could not parse summary)"

                yaml_preview.change(
                    fn=_summarize_config,
                    inputs=yaml_preview,
                    outputs=config_summary,
                )

                # ---- Training tab state ----
                log_accumulator = gr.State(value="")

                def start_training_fn(config_yaml: str) -> tuple[str, str]:
                    msg = _start_training(config_yaml)
                    return msg, ""

                start_btn.click(
                    fn=start_training_fn,
                    inputs=yaml_preview,
                    outputs=[training_status, log_accumulator],
                )

                def stop_training_fn() -> str:
                    if not _training_status["running"]:
                        return "No training in progress."
                    _training_stop_event.set()
                    _training_log_queue.put(
                        "[UI] Stop requested — training will halt after the current step."
                    )
                    return "STOPPING — waiting for current step to finish..."

                stop_btn.click(
                    fn=stop_training_fn,
                    inputs=None,
                    outputs=training_status,
                )

                def refresh_logs_fn(accumulated: str) -> tuple[str, str, str]:
                    new_logs = _get_training_logs()
                    if new_logs:
                        accumulated = accumulated + ("\n" if accumulated else "") + new_logs
                    status = _get_training_status()
                    return accumulated, accumulated, status

                refresh_btn.click(
                    fn=refresh_logs_fn,
                    inputs=log_accumulator,
                    outputs=[training_logs, log_accumulator, training_status],
                )

                detect_hw_btn.click(
                    fn=_get_system_info,
                    inputs=None,
                    outputs=sys_info,
                )

                # Auto-refresh logs every 3 seconds while training
                timer = gr.Timer(value=3, active=True)
                timer.tick(
                    fn=refresh_logs_fn,
                    inputs=log_accumulator,
                    outputs=[training_logs, log_accumulator, training_status],
                )

            # ==============================================================
            # TAB 3: Chat
            # ==============================================================
            with gr.Tab("Chat", id="chat"):
                gr.Markdown(
                    "### Chat With Your Fine-Tuned Model\n"
                    "Load a trained model and test it interactively. "
                    "This helps you verify the model's behavior before deploying it."
                )

                # Model path discovery and guidance
                discovered_paths = _discover_model_paths()

                with gr.Accordion("Where to find your model path", open=not bool(discovered_paths)):
                    gr.Markdown(
                        "**After training completes**, your model is saved to the output directory "
                        "you configured. Here's where to look:\n\n"
                        "| What to look for | Path pattern | When to use |\n"
                        "|---|---|---|\n"
                        "| **Merged model** (best) | `outputs/<name>/merged/` | "
                        "After LoRA training with 'Merge Adapter' enabled |\n"
                        "| **LoRA adapter** | `outputs/<name>/` (has `adapter_config.json`) | "
                        "LoRA training without merging (needs base model too) |\n"
                        "| **Full model** | `outputs/<name>/` (has `config.json`) | "
                        "Full fine-tuning or merged output |\n"
                        "| **Checkpoint** | `outputs/<name>/checkpoint-500/` | "
                        "Mid-training checkpoint (for resuming or testing) |\n\n"
                        "**Your project's output directory:** `" + str(_OUTPUTS_DIR) + "/`\n\n"
                        "**Tip:** The merged model is always the best choice for chat. "
                        "It's a standalone model that works without the base model or adapter files."
                    )

                with gr.Row():
                    if discovered_paths:
                        chat_model_path = gr.Dropdown(
                            choices=discovered_paths,
                            value=discovered_paths[0] if discovered_paths else "",
                            label="Model Path",
                            allow_custom_value=True,
                            scale=3,
                            info=(
                                "Select a discovered model or type a custom path. "
                                "These were found in your outputs/ directory."
                            ),
                        )
                    else:
                        chat_model_path = gr.Textbox(
                            label="Model Path",
                            value="",
                            scale=3,
                            info=(
                                "No trained models found in outputs/. "
                                "After training, your model will appear here. "
                                "Enter a path like: outputs/my-model/merged/"
                            ),
                            placeholder="outputs/my-model/merged/",
                        )
                    chat_load_btn = gr.Button("Load Model", variant="primary", scale=1)

                chat_status = gr.Textbox(
                    label="Model Status",
                    value=(
                        "No model loaded. "
                        + (
                            f"Found {len(discovered_paths)} model(s) in outputs/ — select one above."
                            if discovered_paths
                            else "Train a model first, then come back here to test it."
                        )
                    ),
                    interactive=False,
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Chat",
                            height=450,
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Message",
                                placeholder="Type your message and press Enter or click Send...",
                                lines=2,
                                scale=4,
                            )
                            chat_send_btn = gr.Button("Send", variant="primary", scale=1)
                        chat_clear_btn = gr.Button("Clear Chat")

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**Chat Settings**\n\n"
                            "<span class='help-text'>"
                            "These control how the model generates responses."
                            "</span>"
                        )
                        chat_system_prompt = gr.Textbox(
                            label="System Prompt",
                            value="You are a helpful AI assistant.",
                            lines=3,
                            info=(
                                "Instructions that define the model's behavior. "
                                "The model sees this before every conversation. "
                                "Match this to what you used during training."
                            ),
                        )
                        chat_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.1,
                            step=0.05,
                            label="Temperature",
                            info=(
                                "Controls randomness. "
                                "0.0 = deterministic (always picks most likely token). "
                                "0.1 = almost deterministic (good for factual tasks). "
                                "0.7 = creative (good for writing). "
                                "1.0+ = very random."
                            ),
                        )
                        chat_max_tokens = gr.Slider(
                            minimum=16,
                            maximum=2048,
                            value=256,
                            step=16,
                            label="Max Tokens",
                            info=(
                                "Maximum length of the model's response. "
                                "256 = short answers. "
                                "512 = medium responses. "
                                "1024+ = long-form content."
                            ),
                        )

                # ---- Chat state ----
                chat_model_state = gr.State(value=None)

                def load_chat_model(path: str) -> tuple[str, Any]:
                    if not path or not path.strip():
                        return (
                            "Enter a model path first.\n\n"
                            "Look in: " + str(_OUTPUTS_DIR) + "/\n"
                            "For example: outputs/my-model/merged/"
                        ), None
                    model_dir = Path(path.strip())
                    if not model_dir.is_absolute():
                        model_dir = _PROJECT_ROOT / model_dir
                    if not model_dir.exists():
                        return (
                            f"Path does not exist: {model_dir}\n\n"
                            "Common issues:\n"
                            "  - Training hasn't completed yet\n"
                            "  - Wrong directory name\n"
                            "  - Need to use full absolute path\n\n"
                            f"Check: {_OUTPUTS_DIR}/"
                        ), None

                    # Verify it looks like a model directory
                    has_config = (model_dir / "config.json").exists()
                    has_adapter = (model_dir / "adapter_config.json").exists()
                    if not has_config and not has_adapter:
                        return (
                            f"Directory exists but doesn't look like a model: {model_dir}\n\n"
                            "A valid model directory should contain:\n"
                            "  - config.json (for full/merged models), OR\n"
                            "  - adapter_config.json (for LoRA adapters)\n\n"
                            "Try a subdirectory like: " + str(model_dir / "merged") + "/"
                        ), None

                    try:
                        from llm_forge.serving.gradio_app import GradioApp

                        chat_app = GradioApp.__new__(GradioApp)
                        chat_app.model_path = str(model_dir)
                        chat_app.config = None
                        chat_app.model = None
                        chat_app.tokenizer = None
                        chat_app._rag_retriever = None
                        chat_app._model_info = {}
                        chat_app._load_model()
                        info = chat_app._collect_model_info()
                        params = info.get("total_parameters_billions", "?")
                        model_type = info.get("model_type", "unknown")
                        return (
                            f"Model loaded successfully!\n"
                            f"  Type: {model_type}\n"
                            f"  Parameters: {params}B\n"
                            f"  Path: {model_dir}\n\n"
                            "You can now start chatting below."
                        ), chat_app
                    except Exception as exc:
                        return (
                            f"Failed to load model: {exc}\n\n"
                            "Common causes:\n"
                            "  - Not enough RAM/VRAM for this model\n"
                            "  - Corrupted or incomplete model files\n"
                            "  - Missing dependencies (torch, transformers)\n"
                            "  - For gated models: need HuggingFace token"
                        ), None

                chat_load_btn.click(
                    fn=load_chat_model,
                    inputs=chat_model_path,
                    outputs=[chat_status, chat_model_state],
                )

                def chat_user_msg(msg: str, history: list) -> tuple[str, list]:
                    if not msg.strip():
                        return "", history
                    history = history + [{"role": "user", "content": msg}]
                    return "", history

                def chat_bot_respond(
                    history: list,
                    model_state: Any,
                    sys_prompt: str,
                    temp: float,
                    max_tok: int,
                ) -> Generator[list, None, None]:
                    if not history or model_state is None:
                        yield history + [
                            {
                                "role": "assistant",
                                "content": "No model loaded. Load a model first using the controls above.",
                            }
                        ]
                        return

                    last_msg = history[-1]["content"]
                    prior = history[:-1]
                    history = history + [{"role": "assistant", "content": ""}]

                    try:
                        for partial in model_state._generate_streaming(
                            message=last_msg,
                            history=prior,
                            system_prompt=sys_prompt,
                            temperature=temp,
                            top_p=0.9,
                            top_k=40,
                            max_tokens=int(max_tok),
                            use_rag=False,
                        ):
                            history[-1]["content"] = partial
                            yield history
                    except Exception as exc:
                        history[-1]["content"] = f"Generation error: {exc}"
                        yield history

                chat_input.submit(
                    fn=chat_user_msg,
                    inputs=[chat_input, chatbot],
                    outputs=[chat_input, chatbot],
                    queue=False,
                ).then(
                    fn=chat_bot_respond,
                    inputs=[
                        chatbot,
                        chat_model_state,
                        chat_system_prompt,
                        chat_temperature,
                        chat_max_tokens,
                    ],
                    outputs=chatbot,
                )

                chat_send_btn.click(
                    fn=chat_user_msg,
                    inputs=[chat_input, chatbot],
                    outputs=[chat_input, chatbot],
                    queue=False,
                ).then(
                    fn=chat_bot_respond,
                    inputs=[
                        chatbot,
                        chat_model_state,
                        chat_system_prompt,
                        chat_temperature,
                        chat_max_tokens,
                    ],
                    outputs=chatbot,
                )

                chat_clear_btn.click(
                    fn=lambda: [],
                    inputs=None,
                    outputs=chatbot,
                    queue=False,
                )

    return app, _theme, _css


# ---------------------------------------------------------------------------
# Launch function
# ---------------------------------------------------------------------------


def launch_ui(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    desktop: bool = False,
) -> None:
    """Build and launch the LLM Forge dashboard.

    Parameters
    ----------
    host : str
        Network interface to bind to.
    port : int
        Port number for the web server.
    share : bool
        Whether to create a public Gradio share link.
    desktop : bool
        If True, wrap the Gradio app in a native desktop window
        using pywebview instead of opening a browser tab.
    """
    app, theme, css = build_app()

    if desktop:
        _launch_desktop(app, theme, css, host, port)
    else:
        logger.info("Launching LLM Forge UI on %s:%d", host, port)
        app.launch(server_name=host, server_port=port, share=share, theme=theme, css=css)


def _launch_desktop(
    app: gr.Blocks,
    theme: Any,
    css: str,
    host: str,
    port: int,
) -> None:
    """Launch Gradio in a native desktop window via pywebview."""
    try:
        import webview
    except ImportError:
        raise ImportError(
            "pywebview is required for desktop mode. "
            "Install with: pip install 'llm-forge[desktop]' or pip install pywebview"
        ) from None

    # Start Gradio server without blocking the main thread
    logger.info("Starting Gradio server on %s:%d (desktop mode)", host, port)
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        theme=theme,
        css=css,
        prevent_thread_lock=True,
    )

    url = f"http://{host}:{port}" if host != "0.0.0.0" else f"http://127.0.0.1:{port}"

    # Create native window — this blocks the main thread (required on macOS)
    logger.info("Opening native desktop window -> %s", url)
    webview.create_window(
        title="LLM Forge",
        url=url,
        width=1400,
        height=900,
        min_size=(1000, 700),
    )
    webview.start()  # Blocks until window is closed

    # Cleanup: close Gradio server when window closes
    logger.info("Desktop window closed, shutting down server")
    app.close()
