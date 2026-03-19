"""FREE guided wizard fallback for LLM Forge that works WITHOUT any API key.

When users launch ``llm-forge`` without an ``ANTHROPIC_API_KEY`` (or any
other provider key), this module replaces the "no key" error with a
fully interactive, Rich-powered terminal wizard that walks the user
through:

1. Welcome + purpose selection
2. Data source configuration
3. Hardware detection
4. Model recommendation
5. Config generation & validation
6. Training launch

The wizard uses decision trees (if/elif), NOT an LLM API, and calls
existing tools directly from ``llm_forge.chat.tools``.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Purpose catalogue (decision-tree data, no LLM needed)
# ---------------------------------------------------------------------------

PURPOSE_OPTIONS: list[dict[str, str]] = [
    {
        "key": "customer_support",
        "label": "Customer support bot",
        "description": "Answer FAQs, handle tickets, assist customers",
        "system_prompt": (
            "You are a helpful customer support agent. Be polite, concise, "
            "and solution-oriented. Escalate complex issues appropriately."
        ),
        "recommended_data": "Support tickets, FAQ pairs, knowledge-base articles (.jsonl)",
    },
    {
        "key": "knowledge_assistant",
        "label": "Knowledge assistant",
        "description": "Answer questions from documents, manuals, or wikis",
        "system_prompt": (
            "You are a knowledgeable assistant. Answer questions accurately "
            "using the provided context. Cite sources when possible."
        ),
        "recommended_data": "Q&A pairs, documentation, wiki pages (.jsonl, .txt)",
    },
    {
        "key": "code_helper",
        "label": "Code helper",
        "description": "Explain code, debug issues, generate snippets",
        "system_prompt": (
            "You are a senior software engineer assistant. Explain code clearly, "
            "suggest best practices, and help debug issues step by step."
        ),
        "recommended_data": "Code instruction pairs, StackOverflow Q&A (.jsonl)",
    },
    {
        "key": "writing_assistant",
        "label": "Writing assistant",
        "description": "Help write, edit, and improve text",
        "system_prompt": (
            "You are a skilled writing assistant. Help users write, edit, "
            "and improve their text. Adapt to the requested style and tone."
        ),
        "recommended_data": "Writing samples, edit pairs, style examples (.jsonl, .txt)",
    },
    {
        "key": "domain_expert",
        "label": "Domain expert",
        "description": "Specialise in a specific field (finance, legal, medical, etc.)",
        "system_prompt": (
            "You are a domain expert assistant. Provide accurate, detailed "
            "information in your area of expertise."
        ),
        "recommended_data": "Domain-specific Q&A, textbooks, articles (.jsonl)",
    },
    {
        "key": "custom",
        "label": "Custom",
        "description": "Define your own purpose and system prompt",
        "system_prompt": "You are a helpful AI assistant.",
        "recommended_data": "Any instruction-following dataset (.jsonl, .txt, HuggingFace)",
    },
]

# ---------------------------------------------------------------------------
# Model recommendation table (keyed by VRAM/RAM tiers)
# ---------------------------------------------------------------------------

MODEL_RECOMMENDATIONS: list[dict[str, Any]] = [
    {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "params": "135M",
        "min_vram_gb": 0,
        "min_ram_gb": 2,
        "mode": "lora",
        "batch_size": 4,
        "description": "Tiny model -- great for testing, fits on any machine",
    },
    {
        "name": "HuggingFaceTB/SmolLM2-360M",
        "params": "360M",
        "min_vram_gb": 0,
        "min_ram_gb": 4,
        "mode": "lora",
        "batch_size": 4,
        "description": "Small model, good quality/speed balance",
    },
    {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params": "1.1B",
        "min_vram_gb": 4,
        "min_ram_gb": 8,
        "mode": "qlora",
        "batch_size": 2,
        "description": "Solid results for most tasks",
    },
    {
        "name": "unsloth/Llama-3.2-1B-Instruct",
        "params": "1B",
        "min_vram_gb": 6,
        "min_ram_gb": 12,
        "mode": "lora",
        "batch_size": 2,
        "description": "High-quality instruction-tuned base",
    },
    {
        "name": "Qwen/Qwen2.5-3B",
        "params": "3B",
        "min_vram_gb": 12,
        "min_ram_gb": 16,
        "mode": "lora",
        "batch_size": 2,
        "description": "Strong multilingual model, needs decent GPU",
    },
    {
        "name": "meta-llama/Meta-Llama-3.1-8B",
        "params": "8B",
        "min_vram_gb": 24,
        "min_ram_gb": 32,
        "mode": "lora",
        "batch_size": 1,
        "description": "Production quality, requires 24 GB+ VRAM",
    },
]

# ---------------------------------------------------------------------------
# Training presets per purpose (conservative defaults)
# ---------------------------------------------------------------------------

_TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "customer_support": {
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "knowledge_assistant": {
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "code_helper": {
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "writing_assistant": {
        "num_epochs": 4,
        "learning_rate": 2e-4,
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "domain_expert": {
        "num_epochs": 3,
        "learning_rate": 1e-5,
        "lora_r": 8,
        "lora_alpha": 16,
    },
    "custom": {
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_console():
    """Return a Rich Console, or *None* when Rich is unavailable."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


def _plain_print(text: str) -> None:
    """Fallback printer when Rich is not installed."""
    # Strip Rich markup for plain output
    import re

    clean = re.sub(r"\[/?[a-z_ ]+\]", "", text)
    print(clean)


def _recommend_model(hardware: dict[str, Any]) -> dict[str, Any]:
    """Pick the best model from *MODEL_RECOMMENDATIONS* for the detected hardware."""
    vram = hardware.get("vram_gb", 0)
    ram = hardware.get("ram_total_gb", hardware.get("ram_gb", 0))
    if isinstance(ram, str):
        ram = 0
    if isinstance(vram, str):
        vram = 0

    # Walk the list in reverse (biggest first) and pick the largest that fits
    for model in reversed(MODEL_RECOMMENDATIONS):
        if vram >= model["min_vram_gb"] and ram >= model["min_ram_gb"]:
            return model

    # Fallback: tiniest model
    return MODEL_RECOMMENDATIONS[0]


def _parse_hardware_json(hw_json: str) -> dict[str, Any]:
    """Parse the JSON string returned by ``_detect_hardware()``."""
    try:
        return json.loads(hw_json)
    except (json.JSONDecodeError, TypeError):
        return {}


def build_config(
    *,
    purpose_key: str,
    model_name: str,
    training_mode: str,
    data_path: str,
    data_format: str,
    system_prompt: str,
    output_dir: str = "./outputs/wizard/",
    batch_size: int = 2,
) -> dict[str, Any]:
    """Build a config dict suitable for ``_write_config`` / ``_validate_config``.

    Returns a plain dict matching the ``LLMForgeConfig`` schema.
    """
    preset = _TRAINING_PRESETS.get(purpose_key, _TRAINING_PRESETS["custom"])

    config: dict[str, Any] = {
        "model": {
            "name": model_name,
            "max_seq_length": 2048,
            "attn_implementation": "sdpa",
            "torch_dtype": "bf16",
        },
        "data": {
            "train_path": data_path,
            "format": data_format,
            "test_size": 0.1,
            "system_prompt": system_prompt,
        },
        "training": {
            "mode": training_mode,
            "output_dir": output_dir,
            "num_epochs": preset["num_epochs"],
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": preset["learning_rate"],
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "bf16": True,
            "logging_steps": 10,
            "save_steps": 200,
            "report_to": ["none"],
        },
        "evaluation": {
            "enabled": False,
        },
        "serving": {
            "export_format": "safetensors",
            "merge_adapter": True,
        },
    }

    # LoRA / QLoRA
    if training_mode in ("lora", "qlora"):
        config["lora"] = {
            "r": preset["lora_r"],
            "alpha": preset["lora_alpha"],
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        }

    if training_mode == "qlora":
        config["quantization"] = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bf16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

    return config


# ---------------------------------------------------------------------------
# Main wizard entry-point
# ---------------------------------------------------------------------------


def launch_wizard_fallback() -> None:  # noqa: C901 — intentionally linear wizard
    """Launch the FREE guided wizard (no API key required).

    This is a 6-step interactive terminal flow that uses Rich for
    beautiful output and decision trees to guide the user.
    """
    from llm_forge.chat.tools import (
        _detect_hardware,
        _start_training,
        _validate_config,
        _write_config,
    )

    console = _get_console()

    def cprint(text: str) -> None:
        if console is not None:
            console.print(text)
        else:
            _plain_print(text)

    # ------------------------------------------------------------------
    # Step 1: Welcome + Purpose Selection
    # ------------------------------------------------------------------
    try:
        from rich.panel import Panel
        from rich.table import Table

        _rich_ok = True
    except ImportError:
        _rich_ok = False

    if console is not None and _rich_ok:
        console.print(
            Panel(
                "[bold cyan]Welcome to LLM Forge![/bold cyan]\n\n"
                "Build your own AI model -- [bold]no API key needed[/bold].\n"
                "This wizard will guide you through every step.\n\n"
                "[dim]Type the number of your choice at each prompt.[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
    else:
        print("=" * 54)
        print("  Welcome to LLM Forge!")
        print("  Build your own AI model -- no API key needed.")
        print("  This wizard will guide you through every step.")
        print("=" * 54)
    cprint("")

    # Purpose menu
    cprint("[bold]Step 1/6 -- What do you want to build?[/bold]\n")
    for i, p in enumerate(PURPOSE_OPTIONS, 1):
        cprint(f"  [cyan]{i}[/cyan]) {p['label']}")
        cprint(f"     [dim]{p['description']}[/dim]")

    purpose_idx = _prompt_int("Your choice", 1, len(PURPOSE_OPTIONS), default=1) - 1
    selected_purpose = PURPOSE_OPTIONS[purpose_idx]
    cprint(f"\n  Selected: [bold green]{selected_purpose['label']}[/bold green]\n")

    # Custom system prompt
    system_prompt = selected_purpose["system_prompt"]
    if selected_purpose["key"] == "custom":
        cprint("[bold]Enter a custom system prompt[/bold] (or press Enter for default):")
        user_prompt = input("  > ").strip()
        if user_prompt:
            system_prompt = user_prompt

    # ------------------------------------------------------------------
    # Step 2: Data Source
    # ------------------------------------------------------------------
    cprint("\n[bold]Step 2/6 -- Where is your training data?[/bold]\n")
    cprint(f"  [dim]Recommended: {selected_purpose.get('recommended_data', 'Any dataset')}[/dim]\n")
    cprint("  [cyan]1[/cyan]) HuggingFace dataset (e.g. tatsu-lab/alpaca)")
    cprint("  [cyan]2[/cyan]) Local file or folder")
    cprint("  [cyan]3[/cyan]) Use sample dataset (get started quickly)")

    data_choice = _prompt_int("Your choice", 1, 3, default=3)
    data_path: str
    data_format: str

    if data_choice == 1:
        cprint("\n  Enter the HuggingFace dataset ID:")
        data_path = input("  > ").strip() or "tatsu-lab/alpaca"
        data_format = "alpaca"
        cprint(f"  Using: [bold]{data_path}[/bold]")
    elif data_choice == 2:
        cprint("\n  Enter the path to your data file or folder:")
        data_path = input("  > ").strip()
        if not data_path:
            cprint("  [yellow]No path entered -- falling back to sample dataset.[/yellow]")
            data_path = "tatsu-lab/alpaca"
            data_format = "alpaca"
        else:
            # Attempt a quick scan
            from llm_forge.chat.tools import _scan_data

            scan_result = json.loads(_scan_data(data_path))
            fmt = scan_result.get("detected_format", "custom")
            status = scan_result.get("status", "error")
            if status == "ok":
                cprint(f"  Detected format: [bold]{fmt}[/bold]")
                data_format = fmt
            else:
                cprint(
                    f"  [yellow]Could not scan data: {scan_result.get('error', 'unknown')}[/yellow]"
                )
                cprint("  Defaulting to 'alpaca' format. You can change this in the config.")
                data_format = "alpaca"
    else:
        data_path = "tatsu-lab/alpaca"
        data_format = "alpaca"
        cprint("  Using sample dataset: [bold]tatsu-lab/alpaca[/bold]")

    # ------------------------------------------------------------------
    # Step 3: Hardware Detection
    # ------------------------------------------------------------------
    cprint("\n[bold]Step 3/6 -- Detecting your hardware...[/bold]\n")
    hw_json = _detect_hardware()
    hardware = _parse_hardware_json(hw_json)

    gpu_type = hardware.get("gpu_type", "unknown")
    ram_total = hardware.get("ram_total_gb", "?")
    os_name = hardware.get("os", "unknown")

    if console is not None and _rich_ok:
        hw_table = Table(title="Hardware Detected", show_header=False, padding=(0, 2))
        hw_table.add_column("Key", style="bold")
        hw_table.add_column("Value")
        hw_table.add_row("OS", f"{os_name} {hardware.get('os_version', '')}")
        hw_table.add_row("CPU", str(hardware.get("cpu", "unknown")))
        hw_table.add_row("RAM", f"{ram_total} GB")

        if gpu_type == "nvidia_cuda":
            gpus = hardware.get("gpus", [])
            for i, g in enumerate(gpus):
                hw_table.add_row(f"GPU {i}", f"{g.get('name', '?')} ({g.get('vram_gb', '?')} GB)")
        elif gpu_type == "apple_mps":
            hw_table.add_row("GPU", f"Apple MPS ({hardware.get('gpu_name', 'Apple Silicon')})")
        else:
            hw_table.add_row("GPU", "None detected")
        console.print(hw_table)
    else:
        print(f"  OS:  {os_name}")
        print(f"  RAM: {ram_total} GB")
        print(f"  GPU: {gpu_type}")

    # Compute effective VRAM for recommendations
    vram_gb: float = 0.0
    if gpu_type == "nvidia_cuda":
        gpus = hardware.get("gpus", [])
        if gpus:
            vram_gb = gpus[0].get("vram_gb", 0)
    elif gpu_type == "apple_mps":
        # Apple unified memory -- estimate 75% available for GPU
        raw_ram = hardware.get("ram_total_gb", 0)
        if isinstance(raw_ram, (int, float)):
            vram_gb = raw_ram * 0.75
    hardware["vram_gb"] = vram_gb
    hardware["ram_gb"] = hardware.get("ram_total_gb", 0)

    # ------------------------------------------------------------------
    # Step 4: Model Recommendation
    # ------------------------------------------------------------------
    cprint("\n[bold]Step 4/6 -- Recommended model for your hardware[/bold]\n")
    rec = _recommend_model(hardware)

    cprint(f"  Model:  [bold green]{rec['name']}[/bold green] ({rec['params']} parameters)")
    cprint(f"  Mode:   [bold]{rec['mode']}[/bold]")
    cprint(f"  Reason: {rec['description']}")

    cprint("\n  [dim]Press Enter to accept, or type a HuggingFace model name to override:[/dim]")
    override = input("  > ").strip()
    model_name = override if override else rec["name"]
    training_mode = rec["mode"]
    batch_size = rec["batch_size"]

    cprint(f"\n  Using: [bold]{model_name}[/bold] with [bold]{training_mode}[/bold]")

    # ------------------------------------------------------------------
    # Step 5: Config Generation
    # ------------------------------------------------------------------
    cprint("\n[bold]Step 5/6 -- Generating your training config...[/bold]\n")

    output_dir = "./outputs/wizard/"
    config_path = "configs/wizard_config.yaml"

    config = build_config(
        purpose_key=selected_purpose["key"],
        model_name=model_name,
        training_mode=training_mode,
        data_path=data_path,
        data_format=data_format,
        system_prompt=system_prompt,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    # Write via the existing tool
    write_result = json.loads(_write_config(config_path, config))
    if write_result.get("status") == "ok":
        resolved_path = write_result.get("path", config_path)
        cprint(f"  Config saved to: [bold green]{resolved_path}[/bold green]")
    else:
        cprint(f"  [red]Failed to write config: {write_result.get('error')}[/red]")
        return

    # Validate
    val_result = json.loads(_validate_config(config_path))
    if val_result.get("status") == "valid":
        cprint("  Validation:      [bold green]PASSED[/bold green]")
    else:
        cprint(f"  Validation:      [yellow]WARNINGS -- {val_result.get('errors', '')}[/yellow]")
        cprint("  [dim]The config may still work. You can edit it and re-validate later.[/dim]")

    # Show summary
    if console is not None and _rich_ok:
        summary = Table(title="Config Summary", show_header=False, padding=(0, 2))
        summary.add_column("Key", style="bold")
        summary.add_column("Value")
        summary.add_row("Purpose", selected_purpose["label"])
        summary.add_row("Model", model_name)
        summary.add_row("Mode", training_mode)
        summary.add_row("Data", data_path)
        summary.add_row("Epochs", str(config["training"]["num_epochs"]))
        summary.add_row("Batch size", str(config["training"]["per_device_train_batch_size"]))
        summary.add_row("Learning rate", str(config["training"]["learning_rate"]))
        if "lora" in config:
            summary.add_row("LoRA rank", str(config["lora"]["r"]))
        summary.add_row("Output dir", output_dir)
        cprint("")
        console.print(summary)
    else:
        print(f"\n  Purpose: {selected_purpose['label']}")
        print(f"  Model:   {model_name}")
        print(f"  Mode:    {training_mode}")
        print(f"  Data:    {data_path}")

    # ------------------------------------------------------------------
    # Step 6: Offer to Start Training
    # ------------------------------------------------------------------
    cprint("\n[bold]Step 6/6 -- Ready to train![/bold]\n")
    cprint("  [cyan]1[/cyan]) Start training now")
    cprint(
        "  [cyan]2[/cyan]) Exit (train later with: llm-forge train --config " + config_path + ")"
    )

    launch_choice = _prompt_int("Your choice", 1, 2, default=2)

    if launch_choice == 1:
        cprint("\n  Starting training...\n")
        train_result = json.loads(_start_training(config_path))
        status = train_result.get("status", "error")
        if status == "started":
            cprint("[bold green]  Training started![/bold green]")
            cprint(f"  PID: {train_result.get('pid', '?')}")
            cprint("\n  Monitor with: llm-forge status")
            cprint(f"  Config:       {config_path}")
        else:
            cprint(f"  [red]Failed to start training: {train_result.get('error', 'unknown')}[/red]")
            cprint(f"\n  You can start manually:\n    llm-forge train --config {config_path}")
    else:
        cprint("\n  No problem! When you're ready, run:")
        cprint(f"    [bold]llm-forge train --config {config_path}[/bold]")
        cprint("")
        cprint("  [dim]Tip: Set ANTHROPIC_API_KEY for the full AI-guided experience.[/dim]")

    cprint("")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _prompt_int(message: str, low: int, high: int, *, default: int = 1) -> int:
    """Prompt for an integer in [low, high] with a default."""
    while True:
        raw = input(f"  {message} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if low <= val <= high:
                return val
        except ValueError:
            pass
        # Use plain print -- this helper must not depend on Rich
        print(f"  Please enter a number between {low} and {high}.")
