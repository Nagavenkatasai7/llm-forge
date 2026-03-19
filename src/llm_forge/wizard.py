"""Interactive training wizard for non-technical users.

Guides users through model selection, data configuration, and
training setup via a conversational CLI interface.  Generates a
ready-to-use YAML config customised for detected hardware.
"""

from __future__ import annotations

import platform
import textwrap
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Template definitions (purpose-built personal AI presets)
# ---------------------------------------------------------------------------

PERSONAL_AI_TEMPLATES: dict[str, dict[str, Any]] = {
    "journal": {
        "name": "Personal Journal Assistant",
        "emoji": "notebook",
        "description": "Learns your writing style, helps reflect, suggests prompts",
        "optimised_for": "Personal voice matching, empathetic responses",
        "data_hint": "Journal entries, personal notes (.txt, .md files)",
        "system_prompt": (
            "You are a thoughtful personal journal assistant. "
            "Mirror the user's writing style. Be empathetic, reflective, "
            "and encourage self-expression. Ask thought-provoking questions."
        ),
        "model_recommendation": "small",
        "training_config": {
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 4,
            "warmup_ratio": 0.1,
            "label_smoothing_factor": 0.1,
        },
        "lora_config": {"r": 16, "alpha": 32, "dropout": 0.05},
    },
    "study": {
        "name": "Study Buddy",
        "emoji": "books",
        "description": "Helps learn any topic, quizzes you, explains concepts",
        "optimised_for": "Socratic questioning, adaptive difficulty",
        "data_hint": "Study materials, textbooks, your notes (.pdf, .txt)",
        "system_prompt": (
            "You are a patient, encouraging study buddy. "
            "Explain concepts clearly at the user's level. "
            "Use analogies and examples. Quiz the user to test understanding. "
            "Break down complex topics into manageable pieces."
        ),
        "model_recommendation": "medium",
        "training_config": {
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 4,
            "warmup_ratio": 0.05,
            "label_smoothing_factor": 0.1,
        },
        "lora_config": {"r": 32, "alpha": 64, "dropout": 0.05},
    },
    "writer": {
        "name": "Creative Writing Partner",
        "emoji": "pen",
        "description": "Brainstorms ideas, helps with writer's block, edits work",
        "optimised_for": "Creative flair, style adaptation",
        "data_hint": "Your writing samples, favourite authors' style examples",
        "system_prompt": (
            "You are a creative writing partner. "
            "Help brainstorm ideas, overcome writer's block, and develop "
            "characters and plots. Adapt to the user's preferred genre and "
            "style. Offer constructive feedback on writing samples."
        ),
        "model_recommendation": "medium",
        "training_config": {
            "num_epochs": 4,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 4,
            "warmup_ratio": 0.1,
            "label_smoothing_factor": 0.05,
        },
        "lora_config": {"r": 32, "alpha": 64, "dropout": 0.1},
    },
    "professional": {
        "name": "Professional Assistant",
        "emoji": "briefcase",
        "description": "Drafts emails, summarises documents, manages tasks",
        "optimised_for": "Professional tone, conciseness",
        "data_hint": "Past emails, documents, templates (.txt, .jsonl)",
        "system_prompt": (
            "You are a professional assistant. "
            "Write in a clear, professional tone. Be concise and actionable. "
            "Draft emails, summarise documents, and help organise tasks. "
            "Maintain formality appropriate to the business context."
        ),
        "model_recommendation": "medium",
        "training_config": {
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 4,
            "warmup_ratio": 0.05,
            "label_smoothing_factor": 0.1,
        },
        "lora_config": {"r": 16, "alpha": 32, "dropout": 0.05},
    },
    "mindfulness": {
        "name": "Mindfulness Coach",
        "emoji": "lotus",
        "description": "Guides meditation, tracks mood, offers support",
        "optimised_for": "Calming tone, evidence-based techniques",
        "data_hint": "Journal entries about mood, goals, wellness content",
        "system_prompt": (
            "You are a calm, supportive mindfulness coach. "
            "Guide meditation and breathing exercises. Help the user track "
            "their mood and emotional patterns. Offer evidence-based coping "
            "strategies. Maintain a warm, non-judgmental tone."
        ),
        "model_recommendation": "small",
        "training_config": {
            "num_epochs": 3,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 4,
            "warmup_ratio": 0.1,
            "label_smoothing_factor": 0.1,
        },
        "lora_config": {"r": 16, "alpha": 32, "dropout": 0.05},
    },
    "gamemaster": {
        "name": "Game Master",
        "emoji": "game_die",
        "description": "Runs RPG campaigns, creates NPCs, adapts to players",
        "optimised_for": "Narrative creativity, rule knowledge",
        "data_hint": "Campaign notes, character sheets, lore (.txt, .md)",
        "system_prompt": (
            "You are a creative and adaptive Game Master for tabletop RPGs. "
            "Create vivid descriptions, interesting NPCs, and dynamic encounters. "
            "Adapt difficulty and narrative to player choices. "
            "Know the rules but prioritise fun and storytelling."
        ),
        "model_recommendation": "medium",
        "training_config": {
            "num_epochs": 4,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 4,
            "warmup_ratio": 0.1,
            "label_smoothing_factor": 0.05,
        },
        "lora_config": {"r": 32, "alpha": 64, "dropout": 0.1},
    },
}


# ---------------------------------------------------------------------------
# Model recommendations (hardware-aware)
# ---------------------------------------------------------------------------

MODEL_TIERS: dict[str, dict[str, Any]] = {
    "tiny": {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "params": "135M",
        "description": "Ultra-fast experiments, fits anywhere",
        "min_ram_gb": 2,
        "min_vram_gb": 0,
    },
    "small": {
        "name": "HuggingFaceTB/SmolLM2-360M",
        "params": "360M",
        "description": "Good quality/speed balance, MacBook friendly",
        "min_ram_gb": 4,
        "min_vram_gb": 0,
    },
    "medium": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params": "1.1B",
        "description": "Solid results, good for most use cases",
        "min_ram_gb": 8,
        "min_vram_gb": 4,
    },
    "large": {
        "name": "Qwen/Qwen2.5-3B",
        "params": "3B",
        "description": "High quality, needs decent GPU/RAM",
        "min_ram_gb": 16,
        "min_vram_gb": 8,
    },
    "xlarge": {
        "name": "meta-llama/Meta-Llama-3.1-8B",
        "params": "8B",
        "description": "Production quality, requires 24GB+ VRAM",
        "min_ram_gb": 32,
        "min_vram_gb": 16,
    },
}


# ---------------------------------------------------------------------------
# Hardware detection helpers
# ---------------------------------------------------------------------------


def detect_hardware_profile() -> dict[str, Any]:
    """Detect hardware capabilities for model recommendations."""
    profile: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "ram_gb": 0,
        "gpu_available": False,
        "gpu_name": None,
        "vram_gb": 0,
        "mps_available": False,
        "recommended_method": "qlora",
        "recommended_model_tier": "small",
    }

    # RAM
    try:
        import psutil

        profile["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    # GPU detection
    try:
        import torch

        if torch.cuda.is_available():
            profile["gpu_available"] = True
            profile["gpu_name"] = torch.cuda.get_device_name(0)
            profile["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            profile["mps_available"] = True
            profile["gpu_available"] = True
            profile["gpu_name"] = "Apple MPS"
    except ImportError:
        pass

    # Recommendations based on hardware
    vram = profile["vram_gb"]
    ram = profile["ram_gb"]

    if vram >= 24:
        profile["recommended_method"] = "lora"
        profile["recommended_model_tier"] = "xlarge"
    elif vram >= 12:
        profile["recommended_method"] = "lora"
        profile["recommended_model_tier"] = "large"
    elif vram >= 6 or profile["mps_available"] and ram >= 16:
        profile["recommended_method"] = "qlora"
        profile["recommended_model_tier"] = "medium"
    elif profile["mps_available"] or ram >= 8:
        profile["recommended_method"] = "qlora"
        profile["recommended_model_tier"] = "small"
    else:
        profile["recommended_method"] = "qlora"
        profile["recommended_model_tier"] = "tiny"

    return profile


# ---------------------------------------------------------------------------
# Data source helpers
# ---------------------------------------------------------------------------


def scan_data_source(path: str) -> dict[str, Any]:
    """Scan a data source path and report statistics."""
    result: dict[str, Any] = {
        "path": path,
        "is_hf_dataset": False,
        "is_local": False,
        "exists": False,
        "file_count": 0,
        "total_size_mb": 0,
        "detected_format": "custom",
        "extensions": [],
    }

    # Check if it looks like a HuggingFace dataset ID
    if "/" in path and not Path(path).exists():
        result["is_hf_dataset"] = True
        result["exists"] = True
        result["detected_format"] = "alpaca"
        return result

    p = Path(path)
    if not p.exists():
        return result

    result["is_local"] = True
    result["exists"] = True

    if p.is_file():
        result["file_count"] = 1
        result["total_size_mb"] = round(p.stat().st_size / (1024 * 1024), 2)
        ext = p.suffix.lower()
        result["extensions"] = [ext]
        if ext == ".jsonl":
            result["detected_format"] = "alpaca"
        elif ext == ".csv":
            result["detected_format"] = "custom"
        elif ext in (".txt", ".md"):
            result["detected_format"] = "completion"
    elif p.is_dir():
        files = list(p.rglob("*"))
        files = [f for f in files if f.is_file()]
        result["file_count"] = len(files)
        result["total_size_mb"] = round(sum(f.stat().st_size for f in files) / (1024 * 1024), 2)
        exts = list(set(f.suffix.lower() for f in files if f.suffix))
        result["extensions"] = sorted(exts)

        if ".jsonl" in exts or ".json" in exts:
            result["detected_format"] = "alpaca"
        else:
            result["detected_format"] = "completion"

    return result


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def generate_config(
    *,
    purpose: str,
    ai_name: str,
    model_name: str,
    training_method: str,
    data_path: str,
    data_format: str,
    input_field: str = "instruction",
    output_field: str = "output",
    max_samples: int | None = None,
    num_epochs: int = 3,
    system_prompt: str | None = None,
    enable_eval: bool = True,
    enable_iti: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> dict[str, Any]:
    """Generate a complete LLMForgeConfig-compatible dictionary."""
    config: dict[str, Any] = {
        "model": {
            "name": model_name,
            "max_seq_length": 512,
            "attn_implementation": "sdpa",
        },
        "data": {
            "train_path": data_path,
            "format": data_format,
            "input_field": input_field,
            "output_field": output_field,
            "test_size": 0.1,
        },
        "training": {
            "mode": training_method,
            "output_dir": f"./outputs/{ai_name}/",
            "num_epochs": num_epochs,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "logging_steps": 25,
            "report_to": ["none"],
            "label_smoothing_factor": 0.1,
        },
        "serving": {
            "export_format": "safetensors",
            "merge_adapter": True,
        },
    }

    if max_samples is not None and max_samples > 0:
        config["data"]["max_samples"] = max_samples

    if system_prompt:
        config["data"]["system_prompt"] = system_prompt

    # LoRA / QLoRA settings
    if training_method in ("lora", "qlora"):
        config["lora"] = {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": 0.05,
            "target_modules": "all-linear",
        }

    if training_method == "qlora":
        config["quantization"] = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bf16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

    if enable_eval:
        config["evaluation"] = {
            "enabled": True,
            "benchmarks": ["hellaswag"],
            "num_fewshot": 0,
            "generate_report": True,
        }

    if enable_iti:
        config["iti"] = {
            "enabled": True,
            "probing_dataset": "truthful_qa",
            "num_probing_samples": 500,
            "num_heads": 24,
            "alpha": 15.0,
            "bake_in": True,
        }
        config["refusal"] = {
            "enabled": True,
            "refusal_ratio": 0.15,
        }

    # Apply template-specific overrides
    template = PERSONAL_AI_TEMPLATES.get(purpose)
    if template:
        for k, v in template["training_config"].items():
            config["training"][k] = v
        if "lora" in config:
            for k, v in template["lora_config"].items():
                config["lora"][k] = v

    return config


def config_to_yaml(config: dict[str, Any], ai_name: str = "my-ai") -> str:
    """Convert a config dict to a YAML string with comments."""
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")

    header = textwrap.dedent(f"""\
        # llm-forge config — {ai_name}
        # Generated by the llm-forge training wizard
        # Docs: https://github.com/Nagavenkatasai7/llm-forge
        #
        # To train: llm-forge train --config {ai_name}.yaml
        # To validate: llm-forge validate {ai_name}.yaml

    """)

    return header + yaml.dump(config, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Interactive wizard runner
# ---------------------------------------------------------------------------


class TrainingWizard:
    """Interactive conversational CLI wizard for training setup.

    Guides non-technical users through purpose selection, data
    configuration, model selection, and generates a YAML config.
    """

    def __init__(self, console: Any | None = None) -> None:
        if console is not None:
            self.console = console
        elif _RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    def _print(self, text: str) -> None:
        if self.console:
            self.console.print(text)
        else:
            print(text)

    def _prompt(self, message: str, default: str = "") -> str:
        """Simple input prompt with default."""
        if default:
            response = input(f"{message} [{default}]: ").strip()
            return response if response else default
        return input(f"{message}: ").strip()

    def _confirm(self, message: str, default: bool = True) -> bool:
        suffix = " [Y/n]" if default else " [y/N]"
        response = input(f"{message}{suffix}: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")

    def _choose(self, message: str, options: list[str], default: int = 0) -> int:
        """Display numbered options and return selected index."""
        for i, opt in enumerate(options):
            marker = " (recommended)" if i == default else ""
            self._print(f"  [{i + 1}] {opt}{marker}")
        while True:
            response = input(f"\n{message} [{default + 1}]: ").strip()
            if not response:
                return default
            try:
                idx = int(response) - 1
                if 0 <= idx < len(options):
                    return idx
            except ValueError:
                pass
            self._print(f"  Please enter a number between 1 and {len(options)}")

    def run(self, output_path: str = "config.yaml") -> str | None:
        """Run the full interactive wizard. Returns path to generated config."""
        self._print("")
        if self.console and _RICH_AVAILABLE:
            self.console.print(
                Panel(
                    "[bold cyan]Welcome to LLM Forge![/bold cyan]\n"
                    "Let's train your personal AI.\n\n"
                    "[dim]No technical knowledge required — "
                    "I'll handle the details.[/dim]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        else:
            self._print("=== Welcome to LLM Forge! ===")
            self._print("Let's train your personal AI.\n")

        # Step 1: Purpose
        purpose = self._step_purpose()

        # Step 2: AI name
        ai_name = self._step_name(purpose)

        # Step 3: Data source
        data_path, data_format, input_field, output_field = self._step_data(purpose)

        # Step 4: Model selection (hardware-aware)
        hw = detect_hardware_profile()
        model_name, method = self._step_model(purpose, hw)

        # Step 5: Quick options
        num_epochs, max_samples, enable_eval, enable_iti = self._step_options()

        # Step 6: Review & confirm
        template = PERSONAL_AI_TEMPLATES.get(purpose, {})
        system_prompt = template.get("system_prompt")
        lora_cfg = template.get("lora_config", {})

        config = generate_config(
            purpose=purpose,
            ai_name=ai_name,
            model_name=model_name,
            training_method=method,
            data_path=data_path,
            data_format=data_format,
            input_field=input_field,
            output_field=output_field,
            max_samples=max_samples,
            num_epochs=num_epochs,
            system_prompt=system_prompt,
            enable_eval=enable_eval,
            enable_iti=enable_iti,
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
        )

        confirmed = self._step_review(config, ai_name, hw)
        if not confirmed:
            self._print("\n[yellow]Setup cancelled.[/yellow]")
            return None

        # Write config
        yaml_text = config_to_yaml(config, ai_name)
        out = Path(output_path)
        out.write_text(yaml_text, encoding="utf-8")

        self._print("")
        if self.console and _RICH_AVAILABLE:
            self.console.print(
                Panel(
                    f"[bold green]Config saved to:[/bold green] {out.absolute()}\n\n"
                    f"AI Name: [bold]{ai_name}[/bold]\n"
                    f"Model: [bold]{model_name}[/bold]\n"
                    f"Method: [bold]{method}[/bold]\n"
                    f"Data: [bold]{data_path}[/bold]\n\n"
                    "[bold]Next steps:[/bold]\n"
                    f"  1. Review:   cat {output_path}\n"
                    f"  2. Validate: llm-forge validate {output_path}\n"
                    f"  3. Train:    llm-forge train --config {output_path}\n\n"
                    "[dim]Training will auto-detect your hardware and optimise "
                    "settings for the best results.[/dim]",
                    title="[bold green]Setup Complete![/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        return str(out.absolute())

    # ---- Step implementations ----

    def _step_purpose(self) -> str:
        """Step 1: Choose the AI's purpose."""
        self._print("\n[bold]What would you like to create?[/bold]\n")

        options = []
        keys = list(PERSONAL_AI_TEMPLATES.keys())
        for key in keys:
            t = PERSONAL_AI_TEMPLATES[key]
            options.append(f"{t['name']}\n       {t['description']}")
        options.append("Something custom\n       Define your own purpose")

        idx = self._choose("Your choice", options, default=0)
        if idx < len(keys):
            return keys[idx]
        return "custom"

    def _step_name(self, purpose: str) -> str:
        """Step 2: Name the AI."""
        template = PERSONAL_AI_TEMPLATES.get(purpose, {})
        default_name = template.get("name", "MyAI").replace(" ", "")
        self._print("\n[bold]What should we call your AI?[/bold]")
        name = self._prompt("AI name", default=default_name)
        return name.replace(" ", "-").lower()

    def _step_data(self, purpose: str) -> tuple[str, str, str, str]:
        """Step 3: Configure data source."""
        template = PERSONAL_AI_TEMPLATES.get(purpose, {})
        hint = template.get("data_hint", "Training data (.jsonl, .txt, or HuggingFace dataset)")

        self._print("\n[bold]Where is your training data?[/bold]")
        self._print(f"  [dim]Recommended: {hint}[/dim]\n")

        options = [
            "HuggingFace dataset (e.g., tatsu-lab/alpaca)",
            "Local file or folder",
            "I'll add data later (use sample dataset)",
        ]
        source_idx = self._choose("Data source", options, default=2)

        if source_idx == 0:
            data_path = self._prompt("HuggingFace dataset name", default="tatsu-lab/alpaca")
            data_format = "alpaca"
            input_field = "instruction"
            output_field = "output"
        elif source_idx == 1:
            data_path = self._prompt("Path to file or folder")
            scan = scan_data_source(data_path)
            if scan["exists"]:
                self._print(
                    f"  Found {scan['file_count']} file(s), {scan['total_size_mb']:.1f} MB total"
                )
                data_format = scan["detected_format"]
            else:
                self._print(
                    "  [yellow]Path not found. You can fix this in the config later.[/yellow]"
                )
                data_format = "custom"
            input_field = self._prompt("Input column name", default="instruction")
            output_field = self._prompt("Output column name", default="output")
        else:
            data_path = "tatsu-lab/alpaca"
            data_format = "alpaca"
            input_field = "instruction"
            output_field = "output"

        return data_path, data_format, input_field, output_field

    def _step_model(self, purpose: str, hw: dict[str, Any]) -> tuple[str, str]:
        """Step 4: Model selection with hardware recommendations."""
        self._print("\n[bold]Detecting your hardware...[/bold]")
        if hw["gpu_available"]:
            gpu_info = hw["gpu_name"]
            if hw["vram_gb"] > 0:
                gpu_info += f" ({hw['vram_gb']:.0f} GB)"
            self._print(f"  GPU: {gpu_info}")
        else:
            self._print("  GPU: None detected (CPU training)")
        if hw["ram_gb"] > 0:
            self._print(f"  RAM: {hw['ram_gb']:.0f} GB")

        recommended_tier = hw["recommended_model_tier"]
        template = PERSONAL_AI_TEMPLATES.get(purpose, {})
        if template.get("model_recommendation"):
            # Use template recommendation if hardware supports it
            tmpl_tier = template["model_recommendation"]
            tier_order = ["tiny", "small", "medium", "large", "xlarge"]
            if tier_order.index(tmpl_tier) <= tier_order.index(recommended_tier):
                recommended_tier = tmpl_tier

        self._print("\n[bold]Choose your base model:[/bold]\n")

        tier_keys = list(MODEL_TIERS.keys())
        options = []
        default_idx = 0
        for i, key in enumerate(tier_keys):
            tier = MODEL_TIERS[key]
            tag = " [recommended]" if key == recommended_tier else ""
            options.append(f"{tier['name']} ({tier['params']}){tag}\n       {tier['description']}")
            if key == recommended_tier:
                default_idx = i

        idx = self._choose("Model", options, default=default_idx)
        selected_tier = tier_keys[idx]
        model_name = MODEL_TIERS[selected_tier]["name"]

        # Custom model override
        custom = self._prompt(f"Model name (Enter for {model_name})", default=model_name)
        if custom:
            model_name = custom

        # Training method
        recommended_method = hw["recommended_method"]
        self._print("\n[bold]Training method:[/bold]")
        method_options = [
            f"QLoRA 4-bit — Lowest memory, good quality{' [recommended]' if recommended_method == 'qlora' else ''}",
            f"LoRA — Fast, memory-efficient{' [recommended]' if recommended_method == 'lora' else ''}",
            "Full fine-tuning — Best quality, most memory",
        ]
        method_map = ["qlora", "lora", "full"]
        method_default = method_map.index(recommended_method)
        method_idx = self._choose("Method", method_options, default=method_default)
        method = method_map[method_idx]

        return model_name, method

    def _step_options(self) -> tuple[int, int | None, bool, bool]:
        """Step 5: Quick configuration options."""
        self._print("\n[bold]Quick options:[/bold]")

        num_epochs = int(self._prompt("Number of training epochs", default="3"))
        max_str = self._prompt("Max training samples (0 = all)", default="0")
        max_samples = int(max_str) if max_str and int(max_str) > 0 else None
        enable_eval = self._confirm("Enable evaluation benchmarks?", default=True)
        enable_iti = self._confirm(
            "Enable anti-hallucination (ITI + refusal training)?", default=False
        )

        return num_epochs, max_samples, enable_eval, enable_iti

    def _step_review(self, config: dict[str, Any], ai_name: str, hw: dict[str, Any]) -> bool:
        """Step 6: Review configuration and confirm."""
        self._print("\n")
        if self.console and _RICH_AVAILABLE:
            table = Table(title=f"Training Configuration: {ai_name}", expand=True)
            table.add_column("Setting", style="bold cyan", width=24)
            table.add_column("Value")

            table.add_row("Model", config["model"]["name"])
            table.add_row("Training method", config["training"]["mode"])
            table.add_row("Dataset", config["data"]["train_path"])
            table.add_row("Epochs", str(config["training"]["num_epochs"]))
            table.add_row("Batch size", str(config["training"]["per_device_train_batch_size"]))
            table.add_row("Learning rate", str(config["training"]["learning_rate"]))
            if "lora" in config:
                table.add_row("LoRA rank", str(config["lora"]["r"]))
            eval_status = "Yes" if config.get("evaluation", {}).get("enabled") else "No"
            table.add_row("Evaluation", eval_status)
            iti_status = "Yes" if config.get("iti", {}).get("enabled") else "No"
            table.add_row("Anti-hallucination", iti_status)

            self.console.print(table)
        else:
            self._print(f"  Model:   {config['model']['name']}")
            self._print(f"  Method:  {config['training']['mode']}")
            self._print(f"  Data:    {config['data']['train_path']}")
            self._print(f"  Epochs:  {config['training']['num_epochs']}")

        return self._confirm("\nReady to save this configuration?", default=True)
