"""llm-forge CLI -- Config-driven LLM training platform.

Provides the Typer-based command-line interface for every stage of the
LLM workflow: project initialization, config validation, training,
evaluation, serving, export, data cleaning, and RAG operations.

Entry point registered in ``pyproject.toml``::

    [project.scripts]
    llm-forge = "llm_forge.cli:app"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_forge import __version__

# ---------------------------------------------------------------------------
# App / sub-app definitions
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="llm-forge",
    help="Config-driven LLM training platform",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
    invoke_without_command=True,
)

rag_app = typer.Typer(
    name="rag",
    help="RAG (Retrieval-Augmented Generation) operations",
    no_args_is_help=True,
)
app.add_typer(rag_app, name="rag")

console = Console()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Valid preset template names
_VALID_TEMPLATES = ("lora", "qlora", "pretrain", "rag", "full")

# ---------------------------------------------------------------------------
# Re-export MODEL_TIERS from wizard.py as _SETUP_MODELS for backward compat
# ---------------------------------------------------------------------------
try:
    from llm_forge.wizard import MODEL_TIERS as _MODEL_TIERS

    _SETUP_MODELS = {key: (tier["name"], tier["description"]) for key, tier in _MODEL_TIERS.items()}
except ImportError:
    _SETUP_MODELS = {}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]llm-forge[/bold] {__version__}")
        raise typer.Exit()


def _banner() -> None:
    """Print a startup banner."""
    console.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]llm-forge[/bold cyan] v{__version__}  "
                "[dim]Config-driven LLM training platform[/dim]"
            ),
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _setup_verbose(verbose: bool) -> None:
    """Configure logging based on verbosity flag."""
    from llm_forge.utils.logging import setup_logging

    setup_logging(verbose=verbose)


def _load_config(config_path: str) -> LLMForgeConfig:  # noqa: F821
    """Load and validate a config file, exiting on failure."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {path}")
        raise typer.Exit(code=1)

    try:
        from llm_forge.config.validator import validate_config_file

        config = validate_config_file(str(path))
        return config
    except ImportError:
        # Fallback: load raw YAML and wrap in a Pydantic model
        try:
            import yaml

            from llm_forge.config.schema import LLMForgeConfig

            with open(path) as f:
                raw = yaml.safe_load(f)
            if raw is None:
                console.print(f"[red]Error:[/red] Config file is empty: {path}")
                raise typer.Exit(code=1)
            config = LLMForgeConfig(**raw)
            return config
        except ImportError:
            console.print(
                "[red]Error:[/red] Config schema module not found. "
                "Ensure llm_forge.config.schema is implemented."
            )
            raise typer.Exit(code=1)
        except Exception as exc:
            console.print(f"[red]Error:[/red] Failed to load config: {exc}")
            raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Config validation failed: {exc}")
        raise typer.Exit(code=1)


def _show_hardware_summary() -> None:
    """Print a compact hardware summary for GPU-dependent commands."""
    try:
        from llm_forge.utils.gpu_utils import get_system_gpu_info

        info = get_system_gpu_info()
        parts: list[str] = []

        if info.cuda_available:
            gpu_names = ", ".join(g.name for g in info.gpus) or "unknown"
            total_vram = f"{info.total_vram_gb:.1f} GB"
            parts.append(f"[green]CUDA[/green] {gpu_names} ({total_vram})")
        elif info.mps_available:
            parts.append("[green]Apple MPS[/green] (Metal Performance Shaders)")
        else:
            parts.append("[yellow]CPU only[/yellow] (no GPU detected)")

        if info.torch_version:
            parts.append(f"PyTorch {info.torch_version}")

        console.print(
            Panel(
                " | ".join(parts),
                title="[bold]Hardware[/bold]",
                border_style="dim",
                padding=(0, 1),
            )
        )
    except Exception:
        console.print("[dim]Hardware detection unavailable[/dim]")


def _print_config_summary(config: object) -> None:
    """Print a Rich table summarizing the loaded config."""
    table = Table(title="Configuration Summary", show_header=False, expand=True)
    table.add_column("Field", style="bold cyan", width=24)
    table.add_column("Value")

    # Try to iterate over Pydantic model fields
    if hasattr(config, "model_dump"):
        data = config.model_dump()
    elif hasattr(config, "dict"):
        data = config.dict()
    elif isinstance(config, dict):
        data = config
    else:
        console.print("[dim]Cannot display config summary.[/dim]")
        return

    def _add_rows(d: dict, prefix: str = "") -> None:
        for key, val in d.items():
            display_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(val, dict):
                _add_rows(val, display_key)
            elif isinstance(val, (list, tuple)) and len(val) > 5:
                table.add_row(display_key, f"[{len(val)} items]")
            else:
                str_val = str(val)
                if len(str_val) > 80:
                    str_val = str_val[:77] + "..."
                table.add_row(display_key, str_val)

    _add_rows(data)
    console.print(table)


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="LLM provider for chat: anthropic, openai, or ollama.",
        hidden=True,
    ),
) -> None:
    """Build your own AI model. Just type llm-forge and start talking."""
    if ctx.invoked_subcommand is None:
        # No subcommand → launch conversational assistant
        from llm_forge.chat.ui import launch_chat

        launch_chat(provider=provider)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    template: str = typer.Option(
        "lora",
        "--template",
        "-t",
        help=f"Preset template to use: {', '.join(_VALID_TEMPLATES)}",
    ),
    output: str = typer.Option(
        "config.yaml",
        "--output",
        "-o",
        help="Output filename for the generated config.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Initialize a new llm-forge project with a preset config template.

    Copies a preset YAML configuration to the current directory and prints
    getting-started instructions.
    """
    _setup_verbose(verbose)
    _banner()

    if template not in _VALID_TEMPLATES:
        console.print(
            f"[red]Error:[/red] Unknown template '{template}'. "
            f"Valid options: {', '.join(_VALID_TEMPLATES)}"
        )
        raise typer.Exit(code=1)

    output_path = Path(output)
    if output_path.exists():
        overwrite = typer.confirm(f"File '{output}' already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit()

    # Try loading from presets module
    try:
        from llm_forge.config.validator import load_preset

        content = load_preset(template)
    except ImportError:
        # Fallback: generate a reasonable default config
        content = _generate_default_config(template)

    output_path.write_text(content, encoding="utf-8")

    console.print(
        Panel(
            f"[green]Created config file:[/green] {output_path.absolute()}\n\n"
            f"Template: [bold]{template}[/bold]\n\n"
            "[bold]Next steps:[/bold]\n"
            f"  1. Edit [cyan]{output}[/cyan] with your model/dataset settings\n"
            f"  2. Validate: [cyan]llm-forge validate {output}[/cyan]\n"
            f"  3. Train:    [cyan]llm-forge train --config {output}[/cyan]",
            title="[bold green]Project Initialized[/bold green]",
            border_style="green",
        )
    )


def _generate_default_config(template: str) -> str:
    """Generate a default YAML config string for a given template."""
    import textwrap

    configs = {
        "lora": textwrap.dedent("""\
            # llm-forge config -- LoRA fine-tuning
            # Docs: https://github.com/Nagavenkatasai7/llm-forge

            project:
              name: my-lora-finetune
              output_dir: ./outputs
              seed: 42

            model:
              name: meta-llama/Meta-Llama-3.1-8B
              revision: main
              torch_dtype: bf16
              attn_implementation: flash_attention_2

            data:
              train_path: ./data/train.jsonl
              format: alpaca

            training:
              mode: lora
              num_epochs: 3
              per_device_train_batch_size: 4
              gradient_accumulation_steps: 4
              learning_rate: 2.0e-4
              lr_scheduler_type: cosine
              warmup_ratio: 0.03
              weight_decay: 0.01
              max_grad_norm: 1.0
              max_seq_length: 2048
              fp16: false
              bf16: true
              gradient_checkpointing: true

            lora:
              r: 16
              alpha: 32
              dropout: 0.05
              target_modules: all-linear
              bias: none
              task_type: CAUSAL_LM

            logging:
              log_steps: 10
              save_steps: 500
              eval_steps: 500
              report_to: wandb
        """),
        "qlora": textwrap.dedent("""\
            # llm-forge config -- QLoRA fine-tuning (4-bit quantized)
            # Docs: https://github.com/Nagavenkatasai7/llm-forge

            project:
              name: my-qlora-finetune
              output_dir: ./outputs
              seed: 42

            model:
              name: meta-llama/Meta-Llama-3.1-8B
              revision: main
              torch_dtype: bf16
              attn_implementation: flash_attention_2

            quantization:
              load_in_4bit: true
              bnb_4bit_quant_type: nf4
              bnb_4bit_compute_dtype: bf16
              bnb_4bit_use_double_quant: true

            data:
              train_path: ./data/train.jsonl
              format: alpaca

            training:
              mode: qlora
              num_epochs: 3
              per_device_train_batch_size: 4
              gradient_accumulation_steps: 4
              learning_rate: 2.0e-4
              lr_scheduler_type: cosine
              warmup_ratio: 0.03
              weight_decay: 0.01
              max_grad_norm: 1.0
              max_seq_length: 2048
              fp16: false
              bf16: true
              gradient_checkpointing: true

            lora:
              r: 16
              alpha: 32
              dropout: 0.05
              target_modules: all-linear
              bias: none
              task_type: CAUSAL_LM

            logging:
              log_steps: 10
              save_steps: 500
              eval_steps: 500
              report_to: wandb
        """),
        "pretrain": textwrap.dedent("""\
            # llm-forge config -- Pre-training from scratch
            # Docs: https://github.com/Nagavenkatasai7/llm-forge

            project:
              name: my-pretrain
              output_dir: ./outputs
              seed: 42

            model:
              name: meta-llama/Meta-Llama-3.1-8B
              revision: main
              torch_dtype: bf16
              from_scratch: true

            data:
              train_path: ./data/train.jsonl
              streaming: true

            training:
              mode: pretrain
              num_epochs: 1
              per_device_train_batch_size: 8
              gradient_accumulation_steps: 16
              learning_rate: 3.0e-4
              lr_scheduler_type: cosine
              warmup_steps: 2000
              weight_decay: 0.1
              max_grad_norm: 1.0
              max_seq_length: 4096
              fp16: false
              bf16: true
              gradient_checkpointing: true

            distributed:
              enabled: true
              framework: fsdp
        """),
        "rag": textwrap.dedent("""\
            # llm-forge config -- RAG (Retrieval-Augmented Generation)
            # Docs: https://github.com/Nagavenkatasai7/llm-forge

            project:
              name: my-rag-pipeline
              output_dir: ./outputs
              seed: 42

            model:
              name: meta-llama/Meta-Llama-3.1-8B-Instruct
              revision: main
              torch_dtype: bf16

            rag:
              enabled: true
              embedding_model: sentence-transformers/all-MiniLM-L6-v2
              vector_store: chroma
              chunk_size: 512
              chunk_overlap: 50
              top_k: 5
              knowledge_base_dir: ./knowledge_base
              persist_dir: ./vector_store

            data:
              train_path: ./data/train.jsonl

            training:
              mode: lora
              num_epochs: 3
              per_device_train_batch_size: 4
              learning_rate: 2.0e-4
              max_seq_length: 4096

            lora:
              r: 16
              alpha: 32
              dropout: 0.05
              target_modules: all-linear
        """),
        "full": textwrap.dedent("""\
            # llm-forge config -- Full fine-tuning
            # Docs: https://github.com/Nagavenkatasai7/llm-forge

            project:
              name: my-full-finetune
              output_dir: ./outputs
              seed: 42

            model:
              name: meta-llama/Meta-Llama-3.1-8B
              revision: main
              torch_dtype: bf16
              attn_implementation: flash_attention_2

            data:
              train_path: ./data/train.jsonl
              format: alpaca

            training:
              mode: full
              num_epochs: 3
              per_device_train_batch_size: 2
              gradient_accumulation_steps: 8
              learning_rate: 2.0e-5
              lr_scheduler_type: cosine
              warmup_ratio: 0.03
              weight_decay: 0.01
              max_grad_norm: 1.0
              max_seq_length: 2048
              fp16: false
              bf16: true
              gradient_checkpointing: true

            distributed:
              enabled: true
              framework: fsdp
        """),
    }

    return configs.get(template, configs["lora"])


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    config_path: str = typer.Argument(..., help="Path to the YAML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Validate a YAML configuration file.

    Parses the config, checks for errors, verifies hardware compatibility,
    and displays a configuration summary.
    """
    _setup_verbose(verbose)
    _banner()

    config = _load_config(config_path)

    console.print("[green]Config is valid.[/green]\n")
    _print_config_summary(config)

    # Hardware compatibility check
    console.print()
    _show_hardware_summary()

    try:
        from llm_forge.utils.gpu_utils import estimate_model_memory, get_available_vram

        model_name = _get_nested(config, "model", "name")
        if model_name:
            method = _get_nested(config, "training", "method") or "lora"
            precision = _get_nested(config, "model", "torch_dtype") or "float16"
            is_lora = method in ("lora", "qlora")
            lora_rank = None
            if is_lora:
                lora_rank = _get_nested(config, "lora", "rank") or 16

            estimate = estimate_model_memory(
                model_name,
                precision=precision,
                include_optimizer=True,
                include_gradients=True,
                lora_rank=lora_rank,
            )

            est_table = Table(title="Estimated Memory Requirements", expand=True)
            est_table.add_column("Component", style="bold")
            est_table.add_column("Memory (GB)", justify="right")

            est_table.add_row("Model weights", f"{estimate['model_weights_gb']:.2f}")
            est_table.add_row("Gradients", f"{estimate['gradients_gb']:.2f}")
            est_table.add_row("Optimizer states", f"{estimate['optimizer_gb']:.2f}")
            est_table.add_row("Activations (est.)", f"{estimate['activations_gb']:.2f}")
            est_table.add_row(
                "[bold]Total[/bold]",
                f"[bold]{estimate['total_gb']:.2f}[/bold]",
            )
            console.print(est_table)

            # Compare with available VRAM
            available = get_available_vram()
            if available:
                total_free = sum(available.values())
                if estimate["total_gb"] > total_free:
                    console.print(
                        f"\n[yellow]Warning:[/yellow] Estimated memory "
                        f"({estimate['total_gb']:.1f} GB) exceeds available VRAM "
                        f"({total_free:.1f} GB). Consider:\n"
                        "  - Using QLoRA (4-bit quantization)\n"
                        "  - Reducing batch size\n"
                        "  - Enabling gradient checkpointing\n"
                        "  - Using a smaller model"
                    )
                else:
                    console.print(
                        f"\n[green]Memory check passed:[/green] "
                        f"{estimate['total_gb']:.1f} GB required, "
                        f"{total_free:.1f} GB available."
                    )
    except Exception:
        pass  # Non-critical; skip if estimation fails


def _get_nested(obj: object, *keys: str) -> object:
    """Safely get a nested attribute or dict key."""
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key, None)
        else:
            return None
        if current is None:
            return None
    return current


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to the YAML config file."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and show plan without training."
    ),
    stages: str | None = typer.Option(
        None,
        "--stages",
        help="Comma-separated list of pipeline stages to run (e.g. 'data_loading,training').",
    ),
    skip_stages: str | None = typer.Option(
        None,
        "--skip-stages",
        help="Comma-separated list of pipeline stages to skip (e.g. 'iti_baking,evaluation').",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    no_auto_optimize: bool = typer.Option(
        False,
        "--no-auto-optimize",
        help="Disable hardware auto-optimization (use config values as-is).",
    ),
) -> None:
    """Run the training pipeline.

    Loads the configuration, detects hardware, applies auto-optimizations,
    and launches the training loop with Rich progress tracking.
    """
    _setup_verbose(verbose)
    _banner()
    _show_hardware_summary()

    cfg = _load_config(config)
    _print_config_summary(cfg)

    # Auto-optimize config for detected hardware (unless --no-auto-optimize)
    if no_auto_optimize:
        console.print("[dim]Hardware auto-optimization disabled (--no-auto-optimize).[/dim]\n")
    else:
        try:
            from llm_forge.config.hardware_detector import auto_optimize_config, detect_hardware

            hw = detect_hardware()
            cfg = auto_optimize_config(cfg, hw)
            console.print("[green]Config auto-optimized for detected hardware.[/green]\n")
        except ImportError:
            console.print(
                "[dim]Hardware auto-optimization not available "
                "(llm_forge.config.hardware_detector not found).[/dim]\n"
            )
        except Exception as exc:
            console.print(f"[yellow]Warning:[/yellow] Hardware auto-optimization failed: {exc}\n")

    # Parse stage filters
    include_stages = {s.strip() for s in stages.split(",")} if stages else None
    exclude_stages = {s.strip() for s in skip_stages.split(",")} if skip_stages else None

    if include_stages and exclude_stages:
        console.print("[red]Error:[/red] Cannot use --stages and --skip-stages together.")
        raise typer.Exit(code=1)

    # Show stage plan for dry-run or with stage filters
    if dry_run or include_stages or exclude_stages:
        try:
            from llm_forge.pipeline.dag_builder import DAGBuilder

            builder = DAGBuilder()
            dag = builder.build_dag(cfg)
            stage_table = Table(title="Pipeline Stages", expand=True)
            stage_table.add_column("Stage", style="bold")
            stage_table.add_column("Status", justify="center")
            stage_table.add_column("Description")
            for st in dag:
                # Determine effective enabled status
                effective = st.enabled
                if include_stages:
                    effective = st.name in include_stages
                elif exclude_stages:
                    effective = st.enabled and st.name not in exclude_stages
                status = "[green]run[/green]" if effective else "[dim]skip[/dim]"
                stage_table.add_row(st.name, status, st.description)
            console.print(stage_table)
        except Exception:
            pass

    if dry_run:
        console.print(
            Panel(
                "[bold yellow]DRY RUN[/bold yellow] -- No training will be performed.\n"
                "The configuration above shows what would be used.",
                border_style="yellow",
            )
        )
        raise typer.Exit()

    # Run the training pipeline
    console.print("[bold]Starting training pipeline...[/bold]\n")
    start_time = time.time()

    try:
        from llm_forge.pipeline import PipelineRunner

        runner = PipelineRunner()
        run_kwargs: dict = {"auto_optimize": False}
        # Pass stage filters only if PipelineRunner.run() accepts them
        import inspect

        _run_params = set(inspect.signature(runner.run).parameters)
        if include_stages and "include_stages" in _run_params:
            run_kwargs["include_stages"] = include_stages
        if exclude_stages and "exclude_stages" in _run_params:
            run_kwargs["exclude_stages"] = exclude_stages
        runner.run(cfg, **run_kwargs)
    except ImportError:
        # Fallback: try the training module directly
        try:
            from llm_forge.training import Trainer

            trainer = Trainer(cfg)
            trainer.run()
        except ImportError:
            console.print(
                "[red]Error:[/red] Training pipeline not found. "
                "Ensure llm_forge.pipeline or llm_forge.training is implemented."
            )
            raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as exc:
        _show_error_recovery(exc, cfg, verbose)
        raise typer.Exit(code=1)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    hours, minutes = divmod(minutes, 60)

    console.print(
        Panel(
            f"[bold green]Training complete![/bold green]\n"
            f"Total time: {hours}h {minutes}m {seconds}s",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@app.command(name="eval")
def evaluate(
    config: str = typer.Option(..., "--config", "-c", help="Path to the YAML config file."),
    model_path: str | None = typer.Option(
        None, "--model-path", "-m", help="Path to the trained model (overrides config)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run evaluation benchmarks on a trained model.

    Loads the model and runs configured benchmarks (perplexity, BLEU,
    ROUGE, task-specific evals).
    """
    _setup_verbose(verbose)
    _banner()
    _show_hardware_summary()

    cfg = _load_config(config)

    eval_model = model_path or cfg.training.output_dir
    if model_path:
        console.print(f"[bold]Model path override:[/bold] {model_path}")

    mp = Path(eval_model)
    if not mp.exists():
        console.print(f"[red]Error:[/red] Model path not found: {eval_model}")
        console.print("Train a model first, then run evaluation.")
        raise typer.Exit(code=1)

    console.print("[bold]Starting evaluation...[/bold]\n")
    start_time = time.time()

    try:
        from llm_forge.evaluation import BenchmarkRunner

        runner = BenchmarkRunner()
        results = runner.run_benchmarks(
            model_path=eval_model,
            tasks=cfg.evaluation.benchmarks if cfg.evaluation.enabled else None,
            num_fewshot=cfg.evaluation.num_fewshot,
            batch_size=cfg.evaluation.batch_size,
        )

        # Display results
        if results and isinstance(results, dict):
            results_table = Table(title="Evaluation Results", expand=True)
            results_table.add_column("Metric", style="bold cyan")
            results_table.add_column("Score", justify="right")

            for metric, score in results.items():
                if isinstance(score, float):
                    results_table.add_row(metric, f"{score:.4f}")
                else:
                    results_table.add_row(metric, str(score))

            console.print(results_table)
    except (ImportError, RuntimeError):
        console.print(
            "[red]Error:[/red] Evaluation module not found. "
            "Install eval dependencies: [cyan]pip install llm-forge[eval][/cyan]"
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"\n[red]Evaluation failed:[/red] {exc}")
        if verbose:
            console.print_exception(show_locals=True)
        raise typer.Exit(code=1)

    elapsed = time.time() - start_time
    console.print(f"\n[dim]Evaluation completed in {elapsed:.1f}s[/dim]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    config: str = typer.Option(..., "--config", "-c", help="Path to the YAML config file."),
    model_path: str | None = typer.Option(
        None, "--model-path", "-m", help="Path to the trained model (overrides config)."
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Server bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Server port."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Launch a model serving backend.

    Starts a FastAPI/Gradio server for inference with the trained model.
    """
    _setup_verbose(verbose)
    _banner()
    _show_hardware_summary()

    cfg = _load_config(config)

    if model_path:
        mp = Path(model_path)
        if not mp.exists():
            console.print(f"[red]Error:[/red] Model path not found: {model_path}")
            raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]Starting model server[/bold]\n"
            f"  Host: {host}\n"
            f"  Port: {port}\n"
            f"  Model: {model_path or _get_nested(cfg, 'model', 'name') or 'from config'}",
            title="[bold]Serving[/bold]",
            border_style="cyan",
        )
    )

    try:
        from llm_forge.serving import launch_server

        launch_server(
            cfg,
            model_path=model_path,
            host=host,
            port=port,
            console=console,
        )
    except ImportError:
        console.print(
            "[red]Error:[/red] Serving module not found. "
            "Install serving dependencies: [cyan]pip install llm-forge[serve][/cyan]"
        )
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    except Exception as exc:
        console.print(f"\n[red]Server error:[/red] {exc}")
        if verbose:
            console.print_exception(show_locals=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@app.command()
def export(
    config: str = typer.Option(..., "--config", "-c", help="Path to the YAML config file."),
    format: str = typer.Option(
        ...,
        "--format",
        "-f",
        help="Export format: safetensors, gguf, or onnx.",
    ),
    model_path: str | None = typer.Option(
        None, "--model-path", "-m", help="Path to the trained model (overrides config)."
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory for exported model."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Export a trained model to the specified format.

    Supported formats: safetensors (default HF format), GGUF (llama.cpp),
    ONNX (cross-platform inference).
    """
    _setup_verbose(verbose)
    _banner()

    valid_formats = ("safetensors", "gguf", "onnx")
    if format.lower() not in valid_formats:
        console.print(
            f"[red]Error:[/red] Unsupported format '{format}'. "
            f"Valid options: {', '.join(valid_formats)}"
        )
        raise typer.Exit(code=1)

    cfg = _load_config(config)

    export_dir = Path(output_dir) if output_dir else Path("./exported_model")
    export_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]Exporting model[/bold]\n"
            f"  Format: {format}\n"
            f"  Output: {export_dir.absolute()}\n"
            f"  Model:  {model_path or _get_nested(cfg, 'model', 'name') or 'from config'}",
            title="[bold]Export[/bold]",
            border_style="cyan",
        )
    )

    try:
        if format.lower() == "safetensors":
            _export_safetensors(cfg, model_path, export_dir)
        elif format.lower() == "gguf":
            _export_gguf(cfg, model_path, export_dir)
        elif format.lower() == "onnx":
            _export_onnx(cfg, model_path, export_dir)

        console.print(
            Panel(
                f"[bold green]Export complete![/bold green]\n"
                f"Files saved to: {export_dir.absolute()}",
                border_style="green",
            )
        )
    except Exception as exc:
        console.print(f"\n[red]Export failed:[/red] {exc}")
        if verbose:
            console.print_exception(show_locals=True)
        raise typer.Exit(code=1)


def _export_safetensors(cfg: object, model_path: str | None, output_dir: Path) -> None:
    """Export model to safetensors format."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Loading model...", total=None)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = model_path or _get_nested(cfg, "model", "name")
            if not model_name:
                raise ValueError("No model name or path specified.")

            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            progress.update(task, description="Saving as safetensors...")
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)

            progress.update(task, description="Export complete.", completed=True)
        except ImportError:
            raise RuntimeError("transformers library required for safetensors export.")


def _export_gguf(cfg: object, model_path: str | None, output_dir: Path) -> None:
    """Export model to GGUF format (for llama.cpp)."""
    import shutil

    convert_script = shutil.which("python3")
    if convert_script is None:
        raise RuntimeError("Python3 not found for GGUF conversion.")

    model_name = model_path or _get_nested(cfg, "model", "name")
    if not model_name:
        raise ValueError("No model name or path specified.")

    console.print(
        "[yellow]GGUF export requires llama.cpp's convert script.[/yellow]\n"
        "Steps to complete manually:\n"
        "  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp\n"
        "  2. Install requirements: pip install -r llama.cpp/requirements.txt\n"
        f"  3. Run: python llama.cpp/convert_hf_to_gguf.py {model_name} "
        f"--outfile {output_dir / 'model.gguf'}\n"
        f"  4. Optionally quantize: ./llama.cpp/llama-quantize "
        f"{output_dir / 'model.gguf'} {output_dir / 'model-q4_k_m.gguf'} q4_k_m"
    )

    # Try to find and run convert script automatically
    try:
        import subprocess

        llama_cpp_convert = shutil.which("convert_hf_to_gguf.py")
        if llama_cpp_convert:
            subprocess.run(
                [
                    sys.executable,
                    llama_cpp_convert,
                    model_name,
                    "--outfile",
                    str(output_dir / "model.gguf"),
                ],
                check=True,
            )
        else:
            console.print(
                "\n[dim]convert_hf_to_gguf.py not found on PATH. "
                "Follow the manual steps above.[/dim]"
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"GGUF conversion failed: {exc}") from exc


def _export_onnx(cfg: object, model_path: str | None, output_dir: Path) -> None:
    """Export model to ONNX format."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    model_name = model_path or _get_nested(cfg, "model", "name")
    if not model_name:
        raise ValueError("No model name or path specified.")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Exporting to ONNX...", total=None)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            progress.update(task, description="Running ONNX export...")

            try:
                from optimum.exporters.onnx import main_export

                main_export(
                    model_name_or_path=model_name,
                    output=str(output_dir),
                    task="text-generation",
                )
            except ImportError:
                # Fallback to torch.onnx
                import torch

                dummy = tokenizer("Hello world", return_tensors="pt", padding=True)
                torch.onnx.export(
                    model,
                    (dummy["input_ids"], dummy["attention_mask"]),
                    str(output_dir / "model.onnx"),
                    opset_version=17,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch", 1: "seq"},
                        "attention_mask": {0: "batch", 1: "seq"},
                        "logits": {0: "batch", 1: "seq"},
                    },
                )
                tokenizer.save_pretrained(output_dir)

            progress.update(task, description="ONNX export complete.", completed=True)
        except ImportError as exc:
            raise RuntimeError(
                f"Required library not installed for ONNX export: {exc}. "
                "Try: pip install optimum[onnxruntime]"
            ) from exc


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


@app.command()
def clean(
    config: str = typer.Option(..., "--config", "-c", help="Path to the YAML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run the data cleaning pipeline.

    Processes raw data through configured cleaning steps (deduplication,
    PII removal, quality filtering, etc.) without running training.
    """
    _setup_verbose(verbose)
    _banner()

    cfg = _load_config(config)

    console.print("[bold]Starting data cleaning pipeline...[/bold]\n")
    start_time = time.time()

    try:
        from llm_forge.data.cleaning import CleaningPipeline
        from llm_forge.data.loader import DataLoader

        data_cfg = cfg.data
        loader = DataLoader(
            path=data_cfg.train_path,
            streaming=data_cfg.streaming,
            num_workers=data_cfg.num_workers,
            max_samples=data_cfg.max_samples,
            seed=data_cfg.seed,
        )
        dataset = loader.load()
        console.print(f"Loaded {len(dataset):,} samples from {data_cfg.train_path}")

        # Determine text field
        text_field = "text"
        if "text" not in dataset.column_names:
            if data_cfg.output_field and data_cfg.output_field in dataset.column_names:
                text_field = data_cfg.output_field
            elif data_cfg.input_field and data_cfg.input_field in dataset.column_names:
                text_field = data_cfg.input_field
            elif dataset.column_names:
                text_field = dataset.column_names[0]

        pipeline = CleaningPipeline(
            config=data_cfg.cleaning,
            text_field=text_field,
        )
        cleaned_dataset, stats = pipeline.run(dataset)

        clean_table = Table(title="Cleaning Results", expand=True)
        clean_table.add_column("Metric", style="bold cyan")
        clean_table.add_column("Value", justify="right")
        clean_table.add_row("Initial samples", f"{stats.initial_count:,}")
        clean_table.add_row("Final samples", f"{stats.final_count:,}")
        clean_table.add_row("Removed", f"{stats.initial_count - stats.final_count:,}")
        if stats.skipped_steps:
            clean_table.add_row("Skipped steps", ", ".join(stats.skipped_steps))
        console.print(clean_table)

    except ImportError:
        console.print(
            "[red]Error:[/red] Cleaning module not found. "
            "Install cleaning dependencies: [cyan]pip install llm-forge[cleaning][/cyan]"
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"\n[red]Cleaning failed:[/red] {exc}")
        if verbose:
            console.print_exception(show_locals=True)
        raise typer.Exit(code=1)

    elapsed = time.time() - start_time
    console.print(
        Panel(
            f"[bold green]Cleaning complete![/bold green]\nTime: {elapsed:.1f}s",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# rag build
# ---------------------------------------------------------------------------


@rag_app.command("build")
def rag_build(
    config: str = typer.Option(..., "--config", "-c", help="Path to the YAML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Build a RAG knowledge base from documents.

    Processes documents, generates embeddings, and stores them in the
    configured vector store (ChromaDB, FAISS, etc.).
    """
    _setup_verbose(verbose)
    _banner()

    cfg = _load_config(config)

    console.print("[bold]Building RAG knowledge base...[/bold]\n")
    start_time = time.time()

    try:
        from llm_forge.rag import build_knowledge_base

        result = build_knowledge_base(cfg, console=console)

        if result and isinstance(result, dict):
            rag_table = Table(title="RAG Build Results", expand=True)
            rag_table.add_column("Metric", style="bold cyan")
            rag_table.add_column("Value", justify="right")

            for key, val in result.items():
                if isinstance(val, int):
                    rag_table.add_row(key, f"{val:,}")
                else:
                    rag_table.add_row(key, str(val))

            console.print(rag_table)

    except ImportError:
        console.print(
            "[red]Error:[/red] RAG module not found. "
            "Install RAG dependencies: [cyan]pip install llm-forge[rag][/cyan]"
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"\n[red]RAG build failed:[/red] {exc}")
        if verbose:
            console.print_exception(show_locals=True)
        raise typer.Exit(code=1)

    elapsed = time.time() - start_time
    console.print(
        Panel(
            f"[bold green]Knowledge base built![/bold green]\nTime: {elapsed:.1f}s",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# rag query
# ---------------------------------------------------------------------------


@rag_app.command("query")
def rag_query(
    question: str = typer.Argument(..., help="Question to query the RAG system."),
    config: str = typer.Option(..., "--config", "-c", help="Path to the YAML config file."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to retrieve."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Query the RAG knowledge base.

    Retrieves relevant documents and generates a response using the
    configured LLM with retrieval-augmented generation.
    """
    _setup_verbose(verbose)
    _banner()

    cfg = _load_config(config)

    console.print(f'[bold]Query:[/bold] "{question}"\n')

    try:
        from llm_forge.rag import query_knowledge_base

        result = query_knowledge_base(cfg, question=question, top_k=top_k, console=console)

        if result:
            # Display answer
            answer = result.get("answer", result.get("response", ""))
            if answer:
                console.print(
                    Panel(
                        answer,
                        title="[bold green]Answer[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

            # Display sources
            sources = result.get("sources", result.get("documents", []))
            if sources:
                console.print("\n[bold]Sources:[/bold]")
                for i, src in enumerate(sources, 1):
                    if isinstance(src, dict):
                        title = src.get("title", src.get("source", f"Document {i}"))
                        snippet = src.get("content", src.get("text", ""))[:200]
                        console.print(f"  {i}. [cyan]{title}[/cyan]")
                        if snippet:
                            console.print(f"     {snippet}...")
                    else:
                        text = str(src)[:200]
                        console.print(f"  {i}. {text}...")

    except ImportError:
        console.print(
            "[red]Error:[/red] RAG module not found. "
            "Install RAG dependencies: [cyan]pip install llm-forge[rag][/cyan]"
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"\n[red]RAG query failed:[/red] {exc}")
        if verbose:
            console.print_exception(show_locals=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command()
def info(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Display system information.

    Shows GPU details, VRAM, CUDA version, Python version, and installed
    ML package versions.
    """
    _setup_verbose(verbose)
    _banner()

    from llm_forge.utils.gpu_utils import format_gpu_info

    format_gpu_info(console)


# ---------------------------------------------------------------------------
# Error recovery helper
# ---------------------------------------------------------------------------


def _show_error_recovery(exc: Exception, config: object, verbose: bool) -> None:
    """Analyse a training failure and suggest fixes."""
    console.print(f"\n[red]Training failed:[/red] {exc}")
    if verbose:
        console.print_exception(show_locals=True)

    try:
        from llm_forge.utils.error_recovery import diagnose_error

        diagnosis = diagnose_error(exc, config=config)
        suggestions = diagnosis.suggestion_texts
    except ImportError:
        suggestions = [
            "Run 'llm-forge doctor' to check your environment",
            "Use --verbose for detailed error information",
        ]

    if suggestions:
        suggestion_text = "\n".join(f"  {i}. {s}" for i, s in enumerate(suggestions, 1))
        console.print(
            Panel(
                f"[bold yellow]Error Type:[/bold yellow] {type(exc).__name__}\n\n"
                f"[bold yellow]Suggested fixes:[/bold yellow]\n{suggestion_text}",
                title="Error Recovery",
                border_style="yellow",
            )
        )


# ---------------------------------------------------------------------------
# setup (interactive wizard)
# ---------------------------------------------------------------------------


@app.command()
def setup(
    output: str = typer.Option(
        "config.yaml",
        "--output",
        "-o",
        help="Output filename for the generated config.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Interactive training wizard — guided setup for your personal AI.

    Walks you through purpose selection, model choice, data configuration,
    and generates a ready-to-use YAML config optimised for your hardware.
    No technical knowledge required.
    """
    _setup_verbose(verbose)
    _banner()

    try:
        from llm_forge.wizard import TrainingWizard

        wizard = TrainingWizard(console=console)
        result = wizard.run(output_path=output)
        if result is None:
            raise typer.Exit()
    except ImportError:
        console.print(
            "[red]Error:[/red] Wizard module not found.  Ensure llm_forge.wizard is available."
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# doctor (diagnostics)
# ---------------------------------------------------------------------------


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run diagnostic checks on your environment.

    Checks Python version, required packages, GPU/MPS availability, memory,
    and provides recommendations for fixing any issues found.
    """
    _setup_verbose(verbose)
    _banner()

    checks_passed = 0
    checks_failed = 0
    warnings = 0

    table = Table(title="Environment Diagnostics", expand=True)
    table.add_column("Check", style="bold", width=30)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Details")

    # --- Python version ---
    import platform as plat

    py_ver = plat.python_version()
    py_major, py_minor = int(py_ver.split(".")[0]), int(py_ver.split(".")[1])
    if py_major >= 3 and py_minor >= 10:
        table.add_row("Python version", "[green]OK[/green]", f"{py_ver}")
        checks_passed += 1
    elif py_major >= 3 and py_minor >= 9:
        table.add_row("Python version", "[yellow]WARN[/yellow]", f"{py_ver} (3.10+ recommended)")
        warnings += 1
    else:
        table.add_row("Python version", "[red]FAIL[/red]", f"{py_ver} (need 3.10+)")
        checks_failed += 1

    # --- Required packages ---
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("trl", "TRL (SFT)"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
    ]
    optional_packages = [
        ("bitsandbytes", "BitsAndBytes (QLoRA)"),
        ("psutil", "psutil (system monitoring)"),
        ("rich", "Rich (UI)"),
        ("wandb", "Weights & Biases"),
        ("sklearn", "scikit-learn (ITI probing)"),
    ]

    for pkg_name, display_name in required_packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            table.add_row(display_name, "[green]OK[/green]", f"v{ver}")
            checks_passed += 1
        except ImportError:
            table.add_row(display_name, "[red]FAIL[/red]", "Not installed")
            checks_failed += 1

    for pkg_name, display_name in optional_packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            table.add_row(display_name, "[green]OK[/green]", f"v{ver}")
            checks_passed += 1
        except ImportError:
            table.add_row(display_name, "[dim]SKIP[/dim]", "Not installed (optional)")
            warnings += 1

    # --- GPU/MPS detection ---
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            table.add_row("CUDA GPU", "[green]OK[/green]", f"{gpu_name} ({vram_gb:.0f} GB)")
            checks_passed += 1
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            table.add_row("Apple MPS", "[green]OK[/green]", "Metal Performance Shaders available")
            checks_passed += 1
        else:
            table.add_row("GPU", "[yellow]WARN[/yellow]", "No GPU detected — CPU training only")
            warnings += 1
    except ImportError:
        table.add_row("GPU", "[red]FAIL[/red]", "PyTorch not installed")
        checks_failed += 1

    # --- System memory ---
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        avail_gb = mem.available / (1024**3)
        if avail_gb > 4:
            table.add_row(
                "System RAM", "[green]OK[/green]", f"{avail_gb:.1f} / {total_gb:.1f} GB available"
            )
            checks_passed += 1
        else:
            table.add_row(
                "System RAM", "[yellow]WARN[/yellow]", f"{avail_gb:.1f} / {total_gb:.1f} GB (low)"
            )
            warnings += 1
    except ImportError:
        table.add_row("System RAM", "[dim]SKIP[/dim]", "psutil not installed")
        warnings += 1

    # --- Disk space ---
    try:
        import shutil

        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024**3)
        if free_gb > 10:
            table.add_row("Disk space", "[green]OK[/green]", f"{free_gb:.0f} GB free")
            checks_passed += 1
        else:
            table.add_row("Disk space", "[yellow]WARN[/yellow]", f"{free_gb:.0f} GB free (< 10 GB)")
            warnings += 1
    except Exception:
        table.add_row("Disk space", "[dim]SKIP[/dim]", "Could not check")
        warnings += 1

    # --- macOS specific ---
    if plat.system() == "Darwin":
        try:
            from llm_forge.training.mac_utils import get_battery_status, get_thermal_state

            thermal = get_thermal_state()
            thermal_status = (
                "[green]OK[/green]" if thermal in ("nominal", "fair") else "[yellow]WARN[/yellow]"
            )
            table.add_row("Thermal state", thermal_status, thermal)

            batt = get_battery_status()
            if batt["available"]:
                plug = "plugged in" if batt["plugged_in"] else "on battery"
                batt_status = (
                    "[green]OK[/green]"
                    if batt["plugged_in"] or batt["percent"] > 20
                    else "[yellow]WARN[/yellow]"
                )
                table.add_row("Battery", batt_status, f"{batt['percent']}% ({plug})")
        except ImportError:
            pass

    console.print(table)

    # Summary
    checks_passed + checks_failed + warnings
    if checks_failed == 0:
        console.print(
            Panel(
                f"[bold green]All critical checks passed![/bold green]\n"
                f"{checks_passed} passed, {warnings} warnings, {checks_failed} failed",
                border_style="green",
            )
        )
    else:
        suggestions = []
        suggestions.append("pip install -e '.[all]'  # Install all dependencies")
        if py_minor < 10:
            suggestions.append("Upgrade to Python 3.10+")
        suggestion_text = "\n".join(f"  {s}" for s in suggestions)
        console.print(
            Panel(
                f"[bold red]{checks_failed} check(s) failed[/bold red]\n"
                f"{checks_passed} passed, {warnings} warnings\n\n"
                f"[bold]Suggested fixes:[/bold]\n{suggestion_text}",
                border_style="red",
            )
        )


# ---------------------------------------------------------------------------
# ui (Gradio dashboard)
# ---------------------------------------------------------------------------


@app.command()
def ui(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Network interface to bind to."),
    port: int = typer.Option(7860, "--port", "-p", help="Port number for the web server."),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link."),
    desktop: bool = typer.Option(
        False, "--desktop", "-d", help="Launch as a native desktop window instead of browser."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Launch the Gradio web dashboard for config building, training, and chat."""
    _setup_verbose(verbose)
    _banner()

    try:
        from llm_forge.ui import launch_ui
    except ImportError as exc:
        console.print(
            f"[red]Error:[/red] Could not import UI module: {exc}\n"
            "Install gradio with: [bold]pip install gradio[/bold]"
        )
        raise typer.Exit(code=1)

    if desktop:
        console.print(f"[bold cyan]Launching desktop app[/bold cyan] on [bold]{host}:{port}[/bold]")
    else:
        console.print(f"[bold cyan]Starting dashboard[/bold cyan] on [bold]{host}:{port}[/bold]")
        if share:
            console.print("[dim]Public share link will be generated...[/dim]")

    launch_ui(host=host, port=port, share=share, desktop=desktop)


# ---------------------------------------------------------------------------
# runs (run history)
# ---------------------------------------------------------------------------


runs_app = typer.Typer(
    name="runs",
    help="Training run history management",
    no_args_is_help=True,
)
app.add_typer(runs_app, name="runs")


def _get_runs_file(output_dir: str = ".") -> Path:
    """Return the path to the runs history file."""
    return Path(output_dir) / ".llm-forge-runs.json"


def _load_runs(runs_file: Path) -> list:
    """Load runs from the history file."""
    if not runs_file.exists():
        return []
    import json

    with open(runs_file) as f:
        return json.load(f)


def _save_run(runs_file: Path, run_data: dict) -> None:
    """Append a run to the history file."""
    import json

    runs = _load_runs(runs_file)
    runs.append(run_data)
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_file, "w") as f:
        json.dump(runs, f, indent=2, default=str)


@runs_app.command("list")
def runs_list(
    output_dir: str = typer.Option(".", "--dir", "-d", help="Directory to look for run history."),
) -> None:
    """List all recorded training runs."""
    _banner()
    runs_file = _get_runs_file(output_dir)
    runs = _load_runs(runs_file)

    if not runs:
        console.print("[dim]No training runs recorded yet.[/dim]")
        console.print("Runs are recorded automatically when you use 'llm-forge train'.")
        return

    table = Table(title=f"Training Runs ({len(runs)} total)", expand=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Timestamp", width=20)
    table.add_column("Model", width=30)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Loss", justify="right", width=10)
    table.add_column("Duration", justify="right", width=10)

    for i, run in enumerate(runs):
        status_str = (
            "[green]done[/green]"
            if run.get("status") == "completed"
            else "[red]failed[/red]"
            if run.get("status") == "failed"
            else f"[yellow]{run.get('status', '?')}[/yellow]"
        )
        loss = run.get("final_loss")
        loss_str = f"{loss:.4f}" if loss is not None else "—"
        dur = run.get("duration_seconds")
        dur_str = f"{dur / 60:.1f}m" if dur is not None else "—"

        table.add_row(
            str(i + 1),
            run.get("timestamp", "—"),
            run.get("model", "—"),
            status_str,
            loss_str,
            dur_str,
        )

    console.print(table)


@runs_app.command("show")
def runs_show(
    run_id: int = typer.Argument(..., help="Run number (from 'runs list')."),
    output_dir: str = typer.Option(".", "--dir", "-d", help="Directory with run history."),
) -> None:
    """Show details for a specific training run."""
    import json as _json

    _banner()
    runs_file = _get_runs_file(output_dir)
    runs = _load_runs(runs_file)

    idx = run_id - 1
    if idx < 0 or idx >= len(runs):
        console.print(f"[red]Error:[/red] Run #{run_id} not found ({len(runs)} runs total).")
        raise typer.Exit(code=1)

    run = runs[idx]
    console.print(
        Panel(
            _json.dumps(run, indent=2, default=str),
            title=f"Run #{run_id} Details",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
