"""Unified training orchestrator for the llm-forge platform.

The :class:`Trainer` class is the central entry point for all training
workflows.  It reads an ``LLMForgeConfig``, detects available hardware,
selects the appropriate training backend (fine-tuning, pre-training, or
alignment), and drives the full pipeline from data loading through
evaluation and model export.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any

from llm_forge.training.alignment import AlignmentTrainer
from llm_forge.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    GPUMonitorCallback,
    RichProgressCallback,
    WandBCallback,
)
from llm_forge.training.finetuner import FineTuner
from llm_forge.training.pretrainer import PreTrainer
from llm_forge.utils.logging import get_logger

logger = get_logger("training.trainer")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from datasets import Dataset

    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

try:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import sentry_sdk

    _sentry_dsn = os.environ.get("SENTRY_DSN", "")
    if _sentry_dsn:
        sentry_sdk.init(
            dsn=_sentry_dsn,
            traces_sample_rate=0.0,
            send_default_pii=False,
            environment=os.environ.get("LLM_FORGE_ENV", "training"),
        )
except ImportError:
    pass


# ============================================================================
# Trainer
# ============================================================================


class Trainer:
    """Unified training orchestrator.

    Dispatches to :class:`FineTuner`, :class:`PreTrainer`, or
    :class:`AlignmentTrainer` based on ``config.training.mode``.

    The ``run()`` method executes the complete training pipeline:

    1. Load and validate configuration.
    2. Detect hardware (GPU count, VRAM, compute capability).
    3. Auto-optimise settings (batch size, precision, distributed strategy).
    4. Load the base model (or build from scratch for pre-training).
    5. Apply adapters (LoRA / QLoRA) if applicable.
    6. Load and preprocess the dataset.
    7. Train with Rich progress display and configurable callbacks.
    8. Evaluate on the held-out set.
    9. Export / merge the final model.

    Parameters
    ----------
    config : object
        An ``LLMForgeConfig`` instance (or any config object with
        ``.model``, ``.training``, ``.data``, etc. attributes).
    dry_run : bool
        When *True*, print a summary of what *would* happen but do not
        actually train.
    """

    def __init__(self, config: Any, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self.console = Console() if _RICH_AVAILABLE else None
        self._hardware_info: dict[str, Any] = {}
        self._train_result: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    # ================================================================== #
    # Public API
    # ================================================================== #

    def run(self) -> dict[str, Any]:
        """Execute the full training pipeline.

        Returns
        -------
        dict[str, Any]
            Summary of the training run including output path, metrics,
            and elapsed time.
        """
        start_time = time.time()
        results: dict[str, Any] = {"status": "pending"}

        try:
            # Step 1: Detect hardware
            self._detect_hardware()

            # Step 2: Auto-optimise configuration
            self._auto_optimize()

            # Step 3: Print run plan
            self._print_run_plan()

            # Dry-run exits here
            if self.dry_run:
                results["status"] = "dry_run"
                results["plan"] = self._build_plan_dict()
                self._log("Dry run complete. No training was performed.")
                return results

            # Step 4: Dispatch to the appropriate engine
            mode = self.config.training.mode
            if mode == "pretrain":
                results = self._run_pretrain()
            elif mode == "dpo":
                results = self._run_alignment()
            else:
                # lora, qlora, full
                results = self._run_finetune()

            elapsed = time.time() - start_time
            results["elapsed_seconds"] = round(elapsed, 2)
            results["elapsed_human"] = self._format_elapsed(elapsed)
            results["status"] = "success"

            self._print_summary(results)

        except KeyboardInterrupt:
            results["status"] = "interrupted"
            self._log("Training interrupted by user.")
        except Exception as exc:
            results["status"] = "error"
            results["error"] = str(exc)
            logger.error("Training failed: %s", exc, exc_info=True)
            raise

        return results

    # ================================================================== #
    # Hardware detection
    # ================================================================== #

    def _detect_hardware(self) -> None:
        """Detect available hardware and store info."""
        info: dict[str, Any] = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpus": [],
            "mps_available": False,
            "total_vram_gb": 0.0,
        }

        if _TORCH_AVAILABLE:
            info["cuda_available"] = torch.cuda.is_available()
            info["mps_available"] = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )

            if info["cuda_available"]:
                info["gpu_count"] = torch.cuda.device_count()
                total_vram = 0.0
                for i in range(info["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    vram_gb = props.total_memory / (1024**3)
                    total_vram += vram_gb
                    info["gpus"].append(
                        {
                            "index": i,
                            "name": props.name,
                            "vram_gb": round(vram_gb, 2),
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )
                info["total_vram_gb"] = round(total_vram, 2)

                # Check BF16 support
                info["bf16_supported"] = all(
                    torch.cuda.get_device_capability(i)[0] >= 8 for i in range(info["gpu_count"])
                )
            else:
                info["bf16_supported"] = False

        self._hardware_info = info
        logger.info(
            "Hardware: %d GPU(s), %.1f GB total VRAM, CUDA=%s, MPS=%s",
            info["gpu_count"],
            info["total_vram_gb"],
            info["cuda_available"],
            info["mps_available"],
        )

    # ================================================================== #
    # Auto-optimisation
    # ================================================================== #

    def _auto_optimize(self) -> None:
        """Auto-adjust configuration based on detected hardware."""
        hw = self._hardware_info
        training = self.config.training

        # Fall back to fp16 if bf16 is not supported
        if training.bf16 and not hw.get("bf16_supported", False):
            if hw.get("cuda_available"):
                logger.warning("BF16 not supported on this hardware. Falling back to FP16.")
                training.bf16 = False
                training.fp16 = True

        # Auto-detect flash attention availability
        model_cfg = self.config.model
        if model_cfg.attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.info("flash_attn not installed. Falling back to sdpa attention.")
                model_cfg.attn_implementation = "sdpa"

        # Adjust optimizer for QLoRA
        if training.mode == "qlora" and training.optim == "adamw_torch":
            training.optim = "paged_adamw_32bit"
            logger.info("QLoRA mode: using paged_adamw_32bit optimizer")

    # ================================================================== #
    # Fine-tuning pipeline
    # ================================================================== #

    def _run_finetune(self) -> dict[str, Any]:
        """Execute the LoRA / QLoRA / full fine-tuning pipeline."""
        config = self.config
        finetuner = FineTuner(config)

        # Load model and tokenizer
        self._log("Loading model and tokenizer...")
        model, tokenizer = finetuner.setup_model()
        self._model = model
        self._tokenizer = tokenizer

        # Apply LoRA adapters
        if config.training.mode in ("lora", "qlora"):
            self._log("Applying LoRA adapters...")
            model = finetuner.apply_lora(model)
            self._model = model

        # Load and preprocess dataset
        self._log("Loading and preprocessing dataset...")
        train_dataset, eval_dataset = self._load_and_preprocess_data(tokenizer)

        # Build callbacks
        callbacks = self._build_callbacks()

        # Train
        self._log("Starting training...")
        result = finetuner.train(
            model=model,
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        # Merge and save
        if config.training.mode in ("lora", "qlora"):
            if hasattr(config, "serving") and config.serving.merge_adapter:
                self._log("Merging LoRA adapters...")
                merge_dir = finetuner.merge_and_save()
                return {
                    "output_dir": str(config.training.output_dir),
                    "merged_dir": str(merge_dir),
                    "mode": config.training.mode,
                    "train_result": self._extract_metrics(result),
                }

        return {
            "output_dir": str(config.training.output_dir),
            "mode": config.training.mode,
            "train_result": self._extract_metrics(result),
        }

    # ================================================================== #
    # Pre-training pipeline
    # ================================================================== #

    def _run_pretrain(self) -> dict[str, Any]:
        """Execute the pre-training from scratch pipeline."""
        config = self.config
        pretrainer = PreTrainer(config)

        # Train tokenizer if corpus is available
        self._log("Training BPE tokenizer...")
        try:
            tokenizer = pretrainer.train_tokenizer(
                corpus_path=config.data.train_path,
                vocab_size=32000,
            )
        except FileNotFoundError:
            # If train_path is an HF dataset, load a pre-existing tokenizer
            self._log("Using pre-existing tokenizer (HF dataset detected)...")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            pretrainer.tokenizer = tokenizer

        self._tokenizer = tokenizer

        # Build model
        self._log("Building model from scratch...")
        model = pretrainer.build_model(model_size="125M")
        self._model = model

        # Load and preprocess data
        self._log("Loading training data...")
        train_dataset, eval_dataset = self._load_and_preprocess_data(tokenizer)

        # Build callbacks
        callbacks = self._build_callbacks()

        # Train
        self._log("Starting pre-training...")
        result = pretrainer.train(
            model=model,
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        return {
            "output_dir": str(config.training.output_dir),
            "mode": "pretrain",
            "train_result": self._extract_metrics(result),
        }

    # ================================================================== #
    # Alignment pipeline
    # ================================================================== #

    def _run_alignment(self) -> dict[str, Any]:
        """Execute the DPO alignment pipeline."""
        config = self.config
        aligner = AlignmentTrainer(config)

        # Setup DPO (loads model + ref model + tokenizer)
        self._log("Setting up DPO training...")
        model, ref_model, tokenizer = aligner.setup_dpo()
        self._model = model
        self._tokenizer = tokenizer

        # Load preference dataset
        self._log("Loading preference dataset...")
        from llm_forge.data.loader import DataLoader

        loader = DataLoader(
            path=config.data.train_path,
            max_samples=config.data.max_samples,
            seed=config.data.seed,
        )
        raw_dataset = loader.load()

        # Prepare preference pairs
        pref_dataset = aligner.prepare_preference_dataset(raw_dataset)

        # Split for eval
        eval_dataset = None
        if config.data.eval_path:
            eval_loader = DataLoader(path=config.data.eval_path, seed=config.data.seed)
            eval_raw = eval_loader.load()
            eval_dataset = aligner.prepare_preference_dataset(eval_raw)
        elif config.data.test_size > 0:
            split = pref_dataset.train_test_split(
                test_size=config.data.test_size, seed=config.data.seed
            )
            pref_dataset = split["train"]
            eval_dataset = split["test"]

        # Build callbacks
        callbacks = self._build_callbacks()

        # DPO Training
        self._log("Starting DPO training...")
        result = aligner.train_dpo(
            model=model,
            dataset=pref_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        return {
            "output_dir": str(config.training.output_dir),
            "mode": "dpo",
            "train_result": self._extract_metrics(result),
        }

    # ================================================================== #
    # Data loading
    # ================================================================== #

    def _load_and_preprocess_data(
        self,
        tokenizer: Any,
    ) -> tuple:
        """Load dataset and preprocess for training.

        Returns
        -------
        tuple[Dataset, Dataset | None]
            ``(train_dataset, eval_dataset)``
        """
        config = self.config
        data_cfg = config.data

        from llm_forge.data.loader import DataLoader
        from llm_forge.data.preprocessor import DataPreprocessor

        # Load training data
        loader = DataLoader(
            path=data_cfg.train_path,
            streaming=data_cfg.streaming,
            num_workers=data_cfg.num_workers,
            max_samples=data_cfg.max_samples,
            seed=data_cfg.seed,
        )
        train_dataset = loader.load()
        logger.info("Training data loaded: %d samples", len(train_dataset))

        # Load or split eval data
        eval_dataset = None
        if data_cfg.eval_path:
            eval_loader = DataLoader(
                path=data_cfg.eval_path,
                seed=data_cfg.seed,
            )
            eval_dataset = eval_loader.load()
            logger.info("Evaluation data loaded: %d samples", len(eval_dataset))

        # Format dataset
        preprocessor = DataPreprocessor(
            format_type=data_cfg.format,
            input_field=data_cfg.input_field,
            output_field=data_cfg.output_field,
            context_field=data_cfg.context_field,
            system_prompt=data_cfg.system_prompt,
            max_seq_length=config.model.max_seq_length,
        )

        # Use chat template if available, otherwise standard formatting
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            train_dataset = preprocessor.format_for_chat_template(train_dataset, tokenizer)
            if eval_dataset is not None:
                eval_dataset = preprocessor.format_for_chat_template(eval_dataset, tokenizer)
        else:
            train_dataset = preprocessor.format_dataset(train_dataset)
            if eval_dataset is not None:
                eval_dataset = preprocessor.format_dataset(eval_dataset)

        # Split if no eval dataset
        if eval_dataset is None and data_cfg.test_size > 0:
            train_dataset, eval_dataset = preprocessor.split_dataset(
                train_dataset,
                test_size=data_cfg.test_size,
                seed=data_cfg.seed,
            )

        logger.info(
            "Datasets ready: %d train, %s eval",
            len(train_dataset),
            len(eval_dataset) if eval_dataset is not None else "none",
        )

        return train_dataset, eval_dataset

    # ================================================================== #
    # Callbacks
    # ================================================================== #

    def _build_callbacks(self) -> list[Any]:
        """Build the list of training callbacks based on config."""
        callbacks: list[Any] = []
        config = self.config

        # Rich progress
        if _RICH_AVAILABLE:
            with contextlib.suppress(ImportError):
                callbacks.append(RichProgressCallback())

        # GPU monitoring
        if self._hardware_info.get("cuda_available"):
            callbacks.append(GPUMonitorCallback(log_every_n_steps=50))

        # WandB
        if "wandb" in config.training.report_to:
            try:
                callbacks.append(
                    WandBCallback(
                        project="llm-forge",
                        run_name=f"{config.model.name}-{config.training.mode}",
                        tags=[config.training.mode],
                    )
                )
            except ImportError:
                logger.warning("wandb not installed; skipping WandB callback.")

        # Early stopping
        if config.training.eval_strategy != "no":
            callbacks.append(EarlyStoppingCallback(patience=3, min_delta=0.001))

        # Timed checkpoints
        callbacks.append(
            CheckpointCallback(
                save_every_n_minutes=30.0,
                max_checkpoints=config.training.save_total_limit,
            )
        )

        return callbacks

    # ================================================================== #
    # Display helpers
    # ================================================================== #

    def _print_run_plan(self) -> None:
        """Print a Rich table summarising the planned training run."""
        if not _RICH_AVAILABLE or self.console is None:
            self._log_plan_text()
            return

        config = self.config
        hw = self._hardware_info

        plan_table = Table(
            title="llm-forge Training Plan",
            show_header=False,
            expand=True,
            border_style="blue",
        )
        plan_table.add_column("Setting", style="bold cyan", width=30)
        plan_table.add_column("Value")

        plan_table.add_row("Mode", config.training.mode)
        plan_table.add_row("Model", config.model.name)
        plan_table.add_row("Precision", config.model.torch_dtype)
        plan_table.add_row("Max Seq Length", str(config.model.max_seq_length))
        plan_table.add_row("Attention", config.model.attn_implementation)

        plan_table.add_row("", "")  # spacer
        plan_table.add_row("Dataset", config.data.train_path)
        plan_table.add_row("Format", config.data.format)

        plan_table.add_row("", "")
        plan_table.add_row("Epochs", str(config.training.num_epochs))
        plan_table.add_row("Batch Size", str(config.training.per_device_train_batch_size))
        plan_table.add_row("Grad Accum Steps", str(config.training.gradient_accumulation_steps))
        plan_table.add_row("Learning Rate", f"{config.training.learning_rate:.2e}")
        plan_table.add_row("Optimizer", config.training.optim)
        plan_table.add_row("Scheduler", config.training.lr_scheduler_type)
        plan_table.add_row("Grad Checkpointing", str(config.training.gradient_checkpointing))

        if config.training.mode in ("lora", "qlora"):
            plan_table.add_row("", "")
            plan_table.add_row("LoRA Rank", str(config.lora.r))
            plan_table.add_row("LoRA Alpha", str(config.lora.alpha))
            plan_table.add_row("LoRA Dropout", f"{config.lora.dropout:.2f}")
            targets = config.lora.target_modules
            targets_display = targets if isinstance(targets, str) else ", ".join(targets)
            plan_table.add_row("LoRA Targets", targets_display)

        plan_table.add_row("", "")
        plan_table.add_row("Output Dir", config.training.output_dir)
        plan_table.add_row("GPUs", str(hw.get("gpu_count", 0)))
        plan_table.add_row("Total VRAM", f"{hw.get('total_vram_gb', 0):.1f} GB")

        if self.dry_run:
            plan_table.add_row("", "")
            plan_table.add_row(
                "[yellow]DRY RUN[/yellow]", "[yellow]No training will be performed[/yellow]"
            )

        self.console.print(plan_table)

    def _log_plan_text(self) -> None:
        """Fallback plain-text plan output when Rich is not available."""
        config = self.config
        lines = [
            "=== llm-forge Training Plan ===",
            f"Mode:        {config.training.mode}",
            f"Model:       {config.model.name}",
            f"Dataset:     {config.data.train_path}",
            f"Epochs:      {config.training.num_epochs}",
            f"Batch Size:  {config.training.per_device_train_batch_size}",
            f"LR:          {config.training.learning_rate}",
            f"Output:      {config.training.output_dir}",
        ]
        if self.dry_run:
            lines.append("** DRY RUN -- no training will be performed **")
        for line in lines:
            logger.info(line)

    def _print_summary(self, results: dict[str, Any]) -> None:
        """Print a summary panel after training completes."""
        if not _RICH_AVAILABLE or self.console is None:
            logger.info("Training complete: %s", json.dumps(results, indent=2, default=str))
            return

        summary_parts = [
            f"[bold green]Status:[/bold green] {results.get('status', 'unknown')}",
            f"[bold]Mode:[/bold] {results.get('mode', 'N/A')}",
            f"[bold]Output:[/bold] {results.get('output_dir', 'N/A')}",
            f"[bold]Time:[/bold] {results.get('elapsed_human', 'N/A')}",
        ]

        train_result = results.get("train_result", {})
        if train_result:
            if "train_loss" in train_result:
                summary_parts.append(f"[bold]Final Loss:[/bold] {train_result['train_loss']:.4f}")
            if "train_runtime" in train_result:
                summary_parts.append(
                    f"[bold]Train Runtime:[/bold] {train_result['train_runtime']:.1f}s"
                )
            if "train_samples_per_second" in train_result:
                summary_parts.append(
                    f"[bold]Throughput:[/bold] {train_result['train_samples_per_second']:.1f} samples/s"
                )

        if "merged_dir" in results:
            summary_parts.append(f"[bold]Merged Model:[/bold] {results['merged_dir']}")

        self.console.print(
            Panel(
                "\n".join(summary_parts),
                title="llm-forge Training Complete",
                border_style="green",
            )
        )

    # ================================================================== #
    # Utility helpers
    # ================================================================== #

    def _build_plan_dict(self) -> dict[str, Any]:
        """Build a dictionary representation of the training plan for dry-run."""
        config = self.config
        plan = {
            "mode": config.training.mode,
            "model": config.model.name,
            "precision": config.model.torch_dtype,
            "max_seq_length": config.model.max_seq_length,
            "dataset": config.data.train_path,
            "epochs": config.training.num_epochs,
            "batch_size": config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
            "optimizer": config.training.optim,
            "scheduler": config.training.lr_scheduler_type,
            "output_dir": config.training.output_dir,
            "hardware": self._hardware_info,
        }

        if config.training.mode in ("lora", "qlora"):
            plan["lora"] = {
                "rank": config.lora.r,
                "alpha": config.lora.alpha,
                "dropout": config.lora.dropout,
                "target_modules": config.lora.target_modules,
            }

        return plan

    @staticmethod
    def _extract_metrics(result: Any) -> dict[str, Any]:
        """Extract key metrics from a HF TrainOutput."""
        if result is None:
            return {}
        metrics: dict[str, Any] = {}
        if hasattr(result, "metrics"):
            metrics = dict(result.metrics)
        if hasattr(result, "training_loss"):
            metrics["train_loss"] = result.training_loss
        return metrics

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}m"
        hours = minutes / 60
        remaining_min = minutes % 60
        return f"{hours:.0f}h {remaining_min:.0f}m"

    def _log(self, message: str) -> None:
        """Log a message and optionally display via Rich."""
        logger.info(message)
        if self.console is not None and _RICH_AVAILABLE:
            self.console.print(f"  [dim]>>>[/dim] {message}")
