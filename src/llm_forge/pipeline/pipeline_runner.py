"""End-to-end pipeline runner for llm-forge.

Orchestrates the complete training pipeline from configuration loading
through data preparation, training, evaluation, and model export with
stage tracking, error handling, and resume support.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_forge.config.hardware_detector import auto_optimize_config, detect_hardware
from llm_forge.config.schema import LLMForgeConfig
from llm_forge.config.validator import (
    validate_config_dict,
    validate_config_file,
)
from llm_forge.pipeline.config_translator import ConfigTranslator
from llm_forge.pipeline.dag_builder import DAGBuilder, PipelineStage
from llm_forge.pipeline.preset_resolver import PresetResolver
from llm_forge.utils.logging import get_logger

logger = get_logger("pipeline.runner")

# ---------------------------------------------------------------------------
# Optional: Rich progress display
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Stage status tracking
# ---------------------------------------------------------------------------

_STATUS_PENDING = "pending"
_STATUS_RUNNING = "running"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_SKIPPED = "skipped"

# ---------------------------------------------------------------------------
# Error recovery suggestion map
# ---------------------------------------------------------------------------

_RECOVERY_HINTS: dict[str, list[str]] = {
    "CUDA out of memory": [
        "Reduce per_device_train_batch_size (try 1 or 2)",
        "Enable gradient_checkpointing: true",
        "Switch to mode: qlora for 4-bit training",
        "Reduce max_seq_length (try 512 or 1024)",
    ],
    "No module named": [
        "Install missing package: pip install <package_name>",
        "Or install all dependencies: pip install llm-forge[all]",
    ],
    "Unable to load model": [
        "Check model name is correct on HuggingFace Hub",
        "Try adding trust_remote_code: true to model config",
        "Check your internet connection for model download",
    ],
    "NaN": [
        "Reduce learning_rate (try 1e-5)",
        "Check your training data for empty or corrupt samples",
        "Disable neftune_noise_alpha (set to null)",
    ],
    "expected scalar": [
        "Your tokenizer may be missing a pad_token",
        "Try a different model or add pad_token manually",
    ],
    "out of memory": [
        "Reduce per_device_train_batch_size (try 1 or 2)",
        "Enable gradient_checkpointing: true",
        "Switch to mode: qlora for 4-bit training",
        "Reduce max_seq_length (try 512 or 1024)",
    ],
    "Connection error": [
        "Check your internet connection",
        "If behind a proxy, set HTTPS_PROXY environment variable",
        "Try downloading the model/dataset manually first",
    ],
    "Permission denied": [
        "Check file/directory permissions on the output path",
        "Ensure the output directory is writable",
        "Try running with a different output_dir",
    ],
    "Tokenizer": [
        "Ensure the tokenizer matches the model",
        "Try adding trust_remote_code: true",
        "Check if the model repo has a tokenizer_config.json",
    ],
}


def _suggest_recovery(error_msg: str) -> list[str]:
    """Return recovery suggestions based on the error message."""
    suggestions: list[str] = []
    error_lower = error_msg.lower()
    for pattern, hints in _RECOVERY_HINTS.items():
        if pattern.lower() in error_lower:
            suggestions.extend(hints)
    return suggestions or ["Check the full error log above", "Try: llm-forge doctor"]


class _StageStatus:
    """Track execution status for a single pipeline stage."""

    __slots__ = ("name", "status", "start_time", "end_time", "error", "duration_seconds")

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = _STATUS_PENDING
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.error: str | None = None
        self.duration_seconds: float = 0.0

    def mark_running(self) -> None:
        self.status = _STATUS_RUNNING
        self.start_time = time.time()

    def mark_completed(self) -> None:
        self.status = _STATUS_COMPLETED
        self.end_time = time.time()
        if self.start_time:
            self.duration_seconds = self.end_time - self.start_time

    def mark_failed(self, error: str) -> None:
        self.status = _STATUS_FAILED
        self.end_time = time.time()
        self.error = error
        if self.start_time:
            self.duration_seconds = self.end_time - self.start_time

    def mark_skipped(self) -> None:
        self.status = _STATUS_SKIPPED


# ============================================================================
# PipelineRunner
# ============================================================================


class PipelineRunner:
    """Orchestrate the full llm-forge training pipeline.

    Handles configuration loading, hardware detection, DAG construction,
    ordered stage execution, error recovery, and artifact persistence.

    Examples
    --------
    >>> runner = PipelineRunner()
    >>> runner.run("config.yaml")

    >>> runner.dry_run("config.yaml")
    """

    def __init__(self) -> None:
        self._dag_builder = DAGBuilder()
        self._preset_resolver = PresetResolver()
        self._console = Console() if _RICH_AVAILABLE else None

    # ------------------------------------------------------------------ #
    # Public API: run
    # ------------------------------------------------------------------ #

    def run(
        self,
        config_path_or_config: str | Path | dict[str, Any] | LLMForgeConfig,
        resume_from: str | None = None,
        auto_optimize: bool = True,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Execute the full pipeline end-to-end.

        Parameters
        ----------
        config_path_or_config : str, Path, dict, or LLMForgeConfig
            Path to a YAML config file, a raw config dict, or a validated
            ``LLMForgeConfig`` object.  If the dict contains a ``"preset"``
            key, the preset is resolved and merged with overrides.
        resume_from : str, optional
            Name of the stage to resume from.  Stages before this one
            are skipped (their outputs should exist on disk).
        auto_optimize : bool
            Whether to run hardware detection and auto-optimise the config.
        stop_event : threading.Event, optional
            If provided, the pipeline checks this event between stages
            and passes it to the training stage for graceful cancellation.

        Returns
        -------
        dict
            Pipeline context with all intermediate and final artifacts.
        """
        pipeline_start = time.time()
        logger.info("=" * 60)
        logger.info("llm-forge pipeline starting")
        logger.info("=" * 60)

        # Step 1: Load and validate configuration
        config = self._load_config(config_path_or_config)
        logger.info("Configuration loaded and validated")

        # Step 2: Hardware detection and auto-optimisation
        if auto_optimize:
            hardware = detect_hardware()
            logger.info("\n%s", hardware.summary())
            config = auto_optimize_config(config, hardware)
            logger.info("Configuration auto-optimised for detected hardware")

        # Step 3: Build the pipeline DAG
        stages = self._dag_builder.build_dag(config)

        # Step 4: Initialise stage tracking
        statuses = {s.name: _StageStatus(s.name) for s in stages}

        # Step 5: Determine resume point
        skip_stages: set = set()
        if resume_from is not None:
            stage_names = [s.name for s in stages]
            if resume_from not in stage_names:
                raise ValueError(
                    f"Cannot resume from unknown stage '{resume_from}'. "
                    f"Available stages: {stage_names}"
                )
            for name in stage_names:
                if name == resume_from:
                    break
                skip_stages.add(name)
            logger.info(
                "Resuming from stage '%s' (skipping: %s)",
                resume_from,
                ", ".join(skip_stages) if skip_stages else "none",
            )

        # Step 6: Display pipeline plan
        self._display_pipeline_plan(stages, statuses, skip_stages)

        # Step 7: Execute stages in order
        context: dict[str, Any] = {"config": config}
        if stop_event is not None:
            context["stop_event"] = stop_event

        for stage in stages:
            # Check if stop was requested between stages
            if stop_event is not None and stop_event.is_set():
                logger.info("Pipeline stop requested — halting before stage '%s'", stage.name)
                break

            if not stage.enabled:
                statuses[stage.name].mark_skipped()
                logger.info("Stage '%s' disabled, skipping", stage.name)
                continue

            if stage.name in skip_stages:
                statuses[stage.name].mark_skipped()
                logger.info("Stage '%s' skipped (resuming)", stage.name)
                continue

            # Execute the stage
            statuses[stage.name].mark_running()
            self._display_stage_start(stage, statuses)

            try:
                logger.info(
                    "--- Stage '%s' starting: %s ---",
                    stage.name,
                    stage.description,
                )
                context = stage.callable(context)
                statuses[stage.name].mark_completed()
                logger.info(
                    "--- Stage '%s' completed in %.1fs ---",
                    stage.name,
                    statuses[stage.name].duration_seconds,
                )

            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                statuses[stage.name].mark_failed(error_msg)
                logger.error(
                    "--- Stage '%s' FAILED: %s ---",
                    stage.name,
                    error_msg,
                )
                logger.debug(traceback.format_exc())

                # Suggest recovery actions based on error message
                suggestions = _suggest_recovery(str(exc))
                if suggestions:
                    logger.error("Recovery suggestions:")
                    for i, hint in enumerate(suggestions, 1):
                        logger.error("  %d. %s", i, hint)

                # Save progress checkpoint so the user can resume
                self._save_checkpoint(config, context, statuses, stage.name)

                hint_text = "\n".join(f"  - {h}" for h in suggestions)
                raise RuntimeError(
                    f"Pipeline failed at stage '{stage.name}': {error_msg}\n"
                    f"Resume with: PipelineRunner().run(config, resume_from='{stage.name}')\n"
                    f"Recovery suggestions:\n{hint_text}"
                ) from exc

        # Step 8: Save final results and artifacts
        pipeline_duration = time.time() - pipeline_start
        self._save_results(config, context, statuses, pipeline_duration)
        self._display_summary(statuses, pipeline_duration)

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully in %.1f seconds", pipeline_duration)
        logger.info("=" * 60)

        return context

    # ------------------------------------------------------------------ #
    # Public API: dry_run
    # ------------------------------------------------------------------ #

    def dry_run(
        self,
        config_path_or_config: str | Path | dict[str, Any] | LLMForgeConfig,
        auto_optimize: bool = True,
    ) -> None:
        """Show what the pipeline would do without actually executing anything.

        Parameters
        ----------
        config_path_or_config : str, Path, dict, or LLMForgeConfig
            Configuration source (same as ``run``).
        auto_optimize : bool
            Whether to simulate hardware auto-optimisation.
        """
        config = self._load_config(config_path_or_config)

        if auto_optimize:
            hardware = detect_hardware()
            config = auto_optimize_config(config, hardware)

        stages = self._dag_builder.build_dag(config)

        # Translate configs for display
        translated = ConfigTranslator.translate_all(config)

        if _RICH_AVAILABLE and self._console is not None:
            self._console.print()
            self._console.print(Panel("[bold]llm-forge Pipeline Dry Run[/bold]", style="cyan"))

            # Stage table
            table = Table(title="Pipeline Stages", show_lines=True)
            table.add_column("Order", style="dim", width=6)
            table.add_column("Stage", style="bold")
            table.add_column("Enabled", width=8)
            table.add_column("Dependencies")
            table.add_column("Description")

            for i, stage in enumerate(stages, 1):
                enabled_str = "[green]Yes[/green]" if stage.enabled else "[dim]No[/dim]"
                deps = ", ".join(stage.dependencies) if stage.dependencies else "-"
                table.add_row(str(i), stage.name, enabled_str, deps, stage.description)

            self._console.print(table)

            # Config summary
            self._console.print()
            self._console.print("[bold]Model:[/bold]", config.model.name)
            self._console.print("[bold]Training mode:[/bold]", config.training.mode)
            self._console.print("[bold]Data:[/bold]", config.data.train_path)
            self._console.print("[bold]Output:[/bold]", config.training.output_dir)

            if translated.get("bnb_config"):
                self._console.print()
                self._console.print(
                    "[bold]Quantization:[/bold]",
                    json.dumps(translated["bnb_config"], indent=2),
                )

            if translated.get("lora_config"):
                self._console.print()
                self._console.print(
                    "[bold]LoRA config:[/bold]",
                    json.dumps(translated["lora_config"], indent=2),
                )

            if translated.get("deepspeed_config"):
                self._console.print()
                self._console.print(
                    f"[bold]DeepSpeed (ZeRO stage {config.distributed.deepspeed_stage}):[/bold]"
                )
                # Just show key settings, not the full config
                ds = translated["deepspeed_config"]
                self._console.print(
                    f"  Batch size: {ds.get('train_batch_size')}, "
                    f"Micro BS: {ds.get('train_micro_batch_size_per_gpu')}, "
                    f"Grad accum: {ds.get('gradient_accumulation_steps')}"
                )

            self._console.print()
        else:
            # Plain text fallback
            logger.info("=== Pipeline Dry Run ===")
            for i, stage in enumerate(stages, 1):
                status = "ENABLED" if stage.enabled else "DISABLED"
                deps = ", ".join(stage.dependencies) if stage.dependencies else "-"
                logger.info(
                    "  [%d] %-16s  %s  deps=(%s)  %s",
                    i,
                    stage.name,
                    status,
                    deps,
                    stage.description,
                )
            logger.info("Model: %s", config.model.name)
            logger.info("Mode: %s", config.training.mode)
            logger.info("Data: %s", config.data.train_path)
            logger.info("Output: %s", config.training.output_dir)

    # ------------------------------------------------------------------ #
    # Config loading
    # ------------------------------------------------------------------ #

    def _load_config(
        self,
        source: str | Path | dict[str, Any] | LLMForgeConfig,
    ) -> LLMForgeConfig:
        """Load and validate configuration from various sources."""
        # Already validated
        if isinstance(source, LLMForgeConfig):
            return source

        # File path
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.is_file():
                return validate_config_file(path)
            # Maybe it is a preset name
            try:
                preset_cfg = self._preset_resolver.resolve(str(source))
                return validate_config_dict(preset_cfg)
            except FileNotFoundError:
                pass
            raise FileNotFoundError(
                f"Configuration source not found: {source}. "
                "Provide a valid YAML file path or a preset name."
            )

        # Dict
        if isinstance(source, dict):
            # Check for preset + overrides pattern
            preset_name = source.pop("preset", None)
            if preset_name is not None:
                preset_cfg = self._preset_resolver.resolve(preset_name)
                merged = self._preset_resolver.merge_with_overrides(preset_cfg, source)
                return validate_config_dict(merged)

            return validate_config_dict(source)

        raise TypeError(
            f"Unsupported config source type: {type(source).__name__}. "
            "Expected str, Path, dict, or LLMForgeConfig."
        )

    # ------------------------------------------------------------------ #
    # Progress display
    # ------------------------------------------------------------------ #

    def _display_pipeline_plan(
        self,
        stages: list[PipelineStage],
        statuses: dict[str, _StageStatus],
        skip_stages: set,
    ) -> None:
        """Show the pipeline execution plan."""
        if _RICH_AVAILABLE and self._console is not None:
            self._console.print()
            table = Table(title="Pipeline Execution Plan", show_lines=True)
            table.add_column("Stage", style="bold")
            table.add_column("Status", width=12)
            table.add_column("Description")

            for stage in stages:
                if not stage.enabled:
                    status_str = "[dim]disabled[/dim]"
                elif stage.name in skip_stages:
                    status_str = "[yellow]skip[/yellow]"
                else:
                    status_str = "[cyan]pending[/cyan]"
                table.add_row(stage.name, status_str, stage.description)

            self._console.print(table)
            self._console.print()

    def _display_stage_start(
        self,
        stage: PipelineStage,
        statuses: dict[str, _StageStatus],
    ) -> None:
        """Display stage start indicator."""
        if _RICH_AVAILABLE and self._console is not None:
            self._console.print(
                f"\n[bold cyan]>>> Stage: {stage.name}[/bold cyan] - {stage.description}"
            )

    def _display_summary(
        self,
        statuses: dict[str, _StageStatus],
        total_duration: float,
    ) -> None:
        """Display final pipeline execution summary."""
        if _RICH_AVAILABLE and self._console is not None:
            self._console.print()
            table = Table(title="Pipeline Summary", show_lines=True)
            table.add_column("Stage", style="bold")
            table.add_column("Status", width=12)
            table.add_column("Duration", width=12)
            table.add_column("Notes")

            status_styles = {
                _STATUS_COMPLETED: "[green]completed[/green]",
                _STATUS_FAILED: "[red]FAILED[/red]",
                _STATUS_SKIPPED: "[yellow]skipped[/yellow]",
                _STATUS_PENDING: "[dim]pending[/dim]",
                _STATUS_RUNNING: "[cyan]running[/cyan]",
            }

            for status in statuses.values():
                style = status_styles.get(status.status, status.status)
                duration_str = (
                    f"{status.duration_seconds:.1f}s" if status.duration_seconds > 0 else "-"
                )
                notes = status.error or ""
                table.add_row(status.name, style, duration_str, notes)

            self._console.print(table)
            self._console.print(f"\n[bold]Total time:[/bold] {total_duration:.1f} seconds")
        else:
            logger.info("=== Pipeline Summary ===")
            for status in statuses.values():
                duration_str = (
                    f"{status.duration_seconds:.1f}s" if status.duration_seconds > 0 else "-"
                )
                logger.info(
                    "  %-16s  %-10s  %s",
                    status.name,
                    status.status,
                    duration_str,
                )
            logger.info("Total time: %.1f seconds", total_duration)

    # ------------------------------------------------------------------ #
    # Checkpointing and results persistence
    # ------------------------------------------------------------------ #

    def _save_checkpoint(
        self,
        config: LLMForgeConfig,
        context: dict[str, Any],
        statuses: dict[str, _StageStatus],
        failed_stage: str,
    ) -> None:
        """Save a checkpoint to enable resuming from the failed stage."""
        output_dir = Path(config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "failed_stage": failed_stage,
            "stage_statuses": {
                name: {
                    "status": s.status,
                    "duration_seconds": s.duration_seconds,
                    "error": s.error,
                }
                for name, s in statuses.items()
            },
            "resume_hint": f"PipelineRunner().run(config, resume_from='{failed_stage}')",
        }

        checkpoint_path = output_dir / "pipeline_checkpoint.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info("Pipeline checkpoint saved: %s", checkpoint_path)

    def _save_results(
        self,
        config: LLMForgeConfig,
        context: dict[str, Any],
        statuses: dict[str, _StageStatus],
        total_duration: float,
    ) -> None:
        """Save final pipeline results and metadata."""
        output_dir = Path(config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "total_duration_seconds": round(total_duration, 2),
            "stages": {},
        }

        for name, status in statuses.items():
            results["stages"][name] = {
                "status": status.status,
                "duration_seconds": round(status.duration_seconds, 2),
            }
            if status.error:
                results["stages"][name]["error"] = status.error

        # Add key outputs
        if "merged_model_path" in context:
            results["merged_model_path"] = context["merged_model_path"]
        if "export_path" in context:
            results["export_path"] = context["export_path"]
        if "eval_results" in context:
            eval_results = context["eval_results"]
            # Make sure eval results are serialisable
            if isinstance(eval_results, dict):
                results["eval_results"] = eval_results
            else:
                results["eval_results"] = str(eval_results)
        if "eval_report_path" in context:
            results["eval_report_path"] = context["eval_report_path"]

        # Add training stats if available
        if "train_result" in context:
            train_result = context["train_result"]
            if hasattr(train_result, "metrics"):
                results["training_metrics"] = train_result.metrics
            elif isinstance(train_result, dict):
                results["training_metrics"] = train_result

        # Add dataset sizes
        if "train_dataset" in context:
            with contextlib.suppress(TypeError, AttributeError):
                results["train_samples"] = len(context["train_dataset"])
        if "eval_dataset" in context:
            with contextlib.suppress(TypeError, AttributeError):
                results["eval_samples"] = len(context["eval_dataset"])

        results_path = output_dir / "pipeline_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Pipeline results saved: %s", results_path)

        # Remove checkpoint file if pipeline succeeded
        checkpoint_path = output_dir / "pipeline_checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
