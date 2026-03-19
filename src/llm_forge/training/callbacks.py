"""Training callbacks for the llm-forge platform.

Provides pluggable callbacks compatible with the HuggingFace Trainer callback
system.  Includes GPU monitoring, Rich progress bars, checkpoint management,
early stopping, and Weights & Biases integration.
"""

from __future__ import annotations

import platform
import threading
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from llm_forge.utils.logging import get_logger

logger = get_logger("training.callbacks")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ============================================================================
# StopTrainingCallback — for UI-driven graceful cancellation
# ============================================================================


class StopTrainingCallback(TrainerCallback):
    """Check a ``threading.Event`` on each training step and stop if set.

    Used by the Gradio dashboard to allow users to cancel training
    gracefully. The training loop will finish the current step and then
    exit cleanly.

    Parameters
    ----------
    stop_event : threading.Event
        When this event is set, training will stop after the current step.
    """

    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event: threading.Event = stop_event

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self.stop_event.is_set():
            control.should_training_stop = True
            logger.info("Training stop requested by user via UI (step %d)", state.global_step)


# ============================================================================
# WandBCallback
# ============================================================================


class WandBCallback(TrainerCallback):
    """Log training metrics, model architecture info, and system stats to
    Weights & Biases.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str or None
        Run display name.  ``None`` lets W&B auto-generate.
    tags : list[str] or None
        Tags to attach to the run.
    log_model : bool
        Whether to log the final model as a W&B artifact.
    """

    def __init__(
        self,
        project: str = "llm-forge",
        run_name: str | None = None,
        tags: list[str] | None = None,
        log_model: bool = False,
    ) -> None:
        super().__init__()
        if not _WANDB_AVAILABLE:
            raise ImportError(
                "wandb is required for WandBCallback. Install with: pip install wandb"
            )
        self.project = project
        self.run_name = run_name
        self.tags = tags or []
        self.log_model = log_model
        self._run: Any | None = None

    # -- lifecycle -----------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Initialise the W&B run at training start."""
        if _wandb.run is None:
            self._run = _wandb.init(
                project=self.project,
                name=self.run_name,
                tags=self.tags,
                config=args.to_dict(),
                reinit=True,
            )
        else:
            self._run = _wandb.run

        # Log model architecture summary if available
        model = kwargs.get("model")
        if model is not None and hasattr(model, "num_parameters"):
            _wandb.config.update(
                {
                    "total_params": model.num_parameters(),
                    "trainable_params": model.num_parameters(only_trainable=True),
                },
                allow_val_change=True,
            )
        logger.info("WandB run initialised: %s/%s", self.project, self.run_name)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Forward every logged metric dict to W&B."""
        if logs is None:
            return
        step = state.global_step
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if metrics:
            _wandb.log(metrics, step=step)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log evaluation metrics to W&B."""
        if metrics is None:
            return
        eval_metrics = {
            f"eval/{k}" if not k.startswith("eval_") else k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if eval_metrics:
            _wandb.log(eval_metrics, step=state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Finalise the W&B run."""
        if self.log_model and args.output_dir:
            artifact = _wandb.Artifact(
                name=f"model-{_wandb.run.id}" if _wandb.run else "model",
                type="model",
            )
            artifact.add_dir(args.output_dir)
            _wandb.log_artifact(artifact)
            logger.info("Model artifact logged to WandB")

        if self._run is not None:
            self._run.finish()
            logger.info("WandB run finished")


# ============================================================================
# CheckpointCallback
# ============================================================================


class CheckpointCallback(TrainerCallback):
    """Save additional checkpoints at a configurable step interval.

    The built-in HF Trainer checkpointing is step-based; this callback
    adds the ability to save based on wall-clock time or epoch fraction.

    Parameters
    ----------
    save_every_n_minutes : float
        Save a checkpoint every *N* minutes (0 = disabled).
    checkpoint_dir : str or None
        Override directory for checkpoints.  Defaults to ``args.output_dir``.
    max_checkpoints : int
        Maximum number of timed checkpoints to retain.
    """

    def __init__(
        self,
        save_every_n_minutes: float = 30.0,
        checkpoint_dir: str | None = None,
        max_checkpoints: int = 5,
    ) -> None:
        super().__init__()
        self.save_every_n_minutes = save_every_n_minutes
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self._last_save_time: float = 0.0
        self._saved_dirs: list[Path] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._last_save_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self.save_every_n_minutes <= 0:
            return

        elapsed_minutes = (time.time() - self._last_save_time) / 60.0
        if elapsed_minutes >= self.save_every_n_minutes:
            control.should_save = True
            self._last_save_time = time.time()

            # Prune old timed checkpoints
            base = Path(self.checkpoint_dir or args.output_dir)
            ckpt_dir = base / f"checkpoint-timed-{state.global_step}"
            self._saved_dirs.append(ckpt_dir)
            while len(self._saved_dirs) > self.max_checkpoints:
                old = self._saved_dirs.pop(0)
                if old.exists():
                    import shutil

                    shutil.rmtree(old, ignore_errors=True)
                    logger.debug("Pruned old timed checkpoint: %s", old)


# ============================================================================
# EarlyStoppingCallback
# ============================================================================


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when the evaluation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of evaluation rounds with no improvement before stopping.
    min_delta : float
        Minimum decrease in eval loss to qualify as an improvement.
    metric_name : str
        Name of the metric to monitor (from eval logs).
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        metric_name: str = "eval_loss",
    ) -> None:
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self._best_metric: float = float("inf")
        self._wait: int = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        current = metrics.get(self.metric_name)
        if current is None:
            return

        if current < self._best_metric - self.min_delta:
            self._best_metric = current
            self._wait = 0
            logger.debug("Early stopping: %s improved to %.6f", self.metric_name, current)
        else:
            self._wait += 1
            logger.info(
                "Early stopping: %s did not improve. Patience %d/%d",
                self.metric_name,
                self._wait,
                self.patience,
            )
            if self._wait >= self.patience:
                logger.info(
                    "Early stopping triggered after %d evaluations with no improvement.",
                    self.patience,
                )
                control.should_training_stop = True


# ============================================================================
# GPUMonitorCallback
# ============================================================================


class GPUMonitorCallback(TrainerCallback):
    """Log GPU memory utilisation and temperature during training.

    Metrics are emitted every ``log_every_n_steps`` optimiser steps and
    forwarded through the standard Trainer logging pipeline.

    Parameters
    ----------
    log_every_n_steps : int
        Frequency (in optimiser steps) to sample GPU stats.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not _TORCH_AVAILABLE:
            return
        if not torch.cuda.is_available():
            return
        if state.global_step % self.log_every_n_steps != 0:
            return

        gpu_metrics: dict[str, float] = {}
        for i in range(torch.cuda.device_count()):
            allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(i) / (1024**3)
            max_allocated_gb = torch.cuda.max_memory_allocated(i) / (1024**3)
            utilization_pct = (allocated_gb / reserved_gb * 100.0) if reserved_gb > 0 else 0.0

            gpu_metrics[f"gpu_{i}/memory_allocated_gb"] = round(allocated_gb, 3)
            gpu_metrics[f"gpu_{i}/memory_reserved_gb"] = round(reserved_gb, 3)
            gpu_metrics[f"gpu_{i}/memory_peak_gb"] = round(max_allocated_gb, 3)
            gpu_metrics[f"gpu_{i}/memory_utilization_pct"] = round(utilization_pct, 1)

        # Inject metrics into the trainer state for the next log event
        if gpu_metrics:
            state.log_history.append({"step": state.global_step, **gpu_metrics})
            if _WANDB_AVAILABLE and _wandb.run is not None:
                _wandb.log(gpu_metrics, step=state.global_step)


# ============================================================================
# RichProgressCallback
# ============================================================================


def _sparkline(values: list[float], width: int = 20) -> str:
    """Render a list of values as a Unicode sparkline string.

    Parameters
    ----------
    values : list[float]
        Numeric values to render.
    width : int
        Maximum number of characters in the sparkline.

    Returns
    -------
    str
        A string of Unicode block characters representing the values.
    """
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    # Use only the last *width* values
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi > lo else 1.0
    return "".join(blocks[min(int((v - lo) / span * 8), 8)] for v in recent)


def _loss_trend(values: list[float], window: int = 5) -> str:
    """Return a trend indicator for the loss curve.

    Parameters
    ----------
    values : list[float]
        Loss history.
    window : int
        Number of recent values to compare.

    Returns
    -------
    str
        ``"↓"`` (improving), ``"↑"`` (worsening), or ``"→"`` (stable).
    """
    if len(values) < 2:
        return "→"
    recent = values[-window:]
    if len(recent) < 2:
        return "→"
    delta = recent[-1] - recent[0]
    if abs(delta) < 1e-4:
        return "→"
    return "↓" if delta < 0 else "↑"


class RichProgressCallback(TrainerCallback):
    """Rich-based live progress bar for HuggingFace Trainer.

    Displays a progress bar with loss (including sparkline trend), learning
    rate, memory usage, batch size, and throughput information using the Rich
    library.  On macOS, also shows system health (thermal, battery).

    Parameters
    ----------
    sparkline_width : int
        Number of characters for the inline loss sparkline (default 16).
    show_memory : bool
        Whether to show system/MPS memory in the progress line.
    show_mac_status : bool
        Whether to show macOS thermal/battery status (ignored on non-Mac).
    """

    def __init__(
        self,
        sparkline_width: int = 16,
        show_memory: bool = True,
        show_mac_status: bool = True,
    ) -> None:
        super().__init__()
        if not _RICH_AVAILABLE:
            raise ImportError(
                "rich is required for RichProgressCallback. Install with: pip install rich"
            )
        self._console = Console()
        self._progress: Progress | None = None
        self._task_id: Any | None = None
        self._live: Live | None = None
        self._train_start_time: float = 0.0
        self._current_loss: float = 0.0
        self._current_lr: float = 0.0
        self._epoch: float = 0.0
        self._total_steps: int = 0
        self._batch_size: int = 0
        self._grad_accum: int = 1
        # Loss history for sparkline and trend
        self._loss_history: list[float] = []
        self._sparkline_width = sparkline_width
        self._show_memory = show_memory
        self._show_mac_status = show_mac_status and platform.system() == "Darwin"
        # Optional Mac utilities
        self._mac_utils: Any | None = None
        if self._show_mac_status:
            try:
                from llm_forge.training import mac_utils

                self._mac_utils = mac_utils
            except ImportError:
                self._show_mac_status = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._total_steps = state.max_steps
        self._train_start_time = time.time()
        self._batch_size = args.per_device_train_batch_size
        self._grad_accum = args.gradient_accumulation_steps

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[yellow]{task.fields[info]}"),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "Training",
            total=self._total_steps,
            info="starting...",
        )
        self._progress.start()
        logger.info(
            "Training started: %d steps across %d epoch(s), batch_size=%d, grad_accum=%d",
            self._total_steps,
            args.num_train_epochs,
            self._batch_size,
            self._grad_accum,
        )

    def _build_info_string(self, steps_per_sec: float) -> str:
        """Build the info string shown in the progress bar."""
        info_parts: list[str] = []

        # Loss + sparkline + trend
        if self._current_loss > 0:
            trend = _loss_trend(self._loss_history)
            spark = _sparkline(self._loss_history, self._sparkline_width)
            info_parts.append(f"loss: {self._current_loss:.4f}{trend} {spark}")

        # Learning rate
        if self._current_lr > 0:
            info_parts.append(f"lr: {self._current_lr:.2e}")

        # Epoch
        info_parts.append(f"ep: {self._epoch:.2f}")

        # Batch size
        if self._batch_size > 0:
            eff_bs = self._batch_size * self._grad_accum
            info_parts.append(f"bs: {eff_bs}")

        # Throughput
        info_parts.append(f"{steps_per_sec:.1f} steps/s")

        # System memory
        if self._show_memory:
            mem_str = self._get_memory_string()
            if mem_str:
                info_parts.append(mem_str)

        # Mac status
        if self._show_mac_status and self._mac_utils is not None:
            mac_str = self._get_mac_status_string()
            if mac_str:
                info_parts.append(mac_str)

        return " | ".join(info_parts)

    def _get_memory_string(self) -> str:
        """Return a compact memory usage string."""
        parts: list[str] = []
        # MPS memory (Apple Silicon GPU)
        if _TORCH_AVAILABLE:
            try:
                import torch as _torch

                if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                    alloc = _torch.mps.current_allocated_memory() / (1024**3)
                    parts.append(f"mps:{alloc:.1f}G")
                elif _torch.cuda.is_available():
                    alloc = _torch.cuda.memory_allocated() / (1024**3)
                    parts.append(f"gpu:{alloc:.1f}G")
            except (AttributeError, RuntimeError):
                pass
        # System RAM via psutil
        try:
            import psutil

            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            parts.append(f"ram:{used_gb:.0f}/{total_gb:.0f}G")
        except ImportError:
            pass
        return " ".join(parts)

    def _get_mac_status_string(self) -> str:
        """Return a compact macOS health string."""
        if self._mac_utils is None:
            return ""
        parts: list[str] = []
        # Thermal
        try:
            state = self._mac_utils.get_thermal_state()
            if state in ("serious", "critical"):
                parts.append(f"therm:{state}")
        except Exception:
            pass
        # Battery
        try:
            batt = self._mac_utils.get_battery_status()
            if batt.get("available") and not batt.get("plugged_in"):
                parts.append(f"bat:{batt['percent']}%")
        except Exception:
            pass
        return " ".join(parts)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._progress is None or self._task_id is None:
            return

        elapsed = time.time() - self._train_start_time
        steps_done = state.global_step
        steps_per_sec = steps_done / elapsed if elapsed > 0 else 0

        self._progress.update(
            self._task_id,
            completed=steps_done,
            info=self._build_info_string(steps_per_sec),
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        if "loss" in logs:
            self._current_loss = logs["loss"]
            self._loss_history.append(logs["loss"])
        if "learning_rate" in logs:
            self._current_lr = logs["learning_rate"]
        if "epoch" in logs:
            self._epoch = logs["epoch"]

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and self._progress and self._task_id:
            self._progress.update(
                self._task_id,
                info=f"eval_loss: {eval_loss:.4f}",
            )
        if eval_loss is not None:
            self._console.print(
                f"  [cyan]Eval step {state.global_step}[/cyan]: eval_loss={eval_loss:.4f}",
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._progress is not None:
            self._progress.update(
                self._task_id,
                completed=self._total_steps,
                info="complete",
            )
            self._progress.stop()

        elapsed = time.time() - self._train_start_time
        minutes = elapsed / 60.0

        # Build summary with sparkline
        summary_parts = [
            "[bold green]Training complete![/bold green]",
            f"Total steps: {state.global_step}",
            f"Wall time: {minutes:.1f} minutes",
            f"Final loss: {self._current_loss:.4f}",
        ]
        if self._loss_history:
            spark = _sparkline(self._loss_history, 30)
            trend = _loss_trend(self._loss_history)
            summary_parts.append(f"Loss curve: {spark} {trend}")

        self._console.print(
            Panel(
                "\n".join(summary_parts),
                title="llm-forge Training Summary",
                border_style="green",
            )
        )

    @property
    def loss_history(self) -> list[float]:
        """Return a copy of the recorded loss history."""
        return list(self._loss_history)


# ============================================================================
# MacMonitorCallback
# ============================================================================


class MacMonitorCallback(TrainerCallback):
    """Monitor macOS system health during training and auto-pause if needed.

    Checks memory pressure, thermal state, and battery level at a
    configurable step interval.  When dangerous conditions are detected
    (thermal throttling, low battery, extreme memory pressure), the callback
    can automatically pause training.

    Parameters
    ----------
    check_every_n_steps : int
        How often (in optimiser steps) to poll system health.
    memory_threshold : float
        Memory pressure fraction (0–1) above which to warn and suggest a
        smaller batch size.
    thermal_pause_seconds : int
        Seconds to sleep when thermal throttling is detected.
    min_battery_pct : int
        Minimum battery percentage before pausing (ignored when plugged in).
    """

    def __init__(
        self,
        check_every_n_steps: int = 50,
        memory_threshold: float = 0.85,
        thermal_pause_seconds: int = 30,
        min_battery_pct: int = 20,
    ) -> None:
        super().__init__()
        self.check_every_n_steps = check_every_n_steps
        self.memory_threshold = memory_threshold
        self.thermal_pause_seconds = thermal_pause_seconds
        self.min_battery_pct = min_battery_pct
        self._mac_utils: Any | None = None
        self._warnings: list[dict[str, Any]] = []
        try:
            from llm_forge.training import mac_utils

            self._mac_utils = mac_utils
        except ImportError:
            logger.debug("mac_utils not available — MacMonitorCallback is a no-op")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._mac_utils is None:
            return
        if state.global_step % self.check_every_n_steps != 0:
            return

        step = state.global_step

        # --- Memory pressure ---
        if self._mac_utils.is_memory_pressure_high(self.memory_threshold):
            info = self._mac_utils.get_memory_pressure()
            suggested = self._mac_utils.suggest_batch_size(
                args.per_device_train_batch_size,
                threshold=self.memory_threshold,
            )
            msg = (
                f"Step {step}: High memory pressure "
                f"({info['percent']:.0%} used, {info['available_gb']:.1f} GB free). "
                f"Consider batch size {args.per_device_train_batch_size} → {suggested}."
            )
            logger.warning(msg)
            self._warnings.append({"step": step, "type": "memory", "message": msg})

        # --- Thermal throttling ---
        if self._mac_utils.is_thermal_throttling():
            thermal = self._mac_utils.get_thermal_state()
            msg = (
                f"Step {step}: Thermal throttling detected ({thermal}). "
                f"Pausing for {self.thermal_pause_seconds}s to cool down."
            )
            logger.warning(msg)
            self._warnings.append({"step": step, "type": "thermal", "message": msg})
            time.sleep(self.thermal_pause_seconds)

        # --- Battery ---
        if self._mac_utils.should_pause_for_battery(self.min_battery_pct):
            batt = self._mac_utils.get_battery_status()
            msg = (
                f"Step {step}: Low battery ({batt['percent']}%, "
                f"min: {self.min_battery_pct}%). Stopping training."
            )
            logger.warning(msg)
            self._warnings.append({"step": step, "type": "battery", "message": msg})
            control.should_training_stop = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._warnings:
            logger.info(
                "MacMonitorCallback recorded %d warning(s) during training.",
                len(self._warnings),
            )

    @property
    def warnings(self) -> list[dict[str, Any]]:
        """Return a copy of all warnings recorded during training."""
        return list(self._warnings)
