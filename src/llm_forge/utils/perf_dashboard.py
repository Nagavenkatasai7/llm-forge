"""Terminal-based performance monitoring dashboard for training.

Uses Rich Live display to show real-time training metrics:
- GPU/CPU memory utilisation
- Training speed (steps/sec, tokens/sec)
- Loss curve (sparkline)
- ETA and progress
- Thermal state (Mac)

Can be used as a HuggingFace Trainer callback or standalone.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque
from typing import Any

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sparkline helper
# ---------------------------------------------------------------------------

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 30) -> str:
    """Generate a Unicode sparkline string from a list of values."""
    if not values:
        return ""
    if len(values) > width:
        # Downsample
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    chars = []
    for v in values:
        idx = int((v - mn) / rng * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


# ---------------------------------------------------------------------------
# PerformanceMonitor (data collection)
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """Collects and tracks performance metrics during training."""

    def __init__(self, total_steps: int = 0, max_history: int = 200) -> None:
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.last_step_time = self.start_time

        # Metric history (deques with max length)
        self.loss_history: deque = deque(maxlen=max_history)
        self.lr_history: deque = deque(maxlen=max_history)
        self.speed_history: deque = deque(maxlen=max_history)  # steps/sec

        # Current metrics
        self.current_loss: float = 0.0
        self.current_lr: float = 0.0
        self.current_epoch: float = 0.0
        self.gpu_memory_used_gb: float = 0.0
        self.gpu_memory_total_gb: float = 0.0
        self.ram_used_gb: float = 0.0
        self.ram_total_gb: float = 0.0
        self.gpu_utilisation_pct: float = 0.0
        self.thermal_state: str = ""
        self.tokens_per_sec: float = 0.0
        self.batch_size: int = 0

    def update(
        self,
        step: int,
        loss: float | None = None,
        lr: float | None = None,
        epoch: float | None = None,
        tokens_per_sec: float | None = None,
    ) -> None:
        """Record metrics for a training step."""
        now = time.time()
        step_duration = now - self.last_step_time
        self.last_step_time = now

        self.current_step = step

        if loss is not None:
            self.current_loss = loss
            self.loss_history.append(loss)

        if lr is not None:
            self.current_lr = lr
            self.lr_history.append(lr)

        if epoch is not None:
            self.current_epoch = epoch

        if tokens_per_sec is not None:
            self.tokens_per_sec = tokens_per_sec

        # Steps per second
        if step_duration > 0:
            steps_per_sec = 1.0 / step_duration
            self.speed_history.append(steps_per_sec)

        # Collect system metrics
        self._collect_system_metrics()

    def _collect_system_metrics(self) -> None:
        """Gather GPU/RAM/thermal metrics."""
        try:
            import torch

            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info(0)
                self.gpu_memory_total_gb = mem[1] / (1024**3)
                self.gpu_memory_used_gb = (mem[1] - mem[0]) / (1024**3)
                with contextlib.suppress(Exception):
                    self.gpu_utilisation_pct = float(torch.cuda.utilization(0))
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                with contextlib.suppress(Exception):
                    self.gpu_memory_used_gb = torch.mps.current_allocated_memory() / (1024**3)
        except ImportError:
            pass

        try:
            import psutil

            mem = psutil.virtual_memory()
            self.ram_total_gb = mem.total / (1024**3)
            self.ram_used_gb = mem.used / (1024**3)
        except ImportError:
            pass

        # Mac thermal
        try:
            from llm_forge.training.mac_utils import get_thermal_state

            self.thermal_state = get_thermal_state()
        except ImportError:
            pass

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time remaining in seconds."""
        if self.current_step <= 0 or self.total_steps <= 0:
            return None
        elapsed = self.elapsed_seconds
        rate = self.current_step / elapsed
        remaining = self.total_steps - self.current_step
        if rate > 0:
            return remaining / rate
        return None

    @property
    def progress_pct(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, self.current_step / self.total_steps * 100)

    @property
    def avg_speed(self) -> float:
        """Average steps per second."""
        if not self.speed_history:
            return 0.0
        return sum(self.speed_history) / len(self.speed_history)


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------


def render_dashboard(monitor: PerformanceMonitor) -> Any:
    """Render a Rich-formatted dashboard panel.

    Returns a Rich renderable (Panel or Group) suitable for Rich Live display.
    """
    if not _RICH_AVAILABLE:
        return None

    # Memory bar
    gpu_pct = 0
    if monitor.gpu_memory_total_gb > 0:
        gpu_pct = int(monitor.gpu_memory_used_gb / monitor.gpu_memory_total_gb * 100)
    ram_pct = 0
    if monitor.ram_total_gb > 0:
        ram_pct = int(monitor.ram_used_gb / monitor.ram_total_gb * 100)

    def _bar(pct: int, width: int = 20) -> str:
        filled = int(pct / 100 * width)
        empty = width - filled
        color = "green" if pct < 70 else "yellow" if pct < 90 else "red"
        return f"[{color}]{'█' * filled}{'░' * empty}[/{color}] {pct}%"

    # Build metrics table
    table = Table(show_header=False, expand=True, box=None, padding=(0, 1))
    table.add_column("Label", style="bold", width=16)
    table.add_column("Value", width=40)

    # Progress
    table.add_row(
        "Progress",
        f"Step {monitor.current_step}/{monitor.total_steps} ({monitor.progress_pct:.0f}%)",
    )
    table.add_row("Epoch", f"{monitor.current_epoch:.2f}")

    # Speed
    avg_speed = monitor.avg_speed
    table.add_row(
        "Speed",
        f"{avg_speed:.2f} steps/sec"
        + (f" | {monitor.tokens_per_sec:.0f} tok/s" if monitor.tokens_per_sec > 0 else ""),
    )

    # ETA
    eta = monitor.eta_seconds
    if eta is not None:
        mins, secs = divmod(int(eta), 60)
        hrs, mins = divmod(mins, 60)
        eta_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
        table.add_row("ETA", eta_str)

    # Loss + sparkline
    loss_spark = sparkline(list(monitor.loss_history))
    table.add_row(
        "Loss",
        f"{monitor.current_loss:.4f}  {loss_spark}",
    )

    # Learning rate
    table.add_row("LR", f"{monitor.current_lr:.2e}")

    # GPU memory
    if monitor.gpu_memory_total_gb > 0:
        table.add_row(
            "GPU Memory",
            f"{monitor.gpu_memory_used_gb:.1f}/{monitor.gpu_memory_total_gb:.1f} GB  "
            + _bar(gpu_pct, 15),
        )

    # GPU utilisation
    if monitor.gpu_utilisation_pct > 0:
        table.add_row(
            "GPU Util",
            _bar(int(monitor.gpu_utilisation_pct), 15),
        )

    # RAM
    if monitor.ram_total_gb > 0:
        table.add_row(
            "RAM",
            f"{monitor.ram_used_gb:.1f}/{monitor.ram_total_gb:.1f} GB  " + _bar(ram_pct, 15),
        )

    # Thermal state (Mac)
    if monitor.thermal_state:
        thermal_color = "green" if monitor.thermal_state in ("nominal", "fair") else "yellow"
        table.add_row(
            "Thermal",
            f"[{thermal_color}]{monitor.thermal_state}[/{thermal_color}]",
        )

    return Panel(
        table,
        title="[bold]Training Dashboard[/bold]",
        border_style="cyan",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# HuggingFace Trainer callback
# ---------------------------------------------------------------------------


class DashboardCallback:
    """HuggingFace Trainer callback that displays a live performance dashboard.

    Usage::

        from llm_forge.utils.perf_dashboard import DashboardCallback
        trainer = SFTTrainer(..., callbacks=[DashboardCallback()])
    """

    def __init__(self, refresh_rate: float = 2.0) -> None:
        self.monitor = PerformanceMonitor()
        self.refresh_rate = refresh_rate
        self._live: Any | None = None
        self._last_render: float = 0.0

    def on_train_begin(self, args: Any, state: Any, **kwargs: Any) -> None:
        """Called at the start of training."""
        self.monitor.total_steps = state.max_steps or 0
        self.monitor.start_time = time.time()
        if _RICH_AVAILABLE:
            self._live = Live(
                render_dashboard(self.monitor),
                refresh_per_second=1,
                console=Console(),
            )
            self._live.start()

    def on_log(self, args: Any, state: Any, logs: Any = None, **kwargs: Any) -> None:
        """Called on each logging step."""
        if logs is None:
            logs = {}
        self.monitor.update(
            step=state.global_step,
            loss=logs.get("loss"),
            lr=logs.get("learning_rate"),
            epoch=state.epoch,
        )
        if self._live and time.time() - self._last_render > self.refresh_rate:
            self._live.update(render_dashboard(self.monitor))
            self._last_render = time.time()

    def on_train_end(self, args: Any, state: Any, **kwargs: Any) -> None:
        """Called at the end of training."""
        if self._live:
            self._live.stop()
            self._live = None
