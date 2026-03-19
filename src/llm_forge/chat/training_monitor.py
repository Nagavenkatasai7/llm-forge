"""Background training monitor that watches trainer_state.json for real-time progress."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class TrainingMonitor:
    """Polls trainer_state.json in a background thread to provide real-time training metrics.

    Usage::

        monitor = TrainingMonitor("/path/to/output_dir")
        monitor.start()
        status = monitor.get_status()  # {"status": "training", "step": 50, ...}
        monitor.stop()
    """

    def __init__(self, output_dir: str, poll_interval: float = 5.0) -> None:
        self._output_dir = Path(output_dir)
        self._poll_interval = poll_interval

        # Shared state protected by a lock
        self._lock = threading.Lock()
        self._status: dict = {"status": "idle", "message": "Waiting for training to start..."}

        # Background thread lifecycle
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # For ETA calculation
        self._first_step: int | None = None
        self._first_step_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="training-monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 2)
            self._thread = None

    def get_status(self) -> dict:
        """Return a snapshot of the current training metrics (thread-safe)."""
        with self._lock:
            return dict(self._status)

    def is_training_active(self) -> bool:
        """Return True if the monitor's background thread is running."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background loop: read trainer_state.json every *poll_interval* seconds."""
        while not self._stop_event.is_set():
            self._update_status()
            self._stop_event.wait(self._poll_interval)

    def _find_trainer_state(self) -> Path | None:
        """Locate the most recent trainer_state.json under output_dir.

        HuggingFace Trainer writes the file inside checkpoint-N/ dirs as well
        as at the top-level output_dir during training.
        """
        # Prefer the top-level file (written during active training)
        top_level = self._output_dir / "trainer_state.json"
        if top_level.exists():
            return top_level

        # Fall back to the latest checkpoint directory
        checkpoints = sorted(
            self._output_dir.glob("checkpoint-*/trainer_state.json"),
            key=lambda p: p.stat().st_mtime,
        )
        return checkpoints[-1] if checkpoints else None

    def _update_status(self) -> None:
        """Read trainer_state.json and update shared status dict."""
        state_path = self._find_trainer_state()
        if state_path is None:
            with self._lock:
                self._status = {
                    "status": "waiting",
                    "message": "No trainer_state.json found yet — training may still be initializing.",
                }
            return

        try:
            raw = state_path.read_text(encoding="utf-8")
            state = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            # File may be mid-write; skip this cycle
            return

        parsed = self._parse_state(state)
        with self._lock:
            self._status = parsed

    def _parse_state(self, state: dict) -> dict:
        """Extract useful metrics from the trainer state dict."""
        global_step: int = state.get("global_step", 0)
        max_steps: int = state.get("max_steps", 0)
        log_history: list[dict] = state.get("log_history", [])

        # Latest training log entry (skip eval entries that don't have "loss")
        latest: dict = {}
        for entry in reversed(log_history):
            if "loss" in entry:
                latest = entry
                break

        current_loss = latest.get("loss")
        epoch = latest.get("epoch")
        learning_rate = latest.get("learning_rate")
        step = latest.get("step", global_step)

        # ETA calculation
        eta_seconds: float | None = None
        now = time.monotonic()

        if self._first_step is None and step > 0:
            self._first_step = step
            self._first_step_time = now
        elif (
            self._first_step is not None
            and self._first_step_time is not None
            and step > self._first_step
        ):
            elapsed = now - self._first_step_time
            steps_done = step - self._first_step
            steps_remaining = max_steps - step if max_steps > 0 else 0
            if steps_done > 0 and steps_remaining > 0:
                secs_per_step = elapsed / steps_done
                eta_seconds = secs_per_step * steps_remaining

        result: dict = {
            "status": "training",
            "step": step,
            "total_steps": max_steps,
            "loss": current_loss,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "log_entries": len(log_history),
        }

        if eta_seconds is not None:
            minutes, secs = divmod(int(eta_seconds), 60)
            hours, minutes = divmod(minutes, 60)
            result["eta_seconds"] = round(eta_seconds, 1)
            if hours > 0:
                result["eta_display"] = f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                result["eta_display"] = f"{minutes}m {secs}s"
            else:
                result["eta_display"] = f"{secs}s"

        progress = 0.0
        if max_steps > 0:
            progress = round(step / max_steps * 100, 1)
        result["progress_pct"] = progress

        result["message"] = (
            f"Step {step}/{max_steps} ({progress}%) — "
            f"loss: {current_loss if current_loss is not None else '?'}"
        )

        return result
