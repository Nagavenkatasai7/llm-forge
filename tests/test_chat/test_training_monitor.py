"""Tests for the background training monitor."""

from __future__ import annotations

import json
import time
from pathlib import Path

from llm_forge.chat.training_monitor import TrainingMonitor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_trainer_state(output_dir: Path, state: dict) -> Path:
    """Write a trainer_state.json into *output_dir* and return the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "trainer_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path


def _make_trainer_state(
    global_step: int = 50,
    max_steps: int = 100,
    log_entries: int = 5,
    base_loss: float = 2.0,
) -> dict:
    """Build a realistic trainer_state.json payload."""
    log_history = []
    for i in range(1, log_entries + 1):
        step = global_step // log_entries * i
        log_history.append(
            {
                "epoch": round(step / max_steps, 4),
                "loss": round(base_loss - 0.05 * i, 4),
                "learning_rate": 5e-5 * (1 - step / max_steps),
                "step": step,
            }
        )
    return {
        "global_step": global_step,
        "max_steps": max_steps,
        "log_history": log_history,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainingMonitorInit:
    """TrainingMonitor construction and defaults."""

    def test_monitor_init(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(str(tmp_path))
        assert monitor._output_dir == tmp_path
        assert monitor._poll_interval == 5.0
        assert not monitor.is_training_active()

    def test_monitor_custom_poll_interval(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(str(tmp_path), poll_interval=1.0)
        assert monitor._poll_interval == 1.0


class TestGetStatusNoFile:
    """Behaviour when trainer_state.json does not exist."""

    def test_get_status_no_file(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(str(tmp_path))
        status = monitor.get_status()
        # Before starting the thread, it should report idle
        assert status["status"] == "idle"

    def test_get_status_no_file_after_poll(self, tmp_path: Path) -> None:
        """After one manual poll cycle the status should be 'waiting'."""
        monitor = TrainingMonitor(str(tmp_path))
        # Simulate a poll without starting the background thread
        monitor._update_status()
        status = monitor.get_status()
        assert status["status"] == "waiting"
        assert "trainer_state.json" in status["message"]


class TestGetStatusWithFile:
    """Parse a real trainer_state.json and verify metrics."""

    def test_get_status_with_file(self, tmp_path: Path) -> None:
        state = _make_trainer_state(global_step=50, max_steps=100)
        _write_trainer_state(tmp_path, state)

        monitor = TrainingMonitor(str(tmp_path))
        monitor._update_status()
        status = monitor.get_status()

        assert status["status"] == "training"
        assert status["step"] == 50
        assert status["total_steps"] == 100
        assert status["progress_pct"] == 50.0
        assert isinstance(status["loss"], float)
        assert status["loss"] < 2.0  # loss decreases from base_loss=2.0
        assert status["learning_rate"] is not None

    def test_get_status_from_checkpoint_dir(self, tmp_path: Path) -> None:
        """Falls back to checkpoint-N/ subdirectory if no top-level file."""
        ckpt = tmp_path / "checkpoint-100"
        state = _make_trainer_state(global_step=100, max_steps=200)
        _write_trainer_state(ckpt, state)

        monitor = TrainingMonitor(str(tmp_path))
        monitor._update_status()
        status = monitor.get_status()

        assert status["status"] == "training"
        assert status["step"] == 100

    def test_epoch_and_log_entries(self, tmp_path: Path) -> None:
        state = _make_trainer_state(global_step=80, max_steps=100, log_entries=8)
        _write_trainer_state(tmp_path, state)

        monitor = TrainingMonitor(str(tmp_path))
        monitor._update_status()
        status = monitor.get_status()

        assert status["epoch"] is not None
        assert status["log_entries"] == 8


class TestETACalculation:
    """Verify ETA calculation from step rate."""

    def test_eta_calculation(self, tmp_path: Path) -> None:
        # First poll at step 10
        state1 = _make_trainer_state(global_step=10, max_steps=100, log_entries=1)
        _write_trainer_state(tmp_path, state1)

        monitor = TrainingMonitor(str(tmp_path))
        monitor._update_status()
        s1 = monitor.get_status()
        # After only the first reading, there is no second data point yet, so no ETA
        assert "eta_seconds" not in s1

        # Simulate time passing, then poll at step 50
        # We manually set the first_step_time to 10 seconds ago
        monitor._first_step = 10
        monitor._first_step_time = time.monotonic() - 10.0

        state2 = _make_trainer_state(global_step=50, max_steps=100, log_entries=5)
        _write_trainer_state(tmp_path, state2)
        monitor._update_status()

        s2 = monitor.get_status()
        assert "eta_seconds" in s2
        assert s2["eta_seconds"] > 0
        # 40 steps in 10 seconds = 0.25 s/step, 50 remaining => ~12.5s
        assert s2["eta_seconds"] < 20  # rough sanity check
        assert "eta_display" in s2

    def test_eta_display_format_minutes(self, tmp_path: Path) -> None:
        """ETA display should show minutes when >= 60s."""
        state = _make_trainer_state(global_step=10, max_steps=1000, log_entries=1)
        _write_trainer_state(tmp_path, state)

        monitor = TrainingMonitor(str(tmp_path))
        # Pretend we saw step 5 two seconds ago (fast rate)
        monitor._first_step = 5
        monitor._first_step_time = time.monotonic() - 2.0

        monitor._update_status()
        status = monitor.get_status()
        # 5 steps in 2s = 0.4 s/step; 990 remaining => 396s (~6m 36s)
        assert "eta_display" in status
        assert "m" in status["eta_display"]


class TestStartStopLifecycle:
    """start() and stop() work without errors."""

    def test_start_stop_lifecycle(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(str(tmp_path), poll_interval=0.1)

        monitor.start()
        assert monitor.is_training_active()

        # Let one poll cycle complete
        time.sleep(0.3)

        monitor.stop()
        assert not monitor.is_training_active()

    def test_double_start_is_safe(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(str(tmp_path), poll_interval=0.1)
        monitor.start()
        monitor.start()  # should be a no-op
        assert monitor.is_training_active()
        monitor.stop()

    def test_stop_without_start_is_safe(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(str(tmp_path))
        monitor.stop()  # should not raise

    def test_background_thread_picks_up_new_file(self, tmp_path: Path) -> None:
        """Monitor running in background should reflect a newly written file."""
        monitor = TrainingMonitor(str(tmp_path), poll_interval=0.1)
        monitor.start()
        time.sleep(0.15)

        # Initially no file
        s1 = monitor.get_status()
        assert s1["status"] in ("idle", "waiting")

        # Write a trainer_state.json
        state = _make_trainer_state(global_step=30, max_steps=60)
        _write_trainer_state(tmp_path, state)

        # Wait for the monitor to pick it up
        time.sleep(0.3)
        s2 = monitor.get_status()
        assert s2["status"] == "training"
        assert s2["step"] == 30

        monitor.stop()


class TestEdgeCases:
    """Robustness against malformed or partial data."""

    def test_empty_log_history(self, tmp_path: Path) -> None:
        state = {"global_step": 0, "max_steps": 100, "log_history": []}
        _write_trainer_state(tmp_path, state)

        monitor = TrainingMonitor(str(tmp_path))
        monitor._update_status()
        status = monitor.get_status()

        assert status["status"] == "training"
        assert status["step"] == 0
        assert status["loss"] is None

    def test_only_eval_entries_in_log(self, tmp_path: Path) -> None:
        """If log_history has only eval entries (no 'loss' key), loss should be None."""
        state = {
            "global_step": 50,
            "max_steps": 100,
            "log_history": [
                {"eval_loss": 1.5, "step": 50, "epoch": 0.5},
            ],
        }
        _write_trainer_state(tmp_path, state)

        monitor = TrainingMonitor(str(tmp_path))
        monitor._update_status()
        status = monitor.get_status()

        assert status["loss"] is None
        assert status["step"] == 50

    def test_corrupt_json_skipped(self, tmp_path: Path) -> None:
        """A truncated/corrupt JSON file should not crash the monitor."""
        state_path = tmp_path / "trainer_state.json"
        state_path.write_text("{invalid json...", encoding="utf-8")

        monitor = TrainingMonitor(str(tmp_path))
        monitor._update_status()
        # Should remain in idle state (not crash)
        status = monitor.get_status()
        assert status["status"] == "idle"
