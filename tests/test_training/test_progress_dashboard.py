"""Tests for the training progress dashboard enhancements.

Covers sparkline rendering, loss trend detection, RichProgressCallback
enhancements (loss history, memory, Mac status), and MacMonitorCallback.
"""

from __future__ import annotations

from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import the helpers directly (they don't need Rich)
# ---------------------------------------------------------------------------

try:
    from llm_forge.training.callbacks import _loss_trend, _sparkline

    _HELPERS_AVAILABLE = True
except ImportError:
    _HELPERS_AVAILABLE = False

# For callback classes we need Rich — guard imports
try:
    from llm_forge.training.callbacks import (
        MacMonitorCallback,
        RichProgressCallback,
    )

    _CALLBACKS_AVAILABLE = True
except ImportError:
    _CALLBACKS_AVAILABLE = False


# ===================================================================
# Sparkline Rendering Tests
# ===================================================================


@pytest.mark.skipif(not _HELPERS_AVAILABLE, reason="callbacks not importable")
class TestSparkline:
    """Test the _sparkline helper function."""

    def test_empty_values(self) -> None:
        assert _sparkline([]) == ""

    def test_single_value(self) -> None:
        result = _sparkline([5.0])
        assert len(result) == 1

    def test_ascending_values(self) -> None:
        result = _sparkline([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(result) == 5
        # First char should be lowest block, last should be highest
        assert result[0] < result[-1]  # Unicode ordering

    def test_constant_values(self) -> None:
        """All same values → all same block character."""
        result = _sparkline([3.0, 3.0, 3.0, 3.0])
        assert len(set(result)) == 1  # All identical characters

    def test_width_truncation(self) -> None:
        """Values longer than width are truncated (most recent kept)."""
        values = list(range(50))
        result = _sparkline([float(v) for v in values], width=10)
        assert len(result) == 10

    def test_returns_string(self) -> None:
        result = _sparkline([1.0, 2.0, 3.0])
        assert isinstance(result, str)

    def test_unicode_block_characters(self) -> None:
        """Output uses Unicode block characters."""
        blocks = " ▁▂▃▄▅▆▇█"
        result = _sparkline([0.0, 0.5, 1.0])
        for char in result:
            assert char in blocks


# ===================================================================
# Loss Trend Tests
# ===================================================================


@pytest.mark.skipif(not _HELPERS_AVAILABLE, reason="callbacks not importable")
class TestLossTrend:
    """Test the _loss_trend helper function."""

    def test_empty_values(self) -> None:
        assert _loss_trend([]) == "→"

    def test_single_value(self) -> None:
        assert _loss_trend([1.0]) == "→"

    def test_decreasing_loss(self) -> None:
        """Decreasing loss = improving = ↓."""
        assert _loss_trend([2.0, 1.5, 1.0, 0.5]) == "↓"

    def test_increasing_loss(self) -> None:
        """Increasing loss = worsening = ↑."""
        assert _loss_trend([0.5, 1.0, 1.5, 2.0]) == "↑"

    def test_stable_loss(self) -> None:
        """Nearly constant loss = stable = →."""
        assert _loss_trend([1.0, 1.0, 1.0, 1.0]) == "→"

    def test_very_small_delta(self) -> None:
        """Delta < 1e-4 is considered stable."""
        assert _loss_trend([1.0, 1.00005]) == "→"

    def test_custom_window(self) -> None:
        """Window parameter controls how many recent values to compare."""
        # Long history with upward tail — window=2 should see the tail
        values = [5.0, 4.0, 3.0, 2.0, 1.0, 1.5, 2.0]
        assert _loss_trend(values, window=3) == "↑"


# ===================================================================
# RichProgressCallback Tests
# ===================================================================


@pytest.mark.skipif(not _CALLBACKS_AVAILABLE, reason="Rich or callbacks not importable")
class TestRichProgressCallback:
    """Test the enhanced RichProgressCallback."""

    def test_init_defaults(self) -> None:
        cb = RichProgressCallback()
        assert cb._sparkline_width == 16
        assert cb._show_memory is True
        assert cb._loss_history == []

    def test_init_custom_params(self) -> None:
        cb = RichProgressCallback(sparkline_width=10, show_memory=False, show_mac_status=False)
        assert cb._sparkline_width == 10
        assert cb._show_memory is False
        assert cb._show_mac_status is False

    def test_loss_history_property(self) -> None:
        cb = RichProgressCallback()
        cb._loss_history = [1.0, 2.0, 3.0]
        history = cb.loss_history
        assert history == [1.0, 2.0, 3.0]
        # Should be a copy
        history.append(4.0)
        assert len(cb._loss_history) == 3

    def test_on_log_records_loss(self) -> None:
        """on_log should append loss to history."""
        cb = RichProgressCallback()
        # Fake args/state/control
        args = mock.MagicMock()
        state = mock.MagicMock()
        control = mock.MagicMock()

        cb.on_log(args, state, control, logs={"loss": 2.5, "learning_rate": 1e-4, "epoch": 0.5})
        cb.on_log(args, state, control, logs={"loss": 2.0})
        cb.on_log(args, state, control, logs={"loss": 1.5})

        assert cb._loss_history == [2.5, 2.0, 1.5]
        assert cb._current_loss == 1.5
        assert cb._current_lr == 1e-4
        assert cb._epoch == 0.5

    def test_on_log_no_logs(self) -> None:
        """on_log with None should be a no-op."""
        cb = RichProgressCallback()
        args = mock.MagicMock()
        state = mock.MagicMock()
        control = mock.MagicMock()
        cb.on_log(args, state, control, logs=None)
        assert cb._loss_history == []

    def test_build_info_string_basic(self) -> None:
        """Info string contains loss, lr, epoch, batch size, throughput."""
        cb = RichProgressCallback(show_memory=False, show_mac_status=False)
        cb._current_loss = 2.5
        cb._current_lr = 1e-4
        cb._epoch = 1.0
        cb._batch_size = 4
        cb._grad_accum = 2
        cb._loss_history = [3.0, 2.5]

        info = cb._build_info_string(steps_per_sec=10.0)
        assert "loss:" in info
        assert "lr:" in info
        assert "ep:" in info
        assert "bs: 8" in info  # 4 * 2 = 8
        assert "steps/s" in info

    def test_build_info_string_with_sparkline(self) -> None:
        """Info string contains sparkline when loss history is available."""
        cb = RichProgressCallback(show_memory=False, show_mac_status=False)
        cb._current_loss = 1.0
        cb._loss_history = [3.0, 2.5, 2.0, 1.5, 1.0]
        info = cb._build_info_string(steps_per_sec=5.0)
        # Should contain trend arrow
        assert "↓" in info or "→" in info or "↑" in info


# ===================================================================
# MacMonitorCallback Tests
# ===================================================================


@pytest.mark.skipif(not _CALLBACKS_AVAILABLE, reason="callbacks not importable")
class TestMacMonitorCallback:
    """Test the MacMonitorCallback."""

    def test_init_defaults(self) -> None:
        cb = MacMonitorCallback()
        assert cb.check_every_n_steps == 50
        assert cb.memory_threshold == 0.85
        assert cb.thermal_pause_seconds == 30
        assert cb.min_battery_pct == 20
        assert cb.warnings == []

    def test_init_custom(self) -> None:
        cb = MacMonitorCallback(
            check_every_n_steps=25,
            memory_threshold=0.90,
            thermal_pause_seconds=60,
            min_battery_pct=30,
        )
        assert cb.check_every_n_steps == 25
        assert cb.memory_threshold == 0.90

    def test_no_op_without_mac_utils(self) -> None:
        """When mac_utils is None, on_step_end does nothing."""
        cb = MacMonitorCallback()
        cb._mac_utils = None

        args = mock.MagicMock()
        state = mock.MagicMock()
        state.global_step = 50
        control = mock.MagicMock()

        cb.on_step_end(args, state, control)
        assert cb.warnings == []

    def test_skips_non_check_steps(self) -> None:
        """Only checks at multiples of check_every_n_steps."""
        cb = MacMonitorCallback(check_every_n_steps=50)
        cb._mac_utils = mock.MagicMock()
        cb._mac_utils.is_memory_pressure_high.return_value = False
        cb._mac_utils.is_thermal_throttling.return_value = False
        cb._mac_utils.should_pause_for_battery.return_value = False

        args = mock.MagicMock()
        state = mock.MagicMock()
        state.global_step = 25  # Not a multiple of 50
        control = mock.MagicMock()

        cb.on_step_end(args, state, control)
        # Should not have called any monitor functions
        cb._mac_utils.is_memory_pressure_high.assert_not_called()

    def test_memory_warning(self) -> None:
        """High memory pressure triggers a warning."""
        cb = MacMonitorCallback(check_every_n_steps=10)
        cb._mac_utils = mock.MagicMock()
        cb._mac_utils.is_memory_pressure_high.return_value = True
        cb._mac_utils.get_memory_pressure.return_value = {
            "percent": 0.92,
            "available_gb": 1.2,
            "total_gb": 16.0,
            "used_gb": 14.8,
        }
        cb._mac_utils.suggest_batch_size.return_value = 2
        cb._mac_utils.is_thermal_throttling.return_value = False
        cb._mac_utils.should_pause_for_battery.return_value = False

        args = mock.MagicMock()
        args.per_device_train_batch_size = 4
        state = mock.MagicMock()
        state.global_step = 10
        control = mock.MagicMock()

        cb.on_step_end(args, state, control)
        assert len(cb.warnings) == 1
        assert cb.warnings[0]["type"] == "memory"
        assert "92%" in cb.warnings[0]["message"]

    def test_thermal_pause(self) -> None:
        """Thermal throttling triggers a pause (mocked sleep)."""
        cb = MacMonitorCallback(check_every_n_steps=10, thermal_pause_seconds=5)
        cb._mac_utils = mock.MagicMock()
        cb._mac_utils.is_memory_pressure_high.return_value = False
        cb._mac_utils.is_thermal_throttling.return_value = True
        cb._mac_utils.get_thermal_state.return_value = "serious"
        cb._mac_utils.should_pause_for_battery.return_value = False

        args = mock.MagicMock()
        state = mock.MagicMock()
        state.global_step = 10
        control = mock.MagicMock()

        with mock.patch("llm_forge.training.callbacks.time.sleep") as mock_sleep:
            cb.on_step_end(args, state, control)
            mock_sleep.assert_called_once_with(5)

        assert len(cb.warnings) == 1
        assert cb.warnings[0]["type"] == "thermal"

    def test_battery_stops_training(self) -> None:
        """Low battery sets should_training_stop = True."""
        cb = MacMonitorCallback(check_every_n_steps=10, min_battery_pct=20)
        cb._mac_utils = mock.MagicMock()
        cb._mac_utils.is_memory_pressure_high.return_value = False
        cb._mac_utils.is_thermal_throttling.return_value = False
        cb._mac_utils.should_pause_for_battery.return_value = True
        cb._mac_utils.get_battery_status.return_value = {
            "percent": 8,
            "plugged_in": False,
            "available": True,
        }

        args = mock.MagicMock()
        state = mock.MagicMock()
        state.global_step = 10
        control = mock.MagicMock()
        control.should_training_stop = False

        cb.on_step_end(args, state, control)
        assert control.should_training_stop is True
        assert len(cb.warnings) == 1
        assert cb.warnings[0]["type"] == "battery"

    def test_warnings_property_returns_copy(self) -> None:
        cb = MacMonitorCallback()
        cb._warnings = [{"step": 1, "type": "test", "message": "test"}]
        warnings = cb.warnings
        warnings.append({"step": 2, "type": "test2", "message": "test2"})
        assert len(cb._warnings) == 1  # Original unchanged

    def test_all_clear(self) -> None:
        """No warnings when system is healthy."""
        cb = MacMonitorCallback(check_every_n_steps=10)
        cb._mac_utils = mock.MagicMock()
        cb._mac_utils.is_memory_pressure_high.return_value = False
        cb._mac_utils.is_thermal_throttling.return_value = False
        cb._mac_utils.should_pause_for_battery.return_value = False

        args = mock.MagicMock()
        state = mock.MagicMock()
        state.global_step = 10
        control = mock.MagicMock()

        cb.on_step_end(args, state, control)
        assert len(cb.warnings) == 0
