"""Tests for the performance monitoring dashboard.

Covers the sparkline helper, PerformanceMonitor metric tracking,
computed properties (ETA, speed, progress), and DashboardCallback
interface.
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.utils.perf_dashboard import (
        DashboardCallback,
        PerformanceMonitor,
        sparkline,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.utils.perf_dashboard not importable",
)


# ===================================================================
# sparkline() tests
# ===================================================================


class TestSparkline:
    """Test the Unicode sparkline generator."""

    def test_empty_list(self) -> None:
        assert sparkline([]) == ""

    def test_single_value(self) -> None:
        result = sparkline([5.0])
        assert len(result) == 1

    def test_ascending_values(self) -> None:
        result = sparkline([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(result) == 5
        # First should be lowest block, last highest
        assert result[0] <= result[-1]

    def test_descending_values(self) -> None:
        result = sparkline([5.0, 4.0, 3.0, 2.0, 1.0])
        assert len(result) == 5
        assert result[0] >= result[-1]

    def test_constant_values(self) -> None:
        """Constant values → all same character."""
        result = sparkline([3.0, 3.0, 3.0])
        assert len(set(result)) == 1

    def test_width_truncation(self) -> None:
        """Long list is downsampled to fit width."""
        result = sparkline(list(range(100)), width=10)
        assert len(result) == 10

    def test_short_list_not_truncated(self) -> None:
        """Short list is not truncated."""
        result = sparkline([1.0, 2.0, 3.0], width=30)
        assert len(result) == 3

    def test_returns_string(self) -> None:
        result = sparkline([1.0, 2.0])
        assert isinstance(result, str)


# ===================================================================
# PerformanceMonitor tests
# ===================================================================


class TestPerformanceMonitor:
    """Test PerformanceMonitor metric collection."""

    def test_init_defaults(self) -> None:
        pm = PerformanceMonitor(total_steps=100)
        assert pm.total_steps == 100
        assert pm.current_step == 0
        assert pm.current_loss == 0.0
        assert pm.current_lr == 0.0

    def test_update_loss(self) -> None:
        pm = PerformanceMonitor(total_steps=100)
        pm.update(step=1, loss=2.5)
        assert pm.current_loss == 2.5
        assert len(pm.loss_history) == 1
        assert pm.loss_history[0] == 2.5

    def test_update_lr(self) -> None:
        pm = PerformanceMonitor(total_steps=100)
        pm.update(step=1, lr=1e-4)
        assert pm.current_lr == 1e-4
        assert len(pm.lr_history) == 1

    def test_update_epoch(self) -> None:
        pm = PerformanceMonitor(total_steps=100)
        pm.update(step=50, epoch=0.5)
        assert pm.current_epoch == 0.5

    def test_update_tokens_per_sec(self) -> None:
        pm = PerformanceMonitor()
        pm.update(step=1, tokens_per_sec=500.0)
        assert pm.tokens_per_sec == 500.0

    def test_step_tracking(self) -> None:
        pm = PerformanceMonitor(total_steps=100)
        pm.update(step=10)
        pm.update(step=20)
        assert pm.current_step == 20

    def test_speed_history_populated(self) -> None:
        pm = PerformanceMonitor(total_steps=100)
        pm.update(step=1)
        pm.update(step=2)
        assert len(pm.speed_history) >= 1

    def test_max_history_limit(self) -> None:
        """Deque respects max_history."""
        pm = PerformanceMonitor(total_steps=1000, max_history=10)
        for i in range(50):
            pm.update(step=i, loss=float(i))
        assert len(pm.loss_history) == 10

    def test_progress_pct(self) -> None:
        pm = PerformanceMonitor(total_steps=200)
        pm.update(step=100)
        assert pm.progress_pct == pytest.approx(50.0)

    def test_progress_pct_zero_total(self) -> None:
        pm = PerformanceMonitor(total_steps=0)
        assert pm.progress_pct == 0.0

    def test_progress_pct_capped_at_100(self) -> None:
        pm = PerformanceMonitor(total_steps=10)
        pm.current_step = 20
        assert pm.progress_pct == 100.0

    def test_elapsed_seconds(self) -> None:
        pm = PerformanceMonitor()
        assert pm.elapsed_seconds >= 0

    def test_eta_seconds_with_progress(self) -> None:
        pm = PerformanceMonitor(total_steps=100)
        pm.current_step = 50
        eta = pm.eta_seconds
        assert eta is not None
        assert eta > 0

    def test_eta_seconds_none_at_start(self) -> None:
        pm = PerformanceMonitor(total_steps=100)
        assert pm.eta_seconds is None

    def test_eta_seconds_none_zero_total(self) -> None:
        pm = PerformanceMonitor(total_steps=0)
        assert pm.eta_seconds is None

    def test_avg_speed_empty(self) -> None:
        pm = PerformanceMonitor()
        assert pm.avg_speed == 0.0

    def test_avg_speed_with_data(self) -> None:
        pm = PerformanceMonitor()
        pm.speed_history.append(10.0)
        pm.speed_history.append(20.0)
        assert pm.avg_speed == 15.0

    def test_none_values_ignored(self) -> None:
        """None values don't overwrite existing metrics."""
        pm = PerformanceMonitor()
        pm.update(step=1, loss=2.5, lr=1e-4)
        pm.update(step=2, loss=None, lr=None)
        assert pm.current_loss == 2.5
        assert pm.current_lr == 1e-4


# ===================================================================
# DashboardCallback tests
# ===================================================================


class TestDashboardCallback:
    """Test the HuggingFace Trainer callback interface."""

    def test_init_defaults(self) -> None:
        cb = DashboardCallback()
        assert cb.refresh_rate == 2.0
        assert cb.monitor is not None
        assert cb._live is None

    def test_init_custom_refresh(self) -> None:
        cb = DashboardCallback(refresh_rate=5.0)
        assert cb.refresh_rate == 5.0

    def test_monitor_initial_state(self) -> None:
        cb = DashboardCallback()
        assert cb.monitor.total_steps == 0
        assert cb.monitor.current_step == 0

    def test_has_callback_methods(self) -> None:
        """Verify callback has required HF Trainer interface methods."""
        cb = DashboardCallback()
        assert hasattr(cb, "on_train_begin")
        assert hasattr(cb, "on_log")
        assert hasattr(cb, "on_train_end")
        assert callable(cb.on_train_begin)
        assert callable(cb.on_log)
        assert callable(cb.on_train_end)
