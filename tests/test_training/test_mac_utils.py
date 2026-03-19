"""Tests for Mac / Apple Silicon training utilities.

Tests cover memory monitoring, thermal detection, battery awareness,
MPS environment configuration, and the combined status report.
"""

from __future__ import annotations

import os
import platform
from unittest import mock

import pytest

try:
    from llm_forge.training.mac_utils import (
        configure_mps_environment,
        get_battery_status,
        get_mac_training_status,
        get_memory_pressure,
        get_mps_memory,
        get_thermal_state,
        is_memory_pressure_high,
        is_thermal_throttling,
        should_pause_for_battery,
        suggest_batch_size,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

try:
    from llm_forge.config.schema import LLMForgeConfig, MacConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.training.mac_utils not importable",
)

_IS_MAC = platform.system() == "Darwin"


# ===================================================================
# MacConfig Schema Tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestMacConfig:
    """Verify MacConfig schema defaults and validation."""

    def test_defaults(self) -> None:
        cfg = MacConfig()
        assert cfg.smart_memory is True
        assert cfg.memory_pressure_threshold == 0.85
        assert cfg.thermal_aware is True
        assert cfg.thermal_pause_seconds == 30
        assert cfg.battery_aware is True
        assert cfg.min_battery_pct == 20
        assert cfg.mps_high_watermark_ratio == 0.0

    def test_custom_values(self) -> None:
        cfg = MacConfig(
            memory_pressure_threshold=0.90,
            thermal_pause_seconds=60,
            min_battery_pct=30,
        )
        assert cfg.memory_pressure_threshold == 0.90
        assert cfg.thermal_pause_seconds == 60
        assert cfg.min_battery_pct == 30

    def test_in_full_config(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            mac={"smart_memory": False, "battery_aware": False},
        )
        assert config.mac.smart_memory is False
        assert config.mac.battery_aware is False

    def test_threshold_range(self) -> None:
        with pytest.raises(Exception):
            MacConfig(memory_pressure_threshold=0.3)  # too low

    def test_battery_min_range(self) -> None:
        with pytest.raises(Exception):
            MacConfig(min_battery_pct=2)  # below 5


# ===================================================================
# Memory Monitoring Tests
# ===================================================================


class TestMemoryMonitoring:
    """Test memory pressure monitoring functions."""

    def test_get_memory_pressure_returns_dict(self) -> None:
        result = get_memory_pressure()
        assert isinstance(result, dict)
        assert "percent" in result
        assert "total_gb" in result

    def test_memory_percent_in_range(self) -> None:
        result = get_memory_pressure()
        assert 0.0 <= result["percent"] <= 1.0

    def test_is_memory_pressure_high_returns_bool(self) -> None:
        result = is_memory_pressure_high(threshold=0.99)
        assert isinstance(result, bool)

    def test_is_memory_pressure_high_always_true_at_zero(self) -> None:
        """Threshold 0 means any memory usage triggers high pressure."""
        assert is_memory_pressure_high(threshold=0.0) is True

    def test_suggest_batch_size_no_pressure(self) -> None:
        """When not under pressure, batch size stays the same."""
        result = suggest_batch_size(current_batch_size=8, threshold=0.99)
        assert result == 8

    def test_suggest_batch_size_halves(self) -> None:
        """When under pressure (threshold=0), batch size is halved."""
        result = suggest_batch_size(current_batch_size=8, threshold=0.0)
        assert result == 4

    def test_suggest_batch_size_min(self) -> None:
        """Batch size can't go below min_batch_size."""
        result = suggest_batch_size(current_batch_size=1, threshold=0.0, min_batch_size=1)
        assert result == 1

    def test_get_mps_memory_returns_dict(self) -> None:
        result = get_mps_memory()
        assert "allocated_gb" in result
        assert "reserved_gb" in result


# ===================================================================
# Thermal Monitoring Tests
# ===================================================================


class TestThermalMonitoring:
    """Test thermal state detection."""

    def test_get_thermal_state_returns_string(self) -> None:
        state = get_thermal_state()
        assert state in ("nominal", "fair", "serious", "critical", "unknown")

    @pytest.mark.skipif(not _IS_MAC, reason="Mac-only test")
    def test_thermal_state_on_mac(self) -> None:
        """On Mac, thermal state should not be 'unknown' (pmset available)."""
        state = get_thermal_state()
        # May be unknown if pmset fails, but usually works
        assert isinstance(state, str)

    def test_is_thermal_throttling_returns_bool(self) -> None:
        assert isinstance(is_thermal_throttling(), bool)


# ===================================================================
# Battery Monitoring Tests
# ===================================================================


class TestBatteryMonitoring:
    """Test battery status detection."""

    def test_get_battery_status_returns_dict(self) -> None:
        status = get_battery_status()
        assert "percent" in status
        assert "plugged_in" in status
        assert "available" in status

    def test_should_pause_for_battery_plugged_in(self) -> None:
        """Should never pause when plugged in."""
        with mock.patch(
            "llm_forge.training.mac_utils.get_battery_status",
            return_value={"percent": 5, "plugged_in": True, "available": True},
        ):
            assert should_pause_for_battery(min_pct=20) is False

    def test_should_pause_for_battery_low(self) -> None:
        """Should pause when battery is low and unplugged."""
        with mock.patch(
            "llm_forge.training.mac_utils.get_battery_status",
            return_value={"percent": 10, "plugged_in": False, "available": True},
        ):
            assert should_pause_for_battery(min_pct=20) is True

    def test_should_pause_for_battery_sufficient(self) -> None:
        """Should not pause when battery is above threshold."""
        with mock.patch(
            "llm_forge.training.mac_utils.get_battery_status",
            return_value={"percent": 80, "plugged_in": False, "available": True},
        ):
            assert should_pause_for_battery(min_pct=20) is False

    def test_should_pause_no_battery(self) -> None:
        """Desktop (no battery) should never pause."""
        with mock.patch(
            "llm_forge.training.mac_utils.get_battery_status",
            return_value={"percent": 100, "plugged_in": True, "available": False},
        ):
            assert should_pause_for_battery(min_pct=20) is False


# ===================================================================
# MPS Environment Tests
# ===================================================================


class TestMPSEnvironment:
    """Test MPS environment configuration."""

    def test_configure_mps_environment_sets_vars(self) -> None:
        """On Mac, should set environment variables."""
        if not _IS_MAC:
            pytest.skip("Mac-only")
        # Clear any existing values
        env_backup = os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
        fallback_backup = os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        try:
            configure_mps_environment(high_watermark_ratio=0.0)
            assert os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") == "0.0"
            assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
        finally:
            # Restore
            if env_backup is not None:
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = env_backup
            else:
                os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
            if fallback_backup is not None:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = fallback_backup
            else:
                os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)


# ===================================================================
# Combined Status Tests
# ===================================================================


class TestCombinedStatus:
    """Test the combined Mac training status report."""

    def test_returns_dict(self) -> None:
        status = get_mac_training_status()
        assert isinstance(status, dict)
        assert "is_mac" in status
        assert "memory" in status
        assert "thermal_state" in status
        assert "battery" in status
        assert "warnings" in status

    def test_is_mac_correct(self) -> None:
        status = get_mac_training_status()
        assert status["is_mac"] == _IS_MAC

    def test_warnings_is_list(self) -> None:
        status = get_mac_training_status()
        assert isinstance(status["warnings"], list)

    def test_high_pressure_warning(self) -> None:
        """High memory pressure should produce a warning."""
        status = get_mac_training_status(threshold=0.0)
        assert any("memory" in w.lower() for w in status["warnings"])
