"""Mac / Apple Silicon training utilities.

Provides memory pressure monitoring, thermal throttle detection, battery
awareness, and MPS-specific environment configuration.  All functions
degrade gracefully on non-Mac platforms (returning safe defaults).
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

try:
    from llm_forge.utils.logging import get_logger

    logger = get_logger("training.mac_utils")
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


_IS_MAC = platform.system() == "Darwin"


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------


def get_memory_pressure() -> dict[str, Any]:
    """Return current system memory usage.

    Returns
    -------
    dict
        Keys: ``total_gb``, ``used_gb``, ``available_gb``, ``percent``.
        On error, returns ``percent=0.0``.
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
            "percent": mem.percent / 100.0,
        }
    except ImportError:
        return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0.0}


def is_memory_pressure_high(threshold: float = 0.85) -> bool:
    """Check if system memory usage exceeds the threshold.

    Parameters
    ----------
    threshold : float
        Fraction of RAM (0.0–1.0) considered high pressure.

    Returns
    -------
    bool
        True if memory usage exceeds threshold.
    """
    info = get_memory_pressure()
    return info["percent"] >= threshold


def get_mps_memory() -> dict[str, float]:
    """Return MPS (Metal) GPU memory stats if available.

    Returns
    -------
    dict
        Keys: ``allocated_gb``, ``reserved_gb``.  Zeros if MPS unavailable.
    """
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            # driver_allocated is the total reserved by the MPS backend
            driver = torch.mps.driver_allocated_memory() / (1024**3)
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(driver, 2),
            }
    except (ImportError, AttributeError):
        pass
    return {"allocated_gb": 0.0, "reserved_gb": 0.0}


def suggest_batch_size(
    current_batch_size: int,
    threshold: float = 0.85,
    min_batch_size: int = 1,
) -> int:
    """Suggest a reduced batch size if memory pressure is high.

    Halves the batch size when above threshold, down to ``min_batch_size``.

    Parameters
    ----------
    current_batch_size : int
        Current per-device batch size.
    threshold : float
        Memory pressure threshold.
    min_batch_size : int
        Minimum allowed batch size.

    Returns
    -------
    int
        Suggested batch size (unchanged if no pressure).
    """
    if not is_memory_pressure_high(threshold):
        return current_batch_size
    new_bs = max(current_batch_size // 2, min_batch_size)
    if new_bs < current_batch_size:
        logger.warning(
            "Memory pressure high (>%.0f%%). Suggesting batch size %d → %d.",
            threshold * 100,
            current_batch_size,
            new_bs,
        )
    return new_bs


# ---------------------------------------------------------------------------
# Thermal monitoring
# ---------------------------------------------------------------------------


def get_thermal_state() -> str:
    """Return the macOS thermal state.

    Returns
    -------
    str
        One of ``"nominal"``, ``"fair"``, ``"serious"``, ``"critical"``,
        or ``"unknown"`` (non-Mac or detection failure).
    """
    if not _IS_MAC:
        return "unknown"
    try:
        output = subprocess.check_output(
            ["pmset", "-g", "therm"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="replace")
        # pmset -g therm outputs lines like "CPU_Speed_Limit = 100"
        for line in output.splitlines():
            if "CPU_Speed_Limit" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    speed_limit = int(parts[1].strip())
                    if speed_limit >= 95:
                        return "nominal"
                    elif speed_limit >= 75:
                        return "fair"
                    elif speed_limit >= 50:
                        return "serious"
                    else:
                        return "critical"
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass
    return "unknown"


def is_thermal_throttling() -> bool:
    """Check if the Mac is thermally throttled.

    Returns
    -------
    bool
        True if thermal state is ``"serious"`` or ``"critical"``.
    """
    state = get_thermal_state()
    return state in ("serious", "critical")


# ---------------------------------------------------------------------------
# Battery monitoring
# ---------------------------------------------------------------------------


def get_battery_status() -> dict[str, Any]:
    """Return battery information.

    Returns
    -------
    dict
        Keys: ``percent`` (0–100), ``plugged_in`` (bool),
        ``available`` (bool — False on desktops).
    """
    try:
        import psutil

        battery = psutil.sensors_battery()
        if battery is None:
            return {"percent": 100, "plugged_in": True, "available": False}
        return {
            "percent": int(battery.percent),
            "plugged_in": battery.power_plugged,
            "available": True,
        }
    except (ImportError, AttributeError):
        return {"percent": 100, "plugged_in": True, "available": False}


def should_pause_for_battery(min_pct: int = 20) -> bool:
    """Check if training should pause due to low battery.

    Parameters
    ----------
    min_pct : int
        Minimum battery percentage.

    Returns
    -------
    bool
        True if battery is below ``min_pct`` and not plugged in.
    """
    status = get_battery_status()
    if not status["available"] or status["plugged_in"]:
        return False
    return status["percent"] < min_pct


# ---------------------------------------------------------------------------
# MPS environment setup
# ---------------------------------------------------------------------------


def configure_mps_environment(high_watermark_ratio: float = 0.0) -> None:
    """Set MPS-specific environment variables for training.

    Parameters
    ----------
    high_watermark_ratio : float
        ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` value.  0.0 disables the
        limit (recommended for training to avoid premature OOM).
    """
    if not _IS_MAC:
        return

    os.environ.setdefault(
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        str(high_watermark_ratio),
    )
    # MPS fallback for unsupported ops
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    logger.info(
        "MPS environment configured: WATERMARK_RATIO=%s, FALLBACK=1",
        high_watermark_ratio,
    )


# ---------------------------------------------------------------------------
# Combined status
# ---------------------------------------------------------------------------


def get_mac_training_status(
    threshold: float = 0.85,
    min_battery_pct: int = 20,
) -> dict[str, Any]:
    """Get a combined Mac training health status.

    Parameters
    ----------
    threshold : float
        Memory pressure threshold.
    min_battery_pct : int
        Minimum battery percentage.

    Returns
    -------
    dict
        Keys: ``is_mac``, ``memory``, ``thermal_state``,
        ``battery``, ``mps_memory``, ``warnings`` (list of str).
    """
    warnings = []

    memory = get_memory_pressure()
    if memory["percent"] >= threshold:
        warnings.append(
            f"High memory pressure: {memory['percent']:.0%} used "
            f"({memory['available_gb']:.1f} GB available)"
        )

    thermal = get_thermal_state()
    if thermal in ("serious", "critical"):
        warnings.append(f"Thermal throttling detected: {thermal}")

    battery = get_battery_status()
    if battery["available"] and not battery["plugged_in"] and battery["percent"] < min_battery_pct:
        warnings.append(f"Low battery: {battery['percent']}% (min: {min_battery_pct}%)")

    mps = get_mps_memory()

    return {
        "is_mac": _IS_MAC,
        "memory": memory,
        "thermal_state": thermal,
        "battery": battery,
        "mps_memory": mps,
        "warnings": warnings,
    }
