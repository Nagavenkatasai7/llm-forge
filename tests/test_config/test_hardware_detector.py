"""Tests for hardware auto-detection and config optimisation."""

from __future__ import annotations

import platform

import pytest

try:
    from llm_forge.config.hardware_detector import (
        GPUInfo,
        HardwareProfile,
        NVLinkTopology,
        auto_optimize_config,
        detect_hardware,
    )
    from llm_forge.config.schema import LLMForgeConfig

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.config.hardware_detector not importable",
)


def _minimal_config(**overrides) -> LLMForgeConfig:
    base = {"model": {"name": "test-model"}, "data": {"train_path": "test-data"}}
    base.update(overrides)
    return LLMForgeConfig(**base)


def _make_gpu(name: str = "NVIDIA A100", vram_mb: int = 81920, cc: tuple = (8, 0)) -> GPUInfo:
    return GPUInfo(index=0, name=name, vram_mb=vram_mb, compute_capability=cc)


# ===================================================================
# HardwareProfile tests
# ===================================================================


class TestHardwareProfile:
    """Verify HardwareProfile dataclass properties."""

    def test_has_gpu_false_when_empty(self) -> None:
        profile = HardwareProfile()
        assert profile.has_gpu is False

    def test_has_gpu_true(self) -> None:
        profile = HardwareProfile(gpu_count=1, gpus=[_make_gpu()])
        assert profile.has_gpu is True

    def test_total_vram(self) -> None:
        g1 = _make_gpu(vram_mb=40960)
        g2 = _make_gpu(vram_mb=40960)
        g2.index = 1
        profile = HardwareProfile(gpu_count=2, gpus=[g1, g2])
        assert profile.total_vram_mb == 81920

    def test_mps_profile(self) -> None:
        profile = HardwareProfile(
            is_mps=True,
            apple_chip="Apple M4 Pro",
            system_ram_mb=18 * 1024,
        )
        assert profile.is_mps is True
        assert profile.apple_chip == "Apple M4 Pro"
        assert not profile.has_gpu

    def test_summary_nvidia(self) -> None:
        profile = HardwareProfile(
            gpu_count=1,
            gpus=[_make_gpu()],
            cuda_version="12.4",
            system_ram_mb=64 * 1024,
            os_name="Linux 5.15",
            hostname="testhost",
            cpu_count=16,
        )
        s = profile.summary()
        assert "A100" in s
        assert "CUDA" in s

    def test_summary_mps(self) -> None:
        profile = HardwareProfile(
            is_mps=True,
            apple_chip="Apple M4 Pro",
            system_ram_mb=18 * 1024,
            os_name="Darwin 25.3.0",
            hostname="macbook",
            cpu_count=12,
        )
        s = profile.summary()
        assert "MPS" in s
        assert "Apple M4 Pro" in s

    def test_summary_cpu_only(self) -> None:
        profile = HardwareProfile(
            system_ram_mb=32 * 1024,
            os_name="Linux 5.15",
            hostname="cpu-host",
            cpu_count=8,
        )
        s = profile.summary()
        assert "CPU-only" in s


# ===================================================================
# GPUInfo tests
# ===================================================================


class TestGPUInfo:
    """Verify GPU classification properties."""

    def test_ampere_detection(self) -> None:
        gpu = _make_gpu(cc=(8, 0))
        assert gpu.is_ampere_or_newer is True
        assert gpu.supports_bf16 is True

    def test_pre_ampere(self) -> None:
        gpu = _make_gpu(cc=(7, 5))
        assert gpu.is_ampere_or_newer is False
        assert gpu.supports_bf16 is False

    def test_hopper_detection(self) -> None:
        gpu = _make_gpu(name="NVIDIA H100", cc=(9, 0))
        assert gpu.is_hopper_or_newer is True
        assert gpu.supports_fp8 is True

    def test_vram_gb(self) -> None:
        gpu = _make_gpu(vram_mb=81920)
        assert gpu.vram_gb == pytest.approx(80.0, abs=0.1)


# ===================================================================
# detect_hardware tests
# ===================================================================


class TestDetectHardware:
    """Verify detect_hardware() runs without errors."""

    def test_detect_returns_profile(self) -> None:
        """detect_hardware() should return a HardwareProfile on any machine."""
        profile = detect_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.system_ram_mb > 0
        assert profile.cpu_count > 0
        assert profile.os_name != ""

    def test_detect_on_this_machine(self) -> None:
        """Verify detection works for the current platform."""
        profile = detect_hardware()
        if platform.system() == "Darwin":
            # On Mac, MPS may or may not be available depending on torch
            assert profile.system_ram_mb > 0
        else:
            # On Linux/Windows, GPU count is whatever is available
            assert profile.gpu_count >= 0


# ===================================================================
# auto_optimize_config tests
# ===================================================================


class TestAutoOptimize:
    """Verify auto_optimize_config produces valid configs."""

    def test_cpu_only_profile(self) -> None:
        """CPU-only should set fp32, batch_size=1, gradient_checkpointing."""
        cfg = _minimal_config()
        profile = HardwareProfile(
            system_ram_mb=32 * 1024,
            cpu_count=8,
        )
        result = auto_optimize_config(cfg, profile)
        assert result.training.bf16 is False
        assert result.training.fp16 is False
        assert result.training.per_device_train_batch_size == 1
        assert result.training.gradient_checkpointing is True

    def test_mps_16gb(self) -> None:
        """MPS with 16GB should use QLoRA, batch_size=2."""
        cfg = _minimal_config()
        profile = HardwareProfile(
            is_mps=True,
            apple_chip="Apple M4",
            system_ram_mb=16 * 1024,
            cpu_count=10,
        )
        result = auto_optimize_config(cfg, profile)
        assert result.training.bf16 is False
        assert result.training.fp16 is False
        assert result.quantization.load_in_4bit is True
        assert result.training.per_device_train_batch_size == 2
        assert result.training.gradient_checkpointing is True

    def test_mps_8gb(self) -> None:
        """MPS with 8GB should use QLoRA, batch_size=1, limited seq_length."""
        cfg = _minimal_config()
        profile = HardwareProfile(
            is_mps=True,
            apple_chip="Apple M3",
            system_ram_mb=8 * 1024,
            cpu_count=8,
        )
        result = auto_optimize_config(cfg, profile)
        assert result.training.per_device_train_batch_size == 1
        assert result.training.gradient_accumulation_steps == 16
        assert result.model.max_seq_length <= 1024

    def test_mps_36gb(self) -> None:
        """MPS with 36GB should use batch_size=4, no gradient checkpointing."""
        cfg = _minimal_config()
        profile = HardwareProfile(
            is_mps=True,
            apple_chip="Apple M4 Max",
            system_ram_mb=36 * 1024,
            cpu_count=14,
        )
        result = auto_optimize_config(cfg, profile)
        assert result.training.per_device_train_batch_size == 4
        assert result.training.gradient_checkpointing is False

    def test_a100_80gb(self) -> None:
        """A100 80GB should use batch_size=16, no gradient checkpointing."""
        cfg = _minimal_config()
        profile = HardwareProfile(
            gpu_count=1,
            gpus=[_make_gpu("NVIDIA A100-SXM4-80GB", 81920, (8, 0))],
            cuda_version="12.4",
            system_ram_mb=256 * 1024,
            cpu_count=64,
        )
        result = auto_optimize_config(cfg, profile)
        assert result.training.bf16 is True
        assert result.training.per_device_train_batch_size == 16

    def test_rtx_3090(self) -> None:
        """RTX 3090 should use QLoRA, 4-bit, paged optimizer."""
        cfg = _minimal_config()
        profile = HardwareProfile(
            gpu_count=1,
            gpus=[_make_gpu("NVIDIA GeForce RTX 3090", 24576, (8, 6))],
            cuda_version="12.4",
            system_ram_mb=64 * 1024,
            cpu_count=16,
        )
        result = auto_optimize_config(cfg, profile)
        assert result.quantization.load_in_4bit is True
        assert result.training.gradient_checkpointing is True
        assert "paged" in result.training.optim

    def test_config_still_validates(self) -> None:
        """Optimized config should still pass Pydantic validation."""
        cfg = _minimal_config()
        profile = HardwareProfile(
            is_mps=True,
            apple_chip="Apple M4 Pro",
            system_ram_mb=18 * 1024,
            cpu_count=12,
        )
        result = auto_optimize_config(cfg, profile)
        # Re-validate by dumping and reloading
        data = result.model_dump()
        reloaded = LLMForgeConfig(**data)
        assert reloaded.training.bf16 is False
