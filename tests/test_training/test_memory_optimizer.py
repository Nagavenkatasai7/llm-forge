"""Tests for memory estimation and gradient accumulation logic.

All tests are pure-math computations that do not require a GPU.
Tests the HardwareProfile/GPUInfo dataclass math and
auto_optimize_config memory-related heuristics.
"""

from __future__ import annotations

from llm_forge.config.hardware_detector import (
    GPUInfo,
    HardwareProfile,
    auto_optimize_config,
)
from llm_forge.config.schema import LLMForgeConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gpu(
    name: str = "NVIDIA A100-SXM4-40GB",
    vram_mb: int = 40960,
    cc: tuple = (8, 0),
    index: int = 0,
) -> GPUInfo:
    return GPUInfo(index=index, name=name, vram_mb=vram_mb, compute_capability=cc)


def _make_config(**overrides) -> LLMForgeConfig:
    base = {
        "model": {"name": "gpt2"},
        "data": {"train_path": "some/data"},
    }
    base.update(overrides)
    return LLMForgeConfig(**base)


# ===================================================================
# Memory estimation via VRAM-based heuristics
# ===================================================================


class TestMemoryEstimationHeuristics:
    """Test the heuristic memory/batch decisions in auto_optimize_config."""

    def test_low_vram_small_batch(self) -> None:
        """GPUs with < 12GB VRAM should get batch_size=1."""
        gpu = _make_gpu(name="NVIDIA GeForce RTX 3060", vram_mb=8192, cc=(8, 6))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.per_device_train_batch_size == 1
        assert optimized.training.gradient_accumulation_steps >= 8

    def test_mid_vram_moderate_batch(self) -> None:
        """GPUs with ~24GB VRAM (consumer) should get batch_size <= 4."""
        gpu = _make_gpu(name="NVIDIA GeForce RTX 4090", vram_mb=24576, cc=(8, 9))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=16, system_ram_mb=65536)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.per_device_train_batch_size <= 4

    def test_high_vram_larger_batch(self) -> None:
        """GPUs with 80GB VRAM should get batch_size >= 8."""
        gpu = _make_gpu(name="NVIDIA A100-SXM4-80GB", vram_mb=81920, cc=(8, 0))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=64, system_ram_mb=262144)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.per_device_train_batch_size >= 8


# ===================================================================
# Gradient accumulation computation
# ===================================================================


class TestGradientAccumulation:
    """Test gradient accumulation step choices."""

    def test_low_vram_high_accumulation(self) -> None:
        """Low VRAM GPUs need more gradient accumulation steps."""
        gpu = _make_gpu(name="NVIDIA GeForce RTX 3060", vram_mb=8192, cc=(8, 6))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.gradient_accumulation_steps >= 16

    def test_high_vram_low_accumulation(self) -> None:
        """High VRAM GPUs can use fewer accumulation steps."""
        gpu = _make_gpu(name="NVIDIA A100-SXM4-80GB", vram_mb=81920, cc=(8, 0))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=64, system_ram_mb=262144)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.gradient_accumulation_steps <= 2

    def test_cpu_uses_high_accumulation(self) -> None:
        """CPU-only mode uses high gradient accumulation."""
        config = _make_config()
        profile = HardwareProfile(gpu_count=0, gpus=[], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.gradient_accumulation_steps >= 8

    def test_effective_batch_size_relationship(self) -> None:
        """Effective batch = batch_size * gradient_accumulation should be reasonable."""
        gpu = _make_gpu(name="NVIDIA GeForce RTX 3060", vram_mb=8192, cc=(8, 6))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        effective = (
            optimized.training.per_device_train_batch_size
            * optimized.training.gradient_accumulation_steps
        )
        # Effective batch should be at least 8 for reasonable convergence
        assert effective >= 8


# ===================================================================
# Data workers clamped to CPU count
# ===================================================================


class TestDataWorkerClamping:
    """Test that num_workers is clamped to cpu_count - 1."""

    def test_workers_clamped(self) -> None:
        gpu = _make_gpu()
        config = _make_config()
        config.data.num_workers = 32  # Way more than our fake CPU count
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.data.num_workers <= 3  # cpu_count - 1


# ===================================================================
# Gradient checkpointing based on VRAM
# ===================================================================


class TestGradientCheckpointing:
    """Test gradient checkpointing decisions."""

    def test_low_vram_enables_checkpointing(self) -> None:
        gpu = _make_gpu(name="NVIDIA GeForce RTX 3060", vram_mb=8192, cc=(8, 6))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.gradient_checkpointing is True

    def test_high_vram_may_disable_checkpointing(self) -> None:
        gpu = _make_gpu(name="NVIDIA A100-SXM4-80GB", vram_mb=81920, cc=(8, 0))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=64, system_ram_mb=262144)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.gradient_checkpointing is False

    def test_cpu_enables_checkpointing(self) -> None:
        config = _make_config()
        profile = HardwareProfile(gpu_count=0, gpus=[], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.gradient_checkpointing is True


# ===================================================================
# Sequence length clamping on low VRAM
# ===================================================================


class TestSequenceLengthClamping:
    """Test max_seq_length is reduced on very low VRAM GPUs."""

    def test_low_vram_clamps_seq_length(self) -> None:
        gpu = _make_gpu(name="NVIDIA GeForce RTX 3060", vram_mb=8192, cc=(8, 6))
        config = _make_config()
        config.model.max_seq_length = 4096  # Higher than what low VRAM can handle
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.model.max_seq_length <= 1024
