"""Tests for distributed training configuration logic.

Tests framework selection (no GPU needed for the decision matrix),
the HardwareProfile / GPUInfo dataclasses, and auto_optimize_config
with mock hardware profiles.
"""

from __future__ import annotations

import pytest

from llm_forge.config.hardware_detector import (
    GPUInfo,
    HardwareProfile,
    NVLinkTopology,
    auto_optimize_config,
)
from llm_forge.config.schema import LLMForgeConfig

# ---------------------------------------------------------------------------
# Helpers to build mock hardware profiles
# ---------------------------------------------------------------------------


def _make_gpu(
    name: str = "NVIDIA RTX 4090",
    vram_mb: int = 24576,
    cc: tuple = (8, 9),
    index: int = 0,
) -> GPUInfo:
    return GPUInfo(
        index=index,
        name=name,
        vram_mb=vram_mb,
        compute_capability=cc,
    )


def _make_config(**overrides) -> LLMForgeConfig:
    base = {
        "model": {"name": "gpt2"},
        "data": {"train_path": "some/data"},
    }
    base.update(overrides)
    return LLMForgeConfig(**base)


# ===================================================================
# Framework selection logic
# ===================================================================


class TestFrameworkSelection:
    """Test that auto_optimize_config picks the right framework."""

    def test_cpu_only_disables_distributed(self) -> None:
        """CPU-only profile disables distributed training."""
        config = _make_config()
        profile = HardwareProfile(gpu_count=0, gpus=[], cpu_count=8, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.distributed.enabled is False
        assert optimized.training.bf16 is False
        assert optimized.training.fp16 is False

    def test_single_gpu_no_distributed(self) -> None:
        """Single GPU does not enable distributed."""
        gpu = _make_gpu()
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=8, system_ram_mb=32768)

        optimized = auto_optimize_config(config, profile)
        assert optimized.distributed.enabled is False

    def test_multi_gpu_with_nvlink_picks_fsdp(self) -> None:
        """Multi-GPU with NVLink prefers FSDP."""
        gpus = [_make_gpu(index=0), _make_gpu(index=1)]
        nvlink = NVLinkTopology(has_nvlink=True, gpu_pairs_connected=[(0, 1)])
        config = _make_config()
        profile = HardwareProfile(
            gpu_count=2, gpus=gpus, nvlink=nvlink, cpu_count=16, system_ram_mb=65536
        )

        optimized = auto_optimize_config(config, profile)
        assert optimized.distributed.enabled is True
        assert optimized.distributed.framework == "fsdp"
        assert optimized.distributed.num_gpus == 2

    def test_multi_gpu_without_nvlink_picks_deepspeed(self) -> None:
        """Multi-GPU without NVLink prefers DeepSpeed."""
        gpus = [_make_gpu(index=0), _make_gpu(index=1)]
        nvlink = NVLinkTopology(has_nvlink=False)
        config = _make_config()
        profile = HardwareProfile(
            gpu_count=2, gpus=gpus, nvlink=nvlink, cpu_count=16, system_ram_mb=65536
        )

        optimized = auto_optimize_config(config, profile)
        assert optimized.distributed.enabled is True
        assert optimized.distributed.framework == "deepspeed"


# ===================================================================
# GPU classification and per-GPU optimization
# ===================================================================


class TestPerGPUOptimization:
    """Test per-GPU auto-optimization logic."""

    def test_h100_enables_fp8(self) -> None:
        """H100 (Hopper) enables FP8."""
        gpu = _make_gpu(name="NVIDIA H100 80GB", vram_mb=81920, cc=(9, 0))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=64, system_ram_mb=262144)

        optimized = auto_optimize_config(config, profile)
        assert optimized.distributed.fp8_enabled is True
        assert optimized.training.bf16 is True

    def test_rtx_3090_uses_qlora(self) -> None:
        """RTX 3090 (consumer mid-VRAM) is set to QLoRA."""
        gpu = _make_gpu(name="NVIDIA GeForce RTX 3090", vram_mb=24576, cc=(8, 6))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=16, system_ram_mb=65536)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.mode == "qlora"
        assert optimized.quantization.load_in_4bit is True

    def test_low_vram_gpu(self) -> None:
        """GPU with <12GB VRAM uses aggressive QLoRA settings."""
        gpu = _make_gpu(name="NVIDIA GeForce RTX 3060", vram_mb=8192, cc=(8, 6))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=8, system_ram_mb=32768)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.mode == "qlora"
        assert optimized.training.per_device_train_batch_size == 1

    def test_a100_80gb_settings(self) -> None:
        """A100 80GB gets generous batch size."""
        gpu = _make_gpu(name="NVIDIA A100-SXM4-80GB", vram_mb=81920, cc=(8, 0))
        config = _make_config()
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=64, system_ram_mb=262144)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.per_device_train_batch_size >= 8

    def test_fp8_disabled_on_non_hopper(self) -> None:
        """FP8 flag is disabled when GPU does not support it."""
        gpu = _make_gpu(name="NVIDIA A100", vram_mb=81920, cc=(8, 0))
        config = _make_config()
        # Manually set fp8 to True (which should get overridden)
        config.distributed.fp8_enabled = True
        profile = HardwareProfile(gpu_count=1, gpus=[gpu], cpu_count=64, system_ram_mb=262144)

        optimized = auto_optimize_config(config, profile)
        assert optimized.distributed.fp8_enabled is False


# ===================================================================
# CPU-only fallback
# ===================================================================


class TestCPUFallback:
    """Test that CPU-only mode applies safe defaults."""

    def test_cpu_sets_fp32(self) -> None:
        config = _make_config(training={"mode": "full"})
        profile = HardwareProfile(gpu_count=0, gpus=[], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.model.torch_dtype == "fp32"
        assert optimized.model.attn_implementation == "eager"

    def test_cpu_forces_lora_from_full(self) -> None:
        """Full fine-tuning on CPU is downgraded to LoRA."""
        config = _make_config(training={"mode": "full"})
        profile = HardwareProfile(gpu_count=0, gpus=[], cpu_count=4, system_ram_mb=16384)

        optimized = auto_optimize_config(config, profile)
        assert optimized.training.mode == "lora"


# ===================================================================
# GPUInfo properties
# ===================================================================


class TestGPUInfoProperties:
    """Test GPUInfo dataclass property methods."""

    def test_vram_gb(self) -> None:
        gpu = _make_gpu(vram_mb=24576)
        assert gpu.vram_gb == pytest.approx(24.0)

    def test_is_ampere_or_newer(self) -> None:
        assert _make_gpu(cc=(8, 0)).is_ampere_or_newer is True
        assert _make_gpu(cc=(7, 5)).is_ampere_or_newer is False

    def test_is_hopper_or_newer(self) -> None:
        assert _make_gpu(cc=(9, 0)).is_hopper_or_newer is True
        assert _make_gpu(cc=(8, 9)).is_hopper_or_newer is False

    def test_supports_bf16(self) -> None:
        assert _make_gpu(cc=(8, 0)).supports_bf16 is True
        assert _make_gpu(cc=(7, 0)).supports_bf16 is False

    def test_supports_fp8(self) -> None:
        assert _make_gpu(cc=(9, 0)).supports_fp8 is True
        assert _make_gpu(cc=(8, 0)).supports_fp8 is False


# ===================================================================
# HardwareProfile properties
# ===================================================================


class TestHardwareProfileProperties:
    """Test HardwareProfile computed properties."""

    def test_total_vram(self) -> None:
        gpus = [_make_gpu(vram_mb=24576, index=0), _make_gpu(vram_mb=24576, index=1)]
        profile = HardwareProfile(gpu_count=2, gpus=gpus)
        assert profile.total_vram_mb == 49152
        assert profile.total_vram_gb == pytest.approx(48.0)

    def test_has_gpu(self) -> None:
        assert HardwareProfile(gpu_count=1, gpus=[_make_gpu()]).has_gpu is True
        assert HardwareProfile(gpu_count=0, gpus=[]).has_gpu is False

    def test_has_multi_gpu(self) -> None:
        gpus = [_make_gpu(index=0), _make_gpu(index=1)]
        assert HardwareProfile(gpu_count=2, gpus=gpus).has_multi_gpu is True
        assert HardwareProfile(gpu_count=1, gpus=[_make_gpu()]).has_multi_gpu is False

    def test_min_max_vram(self) -> None:
        gpus = [
            _make_gpu(vram_mb=8192, index=0),
            _make_gpu(vram_mb=24576, index=1),
        ]
        profile = HardwareProfile(gpu_count=2, gpus=gpus)
        assert profile.min_gpu_vram_mb == 8192
        assert profile.max_gpu_vram_mb == 24576

    def test_empty_gpu_list_vram(self) -> None:
        profile = HardwareProfile(gpu_count=0, gpus=[])
        assert profile.min_gpu_vram_mb == 0
        assert profile.max_gpu_vram_mb == 0

    def test_summary_string(self) -> None:
        profile = HardwareProfile(
            gpu_count=0,
            gpus=[],
            cpu_count=4,
            system_ram_mb=16384,
            disk_free_mb=102400,
            hostname="testhost",
            os_name="Linux 5.15",
        )
        summary = profile.summary()
        assert "testhost" in summary
        assert "CPU-only" in summary
