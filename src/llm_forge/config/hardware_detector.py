"""Hardware auto-detection and config optimisation for llm-forge.

Detects GPUs (count, model, VRAM, compute capability), CUDA version,
NVLink topology, system RAM, and disk space.  Provides
:func:`auto_optimize_config` to adjust training hyper-parameters based
on the detected hardware profile.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from llm_forge.config.schema import (
    LLMForgeConfig,
    PrecisionMode,
    TrainingMode,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    """Information about a single GPU device."""

    index: int
    name: str
    vram_mb: int
    compute_capability: tuple[int, int]
    uuid: str = ""

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024.0

    @property
    def is_ampere_or_newer(self) -> bool:
        """Ampere = compute capability >= 8.0."""
        return self.compute_capability >= (8, 0)

    @property
    def is_hopper_or_newer(self) -> bool:
        """Hopper = compute capability >= 9.0."""
        return self.compute_capability >= (9, 0)

    @property
    def supports_bf16(self) -> bool:
        return self.is_ampere_or_newer

    @property
    def supports_fp8(self) -> bool:
        return self.is_hopper_or_newer


@dataclass
class NVLinkTopology:
    """Parsed NVLink connectivity information."""

    raw_matrix: str = ""
    gpu_pairs_connected: list[tuple[int, int]] = field(default_factory=list)
    has_nvlink: bool = False


@dataclass
class HardwareProfile:
    """Complete hardware profile of the current machine."""

    # GPU
    gpu_count: int = 0
    gpus: list[GPUInfo] = field(default_factory=list)
    cuda_version: str | None = None
    driver_version: str | None = None
    nvlink: NVLinkTopology = field(default_factory=NVLinkTopology)

    # Apple Silicon / MPS
    is_mps: bool = False
    apple_chip: str | None = None  # e.g. "Apple M4 Pro"

    # System
    system_ram_mb: int = 0
    disk_free_mb: int = 0
    cpu_count: int = 0
    os_name: str = ""
    hostname: str = ""

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_mb for g in self.gpus)

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_mb / 1024.0

    @property
    def system_ram_gb(self) -> float:
        return self.system_ram_mb / 1024.0

    @property
    def disk_free_gb(self) -> float:
        return self.disk_free_mb / 1024.0

    @property
    def min_gpu_vram_mb(self) -> int:
        if not self.gpus:
            return 0
        return min(g.vram_mb for g in self.gpus)

    @property
    def max_gpu_vram_mb(self) -> int:
        if not self.gpus:
            return 0
        return max(g.vram_mb for g in self.gpus)

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0

    @property
    def has_multi_gpu(self) -> bool:
        return self.gpu_count > 1

    @property
    def all_support_bf16(self) -> bool:
        return all(g.supports_bf16 for g in self.gpus)

    @property
    def any_supports_fp8(self) -> bool:
        return any(g.supports_fp8 for g in self.gpus)

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = [
            "=== Hardware Profile ===",
            f"  Host     : {self.hostname} ({self.os_name})",
            f"  CPUs     : {self.cpu_count}",
            f"  RAM      : {self.system_ram_gb:.1f} GB",
            f"  Disk free: {self.disk_free_gb:.1f} GB",
        ]
        if self.has_gpu:
            lines.append(f"  GPUs     : {self.gpu_count}")
            lines.append(f"  CUDA     : {self.cuda_version or 'unknown'}")
            lines.append(f"  Driver   : {self.driver_version or 'unknown'}")
            for g in self.gpus:
                lines.append(
                    f"    [{g.index}] {g.name}  |  "
                    f"{g.vram_gb:.1f} GB  |  "
                    f"CC {g.compute_capability[0]}.{g.compute_capability[1]}"
                )
            if self.nvlink.has_nvlink:
                pairs = ", ".join(f"GPU{a}<->GPU{b}" for a, b in self.nvlink.gpu_pairs_connected)
                lines.append(f"  NVLink   : {pairs}")
        elif self.is_mps:
            chip = self.apple_chip or "Apple Silicon"
            lines.append(f"  GPU      : {chip} (MPS)")
            lines.append(f"  Unified  : {self.system_ram_gb:.1f} GB (shared)")
        else:
            lines.append("  GPUs     : none (CPU-only)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# nvidia-smi helpers
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], timeout: int = 15) -> str | None:
    """Run a shell command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _detect_gpus_nvidia_smi() -> list[GPUInfo]:
    """Query nvidia-smi for GPU details."""
    output = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []

    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            vram = int(float(parts[2]))
            cc_str = parts[3]
            uuid = parts[4] if len(parts) > 4 else ""

            # compute_cap can be "8.6" or "8.9"
            cc_parts = cc_str.split(".")
            cc = (int(cc_parts[0]), int(cc_parts[1]) if len(cc_parts) > 1 else 0)

            gpus.append(
                GPUInfo(
                    index=idx,
                    name=name,
                    vram_mb=vram,
                    compute_capability=cc,
                    uuid=uuid,
                )
            )
        except (ValueError, IndexError):
            continue
    return gpus


def _detect_cuda_version() -> str | None:
    """Detect CUDA runtime version via nvidia-smi or nvcc."""
    # nvidia-smi reports CUDA version in its header
    output = _run_cmd(["nvidia-smi"])
    if output:
        match = re.search(r"CUDA Version:\s*([\d.]+)", output)
        if match:
            return match.group(1)

    # Fallback: nvcc
    output = _run_cmd(["nvcc", "--version"])
    if output:
        match = re.search(r"release\s+([\d.]+)", output)
        if match:
            return match.group(1)

    # Fallback: try torch
    try:
        import torch

        if torch.cuda.is_available():
            return torch.version.cuda
    except ImportError:
        pass

    return None


def _detect_driver_version() -> str | None:
    output = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    if output:
        return output.splitlines()[0].strip()
    return None


def _detect_nvlink_topology() -> NVLinkTopology:
    """Parse ``nvidia-smi topo -m`` to find NVLink connections."""
    output = _run_cmd(["nvidia-smi", "topo", "-m"])
    if output is None:
        return NVLinkTopology()

    topo = NVLinkTopology(raw_matrix=output)

    # The topology matrix has GPU rows and columns.
    # NVLink connections show as "NV<N>" (e.g. NV12, NV4).
    lines = output.splitlines()

    # Find the header row that starts with "GPU" or a tab
    gpu_indices: list[int] = []
    matrix_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("GPU"):
            # Could be header "GPU0  GPU1 ..." or data row "GPU0  X  NV12 ..."
            # Try to parse as data row
            parts = stripped.split()
            if len(parts) > 1 and parts[0].startswith("GPU"):
                try:
                    idx = int(parts[0].replace("GPU", ""))
                    gpu_indices.append(idx)
                    matrix_lines.append(stripped)
                except ValueError:
                    pass

    pairs_seen: set = set()
    for row_line in matrix_lines:
        parts = row_line.split()
        if not parts:
            continue
        try:
            row_gpu = int(parts[0].replace("GPU", ""))
        except ValueError:
            continue
        for col_offset, cell in enumerate(parts[1:]):
            if cell.startswith("NV"):
                col_gpu = col_offset
                if col_gpu != row_gpu:
                    pair = (min(row_gpu, col_gpu), max(row_gpu, col_gpu))
                    if pair not in pairs_seen:
                        pairs_seen.add(pair)
                        topo.gpu_pairs_connected.append(pair)

    topo.has_nvlink = len(topo.gpu_pairs_connected) > 0
    return topo


def _detect_system_ram_mb() -> int:
    """Get total system RAM in MB."""
    try:
        import psutil

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        pass

    # Fallback for Linux
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except (OSError, ValueError):
            pass

    # Fallback for macOS
    if platform.system() == "Darwin":
        output = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if output:
            try:
                return int(output) // (1024 * 1024)
            except ValueError:
                pass

    return 0


def _detect_disk_free_mb(path: str = ".") -> int:
    """Get free disk space in MB at the given path."""
    try:
        usage = shutil.disk_usage(path)
        return int(usage.free / (1024 * 1024))
    except OSError:
        return 0


def _detect_gpus_via_torch() -> list[GPUInfo]:
    """Fallback GPU detection using PyTorch."""
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    gpus: list[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            GPUInfo(
                index=i,
                name=props.name,
                vram_mb=props.total_memory // (1024 * 1024),
                compute_capability=(props.major, props.minor),
            )
        )
    return gpus


def _detect_mps() -> bool:
    """Check if Apple Metal Performance Shaders (MPS) backend is available."""
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def _detect_apple_chip() -> str | None:
    """Detect Apple Silicon chip name (e.g. 'Apple M4 Pro')."""
    if platform.system() != "Darwin":
        return None
    output = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    return output if output else None


# ---------------------------------------------------------------------------
# Public API - detection
# ---------------------------------------------------------------------------


def detect_hardware() -> HardwareProfile:
    """Detect all available hardware and return a :class:`HardwareProfile`.

    This function is safe to call on any machine -- it gracefully handles
    missing GPUs, missing ``nvidia-smi``, and missing optional dependencies.
    """
    # GPUs: prefer nvidia-smi, fall back to torch
    gpus = _detect_gpus_nvidia_smi()
    if not gpus:
        gpus = _detect_gpus_via_torch()

    # Apple Silicon / MPS — detect chip on any Mac, MPS when torch available
    apple_chip = _detect_apple_chip() if not gpus else None
    is_mps = _detect_mps() if (not gpus and apple_chip) else False

    profile = HardwareProfile(
        gpu_count=len(gpus),
        gpus=gpus,
        cuda_version=_detect_cuda_version() if gpus else None,
        driver_version=_detect_driver_version() if gpus else None,
        nvlink=_detect_nvlink_topology() if len(gpus) > 1 else NVLinkTopology(),
        is_mps=is_mps,
        apple_chip=apple_chip,
        system_ram_mb=_detect_system_ram_mb(),
        disk_free_mb=_detect_disk_free_mb(),
        cpu_count=os.cpu_count() or 1,
        os_name=f"{platform.system()} {platform.release()}",
        hostname=platform.node(),
    )
    return profile


# ---------------------------------------------------------------------------
# GPU classification
# ---------------------------------------------------------------------------


class _GPUClass:
    """Internal helper to classify a GPU by family and VRAM."""

    def __init__(self, gpu: GPUInfo) -> None:
        self.gpu = gpu
        name_lower = gpu.name.lower()

        # Determine family
        self.is_consumer = any(
            tag in name_lower
            for tag in ("geforce", "rtx 20", "rtx 30", "rtx 40", "rtx 50", "titan")
        )
        self.is_datacenter = any(
            tag in name_lower
            for tag in ("a100", "a800", "h100", "h200", "h800", "b100", "b200", "l40")
        )

        # Specific GPU detection
        self.is_rtx_3090 = "3090" in name_lower
        self.is_rtx_4090 = "4090" in name_lower
        self.is_rtx_5090 = "5090" in name_lower
        self.is_a100_40 = "a100" in name_lower and gpu.vram_mb < 50_000
        self.is_a100_80 = "a100" in name_lower and gpu.vram_mb >= 50_000
        self.is_h100 = "h100" in name_lower
        self.is_h200 = "h200" in name_lower
        self.is_l40 = "l40" in name_lower

        # VRAM tiers
        self.vram_gb = gpu.vram_gb
        self.is_low_vram = self.vram_gb < 12
        self.is_mid_vram = 12 <= self.vram_gb < 32
        self.is_high_vram = 32 <= self.vram_gb < 64
        self.is_very_high_vram = self.vram_gb >= 64


# ---------------------------------------------------------------------------
# Public API - auto-optimisation
# ---------------------------------------------------------------------------


def auto_optimize_config(
    config: LLMForgeConfig,
    profile: HardwareProfile | None = None,
) -> LLMForgeConfig:
    """Adjust *config* in-place based on the detected (or supplied) hardware.

    This function applies conservative, well-tested defaults for each GPU
    class.  The caller's explicit settings are preserved wherever possible --
    only clearly suboptimal or incompatible values are overridden.

    Parameters
    ----------
    config:
        The configuration to optimise.
    profile:
        Hardware profile.  If ``None``, :func:`detect_hardware` is called.

    Returns
    -------
    LLMForgeConfig
        The same object, mutated in place for convenience.
    """
    if profile is None:
        profile = detect_hardware()

    # ---- Apple Silicon / MPS -----------------------------------------------
    if profile.is_mps or profile.apple_chip:
        ram_gb = profile.system_ram_gb
        config.training.bf16 = False
        config.training.fp16 = False
        config.model.torch_dtype = PrecisionMode.fp32
        config.model.attn_implementation = "eager"
        config.distributed.enabled = False
        config.training.use_unsloth = False

        # Memory-based sizing — unified memory is shared with OS
        usable_gb = ram_gb * 0.65  # ~65% of unified RAM is safely usable
        if usable_gb >= 23:  # 36+ GB Mac
            config.training.per_device_train_batch_size = 4
            config.training.gradient_accumulation_steps = 4
            config.training.gradient_checkpointing = False
        elif usable_gb >= 10:  # 16-32 GB Mac
            config.training.mode = TrainingMode.qlora
            config.quantization.load_in_4bit = True
            config.quantization.bnb_4bit_quant_type = "nf4"
            config.quantization.bnb_4bit_use_double_quant = True
            config.training.per_device_train_batch_size = 2
            config.training.gradient_accumulation_steps = 8
            config.training.gradient_checkpointing = True
        else:  # 8 GB Mac
            config.training.mode = TrainingMode.qlora
            config.quantization.load_in_4bit = True
            config.quantization.bnb_4bit_quant_type = "nf4"
            config.quantization.bnb_4bit_use_double_quant = True
            config.training.per_device_train_batch_size = 1
            config.training.gradient_accumulation_steps = 16
            config.training.gradient_checkpointing = True
            config.model.max_seq_length = min(config.model.max_seq_length, 1024)

        config.data.num_workers = min(config.data.num_workers, max(1, profile.cpu_count - 2))
        return config

    # ---- CPU-only fallback ------------------------------------------------
    if not profile.has_gpu:
        config.training.bf16 = False
        config.training.fp16 = False
        config.model.torch_dtype = PrecisionMode.fp32
        config.model.attn_implementation = "eager"
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 8
        config.training.gradient_checkpointing = True
        config.training.use_unsloth = False
        config.distributed.enabled = False
        if config.training.mode in (TrainingMode.full, TrainingMode.pretrain):
            config.training.mode = TrainingMode.lora
        return config

    # ---- Multi-GPU --------------------------------------------------------
    if profile.has_multi_gpu:
        config.distributed.enabled = True
        config.distributed.num_gpus = profile.gpu_count
        # If all GPUs have NVLink, prefer FSDP; else DeepSpeed
        if profile.nvlink.has_nvlink:
            config.distributed.framework = "fsdp"
            config.distributed.fsdp_sharding_strategy = "FULL_SHARD"
        else:
            config.distributed.framework = "deepspeed"
            config.distributed.deepspeed_stage = 2

    # ---- Per-GPU optimisation (use the weakest GPU as the constraint) -----
    weakest = min(profile.gpus, key=lambda g: g.vram_mb)
    gc = _GPUClass(weakest)

    # Precision
    if weakest.supports_bf16:
        config.training.bf16 = True
        config.training.fp16 = False
        config.model.torch_dtype = PrecisionMode.bf16
    else:
        config.training.bf16 = False
        config.training.fp16 = True
        config.model.torch_dtype = PrecisionMode.fp16

    # Attention implementation: only override if the current setting
    # is incompatible with the hardware. Respect user-specified values.
    if weakest.compute_capability < (8, 0):
        # Pre-Ampere: flash_attention_2 is not supported
        if config.model.attn_implementation == "flash_attention_2":
            config.model.attn_implementation = "sdpa"

    # ---- RTX 3090 (24 GB) -------------------------------------------------
    if gc.is_rtx_3090 or (gc.is_consumer and gc.is_mid_vram):
        config.training.mode = TrainingMode.qlora
        config.quantization.load_in_4bit = True
        config.quantization.bnb_4bit_quant_type = "nf4"
        config.quantization.bnb_4bit_use_double_quant = True
        config.training.gradient_checkpointing = True
        config.training.per_device_train_batch_size = 2
        config.training.gradient_accumulation_steps = 8
        config.training.optim = "paged_adamw_8bit"

    # ---- RTX 4090 (24 GB) -------------------------------------------------
    elif gc.is_rtx_4090 or gc.is_rtx_5090:
        if config.training.mode not in (TrainingMode.qlora, TrainingMode.lora):
            config.training.mode = TrainingMode.lora
        config.training.gradient_checkpointing = True
        config.training.per_device_train_batch_size = 4
        config.training.gradient_accumulation_steps = 4

    # ---- A100 40 GB -------------------------------------------------------
    elif gc.is_a100_40 or (gc.is_datacenter and gc.is_high_vram and gc.vram_gb < 50):
        if config.training.mode == TrainingMode.full:
            config.training.mode = TrainingMode.lora
        config.training.per_device_train_batch_size = 8
        config.training.gradient_accumulation_steps = 2
        config.training.gradient_checkpointing = False

    # ---- A100 80 GB -------------------------------------------------------
    elif gc.is_a100_80 or (
        gc.is_datacenter and gc.is_very_high_vram and not gc.is_h100 and not gc.is_h200
    ):
        # Full fine-tune is feasible for models up to ~7B
        config.training.per_device_train_batch_size = 16
        config.training.gradient_accumulation_steps = 1
        config.training.gradient_checkpointing = False

    # ---- H100 / H200 (80-141 GB) -----------------------------------------
    elif gc.is_h100 or gc.is_h200:
        config.training.per_device_train_batch_size = 16
        config.training.gradient_accumulation_steps = 1
        config.training.gradient_checkpointing = False
        # Enable FP8 for Hopper
        config.distributed.fp8_enabled = True
        config.distributed.fp8_format = "HYBRID"

    # ---- Low VRAM fallback (< 12 GB) -------------------------------------
    elif gc.is_low_vram:
        config.training.mode = TrainingMode.qlora
        config.quantization.load_in_4bit = True
        config.quantization.bnb_4bit_quant_type = "nf4"
        config.quantization.bnb_4bit_use_double_quant = True
        config.training.gradient_checkpointing = True
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 16
        config.training.optim = "paged_adamw_8bit"
        config.model.max_seq_length = min(config.model.max_seq_length, 1024)

    # ---- Generic high-VRAM datacenter fallback ----------------------------
    elif gc.is_high_vram or gc.is_very_high_vram:
        config.training.per_device_train_batch_size = 8
        config.training.gradient_accumulation_steps = 2

    # ---- FP8 guard: disable if hardware doesn't support it ----------------
    if config.distributed.fp8_enabled and not weakest.supports_fp8:
        config.distributed.fp8_enabled = False

    # ---- Data workers: clamp to CPU count ---------------------------------
    config.data.num_workers = min(config.data.num_workers, max(1, profile.cpu_count - 1))

    return config
