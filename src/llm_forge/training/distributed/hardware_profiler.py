"""Hardware profiler for estimating memory requirements and recommending
distributed training configurations.

Provides detailed memory math for model parameters, optimizer states,
gradients, and activations across different parallelism strategies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.distributed.hardware_profiler")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryEstimate:
    """Breakdown of GPU memory requirements in gigabytes."""

    weights_gb: float = 0.0
    gradients_gb: float = 0.0
    optimizer_states_gb: float = 0.0
    activations_gb: float = 0.0
    total_gb: float = 0.0
    per_gpu_gb: float = 0.0
    num_gpus: int = 1
    parallelism_strategy: str = "none"
    precision: str = "bf16"

    def __repr__(self) -> str:
        return (
            f"MemoryEstimate(total={self.total_gb:.2f} GB, "
            f"per_gpu={self.per_gpu_gb:.2f} GB, "
            f"gpus={self.num_gpus}, "
            f"strategy={self.parallelism_strategy})"
        )


@dataclass
class ParallelismRecommendation:
    """Recommended distributed training configuration."""

    framework: str
    strategy: str
    num_gpus: int
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    zero_stage: int = 0
    cpu_offload: bool = False
    nvme_offload: bool = False
    estimated_per_gpu_gb: float = 0.0
    fits_in_vram: bool = True
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bytes per parameter for different precisions
# ---------------------------------------------------------------------------

_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "nf4": 0.5,
}

# AdamW optimizer: fp32 copy + momentum + variance = 3 * 4 bytes = 12 bytes/param
_OPTIMIZER_BYTES_PER_PARAM = {
    "adamw_torch": 12.0,
    "adamw_8bit": 6.0,  # 8-bit optimizer states
    "paged_adamw_32bit": 12.0,
    "paged_adamw_8bit": 6.0,
    "sgd": 4.0,  # just momentum
    "adafactor": 8.0,
}


# ============================================================================
# HardwareProfiler
# ============================================================================


class HardwareProfiler:
    """Estimates memory requirements and recommends parallelism strategies.

    Performs precise memory calculations based on model parameters,
    sequence length, batch size, and available hardware to recommend
    the optimal distributed training configuration.
    """

    def estimate_memory(
        self,
        model_params: float,
        precision: str = "bf16",
        optimizer: str = "adamw_torch",
        num_gpus: int = 1,
        zero_stage: int = 0,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        lora_rank: int | None = None,
        lora_fraction: float = 0.02,
    ) -> MemoryEstimate:
        """Estimate total GPU memory for training a model.

        For mixed-precision AdamW training, the memory per parameter is:
        - Weights: 2B (bf16/fp16)
        - Gradients: 2B (bf16/fp16)
        - Optimizer states: 12B (fp32 master weights + momentum + variance)
        - Total: ~16 bytes/parameter

        With ZeRO-3, optimizer states, gradients, and parameters are
        sharded across N GPUs: 16P/N bytes per GPU.

        Parameters
        ----------
        model_params : float
            Number of model parameters (can be in billions, e.g. 7.0 for 7B).
        precision : str
            Weight precision for forward/backward pass.
        optimizer : str
            Optimizer name for state size estimation.
        num_gpus : int
            Number of GPUs for partitioning.
        zero_stage : int
            DeepSpeed ZeRO stage (0-3).
        tensor_parallel : int
            Tensor parallelism degree.
        pipeline_parallel : int
            Pipeline parallelism degree.
        lora_rank : int, optional
            LoRA rank.  When set, only the LoRA parameters require
            optimizer states and gradients.
        lora_fraction : float
            Approximate fraction of parameters that are trainable with LoRA.

        Returns
        -------
        MemoryEstimate
            Detailed memory breakdown.
        """
        # Normalise: if model_params < 1000, treat as billions
        params = model_params * 1000000000.0 if model_params < 1000 else model_params

        bytes_per = _BYTES_PER_PARAM.get(precision.lower(), 2.0)
        opt_bytes = _OPTIMIZER_BYTES_PER_PARAM.get(optimizer, 12.0)

        # Effective parallelism
        tp = tensor_parallel
        pp = pipeline_parallel
        dp = max(1, num_gpus // (tp * pp))

        # Model weights per GPU (tensor parallel splits weights)
        weights_bytes = params * bytes_per / tp

        # Trainable parameters
        trainable_params = params * lora_fraction if lora_rank is not None else params

        # Gradients (only for trainable parameters)
        grad_bytes = trainable_params * bytes_per / tp

        # Optimizer states (only for trainable parameters)
        opt_state_bytes = trainable_params * opt_bytes

        # ZeRO sharding (across data-parallel ranks)
        if zero_stage == 0:
            # No sharding
            sharded_opt = opt_state_bytes
            sharded_grad = grad_bytes
            sharded_weights = weights_bytes
        elif zero_stage == 1:
            # Shard optimizer states
            sharded_opt = opt_state_bytes / dp
            sharded_grad = grad_bytes
            sharded_weights = weights_bytes
        elif zero_stage == 2:
            # Shard optimizer states + gradients
            sharded_opt = opt_state_bytes / dp
            sharded_grad = grad_bytes / dp
            sharded_weights = weights_bytes
        elif zero_stage == 3:
            # Shard everything
            sharded_opt = opt_state_bytes / dp
            sharded_grad = grad_bytes / dp
            sharded_weights = weights_bytes / dp
        else:
            sharded_opt = opt_state_bytes
            sharded_grad = grad_bytes
            sharded_weights = weights_bytes

        # Pipeline parallel: each stage holds 1/pp of the layers
        sharded_weights /= pp
        sharded_grad /= pp
        sharded_opt /= pp

        # Convert to GB
        weights_gb = sharded_weights / (1024**3)
        grad_gb = sharded_grad / (1024**3)
        opt_gb = sharded_opt / (1024**3)

        total_per_gpu = weights_gb + grad_gb + opt_gb

        # Strategy description
        parts = []
        if zero_stage > 0:
            parts.append(f"ZeRO-{zero_stage}")
        if tp > 1:
            parts.append(f"TP={tp}")
        if pp > 1:
            parts.append(f"PP={pp}")
        if dp > 1:
            parts.append(f"DP={dp}")
        strategy = " + ".join(parts) if parts else "single GPU"

        estimate = MemoryEstimate(
            weights_gb=round(weights_gb, 2),
            gradients_gb=round(grad_gb, 2),
            optimizer_states_gb=round(opt_gb, 2),
            activations_gb=0.0,  # filled by compute_activation_memory
            total_gb=round(total_per_gpu, 2),
            per_gpu_gb=round(total_per_gpu, 2),
            num_gpus=num_gpus,
            parallelism_strategy=strategy,
            precision=precision,
        )

        logger.debug("Memory estimate: %s", estimate)
        return estimate

    def compute_activation_memory(
        self,
        seq_len: int,
        batch_size: int,
        hidden_size: int,
        num_layers: int,
        tp_degree: int = 1,
        gradient_checkpointing: bool = False,
        num_attention_heads: int = 32,
    ) -> float:
        """Estimate activation memory in gigabytes.

        Activation memory for a transformer layer includes:
        - Attention scores: ``batch * heads * seq_len^2``
        - Intermediate activations: ``batch * seq_len * hidden * expansion``
        - Layer norm buffers, residuals

        With gradient checkpointing, only ~sqrt(L) activations are stored.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        batch_size : int
            Micro-batch size per GPU.
        hidden_size : int
            Model hidden dimension.
        num_layers : int
            Number of transformer layers.
        tp_degree : int
            Tensor parallelism degree (splits attention heads).
        gradient_checkpointing : bool
            Whether gradient checkpointing is enabled.
        num_attention_heads : int
            Number of attention heads.

        Returns
        -------
        float
            Estimated activation memory in GB.
        """
        # Attention activations per layer (bf16 = 2 bytes)
        heads_per_gpu = max(1, num_attention_heads // tp_degree)
        attn_bytes = 2.0 * batch_size * heads_per_gpu * seq_len * seq_len

        # MLP activations per layer: input + intermediate + output
        intermediate_size = hidden_size * 4  # typical expansion
        mlp_bytes = (
            2.0 * batch_size * seq_len * (hidden_size + intermediate_size / tp_degree + hidden_size)
        )

        # Residual + layer norm buffers
        residual_bytes = 2.0 * 2 * batch_size * seq_len * hidden_size

        per_layer_bytes = attn_bytes + mlp_bytes + residual_bytes

        # With gradient checkpointing: store activations for only ~sqrt(L) layers
        if gradient_checkpointing:
            effective_layers = math.ceil(math.sqrt(num_layers))
        else:
            effective_layers = num_layers

        total_bytes = per_layer_bytes * effective_layers
        activation_gb = total_bytes / (1024**3)

        logger.debug(
            "Activation memory: %.2f GB (seq=%d, batch=%d, hidden=%d, layers=%d, tp=%d, ckpt=%s)",
            activation_gb,
            seq_len,
            batch_size,
            hidden_size,
            num_layers,
            tp_degree,
            gradient_checkpointing,
        )

        return round(activation_gb, 2)

    def recommend_config(
        self,
        model_params: float,
        available_gpus: list[dict[str, Any]],
        precision: str = "bf16",
        optimizer: str = "adamw_torch",
        seq_len: int = 2048,
        batch_size: int = 4,
        lora_rank: int | None = None,
    ) -> ParallelismRecommendation:
        """Auto-recommend an optimal parallelism strategy.

        Uses a heuristic decision matrix based on model size and available
        hardware to select the best framework and configuration.

        Parameters
        ----------
        model_params : float
            Model size in billions of parameters.
        available_gpus : list[dict]
            List of GPU info dicts with ``"vram_gb"`` keys.
        precision : str
            Training precision.
        optimizer : str
            Optimizer name.
        seq_len : int
            Sequence length.
        batch_size : int
            Target micro-batch size.
        lora_rank : int, optional
            LoRA rank (if using LoRA).

        Returns
        -------
        ParallelismRecommendation
            Recommended configuration.
        """
        num_gpus = len(available_gpus) if available_gpus else 1
        vram_per_gpu = (
            min(g.get("vram_gb", 80.0) for g in available_gpus) if available_gpus else 80.0
        )

        # Normalise model params to billions
        model_params_b = model_params / 1000000000.0 if model_params >= 1000 else model_params

        rec = ParallelismRecommendation(
            framework="auto",
            strategy="auto",
            num_gpus=num_gpus,
        )

        # ---- Decision matrix ----

        if model_params_b <= 1.0:
            # Small model: single GPU or DDP
            rec.framework = "fsdp"
            rec.strategy = "NO_SHARD" if num_gpus <= 1 else "SHARD_GRAD_OP"
            rec.data_parallel = num_gpus
            rec.zero_stage = 0
            rec.notes.append("Small model (<= 1B): DDP or basic FSDP sufficient")

        elif model_params_b <= 8.0:
            # Medium model: FSDP FULL_SHARD
            rec.framework = "fsdp"
            rec.strategy = "FULL_SHARD"
            rec.data_parallel = num_gpus
            rec.zero_stage = 0
            rec.notes.append("Medium model (1-8B): FSDP with FULL_SHARD")

            # Check if QLoRA would fit on single GPU
            if lora_rank is not None:
                single_gpu_est = self.estimate_memory(
                    model_params_b,
                    precision="nf4",
                    optimizer=optimizer,
                    lora_rank=lora_rank,
                    num_gpus=1,
                )
                if single_gpu_est.per_gpu_gb < vram_per_gpu * 0.9:
                    rec.notes.append(
                        f"QLoRA fits on single GPU ({single_gpu_est.per_gpu_gb:.1f} GB)"
                    )

        elif model_params_b <= 20.0:
            # Large model: DeepSpeed ZeRO-3
            rec.framework = "deepspeed"
            rec.strategy = "ZeRO-3"
            rec.zero_stage = 3
            rec.data_parallel = num_gpus
            rec.notes.append("Large model (8-20B): DeepSpeed ZeRO-3")

            # Check if CPU offload is needed
            estimate = self.estimate_memory(
                model_params_b,
                precision,
                optimizer,
                num_gpus=num_gpus,
                zero_stage=3,
            )
            if estimate.per_gpu_gb > vram_per_gpu * 0.85:
                rec.cpu_offload = True
                rec.notes.append("CPU offloading recommended to fit in VRAM")

        elif model_params_b <= 70.0:
            # Very large model: ZeRO-3 + TP
            rec.framework = "deepspeed"
            rec.strategy = "ZeRO-3 + TP"
            rec.zero_stage = 3

            # TP across GPUs within a node (typically 8)
            tp = min(num_gpus, 8)
            rec.tensor_parallel = tp
            rec.data_parallel = max(1, num_gpus // tp)
            rec.notes.append("Very large model (20-70B): ZeRO-3 + Tensor Parallelism")

            # Check CPU offload
            estimate = self.estimate_memory(
                model_params_b,
                precision,
                optimizer,
                num_gpus=num_gpus,
                zero_stage=3,
                tensor_parallel=tp,
            )
            if estimate.per_gpu_gb > vram_per_gpu * 0.85:
                rec.cpu_offload = True
                rec.notes.append("CPU offloading recommended")

        else:
            # Massive model: Megatron-Core
            rec.framework = "megatron"
            rec.strategy = "3D Parallelism"

            # Determine TP, PP, DP
            tp = min(8, num_gpus)
            remaining = max(1, num_gpus // tp)
            pp = min(remaining, max(1, int(model_params_b / 35)))
            dp = max(1, num_gpus // (tp * pp))

            rec.tensor_parallel = tp
            rec.pipeline_parallel = pp
            rec.data_parallel = dp
            rec.notes.append(
                f"Massive model (>70B): Megatron 3D parallelism (TP={tp}, PP={pp}, DP={dp})"
            )

        # Compute final memory estimate
        final_estimate = self.estimate_memory(
            model_params_b,
            precision=precision,
            optimizer=optimizer,
            num_gpus=num_gpus,
            zero_stage=rec.zero_stage,
            tensor_parallel=rec.tensor_parallel,
            pipeline_parallel=rec.pipeline_parallel,
            lora_rank=lora_rank,
        )

        rec.estimated_per_gpu_gb = final_estimate.per_gpu_gb
        rec.fits_in_vram = final_estimate.per_gpu_gb < vram_per_gpu * 0.95

        if not rec.fits_in_vram:
            rec.notes.append(
                f"WARNING: estimated {final_estimate.per_gpu_gb:.1f} GB/GPU "
                f"exceeds available {vram_per_gpu:.1f} GB"
            )
            if not rec.cpu_offload:
                rec.cpu_offload = True
                rec.notes.append("Enabling CPU offloading as fallback")

        logger.info(
            "Recommendation: %s %s, TP=%d PP=%d DP=%d, ~%.1f GB/GPU, fits=%s",
            rec.framework,
            rec.strategy,
            rec.tensor_parallel,
            rec.pipeline_parallel,
            rec.data_parallel,
            rec.estimated_per_gpu_gb,
            rec.fits_in_vram,
        )

        return rec
