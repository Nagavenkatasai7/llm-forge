"""DeepSpeed configuration generator.

Generates DeepSpeed JSON configuration for ZeRO stages 0-3 with optional
CPU and NVMe offloading, mixed-precision settings, and communication
optimisations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.distributed.deepspeed_config")


def generate_deepspeed_config(
    config: Any,
    zero_stage: int | None = None,
    cpu_offload: bool | None = None,
    nvme_offload: bool = False,
    nvme_path: str = "/tmp/deepspeed_nvme",
    gradient_accumulation_steps: int | None = None,
    train_micro_batch_size: int | None = None,
    gradient_clipping: float | None = None,
    fp16_enabled: bool | None = None,
    bf16_enabled: bool | None = None,
    fp16_loss_scale: float = 0.0,
    fp16_loss_scale_window: int = 1000,
    fp16_initial_scale_power: int = 16,
    communication_overlap: bool = True,
    contiguous_gradients: bool = True,
    reduce_bucket_size: int = 500_000_000,
    allgather_bucket_size: int = 500_000_000,
    allgather_partitions: bool = True,
    reduce_scatter: bool = True,
    wall_clock_breakdown: bool = False,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Generate a DeepSpeed JSON configuration.

    Parameters
    ----------
    config : object
        An ``LLMForgeConfig`` instance.
    zero_stage : int, optional
        ZeRO optimisation stage (0-3).  Defaults to
        ``config.distributed.deepspeed_stage``.
    cpu_offload : bool, optional
        Offload optimizer states and/or parameters to CPU.
    nvme_offload : bool
        Offload to NVMe (ZeRO-3 Infinity).
    nvme_path : str
        File-system path for NVMe offloading.
    gradient_accumulation_steps : int, optional
        Gradient accumulation steps.
    train_micro_batch_size : int, optional
        Micro-batch size per GPU.
    gradient_clipping : float, optional
        Maximum gradient norm for clipping.
    fp16_enabled : bool, optional
        Enable FP16 mixed precision.
    bf16_enabled : bool, optional
        Enable BF16 mixed precision.
    fp16_loss_scale : float
        Loss scale for FP16 (0 = dynamic scaling).
    fp16_loss_scale_window : int
        Window for dynamic loss scaling.
    fp16_initial_scale_power : int
        Initial scale power for dynamic scaling.
    communication_overlap : bool
        Overlap communication with computation.
    contiguous_gradients : bool
        Copy gradients to a contiguous buffer.
    reduce_bucket_size : int
        Bucket size for reduce operations.
    allgather_bucket_size : int
        Bucket size for all-gather operations.
    allgather_partitions : bool
        All-gather partitions instead of pairwise exchanges.
    reduce_scatter : bool
        Use reduce-scatter instead of all-reduce.
    wall_clock_breakdown : bool
        Enable wall-clock time breakdown logging.
    save_path : str, optional
        Save the generated config to a JSON file.

    Returns
    -------
    dict[str, Any]
        Complete DeepSpeed configuration dictionary.
    """
    dist_cfg = config.distributed if hasattr(config, "distributed") else None
    training_cfg = config.training if hasattr(config, "training") else None

    # Resolve parameters from config
    stage = zero_stage
    if stage is None and dist_cfg is not None:
        stage = dist_cfg.deepspeed_stage
    stage = stage if stage is not None else 2

    if cpu_offload is None and dist_cfg is not None:
        cpu_offload = dist_cfg.deepspeed_offload
    cpu_offload = cpu_offload or False

    if gradient_accumulation_steps is None and training_cfg is not None:
        gradient_accumulation_steps = training_cfg.gradient_accumulation_steps
    gradient_accumulation_steps = gradient_accumulation_steps or 4

    if train_micro_batch_size is None and training_cfg is not None:
        train_micro_batch_size = training_cfg.per_device_train_batch_size
    train_micro_batch_size = train_micro_batch_size or 4

    if gradient_clipping is None and training_cfg is not None:
        gradient_clipping = training_cfg.max_grad_norm
    gradient_clipping = gradient_clipping or 1.0

    if fp16_enabled is None and bf16_enabled is None:
        if training_cfg is not None:
            bf16_enabled = training_cfg.bf16
            fp16_enabled = training_cfg.fp16
        else:
            bf16_enabled = True
            fp16_enabled = False

    # Ensure mutual exclusivity
    if bf16_enabled and fp16_enabled:
        fp16_enabled = False

    # Build config
    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": wall_clock_breakdown,
    }

    # ---- ZeRO optimisation -----------------------------------------------

    zero_config: dict[str, Any] = {
        "stage": stage,
        "contiguous_gradients": contiguous_gradients,
        "overlap_comm": communication_overlap,
        "reduce_scatter": reduce_scatter,
        "reduce_bucket_size": reduce_bucket_size,
        "allgather_bucket_size": allgather_bucket_size,
        "allgather_partitions": allgather_partitions,
    }

    if stage >= 1:
        # Stage 1: Partition optimizer states
        zero_config["stage1_reduce_bucket_size"] = reduce_bucket_size

    if stage >= 2:
        # Stage 2: + partition gradients
        zero_config["stage2_allgather_bucket_size"] = allgather_bucket_size

    if stage >= 3:
        # Stage 3: + partition parameters
        zero_config["stage3_max_live_parameters"] = 1_000_000_000
        zero_config["stage3_max_reuse_distance"] = 1_000_000_000
        zero_config["stage3_prefetch_bucket_size"] = 500_000_000
        zero_config["stage3_param_persistence_threshold"] = 100_000
        zero_config["stage3_gather_16bit_weights_on_model_save"] = True

    # ---- CPU / NVMe offloading -------------------------------------------

    if cpu_offload:
        if stage >= 2:
            zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": True,
            }
        if stage >= 3:
            zero_config["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 100_000_000,
                "max_in_cpu": 1_000_000_000,
            }

    if nvme_offload and stage >= 3:
        nvme_dir = Path(nvme_path)
        nvme_dir.mkdir(parents=True, exist_ok=True)
        zero_config["offload_optimizer"] = {
            "device": "nvme",
            "nvme_path": str(nvme_dir),
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": True,
        }
        zero_config["offload_param"] = {
            "device": "nvme",
            "nvme_path": str(nvme_dir),
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 100_000_000,
            "max_in_cpu": 1_000_000_000,
        }

    ds_config["zero_optimization"] = zero_config

    # ---- Mixed precision -------------------------------------------------

    if bf16_enabled:
        ds_config["bf16"] = {
            "enabled": True,
        }
        ds_config["fp16"] = {"enabled": False}
    elif fp16_enabled:
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": fp16_loss_scale,
            "loss_scale_window": fp16_loss_scale_window,
            "initial_scale_power": fp16_initial_scale_power,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
        ds_config["bf16"] = {"enabled": False}
    else:
        ds_config["fp16"] = {"enabled": False}
        ds_config["bf16"] = {"enabled": False}

    # ---- Optimizer (let HF Trainer manage it by default) ------------------
    # When integrating with HuggingFace Trainer, the optimizer is typically
    # configured through TrainingArguments.  We set "auto" to delegate.
    ds_config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    }

    ds_config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto",
        },
    }

    # ---- Communication ---------------------------------------------------

    ds_config["communication_data_type"] = "bf16" if bf16_enabled else "fp16"

    # ---- Activation checkpointing ----------------------------------------

    if training_cfg is not None and training_cfg.gradient_checkpointing:
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": cpu_offload,
            "number_checkpoints": None,  # checkpoint all layers
            "synchronize_checkpoint_boundary": False,
        }

    # ---- Save to file ----------------------------------------------------

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(ds_config, f, indent=2)
        logger.info("DeepSpeed config saved to %s", save_file)

    logger.info(
        "Generated DeepSpeed config: ZeRO-%d, cpu_offload=%s, nvme=%s, "
        "bf16=%s, fp16=%s, micro_batch=%d, grad_accum=%d",
        stage,
        cpu_offload,
        nvme_offload,
        bf16_enabled,
        fp16_enabled,
        train_micro_batch_size,
        gradient_accumulation_steps,
    )

    return ds_config
