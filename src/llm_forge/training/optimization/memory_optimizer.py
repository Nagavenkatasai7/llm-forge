"""Memory optimisation utilities for training.

Provides gradient checkpointing, CPU offloading, and automatic VRAM
optimisation to fit larger models or batch sizes within available memory.
"""

from __future__ import annotations

import gc
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.optimization.memory_optimizer")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import PreTrainedModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ============================================================================
# MemoryOptimizer
# ============================================================================


class MemoryOptimizer:
    """Memory optimisation manager for training pipelines.

    Provides methods to reduce GPU memory footprint through gradient
    checkpointing, CPU offloading, and automatic configuration based
    on available VRAM.
    """

    def enable_gradient_checkpointing(
        self,
        model: Any,
        use_reentrant: bool = False,
        selective_layers: list[int] | None = None,
    ) -> Any:
        """Enable gradient checkpointing (activation recomputation).

        Trades compute for memory by recomputing activations during the
        backward pass instead of storing them.  Reduces memory by up to
        60% at the cost of ~30% slower training.

        Parameters
        ----------
        model : PreTrainedModel
            Model to enable gradient checkpointing on.
        use_reentrant : bool
            Whether to use reentrant checkpointing.  ``False`` is
            recommended for compatibility with FSDP and LoRA.
        selective_layers : list[int], optional
            Layer indices to checkpoint.  ``None`` checkpoints all layers.

        Returns
        -------
        model
            The model with gradient checkpointing enabled.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available; skipping gradient checkpointing")
            return model

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )
            logger.info("Gradient checkpointing enabled (use_reentrant=%s)", use_reentrant)
        elif hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled (legacy API)")
        else:
            # Manual checkpointing for custom models
            self._apply_manual_checkpointing(model, selective_layers)

        # Ensure input embeddings require gradients (needed for checkpointing + LoRA)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()
            if embeddings is not None:

                def make_inputs_require_grad(module: nn.Module, input: Any, output: Any) -> None:
                    output.requires_grad_(True)

                embeddings.register_forward_hook(make_inputs_require_grad)

        return model

    def enable_cpu_offloading(
        self,
        model: Any,
        offload_optimizer: bool = True,
        offload_parameters: bool = False,
        pin_memory: bool = True,
    ) -> dict[str, Any]:
        """Configure CPU offloading for optimizer states and optionally parameters.

        This returns configuration settings rather than modifying the model
        directly, as CPU offloading is typically handled by the training
        framework (DeepSpeed, FSDP).

        Parameters
        ----------
        model : PreTrainedModel
            Model reference (used for memory estimation).
        offload_optimizer : bool
            Offload optimizer states (fp32 copies, momentum, variance) to CPU.
        offload_parameters : bool
            Offload model parameters to CPU (ZeRO-3 / FSDP with offload).
        pin_memory : bool
            Pin CPU memory for faster CPU-GPU transfers.

        Returns
        -------
        dict[str, Any]
            Offloading configuration for the training framework.
        """
        config: dict[str, Any] = {
            "offload_optimizer": offload_optimizer,
            "offload_parameters": offload_parameters,
            "pin_memory": pin_memory,
        }

        # Estimate memory savings
        if _TORCH_AVAILABLE and hasattr(model, "num_parameters"):
            total_params = model.num_parameters()
            trainable_params = model.num_parameters(only_trainable=True)

            # AdamW optimizer states: 12 bytes per trainable param
            opt_memory_gb = (trainable_params * 12) / (1024**3)
            # Parameters: 2 bytes per param (bf16)
            param_memory_gb = (total_params * 2) / (1024**3)

            saved_gb = 0.0
            if offload_optimizer:
                saved_gb += opt_memory_gb
            if offload_parameters:
                saved_gb += param_memory_gb

            config["estimated_gpu_savings_gb"] = round(saved_gb, 2)
            config["optimizer_memory_gb"] = round(opt_memory_gb, 2)
            config["parameter_memory_gb"] = round(param_memory_gb, 2)

            logger.info(
                "CPU offloading configured: optimizer=%s, params=%s, estimated GPU savings=%.2f GB",
                offload_optimizer,
                offload_parameters,
                saved_gb,
            )
        else:
            logger.info(
                "CPU offloading configured: optimizer=%s, params=%s",
                offload_optimizer,
                offload_parameters,
            )

        # DeepSpeed offloading config
        if offload_optimizer:
            config["deepspeed_offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": pin_memory,
                "buffer_count": 4,
                "fast_init": True,
            }
        if offload_parameters:
            config["deepspeed_offload_param"] = {
                "device": "cpu",
                "pin_memory": pin_memory,
                "buffer_count": 5,
                "buffer_size": 100_000_000,
                "max_in_cpu": 1_000_000_000,
            }

        return config

    def optimize_for_vram(
        self,
        model: Any,
        available_vram_gb: float,
        target_batch_size: int = 4,
        seq_len: int = 2048,
        precision: str = "bf16",
    ) -> dict[str, Any]:
        """Automatically select memory optimisations based on available VRAM.

        Applies a cascade of optimisations until the estimated memory
        usage fits within the available VRAM:

        1. Gradient checkpointing (saves ~40-60% activation memory)
        2. 8-bit optimizer (saves ~50% optimizer memory)
        3. Reduce batch size
        4. CPU offloading

        Parameters
        ----------
        model : PreTrainedModel
            Model to optimise.
        available_vram_gb : float
            Available GPU VRAM in gigabytes.
        target_batch_size : int
            Desired micro-batch size.
        seq_len : int
            Training sequence length.
        precision : str
            Training precision (``"bf16"``, ``"fp16"``, ``"fp32"``).

        Returns
        -------
        dict[str, Any]
            Applied optimisations and recommended settings.
        """
        optimisations: dict[str, Any] = {
            "gradient_checkpointing": False,
            "optimizer": "adamw_torch",
            "batch_size": target_batch_size,
            "cpu_offload": False,
            "precision": precision,
            "notes": [],
        }

        if not _TORCH_AVAILABLE:
            return optimisations

        # Estimate model memory
        param_count = 0
        trainable_count = 0
        if hasattr(model, "num_parameters"):
            param_count = model.num_parameters()
            trainable_count = model.num_parameters(only_trainable=True)
        elif hasattr(model, "parameters"):
            param_count = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "nf4": 0.5}
        bpp = bytes_per_param.get(precision, 2)

        # Model weights
        model_gb = (param_count * bpp) / (1024**3)

        # Optimizer states (AdamW: 12 bytes per trainable param)
        opt_gb = (trainable_count * 12) / (1024**3)

        # Gradients
        grad_gb = (trainable_count * bpp) / (1024**3)

        # Activation memory (rough estimate)
        hidden_size = 4096  # default estimate
        num_layers = 32
        if hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", 4096)
            num_layers = getattr(model.config, "num_hidden_layers", 32)

        act_per_layer = 2 * target_batch_size * seq_len * hidden_size * 4  # bytes
        act_gb = (act_per_layer * num_layers) / (1024**3)

        total_gb = model_gb + opt_gb + grad_gb + act_gb
        available_vram_gb - model_gb  # must fit rest in remaining

        logger.info(
            "Memory estimate: model=%.1f GB, optimizer=%.1f GB, "
            "gradients=%.1f GB, activations=%.1f GB, total=%.1f GB, "
            "available=%.1f GB",
            model_gb,
            opt_gb,
            grad_gb,
            act_gb,
            total_gb,
            available_vram_gb,
        )

        # Step 1: Gradient checkpointing
        if total_gb > available_vram_gb * 0.9:
            self.enable_gradient_checkpointing(model)
            optimisations["gradient_checkpointing"] = True
            act_gb *= 0.4  # ~60% reduction
            total_gb = model_gb + opt_gb + grad_gb + act_gb
            optimisations["notes"].append(
                "Enabled gradient checkpointing (activation memory reduced ~60%)"
            )

        # Step 2: 8-bit optimizer
        if total_gb > available_vram_gb * 0.9:
            optimisations["optimizer"] = "paged_adamw_8bit"
            opt_gb *= 0.5  # ~50% reduction
            total_gb = model_gb + opt_gb + grad_gb + act_gb
            optimisations["notes"].append(
                "Switched to 8-bit paged optimizer (optimizer memory reduced ~50%)"
            )

        # Step 3: Reduce batch size
        while total_gb > available_vram_gb * 0.9 and optimisations["batch_size"] > 1:
            optimisations["batch_size"] -= 1
            # Recalculate activation memory
            act_factor = optimisations["batch_size"] / target_batch_size
            current_act_gb = act_gb * act_factor
            total_gb = model_gb + opt_gb + grad_gb + current_act_gb
            optimisations["notes"].append(f"Reduced batch size to {optimisations['batch_size']}")

        # Step 4: CPU offloading
        if total_gb > available_vram_gb * 0.9:
            optimisations["cpu_offload"] = True
            total_gb -= opt_gb * 0.8  # offload most optimizer states
            optimisations["notes"].append("Enabled CPU offloading for optimizer states")

        optimisations["estimated_total_gb"] = round(total_gb, 2)
        optimisations["fits_in_vram"] = total_gb <= available_vram_gb

        if not optimisations["fits_in_vram"]:
            optimisations["notes"].append(
                f"WARNING: Still estimated {total_gb:.1f} GB vs {available_vram_gb:.1f} GB available. "
                "Consider using quantization (QLoRA) or more GPUs."
            )

        logger.info(
            "Memory optimisations applied: ckpt=%s, opt=%s, batch=%d, offload=%s, "
            "estimated=%.1f GB",
            optimisations["gradient_checkpointing"],
            optimisations["optimizer"],
            optimisations["batch_size"],
            optimisations["cpu_offload"],
            total_gb,
        )

        return optimisations

    def clear_memory(self) -> None:
        """Force garbage collection and clear GPU memory caches."""
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory caches cleared")

    def get_memory_stats(self) -> dict[str, float]:
        """Return current GPU memory usage statistics.

        Returns
        -------
        dict[str, float]
            Memory statistics in gigabytes for each GPU.
        """
        stats: dict[str, float] = {}

        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return stats

        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            peak = torch.cuda.max_memory_allocated(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)

            stats[f"gpu_{i}_allocated_gb"] = round(allocated, 3)
            stats[f"gpu_{i}_reserved_gb"] = round(reserved, 3)
            stats[f"gpu_{i}_peak_gb"] = round(peak, 3)
            stats[f"gpu_{i}_total_gb"] = round(total, 3)
            stats[f"gpu_{i}_free_gb"] = round(total - reserved, 3)

        return stats

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_manual_checkpointing(
        model: Any,
        selective_layers: list[int] | None = None,
    ) -> None:
        """Apply gradient checkpointing manually to transformer layers."""
        if not _TORCH_AVAILABLE:
            return

        from torch.utils.checkpoint import checkpoint

        checkpointed = 0
        for idx, (_name, module) in enumerate(model.named_modules()):
            # Look for decoder layers
            class_name = type(module).__name__
            is_decoder_layer = any(
                kw in class_name.lower() for kw in ["decoderlayer", "block", "transformerlayer"]
            )

            if not is_decoder_layer:
                continue

            if selective_layers is not None and idx not in selective_layers:
                continue

            # Wrap the forward method
            original_forward = module.forward

            def make_ckpt_forward(fwd: Any) -> Any:
                def ckpt_forward(*args: Any, **kwargs: Any) -> Any:
                    return checkpoint(fwd, *args, use_reentrant=False, **kwargs)

                return ckpt_forward

            module.forward = make_ckpt_forward(original_forward)
            checkpointed += 1

        if checkpointed > 0:
            logger.info("Manual gradient checkpointing applied to %d layers", checkpointed)
        else:
            logger.warning("No decoder layers found for manual checkpointing")
