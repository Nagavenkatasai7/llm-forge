"""Distributed training framework selection engine.

The :class:`DistributedOrchestrator` selects the optimal distributed
training framework (FSDP, DeepSpeed, or Megatron-Core) based on model
size, GPU count, VRAM, and generates the corresponding configuration.
"""

from __future__ import annotations

from typing import Any

from llm_forge.training.distributed.deepspeed_config import generate_deepspeed_config
from llm_forge.training.distributed.fsdp_config import generate_fsdp_config
from llm_forge.training.distributed.hardware_profiler import (
    HardwareProfiler,
    ParallelismRecommendation,
)
from llm_forge.training.distributed.megatron_config import generate_megatron_config
from llm_forge.utils.logging import get_logger

logger = get_logger("training.distributed.orchestrator")


# ============================================================================
# DistributedOrchestrator
# ============================================================================


class DistributedOrchestrator:
    """Framework selection engine for distributed training.

    Analyses model size and available hardware to select the optimal
    distributed training strategy, then generates the appropriate
    framework-specific configuration.

    Parameters
    ----------
    config : object
        An ``LLMForgeConfig`` instance.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.profiler = HardwareProfiler()
        self._recommendation: ParallelismRecommendation | None = None
        self._framework_config: dict[str, Any] | None = None

    # ================================================================== #
    # Framework selection
    # ================================================================== #

    def select_framework(
        self,
        model_size_b: float,
        num_gpus: int,
        gpu_type: str = "A100",
        vram_per_gpu: float = 80.0,
    ) -> str:
        """Select the optimal distributed training framework.

        Decision matrix:

        +------------------+------------+-------------------------------+
        | Model Size       | GPUs       | Recommendation                |
        +==================+============+===============================+
        | <= 1B            | any        | DDP / FSDP NO_SHARD           |
        +------------------+------------+-------------------------------+
        | 1B - 8B          | 1-8        | FSDP FULL_SHARD               |
        +------------------+------------+-------------------------------+
        | 8B - 20B         | any        | DeepSpeed ZeRO-3              |
        +------------------+------------+-------------------------------+
        | 20B - 70B        | 2+         | DeepSpeed ZeRO-3 + TP (+ PP)  |
        +------------------+------------+-------------------------------+
        | > 70B            | 8+         | Megatron-Core 3D parallelism  |
        +------------------+------------+-------------------------------+

        Parameters
        ----------
        model_size_b : float
            Model size in billions of parameters.
        num_gpus : int
            Total number of GPUs available.
        gpu_type : str
            GPU type string (e.g. ``"A100"``, ``"H100"``, ``"4090"``).
        vram_per_gpu : float
            Available VRAM per GPU in gigabytes.

        Returns
        -------
        str
            Selected framework: ``"fsdp"``, ``"deepspeed"``, or ``"megatron"``.
        """
        if model_size_b <= 1.0:
            framework = "fsdp"
            reason = "Small model (<= 1B): FSDP / DDP sufficient"
        elif model_size_b <= 8.0:
            framework = "fsdp"
            reason = "Medium model (1-8B): FSDP FULL_SHARD"
        elif model_size_b <= 20.0:
            framework = "deepspeed"
            reason = "Large model (8-20B): DeepSpeed ZeRO-3"
        elif model_size_b <= 70.0:
            framework = "deepspeed"
            reason = "Very large model (20-70B): DeepSpeed ZeRO-3 + TP"
        else:
            framework = "megatron"
            reason = "Massive model (> 70B): Megatron-Core 3D parallelism"

        # Override: if only 1 GPU and model fits, use FSDP/DDP
        if num_gpus == 1 and model_size_b <= 8.0:
            framework = "fsdp"
            reason = "Single GPU: using FSDP NO_SHARD (equivalent to DDP)"

        logger.info(
            "Framework selected: %s (model=%.1fB, gpus=%d, gpu=%s, vram=%.0fGB) - %s",
            framework,
            model_size_b,
            num_gpus,
            gpu_type,
            vram_per_gpu,
            reason,
        )

        return framework

    # ================================================================== #
    # Configuration generation
    # ================================================================== #

    def configure(
        self,
        config: Any | None = None,
        model_size_b: float | None = None,
    ) -> dict[str, Any]:
        """Generate framework-specific distributed training configuration.

        Automatically selects the best framework based on the model size
        and hardware, then generates the full configuration dictionary.

        Parameters
        ----------
        config : object, optional
            Override config.
        model_size_b : float, optional
            Model size in billions.  If ``None``, estimated from config.

        Returns
        -------
        dict[str, Any]
            Framework-specific configuration dictionary with keys:
            - ``"framework"`` -- selected framework name
            - ``"config"`` -- framework configuration dict
            - ``"recommendation"`` -- parallelism recommendation
        """
        cfg = config or self.config
        dist_cfg = cfg.distributed

        # Determine model size
        if model_size_b is None:
            model_size_b = self._estimate_model_size(cfg)

        # Get GPU info
        num_gpus = dist_cfg.num_gpus
        gpus = self._detect_gpus(num_gpus)
        vram_per_gpu = min(g.get("vram_gb", 80.0) for g in gpus) if gpus else 80.0
        gpu_type = gpus[0].get("name", "unknown") if gpus else "unknown"

        # Select framework
        if dist_cfg.framework == "auto":
            framework = self.select_framework(model_size_b, num_gpus, gpu_type, vram_per_gpu)
        else:
            framework = dist_cfg.framework

        # Get recommendation from hardware profiler
        self._recommendation = self.profiler.recommend_config(
            model_params=model_size_b,
            available_gpus=gpus,
            precision=str(cfg.model.torch_dtype),
            optimizer=cfg.training.optim,
            seq_len=cfg.model.max_seq_length,
            batch_size=cfg.training.per_device_train_batch_size,
        )

        # Generate framework config
        if framework == "fsdp":
            fw_config = generate_fsdp_config(cfg)
        elif framework == "deepspeed":
            fw_config = generate_deepspeed_config(cfg)
        elif framework == "megatron":
            fw_config = generate_megatron_config(cfg)
        else:
            raise ValueError(f"Unknown distributed framework: {framework}")

        self._framework_config = fw_config

        result = {
            "framework": framework,
            "config": fw_config,
            "recommendation": {
                "framework": self._recommendation.framework,
                "strategy": self._recommendation.strategy,
                "tensor_parallel": self._recommendation.tensor_parallel,
                "pipeline_parallel": self._recommendation.pipeline_parallel,
                "data_parallel": self._recommendation.data_parallel,
                "zero_stage": self._recommendation.zero_stage,
                "cpu_offload": self._recommendation.cpu_offload,
                "estimated_per_gpu_gb": self._recommendation.estimated_per_gpu_gb,
                "fits_in_vram": self._recommendation.fits_in_vram,
                "notes": self._recommendation.notes,
            },
        }

        logger.info(
            "Distributed config generated: framework=%s, model=%.1fB, gpus=%d",
            framework,
            model_size_b,
            num_gpus,
        )

        return result

    # ================================================================== #
    # Training arguments
    # ================================================================== #

    def get_training_args(
        self,
        config: Any | None = None,
    ) -> dict[str, Any]:
        """Return distributed training arguments for HuggingFace Trainer.

        Parameters
        ----------
        config : object, optional
            Override config.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to pass to ``TrainingArguments``.
        """
        cfg = config or self.config

        # Generate config if not already done
        if self._framework_config is None:
            self.configure(cfg)

        dist_cfg = cfg.distributed
        rec = self._recommendation
        fw_config = self._framework_config

        args: dict[str, Any] = {}

        # Determine the active framework
        framework = dist_cfg.framework
        if framework == "auto" and rec is not None:
            framework = rec.framework

        if framework == "fsdp" and fw_config is not None:
            # FSDP training arguments
            fsdp_flags = fw_config.get("hf_fsdp_flags", [])
            if fsdp_flags:
                args["fsdp"] = " ".join(fsdp_flags)

            fsdp_config_data = fw_config.get("accelerate_fsdp_config", {}).get("fsdp", {})
            if fsdp_config_data:
                args["fsdp_config"] = fsdp_config_data

        elif framework == "deepspeed" and fw_config is not None:
            # DeepSpeed: pass the entire config dict
            args["deepspeed"] = fw_config

        elif framework == "megatron":
            # Megatron is typically launched with its own scripts,
            # but we store the config for the launcher
            args["megatron_config"] = fw_config

        # Local rank / distributed launch settings
        if dist_cfg.num_gpus > 1:
            args["local_rank"] = -1  # let launcher set this
            args["ddp_find_unused_parameters"] = False

        logger.info(
            "Training args generated for framework=%s with %d GPU(s)",
            framework,
            dist_cfg.num_gpus,
        )

        return args

    # ================================================================== #
    # Private helpers
    # ================================================================== #

    @staticmethod
    def _estimate_model_size(config: Any) -> float:
        """Estimate model size in billions from config or model name."""
        import re

        model_name = config.model.name

        # Try to parse from name (e.g. "Llama-2-7b" -> 7.0)
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
        if match:
            return float(match.group(1))

        # Use gpu_utils for a more thorough lookup
        try:
            from llm_forge.utils.gpu_utils import _resolve_params_b

            return _resolve_params_b(model_name)
        except Exception:
            pass

        # Default to 7B
        return 7.0

    @staticmethod
    def _detect_gpus(num_gpus: int) -> list[dict[str, Any]]:
        """Detect available GPUs and return info dicts."""
        gpus: list[dict[str, Any]] = []

        try:
            import torch

            if torch.cuda.is_available():
                actual_count = torch.cuda.device_count()
                for i in range(min(num_gpus, actual_count)):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append(
                        {
                            "index": i,
                            "name": props.name,
                            "vram_gb": round(props.total_memory / (1024**3), 2),
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )
        except ImportError:
            pass

        # If no GPUs detected, provide placeholder entries
        if not gpus:
            for i in range(num_gpus):
                gpus.append(
                    {
                        "index": i,
                        "name": "unknown",
                        "vram_gb": 80.0,
                        "compute_capability": "8.0",
                    }
                )

        return gpus
