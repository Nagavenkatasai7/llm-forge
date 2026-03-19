"""Translate high-level llm-forge configs to backend-specific formats.

Converts the unified ``LLMForgeConfig`` schema into the specific dictionaries
expected by HuggingFace ``TrainingArguments``, PEFT ``LoraConfig``,
``BitsAndBytesConfig``, and DeepSpeed JSON configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("pipeline.config_translator")


# ============================================================================
# ConfigTranslator
# ============================================================================


class ConfigTranslator:
    """Translate high-level user configuration to backend-specific configs.

    All methods are static or class methods -- instantiation is optional
    and exists only for organisational clarity.
    """

    # ------------------------------------------------------------------ #
    # HuggingFace TrainingArguments
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_training_args(config: Any) -> dict[str, Any]:
        """Generate a dict suitable for HuggingFace ``TrainingArguments``.

        Parameters
        ----------
        config : LLMForgeConfig
            The master configuration object.

        Returns
        -------
        dict
            Keyword arguments for ``transformers.TrainingArguments``.
        """
        t = config.training
        d = config.data

        args: dict[str, Any] = {
            "output_dir": t.output_dir,
            "num_train_epochs": t.num_epochs,
            "per_device_train_batch_size": t.per_device_train_batch_size,
            "per_device_eval_batch_size": t.per_device_eval_batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "learning_rate": t.learning_rate,
            "weight_decay": t.weight_decay,
            "lr_scheduler_type": t.lr_scheduler_type,
            "max_grad_norm": t.max_grad_norm,
            "logging_steps": t.logging_steps,
            "save_steps": t.save_steps,
            "save_total_limit": t.save_total_limit,
            "bf16": t.bf16,
            "fp16": t.fp16,
            "gradient_checkpointing": t.gradient_checkpointing,
            "optim": t.optim,
            "group_by_length": t.group_by_length,
            "report_to": t.report_to,
            "remove_unused_columns": False,
            "seed": d.seed,
            "dataloader_num_workers": d.num_workers,
        }

        # Evaluation strategy
        if hasattr(config, "evaluation") and config.evaluation.enabled:
            args["eval_strategy"] = t.eval_strategy
            if t.eval_steps is not None:
                args["eval_steps"] = t.eval_steps
            args["load_best_model_at_end"] = True
            args["metric_for_best_model"] = "eval_loss"
        else:
            args["eval_strategy"] = "no"

        # Warmup: prefer explicit warmup_steps over ratio
        if t.warmup_steps is not None:
            args["warmup_steps"] = t.warmup_steps
        else:
            args["warmup_ratio"] = t.warmup_ratio

        # Gradient checkpointing kwargs
        if t.gradient_checkpointing:
            args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        # NEFTune
        if t.neftune_noise_alpha is not None and t.neftune_noise_alpha > 0:
            args["neftune_noise_alpha"] = t.neftune_noise_alpha

        # Label smoothing
        if t.label_smoothing_factor > 0:
            args["label_smoothing_factor"] = t.label_smoothing_factor

        # Completion-only loss (mask prompt tokens)
        if t.completion_only_loss is not None:
            args["completion_only_loss"] = t.completion_only_loss

        # Resume from checkpoint
        if t.resume_from_checkpoint:
            args["resume_from_checkpoint"] = t.resume_from_checkpoint

        # Distributed settings
        if hasattr(config, "distributed") and config.distributed.enabled:
            dist = config.distributed
            if dist.framework == "fsdp":
                args["fsdp"] = dist.fsdp_sharding_strategy
                args["fsdp_config"] = {
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_backward_prefetch": "BACKWARD_PRE",
                    "fsdp_offload_params": False,
                    "fsdp_sharding_strategy": dist.fsdp_sharding_strategy,
                    "fsdp_state_dict_type": "FULL_STATE_DICT",
                }
            elif dist.framework == "deepspeed":
                ds_config = ConfigTranslator.to_deepspeed_config(config)
                args["deepspeed"] = ds_config

        return args

    # ------------------------------------------------------------------ #
    # PEFT LoRA config
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_lora_config(config: Any) -> dict[str, Any]:
        """Generate a dict suitable for PEFT ``LoraConfig``.

        Parameters
        ----------
        config : LLMForgeConfig
            The master configuration object.

        Returns
        -------
        dict
            Keyword arguments for ``peft.LoraConfig``.
        """
        lora = config.lora

        lora_dict: dict[str, Any] = {
            "r": lora.r,
            "lora_alpha": lora.alpha,
            "lora_dropout": lora.dropout,
            "target_modules": list(lora.target_modules),
            "bias": lora.bias,
            "task_type": lora.task_type,
        }

        if lora.use_rslora:
            lora_dict["use_rslora"] = True

        if lora.use_dora:
            lora_dict["use_dora"] = True

        return lora_dict

    # ------------------------------------------------------------------ #
    # BitsAndBytes config
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_bnb_config(config: Any) -> dict[str, Any]:
        """Generate a dict suitable for ``BitsAndBytesConfig``.

        Parameters
        ----------
        config : LLMForgeConfig
            The master configuration object.

        Returns
        -------
        dict
            Keyword arguments for ``transformers.BitsAndBytesConfig``.
            Returns an empty dict if quantization is not enabled.
        """
        q = config.quantization

        if not q.load_in_4bit and not q.load_in_8bit:
            return {}

        bnb_dict: dict[str, Any] = {}

        if q.load_in_4bit:
            bnb_dict["load_in_4bit"] = True
            bnb_dict["bnb_4bit_quant_type"] = q.bnb_4bit_quant_type
            bnb_dict["bnb_4bit_use_double_quant"] = q.bnb_4bit_use_double_quant

            # Map compute dtype string to torch dtype string
            compute_dtype_map = {
                "bf16": "bfloat16",
                "bfloat16": "bfloat16",
                "fp16": "float16",
                "float16": "float16",
                "fp32": "float32",
                "float32": "float32",
            }
            compute_dtype_str = str(q.bnb_4bit_compute_dtype)
            bnb_dict["bnb_4bit_compute_dtype"] = compute_dtype_map.get(
                compute_dtype_str, compute_dtype_str
            )

        elif q.load_in_8bit:
            bnb_dict["load_in_8bit"] = True

        return bnb_dict

    # ------------------------------------------------------------------ #
    # DeepSpeed config
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_deepspeed_config(config: Any) -> dict[str, Any]:
        """Generate a DeepSpeed JSON configuration dict.

        Parameters
        ----------
        config : LLMForgeConfig
            The master configuration object.

        Returns
        -------
        dict
            A complete DeepSpeed configuration ready to be serialised
            as JSON or passed directly to HuggingFace Trainer.
        """
        dist = config.distributed
        training = config.training

        stage = dist.deepspeed_stage

        # Effective batch size
        micro_bs = training.per_device_train_batch_size
        grad_accum = training.gradient_accumulation_steps
        effective_bs = micro_bs * grad_accum * dist.num_gpus

        ds_config: dict[str, Any] = {
            "train_batch_size": effective_bs,
            "train_micro_batch_size_per_gpu": micro_bs,
            "gradient_accumulation_steps": grad_accum,
            "gradient_clipping": training.max_grad_norm,
            "steps_per_print": training.logging_steps,
        }

        # Precision
        if training.bf16:
            ds_config["bf16"] = {"enabled": True}
        elif training.fp16:
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }

        # FP8 via Transformer Engine
        if dist.fp8_enabled:
            ds_config["fp8"] = {
                "enabled": True,
                "format": dist.fp8_format,
            }

        # ZeRO optimisation
        zero_config: dict[str, Any] = {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        }

        if stage >= 2:
            zero_config["allgather_bucket_size"] = 2e8
            zero_config["reduce_bucket_size"] = 2e8

        if stage >= 3:
            zero_config["stage3_prefetch_bucket_size"] = 5e7
            zero_config["stage3_param_persistence_threshold"] = 1e4
            zero_config["stage3_max_live_parameters"] = 1e9
            zero_config["stage3_max_reuse_distance"] = 1e9
            zero_config["stage3_gather_16bit_weights_on_model_save"] = True

        # CPU offloading
        if dist.deepspeed_offload:
            if stage >= 2:
                zero_config["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
            if stage >= 3:
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

        ds_config["zero_optimization"] = zero_config

        # Optimizer configuration
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": training.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training.weight_decay,
            },
        }

        # Scheduler configuration
        scheduler_type_map = {
            "cosine": "WarmupCosineLR",
            "linear": "WarmupDecayLR",
            "constant": "WarmupLR",
            "constant_with_warmup": "WarmupLR",
        }

        scheduler_type = scheduler_type_map.get(training.lr_scheduler_type, "WarmupDecayLR")

        # Estimate total steps for scheduler (approximate)
        warmup_steps = training.warmup_steps
        if warmup_steps is None:
            # Use warmup_ratio -- we do not know total steps here,
            # so we set a reasonable default that the trainer will override
            warmup_steps = 100

        ds_config["scheduler"] = {
            "type": scheduler_type,
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training.learning_rate,
                "warmup_num_steps": warmup_steps,
            },
        }

        # Activation checkpointing
        if training.gradient_checkpointing:
            ds_config["activation_checkpointing"] = {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": True,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
            }

        # Communication optimisations
        ds_config["wall_clock_breakdown"] = False
        ds_config["comms_logger"] = {"enabled": False}

        return ds_config

    # ------------------------------------------------------------------ #
    # Convenience: save DeepSpeed config to file
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_deepspeed_config(
        config: Any,
        output_path: str,
    ) -> Path:
        """Generate and save a DeepSpeed JSON config to disk.

        Parameters
        ----------
        config : LLMForgeConfig
            The master configuration.
        output_path : str
            File path for the JSON file.

        Returns
        -------
        Path
            The path to the saved file.
        """
        ds_config = ConfigTranslator.to_deepspeed_config(config)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(ds_config, f, indent=2)

        logger.info("DeepSpeed config saved to %s", out)
        return out

    # ------------------------------------------------------------------ #
    # Full translation: produce all backend configs at once
    # ------------------------------------------------------------------ #

    @staticmethod
    def translate_all(config: Any) -> dict[str, dict[str, Any]]:
        """Translate the full config into all backend-specific formats.

        Parameters
        ----------
        config : LLMForgeConfig
            The master configuration.

        Returns
        -------
        dict
            A mapping with keys ``"training_args"``, ``"lora_config"``,
            ``"bnb_config"``, and ``"deepspeed_config"`` (if applicable).
        """
        result: dict[str, dict[str, Any]] = {
            "training_args": ConfigTranslator.to_training_args(config),
            "lora_config": ConfigTranslator.to_lora_config(config),
            "bnb_config": ConfigTranslator.to_bnb_config(config),
        }

        if hasattr(config, "distributed") and config.distributed.enabled:
            if config.distributed.framework == "deepspeed":
                result["deepspeed_config"] = ConfigTranslator.to_deepspeed_config(config)

        return result
