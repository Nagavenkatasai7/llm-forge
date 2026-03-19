"""FSDP (Fully Sharded Data Parallel) configuration generator.

Generates HuggingFace-compatible FSDP configuration dicts for use with
the ``accelerate`` library and ``TrainingArguments.fsdp``.
"""

from __future__ import annotations

from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.distributed.fsdp_config")


# ---------------------------------------------------------------------------
# FSDP sharding strategies
# ---------------------------------------------------------------------------

FSDP_STRATEGIES = {
    "FULL_SHARD": "FULL_SHARD",
    "SHARD_GRAD_OP": "SHARD_GRAD_OP",
    "NO_SHARD": "NO_SHARD",
    "HYBRID_SHARD": "HYBRID_SHARD",
}

# Transformer layer class names for auto-wrap policy
_DEFAULT_TRANSFORMER_LAYERS = [
    "LlamaDecoderLayer",
    "MistralDecoderLayer",
    "Phi3DecoderLayer",
    "GPT2Block",
    "GPTNeoXLayer",
    "BloomBlock",
    "FalconDecoderLayer",
    "GemmaDecoderLayer",
    "Qwen2DecoderLayer",
    "CohereDecoderLayer",
    "OPTDecoderLayer",
]


def generate_fsdp_config(
    config: Any,
    sharding_strategy: str | None = None,
    auto_wrap_policy: str = "TRANSFORMER_BASED_WRAP",
    transformer_layer_cls: list[str] | None = None,
    min_num_params: int = 1_000_000,
    backward_prefetch: str = "BACKWARD_PRE",
    forward_prefetch: bool = True,
    cpu_offload: bool = False,
    sync_module_states: bool = True,
    use_orig_params: bool = True,
    limit_all_gathers: bool = True,
    activation_checkpointing: bool = False,
    mixed_precision: str | None = None,
) -> dict[str, Any]:
    """Generate an FSDP configuration dictionary.

    The returned dictionary is compatible with HuggingFace ``TrainingArguments``
    and the ``accelerate`` FSDP plugin.

    Parameters
    ----------
    config : object
        An ``LLMForgeConfig`` instance.
    sharding_strategy : str, optional
        FSDP sharding strategy.  One of ``FULL_SHARD``, ``SHARD_GRAD_OP``,
        ``NO_SHARD``, ``HYBRID_SHARD``.  Defaults to
        ``config.distributed.fsdp_sharding_strategy``.
    auto_wrap_policy : str
        Wrap policy: ``TRANSFORMER_BASED_WRAP`` or ``SIZE_BASED_WRAP``.
    transformer_layer_cls : list[str], optional
        Transformer layer class names for the auto-wrap policy.
        Defaults to a list of common decoder layer classes.
    min_num_params : int
        Minimum parameter count for ``SIZE_BASED_WRAP``.
    backward_prefetch : str
        Backward prefetch strategy: ``BACKWARD_PRE`` or ``BACKWARD_POST``.
    forward_prefetch : bool
        Enable forward prefetching.
    cpu_offload : bool
        Offload parameters and gradients to CPU.
    sync_module_states : bool
        Synchronise module states across ranks at init.
    use_orig_params : bool
        Use original parameters (required for mixed precision and
        gradient checkpointing).
    limit_all_gathers : bool
        Limit the number of all-gathers in flight.
    activation_checkpointing : bool
        Enable FSDP activation checkpointing.
    mixed_precision : str, optional
        Mixed precision policy: ``"fp16"``, ``"bf16"``, or ``None``.
        Defaults to the training precision from config.

    Returns
    -------
    dict[str, Any]
        FSDP configuration dictionary.
    """
    dist_cfg = config.distributed if hasattr(config, "distributed") else None
    training_cfg = config.training if hasattr(config, "training") else None

    # Resolve sharding strategy
    strategy = sharding_strategy
    if strategy is None and dist_cfg is not None:
        strategy = dist_cfg.fsdp_sharding_strategy
    strategy = strategy or "FULL_SHARD"

    if strategy not in FSDP_STRATEGIES:
        raise ValueError(
            f"Unknown FSDP sharding strategy '{strategy}'. "
            f"Available: {list(FSDP_STRATEGIES.keys())}"
        )

    # Resolve mixed precision
    if mixed_precision is None and training_cfg is not None:
        if training_cfg.bf16:
            mixed_precision = "bf16"
        elif training_cfg.fp16:
            mixed_precision = "fp16"

    # Resolve activation checkpointing
    if not activation_checkpointing and training_cfg is not None:
        activation_checkpointing = training_cfg.gradient_checkpointing

    # Resolve CPU offload
    if not cpu_offload and dist_cfg is not None:
        cpu_offload = dist_cfg.deepspeed_offload

    # Build the FSDP config dict
    fsdp_config: dict[str, Any] = {
        "fsdp_sharding_strategy": FSDP_STRATEGIES[strategy],
        "fsdp_auto_wrap_policy": auto_wrap_policy,
        "fsdp_backward_prefetch": backward_prefetch,
        "fsdp_forward_prefetch": forward_prefetch,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_sync_module_states": sync_module_states,
        "fsdp_use_orig_params": use_orig_params,
        "fsdp_offload_params": cpu_offload,
    }

    # Transformer layer auto-wrap policy
    if auto_wrap_policy == "TRANSFORMER_BASED_WRAP":
        layer_cls = transformer_layer_cls or _DEFAULT_TRANSFORMER_LAYERS
        fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = layer_cls
    elif auto_wrap_policy == "SIZE_BASED_WRAP":
        fsdp_config["fsdp_min_num_params"] = min_num_params

    # Activation checkpointing
    if activation_checkpointing:
        fsdp_config["fsdp_activation_checkpointing"] = True

    # Build the accelerate fsdp_plugin-style config
    accelerate_config: dict[str, Any] = {
        "fsdp": {
            "sharding_strategy": strategy,
            "backward_prefetch_policy": backward_prefetch,
            "forward_prefetch": forward_prefetch,
            "auto_wrap_policy": auto_wrap_policy,
            "cpu_offload": cpu_offload,
            "sync_module_states": sync_module_states,
            "use_orig_params": use_orig_params,
            "limit_all_gathers": limit_all_gathers,
        }
    }

    if auto_wrap_policy == "TRANSFORMER_BASED_WRAP":
        accelerate_config["fsdp"]["transformer_layer_cls_to_wrap"] = (
            transformer_layer_cls or _DEFAULT_TRANSFORMER_LAYERS
        )
    else:
        accelerate_config["fsdp"]["min_num_params"] = min_num_params

    if mixed_precision:
        mp_config = {}
        if mixed_precision == "bf16":
            mp_config = {
                "param_dtype": "bfloat16",
                "reduce_dtype": "bfloat16",
                "buffer_dtype": "bfloat16",
            }
        elif mixed_precision == "fp16":
            mp_config = {
                "param_dtype": "float16",
                "reduce_dtype": "float16",
                "buffer_dtype": "float16",
            }
        accelerate_config["fsdp"]["mixed_precision_policy"] = mp_config

    fsdp_config["accelerate_fsdp_config"] = accelerate_config

    # HuggingFace TrainingArguments fsdp field (list of strings)
    fsdp_flags = [strategy.lower()]
    if auto_wrap_policy == "TRANSFORMER_BASED_WRAP":
        fsdp_flags.append("auto_wrap")
    if cpu_offload:
        fsdp_flags.append("offload")
    fsdp_config["hf_fsdp_flags"] = fsdp_flags

    logger.info(
        "Generated FSDP config: strategy=%s, auto_wrap=%s, offload=%s, "
        "mixed_precision=%s, activation_ckpt=%s",
        strategy,
        auto_wrap_policy,
        cpu_offload,
        mixed_precision,
        activation_checkpointing,
    )

    return fsdp_config
