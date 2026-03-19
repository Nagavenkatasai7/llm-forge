"""Megatron-Core configuration generator.

Generates configuration dictionaries for Megatron-LM / Megatron-Core
3D parallelism: tensor parallelism (TP), pipeline parallelism (PP),
sequence parallelism (SP), and expert parallelism (EP).
"""

from __future__ import annotations

from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.distributed.megatron_config")


def generate_megatron_config(
    config: Any,
    tensor_parallel_size: int | None = None,
    pipeline_parallel_size: int | None = None,
    sequence_parallel: bool = True,
    expert_parallel_size: int = 1,
    context_parallel_size: int = 1,
    micro_batch_size: int | None = None,
    global_batch_size: int | None = None,
    num_layers: int | None = None,
    hidden_size: int | None = None,
    num_attention_heads: int | None = None,
    ffn_hidden_size: int | None = None,
    num_key_value_heads: int | None = None,
    seq_length: int | None = None,
    vocab_size: int = 32000,
    max_position_embeddings: int | None = None,
    use_flash_attention: bool = True,
    use_rotary_position_embeddings: bool = True,
    rotary_percent: float = 1.0,
    normalization: str = "RMSNorm",
    activation_function: str = "swiglu",
    use_distributed_optimizer: bool = True,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    fp16: bool | None = None,
    bf16: bool | None = None,
    accumulate_allreduce_grads_in_fp32: bool = True,
    initial_loss_scale: float = 4294967296.0,
    min_loss_scale: float = 1.0,
    loss_scale_window: int = 1000,
    recompute_method: str | None = "uniform",
    recompute_granularity: str | None = "selective",
    recompute_num_layers: int | None = None,
    data_path: str | None = None,
    tokenizer_type: str = "HuggingFaceTokenizer",
    tokenizer_model: str | None = None,
    save_interval: int = 1000,
    log_interval: int = 10,
    eval_interval: int = 500,
    eval_iters: int = 50,
    checkpoint_dir: str | None = None,
) -> dict[str, Any]:
    """Generate a Megatron-Core configuration dictionary.

    Megatron-Core uses a flat argument namespace.  This function produces
    a dictionary matching the Megatron command-line arguments that can be
    converted to CLI flags or passed programmatically.

    Parameters
    ----------
    config : object
        An ``LLMForgeConfig`` instance.
    tensor_parallel_size : int, optional
        Number of GPUs for tensor-model parallelism.  Splits attention
        heads and MLP columns across GPUs.
    pipeline_parallel_size : int, optional
        Number of pipeline stages.  Each stage holds a contiguous
        subset of transformer layers.
    sequence_parallel : bool
        Enable sequence parallelism (reduces activation memory by
        partitioning along the sequence dimension).
    expert_parallel_size : int
        Expert parallelism degree for MoE models.
    context_parallel_size : int
        Context parallelism degree for very long sequences.
    micro_batch_size : int, optional
        Per-GPU micro-batch size.
    global_batch_size : int, optional
        Global batch size across all GPUs and accumulation steps.
    num_layers : int, optional
        Number of transformer layers.
    hidden_size : int, optional
        Model hidden dimension.
    num_attention_heads : int, optional
        Number of attention heads.
    ffn_hidden_size : int, optional
        Feed-forward network hidden size.
    num_key_value_heads : int, optional
        Number of key-value heads (for GQA).
    seq_length : int, optional
        Training sequence length.
    vocab_size : int
        Vocabulary size.
    max_position_embeddings : int, optional
        Maximum position embeddings.
    use_flash_attention : bool
        Use FlashAttention kernels.
    use_rotary_position_embeddings : bool
        Use rotary position embeddings (RoPE).
    rotary_percent : float
        Fraction of head dim to apply rotary embeddings to.
    normalization : str
        Normalisation type: ``"RMSNorm"`` or ``"LayerNorm"``.
    activation_function : str
        Activation: ``"swiglu"``, ``"gelu"``, ``"relu"``.
    use_distributed_optimizer : bool
        Shard optimizer states across data-parallel ranks.
    overlap_grad_reduce : bool
        Overlap gradient reduction with backward pass.
    overlap_param_gather : bool
        Overlap parameter all-gather with forward pass.
    fp16 : bool, optional
        Enable FP16 training.
    bf16 : bool, optional
        Enable BF16 training.
    accumulate_allreduce_grads_in_fp32 : bool
        Accumulate gradients in FP32 before all-reduce.
    initial_loss_scale : float
        Initial loss scale for FP16 training.
    min_loss_scale : float
        Minimum loss scale.
    loss_scale_window : int
        Window size for dynamic loss scaling.
    recompute_method : str, optional
        Activation recomputation method: ``"uniform"`` or ``"block"``.
    recompute_granularity : str, optional
        Granularity: ``"selective"`` (recommended) or ``"full"``.
    recompute_num_layers : int, optional
        Number of layers to recompute (for ``block`` method).
    data_path : str, optional
        Path to Megatron-format data files.
    tokenizer_type : str
        Tokenizer type for Megatron.
    tokenizer_model : str, optional
        Path to tokenizer model file.
    save_interval : int
        Checkpoint save interval in iterations.
    log_interval : int
        Logging interval in iterations.
    eval_interval : int
        Evaluation interval in iterations.
    eval_iters : int
        Number of evaluation iterations.
    checkpoint_dir : str, optional
        Checkpoint save directory.

    Returns
    -------
    dict[str, Any]
        Megatron-Core configuration dictionary.
    """
    dist_cfg = config.distributed if hasattr(config, "distributed") else None
    training_cfg = config.training if hasattr(config, "training") else None
    model_cfg = config.model if hasattr(config, "model") else None

    # Resolve from config
    tp = tensor_parallel_size
    if tp is None and dist_cfg is not None:
        tp = dist_cfg.tensor_parallel_degree
    tp = tp or 1

    pp = pipeline_parallel_size
    if pp is None and dist_cfg is not None:
        pp = dist_cfg.pipeline_parallel_degree
    pp = pp or 1

    if micro_batch_size is None and training_cfg is not None:
        micro_batch_size = training_cfg.per_device_train_batch_size
    micro_batch_size = micro_batch_size or 1

    if seq_length is None and model_cfg is not None:
        seq_length = model_cfg.max_seq_length
    seq_length = seq_length or 2048

    if max_position_embeddings is None:
        max_position_embeddings = seq_length

    if fp16 is None and bf16 is None:
        if training_cfg is not None:
            bf16 = training_cfg.bf16
            fp16 = training_cfg.fp16
        else:
            bf16 = True
            fp16 = False

    if bf16 and fp16:
        fp16 = False

    # Calculate global batch size if not provided
    if global_batch_size is None:
        num_gpus = 1
        if dist_cfg is not None:
            num_gpus = dist_cfg.num_gpus
        dp = max(1, num_gpus // (tp * pp))
        grad_accum = 1
        if training_cfg is not None:
            grad_accum = training_cfg.gradient_accumulation_steps
        global_batch_size = micro_batch_size * dp * grad_accum

    # Build Megatron config
    megatron_config: dict[str, Any] = {}

    # ---- Parallelism -----------------------------------------------------

    megatron_config["tensor_model_parallel_size"] = tp
    megatron_config["pipeline_model_parallel_size"] = pp
    megatron_config["sequence_parallel"] = sequence_parallel and tp > 1
    megatron_config["expert_model_parallel_size"] = expert_parallel_size
    megatron_config["context_parallel_size"] = context_parallel_size

    # ---- Model architecture ----------------------------------------------

    if num_layers is not None:
        megatron_config["num_layers"] = num_layers
    if hidden_size is not None:
        megatron_config["hidden_size"] = hidden_size
    if num_attention_heads is not None:
        megatron_config["num_attention_heads"] = num_attention_heads
    if ffn_hidden_size is not None:
        megatron_config["ffn_hidden_size"] = ffn_hidden_size
    if num_key_value_heads is not None:
        megatron_config["num_query_groups"] = num_key_value_heads  # GQA

    megatron_config["seq_length"] = seq_length
    megatron_config["max_position_embeddings"] = max_position_embeddings
    megatron_config["vocab_size"] = vocab_size

    # Attention
    megatron_config["use_flash_attn"] = use_flash_attention
    megatron_config["use_rotary_position_embeddings"] = use_rotary_position_embeddings
    megatron_config["rotary_percent"] = rotary_percent

    # Normalisation and activation
    megatron_config["normalization"] = normalization
    if activation_function == "swiglu":
        megatron_config["swiglu"] = True
        megatron_config["activation_func"] = "silu"
    else:
        megatron_config["activation_func"] = activation_function

    megatron_config["untie_embeddings_and_output_weights"] = True
    megatron_config["position_embedding_type"] = "rope"

    # ---- Training --------------------------------------------------------

    megatron_config["micro_batch_size"] = micro_batch_size
    megatron_config["global_batch_size"] = global_batch_size

    # Learning rate (delegate to Megatron scheduler)
    if training_cfg is not None:
        megatron_config["lr"] = training_cfg.learning_rate
        megatron_config["weight_decay"] = training_cfg.weight_decay
        megatron_config["clip_grad"] = training_cfg.max_grad_norm

        # Warmup and decay
        megatron_config["lr_warmup_fraction"] = training_cfg.warmup_ratio
        megatron_config["lr_decay_style"] = "cosine"
        megatron_config["min_lr"] = training_cfg.learning_rate * 0.1

    # ---- Precision -------------------------------------------------------

    if bf16:
        megatron_config["bf16"] = True
        megatron_config["fp16"] = False
    elif fp16:
        megatron_config["fp16"] = True
        megatron_config["bf16"] = False
        megatron_config["initial_loss_scale"] = initial_loss_scale
        megatron_config["min_loss_scale"] = min_loss_scale
        megatron_config["loss_scale_window"] = loss_scale_window

    megatron_config["accumulate_allreduce_grads_in_fp32"] = accumulate_allreduce_grads_in_fp32

    # ---- Optimiser -------------------------------------------------------

    megatron_config["use_distributed_optimizer"] = use_distributed_optimizer
    megatron_config["overlap_grad_reduce"] = overlap_grad_reduce
    megatron_config["overlap_param_gather"] = overlap_param_gather

    megatron_config["optimizer"] = "adam"
    megatron_config["adam_beta1"] = 0.9
    megatron_config["adam_beta2"] = 0.95
    megatron_config["adam_eps"] = 1e-8

    # ---- Activation recomputation ----------------------------------------

    if recompute_granularity is not None:
        megatron_config["recompute_granularity"] = recompute_granularity
        if recompute_method is not None:
            megatron_config["recompute_method"] = recompute_method
        if recompute_num_layers is not None:
            megatron_config["recompute_num_layers"] = recompute_num_layers

    # ---- Data ------------------------------------------------------------

    if data_path is not None:
        megatron_config["data_path"] = data_path

    megatron_config["tokenizer_type"] = tokenizer_type
    if tokenizer_model is not None:
        megatron_config["tokenizer_model"] = tokenizer_model

    megatron_config["split"] = "98,1,1"  # train, val, test

    # ---- Checkpointing and logging ---------------------------------------

    save_dir = checkpoint_dir
    if save_dir is None and training_cfg is not None:
        save_dir = training_cfg.output_dir
    if save_dir is not None:
        megatron_config["save"] = save_dir
        megatron_config["load"] = save_dir

    megatron_config["save_interval"] = save_interval
    megatron_config["log_interval"] = log_interval
    megatron_config["eval_interval"] = eval_interval
    megatron_config["eval_iters"] = eval_iters

    # Tensorboard
    megatron_config["tensorboard_dir"] = (
        f"{save_dir}/tensorboard" if save_dir else "tensorboard_logs"
    )
    megatron_config["log_timers_to_tensorboard"] = True
    megatron_config["log_validation_ppl_to_tensorboard"] = True

    logger.info(
        "Generated Megatron config: TP=%d, PP=%d, SP=%s, micro_bs=%d, "
        "global_bs=%d, seq_len=%d, bf16=%s",
        tp,
        pp,
        sequence_parallel and tp > 1,
        micro_batch_size,
        global_batch_size,
        seq_length,
        bf16,
    )

    return megatron_config


def megatron_config_to_args(config_dict: dict[str, Any]) -> list[str]:
    """Convert a Megatron config dictionary to CLI argument list.

    Parameters
    ----------
    config_dict : dict
        Megatron configuration dictionary.

    Returns
    -------
    list[str]
        List of CLI-style arguments (e.g. ``["--tensor-model-parallel-size", "8"]``).
    """
    args: list[str] = []

    for key, value in config_dict.items():
        # Convert underscores to hyphens for CLI args
        flag = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                args.append(flag)
            else:
                # For explicitly false flags, use --no-<flag> where applicable
                no_flag = flag.replace("--", "--no-", 1)
                args.append(no_flag)
        elif isinstance(value, (int, float)):
            args.extend([flag, str(value)])
        elif isinstance(value, str):
            args.extend([flag, value])
        elif isinstance(value, list):
            args.extend([flag] + [str(v) for v in value])
        # Skip None values and nested dicts

    return args
