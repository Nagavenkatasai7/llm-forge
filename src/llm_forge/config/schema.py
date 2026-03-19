"""Pydantic v2 configuration models for the llm-forge training platform.

This module defines the complete configuration schema used across
all llm-forge pipelines: data preparation, training, evaluation,
RAG, and serving.  Every field carries a ``Field`` description and
sensible defaults so that a minimal YAML file is enough to get started.
"""

from __future__ import annotations

import enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrainingMode(str, enum.Enum):
    """Supported fine-tuning / training strategies."""

    lora = "lora"
    qlora = "qlora"
    full = "full"
    pretrain = "pretrain"
    dpo = "dpo"
    orpo = "orpo"
    grpo = "grpo"


class DataFormat(str, enum.Enum):
    """Dataset conversation format."""

    alpaca = "alpaca"
    sharegpt = "sharegpt"
    completion = "completion"
    custom = "custom"


class PrecisionMode(str, enum.Enum):
    """Numeric precision for weights and compute."""

    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"
    fp8 = "fp8"
    int8 = "int8"
    int4 = "int4"


class DeduplicationTier(str, enum.Enum):
    """Deduplication strategy tiers, ordered by cost."""

    exact = "exact"
    fuzzy = "fuzzy"
    semantic = "semantic"


class QualityPreset(str, enum.Enum):
    """Data-cleaning strictness presets."""

    permissive = "permissive"
    balanced = "balanced"
    strict = "strict"


# ---------------------------------------------------------------------------
# Sub-config: Model
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Which pretrained model to load and how."""

    name: str = Field(
        ...,
        description="HuggingFace model name or local path (e.g. 'meta-llama/Llama-3.2-1B').",
    )
    revision: str | None = Field(
        default=None,
        description="Git revision (branch, tag, or commit SHA) of the model repo.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust and execute code shipped inside the model repo.",
    )
    torch_dtype: PrecisionMode = Field(
        default=PrecisionMode.bf16,
        description="Dtype used when loading model weights.",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=131072,
        description="Maximum sequence length (context window) for training.",
    )
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = Field(
        default="flash_attention_2",
        description="Attention kernel implementation.",
    )
    rope_scaling: dict[str, Any] | None = Field(
        default=None,
        description=("RoPE scaling configuration dict, e.g. {'type': 'dynamic', 'factor': 2.0}."),
    )


# ---------------------------------------------------------------------------
# Sub-config: LoRA
# ---------------------------------------------------------------------------


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation hyper-parameters (peft)."""

    r: int = Field(
        default=16,
        ge=1,
        le=256,
        description="LoRA rank – number of low-rank dimensions.",
    )
    alpha: int = Field(
        default=32,
        description="LoRA scaling factor (alpha / r is the effective scale).",
    )
    dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Dropout applied to LoRA layers.",
    )
    target_modules: list[str] | str = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Module name patterns to apply LoRA to.  "
        "Default targets attention layers only — safest for knowledge preservation.  "
        "Use 'all-linear' to include MLP layers (gate/up/down_proj + lm_head) "
        "for maximum adaptability, but this increases catastrophic forgetting risk.  "
        "Or provide an explicit list like ['q_proj', 'v_proj'].",
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Which biases to train alongside LoRA weights.",
    )
    task_type: Literal["CAUSAL_LM", "SEQ_2_SEQ_LM", "TOKEN_CLS", "SEQ_CLS"] = Field(
        default="CAUSAL_LM",
        description="PEFT task type.",
    )
    use_rslora: bool = Field(
        default=False,
        description="Enable Rank-Stabilized LoRA (RSLoRA) scaling.",
    )
    use_dora: bool = Field(
        default=False,
        description="Enable Weight-Decomposed Low-Rank Adaptation (DoRA).",
    )

    @field_validator("target_modules")
    @classmethod
    def _validate_target_modules(cls, v: list[str] | str) -> list[str] | str:
        """Ensure target_modules is not an empty list."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError(
                "target_modules cannot be an empty list — LoRA would have no "
                "modules to adapt. Use 'all-linear' or provide specific module "
                "names like ['q_proj', 'v_proj']."
            )
        return v


# ---------------------------------------------------------------------------
# Sub-config: Quantization
# ---------------------------------------------------------------------------


class QuantizationConfig(BaseModel):
    """BitsAndBytes quantization settings."""

    load_in_4bit: bool = Field(
        default=False,
        description="Load the model in 4-bit precision.",
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Load the model in 8-bit precision.",
    )
    bnb_4bit_compute_dtype: PrecisionMode = Field(
        default=PrecisionMode.bf16,
        description="Compute dtype when using 4-bit quantized weights.",
    )
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = Field(
        default="nf4",
        description="4-bit quantization data type (NF4 is generally preferred).",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Use nested/double quantization to save more memory.",
    )

    @field_validator("load_in_4bit", "load_in_8bit")
    @classmethod
    def _no_dual_quant(cls, v: bool, info: Any) -> bool:  # noqa: ANN401
        """Ensure 4-bit and 8-bit are not both enabled."""
        data = info.data
        if v is True:
            field = info.field_name
            other = "load_in_8bit" if field == "load_in_4bit" else "load_in_4bit"
            if data.get(other) is True:
                raise ValueError("Cannot enable both load_in_4bit and load_in_8bit simultaneously.")
        return v


# ---------------------------------------------------------------------------
# Sub-config: Data Cleaning
# ---------------------------------------------------------------------------


class DataCleaningConfig(BaseModel):
    """Controls for the data-cleaning pipeline."""

    enabled: bool = Field(
        default=True,
        description="Master switch for data cleaning.",
    )
    quality_preset: QualityPreset = Field(
        default=QualityPreset.balanced,
        description="Overall quality-filtering strictness.",
    )

    # --- text normalisation ---
    unicode_fix: bool = Field(
        default=True,
        description="Apply ftfy unicode fixing (mojibake repair, etc.).",
    )

    # --- language filtering ---
    language_filter: list[str] | None = Field(
        default=None,
        description="Keep only texts detected as these ISO-639 codes, e.g. ['en'].",
    )
    language_confidence_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum fasttext confidence to accept language detection.",
    )

    # --- heuristic quality filtering ---
    heuristic_filter: bool = Field(
        default=True,
        description="Enable rule-based quality heuristics (word count, symbol ratio, etc.).",
    )
    min_word_count: int = Field(
        default=5,
        ge=0,
        description="Discard samples with fewer words than this.",
    )
    max_word_count: int = Field(
        default=100_000,
        ge=1,
        description="Discard samples with more words than this.",
    )
    min_char_count: int = Field(
        default=20,
        ge=0,
        description="Discard samples shorter than this many characters.",
    )
    max_char_count: int = Field(
        default=5_000_000,
        ge=1,
        description="Discard samples longer than this many characters.",
    )
    alpha_ratio_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of alphabetic characters.",
    )
    symbol_to_word_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Maximum ratio of symbols (#, ...) to total words.",
    )
    max_duplicate_line_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of lines that are duplicates within one sample.",
    )
    max_duplicate_para_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of paragraphs that are duplicates within one sample.",
    )

    # --- toxicity ---
    toxicity_filter: bool = Field(
        default=False,
        description="Enable toxicity scoring (requires detoxify).",
    )
    toxicity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Drop samples whose toxicity score exceeds this threshold.",
    )

    # --- PII ---
    pii_redaction: bool = Field(
        default=False,
        description="Enable PII detection / redaction (requires presidio).",
    )
    pii_entities: list[str] = Field(
        default=[
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "US_SSN",
            "IP_ADDRESS",
        ],
        description="Named entity types to redact.",
    )

    # --- deduplication ---
    dedup_enabled: bool = Field(
        default=True,
        description="Enable cross-document deduplication.",
    )
    dedup_tiers: list[DeduplicationTier] = Field(
        default=[DeduplicationTier.exact, DeduplicationTier.fuzzy],
        description="Deduplication strategies to apply in order.",
    )
    dedup_jaccard_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Jaccard similarity threshold for fuzzy dedup (MinHash).",
    )
    dedup_num_perm: int = Field(
        default=128,
        ge=16,
        description="Number of permutations for MinHash.",
    )
    dedup_shingle_size: int = Field(
        default=5,
        ge=1,
        description="Shingle (n-gram) size for MinHash.",
    )
    semantic_dedup_enabled: bool = Field(
        default=False,
        description="Enable embedding-based semantic deduplication.",
    )
    semantic_dedup_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for semantic dedup.",
    )
    semantic_dedup_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for semantic dedup.",
    )

    @model_validator(mode="after")
    def _validate_word_char_bounds(self) -> DataCleaningConfig:
        if self.min_word_count > self.max_word_count:
            raise ValueError(
                f"min_word_count ({self.min_word_count}) must be <= "
                f"max_word_count ({self.max_word_count})"
            )
        if self.min_char_count > self.max_char_count:
            raise ValueError(
                f"min_char_count ({self.min_char_count}) must be <= "
                f"max_char_count ({self.max_char_count})"
            )
        return self


# ---------------------------------------------------------------------------
# Sub-config: Data
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """Dataset loading, formatting, and cleaning configuration."""

    train_path: str = Field(
        ...,
        description=(
            "Path or HuggingFace dataset identifier for training data "
            "(e.g. 'data/train.jsonl' or 'tatsu-lab/alpaca')."
        ),
    )
    eval_path: str | None = Field(
        default=None,
        description="Path or HF identifier for evaluation data.  "
        "If omitted, a split is created from train_path using test_size.",
    )
    format: DataFormat = Field(
        default=DataFormat.alpaca,
        description="Conversation / prompt format of the dataset.",
    )
    input_field: str = Field(
        default="instruction",
        description="Column name containing the user instruction / input.",
    )
    output_field: str = Field(
        default="output",
        description="Column name containing the expected model output.",
    )
    context_field: str | None = Field(
        default="input",
        description="Column name for optional context / input supplement.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt prepended to every sample.",
    )
    max_samples: int | None = Field(
        default=None,
        ge=1,
        description="Cap the number of training samples (useful for debugging).",
    )
    test_size: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Fraction of training data held out for evaluation when eval_path is None.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible dataset splitting / shuffling.",
    )
    streaming: bool = Field(
        default=False,
        description="Stream the dataset instead of loading it entirely into RAM.",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of data-loader worker processes.",
    )
    cleaning: DataCleaningConfig = Field(
        default_factory=DataCleaningConfig,
        description="Data-cleaning sub-configuration.",
    )

    @model_validator(mode="after")
    def _validate_paths(self) -> DataConfig:
        """Warn if local file paths don't exist on disk.

        HuggingFace dataset IDs (e.g. ``tatsu-lab/alpaca``) look like
        paths but should not be validated as local files.  We only warn
        for values that clearly reference the local filesystem: starting
        with ``./``, ``/``, ``~``, or containing no ``/`` at all but
        having a file-like extension.
        """
        import warnings
        from pathlib import Path

        for field_name in ("train_path", "eval_path"):
            value = getattr(self, field_name)
            if value is None:
                continue
            # Skip URLs
            if value.startswith(("http://", "https://", "hf://")):
                continue
            # Heuristic: if the first path component contains no "/",
            # it could be a HF dataset ID like "tatsu-lab/alpaca".
            # Only warn for paths that look local.
            _looks_local = value.startswith((".", "/", "~")) or (
                "/" not in value and "." in value.rsplit("/", 1)[-1]
            )
            if not _looks_local:
                continue
            p = Path(value).expanduser()
            if not p.exists():
                warnings.warn(
                    f"{field_name} '{value}' not found at {p.resolve()}. "
                    f"Training will fail unless this is a HuggingFace dataset ID.",
                    UserWarning,
                    stacklevel=2,
                )
        return self


# ---------------------------------------------------------------------------
# Sub-config: Training
# ---------------------------------------------------------------------------


class TrainingConfig(BaseModel):
    """Core training hyper-parameters (maps mostly to HF TrainingArguments)."""

    mode: TrainingMode = Field(
        default=TrainingMode.lora,
        description="Training strategy.",
    )
    output_dir: str = Field(
        default="outputs",
        description="Directory for checkpoints, logs, and final artefacts.",
    )

    @field_validator("output_dir")
    @classmethod
    def _expand_output_dir(cls, v: str) -> str:
        """Expand $HOME, ~ and other env vars in output_dir."""
        import os

        return os.path.expandvars(os.path.expanduser(v))

    num_epochs: int = Field(
        default=1,
        ge=1,
        description="Total number of training epochs.  For LoRA instruction tuning "
        "on <50K samples, 1 epoch is usually sufficient and prevents overfitting.",
    )
    per_device_train_batch_size: int = Field(
        default=4,
        ge=1,
        description="Micro-batch size per GPU for training.",
    )
    per_device_eval_batch_size: int = Field(
        default=4,
        ge=1,
        description="Micro-batch size per GPU for evaluation.",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Number of micro-batches accumulated before a gradient step.",
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0.0,
        description="Peak learning rate.  For LoRA on Instruct models, 1e-5 to 5e-5 "
        "is recommended.  Higher values (>1e-4) risk catastrophic forgetting.",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="L2 weight decay coefficient.",
    )
    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Fraction of total steps used for linear warmup.  "
        "Ignored when warmup_steps is set.",
    )
    warmup_steps: int | None = Field(
        default=None,
        ge=0,
        description="Exact number of warmup steps (overrides warmup_ratio).",
    )
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "reduce_on_plateau",
    ] = Field(
        default="cosine",
        description="Learning-rate scheduler type.",
    )
    max_grad_norm: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum gradient norm for clipping.",
    )
    logging_steps: int = Field(
        default=10,
        ge=1,
        description="Log training metrics every N steps.",
    )
    eval_steps: int | None = Field(
        default=None,
        ge=1,
        description="Run evaluation every N steps.  None = every epoch.",
    )
    eval_strategy: Literal["no", "steps", "epoch"] = Field(
        default="epoch",
        description="When to run evaluation.",
    )
    save_steps: int = Field(
        default=500,
        ge=1,
        description="Save a checkpoint every N steps.",
    )
    save_total_limit: int = Field(
        default=3,
        ge=1,
        description="Maximum number of checkpoints to keep.",
    )
    bf16: bool = Field(
        default=True,
        description="Enable bfloat16 mixed precision training.",
    )
    fp16: bool = Field(
        default=False,
        description="Enable float16 mixed precision training.",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Trade compute for memory by re-computing activations.",
    )
    optim: str = Field(
        default="adamw_torch",
        description="Optimizer name (e.g. 'adamw_torch', 'adamw_8bit', 'paged_adamw_32bit').",
    )
    group_by_length: bool = Field(
        default=True,
        description="Group samples of similar length to reduce padding waste.",
    )
    report_to: list[str] = Field(
        default=["wandb"],
        description="Experiment trackers to report to.",
    )
    resume_from_checkpoint: str | None = Field(
        default=None,
        description="Path to checkpoint directory to resume from.",
    )
    neftune_noise_alpha: float | None = Field(
        default=None,
        ge=0.0,
        description="NEFTune noise alpha for regularisation.  "
        "Set to null/None to disable (default).  Use 5.0 for datasets >50K samples.  "
        "On small datasets (<10K), NEFTune noise amplifies forgetting.  "
        "Ref: arxiv:2310.05914.",
    )
    label_smoothing_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Label-smoothing coefficient.  Set to 0.1 for large datasets "
        "(>50K samples) to reduce overconfidence.  On small datasets, label smoothing "
        "can degrade MC-style benchmark performance.  Default 0.0 (disabled).",
    )
    completion_only_loss: bool | None = Field(
        default=True,
        description="Compute loss on completion tokens only (for prompt-completion datasets).  "
        "True = mask prompt tokens.  False = loss on full sequence.  "
        "None = auto-detect from dataset format.",
    )
    assistant_only_loss: bool = Field(
        default=True,
        description="Compute loss on assistant response tokens only (for conversational/messages "
        "datasets).  When True, TRL uses the chat template to identify assistant turns and "
        "masks all system/user tokens with label=-100.  This is THE critical setting for "
        "instruction tuning — prevents system-prompt regurgitation and loss inflation.  "
        "Requires dataset to have a 'messages' column with role/content dicts.",
    )
    average_tokens_across_devices: bool = Field(
        default=True,
        description="Synchronise token counts across GPUs for correct gradient-accumulation "
        "loss scaling.  Requires transformers>=4.46.  "
        "Ref: https://github.com/huggingface/transformers/issues/34242",
    )
    use_unsloth: bool = Field(
        default=False,
        description="Use Unsloth accelerated kernels when available.",
    )
    pack_sequences: bool = Field(
        default=False,
        description="Pack multiple short sequences into max_seq_length to reduce padding waste. "
        "Uses TRL's built-in PackedDataset with attention masking to prevent "
        "cross-sample attention leakage.  Improves throughput 2-4x on variable-length "
        "instruction-tuning datasets.  Not recommended for long-form or chat data.",
    )

    @field_validator("bf16", "fp16")
    @classmethod
    def _no_dual_precision(cls, v: bool, info: Any) -> bool:  # noqa: ANN401
        data = info.data
        if v is True:
            field = info.field_name
            other = "fp16" if field == "bf16" else "bf16"
            if data.get(other) is True:
                raise ValueError("Cannot enable both bf16 and fp16 simultaneously.")
        return v

    @model_validator(mode="after")
    def _validate_training_cross_fields(self) -> TrainingConfig:
        """Validate cross-field consistency in training config."""
        # eval_steps requires eval_strategy="steps"
        if self.eval_steps is not None and self.eval_strategy == "no":
            raise ValueError(
                f"eval_steps={self.eval_steps} has no effect when eval_strategy='no'. "
                "Set eval_strategy='steps' to enable step-based evaluation, "
                "or remove eval_steps."
            )

        # warmup_ratio and warmup_steps shouldn't both be non-default
        if self.warmup_steps is not None and self.warmup_steps > 0 and self.warmup_ratio > 0:
            import warnings

            warnings.warn(
                f"Both warmup_steps={self.warmup_steps} and warmup_ratio={self.warmup_ratio} "
                f"are set. warmup_steps takes precedence; warmup_ratio will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Guard against dangerously high learning rates
        if self.learning_rate > 1e-3:
            import warnings

            warnings.warn(
                f"learning_rate={self.learning_rate:.2e} is very high. "
                f"For fine-tuning, values above 1e-3 almost always cause "
                f"catastrophic forgetting. Typical range: 1e-5 to 5e-5.",
                UserWarning,
                stacklevel=2,
            )

        return self


# ---------------------------------------------------------------------------
# Sub-config: Distributed
# ---------------------------------------------------------------------------


class DistributedConfig(BaseModel):
    """Multi-GPU / multi-node distributed training settings."""

    enabled: bool = Field(
        default=False,
        description="Enable distributed training.",
    )
    framework: Literal["auto", "fsdp", "deepspeed", "megatron"] = Field(
        default="auto",
        description="Distributed framework to use.",
    )
    num_gpus: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs to use.",
    )
    num_nodes: int = Field(
        default=1,
        ge=1,
        description="Number of nodes in the cluster.",
    )
    fsdp_sharding_strategy: Literal[
        "FULL_SHARD",
        "SHARD_GRAD_OP",
        "NO_SHARD",
        "HYBRID_SHARD",
    ] = Field(
        default="FULL_SHARD",
        description="FSDP sharding strategy.",
    )
    deepspeed_stage: Literal[0, 1, 2, 3] = Field(
        default=2,
        description="DeepSpeed ZeRO stage.",
    )
    deepspeed_offload: bool = Field(
        default=False,
        description="Offload optimizer / parameters to CPU (ZeRO-Offload).",
    )
    tensor_parallel_degree: int = Field(
        default=1,
        ge=1,
        description="Tensor-parallelism degree (Megatron-LM).",
    )
    pipeline_parallel_degree: int = Field(
        default=1,
        ge=1,
        description="Pipeline-parallelism degree (Megatron-LM).",
    )
    fp8_enabled: bool = Field(
        default=False,
        description="Enable FP8 compute via Transformer Engine.",
    )
    fp8_format: Literal["E4M3", "HYBRID"] = Field(
        default="HYBRID",
        description="FP8 format: E4M3 for forward, E5M2 for backward (HYBRID).",
    )
    auto_micro_batch: bool = Field(
        default=False,
        description="Automatically find the largest micro-batch that fits in VRAM.",
    )


# ---------------------------------------------------------------------------
# Sub-config: Evaluation
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    """Post-training evaluation configuration."""

    enabled: bool = Field(
        default=True,
        description="Run benchmarks after training.",
    )
    benchmarks: list[str] = Field(
        default=["hellaswag", "arc_easy", "mmlu", "truthfulqa_mc2", "ifeval"],
        description="lm-eval benchmark task names.",
    )
    custom_eval_path: str | None = Field(
        default=None,
        description="Path to a custom evaluation script or dataset.",
    )
    num_fewshot: int = Field(
        default=0,
        ge=0,
        description="Number of few-shot examples for benchmarks.",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for evaluation inference.",
    )
    generate_report: bool = Field(
        default=True,
        description="Generate an HTML / Markdown evaluation report.",
    )
    regression_check: bool = Field(
        default=True,
        description="Compare fine-tuned model against base model and warn "
        "if performance degrades on any benchmark.",
    )
    regression_threshold: float = Field(
        default=-0.02,
        le=0.0,
        description="Maximum acceptable score drop (negative) per benchmark.  "
        "Scores dropping below this trigger a regression warning.",
    )
    llm_judge: bool = Field(
        default=False,
        description="Run LLM-as-Judge evaluation on generated outputs.",
    )
    judge_model: str | None = Field(
        default=None,
        description="Model path or HuggingFace ID for the judge model.  "
        "If None, uses the trained model itself as judge.",
    )
    judge_criteria: list[str] = Field(
        default=["helpfulness", "coherence"],
        description="Criteria for LLM-as-Judge evaluation.",
    )
    judge_samples: int = Field(
        default=50,
        ge=1,
        description="Number of samples to evaluate with LLM-as-Judge.",
    )
    retention_probes: bool = Field(
        default=False,
        description="Run knowledge retention probes (100 factual questions) to "
        "detect catastrophic forgetting after fine-tuning.",
    )
    retention_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable retention rate.  If the fine-tuned model "
        "forgets more than (1 - threshold) of the base model's correct answers, "
        "a retention warning is raised.",
    )


# ---------------------------------------------------------------------------
# Sub-config: RAG
# ---------------------------------------------------------------------------


class RAGConfig(BaseModel):
    """Retrieval-Augmented Generation pipeline settings."""

    enabled: bool = Field(
        default=False,
        description="Enable the RAG pipeline.",
    )
    knowledge_base_path: str | None = Field(
        default=None,
        description="Path to the knowledge-base directory (PDFs, text, etc.).",
    )
    chunk_strategy: Literal["fixed", "recursive", "semantic", "sentence"] = Field(
        default="recursive",
        description="Text chunking strategy.",
    )
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Target chunk size in tokens.",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Overlap between consecutive chunks in tokens.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model used for vectorisation.",
    )
    vectorstore: Literal["chromadb", "faiss", "qdrant", "weaviate"] = Field(
        default="chromadb",
        description="Vector store backend.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of chunks to retrieve per query.",
    )
    reranker_model: str | None = Field(
        default=None,
        description="Cross-encoder model for re-ranking retrieved chunks.",
    )
    hybrid_search: bool = Field(
        default=False,
        description="Combine dense retrieval with BM25 sparse retrieval.",
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to keep a retrieved chunk.",
    )

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> RAGConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})."
            )
        return self


# ---------------------------------------------------------------------------
# Sub-config: Serving
# ---------------------------------------------------------------------------


class ServingConfig(BaseModel):
    """Model serving / export settings."""

    backend: Literal["gradio", "fastapi", "vllm"] = Field(
        default="gradio",
        description="Serving backend.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the server to.",
    )
    port: int = Field(
        default=7860,
        ge=1,
        le=65535,
        description="Port number for the server.",
    )
    export_format: Literal["gguf", "onnx", "safetensors", "awq", "gptq"] | None = Field(
        default=None,
        description="Export the model to an alternative format after training.",
    )
    gguf_quantization: str | None = Field(
        default=None,
        description="GGUF quantization level (e.g. 'Q4_K_M', 'Q5_K_S').",
    )
    merge_adapter: bool = Field(
        default=True,
        description="Merge LoRA adapter into base model before serving / export.",
    )

    # -- Ollama / Modelfile generation --
    generate_modelfile: bool = Field(
        default=True,
        description="Auto-generate an Ollama Modelfile alongside GGUF export.  "
        "Only used when export_format='gguf'.",
    )
    ollama_system_prompt: str | None = Field(
        default=None,
        description="System prompt for the Ollama Modelfile.  "
        "If None, uses data.system_prompt from the training config.",
    )

    # -- Inference parameters (written into Modelfile) --
    inference_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for inference.  Small models (<3B) need "
        "low temperatures (0.1-0.3) to avoid degenerate outputs.  "
        "Higher values (0.7+) risk incoherent text on small models.",
    )
    inference_top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling threshold.",
    )
    inference_top_k: int = Field(
        default=40,
        ge=0,
        description="Top-k sampling limit.  0 = disabled.",
    )
    inference_repeat_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Repetition penalty.  Values above 1.2 can cause "
        "empty list items and awkward word choices on small models.",
    )
    inference_num_predict: int = Field(
        default=256,
        ge=32,
        le=4096,
        description="Maximum tokens to generate per response.  "
        "Small models (<3B) degrade on long outputs — keep this "
        "at 256 or lower for best quality.",
    )
    inference_num_ctx: int = Field(
        default=2048,
        ge=512,
        le=131072,
        description="KV-cache context window for Ollama inference.  "
        "Should match or slightly exceed the training max_seq_length.  "
        "Controls how many tokens of conversation history the model sees.",
    )


# ---------------------------------------------------------------------------
# Sub-config: Mac Optimisation
# ---------------------------------------------------------------------------


class MacConfig(BaseModel):
    """MacOS / Apple Silicon training optimisations."""

    smart_memory: bool = Field(
        default=True,
        description="Enable smart memory management: auto-reduce batch size on "
        "memory pressure, OOM recovery, and memory usage reporting.",
    )
    memory_pressure_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=0.99,
        description="System RAM usage fraction that triggers batch-size reduction.",
    )
    thermal_aware: bool = Field(
        default=True,
        description="Detect thermal throttling and automatically pause or reduce "
        "training intensity to prevent system instability.",
    )
    thermal_pause_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Seconds to pause training when thermal throttling is detected.",
    )
    battery_aware: bool = Field(
        default=True,
        description="Pause training when battery level is critically low.",
    )
    min_battery_pct: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Minimum battery percentage.  Training pauses below this level.",
    )
    mps_high_watermark_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="PYTORCH_MPS_HIGH_WATERMARK_RATIO value.  0.0 = no limit "
        "(recommended for training), 1.0 = strict limit.",
    )


# ---------------------------------------------------------------------------
# Sub-config: ITI (Inference-Time Intervention)
# ---------------------------------------------------------------------------


class ITIConfig(BaseModel):
    """Configuration for Inference-Time Intervention (ITI).

    ITI finds 'truthfulness directions' in attention heads and bakes them
    into model weights as o_proj biases, reducing hallucination at zero
    inference cost.
    """

    enabled: bool = Field(
        default=False,
        description="Enable ITI probing and baking pipeline stages.",
    )
    probing_dataset: str = Field(
        default="truthful_qa",
        description="HuggingFace dataset used for probing truthfulness directions.",
    )
    num_probing_samples: int = Field(
        default=500,
        ge=10,
        description="Number of samples to use from the probing dataset.",
    )
    num_heads: int = Field(
        default=48,
        ge=1,
        description="Top-K attention heads to intervene on.",
    )
    alpha: float = Field(
        default=15.0,
        gt=0.0,
        description="Intervention strength (scales the truthfulness direction).",
    )
    method: Literal["center_of_mass", "linear_probe"] = Field(
        default="center_of_mass",
        description="Method for computing truthfulness directions.",
    )
    bake_in: bool = Field(
        default=True,
        description="Bake directions into model weights (Ollama/GGUF compatible).",
    )


# ---------------------------------------------------------------------------
# Sub-config: Refusal-Aware Training (R-Tuning)
# ---------------------------------------------------------------------------


class RefusalConfig(BaseModel):
    """Configuration for Refusal-Aware Training (R-Tuning).

    Mixes 'I don't know' refusal examples into training data so the model
    learns to refuse rather than hallucinate on questions beyond its knowledge.
    """

    enabled: bool = Field(
        default=False,
        description="Enable refusal data augmentation before training.",
    )
    refusal_ratio: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
        description="Fraction of training data to replace with refusal examples.",
    )
    refusal_responses: list[str] = Field(
        default=[
            "I don't have enough information to answer that accurately.",
            "I'm not confident in my knowledge about this topic.",
            "I don't know the answer to that question.",
        ],
        description="Pool of refusal response templates.",
    )


class IFDConfig(BaseModel):
    """Configuration for IFD (Instruction-Following Difficulty) data scoring.

    Computes per-sample IFD scores as the ratio of conditioned to unconditioned
    perplexity: IFD(Q, A) = s(A|Q) / s(A).  High-IFD samples are harder for
    the model to follow, and are often more valuable for training.
    """

    enabled: bool = Field(
        default=False,
        description="Enable IFD scoring and data filtering.",
    )
    select_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of training data to keep after IFD filtering (top-k by IFD score).",
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        description="Batch size for IFD scoring forward passes.",
    )
    max_length: int = Field(
        default=512,
        ge=64,
        description="Maximum sequence length for IFD scoring.",
    )


class MergeConfig(BaseModel):
    """Configuration for model merging (TIES, SLERP, linear averaging)."""

    enabled: bool = Field(
        default=False,
        description="Enable model merging as a post-training step.",
    )
    method: Literal["linear", "slerp", "ties"] = Field(
        default="linear",
        description="Merge method: linear (weighted avg), slerp (spherical interpolation), "
        "ties (Trim, Elect Sign & Merge).",
    )
    models: list[str] = Field(
        default_factory=list,
        description="List of model paths or HuggingFace IDs to merge.",
    )
    weights: list[float] = Field(
        default_factory=list,
        description="Per-model weights for linear/TIES merging. If empty, equal weights are used.",
    )
    base_model: str | None = Field(
        default=None,
        description="Base model for TIES merging (task vectors are computed "
        "relative to this model). Required for TIES.",
    )
    slerp_t: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Interpolation parameter for SLERP (0.0 = first model, "
        "1.0 = second model). Only used with method='slerp'.",
    )
    ties_density: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of top-magnitude parameters to keep in TIES "
        "trimming step. Lower values are more aggressive.",
    )
    output_path: str | None = Field(
        default=None,
        description="Where to save the merged model. If None, saves to "
        "training output_dir / 'merged'.",
    )


class AlignmentConfig(BaseModel):
    """Configuration for preference-based alignment training (DPO, ORPO)."""

    preference_dataset: str | None = Field(
        default=None,
        description="Path or HuggingFace ID for the preference dataset "
        "with prompt/chosen/rejected columns.",
    )
    prompt_field: str = Field(
        default="prompt",
        description="Column name for prompts in the preference dataset.",
    )
    chosen_field: str = Field(
        default="chosen",
        description="Column name for preferred responses.",
    )
    rejected_field: str = Field(
        default="rejected",
        description="Column name for dispreferred responses.",
    )
    beta: float = Field(
        default=0.1,
        gt=0.0,
        description="KL divergence penalty coefficient (DPO/ORPO).",
    )
    max_prompt_length: int = Field(
        default=512,
        ge=64,
        description="Maximum prompt length in tokens.",
    )
    max_length: int = Field(
        default=1024,
        ge=128,
        description="Maximum total sequence length (prompt + response).",
    )
    loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = Field(
        default="sigmoid",
        description="DPO loss variant.",
    )
    num_generations: int = Field(
        default=4,
        ge=2,
        description="Number of completions per prompt for GRPO group scoring.",
    )
    max_completion_length: int = Field(
        default=256,
        ge=16,
        description="Maximum completion length for GRPO generation.",
    )


# ---------------------------------------------------------------------------
# Sub-config: MLX (Apple Silicon native training)
# ---------------------------------------------------------------------------


class MLXConfig(BaseModel):
    """Configuration for MLX-based training on Apple Silicon.

    When enabled, training uses Apple's MLX framework (via ``mlx-lm``)
    instead of PyTorch.  MLX leverages unified memory on M-series chips
    for efficient LLM fine-tuning.  Supports LoRA, DoRA, and full
    fine-tuning with automatic QLoRA for quantised models.

    Requires: ``pip install 'mlx-lm[train]'``  (Apple Silicon only).
    """

    enabled: bool = Field(
        default=False,
        description="Use MLX backend instead of PyTorch for training.",
    )
    fine_tune_type: Literal["lora", "dora", "full"] = Field(
        default="lora",
        description="Fine-tuning strategy: 'lora' (adapters), 'dora' "
        "(weight-decomposed LoRA), or 'full' (unfreeze last N layers).",
    )
    num_layers: int = Field(
        default=16,
        description="Number of transformer layers to adapt (from the end).  -1 = all layers.",
    )
    lora_rank: int = Field(
        default=8,
        ge=1,
        description="LoRA rank (r) for the low-rank adapters.",
    )
    lora_scale: float = Field(
        default=20.0,
        gt=0.0,
        description="LoRA scaling factor (analogous to alpha/r in PEFT).",
    )
    lora_dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout applied to the LoRA input path.",
    )
    iters: int = Field(
        default=1000,
        ge=1,
        description="Total number of training iterations.",
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        description="Training batch size.",
    )
    learning_rate: float = Field(
        default=1e-5,
        gt=0.0,
        description="Peak learning rate.",
    )
    optimizer: Literal["adam", "adamw", "sgd", "adafactor"] = Field(
        default="adam",
        description="Optimizer to use for MLX training.",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=64,
        description="Maximum sequence length for training.",
    )
    grad_checkpoint: bool = Field(
        default=False,
        description="Enable gradient checkpointing (trades compute for memory).",
    )
    grad_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="Number of gradient accumulation steps.",
    )
    steps_per_report: int = Field(
        default=10,
        ge=1,
        description="Report training loss every N steps.",
    )
    steps_per_eval: int = Field(
        default=200,
        ge=1,
        description="Run validation every N steps.",
    )
    steps_per_save: int = Field(
        default=100,
        ge=1,
        description="Save adapter checkpoint every N steps.",
    )
    mask_prompt: bool = Field(
        default=True,
        description="Train only on assistant/completion tokens (mask the prompt).",
    )
    lr_schedule: str | None = Field(
        default="cosine_decay",
        description="Learning-rate schedule name from mlx.optimizers.schedulers "
        "(e.g. 'cosine_decay', 'linear_decay').  None = constant LR.",
    )
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of linear warmup steps before the main schedule.",
    )
    adapter_path: str = Field(
        default="adapters",
        description="Directory to save adapter weights (relative to output_dir).",
    )
    fuse_after_training: bool = Field(
        default=True,
        description="Fuse LoRA adapters into base weights after training.",
    )


# ---------------------------------------------------------------------------
# Sub-config: Compute Backend
# ---------------------------------------------------------------------------


class ComputeBackend(str, enum.Enum):
    """Where training runs: local machine, SLURM cluster, or cloud."""

    local = "local"
    slurm = "slurm"
    aws = "aws"
    gcp = "gcp"
    azure = "azure"
    lambda_cloud = "lambda"
    runpod = "runpod"
    ssh = "ssh"


class SLURMConfig(BaseModel):
    """SLURM job scheduler settings for HPC clusters."""

    partition: str = Field(
        default="gpuq",
        description="SLURM partition to submit to.",
    )
    qos: str | None = Field(
        default="gpu",
        description="Quality-of-service tier (e.g. 'gpu', 'normal').",
    )
    gres: str = Field(
        default="gpu:A100.80gb:1",
        description="Generic resource specification (e.g. 'gpu:A100.80gb:1').",
    )
    cpus_per_task: int = Field(
        default=8,
        ge=1,
        le=128,
        description="CPU cores per task.",
    )
    mem_gb: int = Field(
        default=64,
        ge=1,
        description="Memory in GB.",
    )
    time_limit: str = Field(
        default="2:00:00",
        description="Wall time limit (HH:MM:SS).",
    )
    exclude_nodes: str | None = Field(
        default=None,
        description="Comma-separated list of nodes to exclude.",
    )
    modules: list[str] = Field(
        default_factory=list,
        description="Environment modules to load before training.",
    )
    conda_env: str | None = Field(
        default=None,
        description="Conda environment name to activate.",
    )
    conda_prefix: str | None = Field(
        default=None,
        description="Path to conda/miniforge installation (e.g. '~/miniforge').",
    )
    extra_env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables to export in the SBATCH script.",
    )
    extra_sbatch_flags: dict[str, str] = Field(
        default_factory=dict,
        description="Additional #SBATCH directives as key-value pairs.",
    )

    model_config = {"extra": "forbid"}


class SSHConfig(BaseModel):
    """SSH connection settings for remote training (SLURM or bare-metal)."""

    host: str = Field(
        ...,
        description="SSH hostname or alias (must match ~/.ssh/config or be a valid host).",
    )
    user: str | None = Field(
        default=None,
        description="SSH username (if not specified in ~/.ssh/config).",
    )
    key_path: str | None = Field(
        default=None,
        description="Path to SSH private key (if not using ssh-agent).",
    )
    remote_dir: str = Field(
        default="~/llm-forge",
        description="Working directory on the remote machine.",
    )
    sync_exclude: list[str] = Field(
        default=[
            ".venv",
            "__pycache__",
            "*.pyc",
            "outputs/",
            "eval_results/",
            ".git",
            "*.egg-info",
        ],
        description="Patterns to exclude when rsyncing code to remote.",
    )

    model_config = {"extra": "forbid"}


class CloudGPUConfig(BaseModel):
    """Cloud GPU instance settings (AWS, GCP, Azure, Lambda, RunPod)."""

    instance_type: str | None = Field(
        default=None,
        description="Instance type (e.g. 'p4d.24xlarge' for AWS, 'a2-highgpu-1g' for GCP).",
    )
    region: str | None = Field(
        default=None,
        description="Cloud region (e.g. 'us-east-1', 'us-central1').",
    )
    gpu_type: str | None = Field(
        default=None,
        description="GPU type (e.g. 'A100', 'H100', 'A10G'). Used for auto-selecting instance.",
    )
    num_gpus: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs to request.",
    )
    disk_gb: int = Field(
        default=200,
        ge=50,
        description="Disk size in GB (models + data + outputs).",
    )
    spot: bool = Field(
        default=False,
        description="Use spot/preemptible instances (cheaper but can be interrupted).",
    )
    max_price_per_hour: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum price per hour for spot instances (USD).",
    )
    auto_shutdown: bool = Field(
        default=True,
        description="Automatically stop the instance after training completes.",
    )
    docker_image: str | None = Field(
        default=None,
        description="Docker image to use (e.g. 'nvcr.io/nvidia/pytorch:24.01-py3').",
    )
    setup_commands: list[str] = Field(
        default_factory=list,
        description="Shell commands to run before training (install deps, etc.).",
    )

    model_config = {"extra": "forbid"}


class ComputeConfig(BaseModel):
    """Compute backend configuration — where and how to run training.

    Controls whether training runs locally, on a SLURM cluster,
    or on a cloud GPU provider. When backend is not 'local',
    the SSH settings connect to the remote, and backend-specific
    settings (slurm/cloud) configure the job.
    """

    backend: ComputeBackend = Field(
        default=ComputeBackend.local,
        description="Where to run training: 'local', 'slurm', 'aws', 'gcp', 'azure', 'lambda', 'runpod', 'ssh'.",
    )
    ssh: SSHConfig | None = Field(
        default=None,
        description="SSH connection settings for remote backends.",
    )
    slurm: SLURMConfig | None = Field(
        default=None,
        description="SLURM job scheduler settings (required when backend='slurm').",
    )
    cloud: CloudGPUConfig | None = Field(
        default=None,
        description="Cloud GPU instance settings (for aws/gcp/azure/lambda/runpod backends).",
    )
    sync_code: bool = Field(
        default=True,
        description="Rsync local code to remote before training.",
    )
    stream_logs: bool = Field(
        default=True,
        description="Stream training logs back to the local terminal.",
    )
    pull_outputs: bool = Field(
        default=True,
        description="Pull output artifacts (model, GGUF) back to local after training.",
    )
    local_output_dir: str | None = Field(
        default=None,
        description="Local directory to pull remote outputs to (defaults to training.output_dir).",
    )

    @model_validator(mode="after")
    def _validate_backend_settings(self) -> ComputeConfig:
        """Ensure required sub-configs are present for the selected backend."""
        if self.backend == ComputeBackend.slurm:
            if self.ssh is None:
                raise ValueError(
                    "compute.ssh is required when backend='slurm'. "
                    "Set ssh.host to your cluster login node."
                )
            if self.slurm is None:
                self.slurm = SLURMConfig()
        elif self.backend in (
            ComputeBackend.aws,
            ComputeBackend.gcp,
            ComputeBackend.azure,
            ComputeBackend.lambda_cloud,
            ComputeBackend.runpod,
        ):
            if self.cloud is None:
                self.cloud = CloudGPUConfig()
        elif self.backend == ComputeBackend.ssh:
            if self.ssh is None:
                raise ValueError(
                    "compute.ssh is required when backend='ssh'. "
                    "Set ssh.host to your remote machine."
                )
        return self

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------


class LLMForgeConfig(BaseModel):
    """Top-level configuration for an llm-forge run.

    A single YAML file parsed into this model drives the entire pipeline:
    data cleaning -> training -> evaluation -> serving.
    """

    model: ModelConfig = Field(
        ...,
        description="Pretrained model selection and loading options.",
    )
    lora: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description="LoRA / QLoRA adapter hyper-parameters.",
    )
    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig,
        description="BitsAndBytes quantization options.",
    )
    data: DataConfig = Field(
        ...,
        description="Dataset paths, format, and cleaning options.",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Core training hyper-parameters.",
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="Distributed / multi-GPU training settings.",
    )
    evaluation: EvalConfig = Field(
        default_factory=EvalConfig,
        description="Post-training evaluation settings.",
    )
    rag: RAGConfig = Field(
        default_factory=RAGConfig,
        description="RAG pipeline configuration.",
    )
    serving: ServingConfig = Field(
        default_factory=ServingConfig,
        description="Model serving and export configuration.",
    )
    mac: MacConfig = Field(
        default_factory=MacConfig,
        description="MacOS / Apple Silicon training optimisations.",
    )
    iti: ITIConfig = Field(
        default_factory=ITIConfig,
        description="Inference-Time Intervention (anti-hallucination) settings.",
    )
    refusal: RefusalConfig = Field(
        default_factory=RefusalConfig,
        description="Refusal-aware training (R-Tuning) settings.",
    )
    alignment: AlignmentConfig = Field(
        default_factory=AlignmentConfig,
        description="Preference-based alignment (DPO/ORPO) settings.",
    )
    ifd: IFDConfig = Field(
        default_factory=IFDConfig,
        description="IFD (Instruction-Following Difficulty) data scoring and filtering.",
    )
    merge: MergeConfig = Field(
        default_factory=MergeConfig,
        description="Model merging (TIES/SLERP/linear) settings.",
    )
    mlx: MLXConfig = Field(
        default_factory=MLXConfig,
        description="MLX-based training on Apple Silicon (alternative to PyTorch).",
    )
    compute: ComputeConfig = Field(
        default_factory=ComputeConfig,
        description="Compute backend: local, SLURM cluster, or cloud GPU provider.",
    )

    # ------------------------------------------------------------------
    # Automatic adjustments
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _auto_configure_qlora(self) -> LLMForgeConfig:
        """When training mode is 'qlora', ensure quantization is enabled."""
        if self.training.mode == TrainingMode.qlora:
            if not self.quantization.load_in_4bit and not self.quantization.load_in_8bit:
                self.quantization.load_in_4bit = True
                self.quantization.bnb_4bit_quant_type = "nf4"
                self.quantization.bnb_4bit_use_double_quant = True
                self.quantization.bnb_4bit_compute_dtype = PrecisionMode.bf16
        return self

    @model_validator(mode="after")
    def _warn_forgetting_risk(self) -> LLMForgeConfig:
        """Warn about config combinations that risk catastrophic forgetting."""
        import warnings

        mode = self.training.mode
        if mode not in (TrainingMode.lora, TrainingMode.qlora):
            return self

        lr = self.training.learning_rate
        r = self.lora.r
        alpha = self.lora.alpha
        targets = self.lora.target_modules
        neftune = self.training.neftune_noise_alpha
        rslora = self.lora.use_rslora
        max_samples = self.data.max_samples
        epochs = self.training.num_epochs

        # Compute effective scaling factor
        if rslora:
            import math

            effective_scale = alpha / math.sqrt(r)
        else:
            effective_scale = alpha / r

        # Warn: high effective LR (learning_rate * effective_scale)
        effective_lr = lr * effective_scale
        if effective_lr > 1e-3:
            warnings.warn(
                f"HIGH FORGETTING RISK: effective learning rate is {effective_lr:.2e} "
                f"(lr={lr:.2e} × scale={effective_scale:.1f}).  "
                f"For LoRA on Instruct models, keep effective LR < 5e-4.  "
                f"Consider reducing learning_rate or alpha.",
                UserWarning,
                stacklevel=2,
            )

        # Warn: all-linear targets on small datasets
        is_all_linear = targets == "all-linear" or (isinstance(targets, list) and len(targets) > 5)
        is_small_dataset = max_samples is not None and max_samples < 20000
        if is_all_linear and is_small_dataset:
            warnings.warn(
                f"FORGETTING RISK: target_modules covers many layers with only "
                f"{max_samples} samples.  Consider targeting attention-only modules "
                f"(['q_proj', 'v_proj', 'k_proj', 'o_proj']) to preserve reasoning.",
                UserWarning,
                stacklevel=2,
            )

        # Warn: NEFTune on small datasets
        if neftune is not None and neftune > 0 and is_small_dataset:
            warnings.warn(
                f"FORGETTING RISK: NEFTune alpha={neftune} with only {max_samples} "
                f"samples.  NEFTune noise amplifies forgetting on small datasets.  "
                f"Consider disabling (neftune_noise_alpha: null).",
                UserWarning,
                stacklevel=2,
            )

        # Warn: multiple epochs on small datasets
        if epochs > 1 and is_small_dataset:
            warnings.warn(
                f"OVERFITTING RISK: {epochs} epochs on {max_samples} samples.  "
                f"For LoRA on <20K samples, 1 epoch is usually sufficient.",
                UserWarning,
                stacklevel=2,
            )

        return self

    model_config = {
        "use_enum_values": True,
        "validate_default": True,
        "extra": "forbid",
    }
