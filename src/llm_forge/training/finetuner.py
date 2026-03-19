"""Fine-tuning engine supporting LoRA, QLoRA, and full parameter training.

Handles model loading with proper quantization, PEFT adapter application,
SFT training via TRL, and merged model export.  Includes optional Unsloth
integration for accelerated kernels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.finetuner")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        TrainingArguments,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

try:
    from trl import SFTConfig, SFTTrainer

    _TRL_AVAILABLE = True
except (ImportError, RuntimeError):
    _TRL_AVAILABLE = False

try:
    from datasets import Dataset

    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

try:
    from unsloth import FastLanguageModel as _UnslothModel

    _UNSLOTH_AVAILABLE = True
except ImportError:
    _UNSLOTH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Precision mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, torch.dtype] = {}
if _TORCH_AVAILABLE:
    _DTYPE_MAP = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }


# ============================================================================
# FineTuner
# ============================================================================


class FineTuner:
    """Unified fine-tuning engine for LoRA, QLoRA, and full parameter training.

    This class orchestrates the entire fine-tuning pipeline:

    1. Load a pretrained model with the appropriate precision and
       quantization configuration.
    2. (Optionally) apply PEFT LoRA adapters.
    3. Train using TRL's ``SFTTrainer`` with full callback support.
    4. Merge adapter weights back into the base model and save.

    Parameters
    ----------
    config : object
        An ``LLMForgeConfig`` instance (or any object exposing ``.model``,
        ``.training``, ``.lora``, ``.quantization`` sub-configs).
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._is_peft: bool = False

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def setup_model(
        self, config: Any | None = None
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the base model with the correct dtype and quantization.

        Parameters
        ----------
        config : object, optional
            Override config for this call.  Falls back to ``self.config``.

        Returns
        -------
        tuple[PreTrainedModel, PreTrainedTokenizerBase]
            Loaded model and tokenizer.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        cfg = config or self.config
        model_cfg = cfg.model
        training_cfg = cfg.training
        mode = training_cfg.mode  # lora | qlora | full

        model_name = model_cfg.name
        torch_dtype = _DTYPE_MAP.get(
            str(model_cfg.torch_dtype), torch.bfloat16 if _TORCH_AVAILABLE else None
        )
        max_seq_length = model_cfg.max_seq_length
        trust_remote_code = model_cfg.trust_remote_code
        attn_implementation = model_cfg.attn_implementation

        # ----- Unsloth fast path ------------------------------------------
        if training_cfg.use_unsloth and _UNSLOTH_AVAILABLE:
            logger.info("Loading model via Unsloth fast path: %s", model_name)
            load_in_4bit = mode == "qlora"
            model, tokenizer = _UnslothModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=torch_dtype,
                load_in_4bit=load_in_4bit,
                trust_remote_code=trust_remote_code,
            )
            self.model = model
            self.tokenizer = tokenizer
            return model, tokenizer

        # ----- Standard loading -------------------------------------------
        # Tokenizer
        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            model_max_length=max_seq_length,
        )
        if tokenizer.pad_token is None:
            # Prefer Llama 3's dedicated fine-tune pad token over eos_token
            # (training on eos_token teaches the model to predict padding).
            vocab = tokenizer.get_vocab()
            if "<|finetune_right_pad_id|>" in vocab:
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
                tokenizer.pad_token_id = vocab["<|finetune_right_pad_id|>"]
                logger.info("Using Llama 3 finetune pad token")
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Ensure chat template with {% generation %} markers exists.
        # TRL v0.20's assistant_only_loss requires {% generation %}...{% endgeneration %}
        # tags in the Jinja template to identify assistant content for loss masking.
        # - Base Llama 3.x models have NO chat template → set our template
        # - Instruct variants have a template but WITHOUT generation markers → override it
        _has_llama3_tokens = "<|start_header_id|>" in tokenizer.get_vocab()
        _needs_generation_template = (
            _has_llama3_tokens
            and getattr(training_cfg, "assistant_only_loss", False)
            and (
                not getattr(tokenizer, "chat_template", None)
                or "{% generation %}" not in (tokenizer.chat_template or "")
            )
        )
        if _needs_generation_template:
            tokenizer.chat_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set header = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' %}"
                "{% if loop.first %}{{ bos_token + header }}{% else %}{{ header }}{% endif %}"
                "{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] | trim }}<|eot_id|>{% endgeneration %}"
                "{% else %}{{ message['content'] | trim }}<|eot_id|>"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            )
            logger.info(
                "Set Llama 3 chat template with generation markers (for assistant_only_loss)"
            )
        elif not getattr(tokenizer, "chat_template", None) and _has_llama3_tokens:
            tokenizer.chat_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set header = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' %}"
                "{% if loop.first %}{{ bos_token + header }}{% else %}{{ header }}{% endif %}"
                "{{ message['content'] | trim }}<|eot_id|>"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            )
            logger.info("Set Llama 3 chat template on base model tokenizer")

        # Build model kwargs
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        # Attention implementation
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        # Revision
        if model_cfg.revision:
            model_kwargs["revision"] = model_cfg.revision

        # RoPE scaling
        if model_cfg.rope_scaling:
            model_kwargs["rope_scaling"] = model_cfg.rope_scaling

        # Device map — disable in distributed mode (FSDP/DDP handles placement)
        # Check env vars set by torchrun/accelerate before distributed init
        _world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if _world_size <= 1:
            model_kwargs["device_map"] = "auto"

        # ---- QLoRA quantization ------------------------------------------
        if mode == "qlora":
            quant_cfg = cfg.quantization
            compute_dtype = _DTYPE_MAP.get(str(quant_cfg.bnb_4bit_compute_dtype), torch.bfloat16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
            )
            model_kwargs["quantization_config"] = bnb_config
            logger.info(
                "QLoRA quantization: 4-bit NF4, double-quant=%s, compute_dtype=%s",
                quant_cfg.bnb_4bit_use_double_quant,
                compute_dtype,
            )

        # ---- 8-bit quantization ------------------------------------------
        elif mode == "lora" and hasattr(cfg, "quantization") and cfg.quantization.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config
            logger.info("Loading model in 8-bit quantization")

        # ---- Load model --------------------------------------------------
        logger.info("Loading model: %s (mode=%s)", model_name, mode)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Prepare for k-bit training (QLoRA)
        if mode == "qlora" and _PEFT_AVAILABLE:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_cfg.gradient_checkpointing,
            )

        # Gradient checkpointing for full fine-tuning
        if mode == "full" and training_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Resize embeddings if tokenizer was expanded
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(
                "Resized embeddings: %d -> %d",
                model.config.vocab_size,
                len(tokenizer),
            )

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    # ------------------------------------------------------------------ #
    # LoRA adapter application
    # ------------------------------------------------------------------ #

    def apply_lora(
        self,
        model: PreTrainedModel,
        config: Any | None = None,
    ) -> PeftModel:
        """Apply PEFT LoRA adapters to the model.

        Parameters
        ----------
        model : PreTrainedModel
            Base model (possibly quantized) to wrap with LoRA adapters.
        config : object, optional
            Override config.  Falls back to ``self.config``.

        Returns
        -------
        PeftModel
            Model with LoRA adapters attached.
        """
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        cfg = config or self.config
        lora_cfg = cfg.lora
        training_cfg = cfg.training

        # Unsloth fast path
        if training_cfg.use_unsloth and _UNSLOTH_AVAILABLE:
            logger.info("Applying LoRA via Unsloth")
            model = _UnslothModel.get_peft_model(
                model,
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=lora_cfg.target_modules,
                bias=lora_cfg.bias,
                use_rslora=lora_cfg.use_rslora,
                use_gradient_checkpointing="unsloth",
            )
            self.model = model
            self._is_peft = True
            return model

        # Map task type string to TaskType enum
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }
        task_type = task_type_map.get(lora_cfg.task_type, TaskType.CAUSAL_LM)

        # target_modules can be "all-linear" (PEFT discovers all nn.Linear)
        # or an explicit list of module names.
        target_modules = lora_cfg.target_modules
        if isinstance(target_modules, list):
            target_modules = list(target_modules)

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=target_modules,
            bias=lora_cfg.bias,
            task_type=task_type,
            use_rslora=lora_cfg.use_rslora,
            use_dora=lora_cfg.use_dora,
        )

        logger.info(
            "Applying LoRA: r=%d, alpha=%d, dropout=%.2f, targets=%s",
            lora_cfg.r,
            lora_cfg.alpha,
            lora_cfg.dropout,
            lora_cfg.target_modules,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Enable gradient checkpointing if requested
        if training_cfg.gradient_checkpointing:
            model.enable_input_require_grads()

        self.model = model
        self._is_peft = True
        return model

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        config: Any | None = None,
        eval_dataset: Dataset | None = None,
        callbacks: list[Any] | None = None,
    ) -> Any:
        """Run SFT training with the given model and dataset.

        Parameters
        ----------
        model : PreTrainedModel
            Model (possibly with LoRA adapters) to train.
        dataset : Dataset
            Training dataset with a ``"text"`` column.
        config : object, optional
            Override config.
        eval_dataset : Dataset, optional
            Evaluation dataset.
        callbacks : list, optional
            Additional HF Trainer callbacks.

        Returns
        -------
        TrainOutput
            Training result from the HF Trainer.
        """
        if not _TRL_AVAILABLE:
            raise ImportError("trl is required for training. Install with: pip install trl")

        cfg = config or self.config
        training_cfg = cfg.training
        model_cfg = cfg.model

        # Build TrainingArguments
        training_args_dict: dict[str, Any] = {
            "output_dir": training_cfg.output_dir,
            "num_train_epochs": training_cfg.num_epochs,
            "per_device_train_batch_size": training_cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": training_cfg.per_device_eval_batch_size,
            "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
            "learning_rate": training_cfg.learning_rate,
            "weight_decay": training_cfg.weight_decay,
            "warmup_ratio": training_cfg.warmup_ratio,
            "lr_scheduler_type": training_cfg.lr_scheduler_type,
            "max_grad_norm": training_cfg.max_grad_norm,
            "logging_steps": training_cfg.logging_steps,
            "save_steps": training_cfg.save_steps,
            "save_total_limit": training_cfg.save_total_limit,
            "bf16": training_cfg.bf16,
            "fp16": training_cfg.fp16,
            "gradient_checkpointing": training_cfg.gradient_checkpointing,
            "optim": training_cfg.optim,
            "group_by_length": training_cfg.group_by_length,
            "report_to": training_cfg.report_to,
            "seed": cfg.data.seed if hasattr(cfg, "data") else 42,
            "dataloader_num_workers": cfg.data.num_workers if hasattr(cfg, "data") else 4,
            "remove_unused_columns": False,
            # Label smoothing (reduces overconfidence, +1-3% quality).
            "label_smoothing_factor": training_cfg.label_smoothing_factor,
            # Gradient accumulation loss fix (transformers >= 4.46):
            # Synchronise non-padding token counts across devices so loss
            # is computed as total_loss / total_tokens, not mean(per_batch).
            "average_tokens_across_devices": training_cfg.average_tokens_across_devices,
        }

        # Evaluation settings
        if eval_dataset is not None:
            eval_strat = training_cfg.eval_strategy
            training_args_dict["eval_strategy"] = eval_strat
            if training_cfg.eval_steps is not None:
                training_args_dict["eval_steps"] = training_cfg.eval_steps
            # Align save_strategy with eval_strategy for load_best_model_at_end
            training_args_dict["save_strategy"] = eval_strat
            training_args_dict["load_best_model_at_end"] = True
            training_args_dict["metric_for_best_model"] = "eval_loss"

        # Warmup steps override
        if training_cfg.warmup_steps is not None:
            training_args_dict["warmup_steps"] = training_cfg.warmup_steps
            training_args_dict.pop("warmup_ratio", None)

        # Resume from checkpoint
        resume_from = training_cfg.resume_from_checkpoint

        # Gradient checkpointing kwargs
        if training_cfg.gradient_checkpointing:
            training_args_dict["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        # NEFTune noise
        if training_cfg.neftune_noise_alpha is not None and training_cfg.neftune_noise_alpha > 0:
            training_args_dict["neftune_noise_alpha"] = training_cfg.neftune_noise_alpha
            logger.info("NEFTune noise enabled: alpha=%.2f", training_cfg.neftune_noise_alpha)

        # Completion-only loss (for prompt-completion datasets)
        if training_cfg.completion_only_loss is not None:
            training_args_dict["completion_only_loss"] = training_cfg.completion_only_loss
            logger.info("Completion-only loss: %s", training_cfg.completion_only_loss)

        # Assistant-only loss (for conversational/messages datasets) — THE
        # critical setting for instruction tuning.  TRL v0.20+ uses this to
        # call apply_chat_template(return_assistant_tokens_mask=True), which
        # creates a binary mask over assistant turns.  Non-assistant tokens
        # get label=-100 so the model only learns to predict responses.
        if getattr(training_cfg, "assistant_only_loss", False):
            training_args_dict["assistant_only_loss"] = True
            logger.info("Assistant-only loss: True (masking non-assistant tokens)")

        # Build SFTConfig (extends TrainingArguments in TRL)
        # Filter out kwargs not accepted by current TRL version's SFTConfig
        import inspect

        _sft_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
        _filtered_args = {k: v for k, v in training_args_dict.items() if k in _sft_params}
        _dropped = set(training_args_dict.keys()) - set(_filtered_args.keys())
        if _dropped:
            logger.warning("Filtered out unsupported SFTConfig args: %s", _dropped)

        # Add SFT-specific args (names vary across TRL versions)
        if "max_seq_length" in _sft_params:
            _filtered_args["max_seq_length"] = model_cfg.max_seq_length
        elif "max_length" in _sft_params:
            _filtered_args["max_length"] = model_cfg.max_seq_length
        if "packing" in _sft_params:
            use_packing = training_cfg.pack_sequences
            if use_packing:
                # Packing with BFD strategy requires FlashAttention2 for correct
                # attention boundaries. Without it, SDPA treats packed samples as
                # one giant sequence, causing cross-sample contamination and
                # catastrophically high loss (2x+ worse than random chance).
                _flash_ok = False
                try:
                    import flash_attn  # noqa: F401

                    _flash_ok = True
                except ImportError:
                    pass
                attn_impl = getattr(model.config, "_attn_implementation", None)
                if not _flash_ok and attn_impl != "flash_attention_2":
                    logger.warning(
                        "pack_sequences=True requires flash_attention_2 for correct "
                        "attention boundaries. FlashAttention not available — "
                        "DISABLING packing to prevent cross-sample contamination."
                    )
                    use_packing = False
                else:
                    logger.info(
                        "Sample packing enabled with FlashAttention2 — TRL will "
                        "concatenate short examples with proper attention masking"
                    )
            _filtered_args["packing"] = use_packing
        # Detect whether dataset has a 'messages' column (TRL chat-template
        # pipeline) or a 'text' column (legacy flat-text pipeline).  The
        # messages path lets TRL apply the model's chat template AND enables
        # assistant-only loss masking — critical for instruction tuning.
        _has_messages = "messages" in dataset.column_names
        _has_text = "text" in dataset.column_names

        if _has_messages and self.tokenizer is not None:
            # Messages path: TRL applies the chat template automatically
            # and can mask non-assistant tokens for completion-only loss.
            logger.info(
                "Dataset has 'messages' column — using TRL chat-template pipeline "
                "with assistant-only loss masking"
            )
            # Do NOT set dataset_text_field — TRL detects 'messages' automatically
        elif _has_text and "dataset_text_field" in _sft_params:
            # Legacy text path
            _filtered_args["dataset_text_field"] = "text"
            logger.info(
                "Dataset has 'text' column — using legacy text pipeline "
                "(completion_only_loss may not work correctly)"
            )
        else:
            # Fallback
            if "dataset_text_field" in _sft_params:
                _filtered_args["dataset_text_field"] = "text"

        sft_config = SFTConfig(**_filtered_args)

        # Build trainer
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": sft_config,
            "train_dataset": dataset,
            "processing_class": self.tokenizer,
        }

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer = SFTTrainer(**trainer_kwargs)

        # ---- Pre-training diagnostics ----
        # Verify label masking is working (critical for instruction tuning).
        try:
            _diag_batch = next(iter(trainer.get_train_dataloader()))
            _diag_labels = _diag_batch.get("labels")
            if _diag_labels is not None:
                _masked = (_diag_labels == -100).sum().item()
                _total = _diag_labels.numel()
                _pct = _masked / _total * 100 if _total > 0 else 0
                logger.info(
                    "PRE-TRAINING DIAGNOSTIC: Masked tokens: %d/%d (%.1f%%) "
                    "— expected >50%% for conversational data",
                    _masked,
                    _total,
                    _pct,
                )
                if _pct < 30 and _has_messages:
                    logger.warning(
                        "LOW MASKING (%.1f%%): Model will train on ALL tokens "
                        "including system/user prompts. This inflates loss and "
                        "can cause system-prompt regurgitation. Check that "
                        "completion_only_loss is enabled and the chat template "
                        "is applied correctly.",
                        _pct,
                    )
        except Exception as _diag_exc:
            logger.debug("Pre-training diagnostic skipped: %s", _diag_exc)

        logger.info(
            "Starting SFT training: %d epochs, batch_size=%d, grad_accum=%d, lr=%.2e",
            training_cfg.num_epochs,
            training_cfg.per_device_train_batch_size,
            training_cfg.gradient_accumulation_steps,
            training_cfg.learning_rate,
        )

        # Train
        result = trainer.train(resume_from_checkpoint=resume_from)

        # Save final model
        trainer.save_model(training_cfg.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(training_cfg.output_dir)

        logger.info("Training complete. Model saved to %s", training_cfg.output_dir)
        return result

    # ------------------------------------------------------------------ #
    # Merge and save
    # ------------------------------------------------------------------ #

    def merge_and_save(
        self,
        model: PreTrainedModel | None = None,
        output_dir: str | None = None,
    ) -> Path:
        """Merge LoRA adapter weights into the base model and save.

        Parameters
        ----------
        model : PreTrainedModel, optional
            PeftModel to merge.  Defaults to ``self.model``.
        output_dir : str, optional
            Directory for the merged model.  Defaults to
            ``{config.training.output_dir}/merged``.

        Returns
        -------
        Path
            Directory containing the merged model.
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model to merge. Call setup_model() and apply_lora() first.")

        output_path = Path(output_dir or os.path.join(self.config.training.output_dir, "merged"))
        output_path.mkdir(parents=True, exist_ok=True)

        # Unsloth merge path
        if self.config.training.use_unsloth and _UNSLOTH_AVAILABLE:
            logger.info("Merging and saving via Unsloth to %s", output_path)
            model.save_pretrained_merged(
                str(output_path),
                self.tokenizer,
                save_method="merged_16bit",
            )
            return output_path

        # Standard PEFT merge
        if _PEFT_AVAILABLE and isinstance(model, PeftModel):
            # QLoRA fix: When training with 4-bit quantization, the in-memory
            # model has bitsandbytes .absmax metadata tensors.  Calling
            # merge_and_unload() on the quantized model produces a checkpoint
            # with these artifacts, which breaks GGUF conversion.
            # Fix: reload the base model in full precision, apply the adapter,
            # then merge — producing clean float16 weights.
            _is_quantized = (
                getattr(getattr(model, "config", None), "quantization_config", None) is not None
                or self.config.training.mode == "qlora"
            )

            if _is_quantized and _TRANSFORMERS_AVAILABLE:
                logger.info("QLoRA detected — reloading base model in float16 for clean merge")
                base_name = self.config.model.name
                adapter_dir = self.config.training.output_dir

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                )
                fresh_peft = PeftModel.from_pretrained(base_model, adapter_dir)
                merged_model = fresh_peft.merge_and_unload()

                del base_model, fresh_peft
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                logger.info("Merging LoRA adapters into base model")
                merged_model = model.merge_and_unload()

            merged_model.save_pretrained(
                str(output_path),
                safe_serialization=True,
            )
        else:
            logger.info("Saving model (no adapters to merge)")
            model.save_pretrained(
                str(output_path),
                safe_serialization=True,
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(output_path))

        # Strip {% generation %} markers from the saved template so the
        # merged model works with llama.cpp / Ollama / vLLM out of the box.
        from llm_forge.serving.export import _clean_chat_template_for_export

        _clean_chat_template_for_export(output_path)

        logger.info("Merged model saved to %s", output_path)
        return output_path
