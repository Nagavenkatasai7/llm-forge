"""Pre-training engine for training language models from scratch.

Supports configurable transformer architectures based on the Llama family,
custom BPE tokenizer training, and causal language modelling with proper
warmup and cosine decay scheduling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.pretrainer")

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
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
        Trainer,
        TrainingArguments,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )

    _TOKENIZERS_AVAILABLE = True
except ImportError:
    _TOKENIZERS_AVAILABLE = False

try:
    from datasets import Dataset

    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model size presets
# ---------------------------------------------------------------------------

MODEL_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "125M": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "num_key_value_heads": 12,
    },
    "350M": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "num_key_value_heads": 16,
    },
    "760M": {
        "hidden_size": 1536,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 6144,
        "num_key_value_heads": 16,
    },
    "1B": {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 8192,
        "num_key_value_heads": 8,
    },
}


# ============================================================================
# PreTrainer
# ============================================================================


class PreTrainer:
    """Pre-training engine for training transformer LMs from scratch.

    Provides the full pipeline: tokenizer training, model architecture
    construction, and causal-LM training with linear warmup and cosine
    decay scheduling.

    Parameters
    ----------
    config : object
        An ``LLMForgeConfig`` instance.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #

    def build_model(
        self,
        config: Any | None = None,
        model_size: str | None = None,
        vocab_size: int = 32000,
        max_position_embeddings: int = 2048,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        intermediate_size: int | None = None,
        num_key_value_heads: int | None = None,
    ) -> PreTrainedModel:
        """Create a transformer model from scratch using the Llama architecture.

        Parameters
        ----------
        config : object, optional
            Override config.  Falls back to ``self.config``.
        model_size : str, optional
            Preset size name: ``"125M"``, ``"350M"``, ``"760M"``, or ``"1B"``.
            Overrides individual dimension parameters when set.
        vocab_size : int
            Vocabulary size for the embedding layer.
        max_position_embeddings : int
            Maximum sequence length the model can handle.
        hidden_size : int, optional
            Model hidden dimension.  Inferred from preset when ``None``.
        num_hidden_layers : int, optional
            Number of transformer layers.
        num_attention_heads : int, optional
            Number of attention heads.
        intermediate_size : int, optional
            Feed-forward intermediate dimension.
        num_key_value_heads : int, optional
            Number of key-value heads for GQA.  If ``None``, equals
            ``num_attention_heads`` (standard MHA).

        Returns
        -------
        PreTrainedModel
            A randomly initialised Llama-architecture causal LM.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        cfg = config or self.config

        # Resolve model dimensions from preset or explicit args
        if model_size is not None:
            preset_key = model_size.upper()
            if preset_key not in MODEL_SIZE_PRESETS:
                raise ValueError(
                    f"Unknown model size preset '{model_size}'. "
                    f"Available: {', '.join(MODEL_SIZE_PRESETS.keys())}"
                )
            dims = MODEL_SIZE_PRESETS[preset_key]
            hidden_size = hidden_size or dims["hidden_size"]
            num_hidden_layers = num_hidden_layers or dims["num_hidden_layers"]
            num_attention_heads = num_attention_heads or dims["num_attention_heads"]
            intermediate_size = intermediate_size or dims["intermediate_size"]
            num_key_value_heads = num_key_value_heads or dims["num_key_value_heads"]
        else:
            # Defaults matching a ~125M model if nothing else is provided
            hidden_size = hidden_size or 768
            num_hidden_layers = num_hidden_layers or 12
            num_attention_heads = num_attention_heads or 12
            intermediate_size = intermediate_size or 3072
            num_key_value_heads = num_key_value_heads or num_attention_heads

        # Use tokenizer vocab size if available
        if self.tokenizer is not None:
            vocab_size = len(self.tokenizer)

        # Use config max_seq_length if available
        if hasattr(cfg, "model") and hasattr(cfg.model, "max_seq_length"):
            max_position_embeddings = cfg.model.max_seq_length

        # Build LlamaConfig
        llama_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            tie_word_embeddings=False,
            use_cache=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )

        # Estimate parameter count
        approx_params = self._estimate_params(llama_config)

        logger.info(
            "Building Llama model from scratch: "
            "hidden=%d, layers=%d, heads=%d, intermediate=%d, "
            "kv_heads=%d, vocab=%d, ~%.1fM params",
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            num_key_value_heads,
            vocab_size,
            approx_params / 1e6,
        )

        model = LlamaForCausalLM(llama_config)

        # Initialize weights with small std for stable training
        self._init_weights(model)

        self.model = model
        return model

    # ------------------------------------------------------------------ #
    # Tokenizer training
    # ------------------------------------------------------------------ #

    def train_tokenizer(
        self,
        corpus_path: str | Path | list[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
        output_dir: str | None = None,
    ) -> PreTrainedTokenizerFast:
        """Train a BPE tokenizer from a text corpus.

        Parameters
        ----------
        corpus_path : str, Path, or list[str]
            Path to a text file, directory of text files, or a list of
            file paths to use as the training corpus.
        vocab_size : int
            Target vocabulary size.
        min_frequency : int
            Minimum frequency for a token pair to be merged.
        special_tokens : list[str], optional
            Additional special tokens.  Default includes ``<pad>``, ``<s>``,
            ``</s>``, ``<unk>``.
        output_dir : str, optional
            Where to save the trained tokenizer.  Defaults to
            ``{config.training.output_dir}/tokenizer``.

        Returns
        -------
        PreTrainedTokenizerFast
            Trained BPE tokenizer wrapped in the HuggingFace fast tokenizer.
        """
        if not _TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers is required. Install with: pip install tokenizers")

        default_special = ["<pad>", "<s>", "</s>", "<unk>"]
        all_special = default_special + (special_tokens or [])

        # Build the BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Normalizer: NFC + lowercase is common; we keep case for LLMs
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.NFC(),
                normalizers.Replace(r"\s+", " "),
                normalizers.Strip(),
            ]
        )

        # Pre-tokenizer: byte-level like GPT-2
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processor
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=all_special,
            show_progress=True,
        )

        # Resolve corpus files
        corpus_files = self._resolve_corpus_files(corpus_path)
        if not corpus_files:
            raise FileNotFoundError(
                f"No text files found at '{corpus_path}'. "
                "Provide a .txt file, a directory of .txt files, or a list of paths."
            )

        logger.info(
            "Training BPE tokenizer: vocab_size=%d, corpus_files=%d",
            vocab_size,
            len(corpus_files),
        )

        tokenizer.train(files=corpus_files, trainer=trainer)

        # Wrap in PreTrainedTokenizerFast
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        # Save
        save_dir = Path(output_dir or os.path.join(self.config.training.output_dir, "tokenizer"))
        save_dir.mkdir(parents=True, exist_ok=True)
        fast_tokenizer.save_pretrained(str(save_dir))
        logger.info("Tokenizer saved to %s (vocab_size=%d)", save_dir, len(fast_tokenizer))

        self.tokenizer = fast_tokenizer
        return fast_tokenizer

    # ------------------------------------------------------------------ #
    # Pre-training
    # ------------------------------------------------------------------ #

    def train(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        config: Any | None = None,
        eval_dataset: Dataset | None = None,
        callbacks: list[Any] | None = None,
    ) -> Any:
        """Run causal-LM pre-training.

        Parameters
        ----------
        model : PreTrainedModel
            Model to train (should be randomly initialised).
        dataset : Dataset
            Training dataset.  Must contain ``"input_ids"`` or ``"text"``
            columns.
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
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        cfg = config or self.config
        training_cfg = cfg.training

        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Call train_tokenizer() or set self.tokenizer first."
            )

        # Tokenize dataset if it has a "text" column but not "input_ids"
        if "text" in dataset.column_names and "input_ids" not in dataset.column_names:
            dataset = self._tokenize_for_clm(dataset)

        if eval_dataset is not None and "text" in eval_dataset.column_names:
            if "input_ids" not in eval_dataset.column_names:
                eval_dataset = self._tokenize_for_clm(eval_dataset)

        # Data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # causal LM, not masked LM
        )

        # Training arguments with linear warmup + cosine decay
        training_args = TrainingArguments(
            output_dir=training_cfg.output_dir,
            num_train_epochs=training_cfg.num_epochs,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
            warmup_ratio=training_cfg.warmup_ratio,
            lr_scheduler_type="cosine",
            max_grad_norm=training_cfg.max_grad_norm,
            logging_steps=training_cfg.logging_steps,
            save_steps=training_cfg.save_steps,
            save_total_limit=training_cfg.save_total_limit,
            bf16=training_cfg.bf16,
            fp16=training_cfg.fp16,
            gradient_checkpointing=training_cfg.gradient_checkpointing,
            optim=training_cfg.optim,
            report_to=training_cfg.report_to,
            seed=cfg.data.seed if hasattr(cfg, "data") else 42,
            dataloader_num_workers=cfg.data.num_workers if hasattr(cfg, "data") else 4,
            remove_unused_columns=False,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=training_cfg.eval_steps if eval_dataset is not None else None,
            load_best_model_at_end=eval_dataset is not None,
        )

        # Override warmup steps if set explicitly
        if training_cfg.warmup_steps is not None:
            training_args.warmup_steps = training_cfg.warmup_steps
            training_args.warmup_ratio = 0.0

        # Gradient checkpointing kwargs
        if training_cfg.gradient_checkpointing:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks or [],
        )

        logger.info(
            "Starting pre-training: %d epochs, lr=%.2e, warmup_ratio=%.3f, scheduler=cosine",
            training_cfg.num_epochs,
            training_cfg.learning_rate,
            training_cfg.warmup_ratio,
        )

        result = trainer.train(resume_from_checkpoint=training_cfg.resume_from_checkpoint)

        # Save final model and tokenizer
        trainer.save_model(training_cfg.output_dir)
        self.tokenizer.save_pretrained(training_cfg.output_dir)
        logger.info("Pre-training complete. Model saved to %s", training_cfg.output_dir)

        return result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _tokenize_for_clm(self, dataset: Dataset) -> Dataset:
        """Tokenize a text dataset and group into fixed-length blocks for CLM."""
        max_length = self.config.model.max_seq_length

        def tokenize_function(examples: dict[str, list[str]]) -> dict[str, Any]:
            return self.tokenizer(
                examples["text"],
                truncation=False,
                add_special_tokens=True,
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing for CLM",
        )

        # Group texts into blocks of max_length
        def group_texts(examples: dict[str, list]) -> dict[str, list]:
            concatenated = {k: sum(examples[k], []) for k in examples}
            total_length = len(concatenated["input_ids"])
            # Drop the remainder
            total_length = (total_length // max_length) * max_length

            result = {
                k: [concatenated[k][i : i + max_length] for i in range(0, total_length, max_length)]
                for k in concatenated
            }
            result["labels"] = [ids[:] for ids in result["input_ids"]]
            return result

        grouped = tokenized.map(
            group_texts,
            batched=True,
            desc="Grouping into CLM blocks",
        )

        logger.info(
            "Tokenized and grouped: %d sequences of length %d",
            len(grouped),
            max_length,
        )
        return grouped

    @staticmethod
    def _resolve_corpus_files(
        corpus_path: str | Path | list[str],
    ) -> list[str]:
        """Resolve corpus path(s) into a list of file paths."""
        if isinstance(corpus_path, (list, tuple)):
            return [str(p) for p in corpus_path if Path(p).exists()]

        path = Path(corpus_path)
        if path.is_file():
            return [str(path)]
        if path.is_dir():
            return sorted(str(f) for f in path.rglob("*.txt") if f.is_file())
        return []

    @staticmethod
    def _estimate_params(config: LlamaConfig) -> int:
        """Rough parameter count estimate for a Llama model."""
        h = config.hidden_size
        l = config.num_hidden_layers
        v = config.vocab_size
        i = config.intermediate_size
        kv = config.num_key_value_heads
        q = config.num_attention_heads

        # Embeddings
        embed = v * h * 2  # input + output embeddings (no tying)

        # Per-layer: attention + FFN + layernorms
        head_dim = h // q
        attn_params = h * q * head_dim + h * kv * head_dim * 2 + h * h  # q,k,v,o
        ffn_params = h * i * 3  # gate, up, down (gated SiLU)
        norm_params = h * 2  # two RMSNorm per layer
        per_layer = attn_params + ffn_params + norm_params

        total = embed + per_layer * l + h  # final norm
        return total

    @staticmethod
    def _init_weights(model: PreTrainedModel) -> None:
        """Initialise model weights with small standard deviation.

        Uses the same initialisation strategy as the original Llama paper:
        normal distribution with std = 0.02.
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
