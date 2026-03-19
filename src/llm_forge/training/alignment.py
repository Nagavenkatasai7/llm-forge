"""Alignment training engine for DPO and RLHF.

Provides preference-based training using Direct Preference Optimisation (DPO)
via TRL's ``DPOTrainer``, and a structured PPO-based RLHF pipeline.
"""

from __future__ import annotations

from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.alignment")

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
        AutoModelForSequenceClassification,
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
    from trl import DPOConfig, DPOTrainer

    _DPO_AVAILABLE = True
except (ImportError, RuntimeError):
    _DPO_AVAILABLE = False

try:
    from trl import PPOConfig, PPOTrainer

    _PPO_AVAILABLE = True
except (ImportError, RuntimeError):
    _PPO_AVAILABLE = False

try:
    from trl import ORPOConfig, ORPOTrainer

    _ORPO_AVAILABLE = True
except (ImportError, RuntimeError):
    _ORPO_AVAILABLE = False

try:
    from trl import GRPOConfig, GRPOTrainer

    _GRPO_AVAILABLE = True
except (ImportError, RuntimeError):
    _GRPO_AVAILABLE = False

try:
    from datasets import Dataset

    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Precision mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, Any] = {}
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
# AlignmentTrainer
# ============================================================================


class AlignmentTrainer:
    """Alignment training engine supporting DPO and RLHF.

    DPO (Direct Preference Optimisation) trains the model directly on
    preference pairs (chosen / rejected) without needing a separate reward
    model.  RLHF uses Proximal Policy Optimisation with an explicit reward
    model.

    Parameters
    ----------
    config : object
        An ``LLMForgeConfig`` instance.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model: PreTrainedModel | None = None
        self.ref_model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.reward_model: PreTrainedModel | None = None

    # ================================================================== #
    # DPO Setup
    # ================================================================== #

    def setup_dpo(
        self,
        model: PreTrainedModel | None = None,
        config: Any | None = None,
    ) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]:
        """Configure DPO training by loading the policy model and reference model.

        Parameters
        ----------
        model : PreTrainedModel, optional
            Pre-loaded policy model.  If ``None``, loads from config.
        config : object, optional
            Override config.

        Returns
        -------
        tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizerBase]
            ``(policy_model, reference_model, tokenizer)``
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        cfg = config or self.config
        model_cfg = cfg.model
        training_cfg = cfg.training

        model_name = model_cfg.name
        torch_dtype = _DTYPE_MAP.get(str(model_cfg.torch_dtype), torch.bfloat16)
        trust_remote_code = model_cfg.trust_remote_code

        # Load tokenizer
        logger.info("Loading tokenizer for DPO: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            model_max_length=model_cfg.max_seq_length,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"  # left padding for DPO

        self.tokenizer = tokenizer

        # Build model kwargs
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
        }

        if model_cfg.attn_implementation:
            model_kwargs["attn_implementation"] = model_cfg.attn_implementation

        # QLoRA quantization for DPO
        if training_cfg.mode == "qlora" or (
            hasattr(cfg, "quantization") and cfg.quantization.load_in_4bit
        ):
            quant_cfg = cfg.quantization
            compute_dtype = _DTYPE_MAP.get(str(quant_cfg.bnb_4bit_compute_dtype), torch.bfloat16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
            )
            model_kwargs["quantization_config"] = bnb_config

        # Load policy model
        if model is None:
            logger.info("Loading policy model: %s", model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model = model

        # Load reference model (frozen copy)
        # For DPO with PEFT, TRL can handle the reference model internally
        # by sharing weights.  When not using PEFT, we load a separate copy.
        if _PEFT_AVAILABLE and training_cfg.mode in ("lora", "qlora", "dpo"):
            logger.info("Reference model will use PEFT weight sharing")
            self.ref_model = None  # DPOTrainer handles this when model is PeftModel
        else:
            logger.info("Loading separate reference model: %s", model_name)
            ref_kwargs = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
                "device_map": "auto",
            }
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_name, **ref_kwargs)
            # Freeze the reference model
            for param in self.ref_model.parameters():
                param.requires_grad = False

        return self.model, self.ref_model, tokenizer

    # ================================================================== #
    # Preference dataset preparation
    # ================================================================== #

    def prepare_preference_dataset(
        self,
        dataset: Dataset,
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
    ) -> Dataset:
        """Prepare a preference dataset for DPO training.

        The dataset must contain columns for the prompt, chosen response,
        and rejected response.  This method validates and normalises the
        column names.

        Parameters
        ----------
        dataset : Dataset
            Raw preference dataset.
        prompt_field : str
            Column containing the prompt text.
        chosen_field : str
            Column containing the preferred response.
        rejected_field : str
            Column containing the dispreferred response.

        Returns
        -------
        Dataset
            Dataset with ``prompt``, ``chosen``, and ``rejected`` columns.
        """
        required = {prompt_field, chosen_field, rejected_field}
        available = set(dataset.column_names)

        missing = required - available
        if missing:
            raise ValueError(
                f"Missing columns in preference dataset: {missing}. "
                f"Available: {available}. "
                f"Expected: prompt={prompt_field}, chosen={chosen_field}, "
                f"rejected={rejected_field}"
            )

        # Rename columns to standard names if needed
        rename_map = {}
        if prompt_field != "prompt":
            rename_map[prompt_field] = "prompt"
        if chosen_field != "chosen":
            rename_map[chosen_field] = "chosen"
        if rejected_field != "rejected":
            rename_map[rejected_field] = "rejected"

        if rename_map:
            dataset = dataset.rename_columns(rename_map)

        # Validate non-empty rows
        def validate_row(example: dict[str, Any]) -> bool:
            return bool(example.get("prompt") and example.get("chosen") and example.get("rejected"))

        initial_len = len(dataset)
        dataset = dataset.filter(validate_row, desc="Validating preference pairs")
        filtered = initial_len - len(dataset)

        if filtered > 0:
            logger.warning(
                "Filtered %d empty preference pairs (kept %d/%d)",
                filtered,
                len(dataset),
                initial_len,
            )

        logger.info("Preference dataset prepared: %d pairs", len(dataset))
        return dataset

    # ================================================================== #
    # DPO Training
    # ================================================================== #

    def train_dpo(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        config: Any | None = None,
        eval_dataset: Dataset | None = None,
        callbacks: list[Any] | None = None,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ) -> Any:
        """Train the model using DPO on a preference dataset.

        Parameters
        ----------
        model : PreTrainedModel
            Policy model to train.
        dataset : Dataset
            Preference dataset with ``prompt``, ``chosen``, ``rejected``.
        config : object, optional
            Override config.
        eval_dataset : Dataset, optional
            Evaluation preference dataset.
        callbacks : list, optional
            Additional HF Trainer callbacks.
        beta : float
            KL divergence regularisation coefficient.  Higher values
            keep the policy closer to the reference model.  Typical
            range: 0.05-0.5.
        loss_type : str
            DPO loss variant: ``"sigmoid"`` (standard), ``"hinge"``,
            ``"ipo"``, ``"kto_pair"``.
        max_length : int
            Maximum combined length of prompt + response.
        max_prompt_length : int
            Maximum length of the prompt portion.

        Returns
        -------
        TrainOutput
            Training result from the DPO Trainer.
        """
        if not _DPO_AVAILABLE:
            raise ImportError("trl is required for DPO training. Install with: pip install trl")

        cfg = config or self.config
        training_cfg = cfg.training

        # Apply LoRA if configured
        if training_cfg.mode in ("lora", "qlora", "dpo") and _PEFT_AVAILABLE:
            lora_cfg = cfg.lora
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
                bias=lora_cfg.bias,
                task_type=TaskType.CAUSAL_LM,
                use_rslora=lora_cfg.use_rslora,
                use_dora=lora_cfg.use_dora,
            )
            logger.info("Applying LoRA for DPO: r=%d, alpha=%d", lora_cfg.r, lora_cfg.alpha)
        else:
            peft_config = None

        # Build DPOConfig
        dpo_config = DPOConfig(
            output_dir=training_cfg.output_dir,
            num_train_epochs=training_cfg.num_epochs,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
            warmup_ratio=training_cfg.warmup_ratio,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
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
            remove_unused_columns=False,
            beta=beta,
            loss_type=loss_type,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )

        if training_cfg.warmup_steps is not None:
            dpo_config.warmup_steps = training_cfg.warmup_steps

        if training_cfg.gradient_checkpointing:
            dpo_config.gradient_checkpointing_kwargs = {"use_reentrant": False}

        # Eval strategy
        if eval_dataset is not None:
            dpo_config.eval_strategy = training_cfg.eval_strategy
            if training_cfg.eval_steps is not None:
                dpo_config.eval_steps = training_cfg.eval_steps

        # Build DPOTrainer
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": dpo_config,
            "train_dataset": dataset,
            "processing_class": self.tokenizer,
        }

        if self.ref_model is not None:
            trainer_kwargs["ref_model"] = self.ref_model

        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer = DPOTrainer(**trainer_kwargs)

        logger.info(
            "Starting DPO training: beta=%.3f, loss=%s, max_len=%d, max_prompt=%d, "
            "epochs=%d, lr=%.2e",
            beta,
            loss_type,
            max_length,
            max_prompt_length,
            training_cfg.num_epochs,
            training_cfg.learning_rate,
        )

        result = trainer.train(resume_from_checkpoint=training_cfg.resume_from_checkpoint)

        # Save
        trainer.save_model(training_cfg.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(training_cfg.output_dir)

        logger.info("DPO training complete. Model saved to %s", training_cfg.output_dir)
        return result

    # ================================================================== #
    # ORPO Training
    # ================================================================== #

    def train_orpo(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        config: Any | None = None,
        eval_dataset: Dataset | None = None,
        callbacks: list[Any] | None = None,
        beta: float = 0.1,
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ) -> Any:
        """Train the model using ORPO (Odds Ratio Preference Optimisation).

        ORPO combines supervised fine-tuning with an odds-ratio preference loss
        in a single stage.  Unlike DPO it does not need a reference model, halving
        the GPU memory requirement.

        Parameters
        ----------
        model : PreTrainedModel
            Policy model to train.
        dataset : Dataset
            Preference dataset with ``prompt``, ``chosen``, ``rejected``.
        config : object, optional
            Override config.
        eval_dataset : Dataset, optional
            Evaluation preference dataset.
        callbacks : list, optional
            Additional HF Trainer callbacks.
        beta : float
            Odds-ratio loss weight.  Typical range: 0.05-0.5.
        max_length : int
            Maximum combined length of prompt + response.
        max_prompt_length : int
            Maximum length of the prompt portion.

        Returns
        -------
        TrainOutput
            Training result from the ORPO Trainer.
        """
        if not _ORPO_AVAILABLE:
            raise ImportError(
                "ORPOTrainer requires trl>=0.24.0. Install with: pip install 'trl>=0.24.0'"
            )

        cfg = config or self.config
        training_cfg = cfg.training

        # Apply LoRA if configured
        if _PEFT_AVAILABLE and hasattr(cfg, "lora"):
            lora_cfg = cfg.lora
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
                bias=lora_cfg.bias,
                task_type=TaskType.CAUSAL_LM,
                use_rslora=lora_cfg.use_rslora,
                use_dora=lora_cfg.use_dora,
            )
            logger.info(
                "Applying LoRA for ORPO: r=%d, alpha=%d",
                lora_cfg.r,
                lora_cfg.alpha,
            )
        else:
            peft_config = None

        # Build ORPOConfig
        orpo_config = ORPOConfig(
            output_dir=training_cfg.output_dir,
            num_train_epochs=training_cfg.num_epochs,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
            warmup_ratio=training_cfg.warmup_ratio,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
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
            remove_unused_columns=False,
            beta=beta,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )

        if training_cfg.warmup_steps is not None:
            orpo_config.warmup_steps = training_cfg.warmup_steps

        if training_cfg.gradient_checkpointing:
            orpo_config.gradient_checkpointing_kwargs = {"use_reentrant": False}

        if eval_dataset is not None:
            orpo_config.eval_strategy = training_cfg.eval_strategy
            if training_cfg.eval_steps is not None:
                orpo_config.eval_steps = training_cfg.eval_steps

        # Build ORPOTrainer — no reference model needed
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": orpo_config,
            "train_dataset": dataset,
            "processing_class": self.tokenizer,
        }

        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer = ORPOTrainer(**trainer_kwargs)

        logger.info(
            "Starting ORPO training: beta=%.3f, max_len=%d, max_prompt=%d, epochs=%d, lr=%.2e",
            beta,
            max_length,
            max_prompt_length,
            training_cfg.num_epochs,
            training_cfg.learning_rate,
        )

        result = trainer.train(resume_from_checkpoint=training_cfg.resume_from_checkpoint)

        trainer.save_model(training_cfg.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(training_cfg.output_dir)

        logger.info("ORPO training complete. Model saved to %s", training_cfg.output_dir)
        return result

    # ================================================================== #
    # GRPO (Group Relative Policy Optimisation) Training
    # ================================================================== #

    def train_grpo(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        config: Any | None = None,
        eval_dataset: Dataset | None = None,
        callbacks: list[Any] | None = None,
        num_generations: int = 4,
        max_completion_length: int = 256,
        beta: float = 0.1,
    ) -> Any:
        """Train using GRPO (Group Relative Policy Optimisation).

        GRPO generates multiple completions per prompt, scores them with a
        reward function, and uses the group-relative advantage for policy
        updates.  Unlike PPO, it does not require a separate critic model,
        making it more memory-efficient.

        Parameters
        ----------
        model : PreTrainedModel
            Policy model to train.
        dataset : Dataset
            Prompt dataset with a ``prompt`` column.
        config : object, optional
            Override config.
        eval_dataset : Dataset, optional
            Evaluation dataset.
        callbacks : list, optional
            Additional Trainer callbacks.
        num_generations : int
            Number of completions to generate per prompt for group scoring.
        max_completion_length : int
            Maximum length of generated completions.
        beta : float
            KL penalty coefficient.

        Returns
        -------
        TrainOutput
            Training result from the GRPO Trainer.
        """
        if not _GRPO_AVAILABLE:
            raise ImportError(
                "GRPOTrainer requires trl>=0.25.0. Install with: pip install 'trl>=0.25.0'"
            )

        cfg = config or self.config
        training_cfg = cfg.training

        # Apply LoRA if configured
        if _PEFT_AVAILABLE and hasattr(cfg, "lora"):
            lora_cfg = cfg.lora
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
                bias=lora_cfg.bias,
                task_type=TaskType.CAUSAL_LM,
                use_rslora=lora_cfg.use_rslora,
                use_dora=lora_cfg.use_dora,
            )
            logger.info(
                "Applying LoRA for GRPO: r=%d, alpha=%d",
                lora_cfg.r,
                lora_cfg.alpha,
            )
        else:
            peft_config = None

        # Build GRPOConfig
        grpo_config = GRPOConfig(
            output_dir=training_cfg.output_dir,
            num_train_epochs=training_cfg.num_epochs,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
            warmup_ratio=training_cfg.warmup_ratio,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
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
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            beta=beta,
        )

        if training_cfg.warmup_steps is not None:
            grpo_config.warmup_steps = training_cfg.warmup_steps

        if training_cfg.gradient_checkpointing:
            grpo_config.gradient_checkpointing_kwargs = {"use_reentrant": False}

        if eval_dataset is not None:
            grpo_config.eval_strategy = training_cfg.eval_strategy
            if training_cfg.eval_steps is not None:
                grpo_config.eval_steps = training_cfg.eval_steps

        # Build GRPOTrainer
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": grpo_config,
            "train_dataset": dataset,
            "processing_class": self.tokenizer,
        }

        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer = GRPOTrainer(**trainer_kwargs)

        logger.info(
            "Starting GRPO training: num_generations=%d, max_completion=%d, "
            "beta=%.3f, epochs=%d, lr=%.2e",
            num_generations,
            max_completion_length,
            beta,
            training_cfg.num_epochs,
            training_cfg.learning_rate,
        )

        result = trainer.train(resume_from_checkpoint=training_cfg.resume_from_checkpoint)

        trainer.save_model(training_cfg.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(training_cfg.output_dir)

        logger.info("GRPO training complete. Model saved to %s", training_cfg.output_dir)
        return result

    # ================================================================== #
    # RLHF (PPO) Training
    # ================================================================== #

    def setup_reward_model(
        self,
        reward_model_name: str,
        config: Any | None = None,
    ) -> PreTrainedModel:
        """Load a reward model for PPO-based RLHF.

        Parameters
        ----------
        reward_model_name : str
            HuggingFace model identifier for the reward model.  Should be
            a sequence classification model that outputs scalar rewards.
        config : object, optional
            Override config.

        Returns
        -------
        PreTrainedModel
            Loaded reward model.
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        cfg = config or self.config
        torch_dtype = _DTYPE_MAP.get(str(cfg.model.torch_dtype), torch.bfloat16)

        logger.info("Loading reward model: %s", reward_model_name)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            num_labels=1,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
        )

        # Freeze reward model
        for param in reward_model.parameters():
            param.requires_grad = False

        self.reward_model = reward_model
        return reward_model

    def train_rlhf(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        config: Any | None = None,
        reward_model: PreTrainedModel | None = None,
        callbacks: list[Any] | None = None,
        kl_penalty: str = "kl",
        init_kl_coef: float = 0.2,
        target_kl: float = 6.0,
        mini_batch_size: int = 4,
        ppo_epochs: int = 4,
    ) -> dict[str, Any]:
        """Train the model using PPO-based RLHF.

        This implements the basic RLHF pipeline:
        1. Generate responses from the policy model.
        2. Score responses with the reward model.
        3. Update the policy using PPO.

        Parameters
        ----------
        model : PreTrainedModel
            Policy model to train.
        dataset : Dataset
            Dataset with a ``"query"`` column containing prompts.
        config : object, optional
            Override config.
        reward_model : PreTrainedModel, optional
            Reward model.  Falls back to ``self.reward_model``.
        callbacks : list, optional
            Additional callbacks (not used directly by PPO but logged).
        kl_penalty : str
            KL penalty method: ``"kl"`` (default), ``"abs"``, ``"mse"``,
            ``"full"``.
        init_kl_coef : float
            Initial KL coefficient.
        target_kl : float
            Target KL divergence value.
        mini_batch_size : int
            Mini-batch size for PPO updates.
        ppo_epochs : int
            Number of PPO optimisation epochs per batch.

        Returns
        -------
        dict[str, Any]
            Summary statistics from the RLHF training run.
        """
        if not _PPO_AVAILABLE:
            raise ImportError(
                "trl with PPO support is required for RLHF. Install with: pip install trl"
            )
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for RLHF training.")

        cfg = config or self.config
        training_cfg = cfg.training

        rm = reward_model or self.reward_model
        if rm is None:
            raise ValueError("No reward model available. Call setup_reward_model() first.")

        if self.tokenizer is None:
            raise ValueError("No tokenizer available. Call setup_dpo() or load a tokenizer first.")

        # Ensure the query column exists
        if "query" not in dataset.column_names:
            # Try to use 'prompt' or 'text' as fallback
            if "prompt" in dataset.column_names:
                dataset = dataset.rename_column("prompt", "query")
            elif "text" in dataset.column_names:
                dataset = dataset.rename_column("text", "query")
            else:
                raise ValueError("Dataset must contain a 'query' (or 'prompt' or 'text') column.")

        # PPO Config
        ppo_config = PPOConfig(
            learning_rate=training_cfg.learning_rate,
            batch_size=training_cfg.per_device_train_batch_size,
            mini_batch_size=mini_batch_size,
            ppo_epochs=ppo_epochs,
            init_kl_coef=init_kl_coef,
            target_kl=target_kl,
            kl_penalty=kl_penalty,
            seed=cfg.data.seed if hasattr(cfg, "data") else 42,
            log_with=training_cfg.report_to[0] if training_cfg.report_to else None,
            output_dir=training_cfg.output_dir,
        )

        # Build PPO Trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        logger.info(
            "Starting RLHF (PPO) training: kl_penalty=%s, init_kl=%.3f, "
            "target_kl=%.1f, ppo_epochs=%d",
            kl_penalty,
            init_kl_coef,
            target_kl,
            ppo_epochs,
        )

        # Training loop
        generation_kwargs = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        all_stats: list[dict[str, float]] = []
        total_steps = 0

        for epoch in range(training_cfg.num_epochs):
            logger.info("RLHF Epoch %d/%d", epoch + 1, training_cfg.num_epochs)

            for batch_idx in range(0, len(dataset), ppo_config.batch_size):
                batch_end = min(batch_idx + ppo_config.batch_size, len(dataset))
                batch = dataset.select(range(batch_idx, batch_end))

                # Tokenize queries
                query_tensors = []
                for query_text in batch["query"]:
                    encoded = self.tokenizer.encode(
                        query_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=cfg.model.max_seq_length // 2,
                    )
                    query_tensors.append(encoded.squeeze(0))

                # Generate responses
                response_tensors = []
                for query in query_tensors:
                    query_device = query.to(model.device if hasattr(model, "device") else "cpu")
                    gen_output = model.generate(
                        query_device.unsqueeze(0),
                        **generation_kwargs,
                    )
                    response = gen_output[0][len(query_device) :]
                    response_tensors.append(response)

                # Compute rewards
                rewards = []
                for query, response in zip(query_tensors, response_tensors, strict=False):
                    full_text = self.tokenizer.decode(
                        torch.cat([query, response]), skip_special_tokens=True
                    )
                    reward_inputs = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=cfg.model.max_seq_length,
                    ).to(rm.device if hasattr(rm, "device") else "cpu")
                    with torch.no_grad():
                        reward_output = rm(**reward_inputs)
                        reward_value = reward_output.logits.squeeze().float()
                    rewards.append(reward_value)

                # PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                all_stats.append(stats)
                total_steps += 1

                if total_steps % training_cfg.logging_steps == 0:
                    mean_reward = sum(r.item() for r in rewards) / len(rewards)
                    logger.info(
                        "Step %d: mean_reward=%.4f",
                        total_steps,
                        mean_reward,
                    )

        # Save final model
        output_dir = training_cfg.output_dir
        model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        logger.info("RLHF training complete. Model saved to %s", output_dir)
        return {
            "total_steps": total_steps,
            "num_epochs": training_cfg.num_epochs,
            "stats": all_stats,
        }
