"""MLX-based fine-tuning engine for Apple Silicon.

Uses Apple's ``mlx-lm`` package to train LLMs natively on M-series chips,
leveraging unified memory for efficient LoRA / DoRA / full fine-tuning.

This module mirrors the :class:`FineTuner` API surface so the pipeline
can transparently swap between PyTorch and MLX backends based on config.

Requires: ``pip install 'mlx-lm[train]'``  (macOS Apple Silicon only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_forge.utils.logging import get_logger

logger = get_logger("training.mlx_trainer")

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import mlx.core as mx
    import mlx.optimizers as opt

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

try:
    from mlx_lm import generate, load
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    _MLX_LM_AVAILABLE = True
except ImportError:
    _MLX_LM_AVAILABLE = False


def is_mlx_available() -> bool:
    """Return True if MLX and mlx-lm are installed and usable."""
    return _MLX_AVAILABLE and _MLX_LM_AVAILABLE


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _prepare_jsonl_data(
    dataset: Any,
    output_dir: Path,
    input_field: str = "instruction",
    output_field: str = "output",
    system_prompt: str | None = None,
    max_samples: int | None = None,
) -> tuple[Path, Path | None]:
    """Convert a HuggingFace dataset to MLX-compatible JSONL files.

    MLX-lm expects ``train.jsonl`` and optionally ``valid.jsonl`` in a
    data directory.  Each line is a JSON object with a ``messages`` list
    in chat format.

    Returns (train_path, valid_path).
    """
    data_dir = output_dir / "mlx_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    def _to_messages(example: dict[str, Any]) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Try chat format first (messages field)
        if "messages" in example:
            return {"messages": example["messages"]}

        # Alpaca/instruction format
        instruction = example.get(input_field, example.get("instruction", ""))
        inp = example.get("input", "")
        response = example.get(output_field, example.get("output", ""))

        user_content = instruction
        if inp:
            user_content = f"{instruction}\n\n{inp}"

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": response})
        return {"messages": messages}

    def _write_jsonl(data: Any, path: Path, max_n: int | None = None) -> int:
        count = 0
        with open(path, "w") as f:
            for i, example in enumerate(data):
                if max_n is not None and i >= max_n:
                    break
                msg = _to_messages(example)
                if msg["messages"]:
                    f.write(json.dumps(msg) + "\n")
                    count += 1
        return count

    # Handle HF Dataset with train/test splits
    if hasattr(dataset, "keys") and callable(dataset.keys):
        # DatasetDict
        if "train" in dataset:
            n = _write_jsonl(dataset["train"], train_path, max_samples)
            logger.info("Wrote %d training examples to %s", n, train_path)
        if "validation" in dataset or "test" in dataset:
            val_split = dataset.get("validation", dataset.get("test"))
            n = _write_jsonl(val_split, valid_path)
            logger.info("Wrote %d validation examples to %s", n, valid_path)
            return train_path, valid_path
        return train_path, None

    # Single dataset (list-like)
    n = _write_jsonl(dataset, train_path, max_samples)
    logger.info("Wrote %d training examples to %s", n, train_path)
    return train_path, None


# ---------------------------------------------------------------------------
# MLXTrainer class
# ---------------------------------------------------------------------------


class MLXTrainer:
    """Fine-tuning engine using Apple's MLX framework.

    Parameters
    ----------
    config : Any
        An ``LLMForgeConfig`` instance.  The ``config.mlx`` sub-config
        controls all MLX-specific parameters.
    """

    def __init__(self, config: Any) -> None:
        if not is_mlx_available():
            raise ImportError(
                "MLX training requires 'mlx-lm'.  Install with:\n"
                "  pip install 'mlx-lm[train]'\n"
                "Note: MLX only works on macOS with Apple Silicon (M1+)."
            )

        self.config = config
        self.mlx_cfg = config.mlx
        self.model: Any = None
        self.tokenizer: Any = None
        self._output_dir = Path(config.training.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def setup_model(self) -> tuple[Any, Any]:
        """Load the model and tokenizer via mlx-lm.

        Returns (model, tokenizer).
        """
        model_name = self.config.model.name
        logger.info("Loading model '%s' with MLX backend", model_name)

        model, tokenizer = load(model_name)

        logger.info(
            "Model loaded — %s parameters",
            sum(p.size for p in model.parameters().values())
            if hasattr(model, "parameters")
            else "unknown",
        )

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    # ------------------------------------------------------------------
    # LoRA application
    # ------------------------------------------------------------------

    def apply_lora(self, model: Any) -> Any:
        """Freeze base weights and apply LoRA adapters.

        If the loaded model uses quantised layers (e.g., 4-bit), this
        automatically becomes QLoRA — the base stays quantised while
        LoRA matrices are in full precision.
        """
        cfg = self.mlx_cfg
        fine_tune_type = cfg.fine_tune_type

        if fine_tune_type == "full":
            logger.info("Full fine-tuning mode — skipping LoRA application")
            return model

        use_dora = fine_tune_type == "dora"

        lora_config = {
            "rank": cfg.lora_rank,
            "scale": cfg.lora_scale,
            "dropout": cfg.lora_dropout,
        }

        model.freeze()
        linear_to_lora_layers(
            model,
            num_layers=cfg.num_layers,
            config=lora_config,
            use_dora=use_dora,
        )

        # Count trainable parameters
        trainable = (
            sum(p.size for name, p in model.trainable_parameters().items())
            if hasattr(model, "trainable_parameters")
            else 0
        )

        logger.info(
            "Applied %s (rank=%d, scale=%.1f, layers=%d) — %s trainable params",
            "DoRA" if use_dora else "LoRA",
            cfg.lora_rank,
            cfg.lora_scale,
            cfg.num_layers,
            f"{trainable:,}",
        )

        return model

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> Any:
        """Build the MLX optimizer with optional LR schedule."""
        cfg = self.mlx_cfg
        lr = cfg.learning_rate

        # Build learning rate schedule if configured
        if cfg.lr_schedule and hasattr(opt.schedulers, cfg.lr_schedule):
            schedule_fn = getattr(opt.schedulers, cfg.lr_schedule)
            # cosine_decay(init, decay_steps, end)
            main_schedule = schedule_fn(lr, cfg.iters, lr * 0.01)

            if cfg.warmup_steps > 0:
                warmup_schedule = opt.schedulers.linear_schedule(lr * 0.01, lr, cfg.warmup_steps)
                lr = opt.schedulers.join_schedules(
                    [warmup_schedule, main_schedule],
                    [cfg.warmup_steps + 1],
                )
            else:
                lr = main_schedule

        # Build optimizer
        optimizer_map = {
            "adam": opt.Adam,
            "adamw": opt.AdamW,
            "sgd": opt.SGD,
            "adafactor": opt.Adafactor,
        }
        opt_cls = optimizer_map.get(cfg.optimizer, opt.Adam)
        optimizer = opt_cls(learning_rate=lr)

        logger.info(
            "Optimizer: %s, LR: %s, schedule: %s, warmup: %d steps",
            cfg.optimizer,
            cfg.learning_rate,
            cfg.lr_schedule or "constant",
            cfg.warmup_steps,
        )

        return optimizer

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        model: Any,
        dataset: Any,
        eval_dataset: Any = None,
    ) -> dict[str, Any]:
        """Run MLX fine-tuning.

        Parameters
        ----------
        model : mlx.nn.Module
            The (possibly LoRA-wrapped) model.
        dataset : Any
            Training dataset (HF Dataset or list of dicts).
        eval_dataset : Any, optional
            Validation dataset.

        Returns
        -------
        dict
            Training result with metrics.
        """
        cfg = self.mlx_cfg
        output_dir = self._output_dir

        # Prepare data in MLX-lm format
        input_field = getattr(self.config.data, "input_field", "instruction")
        output_field = getattr(self.config.data, "output_field", "output")

        train_path, valid_path = _prepare_jsonl_data(
            dataset,
            output_dir,
            input_field=input_field,
            output_field=output_field,
            max_samples=getattr(self.config.data, "max_samples", None),
        )

        # If we have a separate eval dataset, write it too
        if eval_dataset is not None and valid_path is None:
            valid_path = output_dir / "mlx_data" / "valid.jsonl"
            count = 0
            with open(valid_path, "w") as f:
                for example in eval_dataset:
                    messages = []
                    instruction = example.get(input_field, example.get("instruction", ""))
                    inp = example.get("input", "")
                    response = example.get(output_field, example.get("output", ""))
                    user_content = instruction
                    if inp:
                        user_content = f"{instruction}\n\n{inp}"
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": response})
                    f.write(json.dumps({"messages": messages}) + "\n")
                    count += 1
            logger.info("Wrote %d eval examples to %s", count, valid_path)

        # Load datasets via mlx-lm's loader
        import types

        data_args = types.SimpleNamespace(
            data=str(train_path.parent),
            hf_dataset=None,
            train=True,
            test=False,
        )

        from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset

        train_ds, valid_ds, _ = mlx_load_dataset(data_args, self.tokenizer)

        # Build optimizer
        optimizer = self._build_optimizer()

        # Build training args
        adapter_dir = output_dir / cfg.adapter_path
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = str(adapter_dir / "adapters.safetensors")

        training_args = TrainingArgs(
            batch_size=cfg.batch_size,
            iters=cfg.iters,
            val_batches=25,
            steps_per_report=cfg.steps_per_report,
            steps_per_eval=cfg.steps_per_eval,
            steps_per_save=cfg.steps_per_save,
            max_seq_length=cfg.max_seq_length,
            adapter_file=adapter_file,
            grad_checkpoint=cfg.grad_checkpoint,
            grad_accumulation_steps=cfg.grad_accumulation_steps,
        )

        logger.info(
            "Starting MLX training: %d iters, batch=%d, seq_len=%d",
            cfg.iters,
            cfg.batch_size,
            cfg.max_seq_length,
        )

        # Train
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            val_dataset=valid_ds,
            args=training_args,
        )

        logger.info("MLX training complete — adapters saved to %s", adapter_file)

        return {
            "adapter_path": adapter_file,
            "iters": cfg.iters,
            "output_dir": str(output_dir),
        }

    # ------------------------------------------------------------------
    # Fuse adapters
    # ------------------------------------------------------------------

    def fuse_and_save(self, model: Any) -> Path:
        """Fuse LoRA adapters into the base model and save.

        Returns the path to the fused model directory.
        """
        fused_dir = self._output_dir / "fused_model"
        fused_dir.mkdir(parents=True, exist_ok=True)

        try:
            from mlx_lm.fuse import fuse as mlx_fuse

            adapter_path = str(self._output_dir / self.mlx_cfg.adapter_path)
            mlx_fuse(
                model=self.config.model.name,
                adapter_path=adapter_path,
                save_path=str(fused_dir),
            )
            logger.info("Fused model saved to %s", fused_dir)
        except ImportError:
            # Fallback: fuse manually by calling .fuse() on each LoRA layer
            logger.info("mlx_lm.fuse not available — using manual fusion")
            self._manual_fuse(model, fused_dir)

        return fused_dir

    def _manual_fuse(self, model: Any, save_dir: Path) -> None:
        """Manually fuse LoRA layers by calling .fuse() on each one."""
        try:
            from mlx_lm.tuner.lora import LoRALinear

            fused_count = 0
            for _name, module in model.named_modules():
                if isinstance(module, LoRALinear) and hasattr(module, "fuse"):
                    module.fuse()
                    fused_count += 1

            logger.info("Fused %d LoRA layers", fused_count)

            # Save the fused weights
            weights = dict(model.parameters())
            mx.savez(str(save_dir / "weights.npz"), **weights)
        except Exception as exc:
            logger.warning("Manual fusion failed: %s", exc)

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """Generate text using the MLX model (with adapters if loaded)."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded — call setup_model() first")

        result = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
        )
        return result
