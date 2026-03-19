"""Tests for the MLX training backend.

Covers MLXConfig schema validation, data preparation helpers,
availability detection, and module structure.  Actual MLX training
is skipped unless mlx-lm is installed on Apple Silicon.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_forge.config.schema import LLMForgeConfig, MLXConfig

# ===================================================================
# MLXConfig schema tests
# ===================================================================


class TestMLXConfigSchema:
    """Test MLXConfig Pydantic model."""

    def test_defaults(self) -> None:
        cfg = MLXConfig()
        assert cfg.enabled is False
        assert cfg.fine_tune_type == "lora"
        assert cfg.num_layers == 16
        assert cfg.lora_rank == 8
        assert cfg.lora_scale == 20.0
        assert cfg.lora_dropout == 0.0
        assert cfg.iters == 1000
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 1e-5
        assert cfg.optimizer == "adam"
        assert cfg.max_seq_length == 2048
        assert cfg.grad_checkpoint is False
        assert cfg.grad_accumulation_steps == 1
        assert cfg.mask_prompt is True
        assert cfg.fuse_after_training is True

    def test_enabled_true(self) -> None:
        cfg = MLXConfig(enabled=True)
        assert cfg.enabled is True

    def test_fine_tune_type_dora(self) -> None:
        cfg = MLXConfig(fine_tune_type="dora")
        assert cfg.fine_tune_type == "dora"

    def test_fine_tune_type_full(self) -> None:
        cfg = MLXConfig(fine_tune_type="full")
        assert cfg.fine_tune_type == "full"

    def test_invalid_fine_tune_type(self) -> None:
        with pytest.raises(Exception):
            MLXConfig(fine_tune_type="invalid")

    def test_invalid_optimizer(self) -> None:
        with pytest.raises(Exception):
            MLXConfig(optimizer="rmsprop")

    def test_lora_rank_min(self) -> None:
        cfg = MLXConfig(lora_rank=1)
        assert cfg.lora_rank == 1

    def test_lora_rank_zero_fails(self) -> None:
        with pytest.raises(Exception):
            MLXConfig(lora_rank=0)

    def test_lora_dropout_bounds(self) -> None:
        MLXConfig(lora_dropout=0.0)
        MLXConfig(lora_dropout=1.0)
        with pytest.raises(Exception):
            MLXConfig(lora_dropout=-0.1)
        with pytest.raises(Exception):
            MLXConfig(lora_dropout=1.1)

    def test_all_layers_with_negative_one(self) -> None:
        cfg = MLXConfig(num_layers=-1)
        assert cfg.num_layers == -1

    def test_custom_adapter_path(self) -> None:
        cfg = MLXConfig(adapter_path="my_adapters")
        assert cfg.adapter_path == "my_adapters"


# ===================================================================
# MLXConfig in LLMForgeConfig
# ===================================================================


class TestMLXConfigInLLMForge:
    """Test MLXConfig integration in the top-level config."""

    def test_mlx_field_exists(self) -> None:
        assert "mlx" in LLMForgeConfig.model_fields

    def test_mlx_defaults_in_config(self) -> None:
        cfg = LLMForgeConfig(
            model={"name": "test/model"},
            data={"train_path": "test/data"},
        )
        assert cfg.mlx.enabled is False
        assert isinstance(cfg.mlx, MLXConfig)

    def test_mlx_enabled_from_yaml(self) -> None:
        data = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test/data"},
            "mlx": {
                "enabled": True,
                "lora_rank": 16,
                "iters": 500,
                "optimizer": "adamw",
            },
        }
        cfg = LLMForgeConfig(**data)
        assert cfg.mlx.enabled is True
        assert cfg.mlx.lora_rank == 16
        assert cfg.mlx.iters == 500
        assert cfg.mlx.optimizer == "adamw"

    def test_existing_configs_still_validate(self) -> None:
        """Adding MLXConfig doesn't break existing minimal configs."""
        cfg = LLMForgeConfig(
            model={"name": "test/model"},
            data={"train_path": "test/data"},
        )
        assert cfg.model.name == "test/model"


# ===================================================================
# Data preparation helper tests
# ===================================================================


_MLX_TRAINER_AVAILABLE = False
try:
    from llm_forge.training.mlx_trainer import _prepare_jsonl_data

    _MLX_TRAINER_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(
    not _MLX_TRAINER_AVAILABLE,
    reason="mlx_trainer not importable (likely missing rich/torch chain)",
)
class TestPrepareJsonlData:
    """Test the JSONL data preparation for MLX training."""

    def test_alpaca_format(self, tmp_path: Path) -> None:
        """Alpaca-format dicts are converted to chat messages."""
        data = [
            {"instruction": "What is Python?", "input": "", "output": "A programming language."},
            {"instruction": "Translate.", "input": "Hello", "output": "Hola"},
        ]
        train_path, valid_path = _prepare_jsonl_data(data, tmp_path)
        assert train_path.exists()
        lines = train_path.read_text().strip().split("\n")
        assert len(lines) == 2

        msg = json.loads(lines[0])
        assert "messages" in msg
        assert msg["messages"][-1]["role"] == "assistant"
        assert msg["messages"][-1]["content"] == "A programming language."

    def test_chat_format_passthrough(self, tmp_path: Path) -> None:
        """Data with 'messages' key is passed through directly."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        ]
        train_path, _ = _prepare_jsonl_data(data, tmp_path)
        msg = json.loads(train_path.read_text().strip())
        assert msg["messages"][0]["role"] == "user"

    def test_max_samples_limit(self, tmp_path: Path) -> None:
        """Only max_samples examples are written."""
        data = [{"instruction": f"Q{i}", "input": "", "output": f"A{i}"} for i in range(20)]
        train_path, _ = _prepare_jsonl_data(data, tmp_path, max_samples=5)
        lines = train_path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_system_prompt_added(self, tmp_path: Path) -> None:
        """System prompt is prepended to messages."""
        data = [{"instruction": "Q", "input": "", "output": "A"}]
        train_path, _ = _prepare_jsonl_data(data, tmp_path, system_prompt="You are helpful.")
        msg = json.loads(train_path.read_text().strip())
        assert msg["messages"][0]["role"] == "system"
        assert msg["messages"][0]["content"] == "You are helpful."

    def test_output_dir_created(self, tmp_path: Path) -> None:
        """mlx_data subdirectory is created."""
        data = [{"instruction": "Q", "input": "", "output": "A"}]
        _prepare_jsonl_data(data, tmp_path)
        assert (tmp_path / "mlx_data").is_dir()

    def test_with_input_field(self, tmp_path: Path) -> None:
        """Non-empty input field is appended to user content."""
        data = [{"instruction": "Translate", "input": "Hello", "output": "Hola"}]
        train_path, _ = _prepare_jsonl_data(data, tmp_path)
        msg = json.loads(train_path.read_text().strip())
        user_msg = msg["messages"][0]["content"]
        assert "Translate" in user_msg
        assert "Hello" in user_msg


# ===================================================================
# is_mlx_available() tests
# ===================================================================


class TestMLXAvailability:
    """Test MLX availability detection."""

    def test_returns_bool(self) -> None:
        """is_mlx_available returns a boolean."""
        try:
            from llm_forge.training import is_mlx_available

            result = is_mlx_available()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("training module not importable")

    def test_fallback_when_mlx_missing(self) -> None:
        """When mlx is not installed, returns False."""
        try:
            from llm_forge.training import is_mlx_available

            # On non-Apple-Silicon or without mlx-lm, should be False
            # (we don't force it — just check it's a valid bool)
            assert isinstance(is_mlx_available(), bool)
        except ImportError:
            pytest.skip("training module not importable")


# ===================================================================
# MLXTrainer class structure tests
# ===================================================================


@pytest.mark.skipif(
    not _MLX_TRAINER_AVAILABLE,
    reason="mlx_trainer not importable",
)
class TestMLXTrainerStructure:
    """Test MLXTrainer has the expected interface."""

    def test_class_exists(self) -> None:
        from llm_forge.training.mlx_trainer import MLXTrainer

        assert MLXTrainer is not None

    def test_has_setup_model(self) -> None:
        from llm_forge.training.mlx_trainer import MLXTrainer

        assert hasattr(MLXTrainer, "setup_model")

    def test_has_apply_lora(self) -> None:
        from llm_forge.training.mlx_trainer import MLXTrainer

        assert hasattr(MLXTrainer, "apply_lora")

    def test_has_train(self) -> None:
        from llm_forge.training.mlx_trainer import MLXTrainer

        assert hasattr(MLXTrainer, "train")

    def test_has_fuse_and_save(self) -> None:
        from llm_forge.training.mlx_trainer import MLXTrainer

        assert hasattr(MLXTrainer, "fuse_and_save")

    def test_has_generate_text(self) -> None:
        from llm_forge.training.mlx_trainer import MLXTrainer

        assert hasattr(MLXTrainer, "generate_text")
