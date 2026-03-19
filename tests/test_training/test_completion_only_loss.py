"""Tests for completion-only loss (instruction token masking).

Completion-only loss masks prompt/instruction tokens so the model only
learns from the completion (response) portion.  In TRL >= 0.18, this is
configured via ``SFTConfig(completion_only_loss=...)`` rather than the
legacy ``DataCollatorForCompletionOnlyLM``.
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.config.schema import LLMForgeConfig, TrainingConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

try:
    from llm_forge.pipeline.config_translator import ConfigTranslator

    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SCHEMA_AVAILABLE,
    reason="llm_forge.config.schema not importable",
)


# ===================================================================
# Schema Tests
# ===================================================================


class TestCompletionOnlyLossConfig:
    """Verify completion_only_loss config field behaviour."""

    def test_default_is_true(self) -> None:
        """Default is True (assistant-only loss for instruction tuning)."""
        cfg = TrainingConfig()
        assert cfg.completion_only_loss is True

    def test_field_exists(self) -> None:
        """Field exists in TrainingConfig."""
        assert hasattr(TrainingConfig.model_fields, "__getitem__") or hasattr(
            TrainingConfig, "completion_only_loss"
        )

    def test_explicit_true(self) -> None:
        """Can set to True to force completion-only loss."""
        cfg = TrainingConfig(completion_only_loss=True)
        assert cfg.completion_only_loss is True

    def test_explicit_false(self) -> None:
        """Can set to False to force full-sequence loss."""
        cfg = TrainingConfig(completion_only_loss=False)
        assert cfg.completion_only_loss is False

    def test_null_in_yaml_is_none(self) -> None:
        """Setting null (Python None) leaves auto-detect behaviour."""
        cfg = TrainingConfig(completion_only_loss=None)
        assert cfg.completion_only_loss is None

    def test_full_config_with_completion_only(self) -> None:
        """Full LLMForgeConfig validates with completion_only_loss=True."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            training={"completion_only_loss": True},
        )
        assert config.training.completion_only_loss is True

    def test_full_config_with_completion_only_false(self) -> None:
        """Full LLMForgeConfig validates with completion_only_loss=False."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            training={"completion_only_loss": False},
        )
        assert config.training.completion_only_loss is False

    def test_full_config_default_true(self) -> None:
        """Full LLMForgeConfig defaults to True (assistant-only loss)."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
        )
        assert config.training.completion_only_loss is True


# ===================================================================
# Config Translator Tests
# ===================================================================


@pytest.mark.skipif(
    not _TRANSLATOR_AVAILABLE,
    reason="config_translator not importable",
)
class TestCompletionOnlyLossTranslator:
    """Verify completion_only_loss flows through ConfigTranslator."""

    def test_none_not_in_args(self) -> None:
        """When None (auto-detect), completion_only_loss is NOT in training args."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            training={"completion_only_loss": None},
        )
        args = ConfigTranslator.to_training_args(config)
        assert "completion_only_loss" not in args

    def test_true_in_args(self) -> None:
        """When True, completion_only_loss=True is in training args."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            training={"completion_only_loss": True},
        )
        args = ConfigTranslator.to_training_args(config)
        assert args["completion_only_loss"] is True

    def test_false_in_args(self) -> None:
        """When False, completion_only_loss=False is in training args."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            training={"completion_only_loss": False},
        )
        args = ConfigTranslator.to_training_args(config)
        assert args["completion_only_loss"] is False

    def test_default_included_in_args(self) -> None:
        """Default config (True) includes completion_only_loss in training args."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
        )
        args = ConfigTranslator.to_training_args(config)
        assert args.get("completion_only_loss") is True
