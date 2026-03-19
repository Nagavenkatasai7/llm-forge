"""Tests for the FineTuner class.

Most actual training tests require a GPU and real model weights.
This file tests that the class exists, can be instantiated with a
config object, and that GPU-dependent methods are properly guarded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from llm_forge.training.finetuner import _TORCH_AVAILABLE, FineTuner

    _MODULE_AVAILABLE = True
except ImportError:
    _MODULE_AVAILABLE = False
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODULE_AVAILABLE,
    reason="llm_forge.training.finetuner not importable",
)


# ===================================================================
# Instantiation
# ===================================================================


class TestFineTunerInstantiation:
    """Test that FineTuner can be created."""

    def test_class_exists(self) -> None:
        assert FineTuner is not None

    def test_instantiate_with_mock_config(self) -> None:
        """FineTuner accepts any config-like object."""
        mock_config = MagicMock()
        mock_config.model.name = "gpt2"
        mock_config.training.mode = "lora"

        ft = FineTuner(config=mock_config)
        assert ft.config is mock_config
        assert ft.model is None
        assert ft.tokenizer is None

    def test_instantiate_with_real_config(self, minimal_config_dict: dict) -> None:
        """FineTuner can be created with an LLMForgeConfig."""
        from llm_forge.config.schema import LLMForgeConfig

        config = LLMForgeConfig(**minimal_config_dict)
        ft = FineTuner(config=config)
        assert ft.config is config


# ===================================================================
# Attribute checks
# ===================================================================


class TestFineTunerAttributes:
    """Test expected attributes are present."""

    def test_has_setup_model_method(self) -> None:
        assert hasattr(FineTuner, "setup_model")

    def test_has_apply_lora_method(self) -> None:
        assert hasattr(FineTuner, "apply_lora")

    def test_has_train_method(self) -> None:
        assert hasattr(FineTuner, "train")

    def test_has_merge_and_save_method(self) -> None:
        assert hasattr(FineTuner, "merge_and_save")


# ===================================================================
# GPU-dependent tests (skipped by default on CPU)
# ===================================================================

_gpu_available = False
try:
    import torch

    _gpu_available = torch.cuda.is_available()
except ImportError:
    pass


@pytest.mark.skipif(not _gpu_available, reason="GPU not available")
class TestFineTunerGPU:
    """GPU-only tests for FineTuner."""

    @pytest.mark.gpu
    def test_setup_model_with_small_model(self) -> None:
        """Test loading a very small model on GPU."""
        from llm_forge.config.schema import LLMForgeConfig

        config = LLMForgeConfig(
            model={"name": "sshleifer/tiny-gpt2", "attn_implementation": "eager"},
            data={"train_path": "dummy"},
            training={"mode": "lora", "bf16": False, "fp16": False},
        )
        ft = FineTuner(config=config)
        model, tokenizer = ft.setup_model()
        assert model is not None
        assert tokenizer is not None

    @pytest.mark.gpu
    def test_merge_and_save_without_model_raises(self) -> None:
        """Calling merge_and_save without a model raises ValueError."""
        mock_config = MagicMock()
        mock_config.training.output_dir = "/tmp/test_output"
        ft = FineTuner(config=mock_config)
        with pytest.raises(ValueError, match="No model"):
            ft.merge_and_save()
