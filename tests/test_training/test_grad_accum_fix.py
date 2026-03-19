"""Tests for gradient accumulation loss bug fix.

The bug: In transformers < 4.46.0, gradient_accumulation_steps > 1 causes
incorrect loss scaling (mean of per-micro-batch losses instead of
total_loss / total_non_padding_tokens).

The fix: transformers >= 4.46.0 computes num_items_in_batch and passes it
through the training loop. The average_tokens_across_devices parameter
controls whether token counts are synchronised across GPUs.

Ref: https://github.com/huggingface/transformers/issues/34242
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.config.schema import LLMForgeConfig, TrainingConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SCHEMA_AVAILABLE,
    reason="llm_forge.config.schema not importable",
)


# ===================================================================
# Schema / Config tests
# ===================================================================


class TestGradAccumConfig:
    """Verify the average_tokens_across_devices field in TrainingConfig."""

    def test_field_exists(self) -> None:
        """TrainingConfig has the average_tokens_across_devices field."""
        cfg = TrainingConfig()
        assert hasattr(cfg, "average_tokens_across_devices")

    def test_default_is_true(self) -> None:
        """Default value is True (fix is active by default)."""
        cfg = TrainingConfig()
        assert cfg.average_tokens_across_devices is True

    def test_can_disable(self) -> None:
        """Users can explicitly disable the fix."""
        cfg = TrainingConfig(average_tokens_across_devices=False)
        assert cfg.average_tokens_across_devices is False

    def test_can_enable_explicitly(self) -> None:
        """Users can explicitly enable the fix."""
        cfg = TrainingConfig(average_tokens_across_devices=True)
        assert cfg.average_tokens_across_devices is True

    def test_full_config_with_grad_accum(self) -> None:
        """Full config with gradient_accumulation_steps > 1 validates."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            training={
                "gradient_accumulation_steps": 8,
                "average_tokens_across_devices": True,
            },
        )
        assert config.training.gradient_accumulation_steps == 8
        assert config.training.average_tokens_across_devices is True

    def test_yaml_roundtrip(self, tmp_path) -> None:
        """Config with average_tokens_across_devices survives YAML roundtrip."""
        import yaml

        config_dict = {
            "model": {"name": "test-model"},
            "data": {"train_path": "test-data"},
            "training": {
                "gradient_accumulation_steps": 4,
                "average_tokens_across_devices": True,
            },
        }
        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)

        config = LLMForgeConfig(**loaded)
        assert config.training.average_tokens_across_devices is True
        assert config.training.gradient_accumulation_steps == 4


# ===================================================================
# Version requirement tests
# ===================================================================


class TestTransformersVersion:
    """Verify the minimum transformers version for the fix."""

    def test_transformers_version_sufficient(self) -> None:
        """Installed transformers version is >= 4.46.0 (has the fix)."""
        try:
            import transformers

            version_parts = transformers.__version__.split(".")
            major = int(version_parts[0])
            minor = int(version_parts[1])
            # Fix is in 4.46.0+. Also accept 5.x+.
            assert major > 4 or (major == 4 and minor >= 46), (
                f"transformers {transformers.__version__} < 4.46.0: "
                "gradient accumulation loss bug is present. "
                "Upgrade with: pip install 'transformers>=4.46'"
            )
        except ImportError:
            pytest.skip("transformers not installed")

    def test_average_tokens_in_training_arguments(self) -> None:
        """TrainingArguments has the average_tokens_across_devices parameter."""
        try:
            from transformers import TrainingArguments

            assert hasattr(TrainingArguments, "average_tokens_across_devices"), (
                "TrainingArguments does not have average_tokens_across_devices. "
                "This requires transformers >= 4.46.0."
            )
        except ImportError:
            pytest.skip("transformers not installed")


# ===================================================================
# Finetuner integration tests
# ===================================================================


class TestFinetunerGradAccumIntegration:
    """Verify that the finetuner passes average_tokens_across_devices to SFTConfig."""

    def test_training_args_dict_contains_average_tokens(self) -> None:
        """The training_args_dict built by finetuner includes the fix parameter.

        We verify by reading the finetuner source code to confirm
        ``average_tokens_across_devices`` is passed into ``training_args_dict``.
        This avoids complex mocking of SFTConfig.__init__.
        """
        import inspect

        from llm_forge.training.finetuner import FineTuner

        source = inspect.getsource(FineTuner.train)
        assert "average_tokens_across_devices" in source, (
            "FineTuner.train() does not reference average_tokens_across_devices"
        )
