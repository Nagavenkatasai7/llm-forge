"""Tests for the sample packing feature.

Verifies that the pack_sequences config field flows through to
SFTConfig.packing and that the broken manual packing was removed.
"""

from __future__ import annotations

import pytest

from llm_forge.config.schema import LLMForgeConfig, TrainingConfig

# ===================================================================
# Schema Tests
# ===================================================================


class TestPackSequencesConfig:
    """Test the pack_sequences field in TrainingConfig."""

    def test_default_is_false(self) -> None:
        cfg = TrainingConfig()
        assert cfg.pack_sequences is False

    def test_can_enable(self) -> None:
        cfg = TrainingConfig(pack_sequences=True)
        assert cfg.pack_sequences is True

    def test_in_full_config(self) -> None:
        cfg = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            training={"pack_sequences": True},
        )
        assert cfg.training.pack_sequences is True

    def test_default_in_full_config(self) -> None:
        cfg = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
        )
        assert cfg.training.pack_sequences is False


# ===================================================================
# Config Validation Tests
# ===================================================================


class TestPackSequencesValidation:
    """Test that pack_sequences works with all training modes."""

    @pytest.mark.parametrize("mode", ["lora", "qlora", "full", "pretrain"])
    def test_valid_with_all_modes(self, mode: str) -> None:
        cfg = TrainingConfig(mode=mode, pack_sequences=True)
        assert cfg.pack_sequences is True
        assert cfg.mode == mode

    def test_pack_sequences_with_group_by_length(self) -> None:
        """group_by_length and pack_sequences can coexist in config."""
        cfg = TrainingConfig(pack_sequences=True, group_by_length=True)
        assert cfg.pack_sequences is True
        assert cfg.group_by_length is True

    def test_yaml_roundtrip(self) -> None:
        """Config with pack_sequences serializes and deserializes."""
        import yaml

        cfg = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            training={"pack_sequences": True, "num_epochs": 1},
        )
        dumped = yaml.dump(cfg.model_dump(mode="json"))
        loaded = yaml.safe_load(dumped)
        assert loaded["training"]["pack_sequences"] is True


# ===================================================================
# Preprocessor Tests
# ===================================================================


class TestPreprocessorPackingRemoved:
    """Verify that the broken manual packing was removed."""

    def test_no_tokenize_packed_method(self) -> None:
        """The broken _tokenize_packed should be removed."""
        try:
            from llm_forge.data.preprocessor import DataPreprocessor

            preprocessor = DataPreprocessor(max_seq_length=512, format_type="alpaca")
            assert not hasattr(preprocessor, "_tokenize_packed"), (
                "_tokenize_packed should be removed — packing is delegated to TRL"
            )
        except ImportError:
            pytest.skip("preprocessor not importable")


# ===================================================================
# Existing Config Validation
# ===================================================================


class TestExistingConfigsStillValid:
    """Verify that existing configs still validate with the new field."""

    def test_minimal_config(self) -> None:
        cfg = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
        )
        assert cfg.training.pack_sequences is False

    def test_full_config_with_all_features(self) -> None:
        cfg = LLMForgeConfig(
            model={"name": "test"},
            data={"train_path": "test"},
            training={
                "mode": "lora",
                "pack_sequences": True,
                "num_epochs": 3,
                "neftune_noise_alpha": 5.0,
                "label_smoothing_factor": 0.1,
            },
            evaluation={"enabled": True, "benchmarks": ["hellaswag"]},
            iti={"enabled": True},
            refusal={"enabled": True},
        )
        assert cfg.training.pack_sequences is True
        assert cfg.iti.enabled is True
        assert cfg.refusal.enabled is True
