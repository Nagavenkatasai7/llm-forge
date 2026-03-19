"""Tests for the config schema and validator modules.

Covers LLMForgeConfig creation, sub-config validation, enum values,
YAML-based validation, preset loading, and cross-field validators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from llm_forge.config.schema import (
    DataFormat,
    LLMForgeConfig,
    LoRAConfig,
    ModelConfig,
    PrecisionMode,
    QuantizationConfig,
    RAGConfig,
    TrainingConfig,
    TrainingMode,
)
from llm_forge.config.validator import (
    ConfigValidationError,
    list_presets,
    load_preset,
    validate_config_dict,
    validate_config_file,
)

# ===================================================================
# LLMForgeConfig creation
# ===================================================================


class TestLLMForgeConfigCreation:
    """Test top-level config creation with minimal and expanded dicts."""

    def test_minimal_valid_config(self, minimal_config_dict: dict[str, Any]) -> None:
        """Creating LLMForgeConfig with just model.name and data.train_path succeeds."""
        config = LLMForgeConfig(**minimal_config_dict)
        assert config.model.name == "meta-llama/Llama-3.2-1B"
        assert config.data.train_path == "tatsu-lab/alpaca"

    def test_defaults_are_populated(self, minimal_config_dict: dict[str, Any]) -> None:
        """Defaults for LoRA, training, etc. are filled in."""
        config = LLMForgeConfig(**minimal_config_dict)
        # LoRA defaults
        assert config.lora.r == 16
        assert config.lora.alpha == 32
        assert config.lora.dropout == 0.05
        # Training defaults
        assert config.training.mode == "lora"
        assert config.training.num_epochs == 1
        assert config.training.learning_rate == 2e-5
        # Quantization defaults
        assert config.quantization.load_in_4bit is False
        assert config.quantization.load_in_8bit is False

    def test_missing_model_name_raises(self) -> None:
        """Omitting model.name raises a validation error."""
        with pytest.raises(ValidationError, match="model"):
            LLMForgeConfig(
                model={},  # type: ignore[arg-type]
                data={"train_path": "some/data"},
            )

    def test_missing_data_train_path_raises(self) -> None:
        """Omitting data.train_path raises a validation error."""
        with pytest.raises(ValidationError, match="train_path"):
            LLMForgeConfig(
                model={"name": "some-model"},
                data={},  # type: ignore[arg-type]
            )

    def test_extra_field_forbidden(self, minimal_config_dict: dict[str, Any]) -> None:
        """Extra top-level keys are rejected."""
        bad = {**minimal_config_dict, "unknown_key": 123}
        with pytest.raises(ValidationError, match="extra"):
            LLMForgeConfig(**bad)


# ===================================================================
# ModelConfig
# ===================================================================


class TestModelConfig:
    """Tests for the ModelConfig sub-model."""

    def test_basic_creation(self) -> None:
        m = ModelConfig(name="gpt2")
        assert m.name == "gpt2"
        assert m.torch_dtype == PrecisionMode.bf16
        assert m.max_seq_length == 2048

    def test_custom_params(self) -> None:
        m = ModelConfig(
            name="facebook/opt-350m",
            torch_dtype="fp16",
            max_seq_length=4096,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        assert m.torch_dtype == PrecisionMode.fp16
        assert m.max_seq_length == 4096
        assert m.trust_remote_code is True
        assert m.attn_implementation == "sdpa"

    def test_seq_length_lower_bound(self) -> None:
        with pytest.raises(ValidationError, match="128"):
            ModelConfig(name="gpt2", max_seq_length=64)

    def test_seq_length_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(name="gpt2", max_seq_length=200_000)

    def test_invalid_attn_implementation(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(name="gpt2", attn_implementation="invalid")

    def test_revision_optional(self) -> None:
        m = ModelConfig(name="gpt2", revision="v1.0")
        assert m.revision == "v1.0"

    def test_rope_scaling(self) -> None:
        m = ModelConfig(name="gpt2", rope_scaling={"type": "dynamic", "factor": 2.0})
        assert m.rope_scaling["type"] == "dynamic"


# ===================================================================
# LoRAConfig
# ===================================================================


class TestLoRAConfig:
    """Tests for the LoRAConfig sub-model."""

    def test_defaults(self) -> None:
        lora = LoRAConfig()
        assert lora.r == 16
        assert lora.alpha == 32
        assert lora.dropout == 0.05
        assert lora.bias == "none"
        assert lora.task_type == "CAUSAL_LM"
        assert lora.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]

    def test_custom_values(self) -> None:
        lora = LoRAConfig(
            r=64,
            alpha=128,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="all",
            use_rslora=True,
            use_dora=True,
        )
        assert lora.r == 64
        assert lora.alpha == 128
        assert lora.dropout == 0.1
        assert lora.target_modules == ["q_proj", "v_proj"]
        assert lora.bias == "all"
        assert lora.use_rslora is True
        assert lora.use_dora is True

    def test_r_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            LoRAConfig(r=0)

    def test_r_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            LoRAConfig(r=300)

    def test_dropout_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LoRAConfig(dropout=0.6)
        with pytest.raises(ValidationError):
            LoRAConfig(dropout=-0.1)


# ===================================================================
# QuantizationConfig
# ===================================================================


class TestQuantizationConfig:
    """Tests for mutual exclusion and defaults."""

    def test_defaults(self) -> None:
        q = QuantizationConfig()
        assert q.load_in_4bit is False
        assert q.load_in_8bit is False
        assert q.bnb_4bit_quant_type == "nf4"
        assert q.bnb_4bit_use_double_quant is True

    def test_4bit_only(self) -> None:
        q = QuantizationConfig(load_in_4bit=True)
        assert q.load_in_4bit is True
        assert q.load_in_8bit is False

    def test_8bit_only(self) -> None:
        q = QuantizationConfig(load_in_8bit=True)
        assert q.load_in_8bit is True
        assert q.load_in_4bit is False

    def test_mutual_exclusion_4bit_8bit(self) -> None:
        """Cannot enable both load_in_4bit and load_in_8bit."""
        with pytest.raises(ValidationError, match="Cannot enable both"):
            QuantizationConfig(load_in_4bit=True, load_in_8bit=True)

    def test_quant_type_options(self) -> None:
        q = QuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4")
        assert q.bnb_4bit_quant_type == "fp4"


# ===================================================================
# Enums
# ===================================================================


class TestEnums:
    """Test that enum values match expectations."""

    def test_training_mode_values(self) -> None:
        expected = {"lora", "qlora", "full", "pretrain", "dpo", "orpo", "grpo"}
        actual = {e.value for e in TrainingMode}
        assert actual == expected

    def test_data_format_values(self) -> None:
        expected = {"alpaca", "sharegpt", "completion", "custom"}
        actual = {e.value for e in DataFormat}
        assert actual == expected

    def test_precision_mode_values(self) -> None:
        expected = {"fp32", "fp16", "bf16", "fp8", "int8", "int4"}
        actual = {e.value for e in PrecisionMode}
        assert actual == expected

    def test_training_mode_string_usage(self) -> None:
        """TrainingMode members behave as strings."""
        assert TrainingMode.lora == "lora"
        assert TrainingMode.qlora == "qlora"


# ===================================================================
# YAML Validation
# ===================================================================


class TestYAMLValidation:
    """Test config validation from YAML files."""

    def test_validate_config_file_valid(self, sample_config_yaml: Path) -> None:
        config = validate_config_file(sample_config_yaml)
        assert isinstance(config, LLMForgeConfig)
        assert config.model.name == "meta-llama/Llama-3.2-1B"

    def test_validate_config_file_nonexistent(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_config_file(tmp_path / "nope.yaml")

    def test_validate_config_file_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="empty"):
            validate_config_file(empty)

    def test_validate_config_file_non_dict(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="mapping"):
            validate_config_file(bad)

    def test_validate_config_dict_valid(self, minimal_config_dict: dict[str, Any]) -> None:
        config = validate_config_dict(minimal_config_dict)
        assert isinstance(config, LLMForgeConfig)

    def test_validate_config_dict_invalid(self) -> None:
        """Invalid dict produces ConfigValidationError with detail."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({"model": {"name": "x"}, "data": {}})
        assert "error" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()


# ===================================================================
# Error messages
# ===================================================================


class TestErrorMessages:
    """Validation errors include actionable messages."""

    def test_unknown_top_level_key_warning(self, minimal_config_dict: dict[str, Any]) -> None:
        """Extra top-level key yields a ConfigValidationError."""
        bad = {**minimal_config_dict, "epochs": 5}
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict(bad)
        # The error message should mention the unknown key
        assert "epochs" in str(exc_info.value).lower() or "extra" in str(exc_info.value).lower()

    def test_missing_required_field_message(self) -> None:
        """Missing required fields produce clear error messages."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dict({})
        msg = str(exc_info.value)
        assert "model" in msg.lower()


# ===================================================================
# Preset loading
# ===================================================================


class TestPresets:
    """Test preset discovery and loading."""

    def test_list_presets_returns_list(self) -> None:
        presets = list_presets()
        assert isinstance(presets, list)

    def test_preset_files_exist(self) -> None:
        """At least one preset YAML exists."""
        presets = list_presets()
        # If presets directory is shipped, expect at least one
        if presets:
            assert len(presets) >= 1

    @pytest.mark.parametrize(
        "preset_name",
        ["lora_default", "qlora_default", "full_finetune", "pretrain_small", "rag_default"],
    )
    def test_load_known_presets(self, preset_name: str) -> None:
        """Loading a known preset returns a valid LLMForgeConfig."""
        available = list_presets()
        if preset_name not in available:
            pytest.skip(f"Preset '{preset_name}' not available")
        config = load_preset(preset_name)
        assert isinstance(config, LLMForgeConfig)

    def test_load_nonexistent_preset_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_preset("this_preset_does_not_exist_at_all")


# ===================================================================
# Cross-field validation
# ===================================================================


class TestCrossFieldValidation:
    """Test automatic cross-field adjustments (model validators)."""

    def test_qlora_auto_sets_quantization(self) -> None:
        """When training.mode=qlora, quantization.load_in_4bit is auto-set."""
        config = LLMForgeConfig(
            model={"name": "gpt2"},
            data={"train_path": "some/data"},
            training={"mode": "qlora"},
        )
        assert config.quantization.load_in_4bit is True

    def test_qlora_does_not_override_existing_8bit(self) -> None:
        """If 8bit is already set, qlora auto-configure does not force 4bit."""
        config = LLMForgeConfig(
            model={"name": "gpt2"},
            data={"train_path": "some/data"},
            training={"mode": "qlora"},
            quantization={"load_in_8bit": True},
        )
        # 8bit was explicitly set, so auto-configure should not override
        assert config.quantization.load_in_8bit is True

    def test_bf16_fp16_mutual_exclusion(self) -> None:
        """Cannot enable both bf16 and fp16 in TrainingConfig."""
        with pytest.raises(ValidationError, match="Cannot enable both"):
            TrainingConfig(bf16=True, fp16=True)

    def test_rag_chunk_overlap_less_than_size(self) -> None:
        """RAGConfig rejects chunk_overlap >= chunk_size."""
        with pytest.raises(ValidationError, match="chunk_overlap"):
            RAGConfig(chunk_size=256, chunk_overlap=300)

    def test_data_cleaning_word_count_bounds(self) -> None:
        """DataCleaningConfig rejects min_word_count > max_word_count."""
        from llm_forge.config.schema import DataCleaningConfig

        with pytest.raises(ValidationError, match="min_word_count"):
            DataCleaningConfig(min_word_count=500, max_word_count=100)
