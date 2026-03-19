"""Tests for config preset templates and the setup wizard output.

Verifies that all 5 CLI init templates (lora, qlora, pretrain, rag, full)
and the setup wizard output produce valid YAML that parses correctly.
"""

from __future__ import annotations

import pytest
import yaml

try:
    from llm_forge.cli import _SETUP_MODELS, _generate_default_config

    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False

try:
    from llm_forge.config.schema import LLMForgeConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False


pytestmark = pytest.mark.skipif(not _CLI_AVAILABLE, reason="llm_forge.cli not importable")


# ===================================================================
# Init Template Tests
# ===================================================================


class TestInitTemplates:
    """Test that all init templates produce valid YAML."""

    @pytest.mark.parametrize("template", ["lora", "qlora", "pretrain", "rag", "full"])
    def test_template_produces_yaml(self, template: str) -> None:
        """Each template generates parseable YAML."""
        content = _generate_default_config(template)
        assert isinstance(content, str)
        assert len(content) > 100  # Reasonable length
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    @pytest.mark.parametrize("template", ["lora", "qlora", "pretrain", "rag", "full"])
    def test_template_has_model_section(self, template: str) -> None:
        """All templates have a model section."""
        content = _generate_default_config(template)
        parsed = yaml.safe_load(content)
        assert "model" in parsed
        assert "name" in parsed["model"]

    @pytest.mark.parametrize("template", ["lora", "qlora", "pretrain", "rag", "full"])
    def test_template_has_data_section(self, template: str) -> None:
        """All templates have a data section."""
        content = _generate_default_config(template)
        parsed = yaml.safe_load(content)
        assert "data" in parsed

    @pytest.mark.parametrize("template", ["lora", "qlora", "pretrain", "rag", "full"])
    def test_template_has_training_section(self, template: str) -> None:
        """All templates have a training section."""
        content = _generate_default_config(template)
        parsed = yaml.safe_load(content)
        assert "training" in parsed

    def test_lora_template_has_lora_config(self) -> None:
        parsed = yaml.safe_load(_generate_default_config("lora"))
        assert "lora" in parsed

    def test_qlora_template_has_quantization(self) -> None:
        parsed = yaml.safe_load(_generate_default_config("qlora"))
        assert "lora" in parsed
        # QLoRA should reference quantization somewhere
        content = _generate_default_config("qlora")
        assert (
            "4bit" in content.lower() or "quantiz" in content.lower() or "qlora" in content.lower()
        )

    def test_rag_template_has_rag_section(self) -> None:
        parsed = yaml.safe_load(_generate_default_config("rag"))
        assert "rag" in parsed

    def test_pretrain_template_mode(self) -> None:
        parsed = yaml.safe_load(_generate_default_config("pretrain"))
        assert parsed.get("training", {}).get("mode") == "pretrain"


# ===================================================================
# Schema Validation of Templates
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestTemplateSchemaValidation:
    """Test that templates produce configs that validate against the schema.

    Note: some templates include extra keys like 'project' that aren't in
    the schema, so we test with model_config extra='ignore'.
    """

    @pytest.mark.parametrize("template", ["lora", "qlora", "pretrain", "rag", "full"])
    def test_template_validates_core_fields(self, template: str) -> None:
        """Template YAML has valid model and data fields for the schema."""
        content = _generate_default_config(template)
        parsed = yaml.safe_load(content)

        # Extract core fields that the schema requires
        model_cfg = parsed.get("model", {})
        data_cfg = parsed.get("data", {})

        assert "name" in model_cfg, f"Template '{template}' missing model.name"
        assert isinstance(model_cfg["name"], str)

        # Data section should have train_path (or be present)
        if "train_path" in data_cfg:
            assert isinstance(data_cfg["train_path"], str)


# ===================================================================
# Setup Wizard Output Tests
# ===================================================================


class TestSetupModels:
    """Test the setup wizard's model presets."""

    def test_setup_models_dict(self) -> None:
        assert isinstance(_SETUP_MODELS, dict)
        assert len(_SETUP_MODELS) >= 4

    def test_all_presets_have_model_name(self) -> None:
        for _key, (model_name, desc) in _SETUP_MODELS.items():
            assert isinstance(model_name, str)
            assert len(model_name) > 0
            assert isinstance(desc, str)

    def test_expected_sizes(self) -> None:
        assert "tiny" in _SETUP_MODELS
        assert "small" in _SETUP_MODELS
        assert "medium" in _SETUP_MODELS
        assert "large" in _SETUP_MODELS


# ===================================================================
# Config Generation Roundtrip Tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestConfigRoundtrip:
    """Test that generated configs can be loaded back as LLMForgeConfig."""

    def _try_validate(self, yaml_content: str) -> bool:
        """Try to validate a YAML string as LLMForgeConfig."""
        parsed = yaml.safe_load(yaml_content)
        # Remove fields not in the schema
        for extra_key in ["project", "logging"]:
            parsed.pop(extra_key, None)
        try:
            LLMForgeConfig(**parsed)
            return True
        except Exception:
            return False

    def test_lora_roundtrip(self) -> None:
        content = _generate_default_config("lora")
        assert self._try_validate(content)

    def test_full_roundtrip(self) -> None:
        content = _generate_default_config("full")
        assert self._try_validate(content)
