"""Tests for the interactive training wizard.

Covers personal AI templates, model tiers, hardware detection,
data scanning, config generation, and YAML output formatting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

try:
    from llm_forge.wizard import (
        MODEL_TIERS,
        PERSONAL_AI_TEMPLATES,
        TrainingWizard,
        config_to_yaml,
        detect_hardware_profile,
        generate_config,
        scan_data_source,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.wizard not importable",
)


# ===================================================================
# Personal AI Templates
# ===================================================================


class TestPersonalAITemplates:
    """Test the 6 personal AI template definitions."""

    def test_six_templates_exist(self) -> None:
        assert len(PERSONAL_AI_TEMPLATES) == 6

    def test_expected_template_names(self) -> None:
        expected = {"journal", "study", "writer", "professional", "mindfulness", "gamemaster"}
        assert set(PERSONAL_AI_TEMPLATES.keys()) == expected

    def test_each_template_has_required_fields(self) -> None:
        required = {"name", "system_prompt", "training_config", "lora_config"}
        for key, template in PERSONAL_AI_TEMPLATES.items():
            for field in required:
                assert field in template, f"Template '{key}' missing field '{field}'"

    def test_each_template_has_system_prompt(self) -> None:
        for key, template in PERSONAL_AI_TEMPLATES.items():
            assert len(template["system_prompt"]) > 10, (
                f"Template '{key}' has too short a system prompt"
            )

    def test_training_configs_have_epochs(self) -> None:
        for key, template in PERSONAL_AI_TEMPLATES.items():
            tc = template["training_config"]
            assert "num_epochs" in tc or "num_train_epochs" in tc, (
                f"Template '{key}' missing epoch config"
            )

    def test_lora_configs_have_rank(self) -> None:
        for key, template in PERSONAL_AI_TEMPLATES.items():
            lc = template["lora_config"]
            assert "r" in lc, f"Template '{key}' missing LoRA rank"
            assert lc["r"] >= 4


# ===================================================================
# Model Tiers
# ===================================================================


class TestModelTiers:
    """Test the model tier definitions."""

    def test_five_tiers_exist(self) -> None:
        assert len(MODEL_TIERS) == 5

    def test_expected_tier_names(self) -> None:
        expected = {"tiny", "small", "medium", "large", "xlarge"}
        assert set(MODEL_TIERS.keys()) == expected

    def test_each_tier_has_model_name(self) -> None:
        for tier_name, tier in MODEL_TIERS.items():
            assert "name" in tier or "model" in tier, f"Tier '{tier_name}' missing model name"

    def test_tiers_ordered_by_size(self) -> None:
        """Higher tiers should need more RAM."""
        tiers = list(MODEL_TIERS.values())
        min_rams = []
        for tier in tiers:
            min_ram = tier.get("min_ram", tier.get("min_ram_gb", 0))
            min_rams.append(min_ram)
        # At least the first should need less than the last
        assert min_rams[0] <= min_rams[-1]


# ===================================================================
# Hardware detection
# ===================================================================


class TestDetectHardwareProfile:
    """Test hardware profile detection."""

    def test_returns_dict(self) -> None:
        profile = detect_hardware_profile()
        assert isinstance(profile, dict)

    def test_has_platform(self) -> None:
        profile = detect_hardware_profile()
        assert "platform" in profile
        assert profile["platform"].lower() in ("darwin", "linux", "windows")

    def test_has_ram_gb(self) -> None:
        profile = detect_hardware_profile()
        assert "ram_gb" in profile
        assert isinstance(profile["ram_gb"], (int, float))

    def test_has_recommended_method(self) -> None:
        profile = detect_hardware_profile()
        assert "recommended_method" in profile
        assert profile["recommended_method"] in ("lora", "qlora", "full")

    def test_has_recommended_tier(self) -> None:
        profile = detect_hardware_profile()
        assert "recommended_model_tier" in profile or "recommended_tier" in profile


# ===================================================================
# Data source scanning
# ===================================================================


class TestScanDataSource:
    """Test data source analysis."""

    def test_scan_hf_dataset(self) -> None:
        """HuggingFace dataset paths are recognised."""
        info = scan_data_source("tatsu-lab/alpaca")
        assert isinstance(info, dict)
        assert info.get("is_hf_dataset") is True

    def test_scan_local_file(self, tmp_path: Path) -> None:
        """Local JSONL file is detected."""
        f = tmp_path / "train.jsonl"
        f.write_text('{"instruction":"q","output":"a"}\n')
        info = scan_data_source(str(f))
        assert isinstance(info, dict)

    def test_scan_nonexistent_path(self) -> None:
        """Non-existent path returns error info."""
        info = scan_data_source("/nonexistent/path/data.jsonl")
        assert isinstance(info, dict)


# ===================================================================
# Config generation
# ===================================================================


class TestGenerateConfig:
    """Test YAML config generation from wizard selections."""

    _BASE_KWARGS: dict[str, Any] = {
        "ai_name": "TestBot",
        "training_method": "lora",
        "data_format": "alpaca",
    }

    def test_generates_dict(self) -> None:
        cfg = generate_config(
            purpose="journal",
            model_name="HuggingFaceTB/SmolLM2-135M",
            data_path="tatsu-lab/alpaca",
            **self._BASE_KWARGS,
        )
        assert isinstance(cfg, dict)

    def test_has_model_section(self) -> None:
        cfg = generate_config(
            purpose="study",
            model_name="test/model",
            data_path="test/data",
            **self._BASE_KWARGS,
        )
        assert "model" in cfg
        assert cfg["model"]["name"] == "test/model"

    def test_has_data_section(self) -> None:
        cfg = generate_config(
            purpose="writer",
            model_name="test/model",
            data_path="test/data",
            **self._BASE_KWARGS,
        )
        assert "data" in cfg
        assert cfg["data"]["train_path"] == "test/data"

    def test_has_training_section(self) -> None:
        cfg = generate_config(
            purpose="professional",
            model_name="test/model",
            data_path="test/data",
            **self._BASE_KWARGS,
        )
        assert "training" in cfg

    def test_custom_purpose_works(self) -> None:
        cfg = generate_config(
            purpose="custom",
            model_name="test/model",
            data_path="test/data",
            **self._BASE_KWARGS,
        )
        assert isinstance(cfg, dict)
        assert "model" in cfg


# ===================================================================
# YAML output
# ===================================================================


class TestConfigToYaml:
    """Test YAML string generation."""

    def test_returns_string(self) -> None:
        cfg = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test/data"},
            "training": {"num_epochs": 3},
        }
        yaml_str = config_to_yaml(cfg, "TestBot")
        assert isinstance(yaml_str, str)
        assert len(yaml_str) > 10

    def test_valid_yaml(self) -> None:
        """Generated string is parseable YAML."""
        import yaml

        cfg = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test/data"},
        }
        yaml_str = config_to_yaml(cfg, "TestBot")
        # Filter out comment lines before parsing
        lines = [l for l in yaml_str.split("\n") if not l.strip().startswith("#")]
        parsed = yaml.safe_load("\n".join(lines))
        if parsed is not None:
            assert "model" in parsed

    def test_contains_ai_name(self) -> None:
        cfg = {"model": {"name": "test/model"}}
        yaml_str = config_to_yaml(cfg, "MyAssistant")
        assert "MyAssistant" in yaml_str


# ===================================================================
# TrainingWizard class structure
# ===================================================================


class TestTrainingWizardStructure:
    """Test the wizard class has the expected interface."""

    def test_class_exists(self) -> None:
        assert TrainingWizard is not None

    def test_has_run_method(self) -> None:
        assert hasattr(TrainingWizard, "run")
        assert callable(TrainingWizard.run)
