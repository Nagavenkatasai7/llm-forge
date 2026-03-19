"""Tests for the wizard fallback module.

Verifies that the free guided wizard works without any API key:
- The function exists and is importable
- Purpose options are correctly defined
- Hardware detection integration works (mocked)
- Config generation produces valid YAML
- Model recommendation logic is correct
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from llm_forge.chat.wizard_fallback import (
    MODEL_RECOMMENDATIONS,
    PURPOSE_OPTIONS,
    _parse_hardware_json,
    _recommend_model,
    build_config,
    launch_wizard_fallback,
)

# ===================================================================
# Importability and existence
# ===================================================================


class TestWizardFallbackExists:
    """The wizard function exists and is importable."""

    def test_launch_wizard_fallback_importable(self) -> None:
        assert callable(launch_wizard_fallback)

    def test_build_config_importable(self) -> None:
        assert callable(build_config)

    def test_purpose_options_importable(self) -> None:
        assert isinstance(PURPOSE_OPTIONS, list)

    def test_model_recommendations_importable(self) -> None:
        assert isinstance(MODEL_RECOMMENDATIONS, list)


# ===================================================================
# Purpose options
# ===================================================================


class TestPurposeOptions:
    """Test that the 6 purpose options are correctly defined."""

    def test_six_purposes(self) -> None:
        assert len(PURPOSE_OPTIONS) == 6

    def test_expected_keys(self) -> None:
        keys = {p["key"] for p in PURPOSE_OPTIONS}
        expected = {
            "customer_support",
            "knowledge_assistant",
            "code_helper",
            "writing_assistant",
            "domain_expert",
            "custom",
        }
        assert keys == expected

    def test_each_has_required_fields(self) -> None:
        required = {"key", "label", "description", "system_prompt"}
        for p in PURPOSE_OPTIONS:
            for field in required:
                assert field in p, f"Purpose '{p.get('key', '?')}' missing field '{field}'"

    def test_labels_are_nonempty(self) -> None:
        for p in PURPOSE_OPTIONS:
            assert len(p["label"]) > 0

    def test_system_prompts_are_substantial(self) -> None:
        for p in PURPOSE_OPTIONS:
            assert len(p["system_prompt"]) > 10, (
                f"Purpose '{p['key']}' has a too-short system prompt"
            )


# ===================================================================
# Model recommendations
# ===================================================================


class TestModelRecommendations:
    """Test the model recommendation table and logic."""

    def test_at_least_three_models(self) -> None:
        assert len(MODEL_RECOMMENDATIONS) >= 3

    def test_each_has_name_and_mode(self) -> None:
        for m in MODEL_RECOMMENDATIONS:
            assert "name" in m
            assert "mode" in m
            assert m["mode"] in ("lora", "qlora", "full")

    def test_sorted_by_min_ram(self) -> None:
        """Models should be listed from smallest to largest requirements."""
        rams = [m["min_ram_gb"] for m in MODEL_RECOMMENDATIONS]
        assert rams == sorted(rams)

    def test_recommend_tiny_for_no_gpu(self) -> None:
        hw: dict[str, Any] = {"vram_gb": 0, "ram_total_gb": 2}
        rec = _recommend_model(hw)
        assert "135M" in rec["name"] or rec["min_ram_gb"] <= 2

    def test_recommend_large_for_big_gpu(self) -> None:
        hw: dict[str, Any] = {"vram_gb": 48, "ram_total_gb": 64}
        rec = _recommend_model(hw)
        # Should recommend a larger model
        assert rec["min_vram_gb"] >= 12

    def test_recommend_handles_missing_keys(self) -> None:
        hw: dict[str, Any] = {}
        rec = _recommend_model(hw)
        assert "name" in rec

    def test_recommend_handles_string_ram(self) -> None:
        hw: dict[str, Any] = {"vram_gb": "unknown", "ram_total_gb": "unknown"}
        rec = _recommend_model(hw)
        # Should fall back to smallest model
        assert rec["min_ram_gb"] <= 2


# ===================================================================
# Hardware detection integration (mocked)
# ===================================================================


class TestHardwareDetectionIntegration:
    """Test hardware parsing with mocked torch."""

    def test_parse_valid_json(self) -> None:
        hw_str = json.dumps(
            {
                "os": "Darwin",
                "ram_total_gb": 36,
                "gpu_type": "apple_mps",
                "gpu_name": "Apple M3 Pro",
            }
        )
        hw = _parse_hardware_json(hw_str)
        assert hw["os"] == "Darwin"
        assert hw["ram_total_gb"] == 36
        assert hw["gpu_type"] == "apple_mps"

    def test_parse_invalid_json(self) -> None:
        hw = _parse_hardware_json("not valid json")
        assert hw == {}

    def test_parse_none(self) -> None:
        hw = _parse_hardware_json(None)  # type: ignore[arg-type]
        assert hw == {}

    def test_nvidia_gpu_hardware(self) -> None:
        hw_str = json.dumps(
            {
                "os": "Linux",
                "gpu_type": "nvidia_cuda",
                "gpus": [{"name": "A100-SXM", "vram_gb": 80}],
                "ram_total_gb": 128,
            }
        )
        hw = _parse_hardware_json(hw_str)
        rec = _recommend_model(
            {
                "vram_gb": hw["gpus"][0]["vram_gb"],
                "ram_total_gb": hw["ram_total_gb"],
            }
        )
        # A100 80 GB should get a large model
        assert rec["min_vram_gb"] >= 12


# ===================================================================
# Config generation
# ===================================================================


class TestBuildConfig:
    """Test that build_config produces valid, well-structured configs."""

    _BASE = {
        "purpose_key": "customer_support",
        "model_name": "HuggingFaceTB/SmolLM2-135M",
        "training_mode": "lora",
        "data_path": "tatsu-lab/alpaca",
        "data_format": "alpaca",
        "system_prompt": "You are a helpful assistant.",
    }

    def test_returns_dict(self) -> None:
        cfg = build_config(**self._BASE)
        assert isinstance(cfg, dict)

    def test_has_model_section(self) -> None:
        cfg = build_config(**self._BASE)
        assert "model" in cfg
        assert cfg["model"]["name"] == "HuggingFaceTB/SmolLM2-135M"

    def test_has_data_section(self) -> None:
        cfg = build_config(**self._BASE)
        assert "data" in cfg
        assert cfg["data"]["train_path"] == "tatsu-lab/alpaca"
        assert cfg["data"]["format"] == "alpaca"

    def test_has_training_section(self) -> None:
        cfg = build_config(**self._BASE)
        assert "training" in cfg
        assert cfg["training"]["mode"] == "lora"
        assert cfg["training"]["num_epochs"] >= 1

    def test_has_serving_section(self) -> None:
        cfg = build_config(**self._BASE)
        assert "serving" in cfg

    def test_lora_config_for_lora_mode(self) -> None:
        cfg = build_config(**self._BASE)
        assert "lora" in cfg
        assert cfg["lora"]["r"] > 0
        assert cfg["lora"]["alpha"] > 0

    def test_qlora_adds_quantization(self) -> None:
        kwargs = {**self._BASE, "training_mode": "qlora"}
        cfg = build_config(**kwargs)
        assert "quantization" in cfg
        assert cfg["quantization"]["load_in_4bit"] is True
        assert "lora" in cfg

    def test_full_mode_no_lora(self) -> None:
        kwargs = {**self._BASE, "training_mode": "full"}
        cfg = build_config(**kwargs)
        assert "lora" not in cfg
        assert "quantization" not in cfg

    def test_system_prompt_in_data(self) -> None:
        cfg = build_config(**self._BASE)
        assert cfg["data"]["system_prompt"] == "You are a helpful assistant."

    def test_domain_expert_uses_conservative_lr(self) -> None:
        kwargs = {**self._BASE, "purpose_key": "domain_expert"}
        cfg = build_config(**kwargs)
        assert cfg["training"]["learning_rate"] <= 1e-4

    def test_different_purposes_produce_different_configs(self) -> None:
        cfg_support = build_config(**self._BASE)
        cfg_code = build_config(**{**self._BASE, "purpose_key": "code_helper"})
        # They should differ in at least one training parameter
        assert cfg_support["training"]["learning_rate"] != cfg_code["training"][
            "learning_rate"
        ] or cfg_support.get("lora", {}).get("r") != cfg_code.get("lora", {}).get("r")


# ===================================================================
# Config produces valid YAML
# ===================================================================


class TestConfigProducesValidYAML:
    """Config dict can be serialized to valid YAML and parsed back."""

    def test_yaml_round_trip(self) -> None:
        cfg = build_config(
            purpose_key="knowledge_assistant",
            model_name="HuggingFaceTB/SmolLM2-135M",
            training_mode="lora",
            data_path="tatsu-lab/alpaca",
            data_format="alpaca",
            system_prompt="You are helpful.",
        )
        yaml_str = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["model"]["name"] == "HuggingFaceTB/SmolLM2-135M"
        assert parsed["training"]["mode"] == "lora"

    def test_yaml_contains_all_top_level_keys(self) -> None:
        cfg = build_config(
            purpose_key="custom",
            model_name="test/model",
            training_mode="qlora",
            data_path="test/data",
            data_format="alpaca",
            system_prompt="Test prompt.",
        )
        yaml_str = yaml.dump(cfg)
        parsed = yaml.safe_load(yaml_str)
        for key in ("model", "data", "training", "serving"):
            assert key in parsed, f"Missing top-level key: {key}"


# ===================================================================
# Launch wizard smoke test (fully mocked I/O)
# ===================================================================


class TestLaunchWizardSmoke:
    """Smoke-test that the wizard runs end-to-end with mocked input."""

    @patch("llm_forge.chat.wizard_fallback._prompt_int")
    @patch("builtins.input", return_value="")
    @patch("llm_forge.chat.tools._detect_hardware")
    @patch("llm_forge.chat.tools._write_config")
    @patch("llm_forge.chat.tools._validate_config")
    def test_wizard_completes_with_defaults(
        self,
        mock_validate: MagicMock,
        mock_write: MagicMock,
        mock_hw: MagicMock,
        mock_input: MagicMock,
        mock_prompt: MagicMock,
    ) -> None:
        mock_hw.return_value = json.dumps(
            {
                "os": "Darwin",
                "cpu": "arm",
                "ram_total_gb": 36,
                "gpu_type": "apple_mps",
                "gpu_name": "Apple M3 Pro",
            }
        )
        mock_write.return_value = json.dumps({"status": "ok", "path": "/tmp/test.yaml"})
        mock_validate.return_value = json.dumps({"status": "valid"})
        # Simulate: purpose=1, data=3 (sample), model=Enter, launch=2 (exit)
        mock_prompt.side_effect = [1, 3, 2]

        # Should not raise
        launch_wizard_fallback()

        mock_hw.assert_called_once()
        mock_write.assert_called_once()
        mock_validate.assert_called_once()
