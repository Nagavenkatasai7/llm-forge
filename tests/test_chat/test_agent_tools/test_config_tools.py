"""Tests for extracted config tools."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

class TestWriteConfig:
    def test_write_valid_config(self, tmp_path: Path) -> None:
        from llm_forge.chat.agent_tools.config_tools import write_config
        output = tmp_path / "test_config.yaml"
        config = {"project_name": "test", "model": {"name": "SmolLM2-135M"}, "data": {"train_path": "data/train.jsonl"}, "training": {"mode": "lora", "num_train_epochs": 1}}
        result = json.loads(write_config(str(output), config))
        assert result["status"] in ("success", "ok")
        assert output.exists()

    def test_write_config_creates_parent_dirs(self, tmp_path: Path) -> None:
        from llm_forge.chat.agent_tools.config_tools import write_config
        output = tmp_path / "nested" / "dir" / "config.yaml"
        config = {"project_name": "test"}
        result = json.loads(write_config(str(output), config))
        assert output.parent.exists()

class TestValidateConfig:
    def test_validate_nonexistent_file(self) -> None:
        from llm_forge.chat.agent_tools.config_tools import validate_config
        result = json.loads(validate_config("/nonexistent/config.yaml"))
        assert "error" in result.get("status", "").lower() or "error" in json.dumps(result).lower()

class TestListConfigs:
    def test_list_returns_json(self) -> None:
        from llm_forge.chat.agent_tools.config_tools import list_configs
        result = json.loads(list_configs())
        assert isinstance(result, (dict, list))

class TestEstimateTraining:
    def test_estimate_returns_json(self) -> None:
        from llm_forge.chat.agent_tools.config_tools import estimate_training
        result = json.loads(estimate_training(model_name="SmolLM2-135M", mode="lora", num_samples=1000))
        assert isinstance(result, dict)
