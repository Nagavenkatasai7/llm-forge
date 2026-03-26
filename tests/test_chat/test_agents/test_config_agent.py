"""Tests for the Config Agent."""
from __future__ import annotations
import json
import pytest

class TestConfigAgentCreation:
    def test_create_config_agent_rejects_empty_key(self) -> None:
        from llm_forge.chat.agents.config_agent import create_config_agent
        with pytest.raises((ValueError, Exception)):
            create_config_agent("")

    def test_config_agent_system_prompt_exists(self) -> None:
        from llm_forge.chat.agents.config_agent import CONFIG_AGENT_PROMPT
        assert len(CONFIG_AGENT_PROMPT) > 100
        assert "config" in CONFIG_AGENT_PROMPT.lower() or "yaml" in CONFIG_AGENT_PROMPT.lower()

    def test_config_agent_has_tools(self) -> None:
        from llm_forge.chat.agents.config_agent import CONFIG_AGENT_TOOLS
        tool_names = [t.__name__ if callable(t) else t.get("name", "") for t in CONFIG_AGENT_TOOLS]
        assert "write_config" in tool_names
        assert "validate_config" in tool_names

class TestConfigAgentToolDispatch:
    def test_write_config_via_tool(self, tmp_path) -> None:
        from llm_forge.chat.agent_tools.config_tools import write_config
        output = tmp_path / "generated.yaml"
        config = {"project_name": "test", "model": {"name": "SmolLM2-135M"}}
        result = json.loads(write_config(str(output), config))
        assert result["status"] in ("success", "ok")

    def test_estimate_training_via_tool(self) -> None:
        from llm_forge.chat.agent_tools.config_tools import estimate_training
        result = json.loads(estimate_training("SmolLM2-135M", "lora", 500))
        assert isinstance(result, dict)
