"""Tests for BaseAgent wrapper around Google ADK."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest

class TestAgentManager:
    def test_agent_manager_init(self) -> None:
        from llm_forge.chat.agents.base import AgentManager
        with pytest.raises((ValueError, KeyError, Exception)):
            AgentManager(gemini_api_key="")

    def test_agent_manager_has_registry(self) -> None:
        from llm_forge.chat.agents.base import AgentManager
        assert hasattr(AgentManager, "delegate")

    def test_agent_names_constant(self) -> None:
        from llm_forge.chat.agents.base import AGENT_NAMES
        assert "data_agent" in AGENT_NAMES
        assert "config_agent" in AGENT_NAMES

class TestDelegateToolSchema:
    def test_delegate_tool_schema_exists(self) -> None:
        from llm_forge.chat.agents.base import DELEGATE_TOOL
        assert DELEGATE_TOOL["name"] == "delegate_to_agent"
        assert "agent" in DELEGATE_TOOL["input_schema"]["properties"]
        assert "task" in DELEGATE_TOOL["input_schema"]["properties"]

    def test_delegate_tool_agent_enum(self) -> None:
        from llm_forge.chat.agents.base import DELEGATE_TOOL
        agent_enum = DELEGATE_TOOL["input_schema"]["properties"]["agent"]["enum"]
        assert "data_agent" in agent_enum
        assert "config_agent" in agent_enum
