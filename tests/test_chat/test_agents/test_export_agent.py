"""Tests for the Export Agent."""
from __future__ import annotations
import json
import pytest

class TestExportAgentCreation:
    def test_create_export_agent_rejects_empty_key(self) -> None:
        from llm_forge.chat.agents.export_agent import create_export_agent
        with pytest.raises((ValueError, Exception)):
            create_export_agent("")

    def test_export_agent_system_prompt_exists(self) -> None:
        from llm_forge.chat.agents.export_agent import EXPORT_AGENT_PROMPT
        assert len(EXPORT_AGENT_PROMPT) > 100
        assert "export" in EXPORT_AGENT_PROMPT.lower() or "gguf" in EXPORT_AGENT_PROMPT.lower()

    def test_export_agent_has_tools(self) -> None:
        from llm_forge.chat.agents.export_agent import EXPORT_AGENT_TOOLS
        tool_names = [t.__name__ if callable(t) else t.get("name", "") for t in EXPORT_AGENT_TOOLS]
        assert "export_model" in tool_names
        assert "deploy_to_ollama" in tool_names
