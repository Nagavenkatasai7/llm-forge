"""Tests for the OrchestratorEngine (Claude + ADK sub-agents)."""
from __future__ import annotations
import json
import os
from unittest.mock import MagicMock, patch
import pytest


class TestOrchestratorInit:
    def test_requires_anthropic_key(self) -> None:
        from llm_forge.chat.orchestrator import OrchestratorEngine
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises((ValueError, KeyError, Exception)):
                OrchestratorEngine(gemini_api_key="fake-key")

    def test_requires_gemini_key(self) -> None:
        from llm_forge.chat.orchestrator import OrchestratorEngine
        with pytest.raises((ValueError, TypeError)):
            OrchestratorEngine(gemini_api_key="")

    def test_has_send_method(self) -> None:
        from llm_forge.chat.orchestrator import OrchestratorEngine
        assert callable(getattr(OrchestratorEngine, "send", None))

    def test_has_end_session_method(self) -> None:
        from llm_forge.chat.orchestrator import OrchestratorEngine
        assert callable(getattr(OrchestratorEngine, "end_session", None))


class TestOrchestratorSystemPrompt:
    def test_system_prompt_mentions_agents(self) -> None:
        from llm_forge.chat.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
        assert "data_agent" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "config_agent" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_system_prompt_mentions_delegation(self) -> None:
        from llm_forge.chat.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
        assert "delegate" in ORCHESTRATOR_SYSTEM_PROMPT.lower()


class TestOrchestratorToolList:
    def test_delegate_tool_in_tools(self) -> None:
        from llm_forge.chat.orchestrator import ORCHESTRATOR_TOOLS
        tool_names = [t["name"] for t in ORCHESTRATOR_TOOLS]
        assert "delegate_to_agent" in tool_names

    def test_memory_tools_in_tools(self) -> None:
        from llm_forge.chat.orchestrator import ORCHESTRATOR_TOOLS
        tool_names = [t["name"] for t in ORCHESTRATOR_TOOLS]
        assert "save_memory" in tool_names
        assert "recall_memory" in tool_names
