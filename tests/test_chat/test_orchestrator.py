"""Tests for the OrchestratorEngine (Claude + ADK sub-agents)."""
from __future__ import annotations
import json
import os
from unittest.mock import MagicMock, patch
import pytest


class TestOrchestratorInit:
    def test_uses_builtin_keys_when_no_env(self) -> None:
        """OrchestratorEngine uses built-in keys when env vars are missing."""
        from llm_forge.chat.api_keys import get_anthropic_api_key, get_google_api_key

        assert len(get_anthropic_api_key()) > 10
        assert len(get_google_api_key()) > 10

    def test_env_vars_override_builtin(self) -> None:
        """User env vars take priority over built-in keys."""
        from llm_forge.chat.api_keys import get_anthropic_api_key

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-user-custom"}):
            assert get_anthropic_api_key() == "sk-user-custom"

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
