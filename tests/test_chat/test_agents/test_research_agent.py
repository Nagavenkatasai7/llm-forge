"""Tests for the Research Agent."""
from __future__ import annotations
import json
import pytest

class TestResearchAgentCreation:
    def test_create_research_agent_rejects_empty_key(self) -> None:
        from llm_forge.chat.agents.research_agent import create_research_agent
        with pytest.raises((ValueError, Exception)):
            create_research_agent("")

    def test_research_agent_system_prompt_exists(self) -> None:
        from llm_forge.chat.agents.research_agent import RESEARCH_AGENT_PROMPT
        assert len(RESEARCH_AGENT_PROMPT) > 100
        assert "research" in RESEARCH_AGENT_PROMPT.lower() or "search" in RESEARCH_AGENT_PROMPT.lower()

    def test_research_agent_has_tools(self) -> None:
        from llm_forge.chat.agents.research_agent import RESEARCH_AGENT_TOOLS
        tool_names = [t.__name__ if callable(t) else t.get("name", "") for t in RESEARCH_AGENT_TOOLS]
        assert "search_huggingface" in tool_names
        assert "web_search" in tool_names
