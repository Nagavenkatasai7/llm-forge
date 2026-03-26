"""Tests for extracted research tools."""
from __future__ import annotations
import json
import pytest


class TestSearchHuggingface:
    def test_search_via_research_tools(self) -> None:
        from llm_forge.chat.agent_tools.research_tools import search_huggingface
        result = json.loads(search_huggingface("test", "model"))
        assert isinstance(result, (dict, list))


class TestInstallDependencies:
    def test_install_deps_returns_json(self) -> None:
        from llm_forge.chat.agent_tools.research_tools import install_dependencies
        # "training" feature should check deps without actually installing
        result = json.loads(install_dependencies("training"))
        assert isinstance(result, dict)


class TestWebSearch:
    def test_web_search_placeholder(self) -> None:
        from llm_forge.chat.agent_tools.research_tools import web_search
        result = json.loads(web_search("test query"))
        assert isinstance(result, dict)


class TestResearchToolDefinitions:
    def test_definitions_exist(self) -> None:
        from llm_forge.chat.agent_tools.research_tools import RESEARCH_TOOL_DEFINITIONS
        names = [t["name"] for t in RESEARCH_TOOL_DEFINITIONS]
        assert "search_huggingface" in names
        assert "web_search" in names
