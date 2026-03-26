"""Tests for extracted export tools."""
from __future__ import annotations
import json
import pytest

class TestExportModel:
    def test_export_nonexistent_model(self) -> None:
        from llm_forge.chat.agent_tools.export_tools import export_model
        result = json.loads(export_model("/nonexistent/model", "gguf"))
        assert isinstance(result, dict)

class TestDeployToOllama:
    def test_deploy_nonexistent_model(self) -> None:
        from llm_forge.chat.agent_tools.export_tools import deploy_to_ollama
        result = json.loads(deploy_to_ollama("/nonexistent/model", "test-model"))
        assert isinstance(result, dict)

class TestDeployToHuggingface:
    def test_deploy_nonexistent_model(self) -> None:
        from llm_forge.chat.agent_tools.export_tools import deploy_to_huggingface
        result = json.loads(deploy_to_huggingface("/nonexistent/model", "test/repo"))
        assert isinstance(result, dict)

class TestExportToolDefinitions:
    def test_export_tool_definitions_exist(self) -> None:
        from llm_forge.chat.agent_tools.export_tools import EXPORT_TOOL_DEFINITIONS
        names = [t["name"] for t in EXPORT_TOOL_DEFINITIONS]
        assert "export_model" in names
        assert "deploy_to_ollama" in names
