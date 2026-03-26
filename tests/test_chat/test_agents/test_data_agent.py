"""Tests for the Data Agent."""
from __future__ import annotations
import json
import pytest

class TestDataAgentCreation:
    def test_create_data_agent_rejects_empty_key(self) -> None:
        from llm_forge.chat.agents.data_agent import create_data_agent
        with pytest.raises((ValueError, Exception)):
            create_data_agent("")

    def test_data_agent_system_prompt_exists(self) -> None:
        from llm_forge.chat.agents.data_agent import DATA_AGENT_PROMPT
        assert len(DATA_AGENT_PROMPT) > 100
        assert "data" in DATA_AGENT_PROMPT.lower()

    def test_data_agent_has_tools(self) -> None:
        from llm_forge.chat.agents.data_agent import DATA_AGENT_TOOLS
        tool_names = [t.__name__ if callable(t) else t.get("name", "") for t in DATA_AGENT_TOOLS]
        assert "scan_data" in tool_names
        assert "detect_hardware" in tool_names

class TestDataAgentToolDispatch:
    def test_scan_data_via_tool(self, tmp_path) -> None:
        from llm_forge.chat.agent_tools.data_tools import scan_data
        data_file = tmp_path / "test.jsonl"
        data_file.write_text('{"instruction": "test", "output": "ok"}\n')
        result = json.loads(scan_data(str(data_file)))
        assert result["status"] in ("success", "ok")
