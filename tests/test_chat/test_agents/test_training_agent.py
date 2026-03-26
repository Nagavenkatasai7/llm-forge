"""Tests for the Training Agent."""
from __future__ import annotations
import json
import pytest

class TestTrainingAgentCreation:
    def test_create_training_agent_rejects_empty_key(self) -> None:
        from llm_forge.chat.agents.training_agent import create_training_agent
        with pytest.raises((ValueError, Exception)):
            create_training_agent("")

    def test_training_agent_system_prompt_exists(self) -> None:
        from llm_forge.chat.agents.training_agent import TRAINING_AGENT_PROMPT
        assert len(TRAINING_AGENT_PROMPT) > 100
        assert "train" in TRAINING_AGENT_PROMPT.lower()

    def test_training_agent_has_tools(self) -> None:
        from llm_forge.chat.agents.training_agent import TRAINING_AGENT_TOOLS
        tool_names = [t.__name__ if callable(t) else t.get("name", "") for t in TRAINING_AGENT_TOOLS]
        assert "start_training" in tool_names
        assert "check_training_status" in tool_names

class TestTrainingAgentToolDispatch:
    def test_check_status_via_tool(self) -> None:
        from llm_forge.chat.agent_tools.training_tools import check_training_status
        result = json.loads(check_training_status())
        assert isinstance(result, dict)
