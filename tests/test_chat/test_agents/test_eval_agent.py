"""Tests for the Eval Agent."""
from __future__ import annotations
import json
import pytest

class TestEvalAgentCreation:
    def test_create_eval_agent_rejects_empty_key(self) -> None:
        from llm_forge.chat.agents.eval_agent import create_eval_agent
        with pytest.raises((ValueError, Exception)):
            create_eval_agent("")

    def test_eval_agent_system_prompt_exists(self) -> None:
        from llm_forge.chat.agents.eval_agent import EVAL_AGENT_PROMPT
        assert len(EVAL_AGENT_PROMPT) > 100
        assert "eval" in EVAL_AGENT_PROMPT.lower() or "benchmark" in EVAL_AGENT_PROMPT.lower()

    def test_eval_agent_has_tools(self) -> None:
        from llm_forge.chat.agents.eval_agent import EVAL_AGENT_TOOLS
        tool_names = [t.__name__ if callable(t) else t.get("name", "") for t in EVAL_AGENT_TOOLS]
        assert "run_evaluation" in tool_names
        assert "compare_models" in tool_names

class TestEvalAgentToolDispatch:
    def test_run_evaluation_via_tool(self) -> None:
        from llm_forge.chat.agent_tools.eval_tools import run_evaluation
        result = json.loads(run_evaluation("/nonexistent/model"))
        assert isinstance(result, dict)
