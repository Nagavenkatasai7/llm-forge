"""Tests for extracted eval tools."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestRunEvaluation:
    def test_run_evaluation_nonexistent_model(self) -> None:
        from llm_forge.chat.agent_tools.eval_tools import run_evaluation

        result = json.loads(run_evaluation("/nonexistent/model"))
        assert isinstance(result, dict)


class TestTestModel:
    @patch("openai.OpenAI")
    def test_test_model_returns_json(self, mock_openai_cls) -> None:
        """test_model returns valid JSON (mocked to avoid network)."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "4"
        mock_response.usage.total_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        from llm_forge.chat.agent_tools.eval_tools import test_model

        result = json.loads(
            test_model("meta/llama-3.2-1b-instruct", "What is 2+2?")
        )
        assert isinstance(result, dict)


class TestCompareModels:
    @patch("openai.OpenAI")
    def test_compare_models_returns_json(self, mock_openai_cls) -> None:
        """compare_models returns valid JSON (mocked)."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer A is better. Winner: A"
        mock_response.usage.total_tokens = 10
        mock_client.chat.completions.create.return_value = mock_response

        from llm_forge.chat.agent_tools.eval_tools import compare_models

        result = json.loads(
            compare_models("model_a", "model_b", ["test question"])
        )
        assert isinstance(result, dict)


class TestEvalToolDefinitions:
    def test_eval_tool_definitions_exist(self) -> None:
        from llm_forge.chat.agent_tools.eval_tools import EVAL_TOOL_DEFINITIONS

        names = [t["name"] for t in EVAL_TOOL_DEFINITIONS]
        assert "run_evaluation" in names
        assert "test_model" in names
        assert "compare_models" in names
