"""Tests for extracted training tools."""
from __future__ import annotations
import json
import pytest

class TestStartTraining:
    def test_start_training_nonexistent_config(self) -> None:
        from llm_forge.chat.agent_tools.training_tools import start_training
        result = json.loads(start_training("/nonexistent/config.yaml"))
        # Should return error status
        assert "error" in json.dumps(result).lower() or "not" in json.dumps(result).lower() or "fail" in json.dumps(result).lower()

class TestCheckTrainingStatus:
    def test_check_status_returns_json(self) -> None:
        from llm_forge.chat.agent_tools.training_tools import check_training_status
        result = json.loads(check_training_status())
        assert isinstance(result, dict)

class TestReadTrainingLogs:
    def test_read_logs_nonexistent_dir(self) -> None:
        from llm_forge.chat.agent_tools.training_tools import read_training_logs
        result = json.loads(read_training_logs("/nonexistent/output"))
        assert isinstance(result, dict)
