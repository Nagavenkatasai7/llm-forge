"""Tests for the ChatEngine class.

Covers initialization, provider detection, send() behavior,
memory integration, system prompt construction, and tool routing.
Mock all API clients to avoid real network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from llm_forge.chat.engine import ChatEngine
from llm_forge.chat.memory import MemoryManager

# ===================================================================
# Engine initialization
# ===================================================================


class TestEngineInit:
    """Test ChatEngine construction and provider detection."""

    def test_engine_init_no_api_key(self, tmp_path: Path) -> None:
        """ChatEngine with no key sets provider='none'."""
        with patch.dict("os.environ", {}, clear=True):
            engine = ChatEngine(project_dir=str(tmp_path))
        assert engine.provider == "none"

    def test_engine_init_anthropic_key(self, tmp_path: Path) -> None:
        """ChatEngine detects anthropic provider when ANTHROPIC_API_KEY is set."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-fake-key"}, clear=True):
            engine = ChatEngine(project_dir=str(tmp_path))
        assert engine.provider == "anthropic"

    def test_engine_init_openai_key(self, tmp_path: Path) -> None:
        """ChatEngine detects openai provider when only OPENAI_API_KEY is set."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-fake-key"}, clear=True):
            engine = ChatEngine(project_dir=str(tmp_path))
        assert engine.provider == "openai"

    def test_engine_explicit_provider(self, tmp_path: Path) -> None:
        """ChatEngine accepts an explicit provider override."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        assert engine.provider == "none"

    def test_engine_has_memory(self, tmp_path: Path) -> None:
        """engine.memory is a MemoryManager instance."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        assert isinstance(engine.memory, MemoryManager)

    def test_engine_starts_with_empty_messages(self, tmp_path: Path) -> None:
        """ChatEngine starts with an empty messages list."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        assert engine.messages == []


# ===================================================================
# System prompt
# ===================================================================


class TestSystemPrompt:
    """Test system prompt construction with memory context."""

    def test_engine_system_prompt_has_context(self, tmp_path: Path) -> None:
        """engine.system contains 'Current Project State'."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        assert "Current Project State" in engine.system

    def test_engine_system_prompt_includes_base(self, tmp_path: Path) -> None:
        """engine.system includes the base SYSTEM_PROMPT text."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        # The system prompt should contain the LLM Forge identity
        assert "LLM Forge" in engine.system


# ===================================================================
# Send with no provider
# ===================================================================


class TestSendNoProvider:
    """Test send() when no API key is available."""

    def test_engine_send_no_provider(self, tmp_path: Path) -> None:
        """send() returns a helpful error message when no API key is set."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        response = engine.send("Hello")
        assert "No API key found" in response
        assert "ANTHROPIC_API_KEY" in response

    def test_send_no_provider_adds_user_message(self, tmp_path: Path) -> None:
        """send() still adds the user message to the messages list."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        engine.send("Hello there")
        assert len(engine.messages) >= 1
        assert engine.messages[0]["role"] == "user"
        assert engine.messages[0]["content"] == "Hello there"


# ===================================================================
# Tool routing
# ===================================================================


class TestToolRouting:
    """Test _execute_tool() routes memory tools to MemoryManager."""

    def test_execute_tool_routes_memory_save(self, tmp_path: Path) -> None:
        """_execute_tool('save_memory', ...) calls memory.save_memory."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        result_str = engine._execute_tool(
            "save_memory",
            {"category": "user_preference", "content": "Prefers LoRA"},
        )
        result = json.loads(result_str)
        assert result["status"] == "ok"

        # Verify it was actually saved
        recalled = json.loads(engine.memory.recall_memory("LoRA"))
        assert recalled["count"] >= 1

    def test_execute_tool_routes_memory_recall(self, tmp_path: Path) -> None:
        """_execute_tool('recall_memory', ...) calls memory.recall_memory."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        engine.memory.save_memory("test", "GPU is A100")
        result_str = engine._execute_tool("recall_memory", {"query": "GPU"})
        result = json.loads(result_str)
        assert result["count"] >= 1

    def test_execute_tool_routes_get_project_state(self, tmp_path: Path) -> None:
        """_execute_tool('get_project_state', {}) returns project state JSON."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        result_str = engine._execute_tool("get_project_state", {})
        result = json.loads(result_str)
        assert "project_dir" in result

    def test_execute_tool_routes_get_session_history(self, tmp_path: Path) -> None:
        """_execute_tool('get_session_history', {}) calls memory.get_session_history."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        result_str = engine._execute_tool("get_session_history", {})
        result = json.loads(result_str)
        assert "sessions" in result

    def test_execute_tool_routes_log_training_run(self, tmp_path: Path) -> None:
        """_execute_tool('log_training_run', ...) calls memory.log_training_run."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        result_str = engine._execute_tool(
            "log_training_run",
            {
                "config_path": "test.yaml",
                "model_name": "test-model",
                "base_model": "Llama-1B",
                "mode": "lora",
                "output_dir": "outputs/test",
            },
        )
        result = json.loads(result_str)
        assert result["status"] == "ok"

    def test_execute_tool_routes_regular(self, tmp_path: Path) -> None:
        """_execute_tool('detect_hardware', {}) calls tools.execute_tool."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        result_str = engine._execute_tool("detect_hardware", {})
        result = json.loads(result_str)
        assert "os" in result
        assert "cpu" in result


# ===================================================================
# End session
# ===================================================================


class TestEndSession:
    """Test end_session() saves session data."""

    def test_end_session_saves(self, tmp_path: Path) -> None:
        """end_session() completes without error and saves summary."""
        engine = ChatEngine(provider="none", project_dir=str(tmp_path))
        engine.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there. How can I help?"},
        ]
        engine.end_session()

        import sqlite3

        conn = sqlite3.connect(str(engine.memory.db_path))
        row = conn.execute(
            "SELECT summary FROM sessions WHERE id = ?",
            (engine.memory.session_id,),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] is not None
