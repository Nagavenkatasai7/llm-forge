"""Tests for the slash command handling in the LLM Forge chat UI."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_forge.chat.slash_commands import (
    CLEAR_SENTINEL,
    COMMANDS,
    QUIT_SENTINEL,
    handle_slash_command,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(tmp_path: Path) -> MagicMock:
    """Create a minimal mock ChatEngine suitable for slash command tests."""
    engine = MagicMock()

    # Memory manager mock
    engine.memory.project_dir = tmp_path
    engine.memory.db_path = tmp_path / ".llmforge" / "memory.db"
    engine.memory.session_id = "test1234"
    engine.memory.session_start.isoformat.return_value = "2026-03-19T12:00:00"
    engine.memory.project_state = {
        "scanned_at": "2026-03-19T12:00:00",
        "project_dir": str(tmp_path),
        "configs": [
            {"name": "train.yaml", "model": "SmolLM2-135M", "mode": "sft"},
        ],
        "trained_models": [
            {"name": "my-model", "status": "complete", "size_mb": 512.0},
        ],
        "data_sources": [
            {"name": "data.jsonl", "path": "data/data.jsonl", "size_mb": 1.5},
        ],
        "active_config": "config.yaml",
    }

    # Permission system mock
    engine.permissions.auto_approve = False

    # Messages list
    engine.messages = [{"role": "user", "content": "hello"}]

    return engine


def _init_memory_db(db_path: Path) -> None:
    """Create a minimal memory database for testing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            summary TEXT,
            turns INTEGER DEFAULT 0,
            tokens_used INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            relevance_score REAL DEFAULT 1.0
        );
        INSERT INTO sessions (id, started_at, summary, turns)
        VALUES ('s1', '2026-03-18T10:00:00', 'Discussed model training', 5);
        INSERT INTO sessions (id, started_at, summary, turns)
        VALUES ('s2', '2026-03-17T09:00:00', 'Set up project', 3);
        INSERT INTO memories (category, content, relevance_score)
        VALUES ('training', 'Model uses SmolLM2-135M', 1.0);
        INSERT INTO memories (category, content, relevance_score)
        VALUES ('decision', 'Chose LoRA over full fine-tune', 0.9);
        INSERT INTO memories (category, content, relevance_score)
        VALUES ('bug', 'OOM with batch_size 16', 0.8);
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHelpCommand:
    """Tests for /help."""

    def test_help_lists_all_commands(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/help", engine)
        assert result is not None
        assert "Available commands" in result
        # Every registered command should appear in the help output
        for cmd in COMMANDS:
            assert cmd in result

    def test_help_includes_descriptions(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/help", engine)
        for info in COMMANDS.values():
            assert info["description"] in result

    def test_bare_slash_shows_help(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/", engine)
        assert result is not None
        assert "Available commands" in result


class TestStatusCommand:
    """Tests for /status."""

    def test_status_shows_project_dir(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/status", engine)
        assert str(tmp_path) in result

    def test_status_shows_configs(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/status", engine)
        assert "train.yaml" in result
        assert "SmolLM2-135M" in result

    def test_status_shows_models(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/status", engine)
        assert "my-model" in result
        assert "complete" in result

    def test_status_shows_data(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/status", engine)
        assert "data.jsonl" in result

    def test_status_empty_project(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.memory.project_state = {
            "scanned_at": "2026-03-19T12:00:00",
            "project_dir": str(tmp_path),
            "configs": [],
            "trained_models": [],
            "data_sources": [],
        }
        result = handle_slash_command("/status", engine)
        assert "(none found)" in result


class TestHardwareCommand:
    """Tests for /hardware."""

    def test_hardware_returns_info(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        mock_hw = '{"os": "Darwin", "cpu": "arm", "python_version": "3.12.0", "ram_total_gb": 16}'
        with patch("llm_forge.chat.tools.execute_tool", return_value=mock_hw):
            result = handle_slash_command("/hardware", engine)
        assert "Darwin" in result
        assert "arm" in result
        assert "3.12.0" in result

    def test_hardware_handles_json_error(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        with patch("llm_forge.chat.tools.execute_tool", return_value="not-json"):
            result = handle_slash_command("/hardware", engine)
        assert "not-json" in result


class TestMemoryCommand:
    """Tests for /memory."""

    def test_memory_shows_stats(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        _init_memory_db(engine.memory.db_path)
        result = handle_slash_command("/memory", engine)
        assert "test1234" in result
        assert "Past sessions:" in result
        assert "Stored memories:" in result
        # Should show counts from the DB
        assert "2" in result  # 2 sessions with summaries
        assert "3" in result  # 3 memories

    def test_memory_without_db(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.memory.db_path = tmp_path / "nonexistent" / "memory.db"
        result = handle_slash_command("/memory", engine)
        # Should not crash, just show 0s
        assert "Past sessions:" in result
        assert "0" in result


class TestClearCommand:
    """Tests for /clear."""

    def test_clear_returns_sentinel(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/clear", engine)
        assert result == CLEAR_SENTINEL


class TestConfigCommand:
    """Tests for /config."""

    def test_config_shows_active(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/config", engine)
        assert "config.yaml" in result

    def test_config_no_active(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.memory.project_state["active_config"] = None
        result = handle_slash_command("/config", engine)
        assert "train.yaml" in result  # lists available configs

    def test_config_no_configs_at_all(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.memory.project_state["active_config"] = None
        engine.memory.project_state["configs"] = []
        result = handle_slash_command("/config", engine)
        assert "No configs found" in result


class TestModelsCommand:
    """Tests for /models."""

    def test_models_lists_outputs(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        outputs = tmp_path / "outputs"
        model_dir = outputs / "my-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        result = handle_slash_command("/models", engine)
        assert "my-model" in result
        assert "complete" in result

    def test_models_no_outputs_dir(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/models", engine)
        assert "No outputs/ directory found" in result

    def test_models_empty_outputs(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        (tmp_path / "outputs").mkdir()
        result = handle_slash_command("/models", engine)
        assert "no models found" in result.lower()


class TestAutoCommand:
    """Tests for /auto."""

    def test_auto_toggles_on(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.permissions.auto_approve = False
        result = handle_slash_command("/auto", engine)
        assert "ON" in result
        assert engine.permissions.auto_approve is True

    def test_auto_toggles_off(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine.permissions.auto_approve = True
        result = handle_slash_command("/auto", engine)
        assert "OFF" in result
        assert engine.permissions.auto_approve is False


class TestQuitCommand:
    """Tests for /quit."""

    def test_quit_returns_sentinel(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/quit", engine)
        assert result == QUIT_SENTINEL


class TestVersionCommand:
    """Tests for /version."""

    def test_version_shows_version(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/version", engine)
        assert "llm-forge v" in result
        # Should contain a version string like 0.4.1
        from llm_forge import __version__

        assert __version__ in result


class TestUnknownCommand:
    """Tests for unknown / invalid commands."""

    def test_unknown_command(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/foobar", engine)
        assert "Unknown command" in result
        assert "/foobar" in result
        assert "/help" in result

    def test_unknown_command_with_args(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/badcmd some args", engine)
        assert "Unknown command" in result
        assert "/badcmd" in result


class TestNotACommand:
    """Tests for regular (non-slash) input."""

    def test_regular_text_returns_none(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("hello world", engine)
        assert result is None

    def test_empty_string_returns_none(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("", engine)
        assert result is None

    def test_text_with_slash_inside(self, tmp_path: Path) -> None:
        """A slash not at the start should not trigger a command."""
        engine = _make_engine(tmp_path)
        result = handle_slash_command("use /help for info", engine)
        assert result is None


class TestCommandsCaseInsensitive:
    """Test that commands are case-insensitive."""

    def test_help_uppercase(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/HELP", engine)
        assert result is not None
        assert "Available commands" in result

    def test_status_mixed_case(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/Status", engine)
        assert result is not None
        assert "Project Status" in result

    def test_version_upper(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/VERSION", engine)
        assert result is not None
        assert "llm-forge v" in result

    def test_quit_mixed(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = handle_slash_command("/Quit", engine)
        assert result == QUIT_SENTINEL


class TestCommandRegistry:
    """Tests for the COMMANDS registry integrity."""

    def test_all_commands_have_handler(self) -> None:
        for cmd, info in COMMANDS.items():
            assert "handler" in info, f"{cmd} missing 'handler'"
            assert callable(info["handler"]), f"{cmd} handler is not callable"

    def test_all_commands_have_description(self) -> None:
        for cmd, info in COMMANDS.items():
            assert "description" in info, f"{cmd} missing 'description'"
            assert len(info["description"]) > 0, f"{cmd} has empty description"

    def test_all_commands_start_with_slash(self) -> None:
        for cmd in COMMANDS:
            assert cmd.startswith("/"), f"Command {cmd!r} must start with '/'"

    def test_expected_commands_registered(self) -> None:
        expected = {
            "/help",
            "/status",
            "/hardware",
            "/memory",
            "/clear",
            "/config",
            "/models",
            "/auto",
            "/quit",
            "/version",
        }
        assert set(COMMANDS.keys()) == expected
