"""Tests for the MemoryManager class.

Covers database initialization, save/recall memory, session lifecycle,
user profile, training run logging, project scanning, token estimation,
compaction detection, and context block building.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_forge.chat.memory import (
    CHARS_PER_TOKEN,
    COMPACTION_THRESHOLD_TOKENS,
    DB_NAME,
    LLMFORGE_DIR,
    MemoryManager,
)

# ===================================================================
# Database initialization
# ===================================================================


class TestInit:
    """Test MemoryManager initialization and database creation."""

    def test_init_creates_db(self, tmp_path: Path) -> None:
        """MemoryManager creates .llmforge/memory.db on init."""
        mm = MemoryManager(project_dir=tmp_path)
        db_path = tmp_path / LLMFORGE_DIR / DB_NAME
        assert db_path.exists()
        assert mm.db_path == db_path

    def test_init_creates_llmforge_dir(self, tmp_path: Path) -> None:
        """MemoryManager creates the .llmforge directory."""
        mm = MemoryManager(project_dir=tmp_path)
        assert mm.llmforge_dir.is_dir()

    def test_init_sets_session_id(self, tmp_path: Path) -> None:
        """MemoryManager generates a unique session ID."""
        mm = MemoryManager(project_dir=tmp_path)
        assert isinstance(mm.session_id, str)
        assert len(mm.session_id) == 8

    def test_init_registers_session_in_db(self, tmp_path: Path) -> None:
        """The constructor inserts a row into the sessions table."""
        mm = MemoryManager(project_dir=tmp_path)
        import sqlite3

        conn = sqlite3.connect(str(mm.db_path))
        row = conn.execute("SELECT id FROM sessions WHERE id = ?", (mm.session_id,)).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == mm.session_id


# ===================================================================
# Save and recall memory
# ===================================================================


class TestMemoryOperations:
    """Test save_memory() and recall_memory()."""

    def test_save_and_recall_memory(self, tmp_path: Path) -> None:
        """save_memory() then recall_memory() returns the saved item."""
        mm = MemoryManager(project_dir=tmp_path)
        mm.save_memory(category="training_lesson", content="LoRA r=8 works best")
        result = json.loads(mm.recall_memory("LoRA"))
        assert result["count"] >= 1
        assert any("LoRA r=8" in m["content"] for m in result["memories"])

    def test_recall_empty(self, tmp_path: Path) -> None:
        """recall_memory() with no matches returns empty list."""
        mm = MemoryManager(project_dir=tmp_path)
        result = json.loads(mm.recall_memory("nonexistent_xyz_keyword"))
        assert result["count"] == 0
        assert result["memories"] == []

    def test_save_memory_returns_ok(self, tmp_path: Path) -> None:
        """save_memory() returns a JSON string with status=ok."""
        mm = MemoryManager(project_dir=tmp_path)
        result = json.loads(mm.save_memory("user_preference", "prefers QLoRA"))
        assert result["status"] == "ok"

    def test_recall_memory_respects_limit(self, tmp_path: Path) -> None:
        """recall_memory() respects the limit parameter."""
        mm = MemoryManager(project_dir=tmp_path)
        for i in range(5):
            mm.save_memory("test", f"item {i} about training")
        result = json.loads(mm.recall_memory("training", limit=2))
        assert result["count"] == 2

    def test_recall_matches_category(self, tmp_path: Path) -> None:
        """recall_memory() matches against category as well as content."""
        mm = MemoryManager(project_dir=tmp_path)
        mm.save_memory(category="project_decision", content="Use Llama 3.2")
        result = json.loads(mm.recall_memory("project_decision"))
        assert result["count"] >= 1


# ===================================================================
# Session lifecycle
# ===================================================================


class TestSessionLifecycle:
    """Test session creation and end_session()."""

    def test_session_init_creates_session(self, tmp_path: Path) -> None:
        """Initializing MemoryManager creates a session row in the DB."""
        mm = MemoryManager(project_dir=tmp_path)
        import sqlite3

        conn = sqlite3.connect(str(mm.db_path))
        rows = conn.execute("SELECT id FROM sessions").fetchall()
        conn.close()
        assert len(rows) >= 1

    def test_end_session_saves_summary(self, tmp_path: Path) -> None:
        """end_session() updates the session with a summary and turn count."""
        mm = MemoryManager(project_dir=tmp_path)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there. How can I help?"},
            {"role": "user", "content": "Train a model"},
            {"role": "assistant", "content": "Sure. Let me detect your hardware."},
        ]
        mm.end_session(messages)

        import sqlite3

        conn = sqlite3.connect(str(mm.db_path))
        row = conn.execute(
            "SELECT summary, turns, ended_at FROM sessions WHERE id = ?",
            (mm.session_id,),
        ).fetchone()
        conn.close()
        assert row is not None
        summary, turns, ended_at = row
        assert summary is not None
        assert turns == 2  # 2 user messages
        assert ended_at is not None

    def test_get_session_history_empty(self, tmp_path: Path) -> None:
        """get_session_history() returns empty when no sessions have summaries."""
        mm = MemoryManager(project_dir=tmp_path)
        result = json.loads(mm.get_session_history())
        assert result["count"] == 0


# ===================================================================
# User profile
# ===================================================================


class TestUserProfile:
    """Test set_user_profile() and get_user_profile()."""

    def test_user_profile_round_trip(self, tmp_path: Path) -> None:
        """set_user_profile() and get_user_profile() round-trip correctly."""
        mm = MemoryManager(project_dir=tmp_path)
        mm.set_user_profile("hardware", "M2 MacBook Pro 16GB")
        mm.set_user_profile("skill_level", "intermediate")
        profile = mm.get_user_profile()
        assert profile["hardware"] == "M2 MacBook Pro 16GB"
        assert profile["skill_level"] == "intermediate"

    def test_user_profile_empty(self, tmp_path: Path) -> None:
        """get_user_profile() returns empty dict when nothing is set."""
        mm = MemoryManager(project_dir=tmp_path)
        profile = mm.get_user_profile()
        assert profile == {}

    def test_user_profile_overwrite(self, tmp_path: Path) -> None:
        """set_user_profile() overwrites existing keys (INSERT OR REPLACE)."""
        mm = MemoryManager(project_dir=tmp_path)
        mm.set_user_profile("gpu", "A100")
        mm.set_user_profile("gpu", "H100")
        profile = mm.get_user_profile()
        assert profile["gpu"] == "H100"


# ===================================================================
# Training run logging
# ===================================================================


class TestTrainingRunLogging:
    """Test log_training_run() and get_training_history()."""

    def test_log_training_run(self, tmp_path: Path) -> None:
        """log_training_run() stores a run and get_training_history() retrieves it."""
        mm = MemoryManager(project_dir=tmp_path)
        result = json.loads(
            mm.log_training_run(
                config_path="configs/test.yaml",
                model_name="test-model",
                base_model="Llama-3.2-1B",
                mode="lora",
                output_dir="outputs/test",
                final_loss=1.5,
                eval_loss=1.6,
                status="completed",
                notes="test run",
            )
        )
        assert result["status"] == "ok"

        history = mm.get_training_history()
        assert len(history) >= 1
        run = history[0]
        assert run["model_name"] == "test-model"
        assert run["base_model"] == "Llama-3.2-1B"
        assert run["mode"] == "lora"
        assert run["final_loss"] == 1.5
        assert run["eval_loss"] == 1.6
        assert run["status"] == "completed"

    def test_get_training_history_empty(self, tmp_path: Path) -> None:
        """get_training_history() returns empty list when no runs exist."""
        mm = MemoryManager(project_dir=tmp_path)
        history = mm.get_training_history()
        assert history == []

    def test_training_history_respects_limit(self, tmp_path: Path) -> None:
        """get_training_history() respects the limit parameter."""
        mm = MemoryManager(project_dir=tmp_path)
        for i in range(5):
            mm.log_training_run(
                config_path=f"configs/run{i}.yaml",
                model_name=f"model-{i}",
                base_model="base",
                mode="lora",
                output_dir=f"outputs/run{i}",
            )
        history = mm.get_training_history(limit=2)
        assert len(history) == 2


# ===================================================================
# Project scanning
# ===================================================================


class TestProjectScanning:
    """Test _scan_project() behavior."""

    def test_project_scan_empty_dir(self, tmp_path: Path) -> None:
        """Scanning an empty directory returns empty state lists."""
        mm = MemoryManager(project_dir=tmp_path)
        state = mm.project_state
        assert state["configs"] == []
        assert state["trained_models"] == []
        assert state["data_sources"] == []
        assert state["active_training"] is None

    def test_project_scan_with_configs(self, tmp_path: Path) -> None:
        """Scanning a directory with configs/*.yaml lists them."""
        import yaml

        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        config_data = {
            "model": {"name": "Llama-3.2-1B"},
            "training": {"mode": "lora"},
        }
        (configs_dir / "test_config.yaml").write_text(
            yaml.dump(config_data, default_flow_style=False)
        )

        mm = MemoryManager(project_dir=tmp_path)
        assert len(mm.project_state["configs"]) == 1
        cfg = mm.project_state["configs"][0]
        assert cfg["name"] == "test_config.yaml"
        assert cfg["model"] == "Llama-3.2-1B"
        assert cfg["mode"] == "lora"

    def test_project_scan_with_data_files(self, tmp_path: Path) -> None:
        """Scanning detects data files in the data/ directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.jsonl").write_text('{"text": "hello"}\n')

        mm = MemoryManager(project_dir=tmp_path)
        assert len(mm.project_state["data_sources"]) == 1
        assert mm.project_state["data_sources"][0]["name"] == "train.jsonl"

    def test_project_scan_detects_active_config(self, tmp_path: Path) -> None:
        """Scanning detects a config.yaml in the project root."""
        (tmp_path / "config.yaml").write_text("model:\n  name: test\n")
        mm = MemoryManager(project_dir=tmp_path)
        assert mm.project_state.get("active_config") == "config.yaml"

    def test_project_scan_saves_state_file(self, tmp_path: Path) -> None:
        """Scanning saves project_state.json to .llmforge/."""
        _mm = MemoryManager(project_dir=tmp_path)
        state_file = tmp_path / LLMFORGE_DIR / "project_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert "project_dir" in data
        assert data["project_dir"] == str(tmp_path)


# ===================================================================
# Token estimation
# ===================================================================


class TestTokenEstimation:
    """Test estimate_tokens() and its variants."""

    def test_token_estimation(self, tmp_path: Path) -> None:
        """estimate_tokens() returns a reasonable number based on char count."""
        mm = MemoryManager(project_dir=tmp_path)
        messages = [{"role": "user", "content": "a" * 400}]
        tokens = mm.estimate_tokens(messages)
        expected = 400 // CHARS_PER_TOKEN
        assert tokens == expected

    def test_token_estimation_empty(self, tmp_path: Path) -> None:
        """estimate_tokens() returns 0 for empty message list."""
        mm = MemoryManager(project_dir=tmp_path)
        assert mm.estimate_tokens([]) == 0

    def test_token_estimation_list_content(self, tmp_path: Path) -> None:
        """estimate_tokens() handles list-type content (tool results)."""
        mm = MemoryManager(project_dir=tmp_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "some result data"},
                ],
            }
        ]
        tokens = mm.estimate_tokens(messages)
        assert tokens > 0


# ===================================================================
# Compaction detection
# ===================================================================


class TestCompaction:
    """Test needs_compaction() and compact_messages()."""

    def test_needs_compaction_small(self, tmp_path: Path) -> None:
        """Small messages don't need compaction."""
        mm = MemoryManager(project_dir=tmp_path)
        messages = [{"role": "user", "content": "Hello"}]
        assert mm.needs_compaction(messages) is False

    def test_needs_compaction_large(self, tmp_path: Path) -> None:
        """Large messages trigger compaction."""
        mm = MemoryManager(project_dir=tmp_path)
        # Generate enough content to exceed the threshold
        chars_needed = (COMPACTION_THRESHOLD_TOKENS + 1) * CHARS_PER_TOKEN
        messages = [{"role": "user", "content": "x" * chars_needed}]
        assert mm.needs_compaction(messages) is True

    def test_compact_messages_no_op_when_small(self, tmp_path: Path) -> None:
        """compact_messages() returns messages unchanged when below threshold."""
        mm = MemoryManager(project_dir=tmp_path)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = mm.compact_messages(messages)
        assert result == messages


# ===================================================================
# Build context block
# ===================================================================


class TestBuildContextBlock:
    """Test build_context_block() output."""

    def test_build_context_block(self, tmp_path: Path) -> None:
        """build_context_block() returns a string with project state."""
        mm = MemoryManager(project_dir=tmp_path)
        block = mm.build_context_block()
        assert isinstance(block, str)
        assert "Current Project State" in block
        assert str(tmp_path) in block

    def test_build_context_block_includes_profile(self, tmp_path: Path) -> None:
        """build_context_block() includes user profile when set."""
        mm = MemoryManager(project_dir=tmp_path)
        mm.set_user_profile("hardware", "A100 80GB")
        block = mm.build_context_block()
        assert "User Profile" in block
        assert "A100 80GB" in block

    def test_build_context_block_includes_memories(self, tmp_path: Path) -> None:
        """build_context_block() includes saved memories."""
        mm = MemoryManager(project_dir=tmp_path)
        mm.save_memory("training_lesson", "Always use eval split")
        block = mm.build_context_block()
        assert "Key Memories" in block
        assert "Always use eval split" in block

    def test_build_context_block_includes_training_history(self, tmp_path: Path) -> None:
        """build_context_block() includes recent training history."""
        mm = MemoryManager(project_dir=tmp_path)
        mm.log_training_run(
            config_path="test.yaml",
            model_name="my-model",
            base_model="base",
            mode="lora",
            output_dir="outputs/test",
            final_loss=1.2,
            status="completed",
        )
        block = mm.build_context_block()
        assert "Recent Training History" in block
        assert "my-model" in block
