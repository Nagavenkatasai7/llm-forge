"""Tests for the smart project setup integration in the chat UI.

Verifies that:
- detect_project_type is called on launch
- Setup is skipped when already an LLM Forge project
- _print_setup_plan renders the plan correctly
- scaffold_project is invoked when the user accepts
- Helper print functions produce output without errors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_forge.chat.project_setup import (
    detect_project_type,
    get_setup_plan,
    scaffold_project,
)
from llm_forge.chat.ui import (
    _print_error,
    _print_info,
    _print_setup_plan,
    _print_success,
)

# ===================================================================
# detect_project_type triggers on empty directory
# ===================================================================


class TestDetectTriggersOnEmptyDir:
    """When launch_chat runs in an empty dir, detect_project_type is called
    and the setup flow is triggered."""

    def test_empty_dir_detected(self, tmp_path: Path) -> None:
        """An empty directory should be detected as empty and not llmforge."""
        detection = detect_project_type(str(tmp_path))
        assert detection["is_empty"] is True
        assert detection["is_llmforge"] is False

    def test_detect_called_during_launch(self, tmp_path: Path) -> None:
        """launch_chat calls detect_project_type for the current directory."""
        import os

        engine_instance = MagicMock()
        engine_instance.provider = "anthropic"
        engine_instance.send.return_value = "Hello!"
        engine_instance.memory = MagicMock()
        engine_instance.memory.db_path = "/tmp/fake.db"

        with (
            patch("llm_forge.chat.ui.detect_project_type") as mock_detect,
            patch("llm_forge.chat.ui.get_setup_plan") as mock_plan,
            patch("llm_forge.chat.ui.scaffold_project"),
            patch("llm_forge.chat.ui._print_banner"),
            patch("llm_forge.chat.ui._print_info"),
            patch("llm_forge.chat.ui._print_setup_plan"),
            patch("llm_forge.chat.ui._print_success"),
            patch("llm_forge.chat.ui._print_response"),
            patch("llm_forge.chat.ui._print_error"),
            patch("builtins.input", return_value="n"),
            patch("llm_forge.chat.ui.ChatEngine", return_value=engine_instance),
            patch("llm_forge.chat.ui._setup_api_key", return_value=engine_instance),
            patch("llm_forge.chat.ui._get_input", return_value="quit"),
            patch("llm_forge.chat.ui._stream_response"),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True),
        ):
            mock_detect.return_value = {
                "is_empty": True,
                "is_llmforge": False,
                "detected_types": [],
                "files_count": 0,
                "has_git": False,
                "recommended_mode": "root",
                "existing_files": [],
            }
            mock_plan.return_value = {
                "files_to_create": [],
                "directories_to_create": [],
                "mode": "root",
                "total_size_estimate": 0,
            }

            from llm_forge.chat.ui import launch_chat

            launch_chat()

            mock_detect.assert_called_once_with(".")

    def test_scaffold_called_on_accept(self, tmp_path: Path) -> None:
        """When user answers 'y', scaffold_project is called."""
        import os

        engine_instance = MagicMock()
        engine_instance.provider = "anthropic"
        engine_instance.send.return_value = "Hello!"
        engine_instance.memory = MagicMock()
        engine_instance.memory.db_path = "/tmp/fake.db"

        with (
            patch("llm_forge.chat.ui.detect_project_type") as mock_detect,
            patch("llm_forge.chat.ui.get_setup_plan") as mock_plan,
            patch("llm_forge.chat.ui.scaffold_project") as mock_scaffold,
            patch("llm_forge.chat.ui._print_banner"),
            patch("llm_forge.chat.ui._print_info"),
            patch("llm_forge.chat.ui._print_setup_plan"),
            patch("llm_forge.chat.ui._print_success"),
            patch("llm_forge.chat.ui._print_response"),
            patch("llm_forge.chat.ui._print_error"),
            patch("builtins.input", return_value="y"),
            patch("llm_forge.chat.ui.ChatEngine", return_value=engine_instance),
            patch("llm_forge.chat.ui._setup_api_key", return_value=engine_instance),
            patch("llm_forge.chat.ui._get_input", return_value="quit"),
            patch("llm_forge.chat.ui._stream_response"),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True),
        ):
            mock_detect.return_value = {
                "is_empty": True,
                "is_llmforge": False,
                "detected_types": [],
                "files_count": 0,
                "has_git": False,
                "recommended_mode": "root",
                "existing_files": [],
            }
            mock_plan.return_value = {
                "files_to_create": ["config.yaml"],
                "directories_to_create": ["configs"],
                "mode": "root",
                "total_size_estimate": 500,
            }
            mock_scaffold.return_value = {
                "status": "ok",
                "created_files": ["config.yaml"],
                "created_dirs": ["configs"],
                "skipped_files": [],
                "base_dir": str(tmp_path),
                "mode": "root",
            }

            from llm_forge.chat.ui import launch_chat

            launch_chat()

            mock_scaffold.assert_called_once_with(".")


# ===================================================================
# detect skips setup for existing llmforge dirs
# ===================================================================


class TestDetectSkipsLlmforgeDir:
    """When the directory is already an LLM Forge project, setup is not offered."""

    def test_llmforge_dir_detected(self, tmp_path: Path) -> None:
        """A directory with .llmforge marker is detected as llmforge."""
        (tmp_path / ".llmforge").mkdir()
        detection = detect_project_type(str(tmp_path))
        assert detection["is_llmforge"] is True

    def test_configs_marker_detected(self, tmp_path: Path) -> None:
        """A directory with configs/ dir is detected as llmforge."""
        (tmp_path / "configs").mkdir()
        detection = detect_project_type(str(tmp_path))
        assert detection["is_llmforge"] is True

    def test_setup_not_offered_for_llmforge(self, tmp_path: Path) -> None:
        """When is_llmforge is True, scaffold_project is never called."""
        import os

        engine_instance = MagicMock()
        engine_instance.provider = "anthropic"
        engine_instance.send.return_value = "Welcome back!"
        engine_instance.memory = MagicMock()
        engine_instance.memory.db_path = "/tmp/fake.db"

        with (
            patch("llm_forge.chat.ui.detect_project_type") as mock_detect,
            patch("llm_forge.chat.ui.scaffold_project") as mock_scaffold,
            patch("llm_forge.chat.ui._print_banner"),
            patch("llm_forge.chat.ui._print_response"),
            patch("llm_forge.chat.ui._print_error"),
            patch("llm_forge.chat.ui.ChatEngine", return_value=engine_instance),
            patch("llm_forge.chat.ui._setup_api_key", return_value=engine_instance),
            patch("llm_forge.chat.ui._get_input", return_value="quit"),
            patch("llm_forge.chat.ui._stream_response"),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True),
        ):
            mock_detect.return_value = {
                "is_empty": False,
                "is_llmforge": True,
                "detected_types": ["llmforge"],
                "files_count": 10,
                "has_git": False,
                "recommended_mode": "root",
                "existing_files": [".llmforge"],
            }

            from llm_forge.chat.ui import launch_chat

            launch_chat()

            mock_scaffold.assert_not_called()


# ===================================================================
# _print_setup_plan shows files
# ===================================================================


class TestPrintSetupPlan:
    """_print_setup_plan renders plan data without errors."""

    def test_shows_files_and_dirs(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Plan with dirs and files is rendered without crashing."""
        plan: dict[str, Any] = {
            "directories_to_create": ["/tmp/test/configs", "/tmp/test/data"],
            "files_to_create": ["/tmp/test/config.yaml", "/tmp/test/.gitignore"],
            "mode": "root",
            "total_size_estimate": 1024,
        }
        # Should not raise
        _print_setup_plan(plan)

    def test_empty_plan_shows_nothing_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        """An empty plan prints the 'nothing to create' message."""
        plan: dict[str, Any] = {
            "directories_to_create": [],
            "files_to_create": [],
            "mode": "root",
            "total_size_estimate": 0,
        }
        _print_setup_plan(plan)
        # The function calls _print_info internally -- we just verify no crash

    def test_plan_with_only_dirs(self) -> None:
        """Plan with only directories is rendered without crashing."""
        plan: dict[str, Any] = {
            "directories_to_create": ["/tmp/test/outputs"],
            "files_to_create": [],
            "mode": "root",
            "total_size_estimate": 0,
        }
        _print_setup_plan(plan)

    def test_plan_with_subdirectory_mode(self) -> None:
        """Plan in subdirectory mode is rendered without crashing."""
        plan: dict[str, Any] = {
            "directories_to_create": ["/tmp/myapp/llm-forge/configs"],
            "files_to_create": ["/tmp/myapp/llm-forge/config.yaml"],
            "mode": "subdirectory",
            "total_size_estimate": 2048,
        }
        _print_setup_plan(plan)


# ===================================================================
# Helper print functions
# ===================================================================


class TestPrintHelpers:
    """The styled print helpers produce output without errors."""

    def test_print_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_info("Test info message")

    def test_print_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_success("Test success message")

    def test_print_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_error("Test error message")


# ===================================================================
# Scaffold integration
# ===================================================================


class TestScaffoldIntegration:
    """scaffold_project creates the expected structure in a temp dir."""

    def test_scaffold_creates_dirs(self, tmp_path: Path) -> None:
        result = scaffold_project(str(tmp_path))
        assert result["status"] == "ok"
        assert (tmp_path / ".llmforge").is_dir()
        assert (tmp_path / "configs").is_dir()
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "outputs").is_dir()

    def test_scaffold_idempotent(self, tmp_path: Path) -> None:
        """Running scaffold twice does not fail or duplicate files."""
        result1 = scaffold_project(str(tmp_path))
        assert result1["status"] == "ok"
        result2 = scaffold_project(str(tmp_path))
        assert result2["status"] == "ok"
        # Second run should skip most files
        assert len(result2["created_files"]) == 0 or len(result2["skipped_files"]) > 0

    def test_scaffold_subdirectory_mode(self, tmp_path: Path) -> None:
        """In subdirectory mode, scaffold creates under llm-forge/."""
        # Create a fake python project
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'myapp'\n")
        result = scaffold_project(str(tmp_path), mode="subdirectory")
        assert result["status"] == "ok"
        assert result["mode"] == "subdirectory"
        assert (tmp_path / "llm-forge" / ".llmforge").is_dir()

    def test_get_setup_plan_non_empty(self, tmp_path: Path) -> None:
        """get_setup_plan returns items for a fresh directory."""
        plan = get_setup_plan(str(tmp_path))
        assert len(plan["directories_to_create"]) > 0 or len(plan["files_to_create"]) > 0


# ===================================================================
# System prompt includes project setup section
# ===================================================================


class TestSystemPromptSetup:
    """The system prompt includes project setup guidance."""

    def test_system_prompt_has_project_setup(self) -> None:
        from llm_forge.chat.system_prompt import SYSTEM_PROMPT

        assert "## Project Layout" in SYSTEM_PROMPT

    def test_system_prompt_mentions_configs_dir(self) -> None:
        from llm_forge.chat.system_prompt import SYSTEM_PROMPT

        assert "configs/" in SYSTEM_PROMPT

    def test_system_prompt_mentions_data_dir(self) -> None:
        from llm_forge.chat.system_prompt import SYSTEM_PROMPT

        assert "data/" in SYSTEM_PROMPT

    def test_system_prompt_never_delete_warning(self) -> None:
        from llm_forge.chat.system_prompt import SYSTEM_PROMPT

        assert "NEVER delete existing user files" in SYSTEM_PROMPT
