"""Tests for project detection and scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_forge.chat.project_setup import (
    detect_project_type,
    get_setup_plan,
    scaffold_project,
)

# ---------------------------------------------------------------------------
# detect_project_type
# ---------------------------------------------------------------------------


class TestDetectProjectType:
    """Tests for detect_project_type."""

    def test_detect_empty_directory(self, tmp_path: Path) -> None:
        result = detect_project_type(tmp_path)
        assert result["is_empty"] is True
        assert result["is_llmforge"] is False
        assert result["detected_types"] == []
        assert result["recommended_mode"] == "root"

    def test_detect_nodejs_project(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        result = detect_project_type(tmp_path)
        assert result["is_empty"] is False
        assert "nodejs" in result["detected_types"]
        assert "package.json" in result["existing_files"]
        assert result["recommended_mode"] == "subdirectory"

    def test_detect_python_project(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text("torch\n")
        result = detect_project_type(tmp_path)
        assert result["is_empty"] is False
        assert "python" in result["detected_types"]
        assert "requirements.txt" in result["existing_files"]
        assert result["recommended_mode"] == "subdirectory"

    def test_detect_llmforge_project(self, tmp_path: Path) -> None:
        (tmp_path / ".llmforge").mkdir()
        (tmp_path / "configs").mkdir()
        result = detect_project_type(tmp_path)
        assert result["is_llmforge"] is True
        assert "llmforge" in result["detected_types"]
        assert result["recommended_mode"] == "root"

    def test_detect_nonexistent_directory(self, tmp_path: Path) -> None:
        result = detect_project_type(tmp_path / "does-not-exist")
        assert result["is_empty"] is True
        assert result["detected_types"] == []

    def test_detect_has_git(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        result = detect_project_type(tmp_path)
        assert result["has_git"] is True
        # .git is hidden, so the directory should still be considered empty
        assert result["is_empty"] is True

    def test_detect_multiple_types(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "requirements.txt").write_text("flask\n")
        result = detect_project_type(tmp_path)
        assert "nodejs" in result["detected_types"]
        assert "python" in result["detected_types"]

    def test_detect_dotnet_glob_markers(self, tmp_path: Path) -> None:
        (tmp_path / "MyApp.csproj").write_text("<Project/>")
        result = detect_project_type(tmp_path)
        assert "dotnet" in result["detected_types"]


# ---------------------------------------------------------------------------
# scaffold_project
# ---------------------------------------------------------------------------


class TestScaffoldProject:
    """Tests for scaffold_project."""

    def test_scaffold_root_mode(self, tmp_path: Path) -> None:
        result = scaffold_project(tmp_path, mode="root")
        assert result["status"] == "ok"
        assert result["mode"] == "root"

        # Directories created
        assert (tmp_path / "configs").is_dir()
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "outputs").is_dir()
        assert (tmp_path / ".llmforge").is_dir()
        assert (tmp_path / "examples" / "data").is_dir()

        # Files created
        assert (tmp_path / ".gitignore").is_file()
        assert (tmp_path / "config.yaml").is_file()

        # Starter configs
        assert (tmp_path / "configs" / "starter_lora.yaml").is_file()
        assert (tmp_path / "configs" / "starter_qlora.yaml").is_file()

        # Example data files
        assert (tmp_path / "examples" / "data" / "sample_train.jsonl").is_file()
        assert (tmp_path / "examples" / "data" / "sample_sharegpt.jsonl").is_file()
        assert (tmp_path / "examples" / "data" / "sample_completion.txt").is_file()

        # .gitignore content
        gitignore = (tmp_path / ".gitignore").read_text()
        assert "outputs/" in gitignore
        assert "*.safetensors" in gitignore
        assert "!examples/data/*.jsonl" in gitignore

        # created_files should be populated
        assert len(result["created_files"]) > 0
        assert result["base_dir"] == str(tmp_path)

    def test_scaffold_subdirectory_mode(self, tmp_path: Path) -> None:
        # Put something in the directory so it's not empty
        (tmp_path / "package.json").write_text("{}")

        result = scaffold_project(tmp_path, mode="subdirectory")
        assert result["status"] == "ok"
        assert result["mode"] == "subdirectory"

        forge_dir = tmp_path / "llm-forge"
        assert result["base_dir"] == str(forge_dir)
        assert forge_dir.is_dir()
        assert (forge_dir / "configs").is_dir()
        assert (forge_dir / "data").is_dir()
        assert (forge_dir / "outputs").is_dir()
        assert (forge_dir / ".llmforge").is_dir()
        assert (forge_dir / ".gitignore").is_file()
        assert (forge_dir / "config.yaml").is_file()

    def test_scaffold_never_overwrites(self, tmp_path: Path) -> None:
        # Create a file that scaffold would want to create
        (tmp_path / ".gitignore").write_text("# my custom gitignore\n")
        original_content = "# my custom gitignore\n"

        result = scaffold_project(tmp_path, mode="root")
        assert result["status"] == "ok"

        # The original file should NOT be overwritten
        assert (tmp_path / ".gitignore").read_text() == original_content

        # It should appear in skipped_files
        assert str(tmp_path / ".gitignore") in result["skipped_files"]

    def test_scaffold_auto_mode_empty(self, tmp_path: Path) -> None:
        result = scaffold_project(tmp_path, mode="auto")
        assert result["status"] == "ok"
        assert result["mode"] == "root"
        assert result["base_dir"] == str(tmp_path)

    def test_scaffold_auto_mode_existing(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        result = scaffold_project(tmp_path, mode="auto")
        assert result["status"] == "ok"
        assert result["mode"] == "subdirectory"
        assert result["base_dir"] == str(tmp_path / "llm-forge")

    def test_scaffold_no_examples(self, tmp_path: Path) -> None:
        result = scaffold_project(tmp_path, mode="root", include_examples=False)
        assert result["status"] == "ok"
        assert not (tmp_path / "examples" / "data").exists()

    def test_scaffold_subdirectory_appends_parent_gitignore(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / ".gitignore").write_text("node_modules/\n")

        result = scaffold_project(tmp_path, mode="subdirectory")
        assert result["status"] == "ok"

        parent_gitignore = (tmp_path / ".gitignore").read_text()
        assert "llm-forge/" in parent_gitignore
        # Original content preserved
        assert "node_modules/" in parent_gitignore

    def test_scaffold_idempotent(self, tmp_path: Path) -> None:
        """Running scaffold twice should not fail or overwrite files."""
        result1 = scaffold_project(tmp_path, mode="root")
        assert result1["status"] == "ok"
        first_count = len(result1["created_files"])

        result2 = scaffold_project(tmp_path, mode="root")
        assert result2["status"] == "ok"
        # Second run should create nothing new
        assert len(result2["created_files"]) == 0
        # All files should be in skipped
        assert len(result2["skipped_files"]) == first_count


# ---------------------------------------------------------------------------
# get_setup_plan
# ---------------------------------------------------------------------------


class TestGetSetupPlan:
    """Tests for get_setup_plan."""

    def test_get_setup_plan(self, tmp_path: Path) -> None:
        plan = get_setup_plan(tmp_path, mode="root")

        assert "files_to_create" in plan
        assert "directories_to_create" in plan
        assert plan["mode"] == "root"
        assert plan["total_size_estimate"] > 0

        # Should list expected files
        file_names = [Path(f).name for f in plan["files_to_create"]]
        assert "starter_lora.yaml" in file_names
        assert "starter_qlora.yaml" in file_names
        assert "config.yaml" in file_names
        assert ".gitignore" in file_names
        assert "sample_train.jsonl" in file_names

        # Should NOT create anything
        assert not (tmp_path / "configs").exists()
        assert not (tmp_path / ".gitignore").exists()

    def test_plan_auto_mode_empty(self, tmp_path: Path) -> None:
        plan = get_setup_plan(tmp_path, mode="auto")
        assert plan["mode"] == "root"

    def test_plan_auto_mode_existing(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\n")
        plan = get_setup_plan(tmp_path, mode="auto")
        assert plan["mode"] == "subdirectory"

    def test_plan_excludes_existing_files(self, tmp_path: Path) -> None:
        """If files already exist, they should not appear in the plan."""
        (tmp_path / ".gitignore").write_text("# existing\n")
        plan = get_setup_plan(tmp_path, mode="root")
        file_names = [Path(f).name for f in plan["files_to_create"]]
        assert ".gitignore" not in file_names


# ---------------------------------------------------------------------------
# Tool integration (execute_tool dispatch)
# ---------------------------------------------------------------------------


class TestToolIntegration:
    """Verify the tools are wired into execute_tool correctly."""

    def test_detect_project_tool(self, tmp_path: Path) -> None:
        from llm_forge.chat.tools import execute_tool

        result_str = execute_tool("detect_project", {"directory": str(tmp_path)})
        result = json.loads(result_str)
        assert result["status"] == "ok"
        assert "is_empty" in result
        assert "detected_types" in result

    def test_setup_project_tool(self, tmp_path: Path) -> None:
        from llm_forge.chat.tools import execute_tool

        result_str = execute_tool(
            "setup_project",
            {"directory": str(tmp_path), "mode": "root"},
        )
        result = json.loads(result_str)
        assert result["status"] == "ok"
        assert (tmp_path / "configs").is_dir()

    def test_detect_project_in_tools_list(self) -> None:
        from llm_forge.chat.tools import TOOLS

        names = [t["name"] for t in TOOLS]
        assert "detect_project" in names
        assert "setup_project" in names
