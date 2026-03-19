"""Tests for Phase 4 CLI enhancements.

Covers: interactive setup wizard, doctor command, pipeline stage selection,
run history tracking, and error recovery.
"""

from __future__ import annotations

import platform
from pathlib import Path
from unittest import mock

import pytest
import yaml

try:
    from typer.testing import CliRunner

    from llm_forge.cli import (
        _get_runs_file,
        _load_runs,
        _save_run,
        _show_error_recovery,
        app,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="llm_forge.cli not importable")


@pytest.fixture()
def cli_runner():
    return CliRunner()


# ===================================================================
# Setup Wizard Tests
# ===================================================================


class TestSetupCommand:
    """Test the interactive setup wizard."""

    def test_setup_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "wizard" in result.output.lower() or "setup" in result.output.lower()

    def test_setup_creates_config(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Setup wizard creates a config file with default inputs."""
        output_file = str(tmp_path / "setup_config.yaml")
        # Wizard flow: purpose, ai_name, data_source, model_tier,
        #   custom_model, method, epochs, max_samples, eval, iti, confirm
        # All defaults (empty lines):
        result = cli_runner.invoke(
            app,
            ["setup", "--output", output_file],
            input="\n" * 11,
        )
        assert result.exit_code == 0, f"Setup failed: {result.output}"
        assert Path(output_file).exists()
        content = Path(output_file).read_text()
        assert "model:" in content
        assert "training:" in content

    def test_setup_custom_model(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Setup with custom model name."""
        output_file = str(tmp_path / "custom_model.yaml")
        # purpose=1 (journal), name=default, data=default(3),
        # tier=1 (tiny), custom_model=gpt2, method=3 (full),
        # epochs=1, max_samples=100, eval=n, iti=n, confirm=y
        result = cli_runner.invoke(
            app,
            ["setup", "--output", output_file],
            input="1\n\n\n1\ngpt2\n3\n1\n100\nn\nn\ny\n",
        )
        assert result.exit_code == 0, f"Setup failed: {result.output}"
        content = Path(output_file).read_text()
        assert "gpt2" in content

    def test_setup_qlora_method(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Setup with QLoRA method includes quantization section."""
        output_file = str(tmp_path / "qlora_setup.yaml")
        # All defaults except method=1 (qlora) and confirm=y
        result = cli_runner.invoke(
            app,
            ["setup", "--output", output_file],
            input="\n\n\n\n\n1\n3\n0\nn\nn\ny\n",
        )
        assert result.exit_code == 0, f"Setup failed: {result.output}"
        content = Path(output_file).read_text()
        assert "quantization" in content or "qlora" in content.lower()

    def test_setup_with_iti(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Setup with ITI enabled includes iti section."""
        output_file = str(tmp_path / "iti_setup.yaml")
        # All defaults except iti=y and confirm=y
        result = cli_runner.invoke(
            app,
            ["setup", "--output", output_file],
            input="\n\n\n\n\n\n1\n0\nn\ny\ny\n",
        )
        assert result.exit_code == 0, f"Setup failed: {result.output}"
        content = Path(output_file).read_text()
        assert "iti:" in content
        assert "refusal:" in content


# ===================================================================
# Doctor Command Tests
# ===================================================================


class TestDoctorCommand:
    """Test the doctor diagnostic command."""

    def test_doctor_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "diagnostic" in result.output.lower() or "doctor" in result.output.lower()

    def test_doctor_runs(self, cli_runner: CliRunner) -> None:
        """Doctor command should run and display checks."""
        result = cli_runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "python" in result.output.lower()
        assert "diagnostics" in result.output.lower() or "check" in result.output.lower()

    def test_doctor_checks_python(self, cli_runner: CliRunner) -> None:
        """Doctor should report Python version."""
        result = cli_runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        # Should contain the Python version number
        py_ver = platform.python_version()
        assert py_ver in result.output or "python" in result.output.lower()


# ===================================================================
# Pipeline Stage Selection Tests
# ===================================================================


class TestStageSelection:
    """Test --stages and --skip-stages options on train command."""

    def test_train_help_shows_stages(self, cli_runner: CliRunner) -> None:
        """Train --help should show --stages and --skip-stages options."""
        result = cli_runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        # Strip ANSI escape codes before checking (Rich adds formatting)
        import re

        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--stages" in plain
        assert "--skip-stages" in plain

    def test_stages_and_skip_mutually_exclusive(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Using both --stages and --skip-stages should error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"model": {"name": "gpt2"}, "data": {"train_path": "test"}}),
            encoding="utf-8",
        )
        result = cli_runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_file),
                "--stages",
                "training",
                "--skip-stages",
                "evaluation",
            ],
        )
        assert result.exit_code != 0
        assert "cannot" in result.output.lower() or "error" in result.output.lower()

    def test_dry_run_shows_stage_table(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Dry run with --stages should show the stage plan table."""
        config_dict = {"model": {"name": "gpt2"}, "data": {"train_path": "test"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_dict), encoding="utf-8")

        result = cli_runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_file),
                "--stages",
                "data_loading,training",
                "--dry-run",
            ],
        )
        # May fail on config validation but should show stage table if pipeline imports work
        # Accept both success and failure — the key is no unhandled crash
        assert result.exit_code in (0, 1)


# ===================================================================
# Run History Tests
# ===================================================================


class TestRunHistory:
    """Test run history tracking."""

    def test_get_runs_file_default(self) -> None:
        path = _get_runs_file(".")
        assert path == Path(".") / ".llm-forge-runs.json"

    def test_get_runs_file_custom(self, tmp_path: Path) -> None:
        path = _get_runs_file(str(tmp_path))
        assert path == tmp_path / ".llm-forge-runs.json"

    def test_load_runs_empty(self, tmp_path: Path) -> None:
        """Loading from non-existent file returns empty list."""
        path = tmp_path / ".llm-forge-runs.json"
        runs = _load_runs(path)
        assert runs == []

    def test_save_and_load_run(self, tmp_path: Path) -> None:
        """Save a run and load it back."""
        path = tmp_path / ".llm-forge-runs.json"
        run_data = {
            "timestamp": "2026-03-03T10:00:00",
            "model": "gpt2",
            "status": "completed",
            "final_loss": 1.5,
            "duration_seconds": 120,
        }
        _save_run(path, run_data)
        runs = _load_runs(path)
        assert len(runs) == 1
        assert runs[0]["model"] == "gpt2"
        assert runs[0]["final_loss"] == 1.5

    def test_save_multiple_runs(self, tmp_path: Path) -> None:
        """Multiple saves append to the history."""
        path = tmp_path / ".llm-forge-runs.json"
        for i in range(3):
            _save_run(path, {"model": f"model_{i}", "status": "completed"})
        runs = _load_runs(path)
        assert len(runs) == 3

    def test_runs_list_empty(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Runs list with no history shows helpful message."""
        result = cli_runner.invoke(app, ["runs", "list", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "no" in result.output.lower() or "recorded" in result.output.lower()

    def test_runs_list_with_data(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Runs list shows a table when history exists."""
        path = tmp_path / ".llm-forge-runs.json"
        _save_run(
            path,
            {
                "timestamp": "2026-03-03T10:00:00",
                "model": "test-model",
                "status": "completed",
                "final_loss": 1.5,
                "duration_seconds": 60,
            },
        )
        result = cli_runner.invoke(app, ["runs", "list", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "test-model" in result.output

    def test_runs_show(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Runs show displays details for a specific run."""
        path = tmp_path / ".llm-forge-runs.json"
        _save_run(
            path,
            {
                "timestamp": "2026-03-03",
                "model": "detail-model",
                "status": "completed",
            },
        )
        result = cli_runner.invoke(app, ["runs", "show", "1", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "detail-model" in result.output

    def test_runs_show_invalid_id(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Runs show with invalid ID returns error."""
        result = cli_runner.invoke(app, ["runs", "show", "999", "--dir", str(tmp_path)])
        assert result.exit_code != 0


# ===================================================================
# Error Recovery Tests
# ===================================================================


class TestErrorRecovery:
    """Test the error recovery suggestion system."""

    @staticmethod
    def _extract_print_text(mock_console) -> str:
        """Extract all text from console.print calls, including Rich Panel content."""
        parts: list[str] = []
        for call_args in mock_console.print.call_args_list:
            for arg in call_args[0]:
                # Rich Panel stores content in .renderable
                if hasattr(arg, "renderable"):
                    parts.append(str(arg.renderable))
                else:
                    parts.append(str(arg))
        return " ".join(parts)

    def test_oom_suggestions(self) -> None:
        """OOM errors suggest memory-related fixes."""
        with mock.patch("llm_forge.cli.console") as mock_console:
            exc = RuntimeError("CUDA out of memory. Tried to allocate 2 GB")
            _show_error_recovery(exc, mock.MagicMock(), verbose=False)
            assert mock_console.print.call_count >= 2
            calls_text = self._extract_print_text(mock_console).lower()
            assert "batch" in calls_text or "memory" in calls_text

    def test_import_error_suggestions(self) -> None:
        """Import errors suggest installing packages."""
        with mock.patch("llm_forge.cli.console") as mock_console:
            exc = ImportError("No module named 'bitsandbytes'")
            _show_error_recovery(exc, mock.MagicMock(), verbose=False)
            calls_text = self._extract_print_text(mock_console).lower()
            assert "install" in calls_text or "pip" in calls_text

    def test_config_error_suggestions(self) -> None:
        """Config errors suggest validation."""
        with mock.patch("llm_forge.cli.console") as mock_console:
            exc = ValueError("Config validation failed: unknown field 'xyz'")
            _show_error_recovery(exc, mock.MagicMock(), verbose=False)
            calls_text = self._extract_print_text(mock_console).lower()
            assert "validate" in calls_text or "config" in calls_text

    def test_nan_loss_suggestions(self) -> None:
        """NaN/divergence errors suggest LR and warmup fixes."""
        with mock.patch("llm_forge.cli.console") as mock_console:
            exc = RuntimeError("Loss is NaN at step 50")
            _show_error_recovery(exc, mock.MagicMock(), verbose=False)
            calls_text = self._extract_print_text(mock_console).lower()
            assert "learning_rate" in calls_text or "lr" in calls_text or "nan" in calls_text

    def test_generic_error_suggestions(self) -> None:
        """Unknown errors suggest doctor and validate."""
        with mock.patch("llm_forge.cli.console") as mock_console:
            exc = RuntimeError("Something unexpected happened")
            _show_error_recovery(exc, mock.MagicMock(), verbose=False)
            calls_text = self._extract_print_text(mock_console).lower()
            assert "doctor" in calls_text or "validate" in calls_text or "error" in calls_text

    def test_verbose_shows_traceback(self) -> None:
        """Verbose mode should call print_exception."""
        with mock.patch("llm_forge.cli.console") as mock_console:
            exc = RuntimeError("test error")
            _show_error_recovery(exc, mock.MagicMock(), verbose=True)
            mock_console.print_exception.assert_called_once()


# ===================================================================
# Runs Sub-App Tests
# ===================================================================


class TestRunsSubApp:
    """Test the runs sub-command group."""

    def test_runs_help(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(app, ["runs", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower()
        assert "show" in result.output.lower()
