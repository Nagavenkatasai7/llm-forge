"""Tests for the ``llm-forge demo`` CLI command.

Verifies that the command is registered, has proper help text, and that
the Typer app exposes a ``demo`` subcommand.
"""

from __future__ import annotations

from typer.testing import CliRunner

from llm_forge.cli import app

runner = CliRunner()


# ===================================================================
# Command registration
# ===================================================================


class TestDemoCommandExists:
    """Ensure the demo command is registered on the Typer app."""

    def test_demo_command_exists(self) -> None:
        """The app should have a 'demo' command registered."""
        command_names: list[str] = []
        for cmd_info in app.registered_commands:
            if cmd_info.name:
                command_names.append(cmd_info.name)
            elif cmd_info.callback:
                command_names.append(cmd_info.callback.__name__)
        assert "demo" in command_names


# ===================================================================
# Help text
# ===================================================================


class TestDemoHelp:
    """Ensure ``demo --help`` works and describes the command."""

    def test_demo_help(self) -> None:
        result = runner.invoke(app, ["demo", "--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "train" in output_lower or "demo" in output_lower
        assert "tiny" in output_lower or "zero" in output_lower or "5 minutes" in output_lower
