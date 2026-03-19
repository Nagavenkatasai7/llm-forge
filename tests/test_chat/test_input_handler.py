"""Tests for llm_forge.chat.input_handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_forge.chat.input_handler import (
    SLASH_COMMANDS,
    create_input_session,
    get_user_input,
)

# ---------------------------------------------------------------------------
# create_input_session
# ---------------------------------------------------------------------------


def test_create_session_returns_object():
    """create_input_session returns a PromptSession when prompt_toolkit is available."""
    session = create_input_session()
    # prompt_toolkit may or may not be installed in the test env.
    # If installed, we get a PromptSession; if not, None.
    try:
        from prompt_toolkit import PromptSession

        assert isinstance(session, PromptSession)
    except ImportError:
        assert session is None


def test_create_session_returns_none_without_prompt_toolkit():
    """create_input_session returns None when prompt_toolkit is not importable."""
    with patch.dict("sys.modules", {"prompt_toolkit": None}):
        # Force re-import path by calling the function (it does a local import)
        # We need to mock at the import level inside the function
        with patch(
            "builtins.__import__",
            side_effect=_make_import_blocker("prompt_toolkit"),
        ):
            result = create_input_session()
    assert result is None


# ---------------------------------------------------------------------------
# get_user_input — fallback path
# ---------------------------------------------------------------------------


def test_get_user_input_fallback(monkeypatch):
    """With session=None, get_user_input falls back to input()."""
    monkeypatch.setattr("builtins.input", lambda *args: "hello from fallback")
    # Also stub out rich so the fallback path that uses Console doesn't interfere
    # with the plain input() branch.
    with patch(
        "builtins.__import__",
        side_effect=_make_import_blocker("rich"),
    ):
        result = get_user_input(session=None)
    assert result == "hello from fallback"


def test_get_user_input_fallback_with_rich(monkeypatch):
    """With session=None and rich available, the Rich-styled prompt is used."""
    monkeypatch.setattr("builtins.input", lambda: "hello rich")
    result = get_user_input(session=None)
    assert result == "hello rich"


def test_get_user_input_with_session():
    """When a prompt_toolkit session is provided, its prompt() method is called."""
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    session = create_input_session()
    assert session is not None

    # Mock the session.prompt method to avoid actual terminal interaction
    session.prompt = MagicMock(return_value="mocked input")
    result = get_user_input(session=session)
    assert result == "mocked input"
    session.prompt.assert_called_once()


# ---------------------------------------------------------------------------
# Slash command completions
# ---------------------------------------------------------------------------


def test_slash_commands_in_completer():
    """All expected slash commands are present in SLASH_COMMANDS."""
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
        "/model",
        "/paste",
    }
    assert expected == set(SLASH_COMMANDS)


def test_slash_commands_match_registry():
    """SLASH_COMMANDS list is in sync with the slash_commands.py COMMANDS registry."""
    from llm_forge.chat.slash_commands import COMMANDS

    registry_commands = set(COMMANDS.keys())
    completer_commands = set(SLASH_COMMANDS)
    assert completer_commands == registry_commands, (
        f"Mismatch between input_handler.SLASH_COMMANDS and slash_commands.COMMANDS: "
        f"missing from completer={registry_commands - completer_commands}, "
        f"extra in completer={completer_commands - registry_commands}"
    )


def test_completer_has_all_commands_when_prompt_toolkit_available():
    """The WordCompleter in the session contains all slash commands."""
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        pytest.skip("prompt_toolkit not installed")

    session = create_input_session()
    assert session is not None
    # The completer's words attribute holds the completion candidates
    completer_words = session.completer.words
    for cmd in SLASH_COMMANDS:
        assert cmd in completer_words, f"{cmd} missing from completer"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_import_blocker(blocked_name: str):
    """Return a side_effect function for builtins.__import__ that blocks one package."""
    _real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocker(name, *args, **kwargs):
        if name == blocked_name or name.startswith(blocked_name + "."):
            raise ImportError(f"Mocked: {name} not available")
        return _real_import(name, *args, **kwargs)

    return _blocker
