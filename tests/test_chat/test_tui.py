"""Tests for the Textual TUI (``llm_forge.chat.tui``)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------


def test_launch_tui_importable():
    """The ``launch_tui`` function can be imported without side effects."""
    from llm_forge.chat.tui import launch_tui  # noqa: F401

    assert callable(launch_tui)


# ---------------------------------------------------------------------------
# 2. App creation
# ---------------------------------------------------------------------------


def test_app_can_be_created():
    """``LLMForgeApp()`` can be instantiated without an engine and without crashing."""
    from llm_forge.chat.tui import LLMForgeApp

    app = LLMForgeApp(engine=None)
    assert app is not None
    assert app.engine is None


def test_app_stores_engine():
    """The engine passed to the constructor is stored on the instance."""
    from llm_forge.chat.tui import LLMForgeApp

    sentinel = object()
    app = LLMForgeApp(engine=sentinel)
    assert app.engine is sentinel


# ---------------------------------------------------------------------------
# 3. Widget composition (uses Textual's async test pilot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_yields_widgets():
    """Header, Input, Footer, and StatusBar are all present after compose."""
    from textual.widgets import Footer, Header, Input

    from llm_forge.chat.tui import LLMForgeApp, StatusBar

    app = LLMForgeApp(engine=None)
    async with app.run_test() as pilot:  # noqa: F841
        # Header should be present
        headers = app.query(Header)
        assert len(headers) >= 1, "Header widget missing"

        # Input bar
        inputs = app.query(Input)
        assert len(inputs) >= 1, "Input widget missing"

        # Footer
        footers = app.query(Footer)
        assert len(footers) >= 1, "Footer widget missing"

        # StatusBar
        status_bars = app.query(StatusBar)
        assert len(status_bars) >= 1, "StatusBar widget missing"


@pytest.mark.asyncio
async def test_chat_view_has_welcome_message():
    """The initial chat view contains the welcome / system message."""
    from llm_forge.chat.tui import LLMForgeApp

    app = LLMForgeApp(engine=None)
    async with app.run_test():
        chat_view = app.query_one("#chat-view")
        # The welcome message should be mounted as a child Static
        children = list(chat_view.children)
        assert len(children) >= 1, "No welcome message in chat-view"


# ---------------------------------------------------------------------------
# 4. StatusBar
# ---------------------------------------------------------------------------


def test_status_bar_render_defaults():
    """StatusBar.render() returns a string with default values."""
    from llm_forge.chat.tui import StatusBar

    bar = StatusBar()
    rendered = bar.render()
    assert "Model:" in rendered
    assert "Provider:" in rendered
    assert "Memory:" in rendered


def test_status_bar_update_info():
    """StatusBar.update_info() changes the stored values."""
    from llm_forge.chat.tui import StatusBar

    bar = StatusBar()
    bar.update_info(model="TestModel", provider="TestProvider", memory=5, sessions=3)
    assert bar.model_name == "TestModel"
    assert bar.provider == "TestProvider"
    assert bar.memory_count == 5
    assert bar.session_count == 3


# ---------------------------------------------------------------------------
# 5. Message helper smoke tests (run in the async pilot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_user_message():
    """_add_user_message adds a child to the chat view."""
    from llm_forge.chat.tui import LLMForgeApp

    app = LLMForgeApp(engine=None)
    async with app.run_test():
        initial_count = len(list(app.query_one("#chat-view").children))
        app._add_user_message("Hello world")
        assert len(list(app.query_one("#chat-view").children)) == initial_count + 1


@pytest.mark.asyncio
async def test_add_system_message():
    """_add_system_message adds a child to the chat view."""
    from llm_forge.chat.tui import LLMForgeApp

    app = LLMForgeApp(engine=None)
    async with app.run_test():
        initial_count = len(list(app.query_one("#chat-view").children))
        app._add_system_message("System info")
        assert len(list(app.query_one("#chat-view").children)) == initial_count + 1


@pytest.mark.asyncio
async def test_add_tool_message():
    """_add_tool_message adds a child to the chat view."""
    from llm_forge.chat.tui import LLMForgeApp

    app = LLMForgeApp(engine=None)
    async with app.run_test():
        initial_count = len(list(app.query_one("#chat-view").children))
        app._add_tool_message("Detecting hardware")
        assert len(list(app.query_one("#chat-view").children)) == initial_count + 1


@pytest.mark.asyncio
async def test_add_forge_message():
    """_add_forge_message adds a Markdown-rendered assistant response."""
    from llm_forge.chat.tui import LLMForgeApp

    app = LLMForgeApp(engine=None)
    async with app.run_test():
        initial_count = len(list(app.query_one("#chat-view").children))
        app._add_forge_message("Hello, I'm **Forge**.")
        assert len(list(app.query_one("#chat-view").children)) == initial_count + 1


# ---------------------------------------------------------------------------
# 6. Key binding registration
# ---------------------------------------------------------------------------


def test_bindings_registered():
    """The expected key bindings are present on the app class."""
    from llm_forge.chat.tui import LLMForgeApp

    keys = {b.key for b in LLMForgeApp.BINDINGS}
    assert "escape" in keys
    assert "ctrl+c" in keys
    assert "ctrl+l" in keys


# ---------------------------------------------------------------------------
# 7. Action: clear chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_action_clear_chat():
    """action_clear_chat reduces the chat-view children count."""
    from llm_forge.chat.tui import LLMForgeApp

    app = LLMForgeApp(engine=None)
    async with app.run_test() as pilot:
        app._add_user_message("one")
        app._add_user_message("two")
        before = len(list(app.query_one("#chat-view").children))
        assert before >= 3  # welcome + 2 user messages

        app.action_clear_chat()
        await pilot.pause()  # let Textual process the DOM changes

        after = len(list(app.query_one("#chat-view").children))
        # After clearing we should have fewer children than before
        assert after < before
