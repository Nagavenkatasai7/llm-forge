"""Textual-based TUI for LLM Forge.

A proper terminal application with:
- Fixed input bar at the bottom
- Scrollable chat area above
- Status bar showing model, memory, provider
- Tool actions displayed cleanly
- Markdown rendering for responses
- Async API calls (non-blocking UI)

This is an ALTERNATIVE to the print-based ``ui.py``.  Users opt in
with ``llm-forge --tui``.  The default ``llm-forge`` command still
launches the print-based UI for maximum compatibility.
"""

from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown, Static

# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------


class StatusBar(Static):
    """Bottom status bar showing model, provider, and memory count."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $primary-background;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_name: str = "unknown"
        self.provider: str = "unknown"
        self.memory_count: int = 0
        self.session_count: int = 0

    def render(self) -> str:  # type: ignore[override]
        return (
            f" Model: {self.model_name} | "
            f"Provider: {self.provider} | "
            f"Memory: {self.memory_count} insights | "
            f"/ help | Esc interrupt | Ctrl+C quit"
        )

    def update_info(
        self,
        model: str = "",
        provider: str = "",
        memory: int = 0,
        sessions: int = 0,
    ) -> None:
        """Refresh the status bar with new values."""
        if model:
            self.model_name = model
        if provider:
            self.provider = provider
        self.memory_count = memory
        self.session_count = sessions
        self.refresh()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class LLMForgeApp(App):
    """The LLM Forge terminal application."""

    CSS = """
    #chat-view {
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 1 1;
    }

    #user-input {
        dock: bottom;
        margin: 0 1;
    }

    .user-message {
        border-left: thick cyan;
        padding-left: 1;
        margin: 1 0;
    }

    .assistant-message {
        border-left: thick green;
        padding-left: 1;
        margin: 1 0;
    }

    .tool-message {
        color: $text-muted;
        padding-left: 2;
        margin: 0;
    }

    .system-message {
        color: $text-muted;
        text-style: italic;
        margin: 0;
    }
    """

    TITLE = "LLM Forge"
    SUB_TITLE = "Build your own AI model"

    BINDINGS = [
        Binding("escape", "interrupt", "Interrupt", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
    ]

    def __init__(self, engine=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.engine = engine

    # -- Compose the widget tree ----------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Static(
                "[bold cyan]LLM Forge[/bold cyan] - Build your own AI model\n"
                "Just tell me what you want to build.\n",
                classes="system-message",
            )
        yield Input(
            placeholder="Type your message... (/ for commands, Esc to interrupt)",
            id="user-input",
        )
        yield StatusBar()
        yield Footer()

    # -- Lifecycle -------------------------------------------------------

    def on_mount(self) -> None:
        """Focus the input bar and optionally send a greeting."""
        self.query_one(Input).focus()
        if self.engine:
            self._update_status()
            self._send_greeting()

    # -- Status helpers --------------------------------------------------

    def _update_status(self) -> None:
        """Sync the status bar with current engine state."""
        if not self.engine:
            return
        status = self.query_one(StatusBar)

        model_name = getattr(self.engine, "model_key", "unknown") or "unknown"
        provider = getattr(self.engine, "provider", "unknown") or "unknown"

        memory_count = 0
        session_count = 0
        try:
            import sqlite3

            conn = sqlite3.connect(str(self.engine.memory.db_path))
            memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            session_count = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE summary IS NOT NULL"
            ).fetchone()[0]
            conn.close()
        except Exception:
            pass

        status.update_info(
            model=model_name,
            provider=provider.upper(),
            memory=memory_count,
            sessions=session_count,
        )

    # -- Message helpers -------------------------------------------------

    def _add_message(self, content: str, css_class: str) -> None:
        """Append a Rich-markup Static to the chat view."""
        chat_view = self.query_one("#chat-view")
        msg = Static(content, classes=css_class)
        chat_view.mount(msg)
        msg.scroll_visible()

    def _add_user_message(self, text: str) -> None:
        self._add_message(f"[bold cyan]You:[/bold cyan] {text}", "user-message")

    def _add_forge_message(self, text: str) -> None:
        """Add a Forge response rendered as Markdown."""
        chat_view = self.query_one("#chat-view")
        container = Static(classes="assistant-message")
        chat_view.mount(container)
        header = Static("[bold green]Forge:[/bold green]")
        container.mount(header)
        md = Markdown(text)
        container.mount(md)
        md.scroll_visible()

    def _add_tool_message(self, text: str) -> None:
        self._add_message(f"  [dim cyan]| {text}[/dim cyan]", "tool-message")

    def _add_system_message(self, text: str) -> None:
        self._add_message(f"  [dim]{text}[/dim]", "system-message")

    def _add_thinking(self) -> None:
        self._add_system_message("thinking...")

    # -- Input handling --------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """React to the user pressing Enter in the input bar."""
        text = event.value.strip()
        if not text:
            return

        event.input.clear()

        # Quit keywords
        if text.lower() in ("quit", "exit", "q", "bye"):
            self._save_session()
            self.exit()
            return

        # Slash commands
        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        # Regular message
        self._add_user_message(text)
        self._send_message(text)

    def _handle_slash_command(self, command: str) -> None:
        """Dispatch a slash command locally."""
        from llm_forge.chat.slash_commands import (
            CLEAR_SENTINEL,
            QUIT_SENTINEL,
            handle_slash_command,
        )

        result = handle_slash_command(command, self.engine)
        if result is None:
            return
        if result == QUIT_SENTINEL:
            self._save_session()
            self.exit()
            return
        if result == CLEAR_SENTINEL:
            chat_view = self.query_one("#chat-view")
            chat_view.remove_children()
            self._add_system_message("Conversation cleared. Memory preserved.")
            if self.engine:
                self.engine.messages.clear()
            return
        # Display command output
        self._add_system_message(result)

    # -- Background workers (non-blocking) --------------------------------

    @work(thread=True)
    def _send_message(self, text: str) -> None:
        """Send a user message to the engine in a background thread."""
        if not self.engine:
            return

        def on_tool_start(name, data):
            from llm_forge.chat.ui import _format_tool_detail

            detail = _format_tool_detail(name, data)
            self.call_from_thread(self._add_tool_message, detail)

        def on_tool_end(name, data, result):
            from llm_forge.chat.ui import _summarize_tool_result

            summary = _summarize_tool_result(name, result)
            if summary:
                self.call_from_thread(self._add_tool_message, f"  {summary}")

        prev_start = self.engine.on_tool_start
        prev_end = self.engine.on_tool_end
        self.engine.on_tool_start = on_tool_start
        self.engine.on_tool_end = on_tool_end

        self.call_from_thread(self._add_thinking)

        try:
            response = self.engine.send(text)
            self.call_from_thread(self._add_forge_message, response)
        except Exception as e:
            self.call_from_thread(self._add_system_message, f"Error: {e}")
        finally:
            self.engine.on_tool_start = prev_start
            self.engine.on_tool_end = prev_end
            self.call_from_thread(self._update_status)

    @work(thread=True)
    def _send_greeting(self) -> None:
        """Send the initial greeting prompt in a background thread."""
        if not self.engine:
            return

        def on_tool_start(name, data):
            from llm_forge.chat.ui import _format_tool_detail

            detail = _format_tool_detail(name, data)
            self.call_from_thread(self._add_tool_message, detail)

        def on_tool_end(name, data, result):
            from llm_forge.chat.ui import _summarize_tool_result

            summary = _summarize_tool_result(name, result)
            if summary:
                self.call_from_thread(self._add_tool_message, f"  {summary}")

        self.engine.on_tool_start = on_tool_start
        self.engine.on_tool_end = on_tool_end

        self.call_from_thread(self._add_thinking)

        try:
            greeting = self.engine.send(
                "The user just launched llm-forge. Check the project state and "
                "session history. If there's past work, welcome them back with "
                "context. If new user, greet them warmly and ask what kind of AI "
                "model they want to build. Also detect hardware if not already in "
                "memory. Keep it to 3-4 sentences."
            )
            self.call_from_thread(self._add_forge_message, greeting)
        except Exception as e:
            self.call_from_thread(self._add_system_message, f"Error: {e}")
        finally:
            self.call_from_thread(self._update_status)

    # -- Actions ---------------------------------------------------------

    def action_interrupt(self) -> None:
        """Cancel any running background workers."""
        self.workers.cancel_all()
        self._add_system_message("Interrupted -- type your next instruction")

    def action_clear_chat(self) -> None:
        """Clear all messages from the chat view."""
        chat_view = self.query_one("#chat-view")
        chat_view.remove_children()
        self._add_system_message("Chat cleared. Memory preserved.")
        if self.engine:
            self.engine.messages.clear()

    # -- Shutdown --------------------------------------------------------

    def _save_session(self) -> None:
        """Persist the session before the app exits.

        Named ``_save_session`` (not ``_shutdown``) to avoid overriding
        Textual's internal ``App._shutdown`` coroutine.
        """
        if self.engine:
            try:
                self.engine.end_session()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def launch_tui(provider: str | None = None) -> None:
    """Launch the Textual TUI version of LLM Forge.

    This is the entry point called by ``llm-forge --tui``.
    """
    from llm_forge.chat.engine import ChatEngine
    from llm_forge.chat.project_setup import detect_project_type, scaffold_project

    # Minimal project setup (non-interactive)
    detection = detect_project_type(".")
    if not detection["is_llmforge"]:
        scaffold_project(".", auto_approve=True)

    # Create engine
    engine = ChatEngine(provider=provider, project_dir=".")

    # Launch the app
    app = LLMForgeApp(engine=engine)
    app.run()
