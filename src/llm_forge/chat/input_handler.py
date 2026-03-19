"""Advanced input handling with prompt_toolkit for LLM Forge.

Features:
- Bracketed paste: multi-line paste stays as one input (no auto-submit on newlines)
- Slash command completion: type / and see available commands
- History: up/down arrows browse previous inputs
- Esc+Enter: insert explicit newline for multi-line messages
- Enter: submit the current input
"""

from __future__ import annotations

# Slash commands offered by the completer — kept in sync with slash_commands.py
SLASH_COMMANDS = [
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
]


def create_input_session():
    """Create a prompt_toolkit session with all features enabled.

    Falls back to ``None`` if prompt_toolkit is not installed, in which case
    :func:`get_user_input` uses plain ``input()``.
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings

        completer = WordCompleter(SLASH_COMMANDS, sentence=True)

        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _insert_newline(event):
            """Esc+Enter inserts a newline (for multi-line input)."""
            event.current_buffer.insert_text("\n")

        session = PromptSession(
            history=InMemoryHistory(),
            completer=completer,
            complete_while_typing=False,  # Only complete on Tab
            key_bindings=bindings,
            # Bracketed paste is enabled by default in prompt_toolkit —
            # pasted text with newlines stays as one input block.
        )

        return session
    except ImportError:
        return None


def get_user_input(session=None, prompt_text: str = "You: ") -> str:
    """Get user input using prompt_toolkit (or fallback to ``input()``).

    With prompt_toolkit:
    - Pasted multi-line text stays as one input (bracketed paste)
    - Tab shows slash command completions
    - Up/Down browses history
    - Esc+Enter for explicit newline
    - Enter submits

    Without prompt_toolkit:
    - Falls back to basic ``input()``
    """
    if session is not None:
        try:
            from prompt_toolkit.formatted_text import HTML

            text = session.prompt(HTML("<cyan><b>You: </b></cyan>"))
            return text
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception:
            pass

    # Fallback
    try:
        from rich.console import Console

        Console().print("[bold cyan]You:[/bold cyan] ", end="")
        return input()
    except ImportError:
        return input(prompt_text)
