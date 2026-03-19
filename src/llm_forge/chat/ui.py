"""Rich terminal UI for the LLM Forge conversational assistant.

Provides clean, visually separated output inspired by modern CLI tools:
- Distinct visual blocks for user input, assistant responses, and system messages
- Proper streaming with a single Console instance (no per-chunk re-creation)
- Tool execution indicators during engine processing
- Markdown rendering for assistant responses
- Esc / Ctrl+C interrupt support during streaming
"""

from __future__ import annotations

import os
import sys
from typing import Any

from llm_forge.chat.engine import ChatEngine
from llm_forge.chat.project_setup import (
    detect_project_type,
    get_setup_plan,
    scaffold_project,
)

# ---------------------------------------------------------------------------
# Module-level console — reused across all output to avoid per-call overhead
# and to prevent the multi-Console streaming corruption bug.
# ---------------------------------------------------------------------------

_console = None


def _get_console():
    """Return (and lazily create) the shared Rich Console instance."""
    global _console  # noqa: PLW0603
    if _console is None:
        try:
            from rich.console import Console

            _console = Console()
        except ImportError:
            _console = None
    return _console


# ---------------------------------------------------------------------------
# Tool action display — shows what the engine is doing
# ---------------------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "detect_hardware": "[Hardware] Detecting system capabilities",
    "scan_data": "[Scan] Analyzing data sources",
    "write_config": "[Config] Writing configuration",
    "validate_config": "[Config] Validating configuration",
    "read_file": "[Read] Reading file",
    "write_file": "[Write] Writing file",
    "run_command": "[Shell] Running command",
    "start_training": "[Train] Starting training",
    "check_training_status": "[Train] Checking training status",
    "export_model": "[Export] Exporting model",
    "deploy_to_ollama": "[Deploy] Deploying to Ollama",
    "deploy_to_huggingface": "[Deploy] Deploying to HuggingFace",
    "search_huggingface": "[Search] Searching HuggingFace Hub",
    "download_model": "[Download] Downloading model",
    "save_memory": "[Memory] Saving insight",
    "recall_memory": "[Memory] Recalling context",
    "get_project_state": "[State] Loading project state",
    "get_session_history": "[History] Loading session history",
    "log_training_run": "[Log] Recording training metrics",
    "install_package": "[Install] Installing package",
    "install_dependencies": "[Install] Installing dependencies",
    "fetch_url": "[Fetch] Fetching URL",
    "convert_document": "[Convert] Converting document",
    "list_configs": "[Config] Listing configurations",
    "run_evaluation": "[Eval] Running evaluation",
    "read_training_logs": "[Logs] Reading training logs",
    "show_model_info": "[Info] Showing model info",
    "estimate_training": "[Estimate] Estimating training cost",
    "detect_project": "[Detect] Detecting project type",
    "setup_project": "[Setup] Setting up project",
}


def _print_tool_action(tool_name: str, description: str = "") -> None:
    """Show a tool execution indicator line.

    Displays a dim, indented label so the user can see what work the
    engine is performing behind the scenes, without cluttering the
    main conversation flow.
    """
    console = _get_console()
    label = description or _TOOL_LABELS.get(tool_name, f"[Tool] {tool_name}")
    if console is not None:
        console.print(f"  [dim]{label}[/dim]")
    else:
        print(f"  {label}")


# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------


def _get_version() -> str:
    """Return the llm-forge version string."""
    try:
        from llm_forge import __version__

        return __version__
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Styled output helpers
# ---------------------------------------------------------------------------


def _print_info(msg: str) -> None:
    """Print an informational message in dim cyan."""
    console = _get_console()
    if console is not None:
        console.print(f"  [dim cyan]{msg}[/dim cyan]")
    else:
        print(f"  [info] {msg}")


def _print_success(msg: str) -> None:
    """Print a success message in green."""
    console = _get_console()
    if console is not None:
        console.print(f"  [bold green]{msg}[/bold green]")
    else:
        print(f"  [ok] {msg}")


def _print_error(msg: str) -> None:
    """Print an error message in red with a distinct visual block."""
    console = _get_console()
    if console is not None:
        console.print()
        console.print(f"  [bold red]Error:[/bold red] [red]{msg}[/red]")
        console.print()
    else:
        print(f"\n  [error] {msg}\n")


def _print_setup_plan(plan: dict[str, Any]) -> None:
    """Display the scaffold plan as a Rich table (or plain text fallback)."""
    dirs = plan.get("directories_to_create", [])
    files = plan.get("files_to_create", [])
    mode = plan.get("mode", "root")
    size = plan.get("total_size_estimate", 0)

    if not dirs and not files:
        _print_info("Nothing to create -- project structure already exists.")
        return

    console = _get_console()
    if console is not None:
        try:
            from rich.table import Table

            table = Table(title="Setup Plan", border_style="cyan", show_lines=False)
            table.add_column("Type", style="bold", width=6)
            table.add_column("Path", style="cyan")

            for d in dirs:
                table.add_row("dir", d)
            for f in files:
                table.add_row("file", f)

            console.print(table)
            console.print(f"  [dim]Mode: {mode} | Estimated size: {size:,} bytes[/dim]")
            console.print()
            return
        except ImportError:
            pass

    # Plain-text fallback
    print(f"Setup plan (mode={mode}):")
    for d in dirs:
        print(f"  [dir]  {d}")
    for f in files:
        print(f"  [file] {f}")
    print(f"  Estimated size: {size:,} bytes")
    print()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    """Print the welcome banner with version and quick-help hints."""
    console = _get_console()
    if console is not None:
        try:
            from rich.panel import Panel
            from rich.text import Text

            version = _get_version()

            banner = Text()
            banner.append("LLM Forge", style="bold cyan")
            banner.append(f" v{version}", style="dim")
            banner.append("\n")
            banner.append("Build your own AI model. Just tell me what you want.\n\n", style="")
            banner.append("Type ", style="dim")
            banner.append("/", style="bold yellow")
            banner.append(" for commands  ", style="dim")
            banner.append("Esc", style="bold yellow")
            banner.append(" to interrupt  ", style="dim")
            banner.append("quit", style="bold red")
            banner.append(" to exit", style="dim")

            console.print(Panel(banner, border_style="cyan", padding=(1, 2)))
            console.print()
            return
        except ImportError:
            pass

    # Plain-text fallback
    print("=" * 56)
    print("  LLM Forge - Build your own AI model")
    print("  Just tell me what you want to build.")
    print("  Type '/' for commands, Esc to interrupt, 'quit' to exit.")
    print("=" * 56)
    print()


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------


def _print_response(text: str) -> None:
    """Print assistant response with clear visual separation.

    Uses thin separator rules and Markdown rendering instead of a
    heavy Panel, giving a cleaner, more modern appearance.
    """
    console = _get_console()
    if console is not None:
        try:
            from rich.markdown import Markdown
            from rich.rule import Rule

            console.print(Rule(style="dim green"))
            console.print("[bold green]Forge[/bold green]")
            console.print()
            console.print(Markdown(text))
            console.print()
            console.print(Rule(style="dim green"))
            console.print()
            return
        except ImportError:
            pass

    # Plain-text fallback
    print(f"\nForge: {text}\n")


# ---------------------------------------------------------------------------
# Esc-key detection
# ---------------------------------------------------------------------------


def _check_esc_pressed() -> bool:
    """Non-blocking check if Esc key was pressed (Unix only)."""
    old_settings = None
    try:
        import select
        import termios
        import tty

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    return True
        finally:
            if old_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Streaming response
# ---------------------------------------------------------------------------


def _stream_response(engine: ChatEngine, user_input: str) -> None:
    """Stream the assistant response with clean formatting.

    Key improvements over the previous implementation:
    - Uses a SINGLE shared Console (no ``Console()`` per chunk)
    - Prints header/footer rules for clear visual separation
    - Shows an interrupt hint only when actually interrupted
    """
    console = _get_console()
    interrupted = False
    text_chunks: list[str] = []
    first_token_received = False

    def on_text(chunk: str) -> None:
        """Called for each streamed text chunk."""
        nonlocal first_token_received
        text_chunks.append(chunk)
        if not first_token_received:
            first_token_received = True
        if console is not None:
            console.print(chunk, end="", highlight=False, soft_wrap=True)
        else:
            print(chunk, end="", flush=True)

    def interrupt_check() -> bool:
        nonlocal interrupted
        if interrupted:
            return True
        if _check_esc_pressed():
            interrupted = True
            return True
        return False

    # -- Header --
    if console is not None:
        try:
            from rich.rule import Rule

            console.print(Rule(style="dim green"))
            console.print("[bold green]Forge[/bold green]")
            console.print()
        except ImportError:
            console.print("[bold green]Forge:[/bold green] ", end="")
    else:
        print("Forge: ", end="")

    # -- Wire tool-action display into the engine for this request --
    prev_callback = engine.on_tool_start

    def _on_tool(name: str, _input_data: dict) -> None:
        _print_tool_action(name)

    engine.on_tool_start = _on_tool

    try:
        engine.send(user_input, on_text=on_text, interrupt_check=interrupt_check)
    finally:
        engine.on_tool_start = prev_callback

    # -- Footer --
    if console is not None:
        console.print()  # newline after last streamed chunk
        try:
            from rich.rule import Rule

            console.print(Rule(style="dim green"))
        except ImportError:
            pass
        console.print()
    else:
        print()
        print()

    if interrupted:
        if console is not None:
            console.print("  [dim italic]Interrupted -- type your next instruction[/dim italic]")
            console.print()
        else:
            print("  [interrupted -- type your next instruction]")
            print()


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------


def _get_input() -> str:
    """Get user input with a styled prompt."""
    console = _get_console()
    if console is not None:
        console.print("[bold cyan]You:[/bold cyan] ", end="")
        return input()
    return input("You: ")


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------


def _persist_api_key(key: str) -> None:
    """Write a validated API key to disk for future sessions."""
    from pathlib import Path

    llmforge_dir = Path(".llmforge")
    key_file = llmforge_dir / ".api_key"
    llmforge_dir.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key)
    key_file.chmod(0o600)

    _print_success("API key verified and saved! You won't need to enter it again.")


def _setup_api_key(engine: ChatEngine, provider: str | None) -> tuple[ChatEngine, str | None]:
    """Handle API key setup: load saved key, ask user, or fall back to wizard.

    Returns ``(engine, pending_key)`` where *pending_key* is non-None only
    when the user just pasted a brand-new key that has NOT yet been persisted
    to disk.  The caller should persist it after the key is proven valid
    (e.g. after a successful greeting).
    """
    from pathlib import Path

    console = _get_console()
    llmforge_dir = Path(".llmforge")
    key_file = llmforge_dir / ".api_key"

    # Try loading saved key (already validated in a prior session)
    if key_file.exists():
        saved_key = key_file.read_text().strip()
        if saved_key:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
            return ChatEngine(provider="anthropic", project_dir="."), None

    # Ask the user
    try:
        if console is not None:
            console.print(
                "\n[bold cyan]LLM Forge needs an API key "
                "for the conversational experience.[/bold cyan]\n"
            )
            console.print(
                "  Get a free Claude API key at: [link]https://console.anthropic.com/[/link]\n"
            )
            console.print("[bold]Paste your API key[/bold] (or press Enter to skip): ", end="")
            user_key = input().strip()
        else:
            print("\nLLM Forge needs an API key for the conversational experience.")
            print("Get one at: https://console.anthropic.com/")
            user_key = input("Paste your API key (or press Enter to skip): ").strip()
    except EOFError:
        user_key = ""

    if user_key:
        os.environ["ANTHROPIC_API_KEY"] = user_key
        _print_info("Key accepted. Validating with the API...")
        return ChatEngine(provider="anthropic", project_dir="."), user_key
    else:
        _print_info("No API key? No problem. Launching the free guided wizard instead.")
        from llm_forge.chat.wizard_fallback import launch_wizard_fallback

        launch_wizard_fallback()
        sys.exit(0)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def _shutdown(engine: ChatEngine) -> None:
    """Graceful shutdown: save session memory."""
    try:
        engine.end_session()
    except Exception:
        pass

    console = _get_console()
    if console is not None:
        console.print()
        console.print("  [cyan]Session saved. See you next time![/cyan]")
        console.print()
    else:
        print("\n  Session saved. See you next time!\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def launch_chat(provider: str | None = None) -> None:
    """Launch the interactive chat session with memory."""
    console = _get_console()

    _print_banner()

    # Smart project detection
    detection = detect_project_type(".")

    if not detection["is_llmforge"]:
        # Not an LLM Forge project yet -- offer setup
        if detection["is_empty"]:
            _print_info("New directory detected. Setting up LLM Forge project...")
        else:
            types = ", ".join(detection["detected_types"]) or "unknown"
            _print_info(f"Existing {types} project detected. LLM Forge will use a subdirectory.")

        # Show plan
        plan = get_setup_plan(".")
        _print_setup_plan(plan)

        # Ask permission
        try:
            if console is not None:
                console.print("[bold]Set up LLM Forge here?[/bold] [dim](Y/n)[/dim] ", end="")
                answer = input().strip().lower()
            else:
                answer = input("Set up LLM Forge here? (Y/n) ").strip().lower()
        except (ImportError, EOFError):
            answer = input("Set up LLM Forge here? (Y/n) ").strip().lower()

        if answer in ("", "y", "yes"):
            result = scaffold_project(".")
            if result["status"] == "ok":
                total = len(result["created_files"]) + len(result["created_dirs"])
                _print_success(f"Project set up! Created {total} items.")
            else:
                _print_error(f"Setup failed: {result.get('error', 'unknown')}")
                return
        else:
            _print_info("Skipping setup. You can set up later with: llm-forge init")

    # Now continue with normal engine initialization
    engine = ChatEngine(provider=provider, project_dir=".")

    # If no API key, try to load saved key or ask the user
    _pending_api_key: str | None = None
    if engine.provider == "none":
        engine, _pending_api_key = _setup_api_key(engine, provider)

    # Show memory status
    session_count = 0
    memory_count = 0
    try:
        import sqlite3

        conn = sqlite3.connect(str(engine.memory.db_path))
        session_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE summary IS NOT NULL"
        ).fetchone()[0]
        memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
    except Exception:
        pass

    if session_count > 0 or memory_count > 0:
        _print_info(
            f"Memory loaded: {session_count} past session(s), {memory_count} stored insight(s)"
        )

    # Check if anthropic is installed when using anthropic provider
    if engine.provider == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            _print_error(
                "The 'anthropic' package is not installed.\n"
                "  Run: pip install anthropic\n"
                "  Or:  pip install llm-forge-new[chat]"
            )
            # Try to auto-install
            try:
                if console is not None:
                    console.print("\n  [bold]Install it now?[/bold] [dim](Y/n)[/dim] ", end="")
                    answer = input().strip().lower()
                else:
                    answer = input("Install it now? (Y/n) ").strip().lower()
            except (ImportError, EOFError):
                answer = input("Install it now? (Y/n) ").strip().lower()

            if answer in ("", "y", "yes"):
                import subprocess

                _print_info("Installing anthropic SDK...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "anthropic", "-q"],
                    check=False,
                )
                _print_success("Installed! Reconnecting...")
                engine = ChatEngine(provider="anthropic", project_dir=".")
            else:
                _print_info("Launching free wizard instead.")
                from llm_forge.chat.wizard_fallback import launch_wizard_fallback

                launch_wizard_fallback()
                return

    # Wire tool-action display for the initial greeting and all future calls
    engine.on_tool_start = lambda name, _input: _print_tool_action(name)

    # Send initial greeting
    try:
        greeting = engine.send(
            "The user just launched llm-forge. Check the project state and session history. "
            "If there's past work, welcome them back with context. If new user, "
            "greet them warmly and ask what kind of AI model they want to build. "
            "Also detect hardware if not already in memory. Keep it to 3-4 sentences."
        )
        _print_response(greeting)
    except Exception as e:
        # Sanitise error message to avoid leaking API keys in tracebacks
        err_msg = str(e)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key and api_key in err_msg:
            err_msg = err_msg.replace(api_key, "sk-***REDACTED***")
        _print_error(f"Error connecting to API: {err_msg}")
        _print_info("Falling back to free wizard. You can try again later with an API key.")
        from llm_forge.chat.wizard_fallback import launch_wizard_fallback

        launch_wizard_fallback()
        return

    # The greeting succeeded, so the API key is valid -- persist it now.
    if _pending_api_key:
        _persist_api_key(_pending_api_key)
        _pending_api_key = None

    # Main conversation loop with streaming + Esc interrupt
    while True:
        try:
            user_input = _get_input()
        except (KeyboardInterrupt, EOFError):
            _shutdown(engine)
            break

        if not user_input.strip():
            continue

        if user_input.strip().lower() in ("quit", "exit", "q", "bye"):
            _shutdown(engine)
            break

        # Handle slash commands locally (don't send to Claude)
        if user_input.strip().startswith("/"):
            from llm_forge.chat.slash_commands import (
                CLEAR_SENTINEL,
                QUIT_SENTINEL,
                handle_slash_command,
            )

            result = handle_slash_command(user_input.strip(), engine)
            if result is not None:
                if result == QUIT_SENTINEL:
                    _shutdown(engine)
                    break
                if result == CLEAR_SENTINEL:
                    engine.messages.clear()
                    _print_info("Conversation cleared. Memory is preserved.")
                    continue
                _print_response(result)
                continue

        try:
            _stream_response(engine, user_input)
        except KeyboardInterrupt:
            if console is not None:
                console.print(
                    "\n  [dim italic]Interrupted -- type your next instruction[/dim italic]\n"
                )
            else:
                print("\n  [interrupted -- type your next instruction]\n")
            continue
        except Exception as e:
            _print_error(f"{e}")
            _print_info("Something went wrong. You can keep chatting or type 'quit' to exit.")
            continue
