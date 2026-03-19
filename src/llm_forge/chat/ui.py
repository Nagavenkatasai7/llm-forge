"""Rich terminal UI for the LLM Forge conversational assistant."""

from __future__ import annotations

import os
from typing import Any

from llm_forge.chat.engine import ChatEngine
from llm_forge.chat.project_setup import (
    detect_project_type,
    get_setup_plan,
    scaffold_project,
)

# ---------------------------------------------------------------------------
# Styled output helpers
# ---------------------------------------------------------------------------


def _print_info(msg: str) -> None:
    """Print an informational message in dim cyan."""
    try:
        from rich.console import Console

        Console().print(f"[dim cyan]{msg}[/dim cyan]")
    except ImportError:
        print(f"[info] {msg}")


def _print_success(msg: str) -> None:
    """Print a success message in green."""
    try:
        from rich.console import Console

        Console().print(f"[bold green]{msg}[/bold green]")
    except ImportError:
        print(f"[ok] {msg}")


def _print_error(msg: str) -> None:
    """Print an error message in red."""
    try:
        from rich.console import Console

        Console().print(f"[bold red]{msg}[/bold red]")
    except ImportError:
        print(f"[error] {msg}")


def _print_setup_plan(plan: dict[str, Any]) -> None:
    """Display the scaffold plan as a Rich table (or plain text fallback)."""
    dirs = plan.get("directories_to_create", [])
    files = plan.get("files_to_create", [])
    mode = plan.get("mode", "root")
    size = plan.get("total_size_estimate", 0)

    if not dirs and not files:
        _print_info("Nothing to create -- project structure already exists.")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Setup Plan", border_style="cyan", show_lines=False)
        table.add_column("Type", style="bold", width=6)
        table.add_column("Path", style="cyan")

        for d in dirs:
            table.add_row("dir", d)
        for f in files:
            table.add_row("file", f)

        console.print(table)
        console.print(f"[dim]Mode: {mode} | Estimated size: {size:,} bytes[/dim]")
        console.print()
    except ImportError:
        print(f"Setup plan (mode={mode}):")
        for d in dirs:
            print(f"  [dir]  {d}")
        for f in files:
            print(f"  [file] {f}")
        print(f"  Estimated size: {size:,} bytes")
        print()


def _print_banner() -> None:
    """Print the welcome banner."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        banner = Text()
        banner.append("LLM Forge", style="bold cyan")
        banner.append(" - Build your own AI model\n", style="dim")
        banner.append("Just tell me what you want to build.\n\n", style="")
        banner.append("Type ", style="dim")
        banner.append("/help", style="bold yellow")
        banner.append(" for commands, ", style="dim")
        banner.append("quit", style="bold red")
        banner.append(" to exit.", style="dim")

        console.print(Panel(banner, border_style="cyan", padding=(1, 2)))
        console.print()
    except ImportError:
        print("=" * 50)
        print("  LLM Forge - Build your own AI model")
        print("  Just tell me what you want to build.")
        print("  Type '/help' for commands, 'quit' to exit.")
        print("=" * 50)
        print()


def _print_response(text: str) -> None:
    """Print the assistant's response with Rich formatting."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()
        md = Markdown(text)
        console.print(
            Panel(md, border_style="green", title="Forge", title_align="left", padding=(0, 1))
        )
        console.print()
    except ImportError:
        print(f"\nForge: {text}\n")


def _check_esc_pressed() -> bool:
    """Non-blocking check if Esc key was pressed."""
    old_settings = None
    try:
        import select
        import sys
        import termios
        import tty

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            # Non-blocking check: is there input ready?
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":  # Esc key
                    return True
        finally:
            if old_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    except Exception:
        pass
    return False


def _stream_response(engine: ChatEngine, user_input: str) -> None:
    """Stream the assistant's response with Esc interrupt support."""
    interrupted = False
    text_chunks: list[str] = []

    def on_text(chunk: str) -> None:
        """Called for each streamed text chunk."""
        text_chunks.append(chunk)
        try:
            from rich.console import Console

            Console().print(chunk, end="", highlight=False)
        except ImportError:
            print(chunk, end="", flush=True)

    def interrupt_check() -> bool:
        nonlocal interrupted
        if interrupted:
            return True
        if _check_esc_pressed():
            interrupted = True
            return True
        return False

    # Print response header
    try:
        from rich.console import Console

        console = Console()
        console.print("[bold green]Forge:[/bold green] ", end="")
    except ImportError:
        print("Forge: ", end="")

    # Stream the response
    engine.send(user_input, on_text=on_text, interrupt_check=interrupt_check)

    # End the line
    print()

    if interrupted:
        try:
            from rich.console import Console

            Console().print("\n[dim][Esc pressed — type your next instruction][/dim]")
        except ImportError:
            print("\n[Esc pressed — type your next instruction]")

    print()


def _get_input() -> str:
    """Get user input with a styled prompt."""
    try:
        from rich.console import Console

        console = Console()
        console.print("[bold cyan]You:[/bold cyan] ", end="")
        return input()
    except ImportError:
        return input("You: ")


def _persist_api_key(key: str) -> None:
    """Write a validated API key to disk for future sessions."""
    from pathlib import Path

    llmforge_dir = Path(".llmforge")
    key_file = llmforge_dir / ".api_key"
    llmforge_dir.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key)
    key_file.chmod(0o600)  # Read/write only for owner

    try:
        from rich.console import Console

        Console().print(
            "[green]API key verified and saved! You won't need to enter it again.[/green]\n"
        )
    except ImportError:
        print("API key verified and saved! You won't need to enter it again.\n")


def _setup_api_key(engine: ChatEngine, provider: str | None) -> tuple[ChatEngine, str | None]:
    """Handle API key setup: load saved key, ask user, or fall back to wizard.

    Returns ``(engine, pending_key)`` where *pending_key* is non-None only
    when the user just pasted a brand-new key that has NOT yet been persisted
    to disk.  The caller should persist it after the key is proven valid
    (e.g. after a successful greeting).
    """
    from pathlib import Path

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
        from rich.console import Console

        console = Console()
        console.print(
            "\n[bold cyan]LLM Forge needs an API key for the conversational experience.[/bold cyan]\n"
        )
        console.print(
            "  Get a free Claude API key at: [link]https://console.anthropic.com/[/link]\n"
        )
        console.print("[bold]Paste your API key[/bold] (or press Enter to skip): ", end="")
        user_key = input().strip()
    except (ImportError, EOFError):
        print("\nLLM Forge needs an API key for the conversational experience.")
        print("Get one at: https://console.anthropic.com/")
        user_key = input("Paste your API key (or press Enter to skip): ").strip()

    if user_key:
        # Set the key in the environment but do NOT write to disk yet —
        # it will be persisted after the first successful API call.
        os.environ["ANTHROPIC_API_KEY"] = user_key

        try:
            from rich.console import Console

            Console().print("[dim]Key accepted. Validating with the API...[/dim]\n")
        except ImportError:
            print("Key accepted. Validating with the API...\n")

        return ChatEngine(provider="anthropic", project_dir="."), user_key
    else:
        # User skipped — offer free wizard
        try:
            from rich.console import Console

            Console().print(
                "\n[dim]No API key? No problem. Launching the free guided wizard instead.[/dim]\n"
            )
        except ImportError:
            print("\nNo API key? No problem. Launching the free guided wizard instead.\n")

        from llm_forge.chat.wizard_fallback import launch_wizard_fallback

        launch_wizard_fallback()
        import sys

        sys.exit(0)


def launch_chat(provider: str | None = None) -> None:
    """Launch the interactive chat session with memory."""
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
            from rich.console import Console

            Console().print("[bold]Set up LLM Forge here?[/bold] [dim](Y/n)[/dim] ", end="")
            answer = input().strip().lower()
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
    try:
        from rich.console import Console

        console = Console()
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
            console.print(
                f"[dim]Memory loaded: {session_count} past session(s), "
                f"{memory_count} stored insight(s)[/dim]\n"
            )
    except ImportError:
        pass

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
                from rich.console import Console

                Console().print("\n[bold]Install it now?[/bold] [dim](Y/n)[/dim] ", end="")
                answer = input().strip().lower()
            except (ImportError, EOFError):
                answer = input("Install it now? (Y/n) ").strip().lower()

            if answer in ("", "y", "yes"):
                import subprocess
                import sys

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

    # The greeting succeeded, so the API key is valid — persist it now.
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
            # Ctrl+C also interrupts — treat same as Esc
            try:
                from rich.console import Console

                Console().print("\n[dim][interrupted — type your new instruction][/dim]\n")
            except ImportError:
                print("\n[interrupted — type your new instruction]\n")
            continue
        except Exception as e:
            _print_error(f"{e}")
            _print_info("Something went wrong. You can keep chatting or type 'quit' to exit.")
            continue


def _shutdown(engine: ChatEngine) -> None:
    """Graceful shutdown: save session memory."""
    try:
        engine.end_session()
    except Exception:
        pass

    try:
        from rich.console import Console

        Console().print("\n[cyan]Session saved. See you next time![/cyan]\n")
    except ImportError:
        print("\nSession saved. See you next time!\n")
