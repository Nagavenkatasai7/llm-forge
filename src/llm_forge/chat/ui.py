"""Rich terminal UI for the LLM Forge conversational assistant."""

from __future__ import annotations

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
        banner.append("quit", style="bold red")
        banner.append(" to exit.", style="dim")

        console.print(Panel(banner, border_style="cyan", padding=(1, 2)))
        console.print()
    except ImportError:
        print("=" * 50)
        print("  LLM Forge - Build your own AI model")
        print("  Just tell me what you want to build.")
        print("  Type 'quit' to exit.")
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


def _get_input() -> str:
    """Get user input with a styled prompt."""
    try:
        from rich.console import Console

        console = Console()
        console.print("[bold cyan]You:[/bold cyan] ", end="")
        return input()
    except ImportError:
        return input("You: ")


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

    # Check for API key — launch free wizard fallback if none found
    if engine.provider == "none":
        from llm_forge.chat.wizard_fallback import launch_wizard_fallback

        launch_wizard_fallback()
        return

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

    # Send initial greeting — tells Claude to check project state and memories
    try:
        greeting = engine.send(
            "The user just launched llm-forge. Check the project state and session history. "
            "If there's past work, welcome them back with context. If new user, "
            "greet them warmly and ask what kind of AI model they want to build. "
            "Also detect hardware if not already in memory. Keep it to 3-4 sentences."
        )
        _print_response(greeting)
    except Exception as e:
        try:
            from rich.console import Console

            Console().print(f"[red]Error connecting to API:[/red] {e}")
        except ImportError:
            print(f"Error connecting to API: {e}")
        return

    # Main conversation loop
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

        try:
            response = engine.send(user_input)
            _print_response(response)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
            continue
        except Exception as e:
            try:
                from rich.console import Console

                Console().print(f"[red]Error:[/red] {e}")
            except ImportError:
                print(f"Error: {e}")


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
