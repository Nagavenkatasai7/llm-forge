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
    "detect_hardware": "Detect hardware",
    "scan_data": "Scan data",
    "write_config": "Write config",
    "validate_config": "Validate config",
    "read_file": "Read file",
    "write_file": "Write file",
    "run_command": "Run",
    "start_training": "Start training",
    "check_training_status": "Check training status",
    "export_model": "Export model",
    "deploy_to_ollama": "Deploy to Ollama",
    "deploy_to_huggingface": "Deploy to HuggingFace",
    "search_huggingface": "Search HuggingFace",
    "download_model": "Download model",
    "save_memory": "Save memory",
    "recall_memory": "Recall memory",
    "get_project_state": "Load project state",
    "get_session_history": "Load session history",
    "log_training_run": "Log training run",
    "install_package": "Install package",
    "install_dependencies": "Install dependencies",
    "fetch_url": "Fetch URL",
    "convert_document": "Convert document",
    "list_configs": "List configs",
    "run_evaluation": "Run evaluation",
    "read_training_logs": "Read training logs",
    "show_model_info": "Show model info",
    "estimate_training": "Estimate training",
    "detect_project": "Detect project type",
    "setup_project": "Set up project",
}


def _format_tool_detail(tool_name: str, input_data: dict | None) -> str:
    """Build a descriptive one-liner showing what tool is being called and with what input."""
    if input_data is None:
        return _TOOL_LABELS.get(tool_name, tool_name)

    if tool_name == "read_file":
        return f"Read file: {input_data.get('path', '?')}"
    elif tool_name == "write_file":
        return f"Write file: {input_data.get('path', '?')}"
    elif tool_name == "run_command":
        cmd = input_data.get("command", "?")
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"Run: {cmd}"
    elif tool_name == "scan_data":
        return f"Scan data: {input_data.get('path', '?')}"
    elif tool_name == "write_config":
        return f"Write config: {input_data.get('output_path', '?')}"
    elif tool_name == "detect_hardware":
        return "Detect hardware"
    elif tool_name == "save_memory":
        cat = input_data.get("category", "")
        content = input_data.get("content", "")[:50]
        return f"Save memory [{cat}]: {content}"
    elif tool_name == "recall_memory":
        return f"Recall memory: {input_data.get('query', '?')}"
    elif tool_name == "install_package":
        return f"Install: {input_data.get('package_name', '?')}"
    elif tool_name == "fetch_url":
        url = input_data.get("url", "?")
        if len(url) > 50:
            url = url[:47] + "..."
        return f"Fetch: {url}"
    elif tool_name == "convert_document":
        return f"Convert: {input_data.get('input_path', '?')}"
    elif tool_name == "start_training":
        return f"Start training: {input_data.get('config_path', '?')}"
    elif tool_name == "search_huggingface":
        return f"Search HuggingFace: {input_data.get('query', '?')}"
    elif tool_name == "estimate_training":
        return f"Estimate: {input_data.get('model_name', '?')} ({input_data.get('mode', '?')})"
    elif tool_name == "deploy_to_ollama":
        return f"Deploy to Ollama: {input_data.get('model_name', '?')}"
    elif tool_name == "validate_config":
        return f"Validate: {input_data.get('config_path', '?')}"
    elif tool_name == "get_project_state":
        return "Load project state"
    elif tool_name == "get_session_history":
        return "Load session history"
    elif tool_name == "log_training_run":
        model = input_data.get("model_name", "?")
        mode = input_data.get("mode", "?")
        return f"Log training run: {model} ({mode})"
    else:
        return _TOOL_LABELS.get(tool_name, tool_name)


def _summarize_tool_result(tool_name: str, result_json: str) -> str | None:
    """Extract a brief one-line summary from a tool result JSON string."""
    import json as _json

    try:
        data = _json.loads(result_json)
    except (_json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    if tool_name == "detect_hardware":
        gpu = data.get("gpu_type", "no GPU")
        ram = data.get("ram_total_gb", "?")
        rec = data.get("recommendation", {})
        mode = rec.get("mode", "?") if isinstance(rec, dict) else "?"
        return f"{gpu}, {ram} GB RAM -> {mode} recommended"
    elif tool_name == "scan_data":
        count = data.get("sample_count", "?")
        fmt = data.get("detected_format", "?")
        return f"{count} samples, {fmt} format"
    elif tool_name == "write_config":
        return data.get("message", "Config saved")
    elif tool_name == "read_file":
        lines = data.get("line_count", "?")
        size = data.get("size_bytes", 0)
        return f"{lines} lines, {size:,} bytes"
    elif tool_name == "run_command":
        rc = data.get("return_code", "?")
        stdout = (data.get("stdout", "") or "")[:80]
        if rc == 0 and stdout:
            return stdout.strip().split("\n")[0]
        elif rc != 0:
            return f"Exit code {rc}"
        return None
    elif tool_name == "write_file":
        return data.get("status", None)
    elif tool_name == "install_package":
        return data.get("message", None)
    elif tool_name == "estimate_training":
        fits = data.get("fits_in_memory", True)
        time_min = data.get("estimated_time_minutes", "?")
        vram = data.get("estimated_vram_gb", "?")
        if not fits:
            return f"WARNING: needs {vram} GB VRAM, won't fit!"
        return f"~{time_min} min, {vram} GB VRAM"
    elif tool_name in ("save_memory", "recall_memory", "log_training_run"):
        status = data.get("status", "")
        count = data.get("count", "")
        if count:
            return f"{count} result(s)"
        return status or None
    elif tool_name == "get_project_state":
        proj = data.get("project_dir", "")
        return f"project: {proj}" if proj else None
    elif tool_name == "get_session_history":
        sessions = data.get("sessions", [])
        return f"{len(sessions)} past session(s)"
    return None


def _print_tool_action(
    tool_name: str,
    input_data: dict | None = None,
    description: str = "",
) -> None:
    """Show a tool execution indicator line with a vertical bar for grouping.

    Displays a dim, indented label with a pipe prefix showing what tool
    is being called and with what arguments, giving a pipeline look.
    """
    console = _get_console()
    if description:
        label = description
    else:
        label = _format_tool_detail(tool_name, input_data)
    if console is not None:
        console.print(f"  [dim]{label}[/dim]")
    else:
        print(f"  {label}")


def _print_tool_result_summary(tool_name: str, result_json: str) -> None:
    """Show a brief, indented result summary beneath a tool action line."""
    summary = _summarize_tool_result(tool_name, result_json)
    if not summary:
        return
    console = _get_console()
    if console is not None:
        console.print(f"    [dim]{summary}[/dim]")
    else:
        print(f"    {summary}")


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


def _print_model_info(engine: ChatEngine) -> None:
    """Display the active model and provider as an info line below the banner."""
    if engine.provider == "nvidia":
        from llm_forge.chat.nvidia_provider import DEFAULT_NVIDIA_MODEL, NVIDIA_MODELS

        info = NVIDIA_MODELS.get(engine.model_key or DEFAULT_NVIDIA_MODEL, {})
        name = info.get("name", engine.model_key or "?")
        params = info.get("params", "")
        label = f"Model: {name}"
        if params:
            label += f" ({params})"
        label += " -- free via NVIDIA NIM"
        _print_info(label)
        _print_info("Type /model to switch models, /help for all commands")
    elif engine.provider == "anthropic":
        from llm_forge.chat.engine import CLAUDE_MODELS, DEFAULT_MODEL

        info = CLAUDE_MODELS.get(engine.model_key or DEFAULT_MODEL, {})
        name = info.get("name", engine.model_key or "?")
        _print_info(f"Model: {name}")
        _print_info("Type /model to switch models, /help for all commands")
    else:
        _print_info("Type /help for all commands")


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


def _print_banner(model_label: str = "") -> None:
    """Print a minimal welcome — like Claude Code, no heavy boxes."""
    console = _get_console()
    version = _get_version()

    if console is not None:
        console.print()
        console.print(f"[bold cyan]LLM Forge[/bold cyan] [dim]v{version}[/dim]")
        if model_label:
            console.print(f"[dim]{model_label}[/dim]")
        console.print("[dim]Type / for commands, Esc to interrupt, quit to exit[/dim]")
        console.print()
    else:
        print()
        print(f"LLM Forge v{version}")
        if model_label:
            print(model_label)
        print("Type / for commands, Esc to interrupt, quit to exit")
        print()


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------


def _print_response(text: str) -> None:
    """Print assistant response — clean, no decoration, just markdown."""
    console = _get_console()
    if console is not None:
        try:
            from rich.markdown import Markdown

            console.print()
            console.print(Markdown(text))
            console.print()
            return
        except ImportError:
            pass

    print(f"\n{text}\n")


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
    """Stream the response — clean, natural, like Claude Code."""
    console = _get_console()
    interrupted = False
    text_chunks: list[str] = []
    first_token_received = False

    def on_text(chunk: str) -> None:
        nonlocal first_token_received
        text_chunks.append(chunk)
        if not first_token_received:
            first_token_received = True
            # Just a blank line before the response starts
            if console is not None:
                console.print()
            else:
                print()
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

    # Wire tool-action display
    prev_start = engine.on_tool_start
    prev_end = engine.on_tool_end

    def _on_tool_start(name: str, input_data: dict) -> None:
        _print_tool_action(name, input_data=input_data)

    def _on_tool_end(name: str, _input_data: dict, result: str) -> None:
        _print_tool_result_summary(name, result)

    engine.on_tool_start = _on_tool_start
    engine.on_tool_end = _on_tool_end

    try:
        engine.send(user_input, on_text=on_text, interrupt_check=interrupt_check)
    finally:
        engine.on_tool_start = prev_start
        engine.on_tool_end = prev_end

    # -- Footer: consistent spacing --
    if console is not None:
        console.print()  # newline after last streamed chunk
        console.print()
    else:
        print()
        print()

    if interrupted:
        _print_info("Interrupted -- type your next instruction")


# ---------------------------------------------------------------------------
# User input (prompt_toolkit with graceful fallback)
# ---------------------------------------------------------------------------

_input_session = None


def _get_input() -> str:
    """Get user input with prompt_toolkit (or fallback to plain input).

    prompt_toolkit provides bracketed-paste support, slash-command
    completion on Tab, input history via Up/Down, and Esc+Enter for
    explicit newlines.
    """
    global _input_session  # noqa: PLW0603
    if _input_session is None:
        from llm_forge.chat.input_handler import create_input_session

        _input_session = create_input_session()

    from llm_forge.chat.input_handler import get_user_input

    return get_user_input(session=_input_session)


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

    # Show current model info after engine is ready
    _print_model_info(engine)

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
    def _greeting_tool_start(name: str, input_data: dict) -> None:
        _print_tool_action(name, input_data=input_data)

    def _greeting_tool_end(name: str, _input_data: dict, result: str) -> None:
        _print_tool_result_summary(name, result)

    engine.on_tool_start = _greeting_tool_start
    engine.on_tool_end = _greeting_tool_end

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
                PASTE_SENTINEL,
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
                if result == PASTE_SENTINEL:
                    _print_info(
                        "Paste mode: paste your content, then type --- on a new line to submit."
                    )
                    lines: list[str] = []
                    while True:
                        try:
                            line = input()
                            if line.strip() == "---":
                                break
                            lines.append(line)
                        except (EOFError, KeyboardInterrupt):
                            break
                    user_input = "\n".join(lines)
                    if not user_input.strip():
                        continue
                    # Fall through to send pasted content to Claude
                else:
                    _print_response(result)
                    continue

        try:
            _stream_response(engine, user_input)
        except KeyboardInterrupt:
            if console is not None:
                console.print()
            else:
                print()
            _print_info("Interrupted -- type your next instruction")
            continue
        except Exception as e:
            _print_error(f"{e}")
            _print_info("Something went wrong. You can keep chatting or type 'quit' to exit.")
            continue
