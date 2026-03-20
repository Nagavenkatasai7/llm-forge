"""Slash command handling for the LLM Forge chat UI.

When the user types a message starting with ``/``, the command is handled
locally instead of being sent to the Claude API.  Each handler receives the
:class:`ChatEngine` instance and an optional argument string, and returns a
plain-text response to display.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_forge.chat.engine import ChatEngine

# ---------------------------------------------------------------------------
# Sentinel used by /quit and /clear to signal control flow to the caller
# ---------------------------------------------------------------------------

QUIT_SENTINEL = "__QUIT__"
CLEAR_SENTINEL = "__CLEAR__"

# ---------------------------------------------------------------------------
# Individual command handlers
# ---------------------------------------------------------------------------


def _cmd_help(engine: ChatEngine, args: str) -> str:
    """Show all available slash commands."""
    lines = ["Available commands:\n"]
    for cmd, info in sorted(COMMANDS.items()):
        lines.append(f"  {cmd:<12} {info['description']}")
    lines.append("")
    lines.append("Type a command to execute it, or just chat normally.")
    return "\n".join(lines)


def _cmd_status(engine: ChatEngine, args: str) -> str:
    """Show project state (configs, models, data)."""
    state = engine.memory.project_state

    lines = ["Project Status\n"]
    lines.append(f"  Directory:  {state.get('project_dir', '?')}")
    lines.append(f"  Scanned:    {state.get('scanned_at', '?')}")

    # Configs
    configs = state.get("configs", [])
    lines.append(f"\n  Configs ({len(configs)}):")
    if configs:
        for c in configs:
            name = c.get("name", "?")
            model = c.get("model", "?")
            mode = c.get("mode", "?")
            lines.append(f"    - {name}  (model: {model}, mode: {mode})")
    else:
        lines.append("    (none found)")

    # Trained models
    models = state.get("trained_models", [])
    lines.append(f"\n  Trained models ({len(models)}):")
    if models:
        for m in models:
            name = m.get("name", "?")
            status = m.get("status", "?")
            size = m.get("size_mb")
            size_str = f", {size} MB" if size else ""
            lines.append(f"    - {name}  ({status}{size_str})")
    else:
        lines.append("    (none found)")

    # Data sources
    data = state.get("data_sources", [])
    lines.append(f"\n  Data sources ({len(data)}):")
    if data:
        for d in data:
            lines.append(f"    - {d.get('name', '?')}  ({d.get('size_mb', '?')} MB)")
    else:
        lines.append("    (none found)")

    # Active config
    active = state.get("active_config")
    if active:
        lines.append(f"\n  Active config: {active}")

    return "\n".join(lines)


def _cmd_hardware(engine: ChatEngine, args: str) -> str:
    """Show detected hardware."""
    from llm_forge.chat.tools import execute_tool

    raw = execute_tool("detect_hardware", {})
    try:
        info = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return f"Hardware detection result:\n{raw}"

    lines = ["Detected Hardware\n"]
    lines.append(f"  OS:           {info.get('os', '?')} {info.get('os_version', '')}")
    lines.append(f"  CPU:          {info.get('cpu', '?')}")
    lines.append(f"  Python:       {info.get('python_version', '?')}")

    ram = info.get("ram_total_gb", "?")
    avail = info.get("ram_available_gb", "?")
    lines.append(f"  RAM:          {ram} GB total, {avail} GB available")

    gpu_type = info.get("gpu_type", "unknown")
    lines.append(f"  GPU type:     {gpu_type}")

    if gpu_type == "nvidia_cuda":
        for i, gpu in enumerate(info.get("gpus", [])):
            lines.append(
                f"    GPU {i}: {gpu.get('name', '?')} "
                f"({gpu.get('vram_gb', '?')} GB VRAM, "
                f"CC {gpu.get('compute_capability', '?')})"
            )
        lines.append(f"  CUDA:         {info.get('cuda_version', '?')}")
    elif gpu_type == "apple_mps":
        lines.append(f"  GPU name:     {info.get('gpu_name', '?')}")

    rec = info.get("recommendation", {})
    if rec:
        lines.append("\n  Recommendation:")
        for k, v in rec.items():
            lines.append(f"    {k}: {v}")

    return "\n".join(lines)


def _cmd_memory(engine: ChatEngine, args: str) -> str:
    """Show memory stats (sessions, memories stored)."""
    session_count = 0
    memory_count = 0
    try:
        conn = sqlite3.connect(str(engine.memory.db_path))
        session_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE summary IS NOT NULL"
        ).fetchone()[0]
        memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
    except Exception:
        pass

    lines = ["Memory Stats\n"]
    lines.append(f"  Current session:  {engine.memory.session_id}")
    lines.append(f"  Started:          {engine.memory.session_start.isoformat()}")
    lines.append(f"  Past sessions:    {session_count}")
    lines.append(f"  Stored memories:  {memory_count}")
    lines.append(f"  Database:         {engine.memory.db_path}")
    return "\n".join(lines)


def _cmd_clear(engine: ChatEngine, args: str) -> str:
    """Clear conversation history (reset messages, keep memory)."""
    return CLEAR_SENTINEL


def _cmd_config(engine: ChatEngine, args: str) -> str:
    """Show current active config."""
    state = engine.memory.project_state
    active = state.get("active_config")

    if not active:
        configs = state.get("configs", [])
        if configs:
            lines = ["No active config set (config.yaml not found in project root).\n"]
            lines.append(f"Available configs ({len(configs)}):")
            for c in configs:
                lines.append(f"  - {c.get('name', '?')}")
            return "\n".join(lines)
        return "No configs found. Use the chat to create one, or run: llm-forge init"

    lines = [f"Active config: {active}\n"]

    # Try to read it
    config_path = Path(engine.memory.project_dir) / active
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                raw = yaml.safe_load(f)
            if raw and isinstance(raw, dict):
                model = raw.get("model", {})
                training = raw.get("training", {})
                lines.append(f"  Model:    {model.get('name', '?')}")
                lines.append(f"  Mode:     {training.get('mode', '?')}")
                lines.append(f"  Epochs:   {training.get('num_epochs', '?')}")
                lines.append(f"  LR:       {training.get('learning_rate', '?')}")
        except Exception:
            lines.append("  (could not parse config)")

    return "\n".join(lines)


def _cmd_models(engine: ChatEngine, args: str) -> str:
    """List trained models in outputs/."""
    outputs_dir = Path(engine.memory.project_dir) / "outputs"
    if not outputs_dir.exists():
        return "No outputs/ directory found. Train a model first!"

    models = []
    for d in sorted(outputs_dir.iterdir()):
        if not d.is_dir():
            continue

        status = "empty"
        size_str = ""

        safetensors = list(d.rglob("*.safetensors"))
        gguf_files = list(d.rglob("*.gguf"))
        checkpoints = list(d.glob("checkpoint-*"))

        if safetensors:
            status = "complete"
            total_mb = sum(f.stat().st_size for f in safetensors) / (1024 * 1024)
            size_str = f", {total_mb:.1f} MB"
        elif checkpoints:
            status = f"has {len(checkpoints)} checkpoint(s)"
        elif gguf_files:
            status = "GGUF exported"
            total_mb = sum(f.stat().st_size for f in gguf_files) / (1024 * 1024)
            size_str = f", {total_mb:.1f} MB"

        gguf_names = [f.name for f in gguf_files] if gguf_files else []
        entry = f"  - {d.name}  ({status}{size_str})"
        if gguf_names:
            entry += f"\n    GGUF: {', '.join(gguf_names)}"
        models.append(entry)

    if not models:
        return "outputs/ directory exists but no models found."

    lines = [f"Trained Models ({len(models)}):\n"]
    lines.extend(models)
    return "\n".join(lines)


def _cmd_auto(engine: ChatEngine, args: str) -> str:
    """Toggle auto-approve mode for permissions."""
    current = engine.permissions.auto_approve
    engine.permissions.auto_approve = not current
    new_state = "ON" if engine.permissions.auto_approve else "OFF"
    return f"Auto-approve mode: {new_state}"


def _cmd_quit(engine: ChatEngine, args: str) -> str:
    """Exit the chat session."""
    return QUIT_SENTINEL


def _cmd_version(engine: ChatEngine, args: str) -> str:
    """Show llm-forge version."""
    from llm_forge import __version__

    return f"llm-forge v{__version__}"


def _cmd_model(engine: ChatEngine, args: str) -> str:
    """Show current model or switch to a different one."""
    args = args.strip().lower()

    # NVIDIA provider — show NVIDIA models
    if engine.provider == "nvidia":
        from llm_forge.chat.nvidia_provider import NVIDIA_MODELS

        if not args:
            current = NVIDIA_MODELS.get(engine.model_key, {})
            lines = [
                f"Current model: {current.get('name', engine.model_key)}",
                f"  ID: {current.get('id', '?')}",
                f"  Params: {current.get('params', '?')}",
                "  Provider: NVIDIA NIM (free)",
                "",
                "Available models:",
            ]
            # Group by category
            categories: dict[str, list] = {}
            for key, info in NVIDIA_MODELS.items():
                cat = info["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append((key, info))

            for cat, models in categories.items():
                lines.append(f"\n  [{cat.upper()}]")
                for key, info in models:
                    marker = " <-- active" if key == engine.model_key else ""
                    lines.append(f"    {key:<22} {info['name']:<22} {info['params']}{marker}")
            lines.append("")
            lines.append("Switch with: /model <key>  (e.g. /model llama-3.1-8b)")
            return "\n".join(lines)

        if args in NVIDIA_MODELS:
            engine.model_key = args
            info = NVIDIA_MODELS[args]
            return f"Switched to {info['name']} ({info['params']})"
        return f"Unknown model: {args}\nAvailable: {', '.join(NVIDIA_MODELS.keys())}"

    # Anthropic / default — show Claude models
    from llm_forge.chat.engine import CLAUDE_MODELS

    # No argument — show current model and available options
    if not args:
        current = CLAUDE_MODELS.get(engine.model_key, {})
        lines = [
            f"Current model: {current.get('name', engine.model_key)}",
            f"  ID: {current.get('id', '?')}",
            f"  Context: {current.get('context', '?')}",
            f"  Cost: {current.get('cost', '?')} per 1M tokens",
            "",
            "Available models:",
        ]
        for key, info in CLAUDE_MODELS.items():
            marker = " <-- active" if key == engine.model_key else ""
            lines.append(
                f"  {key:<14} {info['name']:<22} {info['context']:<18} {info['cost']}{marker}"
            )
        lines.append("")
        lines.append("Switch with: /model <key>  (e.g. /model opus-4.6)")
        return "\n".join(lines)

    # Shortcuts: allow partial matches
    shortcuts = {
        "opus": "opus-4.6",
        "sonnet": "sonnet-4.6",
        "haiku": "haiku-4.5",
        "opus4.6": "opus-4.6",
        "sonnet4.6": "sonnet-4.6",
        "haiku4.5": "haiku-4.5",
        "opus-4.5": "opus-4.5",
        "sonnet-4.5": "sonnet-4.5",
        "opus4.5": "opus-4.5",
        "sonnet4.5": "sonnet-4.5",
    }
    resolved = shortcuts.get(args, args)

    if resolved not in CLAUDE_MODELS:
        return f"Unknown model: {args}\nAvailable: {', '.join(CLAUDE_MODELS.keys())}"

    engine.model_key = resolved
    info = CLAUDE_MODELS[resolved]
    return f"Switched to {info['name']} ({info['context']}, {info['cost']} per 1M tokens)"


def _cmd_test(engine: ChatEngine, args: str) -> str:
    """Quick-test a model with a question."""
    if not args:
        return (
            "Usage: /test <question>\nTests the current model. Use /model to switch models first."
        )

    from openai import OpenAI

    from llm_forge.chat.nvidia_provider import (
        NVIDIA_BASE_URL,
        NVIDIA_MODELS,
        get_nvidia_api_key,
    )

    # Get current model
    model_info = NVIDIA_MODELS.get(engine.model_key)
    if model_info is None:
        return "No NVIDIA model selected. Use /model to select one."

    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=get_nvidia_api_key())
        response = client.chat.completions.create(
            model=model_info["id"],
            messages=[{"role": "user", "content": args}],
            max_tokens=300,
            temperature=0.1,
        )
        answer = response.choices[0].message.content or "(no response)"
        return f"Model: {model_info['name']}\nQ: {args}\nA: {answer}"
    except Exception as e:
        return f"Error testing model: {e}"


def _cmd_paste(engine: ChatEngine, args: str) -> str:
    """Enter multi-line paste mode (type --- on a new line to submit)."""
    # Return a special sentinel so the UI loop can handle interactive input.
    return "__PASTE__"


PASTE_SENTINEL = "__PASTE__"

# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

COMMANDS: dict[str, dict] = {
    "/help": {"description": "Show all available commands", "handler": _cmd_help},
    "/status": {
        "description": "Show project state (configs, models, data)",
        "handler": _cmd_status,
    },
    "/hardware": {"description": "Show detected hardware", "handler": _cmd_hardware},
    "/memory": {
        "description": "Show memory stats (sessions, memories stored)",
        "handler": _cmd_memory,
    },
    "/clear": {
        "description": "Clear conversation history (keep memory)",
        "handler": _cmd_clear,
    },
    "/config": {"description": "Show current active config", "handler": _cmd_config},
    "/models": {"description": "List trained models in outputs/", "handler": _cmd_models},
    "/auto": {"description": "Toggle auto-approve mode for permissions", "handler": _cmd_auto},
    "/quit": {"description": "Exit the chat session", "handler": _cmd_quit},
    "/version": {"description": "Show llm-forge version", "handler": _cmd_version},
    "/model": {
        "description": "Show or switch model (e.g. /model opus-4.6 or /model llama-3.1-8b)",
        "handler": _cmd_model,
    },
    "/paste": {
        "description": "Enter multi-line paste mode (--- to submit)",
        "handler": _cmd_paste,
    },
    "/test": {
        "description": "Quick-test current model (e.g. /test What is machine learning?)",
        "handler": _cmd_test,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def handle_slash_command(command: str, engine: ChatEngine) -> str | None:
    """Handle a slash command.

    Returns the output text to display, or ``None`` if *command* does not
    start with ``/`` (i.e. it is regular chat input).

    Special return values:
    - :data:`QUIT_SENTINEL` — caller should exit the main loop.
    - :data:`CLEAR_SENTINEL` — caller should clear messages and continue.
    """
    if not command.startswith("/"):
        return None

    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # Bare "/" shows help
    if cmd == "/":
        return _cmd_help(engine, args)

    if cmd not in COMMANDS:
        return f"Unknown command: {cmd}. Type /help for available commands."

    handler = COMMANDS[cmd]["handler"]
    return handler(engine, args)
