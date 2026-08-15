"""API key resolution for LLM Forge AI services.

Keys are supplied by the user via environment variables or a local
``~/.llm-forge/.env`` file. LLM Forge ships no credentials of its own.

Earlier releases embedded obfuscated project-owned keys here. Those keys are
compromised (obfuscation is not encryption, and the values were published in a
public repository) and have been removed. Nothing in this module reads a
bundled secret any more.
"""

from __future__ import annotations

import os
from pathlib import Path

# Where the installer writes user-supplied keys.
ENV_FILE = Path.home() / ".llm-forge" / ".env"

_ANTHROPIC_CONSOLE = "https://console.anthropic.com/settings/keys"
_GOOGLE_CONSOLE = "https://aistudio.google.com/apikey"


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is not configured."""


def _read_env_file(name: str) -> str:
    """Read ``name`` from ``~/.llm-forge/.env`` if that file exists."""
    try:
        if not ENV_FILE.is_file():
            return ""
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != name:
                continue
            value = value.strip().strip('"').strip("'")
            return value
    except OSError:
        return ""
    return ""


def _resolve(name: str) -> str:
    """Resolve an API key from the environment, then the local env file."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return _read_env_file(name)


def get_anthropic_api_key() -> str:
    """Return the user's Anthropic API key, or an empty string if unset.

    Checks ``ANTHROPIC_API_KEY`` first, then ``~/.llm-forge/.env``.
    """
    return _resolve("ANTHROPIC_API_KEY")


def get_google_api_key() -> str:
    """Return the user's Google API key, or an empty string if unset.

    Checks ``GOOGLE_API_KEY`` first, then ``~/.llm-forge/.env``.
    """
    return _resolve("GOOGLE_API_KEY")


def has_anthropic_api_key() -> bool:
    """True when an Anthropic key is configured."""
    return bool(get_anthropic_api_key())


def has_google_api_key() -> bool:
    """True when a Google key is configured."""
    return bool(get_google_api_key())


def require_anthropic_api_key() -> str:
    """Return the Anthropic key or raise with setup instructions."""
    key = get_anthropic_api_key()
    if not key:
        raise MissingAPIKeyError(
            "No Anthropic API key found.\n"
            f"  Create one at {_ANTHROPIC_CONSOLE}, then either:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            f"  or add ANTHROPIC_API_KEY=sk-ant-... to {ENV_FILE}\n"
            "  Or run `llm-forge setup` for the offline guided setup (no key needed)."
        )
    return key


def require_google_api_key() -> str:
    """Return the Google key or raise with setup instructions."""
    key = get_google_api_key()
    if not key:
        raise MissingAPIKeyError(
            "No Google API key found.\n"
            f"  Create one at {_GOOGLE_CONSOLE}, then either:\n"
            "    export GOOGLE_API_KEY=...\n"
            f"  or add GOOGLE_API_KEY=... to {ENV_FILE}"
        )
    return key


def save_key_to_env_file(name: str, value: str) -> Path:
    """Persist ``name=value`` to ``~/.llm-forge/.env`` with 0600 permissions."""
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if ENV_FILE.is_file():
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if raw.strip().startswith(f"{name}="):
                continue
            lines.append(raw)
    lines.append(f"{name}={value}")

    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ENV_FILE.chmod(0o600)
    return ENV_FILE
