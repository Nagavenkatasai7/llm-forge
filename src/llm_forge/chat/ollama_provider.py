"""Ollama Cloud provider for LLM Forge.

Ollama Cloud exposes an OpenAI-compatible endpoint, so the engine's existing
OpenAI request/response plumbing is reused rather than duplicated. What this
module adds is the parts that are Ollama-specific: endpoint and auth, a *live*
model list, and error handling for the failure modes this API actually has.

The model list is fetched from the API rather than hardcoded. A hardcoded
catalogue goes stale the moment Ollama adds a model, and the point of using
Ollama Cloud is picking whichever model you want.

**Tool calling is required.** LLM Forge's assistant is entirely tool-driven --
it detects hardware, scans data, writes configs, and launches training through
tools. A model that cannot call them can hold a conversation but cannot do the
job, so :func:`supports_tools` is checked before a model is accepted.

No credentials are embedded here. The key comes from ``OLLAMA_API_KEY`` or
``~/.llm-forge/.env``.
"""

from __future__ import annotations

import json
import os
from typing import Any

OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/v1"

OLLAMA_SIGNUP_URL = "https://ollama.com/settings/keys"

# Ollama Cloud bills some models against a separate paid balance. Hitting one
# with an empty balance returns 402 rather than anything auth-shaped, so it is
# worth naming distinctly instead of reporting it as a generic failure.
_PAYMENT_REQUIRED = 402


class OllamaError(RuntimeError):
    """An Ollama Cloud call failed, with a message safe to show the user."""


def get_ollama_api_key() -> str:
    """Return the Ollama API key from the environment or the local env file."""
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if key:
        return key

    # Reuse the same env-file convention as the Anthropic key.
    from llm_forge.chat.api_keys import _read_env_file

    return _read_env_file("OLLAMA_API_KEY")


def has_ollama_api_key() -> bool:
    """True when an Ollama Cloud key is configured."""
    return bool(get_ollama_api_key())


def is_local_ollama_running() -> bool:
    """True when a local ``ollama serve`` is reachable.

    Local Ollama needs no API key, so it is a valid provider even with no
    credentials configured at all.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _client(base_url: str | None = None, api_key: str | None = None):
    """Build an OpenAI-SDK client pointed at Ollama."""
    from openai import OpenAI

    if base_url is None:
        base_url = OLLAMA_CLOUD_BASE_URL if has_ollama_api_key() else OLLAMA_LOCAL_BASE_URL

    if api_key is None:
        # Local Ollama ignores the key but the SDK requires a non-empty string.
        api_key = get_ollama_api_key() or "ollama-local"

    return OpenAI(base_url=base_url, api_key=api_key)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def list_models(client: Any = None) -> list[str]:
    """Return every model this key can reach, newest-style names included.

    Fetched live so a model added by Ollama after this release is still
    selectable.
    """
    if client is None:
        client = _client()
    try:
        response = client.models.list()
    except Exception as exc:
        raise OllamaError(_explain(exc)) from exc

    return sorted(m.id for m in response.data)


def supports_tools(model: str, client: Any = None) -> tuple[bool, str]:
    """Probe whether ``model`` can actually call tools.

    Returns ``(ok, reason)``. Ollama's model list does not advertise tool
    support, and a model that silently answers in prose instead of calling a
    tool would make the assistant look broken in a way that is hard to
    attribute -- so this asks the model to make one trivial call and checks.
    """
    if client is None:
        client = _client()

    probe_tool = [
        {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Return the string pong. Call this when asked to ping.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Call the ping tool."}],
            tools=probe_tool,
            max_tokens=200,
        )
    except Exception as exc:
        return False, _explain(exc)

    message = response.choices[0].message
    if getattr(message, "tool_calls", None):
        return True, "tool calling verified"
    return False, (
        f"{model} replied with text instead of calling the tool. LLM Forge "
        "drives everything through tools, so this model cannot run the "
        "assistant. Pick another with /model."
    )


def _explain(exc: Exception) -> str:
    """Turn an SDK exception into something a user can act on."""
    status = getattr(exc, "status_code", None)
    body = str(exc)

    if status == _PAYMENT_REQUIRED or "extra usage" in body:
        return (
            "This model bills against Ollama's paid extra-usage balance, which "
            "is empty. Top up at https://ollama.com/settings/billing, or pick a "
            "model included in your plan with /model."
        )
    if status in (401, 403):
        return (
            "Ollama rejected the API key. Check OLLAMA_API_KEY, or create a new "
            f"key at {OLLAMA_SIGNUP_URL}."
        )
    if status == 404:
        return "That model name was not found. Run /model to list what your key can reach."
    if status == 429:
        return "Ollama is rate-limiting this key. Wait a moment and retry."
    return f"Ollama request failed: {body}"


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------


def to_openai_messages(messages: list[dict], system: str) -> list[dict]:
    """Convert the engine's message log into OpenAI chat format.

    The engine stores a hybrid: plain strings, Anthropic-style ``tool_result``
    blocks, and OpenAI-style assistant turns with ``tool_calls`` (appended by
    ``_handle_openai_tools``). All three shapes have to survive the round trip
    or the tool loop breaks on the second iteration.
    """
    out: list[dict] = [{"role": "system", "content": system}]

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Anthropic-style tool results become individual tool messages.
                text_parts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(block.get("content", "")),
                            }
                        )
                    elif block.get("type") == "text":
                        text_parts.append(str(block.get("text", "")))
                if text_parts:
                    out.append({"role": "user", "content": "\n".join(text_parts)})

        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": content or ""}
            if msg.get("tool_calls"):
                entry["tool_calls"] = msg["tool_calls"]
                # OpenAI rejects a null content alongside tool_calls.
                entry["content"] = content or ""
            out.append(entry)

        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": str(msg.get("content", "")),
                }
            )

    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert LLM Forge's Anthropic-style tool schemas to OpenAI functions."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def call_ollama(
    messages: list[dict],
    system: str,
    model: str,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    client: Any = None,
):
    """Non-streaming completion with tool use."""
    from llm_forge.chat.tools import TOOLS

    if client is None:
        client = _client()
    if tools is None:
        tools = TOOLS

    try:
        return client.chat.completions.create(
            model=model,
            messages=to_openai_messages(messages, system),
            tools=to_openai_tools(tools),
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise OllamaError(_explain(exc)) from exc


def stream_ollama(
    messages: list[dict],
    system: str,
    model: str,
    tools: list[dict] | None = None,
    on_text=None,
    interrupt_check=None,
    max_tokens: int = 4096,
    client: Any = None,
):
    """Streaming completion, reassembled into a non-streaming response shape.

    The engine's tool loop expects a `.choices[0].message` object with
    `.content` and `.tool_calls`, so the deltas are accumulated back into that
    shape. Tool-call arguments arrive fragmented across chunks and must be
    concatenated per index before they parse as JSON.
    """
    from types import SimpleNamespace

    from llm_forge.chat.tools import TOOLS

    if client is None:
        client = _client()
    if tools is None:
        tools = TOOLS

    collected: list[str] = []
    # Accumulated tool calls, in arrival order.
    #
    # Keyed on index *and* id rather than index alone. Some models emit several
    # sequential tool calls all reporting index 0; keying on index alone
    # appends the second call's arguments onto the first, producing
    # `{...}{...}` in one string. That is not recoverable downstream -- it
    # surfaces as `json.JSONDecodeError: Extra data` at exactly the first
    # object's final character.
    partial_calls: list[dict[str, Any]] = []

    def slot_for(index: int, call_id: str | None) -> dict[str, Any]:
        """Find the accumulator for this fragment, or start a new one."""
        for slot in reversed(partial_calls):
            if slot["index"] != index:
                continue
            # A new id at a known index means a *new* call, not a continuation.
            if call_id and slot["id"] and call_id != slot["id"]:
                break
            return slot
        fresh: dict[str, Any] = {"index": index, "id": "", "name": "", "arguments": ""}
        partial_calls.append(fresh)
        return fresh

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=to_openai_messages(messages, system),
            tools=to_openai_tools(tools),
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if interrupt_check and interrupt_check():
                break
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if getattr(delta, "content", None):
                collected.append(delta.content)
                if on_text:
                    on_text(delta.content)

            for tc in getattr(delta, "tool_calls", None) or []:
                index = getattr(tc, "index", 0) or 0
                slot = slot_for(index, getattr(tc, "id", None))
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    # A second name at the same index is another call, not a
                    # rename -- models that reuse index 0 send the name again.
                    if slot["name"] and tc.function.name != slot["name"]:
                        slot = slot_for(index, tc.id)
                        slot["id"] = tc.id or ""
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
    except Exception as exc:
        raise OllamaError(_explain(exc)) from exc

    rebuilt = [
        SimpleNamespace(
            id=call["id"] or f"call_{position}",
            type="function",
            function=SimpleNamespace(
                name=call["name"], arguments=call["arguments"] or "{}"
            ),
        )
        for position, call in enumerate(partial_calls)
        if call["name"]
    ]

    message = SimpleNamespace(content="".join(collected), tool_calls=rebuilt or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def describe_models() -> str:
    """Human-readable model list for the /model slash command."""
    try:
        models = list_models()
    except OllamaError as exc:
        return str(exc)

    if not models:
        return "No models available to this key."

    lines = [f"Ollama Cloud models ({len(models)} available):", ""]
    lines.extend(f"  {m}" for m in models)
    lines.append("")
    lines.append("Switch with: /model <name>   (e.g. /model qwen3.5:397b)")
    lines.append(
        "Some models bill against a separate paid balance and will say so if unavailable."
    )
    return "\n".join(lines)


def default_model(client: Any = None) -> str | None:
    """Pick a sensible starting model from whatever the key can reach.

    Preference order favours models known to handle tool-driven agent work;
    falls back to whatever is available so a new Ollama model still works.
    """
    try:
        models = list_models(client=client)
    except OllamaError:
        return None

    if not models:
        return None

    preferred = ("qwen3.5", "deepseek-v4-pro", "glm-5", "gpt-oss:120b", "kimi-k2")
    for prefix in preferred:
        for model in models:
            if model.startswith(prefix):
                return model
    return models[0]


def save_api_key(key: str) -> str:
    """Persist an Ollama key to ``~/.llm-forge/.env`` (0600) and return the path."""
    from llm_forge.chat.api_keys import save_key_to_env_file

    return str(save_key_to_env_file("OLLAMA_API_KEY", key))


__all__ = [
    "OLLAMA_CLOUD_BASE_URL",
    "OLLAMA_LOCAL_BASE_URL",
    "OllamaError",
    "call_ollama",
    "default_model",
    "describe_models",
    "get_ollama_api_key",
    "has_ollama_api_key",
    "is_local_ollama_running",
    "list_models",
    "save_api_key",
    "stream_ollama",
    "supports_tools",
    "to_openai_messages",
    "to_openai_tools",
]
