"""Conversation engine for the LLM Forge CLI assistant.

Uses the Anthropic Claude API with tool use to guide users through
building their own LLMs via natural conversation. Integrates the
three-layer memory system for cross-session intelligence.
"""

from __future__ import annotations

import json
import os

from llm_forge.chat.execution import (
    EXECUTION_TOOL_NAMES,
    PermissionSystem,
    execute_execution_tool,
)
from llm_forge.chat.memory import MemoryManager
from llm_forge.chat.system_prompt import SYSTEM_PROMPT
from llm_forge.chat.tools import TOOLS, execute_tool

# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


def _get_provider() -> str:
    """Detect which LLM provider to use based on available credentials.

    Anthropic first because the assistant's prompts and tool schemas are tuned
    for it, then Ollama (cloud key, or a local server which needs no key at
    all), then OpenAI.
    """
    from llm_forge.chat.api_keys import get_anthropic_api_key

    if get_anthropic_api_key():
        return "anthropic"

    from llm_forge.chat.ollama_provider import has_ollama_api_key, is_local_ollama_running

    if has_ollama_api_key() or is_local_ollama_running():
        return "ollama"

    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    # Nothing configured. The caller surfaces setup instructions.
    return "none"


def _get_anthropic_client():
    """Get a cached Anthropic client instance."""
    import anthropic

    return anthropic.Anthropic()


# Model catalogue lives in claude_models.py so engine and orchestrator cannot drift.
from llm_forge.chat.claude_models import (  # noqa: E402
    ADAPTIVE_THINKING,
    CLAUDE_MODELS,
    DEFAULT_MODEL,
    model_id as _resolve_model_id,
)


def _call_anthropic(
    messages: list[dict], system: str, client=None, model_key: str | None = None
) -> dict:
    """Call Claude API with tool use. Returns the API response."""
    if client is None:
        client = _get_anthropic_client()

    model_id = _resolve_model_id(model_key)

    response = client.messages.create(
        model=model_id,
        max_tokens=16000,
        thinking=ADAPTIVE_THINKING,
        system=system,
        tools=TOOLS,
        messages=messages,
    )
    return response


def _stream_anthropic(
    messages: list[dict],
    system: str,
    client=None,
    on_text=None,
    interrupt_check=None,
    model_key: str | None = None,
):
    """Stream Claude API response. Calls on_text(chunk) for each text chunk.

    If interrupt_check() returns True, stops streaming and returns partial response.
    Returns the final message.
    """
    if client is None:
        client = _get_anthropic_client()

    model_id = _resolve_model_id(model_key)
    collected_text: list[str] = []

    with client.messages.stream(
        model=model_id,
        # Thinking is on by default on Opus 5, and max_tokens caps thinking and
        # response text together -- 4096 truncated answers mid-sentence.
        max_tokens=16000,
        thinking=ADAPTIVE_THINKING,
        system=system,
        tools=TOOLS,
        messages=messages,
    ) as stream:
        for event in stream:
            # Check for interrupt
            if interrupt_check and interrupt_check():
                break

            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    collected_text.append(event.delta.text)
                    if on_text:
                        on_text(event.delta.text)

        try:
            return stream.get_final_message()
        except Exception:
            # Stream was interrupted or incomplete — construct a minimal
            # response object with the text collected so far.
            from types import SimpleNamespace

            text = "".join(collected_text) or "[incomplete response]"
            block = SimpleNamespace(type="text", text=text)
            return SimpleNamespace(
                content=[block],
                stop_reason="end_turn",
            )


def _parse_tool_arguments(raw: str | None) -> list[dict]:
    """Parse a tool call's ``arguments`` into one or more input dicts.

    A bare ``json.loads`` here is a session-ending hazard. Models -- especially
    open-weight ones behind OpenAI-compatible endpoints -- sometimes emit two
    concatenated objects (``{...}{...}``) for what should be two separate tool
    calls, or wrap the payload in a markdown fence. ``json.loads`` raises
    ``Extra data`` on the first case, the exception escapes ``send()``, and the
    whole turn dies with everything done so far discarded.

    Returns a list because concatenated objects genuinely represent multiple
    calls, and running both is what the model meant. An unparseable payload
    yields a marker dict the tool executor turns into an error the model can
    read and retry from, rather than an exception.
    """
    if not raw or not raw.strip():
        return [{}]

    text = raw.strip()

    # Some models wrap arguments in a markdown fence.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.removeprefix("json").strip()
        if text.endswith("```"):
            text = text[: -len("```")].strip()

    decoder = json.JSONDecoder()
    parsed: list[dict] = []
    position = 0
    length = len(text)

    while position < length:
        while position < length and text[position].isspace():
            position += 1
        if position >= length:
            break
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            if parsed:
                # Recovered at least one call; ignore the trailing garbage
                # rather than throwing away good work.
                break
            return [{"__invalid_tool_arguments__": text[:400], "__error__": str(exc)}]
        parsed.append(value if isinstance(value, dict) else {"value": value})

    return parsed or [{}]


def _call_openai(messages: list[dict], system: str) -> dict:
    """Call OpenAI API with tool use (function calling)."""
    from openai import OpenAI

    client = OpenAI()

    functions = []
    for tool in TOOLS:
        functions.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
        )

    oai_messages = [{"role": "system", "content": system}]
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                oai_messages.append({"role": "user", "content": msg["content"]})
            elif isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result":
                        oai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": block["content"],
                            }
                        )
        elif msg["role"] == "assistant":
            oai_messages.append(msg)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=oai_messages,
        tools=functions,
        max_tokens=4096,
    )
    return response


# ---------------------------------------------------------------------------
# Chat Engine with Memory
# ---------------------------------------------------------------------------


class ChatEngine:
    """Manages the conversation loop with integrated memory system."""

    def __init__(
        self,
        provider: str | None = None,
        project_dir: str | None = None,
        model_key: str | None = None,
    ):
        self.provider = provider or _get_provider()
        self.model_key = model_key or DEFAULT_MODEL
        self.messages: list[dict] = []
        self.memory = MemoryManager(project_dir=project_dir or ".")
        self.permissions = PermissionSystem(auto_approve=True)
        self._client = None

        # When provider is nvidia, ensure model_key is a valid NVIDIA model
        if self.provider == "nvidia":
            from llm_forge.chat.nvidia_provider import DEFAULT_NVIDIA_MODEL, NVIDIA_MODELS

            if self.model_key not in NVIDIA_MODELS:
                self.model_key = DEFAULT_NVIDIA_MODEL

        # Ollama model names are free-form and fetched live, so a Claude model
        # key inherited from the default would 404. Resolve one from the API.
        if self.provider == "ollama":
            from llm_forge.chat.ollama_provider import default_model

            if model_key is None or self.model_key in CLAUDE_MODELS:
                self.model_key = default_model() or "gpt-oss:120b"

        # Track recent tool names for context detection (model output vs instruction)
        self._recent_tool_names: list[str] = []

        # Optional UI callbacks: called before/after each tool execution
        # so the UI can display progress indicators.
        self.on_tool_start: callable | None = None
        # Called with (tool_name, input_data, result_json) after execution.
        self.on_tool_end: callable | None = None

        # Build dynamic system prompt with memory context
        self.system = self._build_system_prompt()

        # Wire memory tools into the tool executor
        self._setup_memory_tools()

    def _build_system_prompt(self) -> str:
        """Build system prompt with injected memory context.

        The final prompt is layered:
          1. Core instructions + deep knowledge base (already in SYSTEM_PROMPT)
          2. Dynamic memory context (session history, project state, user prefs)
        Memory context is appended AFTER the knowledge base so that
        session-specific details have highest recency weight.
        """
        context_block = self.memory.build_context_block()
        if context_block.strip():
            return f"{SYSTEM_PROMPT}\n\n---\n\n{context_block}"
        return SYSTEM_PROMPT

    def _setup_memory_tools(self) -> None:
        """Register memory-specific tool implementations."""
        # These are handled in the execute_tool override below
        pass

    def _get_client(self):
        """Get or create the API client."""
        if self._client is None and self.provider == "anthropic":
            self._client = _get_anthropic_client()
        return self._client

    def send(
        self,
        user_input: str,
        on_text=None,
        interrupt_check=None,
    ) -> str:
        """Send a user message and get the assistant's response.

        Handles the full tool-use loop with memory management:
        - Checks for context compaction before each API call
        - Routes memory tools to the MemoryManager
        - Supports streaming via on_text(chunk) callback
        - Supports Esc interruption via interrupt_check() callback
        """
        self.messages.append({"role": "user", "content": user_input})
        self._interrupted = False

        # Check if we need to compact before calling the API
        if self.memory.needs_compaction(self.messages):
            self.messages = self.memory.compact_messages(self.messages, client=self._get_client())

        max_tool_iterations = 15
        tool_iteration = 0

        while True:
            if self.provider == "anthropic":
                if on_text and not interrupt_check:
                    response = _stream_anthropic(
                        self.messages,
                        self.system,
                        client=self._get_client(),
                        on_text=on_text,
                        model_key=self.model_key,
                    )
                elif on_text and interrupt_check:
                    response = _stream_anthropic(
                        self.messages,
                        self.system,
                        client=self._get_client(),
                        on_text=on_text,
                        interrupt_check=interrupt_check,
                        model_key=self.model_key,
                    )
                else:
                    response = _call_anthropic(
                        self.messages,
                        self.system,
                        client=self._get_client(),
                        model_key=self.model_key,
                    )
                text, tool_calls = self._parse_anthropic_response(response)
            elif self.provider == "openai":
                response = _call_openai(self.messages, self.system)
                text, tool_calls = self._parse_openai_response(response)
            elif self.provider == "ollama":
                from llm_forge.chat.ollama_provider import (
                    OllamaError,
                    call_ollama,
                    stream_ollama,
                )

                try:
                    if on_text:
                        response = stream_ollama(
                            self.messages,
                            self.system,
                            model=self.model_key,
                            on_text=on_text,
                            interrupt_check=interrupt_check,
                        )
                    else:
                        response = call_ollama(
                            self.messages, self.system, model=self.model_key
                        )
                except OllamaError as exc:
                    # Already a user-facing explanation (bad key, empty paid
                    # balance, unknown model) -- surface it rather than a stack
                    # trace, and keep the session alive so /model can fix it.
                    return str(exc)

                # Ollama Cloud is OpenAI-compatible, so the same parser applies.
                text, tool_calls = self._parse_openai_response(response)
            elif self.provider == "nvidia":
                from llm_forge.chat.nvidia_provider import call_nvidia, stream_nvidia

                if on_text:
                    response = stream_nvidia(
                        self.messages,
                        self.system,
                        model_key=self.model_key,
                        on_text=on_text,
                        interrupt_check=interrupt_check,
                    )
                else:
                    response = call_nvidia(
                        self.messages,
                        self.system,
                        model_key=self.model_key,
                    )
                # Parse response (OpenAI format)
                text = response.choices[0].message.content or ""
                # NVIDIA models don't support Claude-style tool use;
                # the manager works through natural language instructions instead.
                tool_calls = []
            else:
                return (
                    "No API key found. LLM Forge can use any of:\n\n"
                    "  ANTHROPIC_API_KEY   https://console.anthropic.com/settings/keys\n"
                    "  OLLAMA_API_KEY      https://ollama.com/settings/keys\n"
                    "  OPENAI_API_KEY      https://platform.openai.com/api-keys\n\n"
                    "A local `ollama serve` also works with no key at all.\n"
                    "Or run `llm-forge setup` for the offline wizard."
                )

            # If interrupted, save partial response and return
            if interrupt_check and interrupt_check():
                self._interrupted = True
                partial = text or "[interrupted by user]"
                self.messages.append({"role": "assistant", "content": partial})
                return partial

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": text})
                return text

            # Guard against unbounded tool-call loops
            tool_iteration += 1
            if tool_iteration >= max_tool_iterations:
                partial = text or ""
                if partial:
                    partial += "\n\n"
                partial += (
                    "[Tool-call limit reached: the assistant used tools "
                    f"{max_tool_iterations} times in a row. Returning partial "
                    "response to avoid an infinite loop.]"
                )
                self.messages.append({"role": "assistant", "content": partial})
                return partial

            if self.provider == "anthropic":
                self._handle_anthropic_tools(response, tool_calls)
            elif self.provider in ("openai", "ollama"):
                self._handle_openai_tools(response, tool_calls)

    def _execute_tool(self, name: str, input_data: dict) -> str:
        """Execute a tool, routing memory tools to MemoryManager.

        Execution tools (run_command, read_file, write_file, convert_document,
        install_package, fetch_url) are gated by the PermissionSystem before
        being dispatched.
        """
        # A payload the parser could not read. Hand the model a description of
        # what it sent so it can correct itself, instead of raising -- a
        # malformed argument string is a recoverable model mistake, not a bug
        # in the tool.
        if "__invalid_tool_arguments__" in input_data:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Could not parse the arguments you sent for this tool.",
                    "detail": input_data.get("__error__", ""),
                    "you_sent": input_data.get("__invalid_tool_arguments__", ""),
                    "fix": (
                        "Send exactly one JSON object matching the tool's schema. "
                        "Do not concatenate multiple objects, and do not wrap it "
                        "in a markdown code fence. To make two calls, emit two "
                        "separate tool calls."
                    ),
                }
            )

        # Track tool names for context detection
        self._recent_tool_names.append(name)
        if len(self._recent_tool_names) > 20:
            self._recent_tool_names = self._recent_tool_names[-20:]

        # Notify UI callback (if registered) so it can display a progress line
        if self.on_tool_start is not None:
            try:
                self.on_tool_start(name, input_data)
            except Exception:
                pass  # Never let a UI callback break tool execution

        # Memory-specific tools
        if name == "save_memory":
            result = self.memory.save_memory(
                category=input_data.get("category", "general"),
                content=input_data["content"],
                relevance=input_data.get("relevance", 1.0),
            )
        elif name == "recall_memory":
            result = self.memory.recall_memory(
                query=input_data["query"],
                limit=input_data.get("limit", 10),
            )
        elif name == "get_project_state":
            result = json.dumps(self.memory.project_state, indent=2)
        elif name == "get_session_history":
            result = self.memory.get_session_history(limit=input_data.get("limit", 5))
        elif name == "log_training_run":
            result = self.memory.log_training_run(**input_data)
        elif name in EXECUTION_TOOL_NAMES:
            # Gate execution tools through the permission system
            allowed, reason = self.permissions.check(name, input_data)
            if not allowed:
                result = json.dumps({"status": "blocked", "reason": reason})
            else:
                result = execute_execution_tool(name, input_data)
        else:
            result = execute_tool(name, input_data)

        # Notify UI callback (if registered) so it can display the result
        if self.on_tool_end is not None:
            try:
                self.on_tool_end(name, input_data, result)
            except Exception:
                pass  # Never let a UI callback break tool execution

        return result

    def end_session(self) -> None:
        """End the session gracefully, saving memory."""
        self.memory.end_session(self.messages, client=self._get_client())

    def _parse_anthropic_response(self, response) -> tuple[str, list]:
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        return "\n".join(text_parts), tool_calls

    def _handle_anthropic_tools(self, response, tool_calls: list) -> None:
        self.messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            result = self._execute_tool(tc["name"], tc["input"])
            tool_results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": result})
        self.messages.append({"role": "user", "content": tool_results})

    def _parse_openai_response(self, response) -> tuple[str, list]:
        import json

        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                for parsed in _parse_tool_arguments(tc.function.arguments):
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": parsed,
                        }
                    )
        return text, tool_calls

    def _handle_openai_tools(self, response, tool_calls: list) -> None:
        choice = response.choices[0]
        self.messages.append(
            {
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": __import__("json").dumps(tc["input"]),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            result = self._execute_tool(tc["name"], tc["input"])
            self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
