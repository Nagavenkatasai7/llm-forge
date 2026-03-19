"""Conversation engine for the LLM Forge CLI assistant.

Uses the Anthropic Claude API with tool use to guide users through
building their own LLMs via natural conversation. Integrates the
three-layer memory system for cross-session intelligence.
"""

from __future__ import annotations

import os

from llm_forge.chat.memory import MemoryManager
from llm_forge.chat.system_prompt import SYSTEM_PROMPT
from llm_forge.chat.tools import TOOLS, execute_tool

# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


def _get_provider() -> str:
    """Detect which LLM provider to use based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def _get_anthropic_client():
    """Get a cached Anthropic client instance."""
    import anthropic

    return anthropic.Anthropic()


def _call_anthropic(messages: list[dict], system: str, client=None) -> dict:
    """Call Claude API with tool use. Returns the API response."""
    if client is None:
        client = _get_anthropic_client()

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=system,
        tools=TOOLS,
        messages=messages,
    )
    return response


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

    def __init__(self, provider: str | None = None, project_dir: str | None = None):
        self.provider = provider or _get_provider()
        self.messages: list[dict] = []
        self.memory = MemoryManager(project_dir=project_dir or ".")
        self._client = None

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

    def send(self, user_input: str) -> str:
        """Send a user message and get the assistant's response.

        Handles the full tool-use loop with memory management:
        - Checks for context compaction before each API call
        - Routes memory tools to the MemoryManager
        - Preserves all context within the active workflow
        """
        self.messages.append({"role": "user", "content": user_input})

        # Check if we need to compact before calling the API
        if self.memory.needs_compaction(self.messages):
            self.messages = self.memory.compact_messages(self.messages, client=self._get_client())

        while True:
            if self.provider == "anthropic":
                response = _call_anthropic(self.messages, self.system, client=self._get_client())
                text, tool_calls = self._parse_anthropic_response(response)
            elif self.provider == "openai":
                response = _call_openai(self.messages, self.system)
                text, tool_calls = self._parse_openai_response(response)
            else:
                return (
                    "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.\n\n"
                    "Get a free Claude API key at: https://console.anthropic.com/\n"
                    "Or set OPENAI_API_KEY for OpenAI."
                )

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": text})
                return text

            if self.provider == "anthropic":
                self._handle_anthropic_tools(response, tool_calls)
            elif self.provider == "openai":
                self._handle_openai_tools(response, tool_calls)

    def _execute_tool(self, name: str, input_data: dict) -> str:
        """Execute a tool, routing memory tools to MemoryManager."""
        # Memory-specific tools
        if name == "save_memory":
            return self.memory.save_memory(
                category=input_data.get("category", "general"),
                content=input_data["content"],
                relevance=input_data.get("relevance", 1.0),
            )
        elif name == "recall_memory":
            return self.memory.recall_memory(
                query=input_data["query"],
                limit=input_data.get("limit", 10),
            )
        elif name == "get_project_state":
            import json

            return json.dumps(self.memory.project_state, indent=2)
        elif name == "get_session_history":
            return self.memory.get_session_history(limit=input_data.get("limit", 5))
        elif name == "log_training_run":
            return self.memory.log_training_run(**input_data)
        else:
            return execute_tool(name, input_data)

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
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
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
