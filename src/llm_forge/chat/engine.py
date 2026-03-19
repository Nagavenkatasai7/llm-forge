"""Conversation engine for the LLM Forge CLI assistant.

Uses the Anthropic Claude API with tool use to guide users through
building their own LLMs via natural conversation.
"""

from __future__ import annotations

import os

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


def _call_anthropic(messages: list[dict], system: str) -> dict:
    """Call Claude API with tool use. Returns the API response."""
    import anthropic

    client = anthropic.Anthropic()

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

    # Convert our tool format to OpenAI function calling format
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

    # Convert messages to OpenAI format
    oai_messages = [{"role": "system", "content": system}]
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                oai_messages.append({"role": "user", "content": msg["content"]})
            elif isinstance(msg["content"], list):
                # Tool results
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
# Chat Engine
# ---------------------------------------------------------------------------


class ChatEngine:
    """Manages the conversation loop between the user and the LLM assistant."""

    def __init__(self, provider: str | None = None):
        self.provider = provider or _get_provider()
        self.messages: list[dict] = []
        self.system = SYSTEM_PROMPT

    def send(self, user_input: str) -> str:
        """Send a user message and get the assistant's response.

        Handles the full tool-use loop: if Claude wants to call tools,
        we execute them and feed results back until Claude produces a
        text response.
        """
        self.messages.append({"role": "user", "content": user_input})

        while True:
            if self.provider == "anthropic":
                response = _call_anthropic(self.messages, self.system)
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

            # If no tool calls, we're done — return the text
            if not tool_calls:
                self.messages.append({"role": "assistant", "content": text})
                return text

            # Execute tools and continue the loop
            if self.provider == "anthropic":
                self._handle_anthropic_tools(response, tool_calls)
            elif self.provider == "openai":
                self._handle_openai_tools(response, tool_calls)

    def _parse_anthropic_response(self, response) -> tuple[str, list]:
        """Parse Claude API response into text and tool calls."""
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        return "\n".join(text_parts), tool_calls

    def _handle_anthropic_tools(self, response, tool_calls: list) -> None:
        """Execute tools from Claude response and add results to messages."""
        # Add assistant's response (with tool_use blocks) to messages
        self.messages.append({"role": "assistant", "content": response.content})

        # Execute each tool
        tool_results = []
        for tc in tool_calls:
            result = execute_tool(tc["name"], tc["input"])
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                }
            )

        # Add tool results as user message
        self.messages.append({"role": "user", "content": tool_results})

    def _parse_openai_response(self, response) -> tuple[str, list]:
        """Parse OpenAI response into text and tool calls."""
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
        """Execute tools from OpenAI response and add results to messages."""
        choice = response.choices[0]
        # Add assistant message with tool calls
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

        # Execute and add results
        for tc in tool_calls:
            result = execute_tool(tc["name"], tc["input"])
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )
