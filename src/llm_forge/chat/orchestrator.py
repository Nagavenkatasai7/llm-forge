"""OrchestratorEngine — Claude orchestrator with ADK sub-agents.

Replaces ChatEngine's single-LLM approach with a hierarchical design:
Claude acts as the orchestrator with ONE meta-tool (``delegate_to_agent``)
to dispatch tasks to specialist sub-agents (data_agent, config_agent).

The send() method implements a standard Claude tool-use loop:
  1. User message -> append to messages
  2. Call Claude with ORCHESTRATOR_TOOLS
  3. If no tool calls -> return text
  4. Handle tool calls (delegate_to_agent, memory tools, execution tools)
  5. Append results -> repeat from 2
  6. Max 15 iterations to prevent infinite loops
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from llm_forge.chat.agents.base import AGENT_NAMES, DELEGATE_TOOL, AgentManager
from llm_forge.chat.execution import (
    EXECUTION_TOOL_NAMES,
    PermissionSystem,
    execute_execution_tool,
)
from llm_forge.chat.memory import MemoryManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Claude model catalogue (mirrors engine.py)
# ---------------------------------------------------------------------------

CLAUDE_MODELS: dict[str, dict[str, str]] = {
    "opus-4.6": {
        "id": "claude-opus-4-6",
        "name": "Claude Opus 4.6",
        "context": "200K (1M beta)",
        "cost": "$5/$25",
    },
    "sonnet-4.6": {
        "id": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "context": "200K (1M beta)",
        "cost": "$3/$15",
    },
    "haiku-4.5": {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "context": "200K",
        "cost": "$1/$5",
    },
    "opus-4.5": {
        "id": "claude-opus-4-5",
        "name": "Claude Opus 4.5",
        "context": "200K",
        "cost": "$5/$25",
    },
    "sonnet-4.5": {
        "id": "claude-sonnet-4-5",
        "name": "Claude Sonnet 4.5",
        "context": "200K",
        "cost": "$3/$15",
    },
}

DEFAULT_MODEL = "opus-4.6"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a router. ALWAYS delegate via `delegate_to_agent`. NEVER answer directly.

Agents: data_agent (data/hardware), config_agent (YAML configs), \
training_agent (train/monitor), eval_agent (benchmarks), \
export_agent (GGUF/Ollama/HF), research_agent (search/docs/concepts).

For multi-step: chain agents (data→config→training→eval→export).

Memory tools (use directly): save_memory, recall_memory, get_project_state, get_session_history.

Be concise. Relay agent results. No extra commentary.
4. If an agent fails, explain the error and suggest alternatives.
5. Always be honest about what you can and cannot do.
"""

# ---------------------------------------------------------------------------
# Memory tool definitions (Anthropic tool-use schema)
# ---------------------------------------------------------------------------

_MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "save_memory",
        "description": (
            "Store an important fact for future recall across sessions. "
            "Use proactively when the user shares preferences, hardware info, "
            "goals, or project decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category: user_preference, hardware, goal, decision, training_result",
                },
                "content": {
                    "type": "string",
                    "description": "The fact to remember",
                },
                "relevance": {
                    "type": "number",
                    "description": "Relevance score 0.0-1.0 (default 1.0)",
                },
            },
            "required": ["category", "content"],
        },
    },
    {
        "name": "recall_memory",
        "description": "Search past memories by keyword. Returns matching memories from previous sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword or phrase",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_project_state",
        "description": "Get current project directory scan: configs, trained models, data files.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_session_history",
        "description": "Get summaries of past conversation sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max sessions to return (default 5)",
                },
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Combined tool list for the orchestrator
# ---------------------------------------------------------------------------

ORCHESTRATOR_TOOLS: list[dict[str, Any]] = [DELEGATE_TOOL] + _MEMORY_TOOLS

# ---------------------------------------------------------------------------
# OrchestratorEngine
# ---------------------------------------------------------------------------


class OrchestratorEngine:
    """Claude-based orchestrator that delegates to ADK sub-agents.

    Parameters
    ----------
    project_dir:
        Root directory of the llm-forge project. Defaults to cwd.
    model_key:
        Which Claude model to use (see ``CLAUDE_MODELS``).
    gemini_api_key:
        Google API key for the Gemini-backed sub-agents. Required.
    """

    def __init__(
        self,
        project_dir: str | None = None,
        model_key: str | None = None,
        gemini_api_key: str = "",
    ) -> None:
        from llm_forge.chat.api_keys import get_anthropic_api_key, get_google_api_key

        # Use built-in keys if not provided
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or get_anthropic_api_key()
        google_key = gemini_api_key or get_google_api_key()

        # Set env var so anthropic.Anthropic() picks it up
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key

        # --- Anthropic client ---
        import anthropic

        self._client = anthropic.Anthropic(api_key=anthropic_key)

        # --- sub-systems ---
        self.provider = "anthropic"  # Compatibility with _print_model_info
        self.model_key = model_key or DEFAULT_MODEL
        self.messages: list[dict[str, Any]] = []
        self.memory = MemoryManager(project_dir=project_dir or ".")
        self.permissions = PermissionSystem(auto_approve=True)
        self.agents = AgentManager(gemini_api_key=google_key)

        # Build system prompt with memory context
        context_block = self.memory.build_context_block()
        if context_block.strip():
            self._system = f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n---\n\n{context_block}"
        else:
            self._system = ORCHESTRATOR_SYSTEM_PROMPT

        # --- optional UI callbacks ---
        self.on_agent_start: Callable[..., Any] | None = None
        self.on_agent_end: Callable[..., Any] | None = None
        self.on_tool_start: Callable[..., Any] | None = None
        self.on_tool_end: Callable[..., Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        user_input: str,
        on_text: Callable[[str], None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> str:
        """Send a user message and return the assistant's response.

        Implements a standard Claude tool-use loop with a max of 15
        iterations to prevent infinite loops.
        """
        self.messages.append({"role": "user", "content": user_input})

        # Context compaction
        if self.memory.needs_compaction(self.messages):
            self.messages = self.memory.compact_messages(
                self.messages, client=self._client
            )

        max_iterations = 8  # Limit Claude round-trips to control Opus costs
        for _iteration in range(max_iterations):
            # Check for interrupt
            if interrupt_check and interrupt_check():
                partial = "[interrupted by user]"
                self.messages.append({"role": "assistant", "content": partial})
                return partial

            # Call Claude
            model_id = CLAUDE_MODELS.get(
                self.model_key, CLAUDE_MODELS[DEFAULT_MODEL]
            )["id"]

            if on_text:
                response = self._stream_call(model_id, on_text, interrupt_check)
            else:
                response = self._client.messages.create(
                    model=model_id,
                    max_tokens=2048,
                    system=self._system,
                    tools=ORCHESTRATOR_TOOLS,
                    messages=self.messages,
                )

            text, tool_calls = self._parse_response(response)

            # Check for interrupt after API call
            if interrupt_check and interrupt_check():
                partial = text or "[interrupted by user]"
                self.messages.append({"role": "assistant", "content": partial})
                return partial

            # No tool calls -> final answer
            if not tool_calls:
                self.messages.append({"role": "assistant", "content": text})
                return text

            # Handle tool calls
            self.messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tc in tool_calls:
                result = self._execute_tool(tc["name"], tc["input"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})

        # Iteration limit reached
        partial = text or ""
        if partial:
            partial += "\n\n"
        partial += (
            "[Tool-call limit reached: the assistant used tools "
            f"{max_iterations} times in a row. Returning partial response "
            "to avoid an infinite loop.]"
        )
        self.messages.append({"role": "assistant", "content": partial})
        return partial

    def end_session(self) -> None:
        """End the session, persisting a summary to memory."""
        self.memory.end_session(self.messages, client=self._client)

    # ------------------------------------------------------------------
    # Internal: API calls
    # ------------------------------------------------------------------

    def _stream_call(
        self,
        model_id: str,
        on_text: Callable[[str], None],
        interrupt_check: Callable[[], bool] | None = None,
    ) -> Any:
        """Stream a Claude call, invoking on_text for each text chunk."""
        collected_text: list[str] = []

        with self._client.messages.stream(
            model=model_id,
            max_tokens=2048,
            system=self._system,
            tools=ORCHESTRATOR_TOOLS,
            messages=self.messages,
        ) as stream:
            for event in stream:
                if interrupt_check and interrupt_check():
                    break
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        collected_text.append(event.delta.text)
                        on_text(event.delta.text)

            try:
                return stream.get_final_message()
            except Exception:
                from types import SimpleNamespace

                text = "".join(collected_text) or "[incomplete response]"
                block = SimpleNamespace(type="text", text=text)
                return SimpleNamespace(
                    content=[block],
                    stop_reason="end_turn",
                )

    # ------------------------------------------------------------------
    # Internal: response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> tuple[str, list[dict[str, Any]]]:
        """Extract text and tool_calls from a Claude response."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )
        return "\n".join(text_parts), tool_calls

    def _extract_text(self, response: Any) -> str:
        """Extract just the text from a Claude response."""
        return self._parse_response(response)[0]

    # ------------------------------------------------------------------
    # Internal: tool routing
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, input_data: dict[str, Any]) -> str:
        """Route a tool call to the appropriate handler."""
        # Notify UI
        if self.on_tool_start is not None:
            try:
                self.on_tool_start(name, input_data)
            except Exception:
                pass

        result: str

        if name == "delegate_to_agent":
            result = self._handle_delegation(
                agent_name=input_data["agent"],
                task=input_data["task"],
                context=input_data.get("context"),
            )
        elif name == "save_memory":
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
            result = self.memory.get_session_history(
                limit=input_data.get("limit", 5)
            )
        elif name in EXECUTION_TOOL_NAMES:
            allowed, reason = self.permissions.check(name, input_data)
            if not allowed:
                result = json.dumps({"status": "blocked", "reason": reason})
            else:
                result = execute_execution_tool(name, input_data)
        else:
            result = json.dumps(
                {"error": f"Unknown tool: {name}. Available: delegate_to_agent, memory tools."}
            )

        # Notify UI
        if self.on_tool_end is not None:
            try:
                self.on_tool_end(name, input_data, result)
            except Exception:
                pass

        return result

    def _handle_delegation(
        self,
        agent_name: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Delegate a task to a sub-agent via AgentManager."""
        if self.on_agent_start is not None:
            try:
                self.on_agent_start(agent_name, task)
            except Exception:
                pass

        logger.info("Delegating to %s: %s", agent_name, task[:100])
        result = self.agents.delegate(agent_name, task, context)

        if self.on_agent_end is not None:
            try:
                self.on_agent_end(agent_name, result)
            except Exception:
                pass

        return result
