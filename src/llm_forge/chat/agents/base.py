"""Base agent infrastructure wrapping Google ADK."""
from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Tiered model selection — Flash for simple tool dispatch, Pro for reasoning
AGENT_MODEL_TIER: dict[str, str] = {
    "data_agent": "gemini-2.5-flash",       # Simple: scan files, detect hardware
    "config_agent": "gemini-2.5-flash",     # Simple: write/validate YAML
    "training_agent": "gemini-3.1-pro-preview",  # Complex: diagnose failures, read loss curves
    "eval_agent": "gemini-3.1-pro-preview",      # Complex: interpret benchmarks, compare models
    "export_agent": "gemini-2.5-flash",     # Simple: run export commands
    "research_agent": "gemini-3.1-pro-preview",  # Complex: synthesize search results, explain concepts
}

AGENT_NAMES: list[str] = [
    "data_agent",
    "config_agent",
    "training_agent",
    "eval_agent",
    "export_agent",
    "research_agent",
]

DELEGATE_TOOL: dict[str, Any] = {
    "name": "delegate_to_agent",
    "description": (
        "Delegate a task to a specialist agent. Available agents:\n"
        "- data_agent: Scan datasets, detect format, validate quality, search HuggingFace\n"
        "- config_agent: Write YAML configs, validate, tune hyperparameters, estimate training\n"
        "- training_agent: Launch training, monitor loss curves, diagnose failures, read logs\n"
        "- eval_agent: Run benchmarks (MMLU, GSM8K, IFEval), compare models, generate reports\n"
        "- export_agent: Export to GGUF/safetensors/ONNX, deploy to Ollama, push to HF Hub\n"
        "- research_agent: Search web, HuggingFace, find models/datasets/papers, read docs\n"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "enum": AGENT_NAMES, "description": "Which specialist agent to invoke"},
            "task": {"type": "string", "description": "What the agent should do"},
            "context": {"type": "object", "description": "Optional context: file paths, previous results"},
        },
        "required": ["agent", "task"],
    },
}


class ADKRunner:
    """Wraps Google ADK InMemoryRunner with sync run(message) -> str interface."""

    def __init__(self, agent: Any, api_key: str) -> None:
        import os
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        from google.adk.runners import InMemoryRunner
        self._runner = InMemoryRunner(agent=agent)
        self._user_id = "orchestrator"
        self._session_id = f"{agent.name}_session"

    def run(self, message: str) -> str:
        async def _run_async() -> list:
            return await self._runner.run_debug(
                message, user_id=self._user_id,
                session_id=self._session_id, quiet=True,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                events = pool.submit(asyncio.run, _run_async()).result()
        else:
            events = asyncio.run(_run_async())

        texts = []
        for event in events:
            if hasattr(event, "content") and event.content:
                if hasattr(event.content, "parts"):
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
        return "\n".join(texts) if texts else "(No response from agent)"


class FallbackRunner:
    """Fallback when google-adk not installed. Keyword-based tool dispatch."""

    def __init__(self, name: str, dispatch_fn: Callable, system_prompt: str) -> None:
        self._name = name
        self._dispatch = dispatch_fn
        self._prompt = system_prompt

    def run(self, message: str) -> str:
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ["scan", "analyze", "check data", "look at"]):
            for word in message.split():
                if "/" in word or "." in word:
                    return self._dispatch("scan_data", {"path": word.strip("'\"")}  )
        if any(kw in msg_lower for kw in ["hardware", "gpu", "cpu", "vram"]):
            return self._dispatch("detect_hardware", {})
        if any(kw in msg_lower for kw in ["search", "find", "huggingface", "hf"]):
            query = message.split("for")[-1].strip() if "for" in message else message
            return self._dispatch("search_huggingface", {"query": query, "search_type": "model"})
        if any(kw in msg_lower for kw in ["config", "yaml", "write", "generate"]):
            return self._dispatch("list_configs", {})
        if any(kw in msg_lower for kw in ["estimate", "time", "memory"]):
            return self._dispatch("estimate_training", {"model_name": "SmolLM2-135M", "mode": "lora", "num_samples": 1000})
        if any(kw in msg_lower for kw in ["validate", "check config"]):
            for word in message.split():
                if word.endswith(".yaml") or word.endswith(".yml"):
                    return self._dispatch("validate_config", {"config_path": word.strip("'\"")})
        return json.dumps({"status": "info", "agent": self._name, "message": f"Agent received: {message}. Please be more specific."})


class AgentManager:
    """Manages Google ADK agent lifecycle. Provides sync delegate() method.

    Cost optimizations:
    - Agents are lazily created (no upfront cost)
    - Results cached per (agent, task) to avoid duplicate calls
    - Cache auto-clears after 50 entries to bound memory
    """

    def __init__(self, gemini_api_key: str) -> None:
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY is required for sub-agents.")
        self._api_key = gemini_api_key
        self._agents: dict[str, Any] = {}
        self._cache: dict[str, str] = {}  # (agent+task hash) → result
        self._cache_max = 50

    def _get_or_create_agent(self, agent_name: str) -> Any:
        if agent_name in self._agents:
            return self._agents[agent_name]
        if agent_name == "data_agent":
            from llm_forge.chat.agents.data_agent import create_data_agent
            agent = create_data_agent(self._api_key)
        elif agent_name == "config_agent":
            from llm_forge.chat.agents.config_agent import create_config_agent
            agent = create_config_agent(self._api_key)
        elif agent_name == "training_agent":
            from llm_forge.chat.agents.training_agent import create_training_agent
            agent = create_training_agent(self._api_key)
        elif agent_name == "eval_agent":
            from llm_forge.chat.agents.eval_agent import create_eval_agent
            agent = create_eval_agent(self._api_key)
        elif agent_name == "export_agent":
            from llm_forge.chat.agents.export_agent import create_export_agent
            agent = create_export_agent(self._api_key)
        elif agent_name == "research_agent":
            from llm_forge.chat.agents.research_agent import create_research_agent
            agent = create_research_agent(self._api_key)
        else:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {AGENT_NAMES}")
        self._agents[agent_name] = agent
        return agent

    def delegate(self, agent_name: str, task: str, context: dict | None = None) -> str:
        if agent_name not in AGENT_NAMES:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {AGENT_NAMES}")

        # Check cache — avoid duplicate API calls for identical requests
        cache_key = f"{agent_name}:{task}"
        if cache_key in self._cache:
            logger.info("Cache hit for %s (saved an API call)", agent_name)
            return self._cache[cache_key]

        agent_runner = self._get_or_create_agent(agent_name)
        message = task
        if context:
            message += f"\n\nContext:\n```json\n{json.dumps(context, indent=2)}\n```"
        logger.info("Delegating to %s (%s): %s",
                     agent_name, AGENT_MODEL_TIER.get(agent_name, "unknown"), task[:100])
        try:
            response = agent_runner.run(message)
            logger.info("Agent %s completed", agent_name)
            # Cache the result (evict oldest if full)
            if len(self._cache) >= self._cache_max:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[cache_key] = response
            return response
        except Exception as e:
            logger.error("Agent %s failed: %s", agent_name, e)
            return json.dumps({"status": "error", "agent": agent_name, "error": str(e)})
