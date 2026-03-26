"""Config Agent — generates, validates, and optimizes YAML training configs.
Backed by Gemini 2.5 Flash via Google ADK.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from llm_forge.chat.agent_tools.config_tools import (
    CONFIG_TOOL_DEFINITIONS, estimate_training, list_configs,
    validate_config, write_config,
)

logger = logging.getLogger(__name__)

CONFIG_AGENT_PROMPT = """\
You are the Config Agent for llm-forge, an LLM training platform. Your job is to generate,
validate, and optimize YAML training configurations.

Key rules for generating configs:
- **LoRA r and alpha**: r must be a power of 2 (8, 16, 32, 64). alpha = 2*r is a good default.
- **Learning rate**: 1e-4 for LoRA, 2e-5 for full fine-tuning, 1e-5 for conservative.
- **Batch size**: Auto-detect based on VRAM: 8GB->bs=1, 16GB->bs=2, 24GB->bs=4, 80GB->bs=16.
- **max_seq_length**: At least 2048 for instruction tuning with multi-turn.
- **assistant_only_loss**: true for chat/instruction data.
- **New features**: Disabled by default (enabled: false).

When generating a config, always:
1. Ask about the model, data, and training mode if not specified
2. Detect hardware to set appropriate batch size
3. Validate the config against the schema before returning
4. Explain your hyperparameter choices
"""

CONFIG_AGENT_TOOLS = [write_config, validate_config, list_configs, estimate_training]


def _dispatch_tool(name: str, args: dict) -> str:
    tool_map = {
        "write_config": write_config,
        "validate_config": validate_config,
        "list_configs": list_configs,
        "estimate_training": estimate_training,
    }
    fn = tool_map.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_config_agent(api_key: str) -> Any:
    if not api_key:
        raise ValueError("Google API key is required for Config Agent")
    try:
        from google.adk.agents import Agent
        agent = Agent(
            name="config_agent",
            model="gemini-2.5-flash",
            description="Generates, validates, and optimizes YAML training configurations",
            instruction=CONFIG_AGENT_PROMPT,
            tools=CONFIG_AGENT_TOOLS,
        )
        from llm_forge.chat.agents.base import ADKRunner
        return ADKRunner(agent, api_key)
    except ImportError:
        logger.warning("google-adk not installed. Config Agent using fallback mode.")
        from llm_forge.chat.agents.base import FallbackRunner
        return FallbackRunner("config_agent", _dispatch_tool, CONFIG_AGENT_PROMPT)
