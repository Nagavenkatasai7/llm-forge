"""Data Agent — scans datasets, validates quality, searches HuggingFace.
Backed by Gemini 2.5 Flash via Google ADK.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from llm_forge.chat.agent_tools.data_tools import (
    DATA_TOOL_DEFINITIONS, detect_hardware, download_model,
    scan_data, search_huggingface, show_model_info,
)

logger = logging.getLogger(__name__)

DATA_AGENT_PROMPT = """\
You are the Data Agent for llm-forge, an LLM training platform. Your job is to analyze
training datasets and help users understand their data before fine-tuning.

When analyzing data, always report:
- **Format**: JSONL, CSV, Parquet, HuggingFace dataset
- **Sample count**: Total number of training examples
- **Columns/fields**: What fields are present
- **Data preview**: First 2-3 examples formatted clearly
- **Estimated token count**: Rough token estimate
- **Quality issues**: Empty fields, duplicates, encoding problems

When the user asks about datasets, search HuggingFace Hub for relevant options.
When asked about hardware, detect and report the user's GPU/CPU/RAM.
Always return structured, actionable information.
"""

DATA_AGENT_TOOLS = [scan_data, detect_hardware, search_huggingface, download_model, show_model_info]

def _dispatch_tool(name: str, args: dict) -> str:
    tool_map = {"scan_data": scan_data, "detect_hardware": detect_hardware,
                "search_huggingface": search_huggingface, "download_model": download_model,
                "show_model_info": show_model_info}
    fn = tool_map.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})

def create_data_agent(api_key: str) -> Any:
    if not api_key:
        raise ValueError("Google API key is required for Data Agent")
    try:
        from google.adk.agents import Agent
        agent = Agent(name="data_agent", model="gemini-2.5-flash",
                      description="Analyzes training datasets",
                      instruction=DATA_AGENT_PROMPT, tools=DATA_AGENT_TOOLS)
        from llm_forge.chat.agents.base import ADKRunner
        return ADKRunner(agent, api_key)
    except ImportError:
        logger.warning("google-adk not installed. Data Agent using fallback mode.")
        from llm_forge.chat.agents.base import FallbackRunner
        return FallbackRunner("data_agent", _dispatch_tool, DATA_AGENT_PROMPT)
