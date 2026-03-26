"""Research Agent — searches HuggingFace, the web, and documentation.
Backed by Gemini 2.5 Flash via Google ADK.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from llm_forge.chat.agent_tools.research_tools import (
    RESEARCH_TOOL_DEFINITIONS, download_model, install_dependencies,
    read_url, search_huggingface, web_search,
)

logger = logging.getLogger(__name__)

RESEARCH_AGENT_PROMPT = """\
You are the Research Agent for llm-forge, an LLM training platform. Your job is to help
users discover and evaluate models, datasets, and research papers for their fine-tuning projects.

When helping users research, always:
- **Search HuggingFace** for relevant models and datasets using search_huggingface
- **Web search** for documentation, tutorials, and research papers using web_search
  (Note: web_search is a placeholder — it will return a not-implemented status for now)
- **Read URLs** to fetch documentation or paper abstracts using read_url
  (Note: read_url is a placeholder — it will return a not-implemented status for now)
- **Cite sources** with full URLs so users can explore further

When recommending a base model for a use case:
- Report model size (parameters), license, and HuggingFace URL
- Note whether the model supports the required task (text generation, classification, etc.)
- Recommend 2-3 options at different size/quality trade-offs

When finding training datasets:
- Report dataset size (number of examples), license, and HuggingFace URL
- Describe the data format (JSONL, Parquet, etc.) and key fields
- Flag any known quality issues or license restrictions

When searching for research papers:
- Provide the paper title, authors, and ArXiv/URL link
- Summarize the key technique and its relevance to the user's question
- Note any publicly available reference implementations

Always return structured, actionable information with URLs for every resource cited.
"""

RESEARCH_AGENT_TOOLS = [search_huggingface, download_model, install_dependencies, web_search, read_url]


def _dispatch_tool(name: str, args: dict) -> str:
    tool_map = {
        "search_huggingface": search_huggingface,
        "download_model": download_model,
        "install_dependencies": install_dependencies,
        "web_search": web_search,
        "read_url": read_url,
    }
    fn = tool_map.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_research_agent(api_key: str) -> Any:
    if not api_key:
        raise ValueError("Google API key is required for Research Agent")
    try:
        from google.adk.agents import Agent
        agent = Agent(
            name="research_agent",
            model="gemini-3.1-pro-preview",
            description="Searches HuggingFace, the web, and documentation for models, datasets, and papers",
            instruction=RESEARCH_AGENT_PROMPT,
            tools=RESEARCH_AGENT_TOOLS,
        )
        from llm_forge.chat.agents.base import ADKRunner
        return ADKRunner(agent, api_key)
    except ImportError:
        logger.warning("google-adk not installed. Research Agent using fallback mode.")
        from llm_forge.chat.agents.base import FallbackRunner
        return FallbackRunner("research_agent", _dispatch_tool, RESEARCH_AGENT_PROMPT)
