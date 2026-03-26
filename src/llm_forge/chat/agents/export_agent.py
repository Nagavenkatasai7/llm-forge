"""Export Agent — converts trained models to GGUF/safetensors/ONNX and deploys to Ollama or HuggingFace Hub.
Backed by Gemini 2.5 Flash via Google ADK.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from llm_forge.chat.agent_tools.export_tools import (
    deploy_to_huggingface,
    deploy_to_ollama,
    export_model,
    show_model_info,
)

logger = logging.getLogger(__name__)

EXPORT_AGENT_PROMPT = """\
You are the Export Agent for llm-forge, an LLM training platform. Your job is to help
users convert their fine-tuned models into deployable formats and publish them.

## GGUF Export
- Use Q4_K_M quantization for 1–3B models (best balance of size and quality, ~763 MB for 1B)
- Use Q8_0 for higher quality when storage is not a concern
- Never include <|begin_of_text|> in the Modelfile — Ollama adds BOS automatically
- The Modelfile MUST use `{{ range .Messages }}` loop for multi-turn conversation support
  (not the outdated .Prompt/.Response single-turn pattern)
- Always set `num_ctx` in the Modelfile to match the model's max_seq_length (default 2048)
- Add `<|start_header_id|>` as a stop token and set repeat_penalty 1.1–1.3 for Llama 3

## Ollama Deployment
- Use `deploy_to_ollama` to export GGUF and create a Modelfile in one step
- The model_name should be lowercase with hyphens (e.g., "finance-specialist-v7")
- Provide a clear system_prompt that describes the model's purpose and capabilities

## HuggingFace Hub Upload
- Use `deploy_to_huggingface` to upload safetensors weights with an auto-generated model card
- Set private=True for experimental models; private=False for public release
- Include a descriptive description covering training data, base model, and intended use

## Supported Formats
- **gguf**: Quantized binary format for Ollama / llama.cpp local inference
- **safetensors**: PyTorch weights in safe, portable format (preferred over .bin)
- **onnx**: Cross-platform format for deployment on non-PyTorch runtimes
- **awq**: 4-bit quantization format optimized for GPU inference

## Workflow
1. Run `show_model_info` to inspect the model path, size, and architecture
2. Choose the export format based on the deployment target
3. Run `export_model` or `deploy_to_ollama` / `deploy_to_huggingface`
4. Verify the exported artifact exists and report file size

Always return structured, actionable information.
"""

EXPORT_AGENT_TOOLS = [export_model, deploy_to_ollama, deploy_to_huggingface, show_model_info]


def _dispatch_tool(name: str, args: dict) -> str:
    tool_map = {
        "export_model": export_model,
        "deploy_to_ollama": deploy_to_ollama,
        "deploy_to_huggingface": deploy_to_huggingface,
        "show_model_info": show_model_info,
    }
    fn = tool_map.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_export_agent(api_key: str) -> Any:
    if not api_key:
        raise ValueError("Google API key is required for Export Agent")
    try:
        from google.adk.agents import Agent
        agent = Agent(
            name="export_agent",
            model="gemini-2.5-flash",
            description="Exports and deploys trained models to GGUF, safetensors, Ollama, and HuggingFace Hub",
            instruction=EXPORT_AGENT_PROMPT,
            tools=EXPORT_AGENT_TOOLS,
        )
        from llm_forge.chat.agents.base import ADKRunner
        return ADKRunner(agent, api_key)
    except ImportError:
        logger.warning("google-adk not installed. Export Agent using fallback mode.")
        from llm_forge.chat.agents.base import FallbackRunner
        return FallbackRunner("export_agent", _dispatch_tool, EXPORT_AGENT_PROMPT)
