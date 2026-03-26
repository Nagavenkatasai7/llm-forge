"""Training Agent — monitors training runs, diagnoses failures, manages training lifecycle.
Backed by Gemini 2.5 Flash via Google ADK.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from llm_forge.chat.agent_tools.training_tools import (
    TRAINING_TOOL_DEFINITIONS, check_training_status, read_training_logs, start_training,
)

logger = logging.getLogger(__name__)

TRAINING_AGENT_PROMPT = """\
You are the Training Agent for llm-forge, an LLM training platform. Your job is to start,
monitor, and diagnose model training runs.

When helping users with training, always:
- **Start training**: Validate the config path exists before starting. Use start_training().
- **Monitor status**: Use check_training_status() to check if training is running.
- **Read logs**: Use read_training_logs() to fetch recent log lines from the output directory.
- **Report metrics**: Parse loss values, steps, and ETA from logs and present them clearly.

Common training failures to diagnose:
- **NaN loss**: Check learning rate (too high?), gradient clipping, input data for NaN/Inf values.
- **OOM (Out of Memory)**: Reduce batch_size, enable gradient_checkpointing, reduce max_seq_length.
- **Loss plateau**: Learning rate may be too low, model may have converged, check data quality.
- **Gibberish output**: completion_only_loss may be False (training on all tokens), tokenizer
  pad_token may equal eos_token, or base model used instead of Instruct variant.
- **Very high initial loss (>10)**: Data format mismatch — check preprocessor output columns.
- **Slow training**: Check GPU utilization, consider increasing batch_size or using bf16.

When a training run finishes:
1. Read the final logs for loss values
2. Report train loss, eval loss, and total training time
3. Recommend next steps (export, benchmark, or adjust hyperparameters)

Always return structured, actionable information.
"""

TRAINING_AGENT_TOOLS = [start_training, check_training_status, read_training_logs]


def _dispatch_tool(name: str, args: dict) -> str:
    tool_map = {
        "start_training": start_training,
        "check_training_status": check_training_status,
        "read_training_logs": read_training_logs,
    }
    fn = tool_map.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_training_agent(api_key: str) -> Any:
    if not api_key:
        raise ValueError("Google API key is required for Training Agent")
    try:
        from google.adk.agents import Agent
        agent = Agent(
            name="training_agent",
            model="gemini-3.1-pro-preview",
            description="Starts, monitors, and diagnoses LLM training runs",
            instruction=TRAINING_AGENT_PROMPT,
            tools=TRAINING_AGENT_TOOLS,
        )
        from llm_forge.chat.agents.base import ADKRunner
        return ADKRunner(agent, api_key)
    except ImportError:
        logger.warning("google-adk not installed. Training Agent using fallback mode.")
        from llm_forge.chat.agents.base import FallbackRunner
        return FallbackRunner("training_agent", _dispatch_tool, TRAINING_AGENT_PROMPT)
