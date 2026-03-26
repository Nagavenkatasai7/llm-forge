"""Eval Agent — runs benchmarks, interprets results, compares models.
Backed by Gemini 2.5 Flash via Google ADK.
"""
from __future__ import annotations
import json
import logging
from typing import Any

from llm_forge.chat.agent_tools.eval_tools import (
    EVAL_TOOL_DEFINITIONS, compare_models, evaluate_with_llm,
    run_evaluation, test_model,
)

logger = logging.getLogger(__name__)

EVAL_AGENT_PROMPT = """\
You are the Eval Agent for llm-forge, an LLM training platform. Your job is to run
benchmarks, interpret evaluation results, and compare models.

Key benchmark baselines for a 1B parameter model (e.g., Llama-3.2-1B-Instruct):
- **MMLU** (general knowledge): ~46% baseline. Measures world knowledge across 57 subjects.
- **GSM8K** (math reasoning): ~33% baseline. Measures grade-school math problem-solving.
- **IFEval** (instruction following): ~43% baseline. Measures adherence to instructions.
- **ARC** (science reasoning): ~40% baseline. Multiple-choice science questions.
- **HellaSwag** (commonsense): ~60% baseline. Common-sense sentence completion.

Catastrophic forgetting thresholds:
- Drop > 5% vs base model = **catastrophic forgetting** — do NOT deploy.
- Drop 2-5% vs base model = **moderate regression** — investigate before deploying.
- Drop < 2% vs base model = **acceptable** — safe to deploy.
- Improvement vs base model = **domain gain** — excellent result.

When interpreting results, always:
1. Compare the fine-tuned model against the base model baseline
2. Flag any benchmark that dropped > 5% as a catastrophic forgetting issue
3. Summarize the trade-off: domain gain vs general capability loss
4. Recommend whether the model is safe to deploy based on these thresholds
5. Suggest config changes (lower LR, fewer epochs, smaller LoRA rank) if forgetting is detected

When running evaluations, prefer smaller benchmark subsets (mmlu_flan_n_shot_generative_5) for
quick checks and full suites for final validation.
"""

EVAL_AGENT_TOOLS = [run_evaluation, test_model, compare_models, evaluate_with_llm]


def _dispatch_tool(name: str, args: dict) -> str:
    tool_map = {
        "run_evaluation": run_evaluation,
        "test_model": test_model,
        "compare_models": compare_models,
        "evaluate_with_llm": evaluate_with_llm,
    }
    fn = tool_map.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_eval_agent(api_key: str) -> Any:
    if not api_key:
        raise ValueError("Google API key is required for Eval Agent")
    try:
        from google.adk.agents import Agent
        agent = Agent(
            name="eval_agent",
            model="gemini-3.1-pro-preview",
            description="Runs benchmarks, interprets evaluation results, and compares models",
            instruction=EVAL_AGENT_PROMPT,
            tools=EVAL_AGENT_TOOLS,
        )
        from llm_forge.chat.agents.base import ADKRunner
        return ADKRunner(agent, api_key)
    except ImportError:
        logger.warning("google-adk not installed. Eval Agent using fallback mode.")
        from llm_forge.chat.agents.base import FallbackRunner
        return FallbackRunner("eval_agent", _dispatch_tool, EVAL_AGENT_PROMPT)
