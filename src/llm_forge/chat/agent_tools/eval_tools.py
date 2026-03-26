"""Evaluation-related tools for the Eval Agent.
Extracted from the monolithic chat/tools.py.
"""
from __future__ import annotations


def run_evaluation(model_path: str, benchmarks: list[str] | None = None) -> str:
    """Run lm-eval benchmarks on a model."""
    from llm_forge.chat.tools import _run_evaluation
    return _run_evaluation(model_path, benchmarks)


def test_model(model: str, question: str, system_prompt: str = "You are a helpful AI assistant.", num_questions: int = 1) -> str:
    """Test a model via NVIDIA NIM."""
    from llm_forge.chat.tools import _test_model
    return _test_model(model, question, system_prompt, num_questions)


def compare_models(model_a: str, model_b: str, questions: list[str], system_prompt: str = "You are a helpful AI assistant.") -> str:
    """A/B test two models on the same questions."""
    from llm_forge.chat.tools import _compare_models
    return _compare_models(model_a, model_b, questions, system_prompt)


def evaluate_with_llm(model_outputs: list[str], questions: list[str], criteria: str = "relevance, accuracy, helpfulness") -> str:
    """Evaluate model outputs using LLM as judge."""
    from llm_forge.chat.tools import _evaluate_with_llm
    return _evaluate_with_llm(model_outputs, questions, criteria)


EVAL_TOOL_DEFINITIONS = [
    {
        "name": "run_evaluation",
        "description": "Run lm-eval benchmarks (MMLU, GSM8K, IFEval, etc.) on a model.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to model"},
                "benchmarks": {"type": "array", "items": {"type": "string"}, "description": "Benchmark names"},
            },
            "required": ["model_path"],
        },
    },
    {
        "name": "test_model",
        "description": "Quick-test a model with a question.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model name or path"},
                "question": {"type": "string"},
                "system_prompt": {"type": "string"},
                "num_questions": {"type": "integer"},
            },
            "required": ["model", "question"],
        },
    },
    {
        "name": "compare_models",
        "description": "A/B test two models on the same questions with AI judging.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_a": {"type": "string"},
                "model_b": {"type": "string"},
                "questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["model_a", "model_b", "questions"],
        },
    },
    {
        "name": "evaluate_with_llm",
        "description": "Score model outputs using LLM-as-judge.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_outputs": {"type": "array", "items": {"type": "string"}},
                "questions": {"type": "array", "items": {"type": "string"}},
                "criteria": {"type": "string"},
            },
            "required": ["model_outputs", "questions"],
        },
    },
]
