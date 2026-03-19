"""Smart context detection for LLM Forge.

Classifies user input as instructions vs. pasted content (model output,
error logs, code, etc.) and wraps it appropriately so Claude responds
as an evaluator rather than responding to the content directly.
"""

from __future__ import annotations


def classify_and_wrap_input(
    user_text: str,
    recent_tool_calls: list[str] | None = None,
    conversation_length: int = 0,
) -> str:
    """Classify user input and optionally wrap it with context markers.

    If the input looks like pasted content (model output, error logs,
    stack traces, code), wraps it with context so Claude knows to
    analyze it rather than respond to it.

    Returns the original text or a context-wrapped version.
    """
    recent_tools = recent_tool_calls or []
    classification = _classify(user_text, recent_tools)

    if classification == "model_output":
        return _wrap_model_output(user_text)
    elif classification == "error_log":
        return _wrap_error_log(user_text)
    elif classification == "code_snippet":
        return _wrap_code_snippet(user_text)
    else:
        return user_text  # Normal instruction, don't wrap


def _classify(text: str, recent_tools: list[str]) -> str:
    """Classify the type of input."""
    lines = text.strip().split("\n")
    line_count = len(lines)
    char_count = len(text)
    has_question = "?" in text

    # Short messages are almost always instructions
    if char_count < 200 and line_count <= 3:
        return "instruction"

    # Check for error/stack trace patterns
    error_indicators = [
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
        "FAILED",
        "error:",
        "RuntimeError",
        "ValueError",
        "ModuleNotFoundError",
        "ImportError",
        "CUDA out of memory",
        "OOM",
        'File "',
        "line ",
    ]
    error_score = sum(1 for ind in error_indicators if ind in text)
    if error_score >= 2:
        return "error_log"

    # Check for code patterns
    code_indicators = [
        "def ",
        "class ",
        "import ",
        "from ",
        "if __name__",
        "return ",
        "self.",
        "```",
        ">>>",
    ]
    code_score = sum(1 for ind in code_indicators if ind in text)
    if code_score >= 3:
        return "code_snippet"

    # Check for model output patterns (after recent training/eval)
    ml_tools = {
        "start_training",
        "read_training_logs",
        "check_training_status",
        "run_evaluation",
        "deploy_to_ollama",
        "show_model_info",
    }
    recent_ml_action = any(t in ml_tools for t in recent_tools[-5:])

    if recent_ml_action and line_count > 3 and not has_question:
        # Long text after ML action with no question = likely model output
        return "model_output"

    # Long text with no question mark — might be pasted content
    if line_count > 10 and not has_question:
        return "model_output"  # Treat as pasted content

    return "instruction"


def _wrap_model_output(text: str) -> str:
    """Wrap text identified as model output."""
    return (
        "The user has pasted the following OUTPUT from their trained model. "
        "Analyze and evaluate this output — do NOT respond as if you are that model. "
        "Comment on quality, relevance, accuracy, and any issues:\n\n"
        "--- BEGIN MODEL OUTPUT ---\n"
        f"{text}\n"
        "--- END MODEL OUTPUT ---"
    )


def _wrap_error_log(text: str) -> str:
    """Wrap text identified as an error/stack trace."""
    return (
        "The user has pasted the following ERROR OUTPUT from their system. "
        "Diagnose the problem, explain what went wrong in simple terms, "
        "and fix it if possible:\n\n"
        "--- BEGIN ERROR LOG ---\n"
        f"{text}\n"
        "--- END ERROR LOG ---"
    )


def _wrap_code_snippet(text: str) -> str:
    """Wrap text identified as code."""
    return (
        "The user has pasted the following CODE. "
        "They likely want you to review, explain, or integrate it:\n\n"
        "--- BEGIN CODE ---\n"
        f"{text}\n"
        "--- END CODE ---"
    )
