"""Tests for the smart context detection system."""

from __future__ import annotations

from llm_forge.chat.context_detector import (
    _classify,
    _wrap_code_snippet,
    _wrap_error_log,
    _wrap_model_output,
    classify_and_wrap_input,
)

# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassification:
    """Tests for input classification logic."""

    def test_short_text_is_instruction(self):
        """Short text (< 200 chars, <= 3 lines) should be classified as instruction."""
        result = _classify("Train my model with LoRA", [])
        assert result == "instruction"

    def test_question_mark_is_instruction(self):
        """Text with a question mark should stay as instruction even if long."""
        # Long text (> 10 lines) but has a question mark
        long_question = "I ran training and got this output:\n" + "line\n" * 12 + "Is this good?"
        result = _classify(long_question, [])
        assert result == "instruction"

    def test_error_traceback_detected(self):
        """Text containing Traceback and error indicators = error_log."""
        error_text = (
            "Running training...\n"
            "Traceback (most recent call last):\n"
            '  File "train.py", line 42, in <module>\n'
            "    trainer.train()\n"
            "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
            "Error: Training failed\n"
        )
        result = _classify(error_text, [])
        assert result == "error_log"

    def test_model_output_after_training(self):
        """Long text with no question after recent ML tool = model_output."""
        model_response = (
            "Hello! I'm your financial advisor.\n"
            "I can help you with investment strategies.\n"
            "Let me explain the basics of portfolio management.\n"
            "First, diversification is key to managing risk.\n"
        )
        recent_tools = ["start_training", "read_training_logs", "deploy_to_ollama"]
        result = _classify(model_response, recent_tools)
        assert result == "model_output"

    def test_code_snippet_detected(self):
        """Text with multiple code indicators = code_snippet."""
        code = (
            "Here is the code I wrote:\n"
            "import torch\n"
            "from transformers import AutoModel\n"
            "\n"
            "class MyModel:\n"
            "    def __init__(self):\n"
            "        self.model = AutoModel.from_pretrained('gpt2')\n"
            "\n"
            "    def forward(self, x):\n"
            "        return self.model(x)\n"
        )
        result = _classify(code, [])
        assert result == "code_snippet"

    def test_long_text_no_question_is_model_output(self):
        """Long text (> 10 lines) with no question = model_output."""
        text = "\n".join([f"This is line {i} of a long response." for i in range(15)])
        result = _classify(text, [])
        assert result == "model_output"

    def test_short_error_still_instruction(self):
        """A single short error mention is still an instruction."""
        result = _classify("I got an error", [])
        assert result == "instruction"

    def test_ml_tool_context_with_short_text(self):
        """Short text after ML tool should still be instruction."""
        result = _classify("looks good", ["start_training"])
        assert result == "instruction"


# ---------------------------------------------------------------------------
# Wrapping tests
# ---------------------------------------------------------------------------


class TestWrapping:
    """Tests for context wrapping functions."""

    def test_wrap_model_output_has_markers(self):
        """Wrapped model output should contain BEGIN/END markers."""
        result = _wrap_model_output("Hello, I am a finance bot.")
        assert "--- BEGIN MODEL OUTPUT ---" in result
        assert "--- END MODEL OUTPUT ---" in result
        assert "Hello, I am a finance bot." in result
        assert "Analyze and evaluate" in result

    def test_wrap_error_has_markers(self):
        """Wrapped error should contain BEGIN/END markers."""
        result = _wrap_error_log("RuntimeError: CUDA OOM")
        assert "--- BEGIN ERROR LOG ---" in result
        assert "--- END ERROR LOG ---" in result
        assert "RuntimeError: CUDA OOM" in result
        assert "Diagnose" in result

    def test_wrap_code_has_markers(self):
        """Wrapped code should contain BEGIN/END markers."""
        result = _wrap_code_snippet("def foo(): pass")
        assert "--- BEGIN CODE ---" in result
        assert "--- END CODE ---" in result
        assert "def foo(): pass" in result
        assert "review" in result

    def test_normal_text_not_wrapped(self):
        """Short instruction text should not be wrapped at all."""
        original = "Train my model please"
        result = classify_and_wrap_input(original)
        assert result == original
        assert "--- BEGIN" not in result

    def test_error_text_gets_wrapped(self):
        """Error text should be wrapped via classify_and_wrap_input."""
        error_text = (
            "Traceback (most recent call last):\n"
            '  File "run.py", line 10\n'
            "ValueError: invalid literal for int()\n"
            "Error: config parsing failed\n"
            "More error details here\n"
        )
        result = classify_and_wrap_input(error_text)
        assert "--- BEGIN ERROR LOG ---" in result

    def test_model_output_gets_wrapped_with_recent_tools(self):
        """Model output with recent ML tools should be wrapped."""
        output = (
            "Welcome to your financial assistant.\n"
            "I can help with investment advice.\n"
            "Please tell me your risk tolerance.\n"
            "I specialize in portfolio management.\n"
        )
        result = classify_and_wrap_input(
            output,
            recent_tool_calls=["start_training", "deploy_to_ollama"],
            conversation_length=5,
        )
        assert "--- BEGIN MODEL OUTPUT ---" in result

    def test_code_gets_wrapped(self):
        """Code text should be wrapped via classify_and_wrap_input."""
        code = (
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "class Config:\n"
            "    def __init__(self):\n"
            "        self.path = Path('.')\n"
            "\n"
            "    def load(self):\n"
            "        return self.path\n"
        )
        result = classify_and_wrap_input(code)
        assert "--- BEGIN CODE ---" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for context detection."""

    def test_empty_string(self):
        """Empty input should be returned as-is (classified as instruction)."""
        result = classify_and_wrap_input("")
        assert result == ""

    def test_none_recent_tools(self):
        """None recent_tool_calls should not crash."""
        result = classify_and_wrap_input("hello", recent_tool_calls=None)
        assert result == "hello"

    def test_mixed_content_error_wins(self):
        """When text has both code and error indicators, error wins (checked first)."""
        mixed = (
            "import torch\n"
            "from transformers import AutoModel\n"
            "Traceback (most recent call last):\n"
            '  File "train.py", line 5\n'
            "RuntimeError: CUDA out of memory\n"
            "Error: training crashed\n"
            "def cleanup(): pass\n"
        )
        result = _classify(mixed, [])
        assert result == "error_log"

    def test_conversation_length_passed_through(self):
        """conversation_length parameter should not cause errors."""
        result = classify_and_wrap_input(
            "short text",
            recent_tool_calls=[],
            conversation_length=100,
        )
        assert result == "short text"
