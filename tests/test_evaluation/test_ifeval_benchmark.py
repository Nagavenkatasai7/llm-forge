"""Tests for IFEval benchmark integration.

IFEval (Instruction-Following Evaluation) tests that models follow
verifiable instructions (e.g. word count, keyword usage, formatting).
Ref: arxiv:2311.07911
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.evaluation.benchmarks import (
        STANDARD_TASKS,
        TASK_ALIASES,
        BenchmarkRunner,
        _resolve_task_name,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

try:
    from llm_forge.config.schema import EvalConfig, LLMForgeConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.evaluation.benchmarks not importable",
)


# ===================================================================
# Task Registry Tests
# ===================================================================


class TestIFEvalTaskRegistry:
    """Verify IFEval is properly registered in STANDARD_TASKS."""

    def test_ifeval_in_standard_tasks(self) -> None:
        """IFEval should be in the STANDARD_TASKS registry."""
        assert "ifeval" in STANDARD_TASKS

    def test_ifeval_task_name(self) -> None:
        """IFEval lm-eval task name is 'ifeval'."""
        assert STANDARD_TASKS["ifeval"]["task_name"] == "ifeval"

    def test_ifeval_display_name(self) -> None:
        """Display name should be 'IFEval'."""
        assert STANDARD_TASKS["ifeval"]["display_name"] == "IFEval"

    def test_ifeval_zero_shot(self) -> None:
        """IFEval default_fewshot should be 0 (always zero-shot)."""
        assert STANDARD_TASKS["ifeval"]["default_fewshot"] == 0

    def test_ifeval_metric_key(self) -> None:
        """IFEval primary metric is prompt_level_strict_acc."""
        assert STANDARD_TASKS["ifeval"]["metric_key"] == "prompt_level_strict_acc"

    def test_ifeval_alias_exists(self) -> None:
        """instruction_following should alias to ifeval."""
        assert TASK_ALIASES.get("instruction_following") == "ifeval"

    def test_resolve_ifeval(self) -> None:
        """_resolve_task_name should resolve 'ifeval' correctly."""
        assert _resolve_task_name("ifeval") == "ifeval"
        assert _resolve_task_name("IFEval") == "ifeval"
        assert _resolve_task_name("instruction_following") == "ifeval"

    def test_total_benchmark_count(self) -> None:
        """STANDARD_TASKS should have 8 benchmarks after adding IFEval."""
        assert len(STANDARD_TASKS) == 8


# ===================================================================
# BenchmarkRunner Tests
# ===================================================================


class TestBenchmarkRunnerIFEval:
    """Verify BenchmarkRunner handles IFEval correctly."""

    def test_list_tasks_includes_ifeval(self) -> None:
        """list_tasks() should include IFEval."""
        runner = BenchmarkRunner()
        tasks = runner.list_tasks()
        task_names = [t["name"] for t in tasks]
        assert "ifeval" in task_names

    def test_apply_chat_template_parameter(self) -> None:
        """run_benchmarks should accept apply_chat_template parameter."""
        runner = BenchmarkRunner()
        import inspect

        sig = inspect.signature(runner.run_benchmarks)
        assert "apply_chat_template" in sig.parameters
        # Default should be True
        assert sig.parameters["apply_chat_template"].default is True


# ===================================================================
# Config Schema Tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestIFEvalConfig:
    """Verify IFEval works in config validation."""

    def test_ifeval_in_benchmark_list(self) -> None:
        """Config with ifeval in benchmarks list should validate."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            evaluation={"benchmarks": ["ifeval", "hellaswag"]},
        )
        assert "ifeval" in config.evaluation.benchmarks

    def test_ifeval_only_config(self) -> None:
        """Config with only ifeval benchmark should validate."""
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            evaluation={"benchmarks": ["ifeval"]},
        )
        assert config.evaluation.benchmarks == ["ifeval"]
