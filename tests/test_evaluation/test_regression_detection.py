"""Tests for regression detection in benchmark comparisons."""

from __future__ import annotations

import pytest

try:
    from llm_forge.evaluation.benchmarks import BenchmarkRunner

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

try:
    from llm_forge.config.schema import EvalConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.evaluation.benchmarks not importable",
)


def _make_comparison(deltas: dict) -> dict:
    """Helper to build a comparison dict with given deltas."""
    comparison = {}
    for task, delta in deltas.items():
        base = 0.5
        comparison[task] = {
            "display_name": task.upper(),
            "base_score": base,
            "finetuned_score": base + delta,
            "delta": delta,
            "pct_change": delta / base * 100,
            "improved": delta > 0,
        }
    all_deltas = list(deltas.values())
    comparison["_summary"] = {
        "avg_delta": sum(all_deltas) / len(all_deltas) if all_deltas else 0,
        "num_improved": sum(1 for d in all_deltas if d > 0),
        "num_degraded": sum(1 for d in all_deltas if d < 0),
        "num_unchanged": sum(1 for d in all_deltas if d == 0),
        "total_tasks": len(all_deltas),
    }
    return comparison


class TestRegressionDetection:
    """Verify check_regression produces correct verdicts."""

    def test_all_improved_passes(self) -> None:
        """All benchmarks improved → passed=True."""
        comparison = _make_comparison({"mmlu": 0.05, "hellaswag": 0.10})
        result = BenchmarkRunner.check_regression(comparison)
        assert result["passed"] is True
        assert len(result["regressions"]) == 0
        assert result["num_improved"] == 2

    def test_one_regression_fails(self) -> None:
        """One benchmark regressed beyond threshold → passed=False."""
        comparison = _make_comparison({"mmlu": 0.05, "hellaswag": -0.05})
        result = BenchmarkRunner.check_regression(comparison, threshold=-0.02)
        assert result["passed"] is False
        assert len(result["regressions"]) == 1
        assert result["regressions"][0]["task"] == "hellaswag"

    def test_small_regression_within_threshold(self) -> None:
        """Small regression within threshold → passed=True."""
        comparison = _make_comparison({"mmlu": 0.05, "hellaswag": -0.01})
        result = BenchmarkRunner.check_regression(comparison, threshold=-0.02)
        assert result["passed"] is True

    def test_all_regressed_fails(self) -> None:
        """All benchmarks regressed → passed=False, grade=D."""
        comparison = _make_comparison({"mmlu": -0.10, "hellaswag": -0.05})
        result = BenchmarkRunner.check_regression(comparison)
        assert result["passed"] is False
        assert result["grade"] == "D"

    def test_grade_a_plus(self) -> None:
        """Large improvement → grade A+."""
        comparison = _make_comparison({"mmlu": 0.25, "hellaswag": 0.20})
        result = BenchmarkRunner.check_regression(comparison)
        assert result["grade"] == "A+"

    def test_grade_b(self) -> None:
        """Small improvement → grade B."""
        comparison = _make_comparison({"mmlu": 0.02, "hellaswag": 0.01})
        result = BenchmarkRunner.check_regression(comparison)
        assert result["grade"] == "B"

    def test_no_change_grade_c(self) -> None:
        """No change → grade C."""
        comparison = _make_comparison({"mmlu": 0.0, "hellaswag": 0.0})
        result = BenchmarkRunner.check_regression(comparison)
        assert result["grade"] == "C"
        assert result["passed"] is True

    def test_custom_threshold(self) -> None:
        """Custom threshold of -0.05 allows small regressions."""
        comparison = _make_comparison({"mmlu": 0.05, "hellaswag": -0.04})
        result = BenchmarkRunner.check_regression(comparison, threshold=-0.05)
        assert result["passed"] is True


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestRegressionConfig:
    """Verify regression_check config fields."""

    def test_regression_check_default_true(self) -> None:
        cfg = EvalConfig()
        assert cfg.regression_check is True

    def test_regression_threshold_default(self) -> None:
        cfg = EvalConfig()
        assert cfg.regression_threshold == -0.02

    def test_can_disable_regression_check(self) -> None:
        cfg = EvalConfig(regression_check=False)
        assert cfg.regression_check is False

    def test_threshold_must_be_negative(self) -> None:
        with pytest.raises(Exception):
            EvalConfig(regression_threshold=0.05)
