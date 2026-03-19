"""Tests for quality report card generation.

The quality report card aggregates benchmark scores, regression analysis,
and training stability into a single pass/fail verdict with a letter grade.
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.evaluation.report_generator import ReportGenerator

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.evaluation.report_generator not importable",
)


def _make_results(avg_score: float = 0.65) -> dict:
    """Build a minimal benchmark results dict."""
    return {
        "hellaswag": {
            "display_name": "HellaSwag",
            "score": avg_score,
            "metric": "acc_norm",
        },
        "_aggregate": {
            "average_score": avg_score,
            "num_tasks": 1,
        },
    }


def _make_comparison(avg_delta: float = 0.05) -> dict:
    """Build a comparison dict with given average delta."""
    base = 0.5
    return {
        "hellaswag": {
            "display_name": "HellaSwag",
            "base_score": base,
            "finetuned_score": base + avg_delta,
            "delta": avg_delta,
            "pct_change": avg_delta / base * 100,
            "improved": avg_delta > 0,
        },
        "_summary": {
            "avg_delta": avg_delta,
            "num_improved": 1 if avg_delta > 0 else 0,
            "num_degraded": 1 if avg_delta < 0 else 0,
            "num_unchanged": 1 if avg_delta == 0 else 0,
            "total_tasks": 1,
        },
    }


def _make_training_history(start: float = 2.5, end: float = 1.2) -> list:
    """Build a training history showing loss decrease."""
    steps = 10
    return [{"step": i * 10, "loss": start - (start - end) * i / (steps - 1)} for i in range(steps)]


# ===================================================================
# generate_quality_card() Tests
# ===================================================================


class TestQualityCardStructure:
    """Verify the quality card has the expected structure."""

    def test_returns_dict(self) -> None:
        card = ReportGenerator.generate_quality_card()
        assert isinstance(card, dict)

    def test_has_verdict(self) -> None:
        card = ReportGenerator.generate_quality_card()
        assert "verdict" in card
        assert card["verdict"] in ("PASS", "FAIL")

    def test_has_grade(self) -> None:
        card = ReportGenerator.generate_quality_card()
        assert "grade" in card
        assert card["grade"] in ("A+", "A", "B+", "B", "C", "D")

    def test_has_overall_score(self) -> None:
        card = ReportGenerator.generate_quality_card()
        assert "overall_score" in card
        assert isinstance(card["overall_score"], float)

    def test_has_dimensions(self) -> None:
        card = ReportGenerator.generate_quality_card()
        assert "dimensions" in card
        dims = card["dimensions"]
        assert "benchmark_performance" in dims
        assert "regression" in dims
        assert "training_stability" in dims

    def test_has_recommendations(self) -> None:
        card = ReportGenerator.generate_quality_card()
        assert "recommendations" in card
        assert isinstance(card["recommendations"], list)

    def test_each_dimension_has_passed(self) -> None:
        card = ReportGenerator.generate_quality_card(
            results=_make_results(),
        )
        for dim in card["dimensions"].values():
            assert "passed" in dim


class TestQualityCardVerdicts:
    """Test pass/fail verdict logic."""

    def test_all_pass_verdict(self) -> None:
        """Good results + good comparison + good training → PASS."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
            comparison=_make_comparison(0.05),
            training_history=_make_training_history(2.5, 1.2),
        )
        assert card["verdict"] == "PASS"

    def test_low_benchmark_fails(self) -> None:
        """Very low benchmark score → FAIL."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.10),
        )
        assert card["verdict"] == "FAIL"
        assert card["dimensions"]["benchmark_performance"]["passed"] is False

    def test_regression_fails(self) -> None:
        """Regression below threshold → FAIL."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
            comparison=_make_comparison(-0.10),
            regression_threshold=-0.02,
        )
        assert card["verdict"] == "FAIL"
        assert card["dimensions"]["regression"]["passed"] is False

    def test_nan_loss_fails(self) -> None:
        """NaN in training loss → FAIL."""
        history = [{"step": 0, "loss": 2.5}, {"step": 10, "loss": float("nan")}]
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
            training_history=history,
        )
        assert card["verdict"] == "FAIL"
        assert card["dimensions"]["training_stability"]["passed"] is False

    def test_no_loss_decrease_fails(self) -> None:
        """Loss increased during training → FAIL."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
            training_history=_make_training_history(1.2, 2.5),
        )
        assert card["dimensions"]["training_stability"]["passed"] is False

    def test_no_results_fails(self) -> None:
        """No results provided → FAIL."""
        card = ReportGenerator.generate_quality_card()
        assert card["verdict"] == "FAIL"

    def test_no_comparison_neutral(self) -> None:
        """No comparison data → regression dimension still passes (neutral)."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
        )
        assert card["dimensions"]["regression"]["passed"] is True

    def test_no_training_history_neutral(self) -> None:
        """No training history → training stability still passes (neutral)."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
        )
        assert card["dimensions"]["training_stability"]["passed"] is True


class TestQualityCardGrades:
    """Test overall grade assignment."""

    def test_high_score_high_grade(self) -> None:
        """High overall score → A or A+."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.95),
            comparison=_make_comparison(0.25),
            training_history=_make_training_history(3.0, 0.5),
        )
        assert card["grade"] in ("A+", "A")

    def test_medium_score_medium_grade(self) -> None:
        """Medium overall score → B range."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.55),
            comparison=_make_comparison(0.03),
            training_history=_make_training_history(2.5, 1.8),
        )
        assert card["grade"] in ("B+", "B", "C")

    def test_regression_grade_d(self) -> None:
        """Regression comparison shows grade D."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
            comparison=_make_comparison(-0.10),
        )
        assert card["dimensions"]["regression"]["grade"] == "D"

    def test_large_improvement_grade_a_plus(self) -> None:
        """Large improvement → regression grade A+."""
        card = ReportGenerator.generate_quality_card(
            comparison=_make_comparison(0.25),
        )
        assert card["dimensions"]["regression"]["grade"] == "A+"


class TestQualityCardRecommendations:
    """Test recommendations generation."""

    def test_all_pass_deployment_ready(self) -> None:
        """When all pass, recommend deployment."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
            comparison=_make_comparison(0.05),
            training_history=_make_training_history(2.5, 1.2),
        )
        assert any("ready" in r.lower() for r in card["recommendations"])

    def test_regression_recommendation(self) -> None:
        """Regression detected → recommendation mentions regressions."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
            comparison=_make_comparison(-0.10),
            regression_threshold=-0.02,
        )
        assert any("regression" in r.lower() for r in card["recommendations"])

    def test_low_benchmark_recommendation(self) -> None:
        """Low benchmark → recommendation to improve."""
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.10),
        )
        assert any(
            "benchmark" in r.lower() or "training" in r.lower() for r in card["recommendations"]
        )


class TestQualityCardHTMLSection:
    """Test the HTML rendering of the quality card."""

    def test_html_contains_verdict(self) -> None:
        gen = ReportGenerator()
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
        )
        html = gen._build_quality_card_section(card)
        assert "PASS" in html or "FAIL" in html

    def test_html_contains_grade(self) -> None:
        gen = ReportGenerator()
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
        )
        html = gen._build_quality_card_section(card)
        assert card["grade"] in html

    def test_html_contains_quality_score(self) -> None:
        gen = ReportGenerator()
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
        )
        html = gen._build_quality_card_section(card)
        assert "Quality Score" in html

    def test_html_contains_dimensions(self) -> None:
        gen = ReportGenerator()
        card = ReportGenerator.generate_quality_card(
            results=_make_results(0.65),
        )
        html = gen._build_quality_card_section(card)
        assert "Benchmark Performance" in html
        assert "Regression Check" in html
        assert "Training Stability" in html
