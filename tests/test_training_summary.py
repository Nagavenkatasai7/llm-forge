"""Tests for the post-training quality report card.

Covers grading logic, TrainingSummary properties (loss improvement,
benchmark improvement, strengths, watch areas, recommendations),
grade assignment, and display formatting.
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.evaluation.training_summary import (
        TrainingSummary,
        compute_grade,
        display_training_summary,
        grade_color,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE,
    reason="llm_forge.evaluation.training_summary not importable",
)


# ===================================================================
# compute_grade() tests
# ===================================================================


class TestComputeGrade:
    """Test the grade assignment from improvement percentages."""

    def test_a_plus_exceptional(self) -> None:
        """>=20% improvement → A+."""
        grade, desc = compute_grade(0.25)
        assert grade == "A+"
        assert "exceptional" in desc.lower()

    def test_a_significant(self) -> None:
        """10-20% improvement → A."""
        grade, _ = compute_grade(0.15)
        assert grade == "A"

    def test_b_plus_good(self) -> None:
        """5-10% improvement → B+."""
        grade, _ = compute_grade(0.07)
        assert grade == "B+"

    def test_b_modest(self) -> None:
        """0-5% improvement → B."""
        grade, _ = compute_grade(0.03)
        assert grade == "B"

    def test_c_no_change(self) -> None:
        """-2% to 0% → C."""
        grade, _ = compute_grade(-0.01)
        assert grade == "C"

    def test_d_regression(self) -> None:
        """< -2% → D."""
        grade, _ = compute_grade(-0.05)
        assert grade == "D"

    def test_boundary_a_plus(self) -> None:
        """Exactly 20% → A+."""
        grade, _ = compute_grade(0.20)
        assert grade == "A+"

    def test_boundary_b(self) -> None:
        """Exactly 0% → B."""
        grade, _ = compute_grade(0.00)
        assert grade == "B"


# ===================================================================
# grade_color() tests
# ===================================================================


class TestGradeColor:
    """Test Rich colour mapping for grades."""

    def test_a_grades_green(self) -> None:
        assert grade_color("A+") == "green"
        assert grade_color("A") == "green"

    def test_b_grades_cyan(self) -> None:
        assert grade_color("B+") == "cyan"
        assert grade_color("B") == "cyan"

    def test_c_yellow(self) -> None:
        assert grade_color("C") == "yellow"

    def test_d_red(self) -> None:
        assert grade_color("D") == "red"


# ===================================================================
# TrainingSummary property tests
# ===================================================================


class TestTrainingSummaryProperties:
    """Test computed properties of TrainingSummary."""

    def test_loss_improvement_positive(self) -> None:
        """Loss decrease → positive improvement."""
        s = TrainingSummary(training_loss_start=2.0, training_loss_end=1.0)
        assert s.loss_improvement == pytest.approx(0.5)

    def test_loss_improvement_zero_start(self) -> None:
        """Zero start loss → 0 improvement (avoid division by zero)."""
        s = TrainingSummary(training_loss_start=0.0, training_loss_end=1.0)
        assert s.loss_improvement == 0.0

    def test_loss_improvement_negative(self) -> None:
        """Loss increase → negative improvement."""
        s = TrainingSummary(training_loss_start=1.0, training_loss_end=2.0)
        assert s.loss_improvement < 0

    def test_mean_benchmark_improvement_with_baselines(self) -> None:
        """Average benchmark improvement across tasks with baselines."""
        s = TrainingSummary(
            benchmark_results={"hellaswag": 0.6, "mmlu": 0.5},
            baseline_results={"hellaswag": 0.5, "mmlu": 0.4},
        )
        # hellaswag: (0.6-0.5)/0.5 = 0.2,  mmlu: (0.5-0.4)/0.4 = 0.25
        expected = (0.2 + 0.25) / 2
        assert s.mean_benchmark_improvement == pytest.approx(expected)

    def test_mean_benchmark_improvement_falls_back_to_loss(self) -> None:
        """Without baselines, falls back to loss improvement."""
        s = TrainingSummary(
            training_loss_start=2.0,
            training_loss_end=1.0,
            benchmark_results={"hellaswag": 0.6},
            baseline_results={},
        )
        assert s.mean_benchmark_improvement == pytest.approx(0.5)

    def test_overall_grade(self) -> None:
        """Overall grade computed from mean benchmark improvement."""
        s = TrainingSummary(
            benchmark_results={"task": 0.75},
            baseline_results={"task": 0.5},
        )
        # improvement = 50% → A+
        grade, desc = s.overall_grade
        assert grade == "A+"

    def test_duration_formatted_hours(self) -> None:
        s = TrainingSummary(duration_seconds=7265.0)
        assert "2h" in s.duration_formatted
        assert "1m" in s.duration_formatted

    def test_duration_formatted_minutes(self) -> None:
        s = TrainingSummary(duration_seconds=125.0)
        assert s.duration_formatted == "2m 5s"

    def test_duration_formatted_seconds_only(self) -> None:
        s = TrainingSummary(duration_seconds=45.0)
        assert s.duration_formatted == "45s"


# ===================================================================
# Strengths, Watch Areas, Recommendations
# ===================================================================


class TestTrainingSummaryAnalysis:
    """Test strengths, watch areas, and recommendations."""

    def test_strengths_with_improvements(self) -> None:
        s = TrainingSummary(
            training_loss_start=3.0,
            training_loss_end=1.0,
            benchmark_results={"hellaswag": 0.7},
            baseline_results={"hellaswag": 0.5},
        )
        strengths = s.strengths()
        assert len(strengths) >= 1
        assert any("hellaswag" in st.lower() for st in strengths)

    def test_strengths_fallback(self) -> None:
        """No improvements → default strength message."""
        s = TrainingSummary()
        strengths = s.strengths()
        assert len(strengths) >= 1
        assert any("completed" in st.lower() for st in strengths)

    def test_watch_areas_regression(self) -> None:
        """Regression detected as watch area."""
        s = TrainingSummary(
            benchmark_results={"task": 0.3},
            baseline_results={"task": 0.5},
        )
        watch = s.watch_areas()
        assert len(watch) >= 1
        assert any("regression" in w.lower() for w in watch)

    def test_watch_areas_loss_increase(self) -> None:
        """Loss increase flagged as watch area."""
        s = TrainingSummary(training_loss_start=1.0, training_loss_end=2.0)
        watch = s.watch_areas()
        assert any("loss increased" in w.lower() for w in watch)

    def test_watch_areas_empty_when_good(self) -> None:
        """No watch areas when everything is fine."""
        s = TrainingSummary(
            training_loss_start=2.0,
            training_loss_end=1.0,
            benchmark_results={"task": 0.7},
            baseline_results={"task": 0.5},
        )
        assert s.watch_areas() == []

    def test_recommendations_poor_grade(self) -> None:
        """Poor grade generates improvement recommendations."""
        s = TrainingSummary(
            benchmark_results={"task": 0.48},
            baseline_results={"task": 0.50},
        )
        recs = s.recommendations()
        assert len(recs) >= 1

    def test_recommendations_good_grade(self) -> None:
        """Good grade recommends deployment."""
        s = TrainingSummary(
            training_loss_start=3.0,
            training_loss_end=1.0,
            benchmark_results={"task": 0.9},
            baseline_results={"task": 0.5},
            training_method="lora",
        )
        recs = s.recommendations()
        assert any("deploy" in r.lower() or "gguf" in r.lower() for r in recs)


# ===================================================================
# Display function (smoke test)
# ===================================================================


class TestDisplayTrainingSummary:
    """Smoke tests for display functions."""

    def test_plain_display(self) -> None:
        """Plain text display doesn't crash."""
        from llm_forge.evaluation.training_summary import _display_plain

        s = TrainingSummary(
            model_name="test-model",
            training_loss_start=2.5,
            training_loss_end=1.2,
            duration_seconds=300,
            benchmark_results={"hellaswag": 0.6},
            baseline_results={"hellaswag": 0.5},
        )
        # Should not raise
        _display_plain(s)
