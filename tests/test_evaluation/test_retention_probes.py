"""Tests for knowledge retention probes.

Knowledge retention probes detect catastrophic forgetting by running
100 factual multiple-choice questions before and after fine-tuning.
"""

from __future__ import annotations

import pytest

try:
    from llm_forge.evaluation.retention_probes import (
        RETENTION_PROBES,
        KnowledgeRetentionProber,
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
    reason="llm_forge.evaluation.retention_probes not importable",
)


# ===================================================================
# Probe Bank Tests
# ===================================================================


class TestProbeBank:
    """Verify the probe question bank structure."""

    def test_probe_count(self) -> None:
        """Should have exactly 100 probes."""
        assert len(RETENTION_PROBES) == 100

    def test_probe_structure(self) -> None:
        """Each probe has required keys."""
        for probe in RETENTION_PROBES:
            assert "id" in probe
            assert "domain" in probe
            assert "question" in probe
            assert "choices" in probe
            assert "answer" in probe

    def test_four_choices(self) -> None:
        """Each probe has exactly 4 choices."""
        for probe in RETENTION_PROBES:
            assert len(probe["choices"]) == 4, (
                f"Probe {probe['id']} has {len(probe['choices'])} choices"
            )

    def test_answer_in_range(self) -> None:
        """Answer index is 0-3."""
        for probe in RETENTION_PROBES:
            assert 0 <= probe["answer"] <= 3, f"Probe {probe['id']} answer out of range"

    def test_ten_domains(self) -> None:
        """Should cover 10 distinct domains."""
        domains = {p["domain"] for p in RETENTION_PROBES}
        assert len(domains) == 10

    def test_ten_per_domain(self) -> None:
        """Each domain should have exactly 10 probes."""
        from collections import Counter

        domain_counts = Counter(p["domain"] for p in RETENTION_PROBES)
        for domain, count in domain_counts.items():
            assert count == 10, f"Domain '{domain}' has {count} probes, expected 10"

    def test_unique_ids(self) -> None:
        """All probe IDs are unique."""
        ids = [p["id"] for p in RETENTION_PROBES]
        assert len(ids) == len(set(ids))

    def test_expected_domains(self) -> None:
        """Expected domain names are present."""
        domains = {p["domain"] for p in RETENTION_PROBES}
        expected = {
            "science",
            "math",
            "geography",
            "history",
            "literature",
            "technology",
            "biology",
            "physics",
            "chemistry",
            "general",
        }
        assert domains == expected


# ===================================================================
# Prober Class Tests
# ===================================================================


class TestKnowledgeRetentionProber:
    """Test the KnowledgeRetentionProber class methods."""

    def test_init_default_probes(self) -> None:
        prober = KnowledgeRetentionProber()
        assert len(prober.probes) == 100

    def test_init_custom_probes(self) -> None:
        custom = [RETENTION_PROBES[0], RETENTION_PROBES[1]]
        prober = KnowledgeRetentionProber(probes=custom)
        assert len(prober.probes) == 2

    def test_get_probes(self) -> None:
        prober = KnowledgeRetentionProber()
        probes = prober.get_probes()
        assert len(probes) == 100
        assert probes is not prober.probes  # returns a copy

    def test_get_probes_by_domain(self) -> None:
        prober = KnowledgeRetentionProber()
        science = prober.get_probes_by_domain("science")
        assert len(science) == 10
        assert all(p["domain"] == "science" for p in science)

    def test_get_probes_by_domain_empty(self) -> None:
        prober = KnowledgeRetentionProber()
        result = prober.get_probes_by_domain("nonexistent")
        assert result == []

    def test_format_prompt(self) -> None:
        prober = KnowledgeRetentionProber()
        probe = RETENTION_PROBES[0]  # "What is the chemical symbol for gold?"
        prompt = prober.format_prompt(probe)
        assert "Question:" in prompt
        assert "Answer:" in prompt
        assert "A." in prompt
        assert "B." in prompt
        assert "C." in prompt
        assert "D." in prompt
        assert probe["question"] in prompt

    def test_format_prompt_contains_choices(self) -> None:
        prober = KnowledgeRetentionProber()
        probe = RETENTION_PROBES[0]
        prompt = prober.format_prompt(probe)
        for choice in probe["choices"]:
            assert choice in prompt

    def test_domains_class_attribute(self) -> None:
        assert len(KnowledgeRetentionProber.DOMAINS) == 10


# ===================================================================
# compare_retention Tests
# ===================================================================


class TestCompareRetention:
    """Test the retention comparison logic."""

    def _make_results(self, correct_ids: set, total: int = 10) -> dict:
        """Build mock evaluation results."""
        per_probe = []
        for i in range(total):
            pid = f"test_{i:02d}"
            per_probe.append(
                {
                    "id": pid,
                    "domain": "test",
                    "correct": pid in correct_ids,
                    "predicted": 0 if pid in correct_ids else 1,
                    "expected": 0,
                    "confidence": 0.9,
                }
            )
        correct_count = len(correct_ids & {f"test_{i:02d}" for i in range(total)})
        return {
            "accuracy": correct_count / max(total, 1),
            "total": total,
            "correct": correct_count,
            "per_domain": {
                "test": {
                    "accuracy": correct_count / max(total, 1),
                    "correct": correct_count,
                    "total": total,
                }
            },
            "per_probe": per_probe,
        }

    def test_perfect_retention(self) -> None:
        """Same answers → 100% retention."""
        base = self._make_results({f"test_{i:02d}" for i in range(8)})
        ft = self._make_results({f"test_{i:02d}" for i in range(8)})
        comp = KnowledgeRetentionProber.compare_retention(base, ft)
        assert comp["retention_rate"] == 1.0
        assert comp["forgotten"] == 0
        assert comp["gained"] == 0

    def test_some_forgotten(self) -> None:
        """Base got 8 right, fine-tuned got 5 of those right → some forgotten."""
        base_correct = {f"test_{i:02d}" for i in range(8)}
        ft_correct = {f"test_{i:02d}" for i in range(5)}
        base = self._make_results(base_correct)
        ft = self._make_results(ft_correct)
        comp = KnowledgeRetentionProber.compare_retention(base, ft)
        assert comp["forgotten"] == 3
        assert comp["retention_rate"] == pytest.approx(5 / 8, abs=0.01)

    def test_gained_knowledge(self) -> None:
        """Fine-tuned model gets new questions right."""
        base_correct = {f"test_{i:02d}" for i in range(5)}
        ft_correct = {f"test_{i:02d}" for i in range(8)}
        base = self._make_results(base_correct)
        ft = self._make_results(ft_correct)
        comp = KnowledgeRetentionProber.compare_retention(base, ft)
        assert comp["gained"] == 3
        assert comp["net_change"] == 3

    def test_all_forgotten(self) -> None:
        """Base got all right, fine-tuned got none → 0% retention."""
        base_correct = {f"test_{i:02d}" for i in range(10)}
        ft_correct: set = set()
        base = self._make_results(base_correct)
        ft = self._make_results(ft_correct)
        comp = KnowledgeRetentionProber.compare_retention(base, ft)
        assert comp["retention_rate"] == 0.0
        assert comp["forgotten"] == 10

    def test_empty_results(self) -> None:
        """Empty results → graceful handling."""
        comp = KnowledgeRetentionProber.compare_retention({"per_probe": []}, {"per_probe": []})
        assert comp["retention_rate"] == 0.0
        assert comp["forgotten"] == 0

    def test_per_domain_delta(self) -> None:
        """Per-domain comparison includes delta."""
        base = self._make_results({f"test_{i:02d}" for i in range(8)})
        ft = self._make_results({f"test_{i:02d}" for i in range(5)})
        comp = KnowledgeRetentionProber.compare_retention(base, ft)
        assert "test" in comp["per_domain"]
        assert "delta" in comp["per_domain"]["test"]


# ===================================================================
# Config Schema Tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema not importable")
class TestRetentionConfig:
    """Verify retention probe config fields."""

    def test_retention_probes_default_false(self) -> None:
        cfg = EvalConfig()
        assert cfg.retention_probes is False

    def test_retention_threshold_default(self) -> None:
        cfg = EvalConfig()
        assert cfg.retention_threshold == 0.80

    def test_enable_retention_probes(self) -> None:
        cfg = EvalConfig(retention_probes=True)
        assert cfg.retention_probes is True

    def test_custom_threshold(self) -> None:
        cfg = EvalConfig(retention_threshold=0.90)
        assert cfg.retention_threshold == 0.90

    def test_full_config_validates(self) -> None:
        config = LLMForgeConfig(
            model={"name": "test-model"},
            data={"train_path": "test-data"},
            evaluation={"retention_probes": True, "retention_threshold": 0.85},
        )
        assert config.evaluation.retention_probes is True
        assert config.evaluation.retention_threshold == 0.85
