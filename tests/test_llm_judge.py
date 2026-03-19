"""Tests for LLM-as-Judge evaluation module.

Covers:
- EvalConfig judge fields
- LLMJudge class interface
- Score/pairwise parsing logic
- Prompt template structure
- Data classes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from llm_forge.config.schema import EvalConfig, LLMForgeConfig

    _SCHEMA_AVAILABLE = True
except ImportError:
    _SCHEMA_AVAILABLE = False

try:
    from llm_forge.evaluation.llm_judge import (
        DEFAULT_CRITERIA,
        PAIRWISE_PROMPT,
        SINGLE_SCORE_PROMPT,
        JudgeResult,
        JudgeScore,
        LLMJudge,
        PairwiseResult,
    )

    _JUDGE_AVAILABLE = True
except ImportError:
    _JUDGE_AVAILABLE = False


# ===================================================================
# EvalConfig judge field tests
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestEvalConfigJudgeFields:
    """Verify judge-related fields in EvalConfig."""

    def test_llm_judge_default_false(self):
        cfg = EvalConfig()
        assert cfg.llm_judge is False

    def test_judge_model_default_none(self):
        cfg = EvalConfig()
        assert cfg.judge_model is None

    def test_judge_criteria_defaults(self):
        cfg = EvalConfig()
        assert cfg.judge_criteria == ["helpfulness", "coherence"]

    def test_judge_samples_default(self):
        cfg = EvalConfig()
        assert cfg.judge_samples == 50

    def test_judge_enabled(self):
        cfg = EvalConfig(llm_judge=True, judge_model="gpt2")
        assert cfg.llm_judge is True
        assert cfg.judge_model == "gpt2"

    def test_judge_custom_criteria(self):
        cfg = EvalConfig(judge_criteria=["accuracy", "relevance", "safety"])
        assert len(cfg.judge_criteria) == 3

    def test_judge_in_master_config(self):
        cfg = LLMForgeConfig.model_validate(
            {
                "model": {"name": "test/model"},
                "data": {"train_path": "test.jsonl"},
                "training": {"output_dir": "/tmp/test"},
                "evaluation": {
                    "llm_judge": True,
                    "judge_model": "meta-llama/Llama-2-7b",
                    "judge_samples": 25,
                },
            }
        )
        assert cfg.evaluation.llm_judge is True
        assert cfg.evaluation.judge_model == "meta-llama/Llama-2-7b"
        assert cfg.evaluation.judge_samples == 25

    def test_yaml_roundtrip(self):
        raw = {
            "model": {"name": "test/model"},
            "data": {"train_path": "test.jsonl"},
            "training": {"output_dir": "/tmp/test"},
            "evaluation": {
                "llm_judge": True,
                "judge_model": "gpt2",
                "judge_criteria": ["helpfulness"],
                "judge_samples": 10,
            },
        }
        cfg = LLMForgeConfig.model_validate(raw)
        dumped = cfg.model_dump(mode="json")
        reloaded = yaml.safe_load(yaml.dump(dumped))
        cfg2 = LLMForgeConfig.model_validate(reloaded)
        assert cfg2.evaluation.llm_judge is True
        assert cfg2.evaluation.judge_model == "gpt2"


# ===================================================================
# LLMJudge class tests
# ===================================================================


@pytest.mark.skipif(not _JUDGE_AVAILABLE, reason="judge module deps missing")
class TestLLMJudgeInterface:
    """Test LLMJudge class interface."""

    def test_class_exists(self):
        assert LLMJudge is not None

    def test_instantiation_defaults(self):
        judge = LLMJudge()
        assert judge.judge_model_name is None
        assert judge.max_new_tokens == 256
        assert len(judge.criteria) > 0

    def test_instantiation_custom(self):
        judge = LLMJudge(
            judge_model="gpt2",
            criteria={"custom": "Custom criterion"},
            max_new_tokens=128,
        )
        assert judge.judge_model_name == "gpt2"
        assert "custom" in judge.criteria

    def test_evaluate_method_exists(self):
        assert hasattr(LLMJudge, "evaluate")
        assert callable(LLMJudge.evaluate)

    def test_pairwise_compare_method_exists(self):
        assert hasattr(LLMJudge, "pairwise_compare")
        assert callable(LLMJudge.pairwise_compare)


# ===================================================================
# Score parsing tests
# ===================================================================


@pytest.mark.skipif(not _JUDGE_AVAILABLE, reason="judge module deps missing")
class TestScoreParsing:
    """Test JSON score parsing from judge output."""

    def test_parse_valid_json(self):
        text = '{"score": 8, "reasoning": "Good response"}'
        score, reasoning = LLMJudge._parse_score(text)
        assert score == 8
        assert "Good response" in reasoning

    def test_parse_json_in_text(self):
        text = 'Here is my evaluation: {"score": 7, "reasoning": "Decent"}'
        score, reasoning = LLMJudge._parse_score(text)
        assert score == 7

    def test_parse_score_clamped_to_range(self):
        text = '{"score": 15, "reasoning": "Off scale"}'
        score, _ = LLMJudge._parse_score(text)
        assert score == 10

    def test_parse_score_minimum(self):
        text = '{"score": 0, "reasoning": "Below min"}'
        score, _ = LLMJudge._parse_score(text)
        assert score == 1

    def test_parse_fallback_number(self):
        text = "I would rate this a 6 out of 10"
        score, _ = LLMJudge._parse_score(text)
        assert score == 6

    def test_parse_no_score_returns_default(self):
        text = "This response is quite good overall"
        score, _ = LLMJudge._parse_score(text)
        assert score == 5  # default middle score

    def test_parse_pairwise_a_wins(self):
        text = '{"winner": "A", "reasoning": "More detailed"}'
        winner, _ = LLMJudge._parse_pairwise(text)
        assert winner == "A"

    def test_parse_pairwise_b_wins(self):
        text = '{"winner": "B", "reasoning": "More accurate"}'
        winner, _ = LLMJudge._parse_pairwise(text)
        assert winner == "B"

    def test_parse_pairwise_tie(self):
        text = '{"winner": "tie", "reasoning": "Equal quality"}'
        winner, _ = LLMJudge._parse_pairwise(text)
        assert winner == "TIE"

    def test_parse_pairwise_fallback_a(self):
        text = "Response A is clearly better because..."
        winner, _ = LLMJudge._parse_pairwise(text)
        assert winner == "A"

    def test_parse_pairwise_fallback_b(self):
        text = 'I prefer Response B, it is "B" quality'
        winner, _ = LLMJudge._parse_pairwise(text)
        assert winner == "B"


# ===================================================================
# Data class tests
# ===================================================================


@pytest.mark.skipif(not _JUDGE_AVAILABLE, reason="judge module deps missing")
class TestJudgeDataClasses:
    """Test data classes for judge results."""

    def test_judge_score(self):
        js = JudgeScore(
            instruction="What is 2+2?",
            response="4",
            criterion="accuracy",
            score=10,
            reasoning="Correct",
        )
        assert js.score == 10
        assert js.criterion == "accuracy"

    def test_judge_result_defaults(self):
        jr = JudgeResult()
        assert jr.scores == []
        assert jr.mean_scores == {}
        assert jr.num_evaluated == 0

    def test_pairwise_result(self):
        pr = PairwiseResult(
            instruction="Explain gravity",
            response_a="Force of attraction",
            response_b="Things fall down",
            winner="A",
            reasoning="More scientific",
        )
        assert pr.winner == "A"

    def test_default_criteria_exists(self):
        assert "helpfulness" in DEFAULT_CRITERIA
        assert "accuracy" in DEFAULT_CRITERIA
        assert "coherence" in DEFAULT_CRITERIA
        assert "relevance" in DEFAULT_CRITERIA


# ===================================================================
# Prompt template tests
# ===================================================================


@pytest.mark.skipif(not _JUDGE_AVAILABLE, reason="judge module deps missing")
class TestPromptTemplates:
    """Verify prompt templates have correct placeholders."""

    def test_single_score_prompt_placeholders(self):
        assert "{criteria}" in SINGLE_SCORE_PROMPT
        assert "{instruction}" in SINGLE_SCORE_PROMPT
        assert "{response}" in SINGLE_SCORE_PROMPT

    def test_single_score_prompt_formats(self):
        filled = SINGLE_SCORE_PROMPT.format(
            criteria="accuracy",
            instruction="What is 2+2?",
            response="4",
        )
        assert "accuracy" in filled
        assert "What is 2+2?" in filled
        assert "4" in filled

    def test_pairwise_prompt_placeholders(self):
        assert "{criteria}" in PAIRWISE_PROMPT
        assert "{instruction}" in PAIRWISE_PROMPT
        assert "{response_a}" in PAIRWISE_PROMPT
        assert "{response_b}" in PAIRWISE_PROMPT


# ===================================================================
# Existing configs still valid
# ===================================================================


@pytest.mark.skipif(not _SCHEMA_AVAILABLE, reason="schema deps missing")
class TestExistingConfigsWithJudge:
    """Ensure adding judge fields didn't break configs."""

    def test_project_configs_validate(self):
        configs_dir = Path(__file__).resolve().parents[1] / "configs"
        if not configs_dir.exists():
            pytest.skip("configs directory not found")
        for c in configs_dir.glob("*.yaml"):
            with open(c) as f:
                raw = yaml.safe_load(f)
            LLMForgeConfig.model_validate(raw)
