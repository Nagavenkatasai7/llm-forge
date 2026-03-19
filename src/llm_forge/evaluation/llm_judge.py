"""LLM-as-Judge evaluation module.

Uses a language model to evaluate generated responses on criteria such as
helpfulness, accuracy, relevance, and coherence.  Supports both local models
(via transformers) and external API judges.

Reference: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
(Zheng et al., NeurIPS 2023, arXiv:2306.05685).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("llm_forge.evaluation.llm_judge")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SINGLE_SCORE_PROMPT = """You are an expert evaluator. Score the following response on a scale of 1-10 based on the criteria below.

**Criteria:** {criteria}

**Question/Instruction:**
{instruction}

**Response:**
{response}

Provide your evaluation in the following JSON format:
{{"score": <integer 1-10>, "reasoning": "<brief explanation>"}}

Your evaluation:"""

PAIRWISE_PROMPT = """You are an expert evaluator. Compare the two responses below and determine which is better based on the criteria.

**Criteria:** {criteria}

**Question/Instruction:**
{instruction}

**Response A:**
{response_a}

**Response B:**
{response_b}

Which response is better? Respond in the following JSON format:
{{"winner": "A" or "B" or "tie", "reasoning": "<brief explanation>"}}

Your evaluation:"""

# Default evaluation criteria
DEFAULT_CRITERIA = {
    "helpfulness": "How helpful and informative is the response? Does it address the user's question?",
    "accuracy": "Is the information in the response factually correct?",
    "coherence": "Is the response well-structured, logical, and easy to understand?",
    "relevance": "Does the response stay on topic and address what was asked?",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class JudgeScore:
    """Score from a single LLM-as-Judge evaluation."""

    instruction: str
    response: str
    criterion: str
    score: int = 0
    reasoning: str = ""
    raw_output: str = ""


@dataclass
class JudgeResult:
    """Aggregate result of LLM-as-Judge evaluation."""

    scores: list[JudgeScore] = field(default_factory=list)
    mean_scores: dict[str, float] = field(default_factory=dict)
    num_evaluated: int = 0


@dataclass
class PairwiseResult:
    """Result of pairwise comparison."""

    instruction: str = ""
    response_a: str = ""
    response_b: str = ""
    winner: str = ""  # "A", "B", or "tie"
    reasoning: str = ""


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Evaluate model outputs using an LLM as judge.

    Parameters
    ----------
    judge_model : str
        Path or HuggingFace ID of the judge model.
    criteria : dict[str, str], optional
        Evaluation criteria mapping name -> description.
        Defaults to helpfulness, accuracy, coherence, relevance.
    max_new_tokens : int
        Maximum tokens for the judge's response.
    """

    def __init__(
        self,
        judge_model: str | None = None,
        criteria: dict[str, str] | None = None,
        max_new_tokens: int = 256,
    ):
        self.judge_model_name = judge_model
        self.criteria = criteria or DEFAULT_CRITERIA
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._pipeline = None

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the judge model."""
        if self._pipeline is not None:
            return

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for LLM-as-Judge. Install with: pip install transformers"
            )

        if self.judge_model_name is None:
            raise ValueError("judge_model must be specified for local evaluation")

        logger.info("Loading judge model: %s", self.judge_model_name)
        self._pipeline = pipeline(
            "text-generation",
            model=self.judge_model_name,
            max_new_tokens=self.max_new_tokens,
            torch_dtype="auto",
            device_map="auto",
        )

    def _generate(self, prompt: str) -> str:
        """Generate a response from the judge model."""
        self._ensure_model_loaded()
        result = self._pipeline(prompt, return_full_text=False)
        return result[0]["generated_text"].strip()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def evaluate(
        self,
        instructions: list[str],
        responses: list[str],
        criteria: list[str] | None = None,
    ) -> JudgeResult:
        """Score responses using the judge model.

        Parameters
        ----------
        instructions : list[str]
            Questions or instructions.
        responses : list[str]
            Model-generated responses.
        criteria : list[str], optional
            Which criteria to evaluate. If None, uses all configured criteria.

        Returns
        -------
        JudgeResult
            Scores per sample per criterion plus aggregates.
        """
        assert len(instructions) == len(responses), (
            "instructions and responses must have the same length"
        )

        eval_criteria = criteria or list(self.criteria.keys())
        all_scores: list[JudgeScore] = []

        for inst, resp in zip(instructions, responses, strict=False):
            for crit_name in eval_criteria:
                crit_desc = self.criteria.get(crit_name, crit_name)
                prompt = SINGLE_SCORE_PROMPT.format(
                    criteria=crit_desc,
                    instruction=inst,
                    response=resp,
                )

                raw_output = self._generate(prompt)
                score, reasoning = self._parse_score(raw_output)

                all_scores.append(
                    JudgeScore(
                        instruction=inst,
                        response=resp,
                        criterion=crit_name,
                        score=score,
                        reasoning=reasoning,
                        raw_output=raw_output,
                    )
                )

        # Compute mean scores per criterion
        mean_scores: dict[str, float] = {}
        for crit_name in eval_criteria:
            crit_scores = [s.score for s in all_scores if s.criterion == crit_name]
            mean_scores[crit_name] = sum(crit_scores) / len(crit_scores) if crit_scores else 0.0

        return JudgeResult(
            scores=all_scores,
            mean_scores=mean_scores,
            num_evaluated=len(instructions),
        )

    def pairwise_compare(
        self,
        instruction: str,
        response_a: str,
        response_b: str,
        criterion: str = "helpfulness",
    ) -> PairwiseResult:
        """Compare two responses head-to-head.

        Parameters
        ----------
        instruction : str
            The shared instruction/question.
        response_a : str
            First response.
        response_b : str
            Second response.
        criterion : str
            Evaluation criterion name.

        Returns
        -------
        PairwiseResult
            Which response won and why.
        """
        crit_desc = self.criteria.get(criterion, criterion)
        prompt = PAIRWISE_PROMPT.format(
            criteria=crit_desc,
            instruction=instruction,
            response_a=response_a,
            response_b=response_b,
        )

        raw_output = self._generate(prompt)
        winner, reasoning = self._parse_pairwise(raw_output)

        return PairwiseResult(
            instruction=instruction,
            response_a=response_a,
            response_b=response_b,
            winner=winner,
            reasoning=reasoning,
        )

    # -----------------------------------------------------------------
    # Parsing helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_score(text: str) -> tuple[int, str]:
        """Parse a score and reasoning from judge output."""
        # Try JSON parsing first
        try:
            # Find JSON-like content
            match = re.search(r"\{[^}]+\}", text)
            if match:
                data = json.loads(match.group())
                score = int(data.get("score", 0))
                score = max(1, min(10, score))
                reasoning = str(data.get("reasoning", ""))
                return score, reasoning
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: look for a number
        numbers = re.findall(r"\b(\d+)\b", text)
        for n in numbers:
            val = int(n)
            if 1 <= val <= 10:
                return val, text

        return 5, text  # default middle score

    @staticmethod
    def _parse_pairwise(text: str) -> tuple[str, str]:
        """Parse a pairwise comparison result."""
        try:
            match = re.search(r"\{[^}]+\}", text)
            if match:
                data = json.loads(match.group())
                winner = str(data.get("winner", "tie")).upper()
                if winner not in ("A", "B", "TIE"):
                    winner = "tie"
                reasoning = str(data.get("reasoning", ""))
                return winner, reasoning
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback
        text_upper = text.upper()
        if "RESPONSE A" in text_upper or '"A"' in text_upper:
            return "A", text
        if "RESPONSE B" in text_upper or '"B"' in text_upper:
            return "B", text
        return "tie", text
