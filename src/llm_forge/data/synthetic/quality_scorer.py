"""LLM-as-judge quality scoring for synthetic data."""

from __future__ import annotations

import logging
import re

from datasets import Dataset

logger = logging.getLogger(__name__)

QUALITY_RUBRIC = """Rate the quality of the following instruction-response pair on a scale of 1-5:

**Instruction:** {instruction}
**Response:** {response}

Scoring criteria:
1 - Irrelevant, incorrect, or incoherent
2 - Partially relevant but with major issues
3 - Adequate but could be improved
4 - Good quality, relevant and mostly accurate
5 - Excellent, comprehensive, accurate, and well-structured

Provide your rating as a single number (1-5) followed by a brief justification.
Rating:"""


class QualityScorer:
    """Score synthetic data quality using heuristic or model-based evaluation."""

    def __init__(
        self,
        method: str = "heuristic",
        model_name: str | None = None,
        min_score: float = 3.0,
    ):
        self.method = method
        self.model_name = model_name
        self.min_score = min_score
        self._pipeline = None

    def score_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        output_field: str = "output",
    ) -> Dataset:
        """Score all examples in a dataset and add quality scores."""
        logger.info(f"Scoring {len(dataset)} examples using {self.method} method")

        def score_example(example: dict) -> dict:
            instruction = example.get(instruction_field, "")
            response = example.get(output_field, "")
            score = self.score_pair(instruction, response)
            example["_quality_score"] = score
            return example

        scored = dataset.map(score_example)
        return scored

    def filter_by_quality(
        self,
        dataset: Dataset,
        min_score: float | None = None,
    ) -> Dataset:
        """Filter dataset to keep only high-quality examples."""
        threshold = min_score or self.min_score

        if "_quality_score" not in dataset.column_names:
            dataset = self.score_dataset(dataset)

        original_count = len(dataset)
        filtered = dataset.filter(lambda x: x.get("_quality_score", 0) >= threshold)
        logger.info(
            f"Quality filter: {original_count} -> {len(filtered)} examples (threshold={threshold})"
        )
        return filtered

    def score_pair(self, instruction: str, response: str) -> float:
        """Score a single instruction-response pair."""
        if self.method == "model" and self._pipeline is not None:
            return self._score_with_model(instruction, response)
        return self._score_heuristic(instruction, response)

    def _score_heuristic(self, instruction: str, response: str) -> float:
        """Heuristic quality scoring based on text properties."""
        score = 0.0

        if not response or not response.strip():
            return 1.0

        resp_words = response.split()
        resp_len = len(resp_words)

        if resp_len >= 10:
            score += 1.0
        elif resp_len >= 5:
            score += 0.5

        if resp_len >= 20:
            score += 0.5

        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if len(sentences) >= 2:
            score += 0.5

        if any(response.strip().endswith(p) for p in ".!?"):
            score += 0.5

        instr_words = set(instruction.lower().split())
        resp_words_set = set(response.lower().split())
        overlap = len(instr_words & resp_words_set) / max(len(instr_words), 1)
        if overlap > 0.1:
            score += 0.5

        unique_words = len(set(response.lower().split()))
        total_words = max(len(response.split()), 1)
        vocab_diversity = unique_words / total_words
        if vocab_diversity > 0.5:
            score += 0.5

        if response.strip() == instruction.strip():
            score = max(score - 2.0, 1.0)

        has_structure = any(
            marker in response
            for marker in ["\n-", "\n*", "\n1.", "\n2.", "First,", "Second,", "Finally,"]
        )
        if has_structure:
            score += 0.5

        return min(max(round(score, 1), 1.0), 5.0)

    def _score_with_model(self, instruction: str, response: str) -> float:
        """Score using a language model as judge."""
        prompt = QUALITY_RUBRIC.format(instruction=instruction, response=response)

        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
            )
            generated = outputs[0]["generated_text"]

            numbers = re.findall(r"[1-5]", generated)
            if numbers:
                return float(numbers[0])
        except Exception as e:
            logger.warning(f"Model scoring failed: {e}")

        return self._score_heuristic(instruction, response)

    def load_judge_model(self, model_name: str) -> None:
        """Load a model to use as quality judge."""
        try:
            from transformers import pipeline

            logger.info(f"Loading judge model: {model_name}")
            self._pipeline = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
            )
            self.method = "model"
        except ImportError:
            raise ImportError("transformers is required for model-based scoring")
