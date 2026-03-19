"""Refusal-Aware Data Augmentation (R-Tuning).

Mixes 'I don't know' refusal examples into the training dataset so the
model learns to refuse questions that are beyond its knowledge rather
than hallucinate answers.

Reference: Zhang et al., "R-Tuning: Instructing Large Language Models
to Say 'I Don't Know'" (2023).
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REFUSAL_RESPONSES = [
    "I don't have enough information to answer that accurately.",
    "I'm not confident in my knowledge about this topic.",
    "I don't know the answer to that question.",
]


class RefusalAugmentor:
    """Augment a training dataset with refusal examples.

    Parameters
    ----------
    refusal_ratio : float
        Fraction of the dataset to convert to refusal examples.
    refusal_responses : list[str]
        Pool of refusal response templates to sample from.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        refusal_ratio: float = 0.15,
        refusal_responses: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.refusal_ratio = refusal_ratio
        self.refusal_responses = refusal_responses or DEFAULT_REFUSAL_RESPONSES
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment_dataset(
        self,
        dataset: Any,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> Any:
        """Augment the training dataset with refusal examples.

        If model and tokenizer are provided, uses model-based gap detection
        to identify which questions to convert to refusals.  Otherwise,
        randomly selects samples.

        Parameters
        ----------
        dataset : datasets.Dataset
            The training dataset (must have a 'text' column with formatted prompts).
        model : optional
            Fine-tuned model for knowledge gap detection.
        tokenizer : optional
            Tokenizer matching the model.

        Returns
        -------
        datasets.Dataset
            Augmented dataset with refusal examples mixed in.
        """
        import datasets as hf_datasets

        num_total = len(dataset)
        num_refusals = int(num_total * self.refusal_ratio)

        if num_refusals == 0:
            logger.info("Refusal ratio too small for dataset size, skipping augmentation")
            return dataset

        rng = random.Random(self.seed)

        if model is not None and tokenizer is not None:
            gap_indices = self.identify_knowledge_gaps(
                model, tokenizer, dataset, max_samples=min(num_total, num_refusals * 3)
            )
            # Use gap indices if enough were found, pad with random if needed
            selected_indices = gap_indices[:num_refusals]
            if len(selected_indices) < num_refusals:
                remaining = set(range(num_total)) - set(selected_indices)
                extra = rng.sample(
                    list(remaining),
                    min(num_refusals - len(selected_indices), len(remaining)),
                )
                selected_indices.extend(extra)
        else:
            selected_indices = rng.sample(range(num_total), num_refusals)

        logger.info(
            "Generating %d refusal examples (%.1f%% of %d samples)",
            len(selected_indices),
            100.0 * len(selected_indices) / num_total,
            num_total,
        )

        # Build refusal examples
        refusal_set = set(selected_indices)
        column_names = dataset.column_names

        if "text" in column_names:
            # Text column with formatted prompts — replace the response portion
            new_texts = []
            for i in range(num_total):
                text = dataset[i]["text"]
                if i in refusal_set:
                    text = self._replace_response_in_text(text, rng)
                new_texts.append(text)

            new_dataset = hf_datasets.Dataset.from_dict({"text": new_texts})

            # Preserve other columns
            for col in column_names:
                if col != "text":
                    new_dataset = new_dataset.add_column(col, dataset[col])

            return new_dataset
        else:
            # Structured format — look for output/response field
            output_field = self._detect_output_field(column_names)
            if output_field is None:
                logger.warning(
                    "Could not detect output field in columns %s, returning dataset unchanged",
                    column_names,
                )
                return dataset

            data_dict: dict[str, list] = {col: list(dataset[col]) for col in column_names}
            for i in selected_indices:
                data_dict[output_field][i] = rng.choice(self.refusal_responses)

            return hf_datasets.Dataset.from_dict(data_dict)

    def identify_knowledge_gaps(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        max_samples: int = 500,
    ) -> list[int]:
        """Identify questions the model gets wrong (knowledge gaps).

        Runs the model on a subset of training questions and checks
        whether the generated answer has low confidence (high perplexity).
        Questions with high perplexity are candidates for refusal labels.

        Returns
        -------
        list[int]
            Indices into the dataset of identified knowledge gaps,
            sorted by descending perplexity.
        """
        import torch

        model.eval()
        device = next(model.parameters()).device

        indices_and_perplexities: list[tuple[int, float]] = []
        num_to_check = min(len(dataset), max_samples)

        for i in range(num_to_check):
            text = self._get_sample_text(dataset, i)
            if not text:
                continue

            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False,
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()

                perplexity = float(torch.exp(torch.tensor(loss)))
                indices_and_perplexities.append((i, perplexity))
            except Exception as exc:
                logger.debug("Skipping sample %d: %s", i, exc)
                continue

        # Sort by perplexity (highest first = least confident)
        indices_and_perplexities.sort(key=lambda x: x[1], reverse=True)

        gap_indices = [idx for idx, _ in indices_and_perplexities]
        logger.info(
            "Identified %d knowledge gap candidates (checked %d samples, top perplexity=%.1f)",
            len(gap_indices),
            num_to_check,
            indices_and_perplexities[0][1] if indices_and_perplexities else 0.0,
        )

        return gap_indices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _replace_response_in_text(self, text: str, rng: random.Random) -> str:
        """Replace the response portion of a formatted prompt with a refusal."""
        refusal = rng.choice(self.refusal_responses)

        # Try common prompt formats
        for marker in ["### Response:", "### Assistant:", "<|assistant|>", "ASSISTANT:"]:
            if marker in text:
                prefix = text[: text.index(marker) + len(marker)]
                return f"{prefix}\n{refusal}"

        # Fallback: if there's a clear instruction/response split, use it
        for marker in ["\n\n", "\nA: ", "\nAnswer: "]:
            if marker in text:
                parts = text.split(marker, 1)
                return f"{parts[0]}{marker}{refusal}"

        # Last resort: just append refusal
        return f"{text}\n{refusal}"

    def _detect_output_field(self, column_names: list[str]) -> str | None:
        """Detect which column contains the model output/response."""
        for candidate in ["output", "response", "answer", "completion", "target"]:
            if candidate in column_names:
                return candidate
        return None

    def _get_sample_text(self, dataset: Any, index: int) -> str:
        """Extract text from a dataset sample."""
        item = dataset[index]
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ["text", "input", "instruction", "question", "prompt"]:
                if key in item and item[key]:
                    return str(item[key])
        return ""
