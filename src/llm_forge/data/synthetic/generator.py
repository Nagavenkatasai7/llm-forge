"""Synthetic data generation pipeline for creating training data from source documents."""

from __future__ import annotations

import json
import logging
import random

from datasets import Dataset

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION_TEMPLATES = [
    "Summarize the following text:\n\n{context}",
    "What are the key points discussed in this passage?\n\n{context}",
    "Based on the following information, answer any questions a reader might have:\n\n{context}",
    "Extract the main facts from this text:\n\n{context}",
    "Explain the concepts described in the following passage:\n\n{context}",
    "Create a Q&A pair based on this information:\n\n{context}",
    "What can be inferred from the following text?\n\n{context}",
    "Provide a detailed analysis of this passage:\n\n{context}",
]

DIFFICULTY_TIERS = {
    "L1_factual": [
        "What is {topic}?",
        "Define {topic} based on the provided context.",
        "List the key facts about {topic} from the text.",
    ],
    "L2_inferential": [
        "What can be inferred about {topic} from the passage?",
        "How does {topic} relate to the broader context described?",
        "What are the implications of {topic} based on the text?",
    ],
    "L3_evaluative": [
        "Evaluate the strengths and limitations of {topic} as described.",
        "Compare and contrast the different aspects of {topic} mentioned.",
        "What is the significance of {topic} in the given context?",
    ],
    "L4_counterfactual": [
        "What would change if {topic} were different? Explain based on the context.",
        "How would the outcome differ without {topic}?",
        "Propose an alternative to {topic} based on the information provided.",
    ],
}


class SyntheticDataGenerator:
    """Generate synthetic instruction-response pairs from source documents."""

    def __init__(
        self,
        teacher_model: str | None = None,
        temperature_range: tuple[float, float] = (0.3, 0.9),
        max_pairs_per_chunk: int = 3,
        seed: int = 42,
    ):
        self.teacher_model = teacher_model
        self.temperature_range = temperature_range
        self.max_pairs_per_chunk = max_pairs_per_chunk
        self.rng = random.Random(seed)
        self._pipeline = None

    def generate_from_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        num_samples: int | None = None,
    ) -> Dataset:
        """Generate synthetic instruction-response pairs from a text dataset."""
        logger.info(f"Generating synthetic data from {len(dataset)} documents")

        all_pairs: list[dict[str, str]] = []

        for _idx, example in enumerate(dataset):
            text = example.get(text_field, "")
            if not text or len(text.strip()) < 50:
                continue

            pairs = self._generate_pairs_from_text(text)
            all_pairs.extend(pairs)

            if num_samples and len(all_pairs) >= num_samples:
                all_pairs = all_pairs[:num_samples]
                break

        logger.info(f"Generated {len(all_pairs)} synthetic instruction-response pairs")
        return Dataset.from_list(all_pairs)

    def generate_from_chunks(
        self,
        chunks: list[str],
        topics: list[str] | None = None,
    ) -> Dataset:
        """Generate synthetic data from pre-chunked text with optional topic labels."""
        all_pairs: list[dict[str, str]] = []

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue

            topic = topics[i] if topics and i < len(topics) else self._extract_topic(chunk)
            pairs = self._generate_pairs_from_text(chunk, topic=topic)
            all_pairs.extend(pairs)

        logger.info(f"Generated {len(all_pairs)} pairs from {len(chunks)} chunks")
        return Dataset.from_list(all_pairs)

    def _generate_pairs_from_text(
        self,
        text: str,
        topic: str | None = None,
    ) -> list[dict[str, str]]:
        """Generate instruction-response pairs from a single text passage."""
        pairs = []

        if self._pipeline is not None:
            return self._generate_with_model(text, topic)

        templates = self.rng.sample(
            DEFAULT_INSTRUCTION_TEMPLATES,
            min(self.max_pairs_per_chunk, len(DEFAULT_INSTRUCTION_TEMPLATES)),
        )

        for template in templates:
            instruction = template.format(context=text[:1000])

            response = self._create_heuristic_response(text, template)
            if response:
                pairs.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": response,
                        "_difficulty": "L1_factual",
                        "_source": "heuristic",
                    }
                )

        if topic:
            tiered_pairs = self._generate_tiered_questions(text, topic)
            pairs.extend(tiered_pairs)

        return pairs

    def _generate_with_model(
        self,
        text: str,
        topic: str | None = None,
    ) -> list[dict[str, str]]:
        """Generate pairs using a teacher model (requires transformers pipeline)."""
        pairs = []

        prompt = (
            f"Given the following text, generate {self.max_pairs_per_chunk} diverse "
            f"instruction-response pairs for training a language model.\n\n"
            f"Text: {text[:2000]}\n\n"
            f"Generate pairs in JSON format with 'instruction' and 'response' keys."
        )

        try:
            temperature = self.rng.uniform(*self.temperature_range)
            outputs = self._pipeline(
                prompt,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
            )

            generated_text = outputs[0]["generated_text"]
            parsed = self._parse_generated_pairs(generated_text)
            pairs.extend(parsed)
        except Exception as e:
            logger.warning(f"Model generation failed, falling back to heuristic: {e}")
            pairs = self._generate_pairs_from_text.__wrapped__(self, text, topic)

        return pairs

    def _create_heuristic_response(self, text: str, template: str) -> str:
        """Create a heuristic response based on the text and template type."""
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

        if not sentences:
            return ""

        if "summarize" in template.lower():
            key_sentences = sentences[: min(3, len(sentences))]
            return ". ".join(key_sentences) + "."

        if "key points" in template.lower():
            points = sentences[: min(5, len(sentences))]
            return "\n".join(f"- {p.strip()}." for p in points)

        if "extract" in template.lower() or "facts" in template.lower():
            facts = sentences[: min(4, len(sentences))]
            return "\n".join(f"{i + 1}. {f.strip()}." for i, f in enumerate(facts))

        return ". ".join(sentences[: min(3, len(sentences))]) + "."

    def _generate_tiered_questions(
        self,
        text: str,
        topic: str,
    ) -> list[dict[str, str]]:
        """Generate difficulty-tiered Q&A pairs."""
        pairs = []
        [s.strip() for s in text.split(".") if s.strip()]

        for tier_name, templates in DIFFICULTY_TIERS.items():
            if not templates:
                continue

            template = self.rng.choice(templates)
            question = template.format(topic=topic)
            answer = self._create_heuristic_response(text, question)

            if answer:
                pairs.append(
                    {
                        "instruction": question,
                        "input": text[:500],
                        "output": answer,
                        "_difficulty": tier_name,
                        "_source": "tiered",
                    }
                )

        return pairs

    def _extract_topic(self, text: str) -> str:
        """Extract a simple topic from text using first sentence or noun phrase."""
        first_sentence = text.split(".")[0].strip()
        words = first_sentence.split()
        if len(words) > 5:
            return " ".join(words[:5])
        return first_sentence

    def _parse_generated_pairs(self, text: str) -> list[dict[str, str]]:
        """Parse model-generated JSON pairs from text output."""
        pairs = []

        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                for item in data:
                    if "instruction" in item and "response" in item:
                        pairs.append(
                            {
                                "instruction": item["instruction"],
                                "input": "",
                                "output": item["response"],
                                "_source": "model",
                            }
                        )
        except (json.JSONDecodeError, KeyError):
            pass

        return pairs

    def load_teacher_model(self, model_name: str) -> None:
        """Load a teacher model for synthetic data generation."""
        try:
            from transformers import pipeline

            logger.info(f"Loading teacher model: {model_name}")
            self._pipeline = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
            )
        except ImportError:
            raise ImportError("transformers is required for model-based generation")
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise
