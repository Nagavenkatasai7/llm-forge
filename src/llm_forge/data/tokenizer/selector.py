"""Adaptive tokenizer selection based on domain and data characteristics."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

RECOMMENDED_TOKENIZERS = {
    "general": "meta-llama/Llama-3.2-1B",
    "code": "Qwen/Qwen2.5-Coder-1.5B",
    "multilingual": "Qwen/Qwen2.5-1.5B",
    "small": "openai-community/gpt2",
}


class TokenizerSelector:
    """Select the best tokenizer based on data characteristics."""

    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size

    def analyze_corpus(self, texts: list[str]) -> dict[str, Any]:
        """Analyze corpus characteristics for tokenizer selection."""
        sample = texts[: self.sample_size]

        all_words: list[str] = []
        total_chars = 0
        non_ascii_chars = 0
        code_indicators = 0

        for text in sample:
            words = text.split()
            all_words.extend(words)
            total_chars += len(text)
            non_ascii_chars += sum(1 for c in text if ord(c) > 127)
            code_indicators += sum(
                1
                for marker in ["{", "}", "def ", "class ", "import ", "//", "/*"]
                if marker in text
            )

        word_freq = Counter(all_words)
        unique_ratio = len(word_freq) / max(len(all_words), 1)
        non_ascii_ratio = non_ascii_chars / max(total_chars, 1)
        avg_word_len = sum(len(w) for w in all_words) / max(len(all_words), 1)

        return {
            "total_documents": len(sample),
            "total_words": len(all_words),
            "unique_words": len(word_freq),
            "vocabulary_diversity": unique_ratio,
            "non_ascii_ratio": non_ascii_ratio,
            "avg_word_length": avg_word_len,
            "code_indicator_count": code_indicators,
            "is_code_heavy": code_indicators > len(sample) * 0.3,
            "is_multilingual": non_ascii_ratio > 0.1,
        }

    def recommend_tokenizer(self, texts: list[str]) -> str:
        """Recommend a tokenizer based on corpus analysis."""
        analysis = self.analyze_corpus(texts)

        if analysis["is_code_heavy"]:
            recommendation = RECOMMENDED_TOKENIZERS["code"]
            logger.info(f"Code-heavy corpus detected, recommending: {recommendation}")
        elif analysis["is_multilingual"]:
            recommendation = RECOMMENDED_TOKENIZERS["multilingual"]
            logger.info(f"Multilingual corpus detected, recommending: {recommendation}")
        else:
            recommendation = RECOMMENDED_TOKENIZERS["general"]
            logger.info(f"General corpus detected, recommending: {recommendation}")

        return recommendation

    def train_custom_tokenizer(
        self,
        texts: list[str],
        vocab_size: int = 32000,
        output_path: str | None = None,
    ) -> Any:
        """Train a custom BPE tokenizer on the provided corpus."""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        logger.info(f"Training custom BPE tokenizer (vocab_size={vocab_size})")

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)

        if output_path:
            tokenizer.save(output_path)
            logger.info(f"Custom tokenizer saved to: {output_path}")

        return tokenizer

    def compare_tokenizers(
        self,
        texts: list[str],
        tokenizer_names: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compare tokenization efficiency across different tokenizers."""
        from transformers import AutoTokenizer

        results = {}
        sample = texts[: min(100, len(texts))]

        for name in tokenizer_names:
            try:
                tokenizer = AutoTokenizer.from_pretrained(name)
                total_tokens = 0
                total_chars = 0

                for text in sample:
                    tokens = tokenizer.encode(text)
                    total_tokens += len(tokens)
                    total_chars += len(text)

                avg_chars_per_token = total_chars / max(total_tokens, 1)
                results[name] = {
                    "total_tokens": total_tokens,
                    "avg_chars_per_token": round(avg_chars_per_token, 2),
                    "compression_ratio": round(total_chars / max(total_tokens, 1), 2),
                }
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {name}: {e}")
                results[name] = {"error": str(e)}

        return results
