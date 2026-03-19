"""Structural redundancy elimination for token optimization."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)


class TokenOptimizer:
    """Reduce token count by eliminating structural redundancy in data."""

    def __init__(self, target_format: str = "auto"):
        self.target_format = target_format

    def optimize_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
    ) -> Dataset:
        """Optimize token usage across a dataset by detecting and converting structured data."""
        logger.info(f"Optimizing token usage for {len(dataset)} examples")

        optimized_records = []
        savings_total = 0

        for example in dataset:
            text = example.get(text_field, "")
            optimized, savings = self._optimize_text(text)
            savings_total += savings
            record = dict(example)
            record[text_field] = optimized
            optimized_records.append(record)

        if savings_total > 0:
            logger.info(f"Token optimization saved ~{savings_total} characters total")

        return Dataset.from_list(optimized_records)

    def _optimize_text(self, text: str) -> tuple[str, int]:
        """Optimize a single text, returning (optimized_text, chars_saved)."""
        original_len = len(text)

        detected = self._detect_format(text)
        if detected == "json_array":
            optimized = self._json_array_to_csv(text)
        elif detected == "json_repeated_keys":
            optimized = self._json_to_compact(text)
        elif detected == "verbose_json":
            optimized = self._compact_json(text)
        else:
            optimized = text

        savings = original_len - len(optimized)
        return optimized, max(savings, 0)

    def _detect_format(self, text: str) -> str:
        """Detect the structural format of text content."""
        stripped = text.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                data = json.loads(stripped)
                if isinstance(data, list) and len(data) > 1 and isinstance(data[0], dict):
                    keys = set(data[0].keys())
                    if all(isinstance(item, dict) and set(item.keys()) == keys for item in data):
                        return "json_array"
                    return "json_repeated_keys"
            except json.JSONDecodeError:
                pass

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
                if len(stripped) > 200:
                    return "verbose_json"
            except json.JSONDecodeError:
                pass

        return "text"

    def _json_array_to_csv(self, text: str) -> str:
        """Convert JSON array with uniform keys to CSV format (schema-once)."""
        try:
            data = json.loads(text)
            if not data or not isinstance(data[0], dict):
                return text

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            result = output.getvalue()

            if len(result) < len(text):
                return result
            return text

        except (json.JSONDecodeError, Exception):
            return text

    def _json_to_compact(self, text: str) -> str:
        """Convert JSON with repeated keys to compact representation."""
        try:
            data = json.loads(text)
            return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        except json.JSONDecodeError:
            return text

    def _compact_json(self, text: str) -> str:
        """Remove unnecessary whitespace from JSON."""
        try:
            data = json.loads(text)
            return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        except json.JSONDecodeError:
            return text

    def estimate_savings(
        self,
        dataset: Dataset,
        text_field: str = "text",
        tokenizer: Any | None = None,
    ) -> dict[str, Any]:
        """Estimate potential token savings without modifying data."""
        total_original_chars = 0
        total_optimized_chars = 0
        format_counts: dict[str, int] = {}

        sample = list(dataset.select(range(min(100, len(dataset)))))

        for example in sample:
            text = example.get(text_field, "")
            total_original_chars += len(text)

            detected = self._detect_format(text)
            format_counts[detected] = format_counts.get(detected, 0) + 1

            optimized, _ = self._optimize_text(text)
            total_optimized_chars += len(optimized)

        char_savings = total_original_chars - total_optimized_chars
        char_savings_pct = (char_savings / max(total_original_chars, 1)) * 100

        result = {
            "sample_size": len(sample),
            "original_chars": total_original_chars,
            "optimized_chars": total_optimized_chars,
            "char_savings": char_savings,
            "char_savings_pct": round(char_savings_pct, 1),
            "format_distribution": format_counts,
        }

        if tokenizer is not None:
            original_tokens = sum(len(tokenizer.encode(ex.get(text_field, ""))) for ex in sample)
            optimized_tokens = sum(
                len(tokenizer.encode(self._optimize_text(ex.get(text_field, ""))[0]))
                for ex in sample
            )
            result["original_tokens"] = original_tokens
            result["optimized_tokens"] = optimized_tokens
            result["token_savings"] = original_tokens - optimized_tokens
            result["token_savings_pct"] = round(
                ((original_tokens - optimized_tokens) / max(original_tokens, 1)) * 100, 1
            )

        return result
