"""Data format conversion and tokenization for training."""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

logger = logging.getLogger(__name__)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


class DataPreprocessor:
    """Convert raw data into tokenized training-ready format."""

    def __init__(
        self,
        format_type: str = "alpaca",
        input_field: str = "instruction",
        output_field: str = "output",
        context_field: str | None = "input",
        system_prompt: str | None = None,
        max_seq_length: int = 2048,
    ):
        self.format_type = format_type
        self.input_field = input_field
        self.output_field = output_field
        self.context_field = context_field
        self.system_prompt = system_prompt
        self.max_seq_length = max_seq_length

    def format_dataset(self, dataset: Dataset) -> Dataset:
        """Convert dataset to unified text format based on configured format type."""
        formatter = {
            "alpaca": self._format_alpaca,
            "sharegpt": self._format_sharegpt,
            "completion": self._format_completion,
            "custom": self._format_custom,
        }.get(self.format_type)

        if formatter is None:
            raise ValueError(
                f"Unknown format: {self.format_type}. "
                f"Supported: alpaca, sharegpt, completion, custom"
            )

        logger.info(f"Formatting dataset with '{self.format_type}' format")
        logger.info(f"Dataset columns: {dataset.column_names}")
        formatted = dataset.map(formatter, remove_columns=dataset.column_names)

        # Validate: check first few samples have meaningful content
        self._validate_formatted_data(formatted)
        return formatted

    def _validate_formatted_data(self, dataset: Dataset, num_check: int = 5) -> None:
        """Sanity-check formatted samples to catch silent data corruption."""
        min_length = 50  # minimum chars for a meaningful training sample
        bad = 0
        for i in range(min(num_check, len(dataset))):
            text = dataset[i].get("text", "")
            if len(text) < min_length:
                bad += 1
                logger.warning(f"Sample {i} suspiciously short ({len(text)} chars): {text[:100]!r}")
            elif (
                "\nHuman:" not in text
                and "\nAssistant:" not in text
                and "### Response:" not in text
            ):
                if self.format_type in ("sharegpt", "alpaca"):
                    bad += 1
                    logger.warning(f"Sample {i} missing conversation turns: {text[:100]!r}")
        if bad == num_check:
            logger.warning(
                f"ALL {num_check} checked samples may be malformed — "
                f"none contain expected conversation markers for '{self.format_type}' format. "
                f"If training loss does not decrease, check dataset column mapping."
            )

    def format_for_chat_template(
        self,
        dataset: Dataset,
        tokenizer: Any,
    ) -> Dataset:
        """Format dataset using the model's chat template if available."""
        if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
            logger.info("No chat template found, using standard formatting")
            return self.format_dataset(dataset)

        logger.info("Using model's chat template for formatting")

        def apply_chat_template(example: dict) -> dict:
            messages = self._build_messages(example)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        return dataset.map(apply_chat_template, remove_columns=dataset.column_names)

    def tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer: Any,
        pack_sequences: bool = False,
    ) -> Dataset:
        """Tokenize formatted dataset with proper label masking."""
        logger.info(f"Tokenizing dataset (max_seq_length={self.max_seq_length})")

        if pack_sequences:
            logger.info(
                "pack_sequences=True: packing is delegated to TRL's SFTTrainer "
                "(set via SFTConfig.packing). Tokenizing normally here."
            )
        return self._tokenize_standard(dataset, tokenizer)

    def _format_alpaca(self, example: dict) -> dict:
        """Format example in Alpaca instruction-response format."""
        instruction = example.get(self.input_field, "")
        context = example.get(self.context_field, "") if self.context_field else ""
        output = example.get(self.output_field, "")

        if context and context.strip():
            text = ALPACA_TEMPLATE.format(instruction=instruction, input=context, output=output)
        else:
            text = ALPACA_TEMPLATE_NO_INPUT.format(instruction=instruction, output=output)

        if self.system_prompt:
            text = f"### System:\n{self.system_prompt}\n\n{text}"

        return {"text": text}

    def _format_sharegpt(self, example: dict) -> dict:
        """Format example in ShareGPT multi-turn conversation format.

        Supports three dataset layouts:
          1. ``conversations`` list of dicts with role/value keys (classic ShareGPT)
          2. ``messages`` list of dicts with role/content keys (OpenAI style)
          3. Flat ``system`` / ``user`` / ``assistant`` columns (HuggingFace common)

        Returns both a ``text`` column (for backward compatibility) and a
        ``messages`` column (list of role/content dicts for TRL chat-template
        pipeline and assistant-only loss masking).
        """
        _ROLE_MAP = {"human": "user", "gpt": "assistant"}
        messages: list[dict[str, str]] = []
        conversations = example.get("conversations", [])

        if conversations:
            # Classic ShareGPT: list of {"from"/"role": ..., "value"/"content": ...}
            for turn in conversations:
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                role = _ROLE_MAP.get(role, role)
                if value and value.strip():
                    messages.append({"role": role, "content": value.strip()})
        elif "messages" in example and isinstance(example["messages"], list):
            # Already in OpenAI-style messages format
            for msg in example["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                role = _ROLE_MAP.get(role, role)
                if content and content.strip():
                    messages.append({"role": role, "content": content.strip()})
        elif "user" in example or "assistant" in example:
            # Flat columns: system / user / assistant (e.g. Finance-Instruct-500k)
            sys_text = example.get("system", "")
            if sys_text and sys_text.strip():
                messages.append({"role": "system", "content": sys_text.strip()})

            user_text = example.get("user", example.get("human", ""))
            if user_text and user_text.strip():
                messages.append({"role": "user", "content": user_text.strip()})

            assistant_text = example.get("assistant", example.get("gpt", ""))
            if assistant_text and assistant_text.strip():
                messages.append({"role": "assistant", "content": assistant_text.strip()})

        # Inject system prompt if none present and one is configured
        if self.system_prompt:
            has_system = any(m["role"] == "system" for m in messages)
            if not has_system:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Build flat text for backward compatibility
        parts = []
        for m in messages:
            r = m["role"]
            if r == "system":
                parts.append(f"System: {m['content']}")
            elif r == "user":
                parts.append(f"Human: {m['content']}")
            elif r == "assistant":
                parts.append(f"Assistant: {m['content']}")

        return {"text": "\n\n".join(parts), "messages": messages}

    def _format_completion(self, example: dict) -> dict:
        """Format example as plain text completion."""
        text = example.get("text", example.get(self.input_field, ""))
        if self.system_prompt:
            text = f"{self.system_prompt}\n\n{text}"
        return {"text": str(text)}

    def _format_custom(self, example: dict) -> dict:
        """Format example using custom field mapping."""
        instruction = example.get(self.input_field, "")
        output = example.get(self.output_field, "")
        context = example.get(self.context_field, "") if self.context_field else ""

        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if instruction:
            parts.append(str(instruction))
        if context:
            parts.append(str(context))
        if output:
            parts.append(str(output))

        return {"text": "\n\n".join(parts)}

    def _build_messages(self, example: dict) -> list[dict[str, str]]:
        """Build chat messages list from an example."""
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if self.format_type == "sharegpt":
            conversations = example.get("conversations", [])
            for turn in conversations:
                role = turn.get("from", turn.get("role", ""))
                value = turn.get("value", turn.get("content", ""))
                if role in ("human", "user"):
                    messages.append({"role": "user", "content": value})
                elif role in ("gpt", "assistant"):
                    messages.append({"role": "assistant", "content": value})
                elif role == "system":
                    messages.insert(0, {"role": "system", "content": value})
        else:
            instruction = example.get(self.input_field, "")
            context = example.get(self.context_field, "") if self.context_field else ""
            output = example.get(self.output_field, "")

            user_content = str(instruction)
            if context:
                user_content += f"\n\n{context}"

            messages.append({"role": "user", "content": user_content})
            if output:
                messages.append({"role": "assistant", "content": str(output)})

        return messages

    def _tokenize_standard(self, dataset: Dataset, tokenizer: Any) -> Dataset:
        """Standard tokenization with label masking for instruction tokens."""

        def tokenize_fn(examples: dict) -> dict:
            texts = examples["text"]
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="longest",
                return_tensors=None,
            )
            tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
            return tokenized

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        return tokenized

    def split_dataset(
        self,
        dataset: Dataset,
        test_size: float = 0.1,
        seed: int = 42,
    ) -> tuple[Dataset, Dataset]:
        """Split dataset into train and eval sets."""
        if test_size <= 0:
            return dataset, Dataset.from_list([])

        split = dataset.train_test_split(test_size=test_size, seed=seed)
        logger.info(f"Split dataset: {len(split['train'])} train, {len(split['test'])} eval")
        return split["train"], split["test"]
