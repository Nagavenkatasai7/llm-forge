"""PII detection and redaction using Microsoft Presidio.

Supports three redaction strategies: ``"redact"`` (replace with entity-type
tags like ``[EMAIL_ADDRESS]``), ``"mask"`` (partial masking), and
``"pseudonymize"`` (deterministic fake values via hashing).
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Literal

from llm_forge.utils.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger("data.cleaning.pii_redactor")

# ---------------------------------------------------------------------------
# Optional dependencies: Presidio
# ---------------------------------------------------------------------------

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default entities to detect
# ---------------------------------------------------------------------------

DEFAULT_PII_ENTITIES: list[str] = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "LOCATION",
    "PERSON",
]

# ---------------------------------------------------------------------------
# Pseudonymization helpers
# ---------------------------------------------------------------------------


def _deterministic_hash(value: str, entity_type: str, length: int = 8) -> str:
    """Generate a deterministic pseudo-value from the original via hashing.

    The same ``(value, entity_type)`` pair always produces the same output,
    which is useful for consistency across a corpus without leaking the
    original PII.

    Parameters
    ----------
    value:
        The original PII value.
    entity_type:
        The entity type (used as part of the hash seed for domain separation).
    length:
        Length of the hex digest to use.

    Returns
    -------
    str
        A deterministic pseudo-value.
    """
    seed = f"{entity_type}:{value}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return digest[:length].upper()


def _pseudonymize_value(value: str, entity_type: str) -> str:
    """Generate a plausible-looking pseudo-value for the given entity type.

    Parameters
    ----------
    value:
        The detected PII value.
    entity_type:
        The entity type (e.g. ``"EMAIL_ADDRESS"``).

    Returns
    -------
    str
        A deterministic fake replacement.
    """
    h = _deterministic_hash(value, entity_type, length=8)

    if entity_type == "EMAIL_ADDRESS":
        return f"user_{h.lower()}@example.com"

    elif entity_type == "PHONE_NUMBER":
        digits = "".join(c for c in h if c.isdigit())
        # Pad with zeros to ensure 10 digits
        digits = (digits + "0000000000")[:10]
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:10]}"

    elif entity_type == "CREDIT_CARD":
        digits = "".join(c for c in h if c.isdigit())
        digits = (digits + "0000000000000000")[:16]
        return f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:16]}"

    elif entity_type == "US_SSN":
        digits = "".join(c for c in h if c.isdigit())
        digits = (digits + "000000000")[:9]
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:9]}"

    elif entity_type == "IP_ADDRESS":
        # Generate 4 octets from hash
        full_hash = hashlib.sha256(f"IP:{value}".encode()).hexdigest()
        octets = [str(int(full_hash[i : i + 2], 16)) for i in range(0, 8, 2)]
        return ".".join(octets)

    elif entity_type == "PERSON":
        return f"Person_{h[:6]}"

    else:
        return f"[{entity_type}_{h[:6]}]"


# ---------------------------------------------------------------------------
# Masking helper
# ---------------------------------------------------------------------------


def _mask_value(value: str, entity_type: str, mask_char: str = "*") -> str:
    """Partially mask a PII value, preserving some characters for context.

    Parameters
    ----------
    value:
        The detected PII value.
    entity_type:
        The entity type.
    mask_char:
        Character to use for masking.

    Returns
    -------
    str
        The partially masked value.
    """
    n = len(value)
    if n <= 2:
        return mask_char * n

    if entity_type == "EMAIL_ADDRESS":
        # Show first char and domain
        at_idx = value.find("@")
        if at_idx > 0:
            local = value[:at_idx]
            domain = value[at_idx:]
            masked_local = local[0] + mask_char * (len(local) - 1)
            return masked_local + domain
        return value[0] + mask_char * (n - 1)

    elif entity_type == "PHONE_NUMBER" or entity_type == "CREDIT_CARD":
        # Show last 4 digits
        digits = re.findall(r"\d", value)
        if len(digits) >= 4:
            last4 = "".join(digits[-4:])
            return mask_char * (n - 4) + last4
        return mask_char * n

    elif entity_type == "US_SSN":
        # Show last 4 digits
        digits = re.findall(r"\d", value)
        if len(digits) >= 4:
            return f"***-**-{''.join(digits[-4:])}"
        return mask_char * n

    elif entity_type == "IP_ADDRESS":
        # Mask first two octets
        parts = value.split(".")
        if len(parts) == 4:
            return f"***.***{'.'.join(parts[2:])}"
        return mask_char * n

    else:
        # Generic: show first and last char
        if n > 3:
            return value[0] + mask_char * (n - 2) + value[-1]
        return mask_char * n


# ---------------------------------------------------------------------------
# PII Redactor class
# ---------------------------------------------------------------------------


class PIIRedactor:
    """Detect and redact PII using Microsoft Presidio.

    Parameters
    ----------
    entities:
        List of entity types to detect.  Defaults to
        :data:`DEFAULT_PII_ENTITIES`.
    strategy:
        Redaction strategy:

        - ``"redact"``: Replace with entity-type tags (e.g. ``[EMAIL_ADDRESS]``)
        - ``"mask"``: Partial masking (e.g. ``j***@example.com``)
        - ``"pseudonymize"``: Deterministic fake values via hashing

    language:
        Language code for the Presidio analyzer. Defaults to ``"en"``.
    score_threshold:
        Minimum confidence score for entity detection. Defaults to ``0.5``.

    Raises
    ------
    ImportError
        If ``presidio-analyzer`` or ``presidio-anonymizer`` is not installed.
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        strategy: Literal["redact", "mask", "pseudonymize"] = "redact",
        language: str = "en",
        score_threshold: float = 0.5,
    ) -> None:
        if not _PRESIDIO_AVAILABLE:
            raise ImportError(
                "Microsoft Presidio is required for PII redaction. "
                "Install it with:\n"
                "  pip install presidio-analyzer presidio-anonymizer\n"
                "  python -m spacy download en_core_web_lg"
            )

        self.entities = entities or DEFAULT_PII_ENTITIES
        self.strategy = strategy
        self.language = language
        self.score_threshold = score_threshold

        logger.info(
            "Initializing PIIRedactor (strategy='%s', entities=%s, threshold=%.2f)",
            strategy,
            self.entities,
            score_threshold,
        )

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def _analyze(self, text: str) -> list[RecognizerResult]:
        """Run the Presidio analyzer on text.

        Parameters
        ----------
        text:
            Text to analyze for PII.

        Returns
        -------
        list[RecognizerResult]
            Detected PII entities.
        """
        results = self._analyzer.analyze(
            text=text,
            entities=self.entities,
            language=self.language,
            score_threshold=self.score_threshold,
        )
        return results

    def redact_text(self, text: str) -> str:
        """Detect and redact PII in a text string.

        Parameters
        ----------
        text:
            The text to redact.

        Returns
        -------
        str
            Text with PII replaced according to the configured strategy.
        """
        if not text or not text.strip():
            return text

        # Analyze for PII
        results = self._analyze(text)

        if not results:
            return text

        if self.strategy == "redact":
            # Use Presidio's built-in replace operator with entity type tags
            operators = {}
            for entity_type in self.entities:
                operators[entity_type] = OperatorConfig(
                    "replace",
                    {"new_value": f"[{entity_type}]"},
                )
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators,
            )
            return anonymized.text

        elif self.strategy == "mask":
            # Sort results by start position in reverse order (process from end)
            sorted_results = sorted(results, key=lambda r: r.start, reverse=True)
            result_text = text
            for result in sorted_results:
                original = text[result.start : result.end]
                masked = _mask_value(original, result.entity_type)
                result_text = result_text[: result.start] + masked + result_text[result.end :]
            return result_text

        elif self.strategy == "pseudonymize":
            # Sort results by start position in reverse order (process from end)
            sorted_results = sorted(results, key=lambda r: r.start, reverse=True)
            result_text = text
            for result in sorted_results:
                original = text[result.start : result.end]
                pseudo = _pseudonymize_value(original, result.entity_type)
                result_text = result_text[: result.start] + pseudo + result_text[result.end :]
            return result_text

        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

    def redact_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
    ) -> Dataset:
        """Apply PII redaction to every record in a HuggingFace Dataset.

        Parameters
        ----------
        dataset:
            A ``datasets.Dataset`` instance.
        text_field:
            Name of the column containing text to redact.

        Returns
        -------
        Dataset
            A new Dataset with PII redacted in the text field.

        Raises
        ------
        ValueError
            If *text_field* does not exist in the dataset.
        """
        if text_field not in dataset.column_names:
            raise ValueError(
                f"Text field '{text_field}' not found in dataset. "
                f"Available columns: {dataset.column_names}"
            )

        initial_count = len(dataset)
        logger.info(
            "Redacting PII in %d records (strategy='%s', field='%s')...",
            initial_count,
            self.strategy,
            text_field,
        )

        # Track how many records had PII
        pii_count = 0

        def _apply(example: dict) -> dict:
            nonlocal pii_count
            original = example[text_field]
            redacted = self.redact_text(original)
            if redacted != original:
                pii_count += 1
            example[text_field] = redacted
            return example

        dataset = dataset.map(
            _apply,
            desc="Redacting PII",
            num_proc=1,  # Presidio analyzer is not fork-safe by default
        )

        logger.info(
            "PII redaction complete. Processed %d records, %d contained PII (%.1f%%).",
            initial_count,
            pii_count,
            (pii_count / initial_count * 100) if initial_count else 0,
        )

        return dataset
