"""Tests for the PII redactor module.

Covers email detection/redaction, phone number redaction, and
pass-through of text without PII. Skips gracefully if Presidio
or other required deps are not available.
"""

from __future__ import annotations

import pytest

# Check if both the module and Presidio are available
_MODULE_AVAILABLE = False
_SKIP_REASON = ""

try:
    from llm_forge.data.cleaning.pii_redactor import (
        _PRESIDIO_AVAILABLE,
        DEFAULT_PII_ENTITIES,
        PIIRedactor,
    )

    if _PRESIDIO_AVAILABLE:
        _MODULE_AVAILABLE = True
    else:
        _SKIP_REASON = "Presidio is not installed"
except ImportError as e:
    _SKIP_REASON = f"pii_redactor not importable: {e}"

pytestmark = pytest.mark.skipif(
    not _MODULE_AVAILABLE,
    reason=_SKIP_REASON or "PII redactor dependencies not available",
)


# ===================================================================
# Email detection and redaction
# ===================================================================


class TestEmailRedaction:
    """Test email PII detection and redaction."""

    def test_email_detected_and_redacted(self) -> None:
        redactor = PIIRedactor(
            entities=["EMAIL_ADDRESS"],
            strategy="redact",
        )
        text = "Contact me at john.doe@example.com for details."
        result = redactor.redact_text(text)
        assert "john.doe@example.com" not in result
        assert "[EMAIL_ADDRESS]" in result

    def test_email_mask_strategy(self) -> None:
        redactor = PIIRedactor(
            entities=["EMAIL_ADDRESS"],
            strategy="mask",
        )
        text = "Email: test@example.com"
        result = redactor.redact_text(text)
        assert "test@example.com" not in result
        # Should still contain some part of the masked email
        assert "@" in result or "*" in result


# ===================================================================
# Phone number redaction
# ===================================================================


class TestPhoneRedaction:
    """Test phone number PII detection and redaction."""

    def test_phone_detected_and_redacted(self) -> None:
        redactor = PIIRedactor(
            entities=["PHONE_NUMBER"],
            strategy="redact",
            score_threshold=0.3,
        )
        text = "Call me at 212-555-1234 tomorrow."
        result = redactor.redact_text(text)
        assert "212-555-1234" not in result
        assert "[PHONE_NUMBER]" in result

    def test_phone_pseudonymize(self) -> None:
        redactor = PIIRedactor(
            entities=["PHONE_NUMBER"],
            strategy="pseudonymize",
            score_threshold=0.3,
        )
        text = "My number is 212-555-1234."
        result = redactor.redact_text(text)
        assert "212-555-1234" not in result


# ===================================================================
# Text without PII
# ===================================================================


class TestNoPII:
    """Test that text without PII passes through unchanged."""

    def test_clean_text_unchanged(self) -> None:
        redactor = PIIRedactor(strategy="redact")
        text = "The weather today is sunny and warm."
        result = redactor.redact_text(text)
        assert result == text

    def test_empty_text(self) -> None:
        redactor = PIIRedactor(strategy="redact")
        assert redactor.redact_text("") == ""
        assert redactor.redact_text("   ") == "   "


# ===================================================================
# Default entities
# ===================================================================


class TestDefaultEntities:
    """Test default PII entity list."""

    def test_default_entities_include_common_types(self) -> None:
        assert "EMAIL_ADDRESS" in DEFAULT_PII_ENTITIES
        assert "PHONE_NUMBER" in DEFAULT_PII_ENTITIES
        assert "CREDIT_CARD" in DEFAULT_PII_ENTITIES


# ===================================================================
# Module-level Presidio flag
# ===================================================================


class TestPresidioFlag:
    """Test the _PRESIDIO_AVAILABLE flag (always True if we get here)."""

    def test_presidio_available(self) -> None:
        assert _PRESIDIO_AVAILABLE is True
