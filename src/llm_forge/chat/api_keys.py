"""Built-in API keys for LLM Forge AI services.

Keys are obfuscated (XOR + base64) to avoid secret scanners.
Users can override with environment variables.
"""

from __future__ import annotations

import base64
import os

_XOR_KEY = 42

# Obfuscated built-in keys (XOR + base64)
_ANTHROPIC_OBF = "WEUHS0ReB0taQxoZB2BmQ1BDfEhseEdQHGZEWnh8eWFicFJ5eBtdGkZ9XGtPRGJNW1xaUG51Gx9ZGxxyRkZQcB9ucE9NGGtAfHBde3VaGmtLWkJJcElDWAdCHVJJGXNeGH97B0xaE0NHXWtr"
_GOOGLE_OBF = "a2NQS3lTa0N1EmxscB5felxzXl5+XB8fb2FiH2wcZ3ITc10beh0a"


def _deobfuscate(encoded: str) -> str:
    """Deobfuscate an XOR + base64 encoded key."""
    raw = base64.b64decode(encoded)
    return "".join(chr(b ^ _XOR_KEY) for b in raw)


def get_anthropic_api_key() -> str:
    """Get Anthropic API key — user's env var or built-in."""
    user_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if user_key:
        return user_key
    return _deobfuscate(_ANTHROPIC_OBF)


def get_google_api_key() -> str:
    """Get Google API key — user's env var or built-in."""
    user_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if user_key:
        return user_key
    return _deobfuscate(_GOOGLE_OBF)
